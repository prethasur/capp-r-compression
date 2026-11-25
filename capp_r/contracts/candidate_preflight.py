#!/usr/bin/env python3
"""Construction, full-model F/B, task-loss, ONNX, and ORT candidate preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from capp_r.contracts.protected_semantics import adapter_for
from capp_r.generator.realize_micrograph import (
    boundary_shape,
    build_module,
    parameter_count,
    replace_module,
)

YOLO_ROUTING_ATTRIBUTES = ("i", "f", "type", "np")


def tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from tensors(value[key])
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from tensors(nested)


def signature(value: Any, path: str = "output") -> list[dict[str, Any]]:
    if isinstance(value, torch.Tensor):
        return [{"path": path, "shape": list(value.shape), "dtype": str(value.dtype)}]
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key in sorted(value):
            rows.extend(signature(value[key], f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            rows.extend(signature(nested, f"{path}[{index}]"))
    return rows


def finite(value: Any) -> bool:
    values = list(tensors(value))
    return bool(values) and all(bool(torch.isfinite(tensor).all()) for tensor in values)


def normalized_output_divergence(parent: Any, child: Any) -> float:
    parent_tensors = list(tensors(parent))
    child_tensors = list(tensors(child))
    if len(parent_tensors) != len(child_tensors) or not parent_tensors:
        raise RuntimeError("parent and child tensor output structures differ")
    divergences = []
    for parent_tensor, child_tensor in zip(parent_tensors, child_tensors):
        if tuple(parent_tensor.shape) != tuple(child_tensor.shape):
            raise RuntimeError("parent and child tensor output shapes differ")
        numerator = (child_tensor.float() - parent_tensor.float()).abs().mean()
        denominator = parent_tensor.float().abs().mean().clamp_min(1e-8)
        divergences.append(numerator / denominator)
    return float(torch.stack(divergences).mean().detach().cpu())


def contract_shape(type_spec: dict[str, Any]) -> tuple[int, ...]:
    return boundary_shape(type_spec)


def initialize(module: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for child in module.modules():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(child.weight, a=math.sqrt(5))
        elif isinstance(child, (nn.BatchNorm2d, nn.LayerNorm)):
            if child.weight is not None:
                nn.init.ones_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def synthetic_yolo_batch(
    image: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "img": image,
        "cls": torch.tensor([[0.0]], dtype=torch.float32, device=device),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32, device=device
        ),
        "batch_idx": torch.tensor([0.0], dtype=torch.float32, device=device),
    }


def preserve_yolo_routing_metadata(original: nn.Module, replacement: nn.Module) -> None:
    """Retain the graph-routing attributes consumed by Ultralytics DetectionModel."""
    for attribute in YOLO_ROUTING_ATTRIBUTES:
        if hasattr(original, attribute):
            setattr(replacement, attribute, getattr(original, attribute))


def normalize_yolo_training_args(model: nn.Module) -> None:
    """Merge legacy checkpoint arguments with current loss defaults."""
    from ultralytics.cfg import DEFAULT_CFG, get_cfg

    overrides = dict(model.args) if isinstance(model.args, dict) else vars(model.args)
    model.args = get_cfg(DEFAULT_CFG, overrides=overrides)
    model.criterion = None


def load_yolo_model(weights: Path) -> nn.Module:
    from ultralytics import YOLO

    return YOLO(str(weights)).model


def construction_and_fb_yolo(
    candidate: dict[str, Any],
    model: nn.Module,
    parent_signature: list[dict[str, Any]],
    parent_output: Any,
    probe_image: torch.Tensor,
    protected_semantics: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    graph = candidate["graph"]
    row: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "architecture": candidate["architecture"],
        "graph_hash": candidate["graph_hash"],
        "location_id": candidate["location_id"],
        "topology": candidate["topology"],
        "target_parameter_reduction_fraction":
            candidate["target_parameter_reduction_fraction"],
        "symbolic_type": True,
        "construction": False,
        "forward": False,
        "backward": False,
        "training_loss": False,
        "boundary_shape": False,
        "stride_feature_level": False,
        "protected_semantics": False,
        "output_signature": False,
        "onnx_export": False,
        "onnx_checker": False,
        "onnxruntime": False,
        "numerical_sanity": False,
        "failure_stage": "",
        "failure_reason": "",
    }
    replacement = build_module(graph)
    initialize(replacement, int(candidate["graph_hash"][:8], 16))
    row["candidate_parameters_actual"] = parameter_count(replacement)
    row["candidate_parameters_estimate"] = int(
        candidate["candidate_parameters_estimate"]
    )
    if row["candidate_parameters_actual"] != row["candidate_parameters_estimate"]:
        raise RuntimeError(
            f"parameter estimate mismatch: actual={row['candidate_parameters_actual']}, "
            f"estimated={row['candidate_parameters_estimate']}"
        )
    row["construction"] = True

    probe = torch.randn(contract_shape(graph["input_type"]))
    replacement.eval()
    with torch.no_grad():
        probe_output = replacement(probe)
    expected_shape = contract_shape(graph["output_type"])
    if tuple(probe_output.shape) != expected_shape:
        raise RuntimeError(
            f"replacement boundary mismatch: {tuple(probe_output.shape)} != "
            f"{expected_shape}"
        )
    row["boundary_shape"] = True
    row["stride_feature_level"] = (
        graph["input_type"].get("height") is None
        or (
            int(graph["input_type"]["height"])
            // int(graph["output_type"]["height"])
            in {1, 2}
        )
    )
    if not row["stride_feature_level"]:
        raise RuntimeError("replacement changes an unsupported spatial stride")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replacement = replacement.to(device)
    for parameter in replacement.parameters():
        parameter.requires_grad_(True)
    original = model.get_submodule(candidate["location_id"])
    preserve_yolo_routing_metadata(original, replacement)
    replace_module(model, candidate["location_id"], replacement)
    try:
        model.eval()
        with torch.no_grad():
            output = model(probe_image)
        row["forward"] = finite(output)
        row["output_signature"] = signature(output) == parent_signature
        if not row["forward"]:
            raise RuntimeError("full-model output contains NaN/Inf or no tensors")
        if not row["output_signature"]:
            raise RuntimeError("full-model output signature changed")
        divergence = normalized_output_divergence(parent_output, output)
        row["normalized_parent_child_output_divergence"] = divergence
        row["numerical_sanity"] = math.isfinite(divergence)
        if not row["numerical_sanity"]:
            raise RuntimeError("parent-child numerical divergence is non-finite")

        adapter = adapter_for(candidate["architecture"])
        row["protected_semantics"] = (
            adapter.output_semantics(model) == protected_semantics
        )
        if not row["protected_semantics"]:
            raise RuntimeError("protected task semantics changed")

        normalize_yolo_training_args(model)
        model.train()
        replacement_modules = set(replacement.modules())
        for module in model.modules():
            if (
                isinstance(module, nn.modules.batchnorm._BatchNorm)
                and module not in replacement_modules
            ):
                module.eval()
        loss_output = model.loss(synthetic_yolo_batch(probe_image, device))
        loss_terms = loss_output[0] if isinstance(loss_output, tuple) else loss_output
        loss = loss_terms.sum()
        row["training_loss_value"] = float(loss.detach().cpu())
        row["training_loss"] = bool(torch.isfinite(loss))
        if not row["training_loss"]:
            raise RuntimeError("task loss is NaN/Inf")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in replacement.parameters()
            if parameter.requires_grad
        ]
        row["backward"] = bool(gradients) and all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in gradients
        )
        if not row["backward"]:
            raise RuntimeError("replacement gradients are absent or non-finite")
    finally:
        replace_module(model, candidate["location_id"], original)
        model.zero_grad(set_to_none=True)
        model.criterion = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row


class YOLODeploymentModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images)
        return output[0] if isinstance(output, (tuple, list)) else output


def onnx_and_ort_yolo(
    row: dict[str, Any],
    candidate: dict[str, Any],
    weights: Path,
    output_dir: Path,
) -> None:
    import onnx
    import onnxruntime

    model = load_yolo_model(weights).eval()
    replacement = build_module(candidate["graph"])
    initialize(replacement, int(candidate["graph_hash"][:8], 16))
    original = model.get_submodule(candidate["location_id"])
    preserve_yolo_routing_metadata(original, replacement)
    replace_module(model, candidate["location_id"], replacement)
    wrapper = YOLODeploymentModel(model).eval()
    images = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        expected = wrapper(images)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{candidate['candidate_id']}.onnx"
    torch.onnx.export(
        wrapper,
        images,
        output,
        input_names=["images"],
        output_names=["predictions"],
        opset_version=16,
        do_constant_folding=True,
    )
    row["onnx_export"] = output.is_file() and output.stat().st_size > 0
    checked = onnx.load(output)
    onnx.checker.check_model(checked)
    row["onnx_checker"] = True
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session = onnxruntime.InferenceSession(
        str(output),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(None, {"images": images.numpy()})
    expected_array = expected.detach().cpu().numpy()
    row["onnxruntime"] = (
        len(actual) == 1
        and actual[0].shape == expected_array.shape
        and bool(np.isfinite(actual[0]).all())
    )
    if not row["onnxruntime"]:
        raise RuntimeError(
            "ONNX Runtime output is missing, non-finite, or shape-incompatible"
        )
    difference = np.abs(actual[0].astype(np.float32) - expected_array.astype(np.float32))
    row["onnx_predictions_max_abs_difference"] = float(np.max(difference))
    row["onnx_predictions_mean_abs_difference"] = float(np.mean(difference))
    row["onnx_predictions_numerically_close"] = bool(
        np.allclose(actual[0], expected_array, rtol=1e-3, atol=1e-3)
    )
    row["onnx_max_abs_difference"] = row[
        "onnx_predictions_max_abs_difference"
    ]
    row["onnx_numerically_close"] = row[
        "onnx_predictions_numerically_close"
    ]
    row["onnx_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    row["onnx_bytes"] = output.stat().st_size
    if not row["onnx_numerically_close"]:
        raise RuntimeError("ONNX Runtime output exceeds preflight numerical tolerances")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        required=True,
        choices=["yolov8n", "yolo11n"],
    )
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--failure-summary", required=True, type=Path)
    parser.add_argument("--onnx-dir", required=True, type=Path)
    parser.add_argument("--construction-max", type=int, default=100)
    parser.add_argument("--onnx-max", type=int, default=60)
    args = parser.parse_args()
    if not (1 <= args.construction_max <= 100 and 1 <= args.onnx_max <= 60):
        raise SystemExit("preflight budgets violate the frozen protocol")
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    if payload.get("official_validation_used") or payload.get("HRSID_test_accessed"):
        raise RuntimeError("STOP: candidate manifest is contaminated by a prohibited split")
    selected = payload["candidates"][: args.construction_max]
    if not selected:
        raise RuntimeError("no candidates available for preflight")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_yolo_model(args.weights)
    model = model.to(device)
    adapter = adapter_for(args.architecture)
    protected = adapter.output_semantics(model)
    model.eval()
    probe_image = torch.rand(1, 3, 640, 640, device=device)
    with torch.no_grad():
        parent_output = model(probe_image)
    parent_signature = signature(parent_output)

    rows: list[dict[str, Any]] = []
    for candidate in selected:
        try:
            row = construction_and_fb_yolo(
                candidate, model, parent_signature, parent_output,
                probe_image, protected, device
            )
        except Exception as error:
            row = {
                "candidate_id": candidate["candidate_id"],
                "architecture": candidate["architecture"],
                "graph_hash": candidate["graph_hash"],
                "location_id": candidate["location_id"],
                "topology": candidate["topology"],
                "target_parameter_reduction_fraction":
                    candidate["target_parameter_reduction_fraction"],
                "failure_stage": "construction_forward_backward",
                "failure_reason": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=5),
            }
        rows.append(row)
        write_csv(args.table, rows)
        print(json.dumps({
            "phase": "construction_forward_backward",
            "candidate_id": candidate["candidate_id"],
            "passed": not bool(row.get("failure_reason")),
            "failure_reason": row.get("failure_reason", ""),
        }), flush=True)

    forward_backward_valid = [
        (row, candidate)
        for row, candidate in zip(rows, selected)
        if all(bool(row.get(gate)) for gate in (
            "symbolic_type", "construction", "forward", "backward", "training_loss",
            "boundary_shape", "stride_feature_level", "protected_semantics",
            "output_signature", "numerical_sanity",
        ))
    ]
    eligible_for_onnx = forward_backward_valid[: args.onnx_max]
    for row, candidate in eligible_for_onnx:
        try:
            onnx_and_ort_yolo(row, candidate, args.weights, args.onnx_dir)
        except Exception as error:
            row["failure_stage"] = "onnx_ort"
            row["failure_reason"] = f"{type(error).__name__}: {error}"
            row["traceback"] = traceback.format_exc(limit=5)
        write_csv(args.table, rows)
        print(json.dumps({
            "phase": "onnx_ort",
            "candidate_id": candidate["candidate_id"],
            "passed": not bool(row.get("failure_reason")),
            "failure_reason": row.get("failure_reason", ""),
        }), flush=True)

    write_csv(args.table, rows)
    failures = Counter(
        row.get("failure_reason", "") or "PASS_TO_CURRENT_STAGE" for row in rows
    )
    summary_rows = [
        {"architecture": args.architecture, "outcome": outcome, "count": count}
        for outcome, count in sorted(failures.items())
    ]
    write_csv(args.failure_summary, summary_rows)
    fully_valid = sum(
        all(bool(row.get(gate)) for gate in (
            "symbolic_type", "construction", "forward", "backward", "training_loss",
            "boundary_shape", "stride_feature_level", "protected_semantics",
            "output_signature", "onnx_export", "onnx_checker", "onnxruntime",
            "onnx_numerically_close", "numerical_sanity",
        ))
        for row in rows
    )
    result = {
        "status": "COMPLETE" if fully_valid else "FAILED_GATE",
        "architecture": args.architecture,
        "construction_attempted": len(rows),
        "forward_backward_valid": len(forward_backward_valid),
        "onnx_attempted": len(eligible_for_onnx),
        "fully_valid": fully_valid,
        "official_validation_used": False,
        "HRSID_test_accessed": False,
    }
    print(json.dumps(result, indent=2))
    if fully_valid < 1:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
