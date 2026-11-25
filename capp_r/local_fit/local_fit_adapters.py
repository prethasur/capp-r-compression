"""Architecture boundary for training-only CAPP-R local fitting."""

from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from capp_r.contracts.candidate_preflight import (
    initialize,
    preserve_yolo_routing_metadata,
)
from capp_r.contracts.protected_semantics import adapter_for
from capp_r.generator.realize_micrograph import (
    boundary_shape,
    build_module,
    parameter_count,
    replace_module,
)


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for key in sorted(value):
            try:
                return first_tensor(value[key])
            except RuntimeError:
                pass
    if isinstance(value, (list, tuple)):
        for nested in value:
            try:
                return first_tensor(nested)
            except RuntimeError:
                pass
    raise RuntimeError("captured feature contains no tensor")


@dataclass
class FeaturePair:
    parent: torch.Tensor | None = None
    candidate: torch.Tensor | None = None


class FeatureHooks(AbstractContextManager["FeatureHooks"]):
    def __init__(
        self, parent_region: nn.Module, candidate_region: nn.Module
    ) -> None:
        self.features = FeaturePair()
        self._handles = (
            parent_region.register_forward_hook(self._parent_hook),
            candidate_region.register_forward_hook(self._candidate_hook),
        )

    def _parent_hook(self, _module, _inputs, output) -> None:
        self.features.parent = first_tensor(output)

    def _candidate_hook(self, _module, _inputs, output) -> None:
        self.features.candidate = first_tensor(output)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()


class LocalFitAdapter(ABC):
    family: str

    @abstractmethod
    def prepare_parent(self, weights: Path, **kwargs) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def prepare_candidate(
        self, parent: nn.Module, candidate: dict[str, Any]
    ) -> tuple[nn.Module, nn.Module]:
        raise NotImplementedError

    def register_feature_hooks(
        self,
        parent: nn.Module,
        candidate_model: nn.Module,
        location: str,
    ) -> FeatureHooks:
        return FeatureHooks(
            parent.get_submodule(location),
            candidate_model.get_submodule(location),
        )

    @abstractmethod
    def capture_parent_targets(
        self, parent: nn.Module, images: torch.Tensor
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def capture_candidate_outputs(
        self, candidate_model: nn.Module, images: torch.Tensor
    ) -> Any:
        raise NotImplementedError

    def compute_feature_fidelity(
        self, parent: torch.Tensor, candidate: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if parent.shape != candidate.shape:
            raise RuntimeError("replacement-boundary feature shape changed")
        batch = parent.shape[0]
        difference = candidate.float() - parent.float()
        mse = difference.square().mean()
        normalized = mse / parent.float().square().mean().clamp_min(1e-8)
        cosine = F.cosine_similarity(
            parent.float().reshape(batch, -1),
            candidate.float().reshape(batch, -1),
            dim=1,
        )
        cosine_distance = (
            mse * 0.0
            if torch.equal(parent, candidate)
            else (1.0 - cosine.mean()).clamp_min(0.0)
        )
        return {
            "feature_mse": mse,
            "normalized_feature_mse": normalized,
            "feature_cosine_distance": cosine_distance,
        }

    @abstractmethod
    def compute_task_output_fidelity(
        self, parent_output: Any, candidate_output: Any
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def compute_training_loss(
        self, model: nn.Module, batch: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def verify_boundary_contract(
        self, candidate: dict[str, Any], replacement: nn.Module
    ) -> None:
        raise NotImplementedError

    def verify_protected_semantics(
        self, parent: nn.Module, candidate_model: nn.Module
    ) -> None:
        architecture = adapter_for(self.family)
        if (
            architecture.output_semantics(parent)
            != architecture.output_semantics(candidate_model)
        ):
            raise RuntimeError("protected task semantics changed")

    @abstractmethod
    def export_candidate(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def compare_onnx_outputs(self, expected: Any, actual: Any) -> dict[str, Any]:
        raise NotImplementedError

    def serialize_result(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)


def yolo_raw_scales(output: Any) -> list[torch.Tensor]:
    if isinstance(output, tuple) and len(output) > 1:
        raw = output[1]
    else:
        raw = output
    if not isinstance(raw, (list, tuple)) or not raw:
        raise RuntimeError("YOLO raw multiscale outputs are unavailable")
    scales = list(raw)
    if not all(isinstance(value, torch.Tensor) and value.ndim == 4 for value in scales):
        raise RuntimeError("YOLO raw scale layout is not BCHW")
    return scales


def yolo_decoded(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.ndim != 3:
        raise RuntimeError("YOLO decoded output is not [B,C,N]")
    return output


def split_yolo_scale(
    scale: torch.Tensor, nc: int, reg_max: int
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = 4 * reg_max + nc
    if scale.shape[1] != expected:
        raise RuntimeError(
            f"YOLO raw channels changed: {scale.shape[1]} != {expected}"
        )
    box = scale[:, : 4 * reg_max]
    classification = scale[:, 4 * reg_max :]
    return box, classification


def _decoded_matching(
    parent: torch.Tensor, candidate: torch.Tensor, nc: int, top_k: int = 100
) -> dict[str, torch.Tensor]:
    if parent.shape != candidate.shape or parent.shape[1] != 4 + nc:
        raise RuntimeError("YOLO decoded layout changed")
    box_values: list[torch.Tensor] = []
    score_values: list[torch.Tensor] = []
    count_values: list[torch.Tensor] = []
    for image_index in range(parent.shape[0]):
        parent_boxes = parent[image_index, :4].transpose(0, 1)
        candidate_boxes = candidate[image_index, :4].transpose(0, 1)
        parent_scores = parent[image_index, 4:].sigmoid().amax(0)
        candidate_scores = candidate[image_index, 4:].sigmoid().amax(0)
        keep = min(top_k, parent_scores.numel(), candidate_scores.numel())
        parent_index = parent_scores.topk(keep).indices
        candidate_index = candidate_scores.topk(keep).indices
        left_boxes = parent_boxes[parent_index]
        right_boxes = candidate_boxes[candidate_index]
        left_scores = parent_scores[parent_index]
        right_scores = candidate_scores[candidate_index]
        cost = torch.cdist(left_boxes.detach(), right_boxes.detach(), p=1)
        cost += (
            left_scores.detach()[:, None] - right_scores.detach()[None, :]
        ).abs()
        left_match, right_match = linear_sum_assignment(cost.cpu().numpy())
        left_match = torch.as_tensor(left_match, device=parent.device)
        right_match = torch.as_tensor(right_match, device=candidate.device)
        box_values.append(
            (left_boxes[left_match] - right_boxes[right_match]).abs().mean()
        )
        score_values.append(
            (left_scores[left_match] - right_scores[right_match]).abs().mean()
        )
        count_values.append(
            (
                (parent_scores >= 0.25).sum()
                - (candidate_scores >= 0.25).sum()
            ).abs().float()
        )
    return {
        "decoded_box_l1": torch.stack(box_values).mean(),
        "decoded_score_divergence": torch.stack(score_values).mean(),
        "diagnostic_prediction_count_delta": torch.stack(count_values).mean(),
    }


class YOLOLocalFitAdapter(LocalFitAdapter):
    def __init__(self, family: str) -> None:
        if family not in {"yolov8n", "yolo11n"}:
            raise ValueError("YOLO adapter supports yolov8n or yolo11n")
        self.family = family

    def prepare_parent(self, weights: Path, **kwargs) -> nn.Module:
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG, get_cfg

        model = YOLO(str(weights)).model
        overrides = dict(model.args) if isinstance(model.args, dict) else vars(model.args)
        model.args = get_cfg(DEFAULT_CFG, overrides=overrides)
        model.criterion = None
        return model

    def prepare_candidate(
        self, parent: nn.Module, candidate: dict[str, Any]
    ) -> tuple[nn.Module, nn.Module]:
        model = copy.deepcopy(parent)
        replacement = build_module(candidate["graph"])
        initialize(replacement, int(candidate["graph_hash"][:8], 16))
        source = model.get_submodule(candidate["location_id"])
        preserve_yolo_routing_metadata(source, replacement)
        replace_module(model, candidate["location_id"], replacement)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in replacement.parameters():
            parameter.requires_grad_(True)
        return model, replacement

    def capture_parent_targets(
        self, parent: nn.Module, images: torch.Tensor
    ) -> Any:
        return parent.eval()(images)

    def capture_candidate_outputs(
        self, candidate_model: nn.Module, images: torch.Tensor
    ) -> Any:
        return candidate_model.eval()(images)

    def _detect_contract(self, model: nn.Module) -> tuple[int, int, int]:
        semantics = adapter_for(self.family).output_semantics(model)
        nc = int(semantics["nc"])
        reg_max = int(semantics["reg_max"])
        levels = int(semantics["feature_levels"])
        return nc, reg_max, levels

    def compute_task_output_fidelity(
        self, parent_output: Any, candidate_output: Any
    ) -> dict[str, Any]:
        parent_scales = yolo_raw_scales(parent_output)
        candidate_scales = yolo_raw_scales(candidate_output)
        if len(parent_scales) != len(candidate_scales):
            raise RuntimeError("YOLO feature-level count changed")
        decoded_parent = yolo_decoded(parent_output)
        decoded_candidate = yolo_decoded(candidate_output)
        nc = decoded_parent.shape[1] - 4
        reg_max = (parent_scales[0].shape[1] - nc) // 4
        scale_rows: list[dict[str, float]] = []
        class_terms: list[torch.Tensor] = []
        dfl_terms: list[torch.Tensor] = []
        raw_box_terms: list[torch.Tensor] = []
        for index, (left, right) in enumerate(
            zip(parent_scales, candidate_scales)
        ):
            if left.shape != right.shape:
                raise RuntimeError(f"YOLO raw scale {index} shape changed")
            left_box, left_class = split_yolo_scale(left, nc, reg_max)
            right_box, right_class = split_yolo_scale(right, nc, reg_max)
            class_divergence = (
                left_class.sigmoid() - right_class.sigmoid()
            ).abs().mean()
            batch, _, height, width = left_box.shape
            left_dfl = left_box.reshape(batch, 4, reg_max, height, width)
            right_dfl = right_box.reshape(batch, 4, reg_max, height, width)
            left_probability = left_dfl.softmax(2)
            dfl_divergence = F.kl_div(
                right_dfl.log_softmax(2),
                left_probability,
                reduction="batchmean",
            ).clamp_min(0.0) / (4 * height * width)
            raw_box = (left_box - right_box).abs().mean()
            class_terms.append(class_divergence)
            dfl_terms.append(dfl_divergence)
            raw_box_terms.append(raw_box)
            scale_rows.append({
                "scale_index": index,
                "height": height,
                "width": width,
                "classification_divergence": float(class_divergence.detach().cpu()),
                "dfl_divergence": float(dfl_divergence.detach().cpu()),
                "raw_box_distribution_divergence": float(raw_box.detach().cpu()),
            })
        decoded = _decoded_matching(
            decoded_parent, decoded_candidate, nc=nc
        )
        return {
            "classification_logit_divergence": torch.stack(class_terms).mean(),
            "dfl_distribution_divergence": torch.stack(dfl_terms).mean(),
            "raw_box_distribution_divergence": torch.stack(raw_box_terms).mean(),
            **decoded,
            "scale_metrics": scale_rows,
        }

    def compute_training_loss(
        self, model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        output = model.loss(batch)
        terms = output[0] if isinstance(output, tuple) else output
        if not isinstance(terms, torch.Tensor):
            raise RuntimeError("YOLO native loss did not return a tensor")
        loss = terms.sum()
        names = ("box_loss", "classification_loss", "dfl_loss")
        flat = terms.detach().flatten().cpu()
        components = {
            names[index] if index < len(names) else f"loss_{index}": float(value)
            for index, value in enumerate(flat)
        }
        return loss, components

    def verify_boundary_contract(
        self, candidate: dict[str, Any], replacement: nn.Module
    ) -> None:
        graph = candidate["graph"]
        probe = torch.randn(boundary_shape(graph["input_type"]))
        with torch.no_grad():
            output = replacement.eval()(probe)
        if tuple(output.shape) != boundary_shape(graph["output_type"]):
            raise RuntimeError("YOLO replacement boundary changed")
        if parameter_count(replacement) != int(
            candidate["candidate_parameters_estimate"]
        ):
            raise RuntimeError("YOLO candidate parameter estimate changed")

    def export_candidate(self, *args, **kwargs) -> Any:
        from capp_r.contracts.candidate_preflight import (
            onnx_and_ort_yolo,
        )

        return onnx_and_ort_yolo(*args, **kwargs)

    def compare_onnx_outputs(
        self, expected: np.ndarray, actual: np.ndarray
    ) -> dict[str, Any]:
        if expected.shape != actual.shape or not np.isfinite(actual).all():
            return {"shape_and_finite": False, "numerically_close": False}
        difference = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
        return {
            "shape_and_finite": True,
            "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
            "numerically_close": bool(
                np.allclose(actual, expected, rtol=1e-3, atol=1e-3)
            ),
        }
