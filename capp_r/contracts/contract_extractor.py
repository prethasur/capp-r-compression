"""Runtime graph tracing and contract extraction without fixed module indices."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .protected_semantics import adapter_for


def _tensors(value: Any) -> Iterable[Any]:
    try:
        import torch
    except ImportError:
        return
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _tensors(nested)
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _tensors(nested)


def _tensor_meta(value: Any) -> list[dict[str, Any]]:
    return [{"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device)}
            for tensor in _tensors(value)]


@dataclass
class RegionContract:
    node_id: str
    module_type: str
    parents: list[str] = field(default_factory=list)
    consumers: list[str] = field(default_factory=list)
    input_tensors: list[dict[str, Any]] = field(default_factory=list)
    output_tensors: list[dict[str, Any]] = field(default_factory=list)
    output_channels: int | None = None
    output_height: int | None = None
    output_width: int | None = None
    stride_height: float | None = None
    stride_width: float | None = None
    parameter_count: int = 0
    direct_parameter_count: int = 0
    macs: int = 0
    residual_relationship: bool = False
    concatenation_relationship: bool = False
    shared_consumers: bool = False
    task_head_relevance: str = ""
    onnx_operator_mapping: str = ""
    protected: bool = False
    protected_reason: str = ""
    region_kind: str = "generic"
    single_entry_single_exit: bool = False
    eligible: bool = False
    ineligible_reason: str = ""


class ContractExtractor:
    def __init__(self, model: Any, family: str, input_size: tuple[int, int] = (640, 640),
                 minimum_parameter_fraction: float = 0.01) -> None:
        self.adapter = adapter_for(family)
        self.model = self.adapter.unwrap(model)
        self.input_size = input_size
        self.minimum_parameter_fraction = minimum_parameter_fraction
        self.records: dict[str, RegionContract] = {}
        self._producer: dict[int, str] = {}
        self._handles = []

    @staticmethod
    def _direct_parameters(module: Any) -> int:
        return sum(int(parameter.numel()) for parameter in module.parameters(recurse=False))

    @staticmethod
    def _all_parameters(module: Any) -> int:
        return sum(int(parameter.numel()) for parameter in module.parameters())

    @staticmethod
    def _leaf_macs(module: Any, inputs: Any, output: Any) -> int:
        try:
            import torch.nn as nn
        except ImportError:
            return 0
        outputs = list(_tensors(output))
        inputs_flat = list(_tensors(inputs))
        if not outputs:
            return 0
        out = outputs[0]
        if isinstance(module, nn.Conv2d) and out.ndim == 4:
            batch, channels, height, width = map(int, out.shape)
            kernel = module.kernel_size[0] * module.kernel_size[1]
            return batch * channels * height * width * (module.in_channels // module.groups) * kernel
        if isinstance(module, nn.Linear):
            elements = int(out.numel() // max(out.shape[-1], 1))
            return elements * module.in_features * module.out_features
        if inputs_flat and module.__class__.__name__.lower().endswith("pool2d"):
            return int(out.numel())
        return 0

    @staticmethod
    def _onnx_mapping(module: Any) -> str:
        name = module.__class__.__name__.lower()
        mappings = {
            "conv": "Conv", "linear": "Gemm/MatMul", "batchnorm": "BatchNormalization",
            "relu": "Relu", "silu": "Sigmoid+Mul", "pool": "AveragePool/MaxPool",
            "upsample": "Resize", "concat": "Concat", "softmax": "Softmax",
        }
        return next((value for token, value in mappings.items() if token in name), module.__class__.__name__)

    def _register(self) -> None:
        module_names = {module: name or "__root__" for name, module in self.model.named_modules()}
        for module, name in module_names.items():
            decision = self.adapter.protected_semantics(name, module)
            self.records[name] = RegionContract(
                node_id=name,
                module_type=self.adapter.module_type(module),
                parameter_count=self._all_parameters(module),
                direct_parameter_count=self._direct_parameters(module),
                task_head_relevance=decision.reason,
                onnx_operator_mapping=self._onnx_mapping(module),
                protected=decision.protected,
                protected_reason=decision.reason,
                region_kind=decision.region_kind,
            )

            def pre_hook(current: Any, inputs: Any, node_id: str = name) -> None:
                record = self.records[node_id]
                record.input_tensors = _tensor_meta(inputs)
                parents = sorted({self._producer[id(tensor)] for tensor in _tensors(inputs)
                                  if id(tensor) in self._producer and self._producer[id(tensor)] != node_id})
                record.parents = parents
                for parent in parents:
                    consumers = self.records[parent].consumers
                    if node_id not in consumers:
                        consumers.append(node_id)

            def post_hook(current: Any, inputs: Any, output: Any, node_id: str = name) -> None:
                record = self.records[node_id]
                record.output_tensors = _tensor_meta(output)
                outputs = list(_tensors(output))
                for tensor in outputs:
                    self._producer[id(tensor)] = node_id
                if outputs:
                    tensor = outputs[0]
                    shape = tuple(int(item) for item in tensor.shape)
                    if len(shape) >= 4:
                        record.output_channels = shape[-3]
                        record.output_height, record.output_width = shape[-2:]
                        record.stride_height = self.input_size[0] / shape[-2]
                        record.stride_width = self.input_size[1] / shape[-1]
                inputs_flat = list(_tensors(inputs))
                record.residual_relationship = bool(
                    inputs_flat and outputs and tuple(inputs_flat[0].shape) == tuple(outputs[0].shape)
                    and ("res" in current.__class__.__name__.lower()
                         or "bottleneck" in current.__class__.__name__.lower())
                )
                record.concatenation_relationship = "concat" in current.__class__.__name__.lower()
                record.macs = max(record.macs, self._leaf_macs(current, inputs, output))

            self._handles.append(module.register_forward_pre_hook(pre_hook))
            self._handles.append(module.register_forward_hook(post_hook))

    def extract(self, sample: Any) -> tuple[list[RegionContract], dict[str, Any]]:
        self._register()
        try:
            self.model.eval()
            import torch
            with torch.no_grad():
                self.model(sample)
        finally:
            for handle in self._handles:
                handle.remove()
        total_parameters = self._all_parameters(self.model)
        for name, record in self.records.items():
            descendant_macs = sum(
                nested.macs for nested_name, nested in self.records.items()
                if nested_name != name and nested_name.startswith(f"{name}.")
            )
            record.macs = max(record.macs, descendant_macs)
            record.consumers.sort()
            record.shared_consumers = len(record.consumers) > 1
            record.single_entry_single_exit = (
                len(record.input_tensors) == 1
                and len(record.output_tensors) == 1
            )
            reasons = []
            if record.protected:
                reasons.append(record.protected_reason or "protected semantics")
            if not record.single_entry_single_exit:
                reasons.append("not single-entry/single-exit")
            if record.parameter_count < total_parameters * self.minimum_parameter_fraction:
                reasons.append("below minimum resource threshold")
            if not record.output_tensors:
                reasons.append("not executed")
            record.ineligible_reason = "; ".join(reasons)
            record.eligible = not reasons and name != "__root__"
        metadata = {
            "architecture": self.adapter.family,
            "input_size": list(self.input_size),
            "total_parameters": total_parameters,
            "output_semantics": self.adapter.output_semantics(self.model),
            "hard_coded_module_indices": False,
            "region_support": "single-entry/single-exit",
        }
        return list(self.records.values()), metadata


def write_outputs(records: list[RegionContract], metadata: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "regions": [asdict(record) for record in records]}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    rows = [asdict(record) for record in records]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=["yolov8n", "yolo11n"])
    parser.add_argument("--weights", required=True,
                        help="Ultralytics checkpoint (.pt) or model YAML (e.g. yolo11n.yaml, random init)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO
    model = YOLO(args.weights).model
    sample = torch.zeros(1, 3, 640, 640)
    records, metadata = ContractExtractor(model, args.family).extract(sample)
    write_outputs(records, metadata, args.output)
    print(json.dumps({"status": "COMPLETE", "regions": len(records),
                      "eligible": sum(record.eligible for record in records), **metadata}, indent=2))


if __name__ == "__main__":
    main()
