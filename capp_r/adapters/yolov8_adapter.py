"""Protected task semantics for YOLOv8 detectors."""

from __future__ import annotations

from typing import Any

from .base_adapter import ArchitectureAdapter, SemanticDecision


class YOLOv8Adapter(ArchitectureAdapter):
    family = "yolov8n"
    native_postprocessing = "YOLO_decode_and_NMS"

    def protected_semantics(self, name: str, module: Any) -> SemanticDecision:
        lowered = f"{name}.{module.__class__.__name__}".lower()
        protected_tokens = {
            "detect": "Detect head and P3/P4/P5 ordering",
            "dfl": "DFL/reg_max semantics",
            "nms": "native NMS semantics",
        }
        for token, reason in protected_tokens.items():
            if token in lowered:
                return SemanticDecision(True, reason, "task_head")
        return SemanticDecision(False, region_kind="spatial")

    def output_semantics(self, model: Any) -> dict[str, Any]:
        root = self.unwrap(model)
        detect = None
        for module in root.modules():
            if module.__class__.__name__.lower() == "detect":
                detect = module
        return {
            **super().output_semantics(model),
            "nc": getattr(detect, "nc", None),
            "reg_max": getattr(detect, "reg_max", None),
            "feature_levels": 3,
            "protected_output": True,
        }
