"""Factory for frozen architecture-semantic adapters."""

from __future__ import annotations

from capp_r.adapters import YOLO11Adapter, YOLOv8Adapter


def adapter_for(family: str):
    normalized = family.lower().replace("-", "").replace("_", "")
    if normalized in {"yolov8", "yolov8n"}:
        return YOLOv8Adapter()
    if normalized in {"yolo11", "yolo11n"}:
        return YOLO11Adapter()
    raise ValueError(f"unsupported frozen architecture family: {family}")
