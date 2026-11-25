"""Protected task semantics for YOLO11 detectors."""

from __future__ import annotations

from .yolov8_adapter import YOLOv8Adapter


class YOLO11Adapter(YOLOv8Adapter):
    family = "yolo11n"
