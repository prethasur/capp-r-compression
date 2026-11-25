"""Architecture-semantic adapters for generic contract extraction."""

from .base_adapter import ArchitectureAdapter
from .yolo11_adapter import YOLO11Adapter
from .yolov8_adapter import YOLOv8Adapter

__all__ = ["ArchitectureAdapter", "YOLOv8Adapter", "YOLO11Adapter"]
