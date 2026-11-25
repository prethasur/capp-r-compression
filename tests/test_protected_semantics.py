from types import SimpleNamespace

from capp_r.adapters.yolov8_adapter import YOLOv8Adapter


def named_module(name: str):
    return type(name, (), {})()


def test_yolo_detect_and_dfl_are_protected():
    adapter = YOLOv8Adapter()
    assert adapter.protected_semantics("head", named_module("Detect")).protected
    assert adapter.protected_semantics("head.dfl", SimpleNamespace()).protected


