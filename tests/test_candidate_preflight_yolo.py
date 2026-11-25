from __future__ import annotations

import torch
from torch import nn

from capp_r.contracts.candidate_preflight import (
    YOLO_ROUTING_ATTRIBUTES,
    preserve_yolo_routing_metadata,
    synthetic_yolo_batch,
)


def test_preserve_yolo_routing_metadata_copies_only_present_attributes() -> None:
    original = nn.Identity()
    original.i = 7
    original.f = [-1, 4]
    original.type = "C2f"
    replacement = nn.Sequential(nn.Identity())

    preserve_yolo_routing_metadata(original, replacement)

    assert replacement.i == 7
    assert replacement.f == [-1, 4]
    assert replacement.type == "C2f"
    assert not hasattr(replacement, "np")
    assert YOLO_ROUTING_ATTRIBUTES == ("i", "f", "type", "np")


def test_synthetic_yolo_batch_has_one_normalized_class_zero_target() -> None:
    image = torch.rand(1, 3, 640, 640)

    batch = synthetic_yolo_batch(image, torch.device("cpu"))

    assert batch["img"] is image
    assert batch["cls"].tolist() == [[0.0]]
    assert batch["bboxes"].tolist() == [[0.5, 0.5, 0.20000000298023224, 0.20000000298023224]]
    assert batch["batch_idx"].tolist() == [0.0]
