from __future__ import annotations

import numpy as np
import pytest
import torch

from capp_r.local_fit.local_fit_adapters import (
    YOLOLocalFitAdapter,
    split_yolo_scale,
    yolo_decoded,
    yolo_raw_scales,
)


def yolo_output() -> tuple[torch.Tensor, list[torch.Tensor]]:
    torch.manual_seed(7)
    raw = [
        torch.randn(1, 65, 8, 8),
        torch.randn(1, 65, 4, 4),
        torch.randn(1, 65, 2, 2),
    ]
    decoded = torch.randn(1, 5, 84)
    return decoded, raw


@pytest.mark.parametrize("family", ["yolov8n", "yolo11n"])
def test_yolo_adapter_family_compatibility(family: str) -> None:
    assert YOLOLocalFitAdapter(family).family == family


def test_raw_scale_order_and_semantic_dimensions_are_preserved() -> None:
    output = yolo_output()
    scales = yolo_raw_scales(output)
    assert [value.shape[-2:] for value in scales] == [(8, 8), (4, 4), (2, 2)]
    for scale in scales:
        box, classification = split_yolo_scale(scale, nc=1, reg_max=16)
        assert box.shape[1] == 64
        assert classification.shape[1] == 1
    assert yolo_decoded(output).shape == (1, 5, 84)


def test_parent_copy_task_fidelity_is_zero() -> None:
    adapter = YOLOLocalFitAdapter("yolov8n")
    output = yolo_output()
    metrics = adapter.compute_task_output_fidelity(output, output)
    tensor_metrics = {
        name: value for name, value in metrics.items()
        if isinstance(value, torch.Tensor)
    }
    assert tensor_metrics
    assert all(torch.allclose(value, torch.tensor(0.0), atol=1e-7)
               for value in tensor_metrics.values())


def test_destructive_output_is_measurably_worse_than_copy() -> None:
    adapter = YOLOLocalFitAdapter("yolo11n")
    parent = yolo_output()
    destructive_scales = []
    for value in parent[1]:
        perturbation = torch.linspace(
            -2.0, 2.0, value.shape[1], dtype=value.dtype
        ).reshape(1, -1, 1, 1)
        destructive_scales.append(value + perturbation)
    destructive = (
        parent[0] + 2.0,
        destructive_scales,
    )
    metrics = adapter.compute_task_output_fidelity(parent, destructive)
    assert metrics["classification_logit_divergence"] > 0
    assert metrics["dfl_distribution_divergence"] > 0
    assert metrics["raw_box_distribution_divergence"] > 0
    assert metrics["decoded_box_l1"] > 0


def test_feature_fidelity_parent_copy_and_shape_guard() -> None:
    adapter = YOLOLocalFitAdapter("yolov8n")
    feature = torch.randn(2, 4, 8, 8)
    metrics = adapter.compute_feature_fidelity(feature, feature)
    assert all(torch.allclose(value, torch.tensor(0.0), atol=1e-7)
               for value in metrics.values())
    with pytest.raises(RuntimeError):
        adapter.compute_feature_fidelity(feature, feature[:, :, :4])


def test_onnx_comparison_is_fail_closed() -> None:
    adapter = YOLOLocalFitAdapter("yolov8n")
    expected = np.ones((1, 5, 10), dtype=np.float32)
    assert adapter.compare_onnx_outputs(expected, expected)["numerically_close"]
    assert not adapter.compare_onnx_outputs(
        expected, np.ones((1, 5, 9), dtype=np.float32)
    )["numerically_close"]


