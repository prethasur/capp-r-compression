#!/usr/bin/env python3
"""Frozen FCOS-R50-FPN model constructors used by qualification and production."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection import FCOS, fcos_resnet50_fpn
from torchvision.models.detection.backbone_utils import BackboneWithFPN, _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelP6P7


NUM_CLASSES = 1
IMAGE_SIZE = 640
PARENT_PARAMETERS = 32_115_270
CAPP_PARAMETERS = {"B": 22_352_998, "A": 17_760_166}
WIDTH_PER_GROUP = {"B": 42, "A": 28}
WIDTH_PARAMETERS = {"B": 22_449_482, "A": 17_691_078}


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def architecture_hash(model: nn.Module) -> str:
    rows = [
        (name, module.__class__.__module__ + "." + module.__class__.__qualname__, tuple(parameter.shape))
        for name, module in model.named_modules()
        for parameter in module.parameters(recurse=False)
    ]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


class GhostSpatial(nn.Module):
    """Interface-preserving frozen CAPP primitive."""

    def __init__(self, old: nn.Conv2d):
        super().__init__()
        primary = (old.out_channels + 1) // 2
        cheap = old.out_channels - primary
        self.primary = nn.Conv2d(old.in_channels, primary, 1, old.stride, 0, bias=old.bias is not None)
        self.cheap = nn.Conv2d(primary, primary, 3, 1, 1, groups=primary, bias=old.bias is not None)
        self.out_channels = old.out_channels
        self.cheap_channels = cheap

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        primary = self.primary(inputs)
        cheap = self.cheap(primary)
        return torch.cat((primary, cheap[:, : self.cheap_channels]), 1)


def parent_model() -> FCOS:
    return fcos_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=NUM_CLASSES,
        min_size=IMAGE_SIZE, max_size=IMAGE_SIZE,
    )


def width_model(width_per_group: int) -> FCOS:
    """Global-width baseline; integer bottleneck widths are legal in ResNet."""
    if not isinstance(width_per_group, int) or width_per_group < 1:
        raise ValueError("width_per_group must be a positive integer")
    body = resnet50(weights=None, norm_layer=nn.BatchNorm2d, width_per_group=width_per_group)
    backbone = _resnet_fpn_extractor(
        body, 5, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )
    return FCOS(backbone, NUM_CLASSES, min_size=IMAGE_SIZE, max_size=IMAGE_SIZE)


def two_factor_width_model(width_per_group: int, fpn_head_channels: int) -> FCOS:
    """Prospective fallback family. Head width must satisfy FCOS GroupNorm(32)."""
    if fpn_head_channels < 32 or fpn_head_channels % 32:
        raise ValueError("FCOS FPN/head width must be divisible by GroupNorm's 32 groups")
    body = resnet50(weights=None, norm_layer=nn.BatchNorm2d, width_per_group=width_per_group)
    returned_layers = [2, 3, 4]
    return_layers = {f"layer{k}": str(index) for index, k in enumerate(returned_layers)}
    in_channels_stage2 = body.inplanes // 8
    in_channels = [in_channels_stage2 * 2 ** (layer - 1) for layer in returned_layers]
    backbone = BackboneWithFPN(
        body, return_layers, in_channels, fpn_head_channels,
        extra_blocks=LastLevelP6P7(fpn_head_channels, fpn_head_channels),
    )
    return FCOS(backbone, NUM_CLASSES, min_size=IMAGE_SIZE, max_size=IMAGE_SIZE)


def set_child(root: nn.Module, name: str, value: nn.Module) -> None:
    bits = name.split(".")
    owner = root
    for bit in bits[:-1]:
        owner = owner[int(bit)] if bit.isdigit() else getattr(owner, bit)
    if bits[-1].isdigit():
        owner[int(bits[-1])] = value
    else:
        setattr(owner, bits[-1], value)


def capp_model(selected_regions: Iterable[str]) -> FCOS:
    model = parent_model()
    for name in selected_regions:
        modules = dict(model.named_modules())
        if name not in modules or not isinstance(modules[name], nn.Conv2d):
            raise RuntimeError(f"frozen CAPP region is absent or not Conv2d: {name}")
        set_child(model, name, GhostSpatial(modules[name]))
    return model


def valid_detection_output(output: object) -> bool:
    return (
        isinstance(output, list) and len(output) == 1 and isinstance(output[0], dict)
        and set(output[0]) == {"boxes", "scores", "labels"}
        and output[0]["boxes"].ndim == 2 and output[0]["boxes"].shape[1] == 4
    )
