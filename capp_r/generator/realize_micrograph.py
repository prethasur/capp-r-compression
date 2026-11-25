"""Realize a serialized typed micrograph as an executable PyTorch module."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, branch: nn.Module) -> None:
        super().__init__()
        self.branch = branch

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.branch(value)


class ParallelSum(nn.Module):
    def __init__(self, left: nn.Module, right: nn.Module) -> None:
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.left(value) + self.right(value)


class ParallelConcatProjection(nn.Module):
    def __init__(self, left: nn.Module, right: nn.Module, projection: nn.Module,
                 dimension: int) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.projection = projection
        self.dimension = dimension

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        merged = torch.cat((self.left(value), self.right(value)), dim=self.dimension)
        return self.projection(merged)


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int) -> None:
        super().__init__()
        if groups < 1:
            raise ValueError("ChannelShuffle groups must be positive")
        self.groups = groups

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise RuntimeError("ChannelShuffle requires NCHW input")
        batch, channels, height, width = value.shape
        if channels % self.groups:
            raise RuntimeError("ChannelShuffle groups do not divide channels")
        value = value.reshape(batch, self.groups, channels // self.groups, height, width)
        return value.transpose(1, 2).contiguous().reshape(batch, channels, height, width)


def _channels(type_spec: dict[str, Any]) -> int:
    channels = int(type_spec["channels"])
    if channels < 1:
        raise ValueError("micrograph tensor channels must be positive")
    return channels


def _conv(spec: dict[str, Any]) -> nn.Conv2d:
    attributes = spec.get("attributes", {})
    kernel = int(attributes["kernel"])
    stride = int(attributes.get("stride", 1))
    groups = int(attributes.get("groups", 1))
    source_channels = _channels(spec["input_type"])
    output_channels = _channels(spec["output_type"])
    if source_channels % groups or output_channels % groups:
        raise ValueError("convolution groups must divide input and output channels")
    return nn.Conv2d(
        source_channels,
        output_channels,
        kernel_size=kernel,
        stride=stride,
        padding=kernel // 2,
        groups=groups,
        bias=False,
    )


def _activation(name: str) -> nn.Module:
    activations = {
        "SiLU": nn.SiLU,
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
    }
    if name not in activations:
        raise ValueError(f"unsupported activation: {name}")
    return activations[name]()


def build_module(spec: dict[str, Any]) -> nn.Module:
    """Build one module and fail closed for graph operators not implemented."""
    op = str(spec["op"])
    children = list(spec.get("children", []))
    attributes = spec.get("attributes", {})
    kind = str(spec["input_type"]["kind"])

    if op in {"Sequential", "Repeat"}:
        if not children:
            raise ValueError(f"{op} requires children")
        return nn.Sequential(*(build_module(child) for child in children))
    if op == "Residual":
        if len(children) != 1:
            raise ValueError("Residual requires one child")
        return Residual(build_module(children[0]))
    if op == "ParallelSum":
        if len(children) != 2:
            raise ValueError("ParallelSum requires two children")
        return ParallelSum(build_module(children[0]), build_module(children[1]))
    if op == "ParallelConcatProjection":
        if len(children) != 3:
            raise ValueError("ParallelConcatProjection requires two branches and a projection")
        dimension = 1 if kind == "spatial" else -1
        return ParallelConcatProjection(
            build_module(children[0]),
            build_module(children[1]),
            build_module(children[2]),
            dimension,
        )
    if op in {
        "Conv1x1", "Conv3x3", "Conv5x5", "DepthwiseConv3x3",
        "DepthwiseConv5x5", "GroupConv3x3", "PointwiseProjection",
    }:
        if kind != "spatial":
            raise ValueError(f"{op} requires a spatial tensor")
        return _conv(spec)
    if op == "AvgPool3x3":
        return nn.AvgPool2d(kernel_size=3, stride=int(attributes.get("stride", 1)), padding=1)
    if op == "MaxPool3x3":
        return nn.MaxPool2d(kernel_size=3, stride=int(attributes.get("stride", 1)), padding=1)
    if op == "Identity":
        return nn.Identity()
    if op == "Normalization":
        channels = _channels(spec["input_type"])
        return nn.BatchNorm2d(channels) if kind == "spatial" else nn.LayerNorm(channels)
    if op == "Activation":
        return _activation(str(attributes.get("name", "SiLU")))
    if op == "ChannelShuffle":
        return ChannelShuffle(int(attributes["groups"]))
    if op == "Linear":
        return nn.Linear(
            _channels(spec["input_type"]),
            _channels(spec["output_type"]),
            bias=False,
        )
    if op == "FactorizedLowRankLinear":
        source = _channels(spec["input_type"])
        output = _channels(spec["output_type"])
        rank = int(attributes["rank"])
        if rank < 1 or rank > min(source, output):
            raise ValueError("invalid low-rank dimension")
        return nn.Sequential(
            nn.Linear(source, rank, bias=False),
            nn.Linear(rank, output, bias=False),
        )
    raise ValueError(f"unsupported micrograph operator: {op}")


def boundary_shape(type_spec: dict[str, Any], batch: int = 1) -> tuple[int, ...]:
    kind = str(type_spec["kind"])
    channels = _channels(type_spec)
    if kind == "spatial":
        return batch, channels, int(type_spec["height"]), int(type_spec["width"])
    if kind == "token":
        return batch, 16, channels
    raise ValueError(f"unsupported tensor kind: {kind}")


def parameter_count(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def replace_module(root: nn.Module, location: str, replacement: nn.Module) -> nn.Module:
    """Replace a named submodule and return the original module."""
    parent_path, separator, leaf = location.rpartition(".")
    parent = root.get_submodule(parent_path) if separator else root
    original = parent.get_submodule(leaf)
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and leaf.isdigit():
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)
    return original
