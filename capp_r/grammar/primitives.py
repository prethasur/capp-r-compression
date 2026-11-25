"""Typed primitive constructors with analytical resource estimates."""

from __future__ import annotations

from .type_system import GraphSpec, TensorType, attributes


def _spatial_macs(output: TensorType, parameters: int) -> int:
    return parameters * int(output.height or 1) * int(output.width or 1)


def conv(source: TensorType, channels: int, kernel: int = 1, groups: int = 1,
         op: str | None = None, stride: int = 1) -> GraphSpec:
    if source.kind != "spatial" or channels <= 0:
        raise TypeError("Conv requires a nondegenerate spatial tensor")
    if stride not in {1, 2}:
        raise TypeError("frozen spatial grammar permits stride 1 or 2 only")
    if source.channels % groups or channels % groups:
        raise TypeError("groups must divide input and output channels")
    output = TensorType(
        "spatial", channels,
        None if source.height is None else source.height // stride,
        None if source.width is None else source.width // stride,
        source.dtype,
    )
    parameters = source.channels * channels * kernel * kernel // groups
    name = op or ("PointwiseProjection" if kernel == 1 else f"Conv{kernel}x{kernel}")
    return GraphSpec(name, source, output, attributes(kernel=kernel, groups=groups, stride=stride),
                     estimated_parameters=parameters, estimated_macs=_spatial_macs(output, parameters))


def depthwise(source: TensorType, kernel: int) -> GraphSpec:
    return conv(source, source.channels, kernel, source.channels, f"DepthwiseConv{kernel}x{kernel}")


def group_conv(source: TensorType, channels: int, groups: int, kernel: int = 3) -> GraphSpec:
    return conv(source, channels, kernel, groups, "GroupConv3x3")


def pool(source: TensorType, kind: str = "avg") -> GraphSpec:
    if source.kind != "spatial" or kind not in {"avg", "max"}:
        raise TypeError("invalid pooling primitive")
    return GraphSpec(f"{kind.title()}Pool3x3", source, source, attributes(kernel=3, stride=1),
                     estimated_macs=source.channels * int(source.height or 1) * int(source.width or 1) * 9)


def identity(source: TensorType) -> GraphSpec:
    return GraphSpec("Identity", source, source, transform_primitives=0)


def normalization(source: TensorType) -> GraphSpec:
    return GraphSpec("Normalization", source, source, estimated_parameters=2 * source.channels)


def activation(source: TensorType, name: str = "SiLU") -> GraphSpec:
    return GraphSpec("Activation", source, source, attributes(name=name),
                     estimated_macs=source.channels * int(source.height or 1) * int(source.width or 1))


def linear(source: TensorType, features: int, rank: int | None = None) -> GraphSpec:
    if source.kind != "token" or features <= 0:
        raise TypeError("Linear requires a token tensor")
    output = TensorType("token", features, dtype=source.dtype)
    if rank is None:
        return GraphSpec("Linear", source, output, estimated_parameters=source.channels * features,
                         estimated_macs=source.channels * features)
    if rank <= 0 or rank > min(source.channels, features):
        raise TypeError("invalid low-rank dimension")
    parameters = rank * (source.channels + features)
    return GraphSpec("FactorizedLowRankLinear", source, output, attributes(rank=rank),
                     transform_primitives=2, estimated_parameters=parameters, estimated_macs=parameters)
