"""Bounded spatial-feature micrograph enumeration."""

from __future__ import annotations

from typing import Iterable

from .combinators import parallel_concat_projection, parallel_sum, residual, sequential
from .primitives import activation, conv, depthwise, group_conv, identity, normalization, pool
from .type_system import GraphSpec, TensorType, assert_boundary


def aligned_widths(channels: int) -> list[int]:
    values = set()
    for ratio in (0.25, 0.35, 0.5, 0.75):
        value = max(8, int(round(channels * ratio / 8)) * 8)
        values.add(min(value, channels))
    return sorted(values)


def enumerate_spatial(source: TensorType, output: TensorType) -> Iterable[GraphSpec]:
    if source.kind != "spatial" or output.kind != "spatial":
        return
    if not source.height or not source.width or not output.height or not output.width:
        return
    if source.height % output.height or source.width % output.width:
        return
    stride_h, stride_w = source.height // output.height, source.width // output.width
    if stride_h != stride_w or stride_h not in {1, 2}:
        return
    entry_stride = stride_h
    for width in aligned_widths(max(source.channels, output.channels)):
        first = conv(source, width, 1, stride=entry_stride)
        for kernel in (3, 5):
            dw = depthwise(first.output_type, kernel)
            projected = conv(dw.output_type, output.channels, 1)
            graph = sequential(first, dw, activation(dw.output_type), projected)
            assert_boundary(graph, source, output)
            yield graph
        for groups in (2, 4, 8):
            if first.output_type.channels % groups:
                continue
            middle = group_conv(first.output_type, first.output_type.channels, groups)
            graph = sequential(first, middle, activation(middle.output_type),
                               conv(middle.output_type, output.channels, 1))
            assert_boundary(graph, source, output)
            yield graph
        pooled = pool(first.output_type, "avg")
        graph = sequential(first, pooled, conv(pooled.output_type, output.channels, 1))
        assert_boundary(graph, source, output)
        yield graph
        left = sequential(first, depthwise(first.output_type, 3))
        right_entry = conv(source, width, 1, stride=entry_stride)
        right = sequential(right_entry, pool(right_entry.output_type, "avg"))
        graph = parallel_concat_projection(left, right, output)
        assert_boundary(graph, source, output)
        yield graph
    if source.compatible(output):
        branch = sequential(normalization(source), activation(source), conv(source, source.channels, 1))
        yield residual(branch)
        yield parallel_sum(identity(source), branch)
