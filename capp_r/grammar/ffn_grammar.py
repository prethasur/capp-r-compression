"""Bounded token/FFN micrograph enumeration."""

from __future__ import annotations

from typing import Iterable

from .combinators import parallel_sum, residual, sequential
from .primitives import activation, identity, linear, normalization
from .type_system import GraphSpec, TensorType, assert_boundary


def enumerate_ffn(source: TensorType, output: TensorType) -> Iterable[GraphSpec]:
    if source.kind != "token" or output.kind != "token":
        return
    for ratio in (0.25, 0.35, 0.5, 0.75):
        hidden = max(8, int(round(max(source.channels, output.channels) * ratio / 8)) * 8)
        graph = sequential(linear(source, hidden), activation(TensorType("token", hidden, dtype=source.dtype)),
                           linear(TensorType("token", hidden, dtype=source.dtype), output.channels))
        assert_boundary(graph, source, output)
        yield graph
        rank = max(4, min(hidden, source.channels, output.channels) // 2)
        low_rank = linear(source, output.channels, rank=rank)
        assert_boundary(low_rank, source, output)
        yield low_rank
    if source.compatible(output):
        branch = sequential(normalization(source), linear(source, source.channels),
                            activation(source), linear(source, source.channels))
        yield residual(branch)
        yield parallel_sum(identity(source), branch)
