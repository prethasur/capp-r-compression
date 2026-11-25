"""Typed graph combinators."""

from __future__ import annotations

from .primitives import conv, linear
from .type_system import GraphSpec, TensorType, attributes


def sequential(*graphs: GraphSpec) -> GraphSpec:
    if not graphs:
        raise TypeError("Sequential requires at least one child")
    for left, right in zip(graphs, graphs[1:]):
        if not left.output_type.compatible(right.input_type):
            raise TypeError("Sequential child boundary mismatch")
    return GraphSpec(
        "Sequential", graphs[0].input_type, graphs[-1].output_type, children=tuple(graphs),
        transform_primitives=sum(graph.transform_primitives for graph in graphs),
        estimated_parameters=sum(graph.estimated_parameters for graph in graphs),
        estimated_macs=sum(graph.estimated_macs for graph in graphs),
    )


def residual(graph: GraphSpec) -> GraphSpec:
    if not graph.input_type.compatible(graph.output_type):
        raise TypeError("Residual requires X -> X")
    return GraphSpec("Residual", graph.input_type, graph.output_type, children=(graph,),
                     transform_primitives=graph.transform_primitives,
                     estimated_parameters=graph.estimated_parameters,
                     estimated_macs=graph.estimated_macs)


def parallel_sum(left: GraphSpec, right: GraphSpec) -> GraphSpec:
    if not left.input_type.compatible(right.input_type) or not left.output_type.compatible(right.output_type):
        raise TypeError("ParallelSum requires X->Y and X->Y")
    return GraphSpec("ParallelSum", left.input_type, left.output_type, children=(left, right),
                     transform_primitives=left.transform_primitives + right.transform_primitives,
                     estimated_parameters=left.estimated_parameters + right.estimated_parameters,
                     estimated_macs=left.estimated_macs + right.estimated_macs)


def parallel_concat_projection(left: GraphSpec, right: GraphSpec, output: TensorType) -> GraphSpec:
    if not left.input_type.compatible(right.input_type):
        raise TypeError("ParallelConcat requires a common input")
    if left.output_type.kind != right.output_type.kind or left.output_type.kind != output.kind:
        raise TypeError("ParallelConcat tensor kinds differ")
    if output.kind == "spatial":
        if (left.output_type.height, left.output_type.width) != (right.output_type.height, right.output_type.width):
            raise TypeError("ParallelConcat spatial sizes differ")
        merged = TensorType("spatial", left.output_type.channels + right.output_type.channels,
                            left.output_type.height, left.output_type.width, left.output_type.dtype)
        projection = conv(merged, output.channels, 1)
    else:
        merged = TensorType("token", left.output_type.channels + right.output_type.channels,
                            dtype=left.output_type.dtype)
        projection = linear(merged, output.channels)
    primitives = left.transform_primitives + right.transform_primitives + projection.transform_primitives
    return GraphSpec("ParallelConcatProjection", left.input_type, output,
                     attributes(merged_channels=merged.channels), (left, right, projection),
                     primitives,
                     left.estimated_parameters + right.estimated_parameters + projection.estimated_parameters,
                     left.estimated_macs + right.estimated_macs + projection.estimated_macs)


def repeat(factory, source: TensorType, count: int) -> GraphSpec:
    if count not in {1, 2, 3}:
        raise TypeError("Repeat count is frozen to 1, 2, or 3")
    graphs = []
    current = source
    for _ in range(count):
        graph = factory(current)
        graphs.append(graph)
        current = graph.output_type
    result = sequential(*graphs)
    return GraphSpec("Repeat", result.input_type, result.output_type, attributes(count=count), result.children,
                     result.transform_primitives, result.estimated_parameters, result.estimated_macs)
