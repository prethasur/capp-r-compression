"""Minimal immutable type system for compositional micrographs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TensorType:
    kind: str
    channels: int
    height: int | None = None
    width: int | None = None
    dtype: str = "float32"

    def compatible(self, other: "TensorType") -> bool:
        return self == other


@dataclass(frozen=True)
class GraphSpec:
    op: str
    input_type: TensorType
    output_type: TensorType
    attributes: tuple[tuple[str, Any], ...] = ()
    children: tuple["GraphSpec", ...] = ()
    transform_primitives: int = 1
    estimated_parameters: int = 0
    estimated_macs: int = 0

    def attribute_dict(self) -> dict[str, Any]:
        return dict(self.attributes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "input_type": asdict(self.input_type),
            "output_type": asdict(self.output_type),
            "attributes": dict(self.attributes),
            "children": [child.to_dict() for child in self.children],
            "transform_primitives": self.transform_primitives,
            "estimated_parameters": self.estimated_parameters,
            "estimated_macs": self.estimated_macs,
        }


def attributes(**values: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(values.items()))


def assert_boundary(graph: GraphSpec, source_input: TensorType, source_output: TensorType) -> None:
    if not graph.input_type.compatible(source_input) or not graph.output_type.compatible(source_output):
        raise TypeError(f"graph boundary mismatch: {graph.input_type}->{graph.output_type}, "
                        f"required {source_input}->{source_output}")
    if graph.transform_primitives > 7:
        raise TypeError("graph exceeds frozen maximum of seven transform primitives")
