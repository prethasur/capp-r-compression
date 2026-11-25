"""Canonical JSON representation for graph-isomorphism heuristics."""

from __future__ import annotations

import json
from typing import Any

from capp_r.grammar.type_system import GraphSpec


COMMUTATIVE = {"ParallelSum", "ParallelConcatProjection"}


def canonical_dict(graph: GraphSpec) -> dict[str, Any]:
    children = [canonical_dict(child) for child in graph.children]
    if graph.op in COMMUTATIVE and len(children) >= 2:
        sortable, suffix = children[:2], children[2:]
        sortable.sort(key=lambda child: json.dumps(child, sort_keys=True, separators=(",", ":")))
        children = sortable + suffix
    return {
        "op": graph.op,
        "input_type": graph.to_dict()["input_type"],
        "output_type": graph.to_dict()["output_type"],
        "attributes": dict(graph.attributes),
        "children": children,
    }


def canonical_json(graph: GraphSpec) -> str:
    return json.dumps(canonical_dict(graph), sort_keys=True, separators=(",", ":"))
