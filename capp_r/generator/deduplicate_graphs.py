"""Deduplicate generated graphs by canonical hash."""

from __future__ import annotations

from collections.abc import Iterable

from capp_r.grammar.type_system import GraphSpec
from .graph_hash import graph_hash


def deduplicate(graphs: Iterable[GraphSpec]) -> list[GraphSpec]:
    unique = {}
    for graph in graphs:
        unique.setdefault(graph_hash(graph), graph)
    return [unique[key] for key in sorted(unique)]
