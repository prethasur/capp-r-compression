"""Stable micrograph identity."""

from __future__ import annotations

import hashlib

from capp_r.grammar.type_system import GraphSpec
from .canonicalize_graph import canonical_json


def graph_hash(graph: GraphSpec) -> str:
    return hashlib.sha256(canonical_json(graph).encode("utf-8")).hexdigest()
