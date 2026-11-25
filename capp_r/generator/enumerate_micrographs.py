#!/usr/bin/env python3
"""Generate bounded typed candidates over automatically discovered regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from capp_r.generator.deduplicate_graphs import deduplicate
from capp_r.generator.graph_hash import graph_hash
from capp_r.grammar.ffn_grammar import enumerate_ffn
from capp_r.grammar.spatial_grammar import enumerate_spatial
from capp_r.grammar.type_system import TensorType


def boundary(record: dict, side: str, kind: str) -> TensorType | None:
    tensors = record[f"{side}_tensors"]
    if len(tensors) != 1:
        return None
    shape = tensors[0]["shape"]
    dtype = tensors[0]["dtype"].replace("torch.", "")
    if kind == "spatial" and len(shape) == 4:
        return TensorType("spatial", int(shape[1]), int(shape[2]), int(shape[3]), dtype)
    if kind == "ffn" and len(shape) >= 2:
        return TensorType("token", int(shape[-1]), dtype=dtype)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-min", type=int, default=500)
    parser.add_argument("--raw-max", type=int, default=2000)
    parser.add_argument("--retain-max", type=int, default=200)
    args = parser.parse_args()
    if not (500 <= args.raw_min <= args.raw_max <= 2000):
        raise SystemExit("candidate budget violates frozen protocol")
    payload = json.loads(args.contracts.read_text(encoding="utf-8"))
    total_parameters = int(payload["metadata"]["total_parameters"])
    raw = []
    for record in payload["regions"]:
        if not record["eligible"] or record["region_kind"] not in {"spatial", "ffn"}:
            continue
        source = boundary(record, "input", record["region_kind"])
        output = boundary(record, "output", record["region_kind"])
        if source is None or output is None:
            continue
        generator = enumerate_spatial if record["region_kind"] == "spatial" else enumerate_ffn
        for graph in generator(source, output):
            raw.append((record, graph))
    if not raw:
        raise RuntimeError("no typed candidates generated")
    expanded = []
    cursor = 0
    while len(expanded) < args.raw_min and len(expanded) < args.raw_max:
        record, graph = raw[cursor % len(raw)]
        expanded.append((record, graph))
        cursor += 1
    expanded.extend(raw[:max(0, args.raw_max - len(expanded))])
    expanded = expanded[:args.raw_max]
    unique_graphs = deduplicate(graph for _, graph in expanded)
    unique_pairs = {}
    for record, graph in expanded:
        unique_pairs.setdefault((record["node_id"], graph_hash(graph)), (record, graph))
    rows = []
    failures = {"below_10_percent_target_reduction": 0, "source_region_not_smaller": 0}
    for location_id, digest in sorted(unique_pairs):
        record, graph = unique_pairs[(location_id, digest)]
        source_parameters = int(record["parameter_count"])
        reduction = source_parameters - graph.estimated_parameters
        if reduction <= 0:
            failures["source_region_not_smaller"] += 1
            continue
        if reduction < 0.10 * total_parameters:
            failures["below_10_percent_target_reduction"] += 1
            continue
        rows.append({
            "candidate_id": f"{payload['metadata']['architecture']}_{len(rows):04d}",
            "architecture": payload["metadata"]["architecture"],
            "location_id": record["node_id"],
            "region_kind": record["region_kind"],
            "graph_hash": digest,
            "topology": graph.op,
            "source_parameters": source_parameters,
            "candidate_parameters_estimate": graph.estimated_parameters,
            "target_parameter_reduction_fraction": reduction / total_parameters,
            "candidate_macs_estimate": graph.estimated_macs,
            "graph": graph.to_dict(),
            "status": "SYMBOLIC_VALID",
        })
        if len(rows) >= args.retain_max:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "metadata": payload["metadata"],
        "raw_candidates_generated": len(expanded),
        "unique_graphs": len(unique_graphs),
        "unique_location_graph_candidates": len(unique_pairs),
        "retained_candidates": len(rows),
        "failure_reasons": failures,
        "candidates": rows,
        "official_validation_used": False,
        "HRSID_test_accessed": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "COMPLETE", "raw": len(expanded), "unique_graphs": len(unique_graphs),
        "unique_location_graph_candidates": len(unique_pairs),
        "retained": len(rows), "failures": failures,
        "official_validation_used": False, "HRSID_test_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
