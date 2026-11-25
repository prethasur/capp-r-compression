"""Deterministic YOLO shortlist selector v1."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


TRUE = {"true", "1", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in TRUE


def number(row: dict[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite selector input: {key}")
    return value


def hard_valid(row: dict[str, Any], config: dict[str, Any]) -> bool:
    gates = config["hard_gates"]
    return (
        row.get("status") == gates["status"]
        and all(as_bool(row.get(key)) is expected for key, expected in (
            ("parent_unchanged", gates["parent_unchanged"]),
            ("candidate_parameters_updated", gates["candidate_parameters_updated"]),
            ("finite_losses_and_gradients", gates["finite_losses_and_gradients"]),
            ("contract_pass", gates["contract_pass"]),
            ("ONNX_export_pass", gates["ONNX_export_pass"]),
            ("ORT_execution_pass", gates["ORT_execution_pass"]),
            ("ONNX_numerical_pass", gates["ONNX_numerical_pass"]),
            ("official_validation_accessed", gates["official_validation_accessed"]),
            ("HRSID_test_accessed", gates["HRSID_test_accessed"]),
        ))
        and number(row, "total_model_parameter_reduction")
            >= float(gates["minimum_total_model_parameter_reduction"])
    )


def add_fractional_ranks(
    rows: list[dict[str, Any]], metrics: list[str]
) -> None:
    for metric in metrics:
        unique = sorted({number(row, metric) for row in rows})
        denominator = max(1, len(unique) - 1)
        ranks = {value: index / denominator for index, value in enumerate(unique)}
        for row in rows:
            row[f"{metric}_rank"] = ranks[number(row, metric)]
    for row in rows:
        row["worst_normalized_fidelity_rank"] = max(
            float(row[f"{metric}_rank"]) for metric in metrics
        )


def dominates(
    left: dict[str, Any], right: dict[str, Any], config: dict[str, Any]
) -> bool:
    minimize = config["pareto"]["minimize"]
    maximize = config["pareto"]["maximize"]
    no_worse = all(number(left, key) <= number(right, key) for key in minimize)
    no_worse &= all(number(left, key) >= number(right, key) for key in maximize)
    strict = any(number(left, key) < number(right, key) for key in minimize)
    strict |= any(number(left, key) > number(right, key) for key in maximize)
    return no_worse and strict


def nondominated(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if not any(other is not row and dominates(other, row, config) for other in rows)
    ]


def tie_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        number(row, "worst_normalized_fidelity_rank"),
        -number(row, "total_model_parameter_reduction"),
        number(row, "ONNX_size"),
        number(row, "graph_depth"),
        number(row, "branch_count") + number(row, "concat_count"),
        row["graph_hash"],
    )


def diversity_key(row: dict[str, Any]) -> tuple[str, str]:
    return row["region_id"], row["topology_signature"]


def select(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]],
]:
    valid = [dict(row) for row in rows if hard_valid(row, config)]
    if not valid:
        return [], [], [], []
    add_fractional_ranks(valid, config["normalization"]["fidelity_metrics"])
    frontier = sorted(nondominated(valid, config), key=tie_key)
    limit = int(config["diversity"]["shortlist_max"])
    per_group = int(config["diversity"]["shortlist_max_per_region_topology"])
    counts: Counter[tuple[str, str]] = Counter()
    shortlist: list[dict[str, Any]] = []
    for row in frontier:
        key = diversity_key(row)
        if counts[key] < per_group and len(shortlist) < limit:
            shortlist.append(row)
            counts[key] += 1
    if len(shortlist) < min(limit, len(frontier)):
        for row in frontier:
            if row not in shortlist:
                shortlist.append(row)
                if len(shortlist) == limit:
                    break
    recovery_limit = int(config["diversity"]["recovery_max"])
    recovery: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    for row in shortlist:
        key = diversity_key(row)
        if key not in used and len(recovery) < recovery_limit:
            recovery.append(row)
            used.add(key)
    if len(recovery) < min(recovery_limit, len(shortlist)):
        for row in shortlist:
            if row not in recovery:
                recovery.append(row)
                if len(recovery) == recovery_limit:
                    break
    return valid, frontier, shortlist, recovery


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
