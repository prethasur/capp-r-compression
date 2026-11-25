from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from capp_r.selection.yolo_shortlist import load_config, select


CONFIG = Path(__file__).parents[1] / "configs/yolo_shortlist_selector_v1.json"


def row(name: str, fidelity: float, reduction: float, size: float, **overrides):
    payload = {
        "architecture": "yolov8n", "queue_index": "0", "candidate_id": name,
        "graph_hash": name, "region_id": "model.8",
        "topology_signature": "Sequential", "status": "FIT_PASS",
        "parent_unchanged": "True", "candidate_parameters_updated": "True",
        "finite_losses_and_gradients": "True", "contract_pass": "True",
        "ONNX_export_pass": "True", "ORT_execution_pass": "True",
        "ONNX_numerical_pass": "True",
        "official_validation_accessed": "False", "HRSID_test_accessed": "False",
        "total_model_parameter_reduction": str(reduction), "ONNX_size": str(size),
        "normalized_feature_MSE": str(fidelity),
        "feature_cosine_distance": str(fidelity),
        "classification_divergence": str(fidelity),
        "DFL_divergence": str(fidelity),
        "raw_box_divergence": str(fidelity),
        "decoded_box_divergence": str(fidelity),
        "score_divergence": str(fidelity), "fitting_improvement": "1",
        "graph_depth": "3", "branch_count": "0", "concat_count": "0",
        "primitive_composition": "Sequential:1",
    }
    payload.update(overrides)
    return payload


def test_hard_gate_rejects_invalid_and_split_contaminated_rows() -> None:
    config = load_config(CONFIG)
    good = row("good", 0.1, 0.12, 100)
    bad_parent = row("bad-parent", 0.01, 0.2, 50, parent_unchanged="False")
    contaminated = row(
        "contaminated", 0.01, 0.2, 50, official_validation_accessed="True"
    )
    valid, _frontier, _shortlist, _recovery = select(
        [good, bad_parent, contaminated], config
    )
    assert [item["candidate_id"] for item in valid] == ["good"]


def test_tradeoff_rows_remain_nondominated() -> None:
    config = load_config(CONFIG)
    fidelity = row("fidelity", 0.01, 0.10, 120)
    compact = row("compact", 0.10, 0.20, 80)
    _valid, frontier, _shortlist, _recovery = select([fidelity, compact], config)
    assert {item["candidate_id"] for item in frontier} == {"fidelity", "compact"}


def test_dominated_row_is_removed() -> None:
    config = load_config(CONFIG)
    better = row("a", 0.01, 0.20, 80)
    worse = row("b", 0.10, 0.10, 120)
    _valid, frontier, _shortlist, _recovery = select([worse, better], config)
    assert [item["candidate_id"] for item in frontier] == ["a"]


def test_tie_break_is_graph_hash_lexical_and_order_independent() -> None:
    config = load_config(CONFIG)
    left = row("aaa", 0.1, 0.12, 100)
    right = row("bbb", 0.1, 0.12, 100)
    outcomes = []
    for rows in ([left, right], [right, left]):
        _valid, frontier, _shortlist, recovery = select(deepcopy(rows), config)
        outcomes.append((
            [item["candidate_id"] for item in frontier],
            [item["candidate_id"] for item in recovery],
        ))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] == ["aaa", "bbb"]


def test_recovery_prefers_region_topology_diversity() -> None:
    config = load_config(CONFIG)
    rows = [
        row("a", 0.01, 0.10, 100),
        row("b", 0.02, 0.11, 95),
        row("c", 0.03, 0.12, 90, region_id="model.7"),
    ]
    _valid, _frontier, _shortlist, recovery = select(rows, config)
    assert len({(item["region_id"], item["topology_signature"]) for item in recovery}) >= 2


