"""End-to-end check of the three CLI stages on a randomly initialised YOLO11n (offline, CPU).

Requires ultralytics, onnx and onnxruntime. Builds the model from Ultralytics' bundled
`yolo11n.yaml`, so no checkpoint or dataset is downloaded.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("onnxruntime")

REPO = Path(__file__).resolve().parents[1]
GATES = ("symbolic_type", "construction", "forward", "backward", "training_loss", "boundary_shape",
         "stride_feature_level", "protected_semantics", "output_signature", "numerical_sanity")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    env = {**os.environ, "PYTHONPATH": str(REPO), "YOLO_CONFIG_DIR": str(out / "ultralytics_cfg"),
           "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, str(REPO / "examples" / "synthetic_pipeline.py"), "--out", str(out),
         "--construction-max", "3", "--onnx-max", "1"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return out


def test_contracts_protect_the_detection_head(run_dir):
    payload = json.loads((run_dir / "contracts.json").read_text(encoding="utf-8"))
    regions = payload["regions"]
    assert payload["metadata"]["architecture"] == "yolo11n"
    assert any(r["eligible"] for r in regions)
    assert not any(r["eligible"] for r in regions if r["protected"])
    assert any(r["protected"] and "Detect" in r["protected_reason"] for r in regions)
    assert all(r["single_entry_single_exit"] for r in regions if r["eligible"])


def test_candidates_are_smaller_and_unique(run_dir):
    payload = json.loads((run_dir / "candidates.json").read_text(encoding="utf-8"))
    rows = payload["candidates"]
    assert rows and payload["retained_candidates"] == len(rows)
    assert len({(r["location_id"], r["graph_hash"]) for r in rows}) == len(rows)
    for r in rows:
        assert r["candidate_parameters_estimate"] < r["source_parameters"]
        assert r["target_parameter_reduction_fraction"] >= 0.10


def test_preflight_passes_gates_and_onnx_parity(run_dir):
    rows = list(csv.DictReader(open(run_dir / "preflight.csv", encoding="utf-8")))
    assert len(rows) == 3
    passed = [r for r in rows if all(r.get(g) == "True" for g in GATES)]
    assert passed, [r.get("failure_reason") for r in rows]
    for r in passed:
        assert int(r["candidate_parameters_actual"]) == int(r["candidate_parameters_estimate"])
    onnx_rows = [r for r in rows if r.get("onnx_numerically_close") == "True"]
    assert len(onnx_rows) == 1 and list((run_dir / "onnx").glob("*.onnx"))
