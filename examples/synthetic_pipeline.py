"""End-to-end CAPP-R pipeline on a randomly initialised YOLO11n (no dataset, no checkpoint).

Runs the three command-line stages in sequence:

1. contract extraction  -> which submodules are replaceable, with their tensor boundaries
2. candidate generation -> typed replacement micrographs that are >= 10% smaller (model-level)
3. candidate preflight  -> build, splice into the detector, forward/backward with the native
                           YOLO loss, contract and semantic checks, ONNX export + ORT parity

Weights are random, so this checks the mechanics of the method, not accuracy.

Usage:  python examples/synthetic_pipeline.py --out runs/synthetic [--model yolo11n.yaml]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("runs/synthetic"))
    parser.add_argument("--model", default="yolo11n.yaml",
                        help="Ultralytics model YAML (random init) or checkpoint (.pt)")
    parser.add_argument("--family", default="yolo11n", choices=["yolo11n", "yolov8n"])
    parser.add_argument("--construction-max", type=int, default=5)
    parser.add_argument("--onnx-max", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    contracts = args.out / "contracts.json"
    candidates = args.out / "candidates.json"
    run("-m", "capp_r.contracts.contract_extractor", "--family", args.family,
        "--weights", args.model, "--output", str(contracts))
    run("-m", "capp_r.generator.enumerate_micrographs", "--contracts", str(contracts),
        "--output", str(candidates))
    run("-m", "capp_r.contracts.candidate_preflight", "--architecture", args.family,
        "--candidates", str(candidates), "--weights", args.model,
        "--table", str(args.out / "preflight.csv"),
        "--failure-summary", str(args.out / "preflight_failures.csv"),
        "--onnx-dir", str(args.out / "onnx"),
        "--construction-max", str(args.construction_max), "--onnx-max", str(args.onnx_max))

    payload = json.loads(candidates.read_text(encoding="utf-8"))
    print(json.dumps({
        "eligible_regions": sum(r["eligible"] for r in json.loads(contracts.read_text())["regions"]),
        "unique_graphs": payload["unique_graphs"],
        "retained_candidates": payload["retained_candidates"],
        "outputs": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
