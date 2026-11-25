#!/usr/bin/env python3
"""Freeze the parameter-only-qualified FCOS generalization protocol."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fcos_models import CAPP_PARAMETERS, PARENT_PARAMETERS, WIDTH_PARAMETERS, WIDTH_PER_GROUP, sha256


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--qualification", required=True, type=Path)
    parser.add_argument("--source-audit", required=True, type=Path)
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--width-b-spec", required=True, type=Path)
    parser.add_argument("--width-a-spec", required=True, type=Path)
    parser.add_argument("--refinement-protocol", required=True, type=Path)
    parser.add_argument("--adaptation-commit-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.launch:
        raise SystemExit("explicit --launch required")
    if args.output.exists():
        raise RuntimeError("append-only FCOS protocol freeze already exists")
    adaptation_commit = args.adaptation_commit_file.read_text(encoding="utf-8").strip()
    if len(adaptation_commit) != 40 or any(character not in "0123456789abcdef" for character in adaptation_commit.lower()):
        raise RuntimeError("IMPLEMENTATION_COMMIT.txt must contain one full Git commit SHA")
    refinement = json.loads(args.refinement_protocol.read_text(encoding="utf-8"))
    if refinement.get("status") != "FCOS_REFINED_WIDTH_QUALIFICATION_PROTOCOL_FROZEN":
        raise RuntimeError("refined width qualification protocol is not frozen")
    qualification = json.loads(args.qualification.read_text(encoding="utf-8"))
    if qualification.get("status") != "FCOS_QUALIFICATION_COMPLETE" or not qualification.get("production_gate"):
        raise RuntimeError("FCOS parameter-only production gate is not satisfied")
    if qualification.get("validation_metric_used") or qualification.get("hardware_metric_used"):
        raise RuntimeError("FCOS qualification was contaminated by prohibited evidence")
    expected_pretrained = "99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7"
    if sha256(args.pretrained) != expected_pretrained:
        raise RuntimeError("canonical torchvision FCOS checkpoint hash mismatch")
    implementation_root = Path(__file__).resolve().parent
    implementation_files = [
        "fcos_models.py", "qualify_fcos_r50_fpn.py", "freeze_fcos_protocol.py",
        "train_fcos_generalization.py", "evaluate_fcos_validation.py",
        "finalize_fcos_generalization.py", "FCOS_REFINED_WIDTH_QUALIFICATION_PROTOCOL.json",
    ]
    implementation_hashes = {name: sha256(implementation_root / name) for name in implementation_files}
    specs = {tier: json.loads(path.read_text(encoding="utf-8")) for tier, path in (("B", args.width_b_spec), ("A", args.width_a_spec))}
    for tier in ("B", "A"):
        if specs[tier].get("width_per_group") != WIDTH_PER_GROUP[tier] or specs[tier].get("parameters") != WIDTH_PARAMETERS[tier]:
            raise RuntimeError(f"Width-{tier} specification changed")
        if specs[tier].get("mismatch_fraction", 1.0) > .05:
            raise RuntimeError(f"Width-{tier} exceeds frozen matching tolerance")
    payload = {
        "status": "FCOS_GENERALIZATION_PROTOCOL_FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_source": {"repository": "https://github.com/pytorch/vision", "version": "0.20.1", "commit": "3ac97aa9120137381ed1060f37237e44485ac2aa", "audit_sha256": sha256(args.source_audit)},
        "adaptation": {"git_commit": adaptation_commit, "commit_file_sha256": sha256(args.adaptation_commit_file), "refinement_protocol_sha256": sha256(args.refinement_protocol), "implementation_sha256": implementation_hashes},
        "initialization": {"checkpoint": str(args.pretrained), "sha256": expected_pretrained, "rule": "for each seed build one parent state, load every exact-name/exact-shape COCO tensor, then transfer every exact-name/exact-shape parent tensor to CAPP/Width; unmatched compressed tensors use deterministic torchvision initialization", "weights_used_for_architecture_selection": False},
        "architecture": {
            "parent": {"parameters": PARENT_PARAMETERS, "architecture_hash": qualification["parent_architecture_hash"], "num_foreground_classes": 1, "input_size": [640, 640]},
            "capp": {tier: {"parameters": CAPP_PARAMETERS[tier], "structure": qualification["capp"][tier]["selected_regions"], "architecture_hash": qualification["capp"][tier].get("architecture_hash")} for tier in ("B", "A")},
            "width": {tier: {"parameters": WIDTH_PARAMETERS[tier], "width_per_group": WIDTH_PER_GROUP[tier], "mismatch_to_capp": abs(WIDTH_PARAMETERS[tier] - CAPP_PARAMETERS[tier]) / CAPP_PARAMETERS[tier], "spec_sha256": sha256(path)} for tier, path in (("B", args.width_b_spec), ("A", args.width_a_spec))},
            "width_family": "single global ResNet width_per_group; fixed 256-channel FPN and coupled FCOS towers",
            "two_factor_fallback_used": False,
        },
        "training": {"epochs": 50, "optimizer": "SGD", "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0001, "scheduler": "MultiStepLR milestones [35,45], gamma 0.1", "augmentation": "random horizontal flip p=0.5 only", "image_size": 640, "batch_per_gpu": 4, "gradient_accumulation": 4, "effective_batch": 16, "amp": True, "seeds": [0, 1, 2], "training_validation": "disabled"},
        "data": {"train_manifest": str(args.train_manifest), "train_manifest_sha256": sha256(args.train_manifest), "validation_manifest": str(args.validation_manifest), "validation_manifest_sha256": sha256(args.validation_manifest), "official_validation_images": 841},
        "evaluation": {"score_threshold": 0.001, "nms_threshold": 0.6, "detections_per_image": 100, "selection_after_training": False, "primary_metrics": ["AP50:95", "AP50", "AP75"]},
        "qualification_sha256": sha256(args.qualification),
        "official_validation_used_for_protocol": False,
        "HRSID_TEST_FINAL_EVALUATION_COMPLETE": True,
        "HRSID_TEST_METRIC_ACCESS": "CLOSED_AFTER_FINAL_EVALUATION",
        "HRSID_TEST_POST_EVALUATION_TUNING_ALLOWED": False,
        "HRSID_TEST_FILES_PRESENT_ON_CLUSTER": False,
        "HARDWARE_AWARE_CAPP_STARTED": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output), "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
