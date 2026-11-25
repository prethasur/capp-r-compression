#!/usr/bin/env python3
"""Fail-closed completeness, identity freeze, and final FCOS aggregation."""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

from fcos_models import sha256

MATRIX = [("parent", "P", seed) for seed in range(3)] + [(family, tier, seed) for family in ("capp", "width") for tier in ("B", "A") for seed in range(3)]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(temporary, path)


def training_rows(root: Path):
    rows=[]
    for family,tier,seed in MATRIX:
        path=root/f"{family}_{tier.lower()}"/f"seed{seed}"/"result.json"
        row=json.loads(path.read_text(encoding="utf-8"))
        if row.get("status")!="FCOS_TRAIN_PASS" or (row["family"],row["tier"],row["seed"])!=(family,tier,seed): raise RuntimeError(f"training gate failed: {path}")
        if row.get("official_validation_used"): raise RuntimeError("official validation contaminated training")
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--launch",action="store_true"); parser.add_argument("--stage",choices=("completeness","identity","aggregate"),required=True); parser.add_argument("--training-root",type=Path,required=True); parser.add_argument("--validation-root",type=Path); parser.add_argument("--protocol",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    if not args.launch: raise SystemExit("explicit --launch required")
    if args.output.exists(): raise RuntimeError("append-only finalization output exists")
    rows=training_rows(args.training_root)
    common={"protocol_sha256":sha256(args.protocol),"HRSID_TEST_METRIC_ACCESS":"CLOSED_AFTER_FINAL_EVALUATION","HRSID_TEST_FILES_PRESENT_ON_CLUSTER":False,"HARDWARE_AWARE_CAPP_STARTED":False}
    if args.stage=="completeness": payload={"status":"FCOS_15_STATE_TRAINING_COMPLETE","states":rows,**common}
    elif args.stage=="identity":
        identities=[{"family":r["family"],"tier":r["tier"],"seed":r["seed"],"parameters":r["parameters"],"architecture_hash":r["architecture_hash"],"checkpoint_sha256":r["checkpoint_sha256"]} for r in rows]
        for family,tier in (("capp","B"),("capp","A"),("width","B"),("width","A")):
            hashes={r["architecture_hash"] for r in rows if r["family"]==family and r["tier"]==tier}
            if len(hashes)!=1: raise RuntimeError(f"architecture differs across seeds: {family}-{tier}")
        payload={"status":"FCOS_CHECKPOINT_ARCHITECTURE_IDENTITY_FROZEN","states":identities,**common}
    else:
        if args.validation_root is None: raise RuntimeError("--validation-root required for aggregation")
        groups={}
        for training in rows:
            key=(training["family"],training["tier"]); path=args.validation_root/f"{key[0]}_{key[1].lower()}"/f"seed{training['seed']}"/"result.json"; row=json.loads(path.read_text(encoding="utf-8"))
            if row.get("status")!="FCOS_VALIDATION_PASS" or row.get("images")!=841 or row.get("failed_images")!=0: raise RuntimeError(f"validation gate failed: {path}")
            groups.setdefault(key,[]).append(row)
        summary=[]
        parent_mean=statistics.mean(r["AP50_95"] for r in groups[("parent","P")])
        for key,group in groups.items():
            values=[r["AP50_95"] for r in group]; summary.append({"model":f"{key[0]}-{key[1]}","parameters":group[0]["parameters"],"reduction_vs_parent":1-group[0]["parameters"]/groups[("parent","P")][0]["parameters"],"AP50_95_mean":statistics.mean(values),"AP50_95_sample_sd":statistics.stdev(values),"AP50_mean":statistics.mean(r["AP50"] for r in group),"AP75_mean":statistics.mean(r["AP75"] for r in group),"delta_parent":statistics.mean(values)-parent_mean})
        lookup={r["model"]:r for r in summary}
        comparisons={tier:{"capp_minus_width_AP50_95":lookup[f"capp-{tier}"]["AP50_95_mean"]-lookup[f"width-{tier}"]["AP50_95_mean"],"parameter_mismatch_fraction":abs(lookup[f"capp-{tier}"]["parameters"]-lookup[f"width-{tier}"]["parameters"])/lookup[f"capp-{tier}"]["parameters"]} for tier in ("B","A")}
        payload={"status":"FCOS_GENERALIZATION_FINAL_REPORT_COMPLETE","models":summary,"primary_comparisons":comparisons,"states":[r for group in groups.values() for r in group],**common}
        report=args.output.with_suffix(".md")
        if report.exists(): raise RuntimeError("append-only final report already exists")
        order={"parent-P":0,"capp-B":1,"width-B":2,"capp-A":3,"width-A":4}
        lines=["# FCOS-R50-FPN Generalization Final Report","","| Model | Params | Reduction | AP50:95 mean +/- SD | AP50 | AP75 | Delta Parent |","|---|---:|---:|---:|---:|---:|---:|"]
        for row in sorted(summary,key=lambda value:order[value["model"]]):
            lines.append(f'| {row["model"]} | {row["parameters"]:,} | {row["reduction_vs_parent"]:.3%} | {row["AP50_95_mean"]:.6f} +/- {row["AP50_95_sample_sd"]:.6f} | {row["AP50_mean"]:.6f} | {row["AP75_mean"]:.6f} | {row["delta_parent"]:+.6f} |')
        lines += ["",f'CAPP-B minus Width-B AP50:95: `{comparisons["B"]["capp_minus_width_AP50_95"]:+.6f}`; parameter mismatch `{comparisons["B"]["parameter_mismatch_fraction"]:.3%}`.',f'CAPP-A minus Width-A AP50:95: `{comparisons["A"]["capp_minus_width_AP50_95"]:+.6f}`; parameter mismatch `{comparisons["A"]["parameter_mismatch_fraction"]:.3%}`.',"","HRSID test metrics remained closed and HRSID test files were absent on Deakin. No hardware-aware CAPP or Jetson experiment was run.",""]
        report.parent.mkdir(parents=True,exist_ok=True); report.write_text("\n".join(lines),encoding="utf-8")
    atomic_json(args.output,payload)


if __name__=="__main__": main()
