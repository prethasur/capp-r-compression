#!/usr/bin/env python3
"""FCOS-native, full 841-image official HRSID validation evaluator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from fcos_models import CAPP_PARAMETERS, PARENT_PARAMETERS, WIDTH_PARAMETERS, WIDTH_PER_GROUP, capp_model, count_parameters, parent_model, sha256, width_model


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def construct(family: str, tier: str, protocol: dict):
    if family == "parent": return parent_model(), PARENT_PARAMETERS
    if family == "width": return width_model(WIDTH_PER_GROUP[tier]), WIDTH_PARAMETERS[tier]
    return capp_model(protocol["architecture"]["capp"][tier]["structure"]), CAPP_PARAMETERS[tier]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-result", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--identity-freeze", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if not args.launch: raise SystemExit("explicit --launch required")
    if args.output_root.exists(): raise RuntimeError("append-only validation output exists")
    if "test" in str(args.dataset_root).lower() or "test" in str(args.validation_manifest).lower(): raise RuntimeError("HRSID test path prohibited")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8")); training = json.loads(args.training_result.read_text(encoding="utf-8"))
    if protocol.get("status") != "FCOS_GENERALIZATION_PROTOCOL_FROZEN" or training.get("status") != "FCOS_TRAIN_PASS": raise RuntimeError("frozen protocol/training gate failed")
    if sha256(args.checkpoint) != training["checkpoint_sha256"]: raise RuntimeError("checkpoint hash mismatch")
    family, tier = training["family"], training["tier"]
    identity=json.loads(args.identity_freeze.read_text(encoding="utf-8"))
    if identity.get("status")!="FCOS_CHECKPOINT_ARCHITECTURE_IDENTITY_FROZEN": raise RuntimeError("identity freeze gate failed")
    matches=[row for row in identity["states"] if row["family"]==family and row["tier"]==tier and row["seed"]==training["seed"] and row["checkpoint_sha256"]==training["checkpoint_sha256"]]
    if len(matches)!=1: raise RuntimeError("checkpoint is absent from the immutable identity freeze")
    model, expected = construct(family, tier, protocol)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    if count_parameters(model) != expected: raise RuntimeError("model identity changed")
    if not torch.cuda.is_available(): raise RuntimeError("FCOS official validation requires CUDA")
    model.score_thresh = float(protocol["evaluation"]["score_threshold"])
    model.nms_thresh = float(protocol["evaluation"]["nms_threshold"])
    model.detections_per_img = int(protocol["evaluation"]["detections_per_image"])
    model = model.cuda().eval()
    if sha256(args.validation_manifest) != protocol["data"]["validation_manifest_sha256"]: raise RuntimeError("official validation manifest hash mismatch")
    manifest=json.loads(args.validation_manifest.read_text(encoding="utf-8"))
    if manifest.get("split") not in {"official_val","validation","val"}: raise RuntimeError("only frozen official validation is accepted")
    for key in ("images_subdir","labels_subdir"):
        if "test" in str(manifest.get(key,"")).lower(): raise RuntimeError("HRSID test path prohibited by manifest")
    image_root=(args.dataset_root/manifest["images_subdir"]).resolve(); label_root=(args.dataset_root/manifest["labels_subdir"]).resolve()
    rows=manifest["images"]
    ground_truth={"images":[],"annotations":[],"categories":[{"id":1,"name":"ship"}]}; annotation_id=1
    for index,row in enumerate(rows):
        image_path=image_root/row["file_name"]; label_path=label_root/row.get("label_file",Path(row["file_name"]).with_suffix(".txt").name)
        if sha256(image_path)!=row["image_sha256"] or sha256(label_path)!=row["label_sha256"]: raise RuntimeError(f"validation artifact hash mismatch: {row['file_name']}")
        with Image.open(image_path) as probe: width,height=probe.size
        image_id=index+1; ground_truth["images"].append({"id":image_id,"file_name":row["file_name"],"width":width,"height":height})
        for line in label_path.read_text(encoding="utf-8").splitlines():
            category,cx,cy,bw,bh=map(float,line.split())
            if int(category)!=0: raise RuntimeError("HRSID validation label is not zero-indexed single-class")
            x=(cx-bw/2)*width; y=(cy-bh/2)*height; w=bw*width; h=bh*height
            ground_truth["annotations"].append({"id":annotation_id,"image_id":image_id,"category_id":1,"bbox":[x,y,w,h],"area":w*h,"iscrowd":0}); annotation_id+=1
    coco=COCO(); coco.dataset=ground_truth; coco.createIndex(); image_ids=sorted(coco.imgs)
    if len(image_ids) != 841: raise RuntimeError(f"official validation must contain 841 images, found {len(image_ids)}")
    predictions, failures = [], []
    with torch.inference_mode():
        for image_id in image_ids:
            info = coco.loadImgs(image_id)[0]
            try:
                image = np.asarray(Image.open(image_root / info["file_name"]).convert("RGB"), dtype=np.float32) / 255.0
                tensor = torch.from_numpy(image).permute(2, 0, 1).cuda()
                output = model([tensor])[0]
                boxes, scores = output["boxes"].cpu(), output["scores"].cpu()
                for box, score in zip(boxes, scores):
                    x0, y0, x1, y1 = box.tolist()
                    predictions.append({"image_id":image_id,"category_id":1,"bbox":[x0,y0,x1-x0,y1-y0],"score":float(score)})
            except Exception as error:
                failures.append({"image_id":image_id,"error":repr(error)})
    if failures: raise RuntimeError(f"validation inference failures: {failures[:3]}")
    args.output_root.mkdir(parents=True)
    atomic_json(args.output_root / "predictions.json", predictions)
    if predictions:
        detection = coco.loadRes(str(args.output_root / "predictions.json"))
    else:
        detection = COCO()
        detection.dataset = {"images": ground_truth["images"], "annotations": [], "categories": ground_truth["categories"]}
        detection.createIndex()
    evaluator = COCOeval(coco, detection, "bbox"); evaluator.params.imgIds = image_ids; evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    result = {"status":"FCOS_VALIDATION_PASS","family":family,"tier":tier,"seed":training["seed"],"parameters":expected,"images":len(image_ids),"failed_images":0,"predictions":len(predictions),"AP50_95":float(evaluator.stats[0]),"AP50":float(evaluator.stats[1]),"AP75":float(evaluator.stats[2]),"checkpoint_sha256":training["checkpoint_sha256"],"predictions_sha256":sha256(args.output_root / "predictions.json"),"protocol_sha256":sha256(args.protocol),"identity_freeze_sha256":sha256(args.identity_freeze),"selection_or_tuning_used":False,"HRSID_TEST_METRIC_ACCESS":"CLOSED_AFTER_FINAL_EVALUATION","HRSID_TEST_FILES_PRESENT_ON_CLUSTER":False,"HARDWARE_AWARE_CAPP_STARTED":False}
    atomic_json(args.output_root / "result.json", result)


if __name__ == "__main__": main()
