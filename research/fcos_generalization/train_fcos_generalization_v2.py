#!/usr/bin/env python3
"""Train one state from the frozen 15-state FCOS generalization matrix.

FCOS_GENERALIZATION_PROTOCOL_V2: byte-identical to train_fcos_generalization.py
EXCEPT that AMP is disabled -- torch.amp.autocast and torch.amp.GradScaler are
removed and the forward/backward/optimizer step run in fp32. This is the only
scientific training change from V1, made to remove the reproduced fp16
dynamic-range overflow in the classification tower
(FCOS_WIDTH_AMP_NUMERICAL_INSTABILITY). Every other hyperparameter -- seeds,
construction, COCO->parent->child transfer, dataset, augmentation, hflip
probability, batch size, gradient accumulation, optimizer, LR, momentum,
weight decay, scheduler, epochs, checkpoint rule, evaluator -- is unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from hrsid_labels import yolo_line_to_coco_bbox
from fcos_models import (CAPP_PARAMETERS, PARENT_PARAMETERS, WIDTH_PARAMETERS,
    WIDTH_PER_GROUP, architecture_hash, capp_model, count_parameters, parent_model, sha256, width_model)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_hrsid_fcos_label(line: str, width: int, height: int) -> tuple[int, list[float]]:
    """Return the frozen HRSID detection target as an XYXY pixel box.

    The canonical project evaluator accepts either a five-value YOLO box or a
    YOLO segmentation polygon and converts polygons to their axis-aligned
    min/max envelope. Reuse that conversion verbatim, then change only the box
    representation from COCO XYWH to the XYXY representation required by FCOS.
    """
    category, (x, y, box_width, box_height) = yolo_line_to_coco_bbox(line, width, height)
    x2, y2 = x + box_width, y + box_height

    # Eight frozen training polygons lie exactly on an image boundary and have
    # zero extent in one dimension after the canonical min/max conversion.
    # Torchvision FCOS rejects non-positive boxes. Keep each object and its
    # canonical boundary geometry, changing only the collapsed endpoint by one
    # image-boundary float32 step toward the interior. Using the resolution at
    # the image extent also survives the frozen horizontal-flip subtraction.
    if not x2 > x:
        if x2 < width:
            x2 = min(float(width), x + float(abs(np.spacing(np.float32(width)))))
        elif x > 0:
            x = float(np.nextafter(np.float32(x), np.float32(0)))
    if not y2 > y:
        if y2 < height:
            y2 = min(float(height), y + float(abs(np.spacing(np.float32(height)))))
        elif y > 0:
            y = float(np.nextafter(np.float32(y), np.float32(0)))
    return category, [x, y, x2, y2]


class HRSIDTrain(Dataset):
    def __init__(self, dataset_root: Path, manifest_path: Path, seed: int):
        if "test" in str(dataset_root).lower() or "test" in str(manifest_path).lower():
            raise RuntimeError("HRSID test paths are prohibited")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("split") not in {"official_train", "train"}:
            raise RuntimeError("only the frozen official training manifest is accepted")
        for key in ("images_subdir", "labels_subdir"):
            if "test" in str(manifest.get(key, "")).lower(): raise RuntimeError("HRSID test path prohibited by manifest")
        self.images = (dataset_root / manifest["images_subdir"]).resolve()
        self.labels = (dataset_root / manifest["labels_subdir"]).resolve()
        self.rows, self.seed = manifest["images"], seed
        for row in self.rows:
            image = self.images / row["file_name"]
            label = self.labels / row.get("label_file", Path(row["file_name"]).with_suffix(".txt").name)
            if sha256(image) != row["image_sha256"] or sha256(label) != row["label_sha256"]:
                raise RuntimeError(f"frozen training artifact hash mismatch: {row['file_name']}")

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_id = index + 1
        image_path = self.images / row["file_name"]
        label_path = self.labels / row.get("label_file", Path(row["file_name"]).with_suffix(".txt").name)
        image = Image.open(image_path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        height, width = tensor.shape[-2:]
        boxes=[]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            category, box = parse_hrsid_fcos_label(line, width, height)
            if category != 0: raise RuntimeError("HRSID training label is not zero-indexed single-class")
            boxes.append(box)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if len(boxes):
            if not torch.all(boxes[:, 2] > boxes[:, 0]) or not torch.all(boxes[:, 3] > boxes[:, 1]):
                raise RuntimeError("HRSID training target contains a non-positive-area box")
            if not torch.all(boxes[:, (0, 2)] >= 0) or not torch.all(boxes[:, (0, 2)] <= width):
                raise RuntimeError("HRSID training target x-coordinate is outside image geometry")
            if not torch.all(boxes[:, (1, 3)] >= 0) or not torch.all(boxes[:, (1, 3)] <= height):
                raise RuntimeError("HRSID training target y-coordinate is outside image geometry")
        labels = torch.zeros((len(boxes),), dtype=torch.int64)  # single FCOS foreground logit is class index 0
        if random.random() < .5:
            tensor = tensor.flip(-1); width = tensor.shape[-1]
            if len(boxes):
                left = width - boxes[:, 2].clone(); right = width - boxes[:, 0].clone()
                boxes[:, 0], boxes[:, 2] = left, right
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return tensor, {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }


def collate(batch): return tuple(zip(*batch))


def construct(family: str, tier: str, protocol: dict):
    if family == "parent": return parent_model(), PARENT_PARAMETERS
    if family == "width": return width_model(WIDTH_PER_GROUP[tier]), WIDTH_PARAMETERS[tier]
    return capp_model(protocol["architecture"]["capp"][tier]["structure"]), CAPP_PARAMETERS[tier]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--family", choices=("parent", "capp", "width"), required=True)
    parser.add_argument("--tier", choices=("P", "B", "A"), required=True)
    parser.add_argument("--seed", choices=(0, 1, 2), type=int, required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if not args.launch: raise SystemExit("explicit --launch required")
    if (args.family == "parent") != (args.tier == "P"):
        raise RuntimeError("parent requires tier P; compressed families require B or A")
    for path in (args.dataset_root, args.train_manifest):
        if "test" in str(path).lower(): raise RuntimeError("HRSID test path prohibited")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FCOS_GENERALIZATION_PROTOCOL_FROZEN": raise RuntimeError("protocol is not frozen")
    if protocol.get("training", {}).get("amp") is not False: raise RuntimeError("V2 requires a protocol with training.amp == false")
    if sha256(args.pretrained) != protocol["initialization"]["sha256"]: raise RuntimeError("pretrained checkpoint hash mismatch")
    output = args.output_root / f"{args.family}_{args.tier.lower()}" / f"seed{args.seed}"
    if output.exists(): raise RuntimeError(f"append-only output exists: {output}")
    output.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Build the same seed-specific parent initialization first for every family.
    # Children inherit every exact-name/exact-shape tensor from that state; only
    # genuinely new or shape-changed compressed tensors retain deterministic init.
    source = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    reference = parent_model()
    reference_state = reference.state_dict()
    pretrained_transferred = {k:v for k,v in source.items() if k in reference_state and tuple(v.shape)==tuple(reference_state[k].shape)}
    reference.load_state_dict(pretrained_transferred, strict=False)
    if args.family == "parent":
        model, expected_parameters = reference, PARENT_PARAMETERS
    else:
        torch.manual_seed(args.seed)
        model, expected_parameters = construct(args.family, args.tier, protocol)
        target_state=model.state_dict(); inherited={k:v for k,v in reference.state_dict().items() if k in target_state and tuple(v.shape)==tuple(target_state[k].shape)}
        model.load_state_dict(inherited, strict=False)
    target_state = model.state_dict()
    transferred = {k:v for k,v in reference.state_dict().items() if k in target_state and tuple(v.shape)==tuple(target_state[k].shape)}
    if count_parameters(model) != expected_parameters: raise RuntimeError("frozen architecture parameter count changed")
    if sha256(args.train_manifest) != protocol["data"]["train_manifest_sha256"]: raise RuntimeError("training manifest hash mismatch")
    dataset = HRSIDTrain(args.dataset_root, args.train_manifest, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate, generator=generator, persistent_workers=True)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available(): raise RuntimeError("FCOS production requires CUDA")
    model.to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 45], gamma=.1)
    started, loss_rows = time.time(), []
    for epoch in range(50):
        optimizer.zero_grad(set_to_none=True)
        for iteration, (images, targets) in enumerate(loader):
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            losses = model(images, targets); loss = sum(losses.values()) / 4
            if not torch.isfinite(loss): raise RuntimeError("non-finite FCOS detector loss")
            loss.backward()
            if (iteration + 1) % 4 == 0 or iteration + 1 == len(loader):
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            loss_rows.append(float(loss.detach().cpu()) * 4)
        scheduler.step()
        torch.save({"state_dict": model.state_dict(), "family": args.family, "tier": args.tier, "seed": args.seed, "architecture_hash": architecture_hash(model)}, output / "last.pt")
        print(json.dumps({"epoch": epoch + 1, "loss": loss_rows[-1], "family": args.family, "tier": args.tier, "seed": args.seed}), flush=True)
    checkpoint = output / "last.pt"
    result = {"status":"FCOS_TRAIN_PASS","family":args.family,"tier":args.tier,"seed":args.seed,"parameters":expected_parameters,"architecture_hash":architecture_hash(model),"checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),"epochs":50,"amp":False,"protocol_variant":"FCOS_GENERALIZATION_PROTOCOL_V2","training_gpu_seconds":time.time()-started,"pretrained_tensor_count":len(pretrained_transferred),"parent_state_transferred_tensor_count":len(transferred),"new_or_shape_changed_tensor_count":len(target_state)-len(transferred),"official_validation_used":False,"protocol_sha256":sha256(args.protocol),"HRSID_TEST_METRIC_ACCESS":"CLOSED_AFTER_FINAL_EVALUATION","HRSID_TEST_FILES_PRESENT_ON_CLUSTER":False,"HARDWARE_AWARE_CAPP_STARTED":False,"slurm_job_id":os.getenv("SLURM_JOB_ID","")}
    atomic_json(output / "result.json", result)


if __name__ == "__main__": main()
