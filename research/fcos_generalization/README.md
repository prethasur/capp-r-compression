# FCOS generalisation (research code)

This folder holds the code, frozen protocol and selected replacement structures used to apply CAPP-R to a second, unrelated detector: torchvision's **FCOS-R50-FPN** (anchor-free, ResNet-50 backbone, FPN, shared GroupNorm towers). It is published so the experiment can be read end to end.

**It is not maintained as runnable software.** The scripts expect a Slurm cluster, the HRSID dataset with train and validation manifests, a COCO-pretrained FCOS checkpoint and an experiment directory layout that is not in this repository. Cluster paths have been replaced with `<experiment_root>` / `$EXPERIMENT_ROOT`. The import of the HRSID label helper was redirected to `hrsid_labels.py`. Apart from those two changes, the files are the code that produced the reported results. Because of the path edits, the SHA-256 values recorded inside the protocol files refer to the original, unedited files.

## Why FCOS

The YOLO pipeline in `capp_r/` relies on an Ultralytics adapter: the Detect head, DFL and NMS are protected, and the native YOLO loss is used. FCOS shares none of that machinery: it has a different backbone, a different head and different post-processing. Re-using the CAPP-R idea here tests whether contract-preserving replacement transfers beyond the YOLO family.

## Adaptation: contract, grammar and selection

| Stage | YOLO (`capp_r/`) | FCOS (this folder) |
|---|---|---|
| **Contract** | Runtime-extracted single-entry/single-exit regions; Detect/DFL/NMS protected | Every dense 3×3 `Conv2d` keeps its input channels, output channels and stride. Protected: the stem and the terminal classification, box-regression and centerness predictors. The `FCOS` boxes/labels/scores output contract is unchanged |
| **Grammar** | Typed `GraphSpec` families (bottleneck → depthwise → projection, grouped, pooled, two-branch, residual), ≤7 primitives | A single interface-preserving primitive from the same family, **`GhostSpatial`** (`fcos_models.py`). A 1×1 primary convolution produces ⌈C/2⌉ channels at the original stride. A depthwise 3×3 "cheap" convolution generates the remaining channels from them, and the two are concatenated back to C |
| **Selection** | Enumeration, preflight gates, fidelity ranks, Pareto shortlist | Parameter-only subset-sum over eligible 3×3 regions, aimed at YOLO11's relative B/A parameter budgets (B = 1,802,835 / 2,590,035 of the parent; A = 1,432,019 / 2,590,035), with a deterministic name tie-break (`qualify_fcos_r50_fpn.py`). Validation accuracy and hardware metrics were **not** used for selection |
| **Matched baseline** | Uniform width scaling | Global ResNet-50 `width_per_group` (B = 42, A = 28), chosen by parameter count only, within 0.5% of the CAPP model (`FCOS_REFINED_WIDTH_QUALIFICATION_PROTOCOL.json`) |

```python
class GhostSpatial(nn.Module):           # fcos_models.py
    def __init__(self, old: nn.Conv2d):
        primary = (old.out_channels + 1) // 2
        cheap = old.out_channels - primary
        self.primary = nn.Conv2d(old.in_channels, primary, 1, old.stride, 0, bias=old.bias is not None)
        self.cheap   = nn.Conv2d(primary, primary, 3, 1, 1, groups=primary, bias=old.bias is not None)
    def forward(self, x):
        p = self.primary(x)
        return torch.cat((p, self.cheap(p)[:, :self.cheap_channels]), 1)
```

The 20 replaced regions for each tier are listed in `protocol/FCOS_GENERALIZATION_PROTOCOL_V2.json` under `architecture.capp.{B,A}.structure`. For both tiers they cover:

- the FPN output convolutions and the P6/P7 extra blocks;
- all eight 3×3 convolutions in the classification and regression towers;
- seven ResNet bottleneck `conv2` layers.

The two tiers choose different bottleneck layers: B includes four layer3 blocks, while A includes all three layer4 blocks.

## Protocol (frozen before training)

| | |
|---|---|
| **Parent** | torchvision `fcos_resnet50_fpn` v0.20.1, 1 foreground class, 640×640 (32,115,270 params) |
| **Initialisation** | COCO-pretrained FCOS → parent; every exact-name/exact-shape tensor is transferred to CAPP and Width children; the rest use torchvision's default initialisation |
| **Training** | 50 epochs, SGD with learning rate 0.01, momentum 0.9 and weight decay 1e-4, MultiStepLR [35, 45] ×0.1, batch 4 × 4 accumulation, horizontal flip only, FP32 (AMP disabled in V2) |
| **Seeds** | 0, 1, 2 for each of the 5 models (15 runs) |
| **Evaluation** | HRSID official validation split (841 images), COCO AP50:95, score threshold 0.001, NMS 0.6, 100 detections; run once after the checkpoint identities were frozen |
| **Hardware** | Jetson Orin Nano, TensorRT 8.6.2 FP32 with TF32 disabled, static 640×640 input, 15 W mode with pinned clocks |

Protocol V2 differs from V1 in one setting only: mixed precision was removed. V1 runs showed a reproducible FP16 overflow in the classification tower for several Width-baseline seeds.

## Files

| File | Role |
|---|---|
| `fcos_models.py` | Parent, `GhostSpatial` CAPP and Width model builders, parameter targets, architecture hash |
| `qualify_fcos_r50_fpn.py` | Region eligibility, subset-sum CAPP selection, Width matching, smoke test (forward → save → rebuild → reload → forward) |
| `freeze_fcos_protocol.py` | Writes the append-only frozen protocol |
| `train_fcos_generalization_v2.py` | Trains one (family, tier, seed) state |
| `hrsid_labels.py` | Converts YOLO-format HRSID labels to COCO boxes |
| `evaluate_fcos_validation.py` | COCO evaluation on the official validation split, gated on the frozen checkpoint identities |
| `finalize_fcos_generalization.py` | Completeness check, identity freeze and aggregation |
| `protocol/` | Qualification protocols, Width specs, canonical torchvision source audit, frozen V2 protocol with the selected structures |
| `slurm/` | The 15-run training and validation job arrays |

## Results

See the main [README](../../README.md#fcos-r50-fpn).
