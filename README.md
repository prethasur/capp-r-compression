# CAPP-R

Contract-preserving structural replacement for compressing object detectors: find submodules whose tensor interface can be held fixed, synthesise smaller typed replacement subgraphs, and verify that the modified detector still behaves like a detector before any retraining is spent on it.

```bash
pip install -e ".[yolo,test]"          # PyTorch first, for your platform
python -m pytest                        # 29 offline CPU tests, ~15 s
python examples/synthetic_pipeline.py   # full pipeline on a random-weight YOLO11n, ~10 s
```

## Results at a glance

| | Parameters | AP50:95 vs parent | AP50:95 vs matched width scaling | On Jetson Orin Nano vs parent |
|---|---:|---:|---:|---|
| **YOLO11n CAPP-B** | −30.4% | −0.0004 | **+0.034** | engine −22%, latency −6%, energy −6% |
| **YOLO11n CAPP-A** | −44.7% | −0.0014 | **+0.040** | engine −36%, latency −8%, energy −9% |
| **FCOS-R50 CAPP-B** | −30.4% | −0.018 | **+0.052** | engine −55%, latency −40%, throughput +63%, energy −41% |
| **FCOS-R50 CAPP-A** | −44.7% | −0.019 | **+0.046** | engine −61%, latency −40%, throughput +62%, energy −41% |

These figures are for SAR ship detection on HRSID. The YOLO11n accuracy is a 3-seed mean; the FCOS rows are the seed-0 models deployed on the Jetson. Latency is end-to-end p95 and energy is dynamic joules per image. The full tables, protocol and caveats are under [Results](#results).

## Overview

Pruning and width scaling shrink a network by removing channels or layers uniformly. CAPP-R takes a different route. It treats a trained detector as a graph of regions with fixed boundaries, and replaces the *inside* of selected regions with cheaper operator graphs. The external tensor contract (shape, dtype, spatial stride) and the task head are preserved. Because every replacement must satisfy the contract it replaces, the rest of the detector, its loss, decoding and export path keep working unchanged.

This repository contains:

- **`capp_r/`:** the replacement machinery and its verification gates for Ultralytics YOLOv8n and YOLO11n models.
- **`research/fcos_generalization/`:** the adaptation of the same idea to torchvision FCOS-R50-FPN, kept as research code.

It also reports results from the original experiments. Datasets, checkpoints and the full training and deployment pipeline are not included.

## Method

1. **Contract extraction** (`capp_r/contracts/contract_extractor.py`). A single forward pass with hooks records, for every submodule:
   - input/output tensors and spatial stride;
   - parameter and MAC counts;
   - producers and consumers.

   A region is *eligible* if it is single-entry/single-exit, executed, holds at least 1% of model parameters, and is not protected. Protection comes from an architecture adapter (`capp_r/adapters/`): for YOLO, the `Detect` head, DFL/`reg_max` and NMS semantics. No module indices are hard-coded.
2. **Typed replacement grammar** (`capp_r/grammar/`). Replacements are `GraphSpec` trees over a small type system (`spatial` and `token` tensors):
   - **Primitives:** pointwise, depthwise and grouped convolution, pooling, normalisation, activation, linear and low-rank linear layers.
   - **Combinators:** sequential, residual, parallel-sum, parallel-concat-projection, repeat.

   Every constructor type-checks its boundary and computes analytical parameter and MAC estimates. Graphs are limited to seven transform primitives. `spatial_grammar.py` and `ffn_grammar.py` enumerate bounded families per boundary, for example bottleneck→depthwise 3×3/5×5→projection, grouped bottlenecks, pooled and two-branch concat-projection forms, and residual forms when input and output types match. Widths are 25–75% of the boundary width, aligned to multiples of 8.
3. **Candidate generation** (`capp_r/generator/enumerate_micrographs.py`).
   - Enumerate grammar graphs for every eligible region.
   - Deduplicate by a canonical, commutativity-aware SHA-256 graph hash (`canonicalize_graph.py`, `graph_hash.py`).
   - Keep (region, graph) pairs that reduce *whole-model* parameters by at least 10%.
4. **Realisation** (`capp_r/generator/realize_micrograph.py`). A serialised graph becomes an `nn.Module`, failing closed on unknown operators, and is spliced into the parent with `replace_module`, preserving the routing attributes Ultralytics uses.
5. **Preflight gates** (`capp_r/contracts/candidate_preflight.py`). Each candidate is accepted only if:
   - its realised parameter count equals the analytical estimate;
   - boundary shape and stride are unchanged;
   - the full detector's output signature is unchanged and finite;
   - the protected head semantics are unchanged;
   - the native YOLO training loss and gradients are finite, with the parent frozen (BatchNorm in eval mode) and only the replacement trainable;
   - ONNX export, the ONNX checker and ONNX Runtime outputs match PyTorch within tolerance.

   Initialisation is deterministic, seeded by the graph hash.
6. **Fidelity measures for local fitting** (`capp_r/local_fit/local_fit_adapters.py`). These compare a candidate with its parent at the replacement boundary (MSE, normalised MSE, cosine distance) and at the detector output:
   - per-scale classification divergence;
   - DFL distribution KL;
   - raw box divergence;
   - decoded box and score divergence after Hungarian matching of top-k predictions.

   The training loop that fits replacements to these targets is **not** included (see Limitations).
7. **Shortlist selection** (`capp_r/selection/yolo_shortlist.py`, rules in `configs/yolo_shortlist_selector_v1.json`):
   - hard gates (fit status, contract, ONNX parity, ≥10% reduction, no validation or test split use);
   - within-architecture fractional ranks of the fidelity metrics;
   - Pareto filtering on worst fidelity rank, ONNX size, graph depth and branch/concat counts versus parameter reduction;
   - region×topology diversity limits;
   - deterministic tie-breaking ending in the graph hash.

What is CAPP-R's own work: the contract model, grammar, generator, gates, fidelity measures and selector. Ultralytics provides the detectors, the training loss and ONNX export; PyTorch, ONNX and ONNX Runtime are used as is.

## Implemented capabilities

| Capability | Code | Tested by |
|---|---|---|
| Typed micrograph grammar with analytical cost estimates | `capp_r/grammar/` | `tests/test_grammar.py`, `test_realize_micrograph.py` |
| Canonical graph hashing and deduplication | `capp_r/generator/{canonicalize_graph,graph_hash,deduplicate_graphs}.py` | `test_grammar.py` |
| Graph → `nn.Module` realisation; module splicing | `capp_r/generator/realize_micrograph.py` | `test_realize_micrograph.py` (every grammar output for 5 boundary types: exact shapes and parameter counts) |
| Runtime contract extraction and eligibility | `capp_r/contracts/contract_extractor.py`, `capp_r/adapters/` | `test_protected_semantics.py`, `test_end_to_end_synthetic.py` |
| Candidate generation (CLI) | `capp_r/generator/enumerate_micrographs.py` | `test_end_to_end_synthetic.py` |
| Preflight gates incl. native loss, gradients and ONNX/ORT parity (CLI) | `capp_r/contracts/candidate_preflight.py` | `test_candidate_preflight_yolo.py`, `test_end_to_end_synthetic.py` |
| Boundary and detector-output fidelity measures | `capp_r/local_fit/local_fit_adapters.py` | `test_local_fit_adapters.py` |
| Pareto shortlist selector | `capp_r/selection/yolo_shortlist.py` | `test_yolo_shortlist_selector.py` |

## Installation

Python ≥ 3.10. Install PyTorch for your platform, then:

```bash
pip install -e ".[yolo,test]"
```

The core (grammar, generator, realisation) needs only `torch`, `numpy` and `scipy`. The `yolo` extra adds `ultralytics` (8.3.x), `onnx` and `onnxruntime` for contract extraction, preflight and the end-to-end example.

**Tested environment:** Windows 11, Python 3.12.3, torch 2.5.1, ultralytics 8.3.145, onnx 1.17.0, onnxruntime 1.25.0, scipy 1.13.1, CPU.

## Quick start

```bash
python examples/synthetic_pipeline.py --out runs/synthetic
```

This builds a randomly initialised YOLO11n from Ultralytics' bundled `yolo11n.yaml` (nothing is downloaded) and runs the three stages:

```bash
python -m capp_r.contracts.contract_extractor --family yolo11n --weights yolo11n.yaml --output runs/synthetic/contracts.json
python -m capp_r.generator.enumerate_micrographs --contracts runs/synthetic/contracts.json --output runs/synthetic/candidates.json
python -m capp_r.contracts.candidate_preflight --architecture yolo11n --candidates runs/synthetic/candidates.json \
  --weights yolo11n.yaml --table runs/synthetic/preflight.csv --failure-summary runs/synthetic/preflight_failures.csv \
  --onnx-dir runs/synthetic/onnx --construction-max 5 --onnx-max 1
```

On the tested environment, the YOLO11n run gives:

- 90 eligible regions, 446 unique replacement graphs, and 26 candidates with at least 10% model-level parameter reduction;
- 5 of 5 candidates passing construction, forward/backward and contract gates;
- 1 of 1 passing ONNX Runtime parity.

These counts and the candidate graph hashes depend only on the architecture and the grammar, not on the random weights; two independent runs were identical. With random weights the example demonstrates the mechanics, not accuracy.

To apply CAPP-R to a trained model, pass its checkpoint as `--weights`.

## Verification

| Level | Status |
|---|---|
| CPU unit and functional tests (`python -m pytest`, 29 tests) | Passing on the tested environment |
| End-to-end synthetic pipeline on random-weight YOLO11n | Passing (in the test suite). Two runs gave identical counts and candidate hashes |
| End-to-end synthetic pipeline on random-weight YOLOv8n | Passing when run manually (`--model yolov8n.yaml --family yolov8n`): 92 eligible regions, 458 unique graphs, 27 retained candidates |
| GPU execution | Supported by the code (CUDA is used when available); not exercised by the tests |
| Trained checkpoints, real datasets | Not part of this repository |
| Training, accuracy evaluation, TensorRT deployment | Run in the original experiments (see [Results](#results)); not reproducible from this repository |
| FCOS research code (`research/fcos_generalization/`) | Published as executed; not tested here and not runnable without the original cluster setup |

## Results

All results come from the original experiments. The task is single-class SAR ship detection on **HRSID**, scored as COCO AP50:95 on the official 841-image validation split. The test split was not used for any number below.

**Hardware protocol:**
- Jetson Orin Nano, TensorRT 8.6.2;
- FP32 with TF32 disabled;
- static 640×640 input;
- 15 W mode with pinned clocks;
- every engine passed a TensorRT-versus-PyTorch accuracy check.

**Reported metrics:**
- **Engine:** serialised TensorRT engine size.
- **Latency:** end-to-end p95 per image.
- **Energy:** dynamic energy per image above the idle baseline.

**Model families:**
- **Width:** uniform width scaling, matched to the CAPP model's parameter count.
- **GReg and Moonshine:** compression baselines at matched size.

### YOLO11n

| Model | Params | Reduction | AP50:95 (3 seeds, mean ± sd) | Engine MiB | Latency p95 ms | Dynamic J/img |
|---|---:|---:|---|---:|---:|---:|
| Parent | 2,590,035 | — | 0.6435 ± 0.0018 | 16.20 | 22.82 | 0.1234 |
| **CAPP-B** | 1,802,835 | 30.4% | **0.6431 ± 0.0038** | 12.58 | 21.43 | 0.1164 |
| **CAPP-A** | 1,432,019 | 44.7% | **0.6420 ± 0.0050** | 10.40 | 21.01 | 0.1122 |
| Width-B | 1,813,479 | 30.0% | 0.6094 ± 0.0028 | 13.24 | 22.18 | 0.1100 |
| Width-A | 1,366,069 | 47.3% | 0.6025 ± 0.0020 | 10.58 | 21.11 | 0.0978 |
| GReg-B | 1,803,811 | 30.4% | 0.5964 ± 0.0022 | 12.78 | 21.16 | 0.0933 |
| Moonshine-B | 1,815,627 | 29.9% | 0.5771 ± 0.0054 | 12.45 | 24.20 | 0.1231 |
| Moonshine-A | 1,553,484 | 40.0% | 0.5398 ± 0.0067 | 11.21 | 23.33 | 0.1097 |

**Pruning baselines** (B tier, 1,815,267 params, accuracy only):
- GroupFisher-B 0.6278 ± 0.0042;
- Random-B 0.6175 ± 0.0025;
- FPGM-B 0.6156 ± 0.0054.

**Result:** CAPP-B and CAPP-A keep the parent's accuracy within one standard deviation while removing 30% and 45% of the parameters. Every matched-size baseline loses at least 0.015 AP; width scaling loses 0.034–0.040.

**Jetson, CAPP-B vs parent:**
- engine −22.3%;
- compute p95 −9.9%;
- end-to-end p95 −6.1%;
- throughput +4.3%;
- dynamic energy −5.7%.

**Jetson, CAPP-A vs parent:**
- engine −35.8%;
- compute p95 −14.4%;
- end-to-end p95 −7.9%;
- throughput +6.2%;
- dynamic energy −9.1%.

### FCOS-R50-FPN

The YOLO11 budgets were carried over to FCOS. This used the interface-preserving `GhostSpatial` replacement, selected on parameter count alone; see [`research/fcos_generalization/`](research/fcos_generalization/) for the adaptation and the code.

**Deployed models (seed 0), Jetson:**

| Model | Params | Reduction | AP50:95 | Engine MiB | Latency p95 ms | Throughput img/s | Dynamic J/img |
|---|---:|---:|---:|---:|---:|---:|---:|
| Parent | 32,115,270 | — | 0.6055 | 208.60 | 161.95 | 5.94 | 1.477 |
| **CAPP-B** | 22,352,998 | 30.4% | **0.5872** | **93.17** | **96.75** | **9.70** | **0.866** |
| **CAPP-A** | 17,760,166 | 44.7% | **0.5861** | **82.23** | **97.52** | **9.61** | **0.874** |
| Width-B | 22,449,482 | 30.1% | 0.5352 | 173.35 | 143.82 | 6.72 | 1.332 |
| Width-A | 17,691,078 | 44.9% | 0.5402 | 145.91 | 125.51 | 7.63 | 1.181 |

**CAPP-B vs Width-B, at the same parameter count:**
- AP +0.052;
- engine −46%;
- compute p95 −39%;
- end-to-end p95 −33%;
- throughput +44%;
- dynamic energy −35%.

**CAPP-A vs Width-A:**
- AP +0.046;
- engine −44%;
- end-to-end p95 −22%;
- dynamic energy −26%.

**Accuracy across 3 training seeds** (official validation):

| Model | Seed 0 | Seed 1 | Seed 2 | Mean ± sd |
|---|---:|---:|---:|---|
| Parent | 0.6055 | 0.5634 | 0.5625 | 0.5771 ± 0.0246 |
| **CAPP-B** | 0.5872 | 0.5860 | 0.5736 | **0.5823 ± 0.0075** |
| CAPP-A | 0.5861 | 0.5853 | 0.4312 | 0.5342 ± 0.0892 |
| Width-B | 0.5352 | 0.0556 | 0.5074 | 0.3661 ± 0.2692 |
| Width-A | 0.5402 | diverged | 0.5257 | 0.5329 ± 0.0103 (n = 2) |

FCOS-R50 CAPP-B was the most stable model: its 3-seed mean was slightly above the parent's, with a third of the parent's spread. Some runs under this fixed recipe were unstable:
- CAPP-A seed 2 and Width-B seed 1 trained to low accuracy.
- Width-A seed 1 diverged in epoch 1.
- A post-hoc rerun of that seed with a learning-rate warm-up reached 0.6089. It is not counted above because it changes the frozen recipe.

### What the results do not show

- **Width scaling and GReg-B are cheaper on the Jetson.** On YOLO11n, at matched size, they have lower latency and energy. GReg-B's compute p95 is 21% lower than CAPP-B's and its dynamic energy is 25% lower, but its AP is 0.047 lower. CAPP's advantage on YOLO11n is accuracy at a given size, not speed.
- **No advantage at mild compression.** At an 11% reduction (the E tier), CAPP-E reaches 0.6467 and the pruning baselines match or slightly exceed it (GroupFisher-E 0.6497, Random-E 0.6488, FPGM-E 0.6486).
- **FCOS hardware rows are single models.** They are the seed-0 models, so they show the deployment effect rather than a seed-averaged estimate. The FCOS recipe was not tuned for the compressed models.

## Repository structure

```
capp_r/
  grammar/      type system, primitives, combinators, spatial and token (FFN) grammars
  generator/    canonical hashing, deduplication, candidate generation CLI, graph → nn.Module
  contracts/    contract extraction CLI, boundary validation, protected semantics, preflight CLI
  adapters/     architecture adapters (YOLOv8n, YOLO11n)
  local_fit/    boundary and detector-output fidelity measures
  selection/    Pareto shortlist selector
configs/        shortlist selection rules
examples/       synthetic end-to-end pipeline
tests/          offline tests
research/
  fcos_generalization/   FCOS-R50-FPN adaptation: GhostSpatial replacement, selection,
                         frozen protocol and selected structures, training/evaluation scripts
```

## Limitations

- **Architectures:** the `capp_r` package supports Ultralytics YOLOv8n and YOLO11n only. The FCOS adaptation is separate research code with a fixed primitive and parameter-only selection, not a `capp_r` adapter. Other families need a new adapter, and generally a different contract.
- **Region types:** only single-entry/single-exit regions are handled. Multi-input regions, and composing replacements across several regions, are not implemented here.
- **Not included:** the YOLO local-fitting training loop, full-network recovery training, YOLO accuracy evaluation and the Jetson benchmarking harness are not in this repository. Selecting replacements for a real model needs those steps, plus a dataset and GPU training.
- **Reported, not reproducible here:** the results above come from the original experiments. Reproducing them needs the dataset, training pipeline and hardware set-up, none of which are included.
- **Cost estimates are analytical:** the grammar's MAC estimates are analytical counts, not measured latency. A smaller parameter count does not guarantee faster inference on a given device.
- **Fixed thresholds:** the preflight probe resolution (640×640), the ≥10% reduction threshold, the candidate budget and the ONNX tolerances are fixed protocol choices in the CLIs.
- **Raw-candidate count:** the candidate generator pads the raw candidate list to a minimum budget before deduplication, so `raw_candidates_generated` can count repeats. `unique_graphs` and `retained_candidates` are the informative counts.

## Acknowledgements and licence


- Uses [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0), [PyTorch](https://pytorch.org/), [ONNX](https://onnx.ai/), [ONNX Runtime](https://onnxruntime.ai/) and [SciPy](https://scipy.org/).

