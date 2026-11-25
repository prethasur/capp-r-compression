# CAPP-R

Contract-preserving structural replacement for compressing object detectors: find submodules whose tensor interface can be held fixed, synthesise smaller typed replacement subgraphs, and verify that the modified detector still behaves like a detector before any retraining is spent on it.

```bash
pip install -e ".[yolo,test]"          # PyTorch first, for your platform
python -m pytest                        # 29 offline CPU tests, ~15 s
python examples/synthetic_pipeline.py   # full pipeline on a random-weight YOLO11n, ~10 s
```

## Overview

Pruning and width scaling shrink a network by removing channels or layers uniformly. CAPP-R takes a different route. It treats a trained detector as a graph of regions with fixed boundaries, and replaces the *inside* of selected regions with cheaper operator graphs. The external tensor contract (shape, dtype, spatial stride) and the task head are preserved. Because every replacement must satisfy the contract it replaces, the rest of the detector, its loss, decoding and export path keep working unchanged.

This repository contains the replacement machinery and its verification gates for Ultralytics YOLOv8n and YOLO11n models. It does not contain training recipes, datasets, checkpoints or experiment results.

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
| Trained checkpoints, real datasets, accuracy | Not part of this repository |
| Embedded or TensorRT deployment | Not part of this repository |

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
```

## Limitations

- **Architectures:** Ultralytics YOLOv8n and YOLO11n only. Other families need a new adapter, and generally a different contract.
- **Region types:** only single-entry/single-exit regions are handled. Multi-input regions, and composing replacements across several regions, are not implemented here.
- **Not included:** the local-fitting training loop, full-network recovery training, accuracy evaluation and hardware benchmarking are not in this repository. Selecting replacements for a real model needs those steps, plus a dataset and GPU training.
- **No benchmark results:** this repository makes no accuracy, latency, energy or size claims about trained compressed models.
- **Cost estimates are analytical:** the grammar's MAC estimates are analytical counts, not measured latency. A smaller parameter count does not guarantee faster inference on a given device.
- **Fixed thresholds:** the preflight probe resolution (640×640), the ≥10% reduction threshold, the candidate budget and the ONNX tolerances are fixed protocol choices in the CLIs.
- **Raw-candidate count:** the candidate generator pads the raw candidate list to a minimum budget before deduplication, so `raw_candidates_generated` can count repeats. `unique_graphs` and `retained_candidates` are the informative counts.

## Acknowledgements and licence


- Uses [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0), [PyTorch](https://pytorch.org/), [ONNX](https://onnx.ai/), [ONNX Runtime](https://onnxruntime.ai/) and [SciPy](https://scipy.org/).

