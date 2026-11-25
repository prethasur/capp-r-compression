#!/bin/bash
#SBATCH --job-name=fcos_v2_val15
#SBATCH --partition=gpu
#SBATCH --array=0-14
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
ROOT="${EXPERIMENT_ROOT:?set EXPERIMENT_ROOT}"
CODE="$ROOT/repo/research/fcos_generalization"
PY="${PYTHON:-python}"
FAMILIES=(parent parent parent capp capp capp capp capp capp width width width width width width)
TIERS=(P P P B B B A A A B B B A A A)
SEEDS=(0 1 2 0 1 2 0 1 2 0 1 2 0 1 2)
I=${SLURM_ARRAY_TASK_ID:?}
KEY="${FAMILIES[$I]}_${TIERS[$I],,}"
STATE="$ROOT/artifacts/fcos_generalization/v2/training/$KEY/seed${SEEDS[$I]}"
FREEZE="$ROOT/artifacts/fcos_generalization/v2/freeze/FCOS_GENERALIZATION_PROTOCOL_V2_FREEZE.json"
export PYTHONPATH="$CODE:$ROOT/repo" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$PY" "$CODE/evaluate_fcos_validation.py" --launch \
  --checkpoint "$STATE/last.pt" --training-result "$STATE/result.json" \
  --protocol "$FREEZE" --dataset-root "$ROOT/data" \
  --identity-freeze "$ROOT/artifacts/fcos_generalization/v2/freeze/FCOS_CHECKPOINT_ARCHITECTURE_IDENTITY_FREEZE.json" \
  --validation-manifest "$ROOT/manifests/hrsid_official_val.json" \
  --output-root "$ROOT/artifacts/fcos_generalization/v2/validation/$KEY/seed${SEEDS[$I]}"
