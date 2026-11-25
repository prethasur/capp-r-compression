#!/bin/bash
#SBATCH --job-name=fcos_v2_train15
#SBATCH --partition=gpu
#SBATCH --array=0-14
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
ROOT="${EXPERIMENT_ROOT:?set EXPERIMENT_ROOT}"
CODE="$ROOT/repo/research/fcos_generalization"
PY="${PYTHON:-python}"
FAMILIES=(parent parent parent capp capp capp capp capp capp width width width width width width)
TIERS=(P P P B B B A A A B B B A A A)
SEEDS=(0 1 2 0 1 2 0 1 2 0 1 2 0 1 2)
INDEX=${SLURM_ARRAY_TASK_ID:?}
FREEZE="$ROOT/artifacts/fcos_generalization/v2/freeze/FCOS_GENERALIZATION_PROTOCOL_V2_FREEZE.json"
test -f "$FREEZE"
export PYTHONPATH="$CODE:$ROOT/repo" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$PY" "$CODE/train_fcos_generalization_v2.py" --launch \
  --family "${FAMILIES[$INDEX]}" --tier "${TIERS[$INDEX]}" --seed "${SEEDS[$INDEX]}" \
  --protocol "$FREEZE" \
  --pretrained "$ROOT/artifacts/fcos_generalization/pretrained/fcos_resnet50_fpn_coco-99b0c9b7.pth" \
  --dataset-root "$ROOT/data" --train-manifest "$ROOT/manifests/hrsid_official_train.json" \
  --output-root "$ROOT/artifacts/fcos_generalization/v2/training"
