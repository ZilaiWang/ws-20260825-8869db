#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${GT:?set GT}"
: "${PRED:?set PRED}"
: "${PSEUDO_ROOT:?set PSEUDO_ROOT}"
: "${P03_DIR:?set P03_DIR}"
: "${IMAGENET_WEIGHT:?set IMAGENET_WEIGHT}"
: "${OUT:?set OUT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BACKGROUND_SAMPLING="${BACKGROUND_SAMPLING:-natural}"

mkdir -p "${OUT}/checkpoints"
printf '%s\n' build_manifest > "${OUT}/status.txt"
cd "${REPO}"

"${PYTHON_BIN}" scripts/build_cv3_pseudo_foreground_manifest.py \
  --gt "${GT}" \
  --pred "${PRED}" \
  --pseudo-root "${PSEUDO_ROOT}" \
  --output "${OUT}/foreground_manifest.csv" \
  --summary "${OUT}/manifest_summary.json" \
  --negative-iou 0.05 \
  --context-ratio 1.0 \
  > "${OUT}/manifest.log" 2>&1

printf '%s\n' training > "${OUT}/status.txt"
for fold in 0 1 2; do
  "${PYTHON_BIN}" scripts/train_pseudo_open_set_verifier.py \
    --manifest "${OUT}/foreground_manifest.csv" \
    --pseudo-root "${PSEUDO_ROOT}" \
    --imagenet-weight "${IMAGENET_WEIGHT}" \
    --p03-checkpoint "${P03_DIR}/p03_fold${fold}.pt" \
    --output "${OUT}/checkpoints/open_set_fold${fold}.pt" \
    --held-out-fold "${fold}" \
    --regime last_stage \
    --epochs 3 \
    --batch-size 64 \
    --batches-per-epoch 40 \
    --background-sampling "${BACKGROUND_SAMPLING}" \
    --resolution 224 \
    --seed 20260829 \
    --device cuda:0 \
    > "${OUT}/train_fold${fold}.log" 2>&1
done

(
  cd "${OUT}"
  sha256sum foreground_manifest.csv manifest_summary.json checkpoints/* > SHA256SUMS
)
printf '%s\n' complete > "${OUT}/status.txt"
