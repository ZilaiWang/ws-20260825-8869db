#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${MANIFEST:?set MANIFEST}"
: "${PSEUDO_ROOT:?set PSEUDO_ROOT}"
: "${P03_DIR:?set P03_DIR}"
: "${IMAGENET_WEIGHT:?set IMAGENET_WEIGHT}"
: "${OUT:?set OUT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUT}/checkpoints"
printf 'training\n' >"${OUT}/status.txt"
cd "${REPO}"
for fold in 0 1 2; do
  pids=()
  for coarse in ship aircraft vehicle; do
    "${PYTHON_BIN}" scripts/train_pseudo_coarse_binary_verifier.py \
      --manifest "${MANIFEST}" \
      --pseudo-root "${PSEUDO_ROOT}" \
      --imagenet-weight "${IMAGENET_WEIGHT}" \
      --p03-checkpoint "${P03_DIR}/p03_fold${fold}.pt" \
      --output "${OUT}/checkpoints/coarse_${coarse}_fold${fold}.pt" \
      --held-out-fold "${fold}" \
      --coarse "${coarse}" \
      --epochs 3 \
      --batch-size 64 \
      --batches-per-epoch 40 \
      --resolution 224 \
      --seed 20260829 \
      --device cuda:0 \
      >"${OUT}/train_${coarse}_fold${fold}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done
(
  cd "${OUT}"
  sha256sum checkpoints/* >SHA256SUMS
)
printf 'complete\n' >"${OUT}/status.txt"
