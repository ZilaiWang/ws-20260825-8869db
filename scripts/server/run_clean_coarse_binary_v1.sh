#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${MANIFEST:?set clean MANIFEST}"
: "${PSEUDO_ROOT:?set PSEUDO_ROOT}"
: "${P03_DIR:?set P03_DIR}"
: "${IMAGENET_WEIGHT:?set IMAGENET_WEIGHT}"
: "${GT:?set GT}"
: "${BASE_PRED:?set BASE_PRED}"
: "${BASELINE_FRONTIER:?set BASELINE_FRONTIER}"
: "${OUT:?set OUT}"

mkdir -p "${OUT}/checkpoints"
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
printf '%s\n' training > "${OUT}/status.txt"
for fold in 0 1 2; do
  pids=()
  for coarse in ship aircraft vehicle; do
    "${PYTHON_BIN}" scripts/train_pseudo_coarse_binary_verifier.py \
      --manifest "${MANIFEST}" --pseudo-root "${PSEUDO_ROOT}" \
      --imagenet-weight "${IMAGENET_WEIGHT}" \
      --p03-checkpoint "${P03_DIR}/p03_fold${fold}.pt" \
      --output "${OUT}/checkpoints/coarse_${coarse}_fold${fold}.pt" \
      --held-out-fold "${fold}" --coarse "${coarse}" \
      --epochs 3 --batch-size 64 --batches-per-epoch 40 \
      --resolution 224 --seed 20260830 --device cuda:0 \
      > "${OUT}/train_${coarse}_fold${fold}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done

printf '%s\n' inference > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/rerank_cv3_pseudo_with_coarse_binary_verifier.py \
  --gt "${GT}" --pred "${BASE_PRED}" --pseudo-root "${PSEUDO_ROOT}" \
  --checkpoint-dir "${OUT}/checkpoints" --imagenet-weight "${IMAGENET_WEIGHT}" \
  --output "${OUT}/coarse_predictions.json" --summary "${OUT}/inference_summary.json" \
  --alpha 0.50 --context-ratio 1.0 --resolution 224 --batch-size 192 \
  --device cuda:0 > "${OUT}/inference.log" 2>&1

printf '%s\n' pixel_oer > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/train_pseudo_pixel_oer.py \
  --ground-truth "${GT}" --crop-predictions "${OUT}/coarse_predictions.json" \
  --output-dir "${OUT}/pixel_oer" --nms-iou 0.60 > "${OUT}/pixel_oer.log" 2>&1
"${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "${GT}" --pred "${OUT}/pixel_oer/identity_predictions.json" \
  --output "${OUT}/identity_frontier.json" > "${OUT}/identity_frontier.log" 2>&1
"${PYTHON_BIN}" scripts/compare_metric_risk_stages.py \
  --baseline "${BASELINE_FRONTIER}" \
  --stage "clean_coarse=${OUT}/identity_frontier.json" \
  --output "${OUT}/decision.json" > "${OUT}/decision.log" 2>&1
(
  cd "${OUT}"
  sha256sum checkpoints/*.pt coarse_predictions.json inference_summary.json \
    pixel_oer/identity_predictions.json identity_frontier.json decision.json > SHA256SUMS
)
printf '%s\n' complete > "${OUT}/status.txt"
