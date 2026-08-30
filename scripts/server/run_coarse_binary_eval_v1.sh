#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${GT:?set GT}"
: "${BASE_PRED:?set BASE_PRED}"
: "${PSEUDO_ROOT:?set PSEUDO_ROOT}"
: "${CHECKPOINT_DIR:?set CHECKPOINT_DIR}"
: "${IMAGENET_WEIGHT:?set IMAGENET_WEIGHT}"
: "${BASELINE_FRONTIER:?set BASELINE_FRONTIER}"
: "${OUT:?set OUT}"

mkdir -p "${OUT}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

on_error() {
  printf '%s\n' failed >"${OUT}/status.txt"
}
trap on_error ERR

printf '%s\n' inference >"${OUT}/status.txt"
"${PYTHON_BIN}" scripts/rerank_cv3_pseudo_with_coarse_binary_verifier.py \
  --gt "${GT}" --pred "${BASE_PRED}" --pseudo-root "${PSEUDO_ROOT}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" --imagenet-weight "${IMAGENET_WEIGHT}" \
  --output "${OUT}/coarse_predictions.json" \
  --summary "${OUT}/inference_summary.json" \
  --alpha 0.50 --context-ratio 1.0 --resolution 224 --batch-size 192 \
  --device cuda:0 >"${OUT}/inference.log" 2>&1

printf '%s\n' pixel_oer >"${OUT}/status.txt"
"${PYTHON_BIN}" scripts/train_pseudo_pixel_oer.py \
  --ground-truth "${GT}" --crop-predictions "${OUT}/coarse_predictions.json" \
  --output-dir "${OUT}/pixel_oer" --nms-iou 0.60 \
  >"${OUT}/pixel_oer.log" 2>&1
"${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "${GT}" --pred "${OUT}/pixel_oer/identity_predictions.json" \
  --output "${OUT}/identity_frontier.json" \
  >"${OUT}/identity_frontier.log" 2>&1
"${PYTHON_BIN}" scripts/compare_metric_risk_stages.py \
  --baseline "${BASELINE_FRONTIER}" \
  --stage "hardscore_coarse=${OUT}/identity_frontier.json" \
  --output "${OUT}/decision.json" >"${OUT}/decision.log" 2>&1
(
  cd "${OUT}"
  sha256sum coarse_predictions.json inference_summary.json \
    pixel_oer/identity_predictions.json identity_frontier.json decision.json \
    >SHA256SUMS
)
trap - ERR
printf '%s\n' complete >"${OUT}/status.txt"
