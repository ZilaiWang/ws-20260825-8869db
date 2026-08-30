#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/xh-202625}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}"
RESULTS="${RESULTS:-/workspace/results}"
INPUT="${INPUT:-${RESULTS}/HERA-GUARD-V4-OMQ-INPUT}"
Q0="${RESULTS}/HERA-GUARD-V4-OMQ-Q0"
Q1="${RESULTS}/HERA-GUARD-V4-OMQ-Q1"
Q2="${RESULTS}/HERA-GUARD-V4-OMQ-Q2"

cd "${REPO}"

train_stage() {
  local stage="$1"
  local data="$2"
  local rank_flag="$3"
  local batch_size="$4"
  mkdir -p "${stage}/train" "${stage}/eval/quality"
  printf 'train\n' >"${stage}/status.txt"
  for fold in 0 1 2; do
    "${PYTHON_BIN}" scripts/train_official_quality_head.py \
      --data "${data}" \
      --output-dir "${stage}/train" \
      --held-out-fold "${fold}" \
      --epochs 20 \
      --batch-size "${batch_size}" \
      --hidden-dim 192 \
      --sampling uniform \
      --robustness erm \
      ${rank_flag} \
      --device cuda:0 \
      >"${stage}/train/fold${fold}.log" 2>&1
  done
  printf 'evaluate\n' >"${stage}/status.txt"
  "${PYTHON_BIN}" scripts/export_omq_oof_predictions.py \
    --cache "${data}" \
    --predictions "${INPUT}/y5-all-preds-d4.json" \
    --formal-crop-manifest "${INPUT}/formal_crop_manifest.csv" \
    --score-dir "${stage}/train" \
    --score-source quality \
    --output-dir "${stage}/eval/quality" \
    >"${stage}/eval/quality/export.log" 2>&1
  "${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_coarse_thresholds.py \
    --gt "${stage}/eval/quality/formal_cv3_ground_truth.json" \
    --pred "${stage}/eval/quality/quality_oof_predictions.json" \
    --output "${stage}/eval/quality/coarse_frontier.json" \
    --threshold-step 0.005 \
    >"${stage}/eval/quality/frontier.log" 2>&1
  printf 'complete\n' >"${stage}/status.txt"
}

train_stage "${Q0}" "${Q0}/omq_metadata.npz" "" 2048
train_stage "${Q1}" "${Q1}/omq_metadata_fpn.npz" "" 1024
train_stage "${Q2}" "${Q1}/omq_metadata_fpn.npz" "--rank-enabled" 2048

printf 'complete\n' >"${RESULTS}/HERA-GUARD-V4-OMQ-Q0-Q2.status"
