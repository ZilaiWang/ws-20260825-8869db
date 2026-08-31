#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/xh-202625}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}"
INPUT="${INPUT:-/workspace/inputs/HERA-GUARD-V5-OPEN-SET-V1}"
OUTPUT="${OUTPUT:-/workspace/results/HERA-GUARD-V5-CROP-ROBUSTNESS-V1}"
BASE_CROP_CACHE="${BASE_CROP_CACHE:-/workspace/results/HERA-GUARD-V4-OMQ-FACTORIAL-GPU/base_crop/cache.npz}"
FORMAL_MANIFEST="${FORMAL_MANIFEST:-/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv}"

mkdir -p "${OUTPUT}/logs"
cd "${REPO}"

run_variant() {
  local name="$1"
  local sampling="$2"
  local robustness="$3"
  local root="${OUTPUT}/${name}"
  mkdir -p "${root}/train" "${root}/eval"
  printf '%s:train\n' "${name}" >"${OUTPUT}/status.txt"
  for fold in 0 1 2; do
    "${PYTHON_BIN}" scripts/train_official_quality_head.py \
      --data "${BASE_CROP_CACHE}" \
      --output-dir "${root}/train" \
      --held-out-fold "${fold}" \
      --epochs 20 --batch-size 2048 --hidden-dim 192 \
      --sampling "${sampling}" --robustness "${robustness}" --device cuda:0 \
      >"${OUTPUT}/logs/${name}_fold${fold}.log" 2>&1
  done
  printf '%s:evaluate\n' "${name}" >"${OUTPUT}/status.txt"
  "${PYTHON_BIN}" scripts/export_omq_oof_predictions.py \
    --cache "${BASE_CROP_CACHE}" \
    --predictions "${INPUT}/y5-all-preds-d4.json" \
    --formal-crop-manifest "${FORMAL_MANIFEST}" \
    --score-dir "${root}/train" \
    --score-source quality \
    --output-dir "${root}/eval" \
    >"${OUTPUT}/logs/${name}_export.log" 2>&1
  "${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_coarse_thresholds.py \
    --gt "${root}/eval/formal_cv3_ground_truth.json" \
    --pred "${root}/eval/quality_oof_predictions.json" \
    --output "${root}/eval/frontier.json" \
    --threshold-step 0.005 \
    >"${OUTPUT}/logs/${name}_frontier.log" 2>&1
}

# D1/D2/D3 complete the frozen robustness factorial around the positive D0
# uniform-ERM crop-only baseline.  No architecture or threshold parameter changes.
run_variant group_balanced_erm group_balanced erm
run_variant uniform_group_dro uniform group_dro
run_variant group_balanced_group_dro group_balanced group_dro

printf 'complete\n' >"${OUTPUT}/status.txt"
sha256sum \
  "${OUTPUT}/group_balanced_erm/eval/frontier.json" \
  "${OUTPUT}/uniform_group_dro/eval/frontier.json" \
  "${OUTPUT}/group_balanced_group_dro/eval/frontier.json" \
  >"${OUTPUT}/SHA256SUMS"
