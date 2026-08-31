#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/xh-202625}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}"
INPUT="${INPUT:-/workspace/inputs/HERA-GUARD-V5-OPEN-SET-V1}"
OUTPUT="${OUTPUT:-/workspace/results/HERA-GUARD-V5-F1-CROP-REVISIT-V1}"
BASE_CROP_CACHE="${BASE_CROP_CACHE:-/workspace/results/HERA-GUARD-V4-OMQ-FACTORIAL-GPU/base_crop/cache.npz}"
FORMAL_MANIFEST="${FORMAL_MANIFEST:-/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv}"

mkdir -p "${OUTPUT}/logs" "${OUTPUT}/quality/train" "${OUTPUT}/quality/eval"
cd "${REPO}"

printf 'augment\n' >"${OUTPUT}/status.txt"
"${PYTHON_BIN}" scripts/augment_omq_with_f1_foreground_logit.py \
  --cache "${BASE_CROP_CACHE}" --proposal-manifest "${INPUT}/open_set_manifest.csv" \
  --foreground-logits "${INPUT}/f1-fg-logits.json" \
  --output "${OUTPUT}/cache.npz" --summary "${OUTPUT}/augment_summary.json" \
  >"${OUTPUT}/logs/augment.log" 2>&1

printf 'quality_train\n' >"${OUTPUT}/status.txt"
for fold in 0 1 2; do
  "${PYTHON_BIN}" scripts/train_official_quality_head.py \
    --data "${OUTPUT}/cache.npz" --output-dir "${OUTPUT}/quality/train" \
    --held-out-fold "${fold}" --epochs 20 --batch-size 2048 --hidden-dim 192 \
    --sampling uniform --robustness erm --device cuda:0 \
    >"${OUTPUT}/logs/quality_fold${fold}.log" 2>&1
done

printf 'evaluate\n' >"${OUTPUT}/status.txt"
"${PYTHON_BIN}" scripts/export_omq_oof_predictions.py \
  --cache "${OUTPUT}/cache.npz" --predictions "${INPUT}/y5-all-preds-d4.json" \
  --formal-crop-manifest "${FORMAL_MANIFEST}" --score-dir "${OUTPUT}/quality/train" \
  --score-source quality --output-dir "${OUTPUT}/quality/eval" \
  >"${OUTPUT}/logs/export.log" 2>&1
"${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_coarse_thresholds.py \
  --gt "${OUTPUT}/quality/eval/formal_cv3_ground_truth.json" \
  --pred "${OUTPUT}/quality/eval/quality_oof_predictions.json" \
  --output "${OUTPUT}/quality/eval/frontier.json" --threshold-step 0.005 \
  >"${OUTPUT}/logs/frontier.log" 2>&1

printf 'complete\n' >"${OUTPUT}/status.txt"
sha256sum "${OUTPUT}/augment_summary.json" "${OUTPUT}/quality/eval/frontier.json" \
  >"${OUTPUT}/SHA256SUMS"
