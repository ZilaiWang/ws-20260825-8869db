#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/xh-202625}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}"
INPUT="${INPUT:-/workspace/inputs/HERA-GUARD-V5-OPEN-SET-V1}"
OUTPUT="${OUTPUT:-/workspace/results/HERA-GUARD-V5-OPEN-SET-V1}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
FORMAL_MANIFEST="${FORMAL_MANIFEST:-/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv}"
P03_ROOT="${P03_ROOT:-/workspace/results/P03-FORMAL-CV3-V2}"
BASE_CROP_CACHE="${BASE_CROP_CACHE:-/workspace/results/HERA-GUARD-V4-OMQ-FACTORIAL-GPU/base_crop/cache.npz}"

mkdir -p "${OUTPUT}/logs" "${OUTPUT}/quality/train" "${OUTPUT}/quality/eval"
cd "${REPO}"

printf 'manifest\n' >"${OUTPUT}/status.txt"
"${PYTHON_BIN}" scripts/build_v5_open_set_manifest.py \
  --proposals "${INPUT}/y5_proposal_inference_manifest.csv" \
  --nodes "${INPUT}/nodes.csv" \
  --output "${OUTPUT}/open_set_manifest.csv" \
  --summary "${OUTPUT}/manifest_summary.json" \
  >"${OUTPUT}/logs/manifest.log" 2>&1

printf 'extract\n' >"${OUTPUT}/status.txt"
"${PYTHON_BIN}" scripts/extract_v5_open_set_features.py \
  --manifest "${OUTPUT}/open_set_manifest.csv" \
  --data-root "${DATA_ROOT}" \
  --checkpoint-pattern "${P03_ROOT}/ft-tight-224-fold{fold}/final_checkpoint.pt" \
  --output "${OUTPUT}/open_set_features.npz" \
  --summary "${OUTPUT}/extraction_summary.json" \
  --resolution 224 --context-scale 1.25 --batch-size 128 --device cuda:0 \
  >"${OUTPUT}/logs/extract.log" 2>&1

printf 'open_set_train\n' >"${OUTPUT}/status.txt"
"${PYTHON_BIN}" scripts/train_v5_open_set_head.py \
  --features "${OUTPUT}/open_set_features.npz" \
  --output-dir "${OUTPUT}/open_set_head" \
  --epochs 10 --batch-size 1024 --hidden-dim 256 \
  --max-sample-weight-ratio 20 --device cuda:0 \
  >"${OUTPUT}/logs/open_set_train.log" 2>&1

printf 'augment\n' >"${OUTPUT}/status.txt"
"${PYTHON_BIN}" scripts/augment_omq_with_open_set.py \
  --cache "${BASE_CROP_CACHE}" \
  --open-set-scores "${OUTPUT}/open_set_head/open_set_oof_scores.npz" \
  --output "${OUTPUT}/crop_open_set_omq.npz" \
  --summary "${OUTPUT}/augment_summary.json" \
  >"${OUTPUT}/logs/augment.log" 2>&1

printf 'quality_train\n' >"${OUTPUT}/status.txt"
for fold in 0 1 2; do
  "${PYTHON_BIN}" scripts/train_official_quality_head.py \
    --data "${OUTPUT}/crop_open_set_omq.npz" \
    --output-dir "${OUTPUT}/quality/train" \
    --held-out-fold "${fold}" \
    --epochs 20 --batch-size 2048 --hidden-dim 192 \
    --sampling uniform --robustness erm --device cuda:0 \
    >"${OUTPUT}/logs/quality_fold${fold}.log" 2>&1
done

printf 'evaluate\n' >"${OUTPUT}/status.txt"
"${PYTHON_BIN}" scripts/export_omq_oof_predictions.py \
  --cache "${OUTPUT}/crop_open_set_omq.npz" \
  --predictions "${INPUT}/y5-all-preds-d4.json" \
  --formal-crop-manifest "${FORMAL_MANIFEST}" \
  --score-dir "${OUTPUT}/quality/train" \
  --score-source quality \
  --output-dir "${OUTPUT}/quality/eval" \
  >"${OUTPUT}/logs/export.log" 2>&1
"${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_coarse_thresholds.py \
  --gt "${OUTPUT}/quality/eval/formal_cv3_ground_truth.json" \
  --pred "${OUTPUT}/quality/eval/quality_oof_predictions.json" \
  --output "${OUTPUT}/quality/eval/frontier.json" \
  --threshold-step 0.005 \
  >"${OUTPUT}/logs/frontier.log" 2>&1

printf 'complete\n' >"${OUTPUT}/status.txt"
sha256sum \
  "${OUTPUT}/manifest_summary.json" \
  "${OUTPUT}/extraction_summary.json" \
  "${OUTPUT}/open_set_head/summary.json" \
  "${OUTPUT}/augment_summary.json" \
  "${OUTPUT}/quality/eval/frontier.json" \
  >"${OUTPUT}/SHA256SUMS"
