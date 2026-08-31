#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/xh-202625}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}"
INPUT="${INPUT:-/workspace/inputs/HERA-GUARD-V5-OPEN-SET-V1}"
OUTPUT="${OUTPUT:-/workspace/results/HERA-GUARD-V5-SELECTIVE-VEHICLE-RESOLUTION-V1}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
P03_ROOT="${P03_ROOT:-/workspace/results/P03-FORMAL-CV3-V2}"
BASE_CROP_CACHE="${BASE_CROP_CACHE:-/workspace/results/HERA-GUARD-V4-OMQ-FACTORIAL-GPU/base_crop/cache.npz}"
FORMAL_MANIFEST="${FORMAL_MANIFEST:-/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv}"

mkdir -p "${OUTPUT}/logs"
cd "${REPO}"

run_resolution() {
  local resolution="$1"
  local name="vehicle_${resolution}"
  local root="${OUTPUT}/${name}"
  mkdir -p "${root}/head" "${root}/quality/train" "${root}/quality/eval"
  printf '%s:extract\n' "${name}" >"${OUTPUT}/status.txt"
  "${PYTHON_BIN}" scripts/extract_v5_open_set_features.py \
    --manifest "${INPUT}/open_set_manifest.csv" \
    --data-root "${DATA_ROOT}" \
    --checkpoint-pattern "${P03_ROOT}/ft-tight-224-fold{fold}/final_checkpoint.pt" \
    --output "${root}/features.npz" --summary "${root}/extraction_summary.json" \
    --resolution "${resolution}" --context-scale 1.25 --coarse-filter vehicle \
    --batch-size 128 --device cuda:0 \
    >"${OUTPUT}/logs/${name}_extract.log" 2>&1
  printf '%s:open_set_train\n' "${name}" >"${OUTPUT}/status.txt"
  "${PYTHON_BIN}" scripts/train_v5_open_set_head.py \
    --features "${root}/features.npz" --output-dir "${root}/head" \
    --epochs 10 --batch-size 1024 --hidden-dim 256 \
    --max-sample-weight-ratio 20 --device cuda:0 \
    >"${OUTPUT}/logs/${name}_head.log" 2>&1
  "${PYTHON_BIN}" scripts/augment_omq_with_selective_open_set.py \
    --cache "${BASE_CROP_CACHE}" --open-set-scores "${root}/head/open_set_oof_scores.npz" \
    --output "${root}/cache.npz" --summary "${root}/augment_summary.json" \
    >"${OUTPUT}/logs/${name}_augment.log" 2>&1
  printf '%s:quality_train\n' "${name}" >"${OUTPUT}/status.txt"
  for fold in 0 1 2; do
    "${PYTHON_BIN}" scripts/train_official_quality_head.py \
      --data "${root}/cache.npz" --output-dir "${root}/quality/train" \
      --held-out-fold "${fold}" --epochs 20 --batch-size 2048 --hidden-dim 192 \
      --sampling uniform --robustness erm --device cuda:0 \
      >"${OUTPUT}/logs/${name}_quality_fold${fold}.log" 2>&1
  done
  printf '%s:evaluate\n' "${name}" >"${OUTPUT}/status.txt"
  "${PYTHON_BIN}" scripts/export_omq_oof_predictions.py \
    --cache "${root}/cache.npz" --predictions "${INPUT}/y5-all-preds-d4.json" \
    --formal-crop-manifest "${FORMAL_MANIFEST}" --score-dir "${root}/quality/train" \
    --score-source quality --output-dir "${root}/quality/eval" \
    >"${OUTPUT}/logs/${name}_export.log" 2>&1
  "${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_coarse_thresholds.py \
    --gt "${root}/quality/eval/formal_cv3_ground_truth.json" \
    --pred "${root}/quality/eval/quality_oof_predictions.json" \
    --output "${root}/quality/eval/frontier.json" --threshold-step 0.005 \
    >"${OUTPUT}/logs/${name}_frontier.log" 2>&1
}

# The 224 arm controls for vehicle-only routing.  The 336 arm changes only
# input resolution and is admitted only if it improves that paired control.
run_resolution 224
run_resolution 336

printf 'complete\n' >"${OUTPUT}/status.txt"
sha256sum \
  "${OUTPUT}/vehicle_224/quality/eval/frontier.json" \
  "${OUTPUT}/vehicle_336/quality/eval/frontier.json" \
  >"${OUTPUT}/SHA256SUMS"
