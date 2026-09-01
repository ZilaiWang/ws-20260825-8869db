#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${OUT:?set OUT}"
: "${OFFICIAL_FOLD0_DATASET:?set OFFICIAL_FOLD0_DATASET}"
: "${Y5_INITIAL:?set Y5_INITIAL}"
: "${Y5_INITIAL_SHA256:?set Y5_INITIAL_SHA256}"
: "${PAIRED_CONTROL_DIR:?set PAIRED_CONTROL_DIR}"
: "${HARD_ROOT:?set HARD_ROOT}"
: "${SENTINEL_ROOT:?set SENTINEL_ROOT}"
: "${BASELINE_WEIGHT_ROOT:?set BASELINE_WEIGHT_ROOT}"
: "${BASELINE_HARD_PRED_ROOT:?set BASELINE_HARD_PRED_ROOT}"
: "${BASELINE_SENTINEL_PRED_ROOT:?set BASELINE_SENTINEL_PRED_ROOT}"

STATUS="${OUT}/status.txt"
CLASSIFICATION_LOSS="${CLASSIFICATION_LOSS:-varifocal}"
[[ "${CLASSIFICATION_LOSS}" = varifocal || \
   "${CLASSIFICATION_LOSS}" = hard_negative_focal || \
   "${CLASSIFICATION_LOSS}" = ship_vehicle_hard_negative_focal ]] || {
  echo "invalid CLASSIFICATION_LOSS" >&2
  exit 2
}
RUN_SLUG="${CLASSIFICATION_LOSS//_/-}"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed_exit_%s\n" "$code" >"${STATUS}"; fi' EXIT
mkdir -p "${OUT}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

printf '%s  %s\n' "${Y5_INITIAL_SHA256}" "${Y5_INITIAL}" | sha256sum -c -
[[ -f "${PAIRED_CONTROL_DIR}/training_result.json" ]] || {
  printf '%s\n' blocked_paired_control_incomplete >"${STATUS}"
  exit 2
}
sha256sum \
  scripts/train_external_initialized_y5_fine.py \
  scripts/run_multifamily_cv3_pseudo_eval.py \
  scripts/merge_coco_predictions.py \
  scripts/analyze_cv3_oof_pseudo_frontier.py \
  src/rsdet/innovation/quality_aware_loss.py \
  src/rsdet/external/transfer.py >"${OUT}/CODE_SHA256.txt"

CANDIDATE="${OUT}/fine/${RUN_SLUG}-fold0"
CHECKPOINT="${CANDIDATE}/runs/foundation/weights/last.pt"
if [[ ! -f "${CANDIDATE}/training_result.json" ]]; then
  [[ ! -f "${CHECKPOINT}" ]] || {
    echo "incomplete existing quality-aware run requires forensic review" >&2
    exit 3
  }
  printf '%s\n' quality_aware_candidate_40ep >"${STATUS}"
  "${PYTHON_BIN}" scripts/train_external_initialized_y5_fine.py \
    --dataset "${OFFICIAL_FOLD0_DATASET}" --external-weights "${Y5_INITIAL}" \
    --expected-weight-sha256 "${Y5_INITIAL_SHA256}" --output-dir "${CANDIDATE}" \
    --epochs 40 --head-warmup-epochs 8 --freeze-layers 10 \
    --imgsz 1024 --batch 12 --workers 8 --seed 20260901 --device cuda:0 \
    --classification-loss "${CLASSIFICATION_LOSS}" \
    --varifocal-alpha 0.75 --varifocal-gamma 2.0 \
    >"${OUT}/fine-varifocal.log" 2>&1
fi

run_fold0_eval() {
  local name=$1 weight=$2 pseudo_root=$3 baseline_root=$4 benchmark=$5
  local run="${OUT}/evaluation/${name}/${benchmark}"
  mkdir -p "${run}"
  if [[ ! -f "${run}/fold0/run_summary.json" ]]; then
    "${PYTHON_BIN}" scripts/run_multifamily_cv3_pseudo_eval.py \
      --pseudo-root "${pseudo_root}" --family yolo \
      --weights "${weight}" \
        "${BASELINE_WEIGHT_ROOT}/fold_1/training/runs/foundation/weights/last.pt" \
        "${BASELINE_WEIGHT_ROOT}/fold_2/training/runs/foundation/weights/last.pt" \
      --folds 0 --output-dir "${run}/fold0" --score-floor 0.03 \
      --batch-size 4 --device cuda:0 >"${run}/fold0.log" 2>&1
  fi
  "${PYTHON_BIN}" scripts/merge_coco_predictions.py \
    --input "${run}/fold0/predictions.json" \
    --input "${baseline_root}/fold_1/predictions.json" \
    --input "${baseline_root}/fold_2/predictions.json" \
    --output "${run}/predictions.json"
  "${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
    --gt "${pseudo_root}/ground_truth.json" --pred "${run}/predictions.json" \
    --output "${run}/frontier.json" --threshold-start 0.001 \
    --threshold-stop 0.996 --threshold-step 0.005 \
    --fdr-levels 0.10 0.12 0.15 0.20 >"${run}/frontier.log" 2>&1
}

printf '%s\n' fixed_benchmark_inference >"${STATUS}"
CONTROL_WEIGHT="${PAIRED_CONTROL_DIR}/runs/foundation/weights/last.pt"
run_fold0_eval candidate "${CHECKPOINT}" "${HARD_ROOT}" "${BASELINE_HARD_PRED_ROOT}" hard
run_fold0_eval control "${CONTROL_WEIGHT}" "${HARD_ROOT}" "${BASELINE_HARD_PRED_ROOT}" hard
run_fold0_eval candidate "${CHECKPOINT}" "${SENTINEL_ROOT}" "${BASELINE_SENTINEL_PRED_ROOT}" sentinel
run_fold0_eval control "${CONTROL_WEIGHT}" "${SENTINEL_ROOT}" "${BASELINE_SENTINEL_PRED_ROOT}" sentinel

find "${OUT}" -type f \( -name training_result.json -o -name frontier.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"${OUT}/RESULT_SHA256.txt"
printf '%s\n' complete_ready_for_analysis >"${STATUS}"
