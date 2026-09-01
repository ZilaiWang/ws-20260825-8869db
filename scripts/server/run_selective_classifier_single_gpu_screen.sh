#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${OUT:?set OUT}"
: "${OFFICIAL_FOLD0_DATASET:?set OFFICIAL_FOLD0_DATASET}"
: "${INCUMBENT_WEIGHT:?set INCUMBENT_WEIGHT}"
: "${INCUMBENT_WEIGHT_SHA256:?set INCUMBENT_WEIGHT_SHA256}"
: "${BASELINE_ROOT:?set BASELINE_ROOT}"
: "${NORMAL_GT:?set NORMAL_GT}"
: "${HARD_ROOT:?set HARD_ROOT}"
: "${HARD_BASELINE_PRED_ROOT:?set HARD_BASELINE_PRED_ROOT}"
: "${SENTINEL_ROOT:?set SENTINEL_ROOT}"
: "${SENTINEL_BASELINE_PRED_ROOT:?set SENTINEL_BASELINE_PRED_ROOT}"

STATUS="${OUT}/status.txt"
CLASSIFICATION_LOSS="${CLASSIFICATION_LOSS:-ship_vehicle_hard_negative_focal}"
TRAIN_SCOPE="${TRAIN_SCOPE:-final_rows}"
PROJECTION_ARGS=()
if [[ -n "${MAX_WEIGHT_RELATIVE_DELTA:-}" || -n "${MAX_BIAS_DELTA:-}" ]]; then
  [[ -n "${MAX_WEIGHT_RELATIVE_DELTA:-}" && -n "${MAX_BIAS_DELTA:-}" ]] || {
    echo "both projection bounds must be supplied" >&2
    exit 2
  }
  PROJECTION_ARGS=(
    --max-weight-relative-delta "${MAX_WEIGHT_RELATIVE_DELTA}"
    --max-bias-delta "${MAX_BIAS_DELTA}"
  )
fi
PROJECTION_ARGS+=(--train-scope "${TRAIN_SCOPE}")
if [[ -n "${MAX_BRANCH_RELATIVE_DELTA:-}" ]]; then
  PROJECTION_ARGS+=(--max-branch-relative-delta "${MAX_BRANCH_RELATIVE_DELTA}")
fi
[[ "${CLASSIFICATION_LOSS}" = bce || \
   "${CLASSIFICATION_LOSS}" = ship_vehicle_hard_negative_focal ]] || {
  echo "invalid CLASSIFICATION_LOSS" >&2
  exit 2
}
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed_exit_%s\n" "$code" >"${STATUS}"; fi' EXIT
mkdir -p "${OUT}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

printf '%s  %s\n' "${INCUMBENT_WEIGHT_SHA256}" "${INCUMBENT_WEIGHT}" | sha256sum -c -
sha256sum \
  scripts/train_selective_classifier_finetune.py \
  scripts/run_multifamily_cv3_pseudo_eval.py \
  scripts/merge_coco_predictions.py \
  scripts/analyze_cv3_oof_pseudo_frontier.py \
  scripts/server/run_y5_fold0_normal_replacement_eval.sh \
  src/rsdet/innovation/quality_aware_loss.py >"${OUT}/CODE_SHA256.txt"

CANDIDATE="${OUT}/fine/selective-classifier-fold0"
CHECKPOINT="${CANDIDATE}/runs/selective_classifier/weights/last.pt"
if [[ ! -f "${CANDIDATE}/training_result.json" ]]; then
  [[ ! -f "${CHECKPOINT}" ]] || {
    echo "incomplete existing selective-classifier run requires forensic review" >&2
    exit 3
  }
  printf '%s\n' selective_classifier_training_12ep >"${STATUS}"
  "${PYTHON_BIN}" scripts/train_selective_classifier_finetune.py \
    --dataset "${OFFICIAL_FOLD0_DATASET}" --weights "${INCUMBENT_WEIGHT}" \
    --expected-weight-sha256 "${INCUMBENT_WEIGHT_SHA256}" \
    --output-dir "${CANDIDATE}" --epochs 12 --imgsz 1024 --batch 12 \
    --workers 8 --seed 20260901 --device cuda:0 --alpha 0.75 --gamma 2.0 \
    --classification-loss "${CLASSIFICATION_LOSS}" \
    "${PROJECTION_ARGS[@]}" \
    >"${OUT}/train.log" 2>&1
fi

run_pseudo_fold0() {
  local benchmark=$1 pseudo_root=$2 baseline_pred_root=$3
  local run="${OUT}/evaluation/candidate/${benchmark}"
  mkdir -p "${run}"
  if [[ ! -f "${run}/fold0/run_summary.json" ]]; then
    "${PYTHON_BIN}" scripts/run_multifamily_cv3_pseudo_eval.py \
      --pseudo-root "${pseudo_root}" --family yolo --weights "${CHECKPOINT}" \
      "${BASELINE_ROOT}/fold_1/training/runs/foundation/weights/last.pt" \
      "${BASELINE_ROOT}/fold_2/training/runs/foundation/weights/last.pt" \
      --folds 0 --output-dir "${run}/fold0" --score-floor 0.03 \
      --batch-size 4 --device cuda:0 >"${run}/fold0.log" 2>&1
  fi
  "${PYTHON_BIN}" scripts/merge_coco_predictions.py \
    --input "${run}/fold0/predictions.json" \
    --input "${baseline_pred_root}/fold_1/predictions.json" \
    --input "${baseline_pred_root}/fold_2/predictions.json" \
    --output "${run}/predictions.json"
  "${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
    --gt "${pseudo_root}/ground_truth.json" --pred "${run}/predictions.json" \
    --output "${run}/frontier.json" --threshold-start 0.001 \
    --threshold-stop 0.996 --threshold-step 0.005 \
    --fdr-levels 0.10 0.12 0.15 0.20 >"${run}/frontier.log" 2>&1
}

printf '%s\n' fixed_three_domain_inference >"${STATUS}"
run_pseudo_fold0 hard "${HARD_ROOT}" "${HARD_BASELINE_PRED_ROOT}"
run_pseudo_fold0 sentinel "${SENTINEL_ROOT}" "${SENTINEL_BASELINE_PRED_ROOT}"
REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" WEIGHT="${CHECKPOINT}" \
  OUT="${OUT}/evaluation/candidate/normal" BASELINE_ROOT="${BASELINE_ROOT}" \
  GROUND_TRUTH="${NORMAL_GT}" bash scripts/server/run_y5_fold0_normal_replacement_eval.sh

mkdir -p "${OUT}/evaluation/incumbent"
for benchmark in hard sentinel; do
  if [[ "${benchmark}" = hard ]]; then
    pseudo_root="${HARD_ROOT}"; pred_root="${HARD_BASELINE_PRED_ROOT}"
  else
    pseudo_root="${SENTINEL_ROOT}"; pred_root="${SENTINEL_BASELINE_PRED_ROOT}"
  fi
  "${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
    --gt "${pseudo_root}/ground_truth.json" --pred "${pred_root}/predictions.json" \
    --output "${OUT}/evaluation/incumbent/${benchmark}.json" \
    --threshold-start 0.001 --threshold-stop 0.996 --threshold-step 0.005 \
    --fdr-levels 0.10 0.12 0.15 0.20 \
    >"${OUT}/evaluation/incumbent/${benchmark}.log" 2>&1
done
"${PYTHON_BIN}" scripts/merge_coco_predictions.py \
  --input "${BASELINE_ROOT}/fold_0/predictions_low.json" \
  --input "${BASELINE_ROOT}/fold_1/predictions_low.json" \
  --input "${BASELINE_ROOT}/fold_2/predictions_low.json" \
  --output "${OUT}/evaluation/incumbent/normal_predictions.json"
"${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "${NORMAL_GT}" --pred "${OUT}/evaluation/incumbent/normal_predictions.json" \
  --output "${OUT}/evaluation/incumbent/normal.json" \
  --threshold-start 0.001 --threshold-stop 0.996 --threshold-step 0.005 \
  --fdr-levels 0.10 0.12 0.15 0.20 \
  >"${OUT}/evaluation/incumbent/normal.log" 2>&1

"${PYTHON_BIN}" scripts/evaluate_pseudo_with_frozen_thresholds.py \
  --gt "${SENTINEL_ROOT}/ground_truth.json" \
  --pred "${OUT}/evaluation/candidate/sentinel/predictions.json" \
  --source-frontier "${OUT}/evaluation/candidate/hard/frontier.json" \
  --fdr-level 0.15 --output "${OUT}/evaluation/candidate/sentinel/frozen.json" \
  >"${OUT}/evaluation/candidate/sentinel/frozen.log" 2>&1
"${PYTHON_BIN}" scripts/evaluate_pseudo_with_frozen_thresholds.py \
  --gt "${SENTINEL_ROOT}/ground_truth.json" \
  --pred "${SENTINEL_BASELINE_PRED_ROOT}/predictions.json" \
  --source-frontier "${OUT}/evaluation/incumbent/hard.json" \
  --fdr-level 0.15 --output "${OUT}/evaluation/incumbent/sentinel_frozen.json" \
  >"${OUT}/evaluation/incumbent/sentinel_frozen.log" 2>&1

"${PYTHON_BIN}" scripts/decide_hera_guard_final_candidate.py \
  --normal-base "${OUT}/evaluation/incumbent/normal.json" \
  --normal-candidate "${OUT}/evaluation/candidate/normal/frontier.json" \
  --hard-base "${OUT}/evaluation/incumbent/hard.json" \
  --hard-candidate "${OUT}/evaluation/candidate/hard/frontier.json" \
  --sentinel-base "${OUT}/evaluation/incumbent/sentinel.json" \
  --sentinel-candidate "${OUT}/evaluation/candidate/sentinel/frontier.json" \
  --sentinel-base-frozen "${OUT}/evaluation/incumbent/sentinel_frozen.json" \
  --sentinel-candidate-frozen "${OUT}/evaluation/candidate/sentinel/frozen.json" \
  --fdr-level 0.150 --selection-mode fdr_level --output "${OUT}/decision.json" \
  >"${OUT}/decision.log" 2>&1

find "${OUT}" -type f \( -name training_result.json -o -name frontier.json \
  -o -name decision.json -o -name frozen.json \) -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/RESULT_SHA256.txt"
printf '%s\n' complete_with_three_domain_decision >"${STATUS}"
