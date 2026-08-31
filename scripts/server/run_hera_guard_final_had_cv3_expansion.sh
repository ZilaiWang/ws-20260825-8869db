#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${OUT:?set OUT}"
: "${TRAIN_ROOT:?set TRAIN_ROOT to the HERA screen HAD directory}"
: "${TEACHER_CACHE:?set TEACHER_CACHE}"
: "${CV3_MANIFEST:?set CV3_MANIFEST}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${BASE_WEIGHT_0:?set BASE_WEIGHT_0}"
: "${BASE_WEIGHT_1:?set BASE_WEIGHT_1}"
: "${BASE_WEIGHT_2:?set BASE_WEIGHT_2}"
: "${BASE_SHA_1:?set BASE_SHA_1}"
: "${BASELINE_EVAL_ROOT:?set BASELINE_EVAL_ROOT}"
: "${NORMAL_ROOT:?set NORMAL_ROOT}"
: "${HARD_ROOT:?set HARD_ROOT}"
: "${SENTINEL_ROOT:?set SENTINEL_ROOT}"

mkdir -p "${OUT}"
STATUS="${OUT}/status.txt"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed_exit_%s\n" "$code" >"${STATUS}"; fi' EXIT
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=2

for path in \
  "${TRAIN_ROOT}/fold0-branch_only/training_result.json" \
  "${TRAIN_ROOT}/fold2-branch_only/training_result.json"; do
  [[ -f "${path}" ]] || { echo "missing admitted branch training result: ${path}" >&2; exit 2; }
done
sha256sum \
  scripts/train_in_model_dfine_agreement.py \
  scripts/run_multifamily_cv3_pseudo_eval.py \
  scripts/analyze_cv3_oof_pseudo_frontier.py \
  scripts/decide_hera_guard_final_candidate.py \
  src/rsdet/innovation/agreement_runtime.py \
  src/rsdet/innovation/yolo_feature_quality.py >"${OUT}/CODE_SHA256.txt"

FOLD1="${TRAIN_ROOT}/fold1-branch_only"
if [[ ! -f "${FOLD1}/training_result.json" ]]; then
  if [[ -f "${FOLD1}/adapter_last.pt" ]]; then
    echo "incomplete existing fold1 HAD checkpoint requires review" >&2
    exit 3
  fi
  printf '%s\n' training_fold1_branch_only >"${STATUS}"
  CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN}" scripts/train_in_model_dfine_agreement.py \
    --teacher-cache "${TEACHER_CACHE}" --split-manifest "${CV3_MANIFEST}" \
    --data-root "${DATA_ROOT}" --base-checkpoint "${BASE_WEIGHT_1}" \
    --expected-base-sha256 "${BASE_SHA_1}" --output-dir "${FOLD1}" \
    --held-out-fold 1 --mode branch_only --epochs 8 --imgsz 1024 \
    --max-proposals-per-image 64 --projection-dim 64 --hidden-dim 128 \
    --seed 20260831 --device cuda:0 >"${OUT}/train-fold1.log" 2>&1
fi

WEIGHTS=("${BASE_WEIGHT_0}" "${BASE_WEIGHT_1}" "${BASE_WEIGHT_2}")
ADAPTERS=(
  "${TRAIN_ROOT}/fold0-branch_only/adapter_last.pt"
  "${FOLD1}/adapter_last.pt"
  "${TRAIN_ROOT}/fold2-branch_only/adapter_last.pt"
)
for condition in normal hard sentinel; do
  case "${condition}" in
    normal) ROOT="${NORMAL_ROOT}" ;;
    hard) ROOT="${HARD_ROOT}" ;;
    sentinel) ROOT="${SENTINEL_ROOT}" ;;
  esac
  BASE_PRED="${BASELINE_EVAL_ROOT}/${condition}/base/predictions.json"
  [[ -f "${BASE_PRED}" ]] || { echo "missing frozen baseline: ${BASE_PRED}" >&2; exit 2; }
  printf 'inference_%s\n' "${condition}" >"${STATUS}"
  if [[ ! -f "${OUT}/${condition}/candidate/predictions.json" ]]; then
    "${PYTHON_BIN}" scripts/run_multifamily_cv3_pseudo_eval.py \
      --pseudo-root "${ROOT}" --family yolo --weights "${WEIGHTS[@]}" \
      --agreement-adapters "${ADAPTERS[@]}" \
      --output-dir "${OUT}/${condition}/candidate" --score-floor 0.03 \
      --batch-size 4 --device cuda:0 >"${OUT}/${condition}-candidate.log" 2>&1
  fi
  for route in base candidate; do
    PRED="${OUT}/${condition}/candidate/predictions.json"
    [[ "${route}" = base ]] && PRED="${BASE_PRED}"
    "${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
      --gt "${ROOT}/ground_truth.json" --pred "${PRED}" \
      --output "${OUT}/${condition}/${route}_frontier.json" \
      --threshold-start 0.0 --threshold-stop 1.0 --threshold-step 0.001 \
      --fdr-levels 0.10 0.12 0.15 0.20 >"${OUT}/${condition}-${route}-frontier.log" 2>&1
  done
done

for route in base candidate; do
  PRED="${OUT}/sentinel/candidate/predictions.json"
  [[ "${route}" = base ]] && PRED="${BASELINE_EVAL_ROOT}/sentinel/base/predictions.json"
  "${PYTHON_BIN}" scripts/evaluate_pseudo_with_frozen_thresholds.py \
    --gt "${SENTINEL_ROOT}/ground_truth.json" --pred "${PRED}" \
    --source-frontier "${OUT}/hard/${route}_frontier.json" --fdr-level 0.15 \
    --output "${OUT}/sentinel/${route}_frozen_from_hard.json" \
    >"${OUT}/sentinel-${route}-frozen.log" 2>&1
done

printf '%s\n' decision >"${STATUS}"
"${PYTHON_BIN}" scripts/decide_hera_guard_final_candidate.py \
  --normal-base "${OUT}/normal/base_frontier.json" \
  --normal-candidate "${OUT}/normal/candidate_frontier.json" \
  --hard-base "${OUT}/hard/base_frontier.json" \
  --hard-candidate "${OUT}/hard/candidate_frontier.json" \
  --sentinel-base "${OUT}/sentinel/base_frontier.json" \
  --sentinel-candidate "${OUT}/sentinel/candidate_frontier.json" \
  --sentinel-base-frozen "${OUT}/sentinel/base_frozen_from_hard.json" \
  --sentinel-candidate-frozen "${OUT}/sentinel/candidate_frozen_from_hard.json" \
  --output "${OUT}/decision.json" >"${OUT}/decision.log" 2>&1
find "${OUT}" -type f \( -name '*frontier.json' -o -name decision.json -o -name training_result.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"${OUT}/SHA256SUMS"
printf '%s\n' complete >"${STATUS}"
