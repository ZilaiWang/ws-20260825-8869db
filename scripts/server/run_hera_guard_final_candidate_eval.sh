#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${OUT:?set OUT}"
: "${CANDIDATE_FOLD:?set CANDIDATE_FOLD to 0, 1, or 2}"
: "${CANDIDATE_WEIGHT:?set CANDIDATE_WEIGHT}"
: "${BASE_WEIGHT_0:?set BASE_WEIGHT_0}"
: "${BASE_WEIGHT_1:?set BASE_WEIGHT_1}"
: "${BASE_WEIGHT_2:?set BASE_WEIGHT_2}"
: "${NORMAL_ROOT:?set NORMAL_ROOT}"
: "${HARD_ROOT:?set HARD_ROOT}"
: "${SENTINEL_ROOT:?set SENTINEL_ROOT}"

[[ "${CANDIDATE_FOLD}" =~ ^[012]$ ]] || { echo invalid CANDIDATE_FOLD >&2; exit 2; }
mkdir -p "${OUT}"
STATUS="${OUT}/status.txt"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed_exit_%s\n" "$code" >"${STATUS}"; fi' EXIT
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

BASE=("${BASE_WEIGHT_0}" "${BASE_WEIGHT_1}" "${BASE_WEIGHT_2}")
CANDIDATE=("${BASE_WEIGHT_0}" "${BASE_WEIGHT_1}" "${BASE_WEIGHT_2}")
CANDIDATE[${CANDIDATE_FOLD}]="${CANDIDATE_WEIGHT}"
ADAPTER_ARGS=()
if [[ -n "${CANDIDATE_ADAPTER:-}" ]]; then
  ADAPTERS=("-" "-" "-")
  ADAPTERS[${CANDIDATE_FOLD}]="${CANDIDATE_ADAPTER}"
  ADAPTER_ARGS=(--agreement-adapters "${ADAPTERS[@]}")
fi

for condition in normal hard sentinel; do
  case "${condition}" in
    normal) ROOT="${NORMAL_ROOT}" ;;
    hard) ROOT="${HARD_ROOT}" ;;
    sentinel) ROOT="${SENTINEL_ROOT}" ;;
  esac
  printf 'inference_%s\n' "${condition}" >"${STATUS}"
  BASE_PRED="${OUT}/${condition}/base/predictions.json"
  if [[ -n "${BASELINE_EVAL_ROOT:-}" ]]; then
    BASE_PRED="${BASELINE_EVAL_ROOT}/${condition}/predictions.json"
    [[ -f "${BASE_PRED}" ]] || { echo "missing cached baseline ${BASE_PRED}" >&2; exit 2; }
  elif [[ ! -f "${BASE_PRED}" ]]; then
    "${PYTHON_BIN}" scripts/run_multifamily_cv3_pseudo_eval.py \
      --pseudo-root "${ROOT}" --family yolo --weights "${BASE[@]}" \
      --output-dir "${OUT}/${condition}/base" --score-floor 0.03 \
      --batch-size 4 --device cuda:0 >"${OUT}/${condition}-base.log" 2>&1
  fi
  if [[ ! -f "${OUT}/${condition}/candidate-fold/predictions.json" ]]; then
    "${PYTHON_BIN}" scripts/run_multifamily_cv3_pseudo_eval.py \
      --pseudo-root "${ROOT}" --family yolo --weights "${CANDIDATE[@]}" \
      "${ADAPTER_ARGS[@]}" --output-dir "${OUT}/${condition}/candidate-fold" \
      --folds "${CANDIDATE_FOLD}" \
      --score-floor 0.03 --batch-size 4 --device cuda:0 \
      >"${OUT}/${condition}-candidate.log" 2>&1
  fi
  mkdir -p "${OUT}/${condition}/candidate"
  "${PYTHON_BIN}" scripts/replace_prediction_fold.py \
    --base "${BASE_PRED}" \
    --candidate-fold "${OUT}/${condition}/candidate-fold/predictions.json" \
    --fold "${CANDIDATE_FOLD}" \
    --output "${OUT}/${condition}/candidate/predictions.json" \
    --audit "${OUT}/${condition}/candidate/replace_audit.json"
  for route in base candidate; do
    PRED="${OUT}/${condition}/${route}/predictions.json"
    [[ "${route}" = base ]] && PRED="${BASE_PRED}"
    "${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
      --gt "${ROOT}/ground_truth.json" --pred "${PRED}" \
      --output "${OUT}/${condition}/${route}_frontier.json" \
      --threshold-start 0.0 --threshold-stop 1.0 --threshold-step 0.001 \
      --fdr-levels 0.10 0.12 0.15 0.20 >"${OUT}/${condition}-${route}-frontier.log" 2>&1
  done
done
printf '%s\n' decision >"${STATUS}"
"${PYTHON_BIN}" scripts/decide_hera_guard_final_candidate.py \
  --normal-base "${OUT}/normal/base_frontier.json" \
  --normal-candidate "${OUT}/normal/candidate_frontier.json" \
  --hard-base "${OUT}/hard/base_frontier.json" \
  --hard-candidate "${OUT}/hard/candidate_frontier.json" \
  --sentinel-base "${OUT}/sentinel/base_frontier.json" \
  --sentinel-candidate "${OUT}/sentinel/candidate_frontier.json" \
  --output "${OUT}/decision.json" >"${OUT}/decision.log" 2>&1
find "${OUT}" -type f \( -name '*frontier.json' -o -name 'decision.json' -o -name 'run_summary.json' \) -print0 | sort -z | xargs -0 sha256sum >"${OUT}/SHA256SUMS"
printf '%s\n' complete >"${STATUS}"
