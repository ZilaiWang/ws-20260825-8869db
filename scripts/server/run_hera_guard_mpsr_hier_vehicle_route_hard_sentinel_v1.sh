#!/usr/bin/env bash
set -Eeuo pipefail

# Source-disjoint confirmation for the fold0 hierarchical detector.  The same
# fold0 checkpoint is intentionally used on every pseudo fold: this is a model
# generalization check, not CV3 threshold fitting.  Only Vehicle is routed from
# the candidate; labels 0..23 remain bitwise owned by P40.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
BASE_WEIGHT=${BASE_WEIGHT:-/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1/s1024/runs/foundation/weights/last.pt}
CAND_WEIGHT=${CAND_WEIGHT:-/root/autodl-tmp/results/HERA-GUARD-MPSR-HIER-S256V128-FOLD0-40EP-3X4080-B60-V1/training/runs/resolution_adaptation/weights/last.pt}
HARD_ROOT=${HARD_ROOT:-/root/autodl-tmp/pseudo10k-trial-mix-local}
SENTINEL_ROOT=${SENTINEL_ROOT:-/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-MPSR-HIER-VEHICLE-ROUTE-HARD-SENTINEL-V1}
STATUS=${OUT}/status.txt

if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}"
exec 9>"${OUT}.lock"
flock -n 9 || exit 4
failed() { code=$?; printf 'failed_exit_%s\n' "${code}" >"${STATUS}"; exit "${code}"; }
trap failed ERR INT TERM
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for path in "${BASE_WEIGHT}" "${CAND_WEIGHT}" \
  "${HARD_ROOT}/ground_truth.json" "${SENTINEL_ROOT}/ground_truth.json"; do
  test -f "${path}"
done
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'RTX 3090'

for condition in hard sentinel; do
  if [[ "${condition}" = hard ]]; then root="${HARD_ROOT}"; else root="${SENTINEL_ROOT}"; fi
  mkdir -p "${OUT}/${condition}"
  printf '%s_baseline_inference\n' "${condition}" >"${STATUS}"
  "${PY}" scripts/run_multifamily_cv3_pseudo_eval.py \
    --pseudo-root "${root}" --family yolo \
    --weights "${BASE_WEIGHT}" "${BASE_WEIGHT}" "${BASE_WEIGHT}" \
    --output-dir "${OUT}/${condition}/baseline" --score-floor 0.001 \
    --batch-size 4 --device cuda:0 --imgsz 1280 --tile-size 1024 --overlap 256 \
    >"${OUT}/${condition}/baseline_infer.log" 2>&1

  printf '%s_candidate_inference\n' "${condition}" >"${STATUS}"
  "${PY}" scripts/run_multifamily_cv3_pseudo_eval.py \
    --pseudo-root "${root}" --family yolo \
    --weights "${CAND_WEIGHT}" "${CAND_WEIGHT}" "${CAND_WEIGHT}" \
    --output-dir "${OUT}/${condition}/candidate" --score-floor 0.001 \
    --batch-size 4 --device cuda:0 --imgsz 1280 --tile-size 1024 --overlap 256 \
    >"${OUT}/${condition}/candidate_infer.log" 2>&1

  printf '%s_vehicle_route_evaluation\n' "${condition}" >"${STATUS}"
  "${PY}" scripts/compose_class_disjoint_predictions.py \
    --primary "${OUT}/${condition}/baseline/predictions.json" \
    --expert "${OUT}/${condition}/candidate/predictions.json" \
    --primary-labels 0-23 --expert-labels 24 \
    --output "${OUT}/${condition}/vehicle_route_predictions.json"
  "${PY}" scripts/evaluate_fixed_score_threshold.py \
    --gt "${root}/ground_truth.json" \
    --pred "${OUT}/${condition}/baseline/predictions.json" --threshold 0.546 \
    --output "${OUT}/${condition}/baseline_fixed_0546.json"
  "${PY}" scripts/evaluate_fixed_score_threshold.py \
    --gt "${root}/ground_truth.json" \
    --pred "${OUT}/${condition}/vehicle_route_predictions.json" --threshold 0.546 \
    --output "${OUT}/${condition}/candidate_fixed_0546.json"
  "${PY}" scripts/compare_candidate_trend.py \
    --baseline "${OUT}/${condition}/baseline_fixed_0546.json" \
    --candidate "${OUT}/${condition}/candidate_fixed_0546.json" \
    --output "${OUT}/${condition}/paired_comparison.json"
done

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
