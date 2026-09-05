#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625-capscale}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
FOLD0_ROOT=${FOLD0_ROOT:-/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1}
CONFIRM_ROOT=${CONFIRM_ROOT:-/root/autodl-tmp/results/YOLO-SCALE-CV3-CONFIRM-V1}
R3_ROOT=${R3_ROOT:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-R3-FIXED-BENCHMARKS-V1}
HARD_ROOT=${HARD_ROOT:-/root/autodl-tmp/pseudo10k-trial-mix-local}
SENTINEL_ROOT=${SENTINEL_ROOT:-/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1}
OUT=${OUT:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-R2-FIXED-BENCHMARKS-V1}
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

S1280=(
  "${FOLD0_ROOT}/s1280/runs/foundation/weights/last.pt"
  "${CONFIRM_ROOT}/s1280/fold_1/runs/foundation/weights/last.pt"
  "${CONFIRM_ROOT}/s1280/fold_2/runs/foundation/weights/last.pt"
)
PRIMARY_FRONTIER="${CONFIRM_ROOT}/s1024/aggregate/crossfit_frontier.json"
HIGHRES_FRONTIER="${CONFIRM_ROOT}/s1280/aggregate/crossfit_frontier.json"

for path in "${PY}" "${PRIMARY_FRONTIER}" "${HIGHRES_FRONTIER}" \
  "${HARD_ROOT}/ground_truth.json" "${SENTINEL_ROOT}/ground_truth.json" \
  "${S1280[@]}"; do
  test -f "${path}"
done

for condition in hard sentinel; do
  if [[ "${condition}" = hard ]]; then root="${HARD_ROOT}"; else root="${SENTINEL_ROOT}"; fi
  primary="${R3_ROOT}/${condition}/primary/predictions.json"
  test -f "${primary}"
  mkdir -p "${OUT}/${condition}"
  printf '%s_s1280_i1280_inference\n' "${condition}" >"${STATUS}"
  "${PY}" scripts/run_multifamily_cv3_pseudo_eval.py \
    --pseudo-root "${root}" --family yolo --weights "${S1280[@]}" \
    --output-dir "${OUT}/${condition}/s1280_i1280" --score-floor 0.001 \
    --batch-size 4 --device cuda:0 --imgsz 1280 --tile-size 1024 --overlap 256 \
    >"${OUT}/${condition}/s1280_i1280_infer.log" 2>&1

  printf '%s_frozen_route_evaluation\n' "${condition}" >"${STATUS}"
  "${PY}" scripts/analyze_cv3_class_resolution_route.py \
    --gt "${root}/ground_truth.json" \
    --primary-pred "${primary}" \
    --highres-pred "${OUT}/${condition}/s1280_i1280/predictions.json" \
    --primary-frontier "${PRIMARY_FRONTIER}" \
    --highres-frontier "${HIGHRES_FRONTIER}" \
    --primary-labels 0-23 --highres-labels 24 --fdr-level 0.150 \
    --transfer-target-folds-from-gt \
    --output "${OUT}/${condition}/frozen_route.json" \
    >"${OUT}/${condition}/frozen_route.log" 2>&1
done

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
