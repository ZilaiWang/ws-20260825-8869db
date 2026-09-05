#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625-capscale}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
FOLD0=${FOLD0:-/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1}
CONFIRM=${CONFIRM:-/root/autodl-tmp/results/YOLO-SCALE-CV3-CONFIRM-V1}
R2=${R2:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-R2-FIXED-BENCHMARKS-V1}
HARD=${HARD:-/root/autodl-tmp/pseudo10k-trial-mix-local}
SENTINEL=${SENTINEL:-/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1}
OUT=${OUT:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-R3-TILE-GEOMETRY-V1}
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

S1024=(
  "${FOLD0}/s1024/runs/foundation/weights/last.pt"
  "${CONFIRM}/s1024/fold_1/runs/foundation/weights/last.pt"
  "${CONFIRM}/s1024/fold_2/runs/foundation/weights/last.pt"
)
PRIMARY_FRONTIER="${CONFIRM}/s1024/aggregate/crossfit_frontier.json"
EXPERT_FRONTIER="${CONFIRM}/s1280/aggregate/crossfit_frontier.json"
for path in "${PY}" "${PRIMARY_FRONTIER}" "${EXPERT_FRONTIER}" "${S1024[@]}" \
  "${HARD}/ground_truth.json" "${SENTINEL}/ground_truth.json" \
  "${R2}/hard/s1280_i1280/predictions.json" \
  "${R2}/sentinel/s1280_i1280/predictions.json"; do test -f "${path}"; done

for condition in hard sentinel; do
  if [[ "${condition}" = hard ]]; then root="${HARD}"; else root="${SENTINEL}"; fi
  condition_out="${OUT}/${condition}"
  mkdir -p "${condition_out}"
  printf '%s_primary_tile1280_i1024\n' "${condition}" >"${STATUS}"
  "${PY}" scripts/run_multifamily_cv3_pseudo_eval.py \
    --pseudo-root "${root}" --family yolo --weights "${S1024[@]}" \
    --output-dir "${condition_out}/primary_tile1280_i1024" --score-floor 0.001 \
    --batch-size 4 --device cuda:0 --imgsz 1024 --tile-size 1280 --overlap 256 \
    >"${condition_out}/primary_infer.log" 2>&1

  printf '%s_frozen_route\n' "${condition}" >"${STATUS}"
  "${PY}" scripts/analyze_cv3_class_resolution_route.py \
    --gt "${root}/ground_truth.json" \
    --primary-pred "${condition_out}/primary_tile1280_i1024/predictions.json" \
    --highres-pred "${R2}/${condition}/s1280_i1280/predictions.json" \
    --primary-frontier "${PRIMARY_FRONTIER}" --highres-frontier "${EXPERT_FRONTIER}" \
    --primary-labels 0-23 --highres-labels 24 --fdr-level 0.150 \
    --transfer-target-folds-from-gt --output "${condition_out}/frozen_route.json" \
    >"${condition_out}/frozen_route.log" 2>&1
done

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
