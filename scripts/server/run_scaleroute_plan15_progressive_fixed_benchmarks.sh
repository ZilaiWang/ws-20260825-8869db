#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625-capscale}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
PROGRESSIVE=${PROGRESSIVE:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE-CV3-V1}
HARD=${HARD:-/root/autodl-tmp/pseudo10k-trial-mix-local}
SENTINEL=${SENTINEL:-/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1}
OUT=${OUT:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE-FIXED-BENCHMARKS-V1}
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

WEIGHTS=(
  "${PROGRESSIVE}/fold_0/adaptation/runs/resolution_adaptation/weights/last.pt"
  "${PROGRESSIVE}/fold_1/adaptation/runs/resolution_adaptation/weights/last.pt"
  "${PROGRESSIVE}/fold_2/adaptation/runs/resolution_adaptation/weights/last.pt"
)
FRONTIER="${PROGRESSIVE}/aggregate/crossfit_frontier.json"
for path in "${PY}" "${FRONTIER}" "${WEIGHTS[@]}" \
  "${HARD}/ground_truth.json" "${SENTINEL}/ground_truth.json"; do
  test -f "${path}"
done

for condition in hard sentinel; do
  if [[ "${condition}" = hard ]]; then root="${HARD}"; else root="${SENTINEL}"; fi
  condition_out="${OUT}/${condition}"
  mkdir -p "${condition_out}"

  printf '%s_progressive_i1280_inference\n' "${condition}" >"${STATUS}"
  "${PY}" scripts/run_multifamily_cv3_pseudo_eval.py \
    --pseudo-root "${root}" --family yolo --weights "${WEIGHTS[@]}" \
    --output-dir "${condition_out}/progressive_i1280" --score-floor 0.001 \
    --batch-size 4 --device cuda:0 --imgsz 1280 --tile-size 1024 --overlap 256 \
    >"${condition_out}/infer.log" 2>&1

  printf '%s_frozen_threshold_evaluation\n' "${condition}" >"${STATUS}"
  "${PY}" scripts/evaluate_pseudo_with_frozen_thresholds.py \
    --gt "${root}/ground_truth.json" \
    --pred "${condition_out}/progressive_i1280/predictions.json" \
    --source-frontier "${FRONTIER}" --selection-mode fdr_level --fdr-level 0.150 \
    --output "${condition_out}/frozen_thresholds.json" \
    >"${condition_out}/frozen_thresholds.log" 2>&1
done

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
