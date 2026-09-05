#!/usr/bin/env bash
set -Eeuo pipefail

CONDITION=${1:?usage: $0 hard|sentinel}
[[ "${CONDITION}" == hard || "${CONDITION}" == sentinel ]] || exit 2
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
TRAIN=${TRAIN:-/root/autodl-tmp/results/R1-5-AIRCRAFT-VIEW-CONSISTENCY-FULL-V1}
OUT_ROOT=${OUT_ROOT:-/root/autodl-tmp/results/P40-V96-AIRCRAFT-D4-FULL-DIAGNOSTIC-V1}
OUT=${OUT_ROOT}/${CONDITION}
P40=${P40:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1}
BASE=${BASE:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-FIXED-BENCHMARKS-V1}
CE=${CE:-/root/autodl-tmp/results/P40-AIRCRAFT-CE-D4-V1}

if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT_ROOT}/logs"
exec 9>"${OUT}.lock"
flock -n 9 || exit 4
failed() { code=$?; printf 'failed_exit_%s\n' "${code}" >"${OUT_ROOT}/${CONDITION}.status"; exit "${code}"; }
trap failed ERR INT TERM

printf 'waiting_for_full_classifier\n' >"${OUT_ROOT}/${CONDITION}.status"
while [[ ! -f "${TRAIN}/training/final_checkpoint.pt" ]]; do
  current=$(cat "${TRAIN}/status.txt" 2>/dev/null || true)
  [[ "${current}" != failed_* ]] || exit 5
  sleep 10
done

if [[ "${CONDITION}" == hard ]]; then
  ROOT=/root/autodl-tmp/pseudo10k-trial-mix-local
  REF="${CE}/comparison.json"
else
  ROOT=/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1
  REF="${CE}/sentinel/comparison.json"
fi
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"
printf 'running_full_d4_%s\n' "${CONDITION}" >"${OUT_ROOT}/${CONDITION}.status"
"${PY}" -u scripts/run_p40_aircraft_ce_d4.py --condition "${CONDITION}" \
  --config configs/experiments/p40_aircraft_view_consistency_v1.json \
  --reference-comparison "${REF}" --pseudo-root "${ROOT}" \
  --pred "${BASE}/${CONDITION}/progressive_i1280/predictions.json" \
  --frontier "${P40}/aggregate/crossfit_frontier.json" \
  --full-classifier "${TRAIN}/training/final_checkpoint.pt" \
  --imagenet /root/autodl-tmp/pretrained/convnext_tiny-983f1562.pth \
  --output "${OUT}" >"${OUT_ROOT}/logs/${CONDITION}.log" 2>&1
trap - ERR INT TERM
printf 'complete_diagnostic_only\n' >"${OUT_ROOT}/${CONDITION}.status"
