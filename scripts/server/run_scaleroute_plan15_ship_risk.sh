#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625-capscale}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
CONFIRM=${CONFIRM:-/root/autodl-tmp/results/YOLO-SCALE-CV3-CONFIRM-V1}
GROUP_MAP=${GROUP_MAP:-/root/autodl-tmp/scaleroute-plan15-assets/macroshift_group_map.csv}
OUT=${OUT:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-SHIP-MACRORISK-V1}
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

GT="${CONFIRM}/s1024/aggregate/ground_truth.json"
PRIMARY="${CONFIRM}/s1024/aggregate/predictions_low.json"
EXPERT="${CONFIRM}/s1280/aggregate/predictions_low.json"
PRIMARY_FRONTIER="${CONFIRM}/s1024/aggregate/crossfit_frontier.json"
EXPERT_FRONTIER="${CONFIRM}/s1280/aggregate/crossfit_frontier.json"
for path in "${PY}" "${GT}" "${PRIMARY}" "${EXPERT}" \
  "${PRIMARY_FRONTIER}" "${EXPERT_FRONTIER}" "${GROUP_MAP}"; do test -f "${path}"; done

printf 'fit_and_bootstrap\n' >"${STATUS}"
"${PY}" scripts/analyze_scaleroute_ship_macro_risk_cv3.py \
  --gt "${GT}" --primary-pred "${PRIMARY}" --expert-pred "${EXPERT}" \
  --primary-frontier "${PRIMARY_FRONTIER}" --expert-frontier "${EXPERT_FRONTIER}" \
  --group-map "${GROUP_MAP}" --fdr-level 0.150 --target-fdr-ship 0.15 \
  --latency 8.0 --bootstrap-iterations 1000 --output "${OUT}/ship_macro_risk.json" \
  >"${OUT}/run.log" 2>&1

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
