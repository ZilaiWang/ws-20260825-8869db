#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-sprint20-ab51106}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
BASE=${BASE:-/root/autodl-tmp/results/HERA-GUARD-ATTEMPT4-SHARED-OTM-RUNTIME-3090-V1}
OPT=${OPT:-/root/autodl-tmp/results/HERA-GUARD-ATTEMPT4-SHARED-OTM-OPTIMIZED-3090-V1}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-ATTEMPT4-SHARED-OTM-OPTIMIZED-QUALITY-3090-V1}

if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
test "$(cat "${BASE}/status.txt")" = complete
test "$(cat "${OPT}/status.txt")" = complete
mkdir -p "${OUT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for condition in hard sentinel; do
  if [[ "${condition}" = hard ]]; then
    ROOT=/root/autodl-tmp/pseudo10k-trial-mix-local
  else
    ROOT=/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1
  fi
  mkdir -p "${OUT}/${condition}"
  "${PY}" scripts/compare_sprint20_runtime.py \
    --gt "${ROOT}/ground_truth.json" \
    --baseline-pred "${BASE}/${condition}/baseline_predictions.json" \
    --candidate-pred "${OPT}/${condition}/candidate_predictions.json" \
    --baseline-summary "${BASE}/${condition}/baseline_runtime.json" \
    --candidate-summary "${OPT}/${condition}/candidate_runtime.json" \
    --alternative-labels 2 3 \
    --output "${OUT}/${condition}/audit.json" \
    >"${OUT}/${condition}/audit.log" 2>&1
done

printf 'complete\n' >"${OUT}/status.txt"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
