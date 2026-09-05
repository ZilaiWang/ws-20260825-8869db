#!/usr/bin/env bash
set -Eeuo pipefail

# RTX 3090 parity/latency validation for the already-selected threshold 0.420
# composition.  No training, threshold selection, Docker build or submission.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
CONFIG=${CONFIG:-/root/autodl-tmp/results/P40-HIER-S256V128-T042-AIRCRAFT-D4-CANDIDATE-V1/runtime_config.json}
OUT=${OUT:-/root/autodl-tmp/results/P40-HIER-S256V128-T042-AIRCRAFT-D4-3090-V1}
STATUS=${OUT}/status.txt

if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}/logs"
exec 9>"${OUT}.lock"
flock -n 9 || exit 4
failed() { code=$?; printf 'failed_exit_%s\n' "${code}" >"${STATUS}"; exit "${code}"; }
trap failed ERR INT TERM
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"
test -f "${CONFIG}"

for condition in hard sentinel; do
  if [[ "${condition}" == hard ]]; then
    root=/root/autodl-tmp/pseudo10k-trial-mix-local
  else
    root=/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1
  fi
  test -f "${root}/ground_truth.json"
  printf 'running_%s_exact_runtime\n' "${condition}" >"${STATUS}"
  mkdir -p "${OUT}/${condition}"
  "${PY}" -u scripts/run_competition_runtime_coco.py \
    --config "${CONFIG}" --pseudo-root "${root}" --device cuda:0 \
    --predictions "${OUT}/${condition}/predictions.json" \
    --summary "${OUT}/${condition}/runtime_summary.json" \
    >"${OUT}/logs/${condition}.log" 2>&1
  "${PY}" scripts/evaluate_fixed_score_threshold.py \
    --gt "${root}/ground_truth.json" --pred "${OUT}/${condition}/predictions.json" \
    --threshold 0 --output "${OUT}/${condition}/metrics.json"
done

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
