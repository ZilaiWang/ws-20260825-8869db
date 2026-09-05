#!/usr/bin/env bash
# Exact-output engineering check only: admitted batch=64 versus batch=128.
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
SOURCE_CONFIG=${SOURCE_CONFIG:-${PROJECT}/configs/experiments/p40_aircraft_d4_full_runtime_candidate_v1.json}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-ATTEMPT4-AIRCRAFT-D4-BATCH128-3090-V1}
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
test "$("${PY}" -c 'import torch; print(torch.cuda.device_count())')" -eq 1

"${PY}" - "${SOURCE_CONFIG}" "${OUT}" <<'PY'
import json, sys
from pathlib import Path
source, output = Path(sys.argv[1]), Path(sys.argv[2])
for batch in (64, 128):
    config = json.loads(source.read_text())
    config['aircraft_classifier_model']['batch_objects'] = batch
    config['workpoint_id'] += f'_paired_batch{batch}'
    (output / f'batch{batch}_config.json').write_text(json.dumps(config, indent=2) + '\n')
PY

for condition in hard sentinel; do
  if [[ "${condition}" = hard ]]; then
    ROOT=/root/autodl-tmp/pseudo10k-trial-mix-local
  else
    ROOT=/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1
  fi
  mkdir -p "${OUT}/${condition}"
  for batch in 64 128; do
    printf '%s_batch%s\n' "${condition}" "${batch}" >"${STATUS}"
    "${PY}" -u scripts/run_competition_runtime_coco.py \
      --config "${OUT}/batch${batch}_config.json" --pseudo-root "${ROOT}" \
      --predictions "${OUT}/${condition}/batch${batch}_predictions.json" \
      --summary "${OUT}/${condition}/batch${batch}_runtime.json" \
      >"${OUT}/${condition}/batch${batch}.log" 2>&1
  done
  printf '%s_compare\n' "${condition}" >"${STATUS}"
  "${PY}" scripts/compare_d4_batch_runtime.py \
    --reference-predictions "${OUT}/${condition}/batch64_predictions.json" \
    --candidate-predictions "${OUT}/${condition}/batch128_predictions.json" \
    --reference-summary "${OUT}/${condition}/batch64_runtime.json" \
    --candidate-summary "${OUT}/${condition}/batch128_runtime.json" \
    --reference-batch 64 --candidate-batch 128 \
    --output "${OUT}/${condition}/audit.json" \
    >"${OUT}/${condition}/audit.log" 2>&1
done

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
