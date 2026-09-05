#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
SOURCE_CONFIG=${SOURCE_CONFIG:-${PROJECT}/configs/experiments/p40_aircraft_d4_full_runtime_candidate_v1.json}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-ATTEMPT4-AIRCRAFT-D4-TENSORIZED-3090-V1}
OPTIMIZED_CHANNELS_LAST=${OPTIMIZED_CHANNELS_LAST:-true}
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

"${PY}" - "${SOURCE_CONFIG}" "${OUT}" "${OPTIMIZED_CHANNELS_LAST}" <<'PY'
import json, sys
from pathlib import Path
source, output = Path(sys.argv[1]), Path(sys.argv[2])
channels_last_raw = sys.argv[3].lower()
if channels_last_raw not in {'true', 'false'}:
    raise ValueError('OPTIMIZED_CHANNELS_LAST must be true or false')
channels_last = channels_last_raw == 'true'
reference = json.loads(source.read_text())
optimized = json.loads(source.read_text())
for config in (reference, optimized):
    config['aircraft_classifier_model']['batch_objects'] = 64
reference['aircraft_classifier_model']['channels_last'] = False
reference['aircraft_classifier_model']['tensorized_views'] = False
optimized['aircraft_classifier_model']['channels_last'] = channels_last
optimized['aircraft_classifier_model']['tensorized_views'] = True
reference['workpoint_id'] += '_legacy_views_b64'
optimized['workpoint_id'] += '_tensorized_views_b64'
if channels_last:
    optimized['workpoint_id'] += '_channels_last'
(output/'reference_config.json').write_text(json.dumps(reference, indent=2)+'\n')
(output/'optimized_config.json').write_text(json.dumps(optimized, indent=2)+'\n')
PY

for condition in hard sentinel; do
  if [[ "${condition}" = hard ]]; then
    ROOT=/root/autodl-tmp/pseudo10k-trial-mix-local
  else
    ROOT=/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1
  fi
  mkdir -p "${OUT}/${condition}"
  for role in reference optimized; do
    printf '%s_%s\n' "${condition}" "${role}" >"${STATUS}"
    "${PY}" -u scripts/run_competition_runtime_coco.py \
      --config "${OUT}/${role}_config.json" --pseudo-root "${ROOT}" \
      --predictions "${OUT}/${condition}/${role}_predictions.json" \
      --summary "${OUT}/${condition}/${role}_runtime.json" \
      >"${OUT}/${condition}/${role}.log" 2>&1
  done
  printf '%s_compare\n' "${condition}" >"${STATUS}"
  "${PY}" scripts/compare_d4_optimized_runtime.py \
    --reference-predictions "${OUT}/${condition}/reference_predictions.json" \
    --optimized-predictions "${OUT}/${condition}/optimized_predictions.json" \
    --reference-summary "${OUT}/${condition}/reference_runtime.json" \
    --optimized-summary "${OUT}/${condition}/optimized_runtime.json" \
    --optimization-label "tensorized_views=true,channels_last=${OPTIMIZED_CHANNELS_LAST}" \
    --output "${OUT}/${condition}/audit.json" \
    >"${OUT}/${condition}/audit.log" 2>&1
done

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
