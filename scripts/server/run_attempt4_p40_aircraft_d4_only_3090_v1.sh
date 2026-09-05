#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
SOURCE_CONFIG=${SOURCE_CONFIG:-${PROJECT}/configs/experiments/p40_aircraft_d4_full_runtime_candidate_v1.json}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-ATTEMPT4-P40-AIRCRAFT-D4-ONLY-3090-V1}
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
candidate = json.loads(source.read_text())
baseline = json.loads(source.read_text())
baseline.pop('aircraft_classifier_model')
baseline['deployment_role'] = 'attempt4_p40_identity_runtime_control'
baseline['workpoint_id'] = 'attempt4_p40_identity_control'
(output/'candidate_config.json').write_text(json.dumps(candidate,indent=2)+'\n')
(output/'baseline_config.json').write_text(json.dumps(baseline,indent=2)+'\n')
PY

for condition in hard sentinel; do
  if [[ "${condition}" = hard ]]; then
    ROOT=/root/autodl-tmp/pseudo10k-trial-mix-local
    REFERENCE=/root/autodl-tmp/results/P40-HIER-S256V128-T042-AIRCRAFT-D4-3090-V1/hard/predictions.json
  else
    ROOT=/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1
    REFERENCE=/root/autodl-tmp/results/P40-HIER-S256V128-T042-AIRCRAFT-D4-3090-V1/sentinel/predictions.json
  fi
  mkdir -p "${OUT}/${condition}"
  printf '%s_baseline\n' "${condition}" >"${STATUS}"
  "${PY}" -u scripts/run_competition_runtime_coco.py \
    --config "${OUT}/baseline_config.json" --pseudo-root "${ROOT}" \
    --predictions "${OUT}/${condition}/baseline_predictions.json" \
    --summary "${OUT}/${condition}/baseline_runtime.json" \
    >"${OUT}/${condition}/baseline.log" 2>&1
  printf '%s_candidate_d4\n' "${condition}" >"${STATUS}"
  "${PY}" -u scripts/run_competition_runtime_coco.py \
    --config "${OUT}/candidate_config.json" --pseudo-root "${ROOT}" \
    --predictions "${OUT}/${condition}/candidate_predictions.json" \
    --summary "${OUT}/${condition}/candidate_runtime.json" \
    >"${OUT}/${condition}/candidate.log" 2>&1
  printf '%s_compare\n' "${condition}" >"${STATUS}"
  "${PY}" scripts/compare_d4_only_runtime.py \
    --gt "${ROOT}/ground_truth.json" \
    --baseline-pred "${OUT}/${condition}/baseline_predictions.json" \
    --candidate-pred "${OUT}/${condition}/candidate_predictions.json" \
    --reference-v3-pred "${REFERENCE}" \
    --baseline-summary "${OUT}/${condition}/baseline_runtime.json" \
    --candidate-summary "${OUT}/${condition}/candidate_runtime.json" \
    --output "${OUT}/${condition}/audit.json" \
    >"${OUT}/${condition}/audit.log" 2>&1
done

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
