#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-sprint20-ab51106}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
BASE_CONFIG=${BASE_CONFIG:-${PROJECT}/configs/experiments/p40_aircraft_d4_full_runtime_candidate_v1.json}
CANDIDATE_CONFIG=${CANDIDATE_CONFIG:-${PROJECT}/configs/experiments/hera_sprint20_p40_d4_otm_ship23_t0560_candidate_v2.json}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-ATTEMPT4-SHARED-OTM-RUNTIME-3090-V1}
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

"${PY}" - "${BASE_CONFIG}" "${CANDIDATE_CONFIG}" "${OUT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

baseline, candidate, output = map(Path, sys.argv[1:])
for path in (baseline, candidate):
    if not path.is_file():
        raise FileNotFoundError(path)

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

payload = {
    "status": "preflight_pass",
    "protocol": "attempt4_shared_otm_paired_runtime_3090_v1",
    "baseline_config": str(baseline),
    "baseline_config_sha256": digest(baseline),
    "candidate_config": str(candidate),
    "candidate_config_sha256": digest(candidate),
}
(output / "preflight.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

run_one() {
  local condition=$1
  local role=$2
  local root=$3
  local config=$4
  local factory=$5
  printf '%s_%s\n' "${condition}" "${role}" >"${STATUS}"
  "${PY}" -u scripts/run_competition_runtime_coco.py \
    --config "${config}" \
    --runtime-factory "${factory}" \
    --pseudo-root "${root}" \
    --predictions "${OUT}/${condition}/${role}_predictions.json" \
    --summary "${OUT}/${condition}/${role}_runtime.json" \
    >"${OUT}/${condition}/${role}.log" 2>&1
}

for condition in hard sentinel; do
  if [[ "${condition}" = hard ]]; then
    ROOT=/root/autodl-tmp/pseudo10k-trial-mix-local
  else
    ROOT=/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1
  fi
  mkdir -p "${OUT}/${condition}"
  # Reverse the execution order on the second condition to reduce one-sided
  # warm-cache bias. Quality comparisons remain paired within each condition.
  if [[ "${condition}" = hard ]]; then
    run_one "${condition}" baseline "${ROOT}" "${BASE_CONFIG}" competition
    run_one "${condition}" candidate "${ROOT}" "${CANDIDATE_CONFIG}" sprint20
  else
    run_one "${condition}" candidate "${ROOT}" "${CANDIDATE_CONFIG}" sprint20
    run_one "${condition}" baseline "${ROOT}" "${BASE_CONFIG}" competition
  fi
  printf '%s_compare\n' "${condition}" >"${STATUS}"
  "${PY}" scripts/compare_sprint20_runtime.py \
    --gt "${ROOT}/ground_truth.json" \
    --baseline-pred "${OUT}/${condition}/baseline_predictions.json" \
    --candidate-pred "${OUT}/${condition}/candidate_predictions.json" \
    --baseline-summary "${OUT}/${condition}/baseline_runtime.json" \
    --candidate-summary "${OUT}/${condition}/candidate_runtime.json" \
    --alternative-labels 2 3 \
    --output "${OUT}/${condition}/audit.json" \
    >"${OUT}/${condition}/audit.log" 2>&1
done

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
