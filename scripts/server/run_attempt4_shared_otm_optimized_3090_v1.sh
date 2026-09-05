#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-sprint20-ab51106}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
REFERENCE_CONFIG=${REFERENCE_CONFIG:-${PROJECT}/configs/experiments/hera_sprint20_p40_d4_otm_ship23_t0560_candidate_v2.json}
CANDIDATE_CONFIG=${CANDIDATE_CONFIG:-${PROJECT}/configs/experiments/hera_sprint20_p40_d4_otm_ship23_t0560_optimized_candidate_v3.json}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-ATTEMPT4-SHARED-OTM-OPTIMIZED-3090-V1}
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

"${PY}" - "${REFERENCE_CONFIG}" "${CANDIDATE_CONFIG}" "${OUT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

reference, candidate, output = map(Path, sys.argv[1:])
for path in (reference, candidate):
    if not path.is_file():
        raise FileNotFoundError(path)

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

reference_payload = json.loads(reference.read_text())
candidate_payload = json.loads(candidate.read_text())
expected_sprint20 = dict(reference_payload["sprint20"])
expected_sprint20["optimized_pipeline"] = True
if candidate_payload["sprint20"] != expected_sprint20:
    raise RuntimeError("candidate changes more than the optimized pipeline flag")
for key in (
    "contract_version", "metric_protocol", "device", "model", "pipeline",
    "post_fusion_score_threshold", "aircraft_classifier_model",
):
    if candidate_payload[key] != reference_payload[key]:
        raise RuntimeError(f"scientific contract changed at {key}")

payload = {
    "status": "preflight_pass",
    "protocol": "attempt4_shared_otm_equivalent_optimization_3090_v1",
    "reference_config": str(reference),
    "reference_config_sha256": digest(reference),
    "candidate_config": str(candidate),
    "candidate_config_sha256": digest(candidate),
    "scientific_contract_exact_except_optimized_pipeline": True,
}
(output / "preflight.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

run_one() {
  local condition=$1
  local role=$2
  local root=$3
  local config=$4
  printf '%s_%s\n' "${condition}" "${role}" >"${STATUS}"
  "${PY}" -u scripts/run_competition_runtime_coco.py \
    --config "${config}" \
    --runtime-factory sprint20 \
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
  # Reverse order on Sentinel to expose warm-cache/order-dependent timing.
  if [[ "${condition}" = hard ]]; then
    run_one "${condition}" reference "${ROOT}" "${REFERENCE_CONFIG}"
    run_one "${condition}" candidate "${ROOT}" "${CANDIDATE_CONFIG}"
  else
    run_one "${condition}" candidate "${ROOT}" "${CANDIDATE_CONFIG}"
    run_one "${condition}" reference "${ROOT}" "${REFERENCE_CONFIG}"
  fi
  printf '%s_compare\n' "${condition}" >"${STATUS}"
  "${PY}" scripts/compare_equivalent_runtime.py \
    --reference-predictions "${OUT}/${condition}/reference_predictions.json" \
    --candidate-predictions "${OUT}/${condition}/candidate_predictions.json" \
    --reference-summary "${OUT}/${condition}/reference_runtime.json" \
    --candidate-summary "${OUT}/${condition}/candidate_runtime.json" \
    --protocol "attempt4_shared_otm_${condition}_equivalent_optimization_3090_v1" \
    --output "${OUT}/${condition}/audit.json" \
    >"${OUT}/${condition}/audit.log" 2>&1
done

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
