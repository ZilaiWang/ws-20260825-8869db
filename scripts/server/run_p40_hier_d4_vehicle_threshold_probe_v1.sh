#!/usr/bin/env bash
set -Eeuo pipefail

# Targeted post-training workpoint probe for the already-trained hierarchical
# Vehicle expert.  P40 (fine 0-23), Aircraft-D4 and every runtime setting stay
# frozen; only the class-disjoint Vehicle threshold changes.  Three predefined
# thresholds run in parallel on three GPUs against the fixed Hard and Sentinel
# proxies.  This is a bounded probe, not an open-ended threshold sweep.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
SOURCE=${SOURCE:-/root/autodl-tmp/results/P40-HIER-S256V128-AIRCRAFT-D4-FULL-VALIDATION-V1}
OUT=${OUT:-/root/autodl-tmp/results/P40-HIER-S256V128-AIRCRAFT-D4-VEHICLE-THRESHOLD-PROBE-V1}
BASE_CONFIG=${BASE_CONFIG:-${PROJECT}/configs/experiments/p40_aircraft_d4_full_runtime_candidate_v1.json}
HARD_ROOT=${HARD_ROOT:-/root/autodl-tmp/pseudo10k-trial-mix-local}
SENTINEL_ROOT=${SENTINEL_ROOT:-/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1}
STATUS=${OUT}/status.txt
THRESHOLD_VALUES=${THRESHOLD_VALUES:-"0.480 0.510 0.535"}
read -r -a THRESHOLDS <<<"${THRESHOLD_VALUES}"
if [[ "${#THRESHOLDS[@]}" -ne 3 ]]; then
  printf 'exactly three thresholds are required for the three-GPU probe\n' >&2
  exit 2
fi

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

expert="${SOURCE}/hier_full_sanitized.pt"
for path in "${expert}" "${BASE_CONFIG}" "${HARD_ROOT}/ground_truth.json" \
  "${SENTINEL_ROOT}/ground_truth.json" \
  "${SOURCE}/hard/baseline/metrics.json" \
  "${SOURCE}/sentinel/baseline/metrics.json"; do
  test -f "${path}"
done

printf 'building_three_frozen_workpoints\n' >"${STATUS}"
for threshold in "${THRESHOLDS[@]}"; do
  tag=${threshold/./p}
  mkdir -p "${OUT}/${tag}"
  "${PY}" scripts/build_p40_hier_d4_runtime_config.py \
    --base-config "${BASE_CONFIG}" --expert-weight "${expert}" \
    --expert-threshold "${threshold}" \
    --workpoint-id "p40_hier_s256v128_t${tag}_aircraft_d4_full_20260904" \
    --output "${OUT}/${tag}/runtime_config.json" \
    >"${OUT}/${tag}/runtime_config_build.json"
done

run_one() {
  local threshold=$1 condition=$2 device=$3 root=$4
  local tag=${threshold/./p}
  local destination="${OUT}/${tag}/${condition}"
  mkdir -p "${destination}"
  CUDA_VISIBLE_DEVICES="${device}" "${PY}" -u scripts/run_competition_runtime_coco.py \
    --config "${OUT}/${tag}/runtime_config.json" --pseudo-root "${root}" \
    --device cuda:0 --predictions "${destination}/predictions.json" \
    --summary "${destination}/runtime_summary.json" \
    >"${OUT}/logs/${tag}_${condition}.log" 2>&1
  "${PY}" scripts/evaluate_fixed_score_threshold.py \
    --gt "${root}/ground_truth.json" --pred "${destination}/predictions.json" \
    --threshold 0 --output "${destination}/metrics.json"
  "${PY}" scripts/compare_candidate_trend.py \
    --baseline "${SOURCE}/${condition}/baseline/metrics.json" \
    --candidate "${destination}/metrics.json" \
    --output "${destination}/paired_comparison.json"
}

printf 'hard_exact_runtime_three_way\n' >"${STATUS}"
for index in 0 1 2; do
  run_one "${THRESHOLDS[$index]}" hard "${index}" "${HARD_ROOT}" & pids[$index]=$!
done
wait "${pids[@]}"

printf 'sentinel_exact_runtime_three_way\n' >"${STATUS}"
for index in 0 1 2; do
  run_one "${THRESHOLDS[$index]}" sentinel "${index}" "${SENTINEL_ROOT}" & pids[$index]=$!
done
wait "${pids[@]}"

printf 'summarizing\n' >"${STATUS}"
"${PY}" - "${OUT}" "${SOURCE}" "${THRESHOLDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
source = Path(sys.argv[2])
thresholds = [float(value) for value in sys.argv[3:]]
rows = []
for threshold in thresholds:
    tag = f"{threshold:.3f}".replace(".", "p")
    conditions = {}
    for condition in ("hard", "sentinel"):
        comparison = json.loads(
            (out / tag / condition / "paired_comparison.json").read_text()
        )
        vehicle = comparison["candidate"]["per_coarse"]["vehicle"]
        runtime = json.loads(
            (out / tag / condition / "runtime_summary.json").read_text()
        )
        conditions[condition] = {
            "delta_quality_contribution": comparison["delta_quality_contribution"],
            "vehicle_recall": vehicle["recall"],
            "vehicle_fdr": vehicle["fdr"],
            "gate_recall": comparison["candidate"]["gate_recall"],
            "gate_fdr": comparison["candidate"]["gate_fdr"],
            "mean_image_seconds": runtime["mean_image_seconds"],
            "predictions_sha256": runtime["predictions_sha256"],
        }
    deltas = [conditions[name]["delta_quality_contribution"] for name in conditions]
    rows.append(
        {
            "expert_threshold": threshold,
            "conditions": conditions,
            "mean_delta_quality": sum(deltas) / len(deltas),
            "worst_delta_quality": min(deltas),
            "both_directions_positive": all(value > 0 for value in deltas),
        }
    )

reference = {}
for condition in ("hard", "sentinel"):
    comparison = json.loads(
        (source / condition / "paired_comparison.json").read_text()
    )
    vehicle = comparison["candidate"]["per_coarse"]["vehicle"]
    reference[condition] = {
        "delta_quality_contribution": comparison["delta_quality_contribution"],
        "vehicle_recall": vehicle["recall"],
        "vehicle_fdr": vehicle["fdr"],
    }
payload = {
    "status": "complete",
    "role": "bounded_vehicle_threshold_probe_not_hidden_score_prediction",
    "frozen_components": ["P40", "Aircraft-D4", "hierarchy checkpoint", "runtime"],
    "reference_threshold_0p546": reference,
    "candidates": rows,
    "selection_rule": "maximize worst_delta_quality; require both directions positive",
}
(out / "threshold_probe_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
)
PY

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
