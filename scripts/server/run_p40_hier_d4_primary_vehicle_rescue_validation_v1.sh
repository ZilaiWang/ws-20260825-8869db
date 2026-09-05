#!/usr/bin/env bash
set -Eeuo pipefail

# Exact Docker-path validation for the bounded, already-selected P40 Vehicle
# fallback.  No training, new threshold search, Docker build, or submission.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
SOURCE=${SOURCE:-/root/autodl-tmp/results/P40-HIER-S256V128-AIRCRAFT-D4-FULL-VALIDATION-V1}
BASE_CONFIG=${BASE_CONFIG:-${PROJECT}/configs/experiments/p40_aircraft_d4_full_runtime_candidate_v1.json}
BASELINE_ROOT=${BASELINE_ROOT:-/root/autodl-tmp/results/P40-HIER-S256V128-T042-AIRCRAFT-D4-3090-V1}
OUT=${OUT:-/root/autodl-tmp/results/P40-HIER-S256V128-T042-P40V060-I070-D4-VALIDATION-V1}
HARD_ROOT=${HARD_ROOT:-/root/autodl-tmp/pseudo10k-trial-mix-local}
SENTINEL_ROOT=${SENTINEL_ROOT:-/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1}
DEVICE=${DEVICE:-cuda:0}
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

for path in "${SOURCE}/hier_full_sanitized.pt" "${BASE_CONFIG}" \
  "${HARD_ROOT}/ground_truth.json" "${SENTINEL_ROOT}/ground_truth.json" \
  "${BASELINE_ROOT}/hard/metrics.json" "${BASELINE_ROOT}/sentinel/metrics.json"; do
  test -f "${path}"
done

printf 'building_frozen_rescue_runtime\n' >"${STATUS}"
"${PY}" scripts/build_p40_hier_d4_runtime_config.py \
  --base-config "${BASE_CONFIG}" \
  --expert-weight "${SOURCE}/hier_full_sanitized.pt" \
  --expert-threshold 0.420 \
  --primary-rescue-threshold 0.600 \
  --primary-rescue-dedup-iou 0.700 \
  --workpoint-id p40_hier_s256v128_t042_p40v060_i070_aircraft_d4_full_20260904 \
  --output "${OUT}/runtime_config.json" \
  >"${OUT}/runtime_config_build.json"

for condition in hard sentinel; do
  if [[ "${condition}" == hard ]]; then
    root=${HARD_ROOT}
  else
    root=${SENTINEL_ROOT}
  fi
  printf 'running_%s_exact_runtime\n' "${condition}" >"${STATUS}"
  mkdir -p "${OUT}/${condition}"
  "${PY}" -u scripts/run_competition_runtime_coco.py \
    --config "${OUT}/runtime_config.json" --pseudo-root "${root}" \
    --device "${DEVICE}" --predictions "${OUT}/${condition}/predictions.json" \
    --summary "${OUT}/${condition}/runtime_summary.json" \
    >"${OUT}/logs/${condition}.log" 2>&1
  "${PY}" scripts/evaluate_fixed_score_threshold.py \
    --gt "${root}/ground_truth.json" --pred "${OUT}/${condition}/predictions.json" \
    --threshold 0 --output "${OUT}/${condition}/metrics.json"
  "${PY}" scripts/compare_candidate_trend.py \
    --baseline "${BASELINE_ROOT}/${condition}/metrics.json" \
    --candidate "${OUT}/${condition}/metrics.json" \
    --output "${OUT}/${condition}/paired_comparison.json"
done

"${PY}" - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
splits = {}
for split in ("hard", "sentinel"):
    comparison = json.loads((root / split / "paired_comparison.json").read_text())
    metrics = json.loads((root / split / "metrics.json").read_text())
    runtime = json.loads((root / split / "runtime_summary.json").read_text())
    splits[split] = {
        "quality_delta": comparison["delta_quality_contribution"],
        "gate_recall": metrics["platform_gate_recall"],
        "gate_fdr": metrics["platform_gate_fdr"],
        "vehicle": metrics["platform"]["per_coarse"]["vehicle"],
        "mean_image_seconds": runtime["mean_image_seconds"],
        "p95_image_seconds": runtime["p95_image_seconds"],
        "predictions_sha256": runtime["predictions_sha256"],
    }
payload = {
    "status": "complete",
    "role": "exact_runtime_fixed_proxy_confirmation_not_hidden_score_prediction",
    "frozen_workpoint": {
        "primary_threshold": 0.536,
        "expert_threshold": 0.420,
        "primary_vehicle_rescue_threshold": 0.600,
        "primary_vehicle_rescue_dedup_iou": 0.700,
        "aircraft_d4_probability": 0.900,
    },
    "splits": splits,
    "both_quality_directions_positive": all(
        row["quality_delta"] > 0.0 for row in splits.values()
    ),
}
(root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
