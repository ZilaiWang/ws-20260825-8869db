#!/usr/bin/env bash
set -Eeuo pipefail

# Resume only the deterministic aggregate tail after the original driver
# stopped because cv3_merge_audit contains prediction-dependent counts in the
# two otherwise identical GT files. No inference, threshold, or model changes.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-MPSR-HIER-VEHICLE-ROUTE-CV3-OOF-3090-V1}
STATUS=${OUT}/status.txt
THRESHOLD=${THRESHOLD:-0.546}

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"
grep -q '^failed_exit_1$' "${STATUS}"
for path in \
  "${OUT}/aggregate/ground_truth.json" \
  "${OUT}/aggregate/candidate_ground_truth.json" \
  "${OUT}/aggregate/baseline_predictions_low.json" \
  "${OUT}/aggregate/candidate_predictions_low.json"; do
  test -f "${path}"
done

"${PY}" - "${OUT}/aggregate/ground_truth.json" \
  "${OUT}/aggregate/candidate_ground_truth.json" <<'PY'
import json
import sys
from pathlib import Path

baseline = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
candidate = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
for field in ("images", "annotations", "categories"):
    if baseline.get(field) != candidate.get(field):
        raise SystemExit(f"merged GT semantic mismatch: {field}")
print("MERGED_GT_SEMANTIC_PARITY_PASS")
PY

printf 'resume_aggregate_evaluation\n' >"${STATUS}"
"${PY}" scripts/evaluate_fixed_score_threshold.py \
  --gt "${OUT}/aggregate/ground_truth.json" \
  --pred "${OUT}/aggregate/baseline_predictions_low.json" --threshold "${THRESHOLD}" \
  --output "${OUT}/aggregate/baseline_fixed_0546.json"
"${PY}" scripts/evaluate_fixed_score_threshold.py \
  --gt "${OUT}/aggregate/ground_truth.json" \
  --pred "${OUT}/aggregate/candidate_predictions_low.json" --threshold "${THRESHOLD}" \
  --output "${OUT}/aggregate/candidate_fixed_0546.json"
"${PY}" scripts/compare_candidate_trend.py \
  --baseline "${OUT}/aggregate/baseline_fixed_0546.json" \
  --candidate "${OUT}/aggregate/candidate_fixed_0546.json" \
  --output "${OUT}/aggregate/paired_comparison.json"
for method in baseline candidate; do
  "${PY}" scripts/analyze_single_split_official_frontier.py \
    --gt "${OUT}/aggregate/ground_truth.json" \
    --pred "${OUT}/aggregate/${method}_predictions_low.json" \
    --output "${OUT}/aggregate/${method}_frontier.json" --step 0.005 \
    >"${OUT}/aggregate/${method}_frontier.log" 2>&1
done
"${PY}" scripts/triage_detector_candidate.py \
  --baseline-fixed "${OUT}/aggregate/baseline_fixed_0546.json" \
  --candidate-fixed "${OUT}/aggregate/candidate_fixed_0546.json" \
  --baseline-frontier "${OUT}/aggregate/baseline_frontier.json" \
  --candidate-frontier "${OUT}/aggregate/candidate_frontier.json" \
  --output "${OUT}/aggregate/candidate_triage.json"

for fold in 0 1 2; do
  if [[ "${fold}" = 0 ]]; then
    checkpoint=/root/autodl-tmp/results/HERA-GUARD-MPSR-HIER-S256V128-FOLD0-40EP-3X4080-B60-V1/training/runs/resolution_adaptation/weights/last.pt
  else
    checkpoint=/root/autodl-tmp/results/HERA-GUARD-MPSR-HIER-S256V128-CV3-REMAINING-3X4080-B60-V1/fold_${fold}/training/runs/resolution_adaptation/weights/last.pt
  fi
  sha256sum "${checkpoint}"
done >"${OUT}/checkpoint_sha256.txt"

printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
