#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${OUT:?set OUT}"
: "${BASELINE_ROOT:?set BASELINE_ROOT}"
: "${NORMAL_GT:?set NORMAL_GT}"
: "${SENTINEL_ROOT:?set SENTINEL_ROOT}"

cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
[[ "$(cat "${OUT}/status.txt")" = complete_ready_for_analysis ]] || {
  echo "EXT-V screen has not completed fixed Hard/Sentinel inference" >&2
  exit 2
}

run_normal() {
  local route=$1 weight=$2
  REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" WEIGHT="${weight}" \
    OUT="${OUT}/evaluation/${route}/normal" BASELINE_ROOT="${BASELINE_ROOT}" \
    GROUND_TRUTH="${NORMAL_GT}" \
    bash scripts/server/run_y5_fold0_normal_replacement_eval.sh
}

CANDIDATE_WEIGHT="${OUT}/fine/extv-fold0/runs/foundation/weights/last.pt"
CONTROL_WEIGHT="${OUT}/fine/control-fold0/runs/foundation/weights/last.pt"
run_normal candidate "${CANDIDATE_WEIGHT}"
run_normal control "${CONTROL_WEIGHT}"

for route in candidate control; do
  "${PYTHON_BIN}" scripts/evaluate_pseudo_with_frozen_thresholds.py \
    --gt "${SENTINEL_ROOT}/ground_truth.json" \
    --pred "${OUT}/evaluation/${route}/sentinel/predictions.json" \
    --source-frontier "${OUT}/evaluation/${route}/hard/frontier.json" \
    --fdr-level 0.15 \
    --output "${OUT}/evaluation/${route}/sentinel/frozen_from_hard_fdr15.json" \
    >"${OUT}/evaluation/${route}/sentinel/frozen_from_hard_fdr15.log" 2>&1
done

"${PYTHON_BIN}" scripts/decide_hera_guard_final_candidate.py \
  --normal-base "${OUT}/evaluation/control/normal/frontier.json" \
  --normal-candidate "${OUT}/evaluation/candidate/normal/frontier.json" \
  --hard-base "${OUT}/evaluation/control/hard/frontier.json" \
  --hard-candidate "${OUT}/evaluation/candidate/hard/frontier.json" \
  --sentinel-base "${OUT}/evaluation/control/sentinel/frontier.json" \
  --sentinel-candidate "${OUT}/evaluation/candidate/sentinel/frontier.json" \
  --sentinel-base-frozen \
    "${OUT}/evaluation/control/sentinel/frozen_from_hard_fdr15.json" \
  --sentinel-candidate-frozen \
    "${OUT}/evaluation/candidate/sentinel/frozen_from_hard_fdr15.json" \
  --fdr-level 0.150 --selection-mode fdr_level \
  --output "${OUT}/decision.json" >"${OUT}/decision.log" 2>&1

find "${OUT}/evaluation" -type f \
  \( -name frontier.json -o -name frozen_from_hard_fdr15.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"${OUT}/FINAL_EVAL_SHA256.txt"
sha256sum "${OUT}/decision.json" >>"${OUT}/FINAL_EVAL_SHA256.txt"
printf '%s\n' complete_with_normal_hard_sentinel_decision >"${OUT}/status.txt"
