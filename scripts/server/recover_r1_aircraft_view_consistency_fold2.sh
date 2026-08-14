#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/xh-202625}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data}"
TRAINING_MANIFEST="${TRAINING_MANIFEST:-/workspace/inputs/R1-1/proposal_crop_manifest.csv}"
INFERENCE_MANIFEST="${INFERENCE_MANIFEST:-/workspace/results/R1-0-P03-TEACHER-M1-OOF/prepare/proposal_inference_manifest.csv}"
BASE_LOGITS_DIR="${BASE_LOGITS_DIR:-/workspace/results/R1-0-P03-TEACHER-M1-OOF/logits}"
AGGREGATE_DIR="${AGGREGATE_DIR:-/workspace/N1A/M1-CV3-OOF-aggregate}"
FORMAL_CROP_MANIFEST="${FORMAL_CROP_MANIFEST:-/workspace/N1A/formal_crop_manifest.csv}"
Y1_CALIBRATION_RESULT="${Y1_CALIBRATION_RESULT:-/workspace/results/Y1-CROSSFIT-CALIBRATION-V1/calibration_result.json}"
CONVNEXT_WEIGHTS="${CONVNEXT_WEIGHTS:-/workspace/pretrained/convnext_tiny-983f1562.pth}"
R11_ROOT="${R11_ROOT:-/workspace/results/R1-1-AIRCRAFT-PROPOSAL-REFINEMENT}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results}"
RUN_ID="R1-5-AIRCRAFT-VIEW-CONSISTENCY"
RUN_ROOT="${RESULTS_ROOT}/${RUN_ID}"
STATUS_PATH="${RESULTS_ROOT}/${RUN_ID}.status"
RECOVERY_LOCK="${RESULTS_ROOT}/.${RUN_ID}.recovery-fold2.lock"
STALE_LOCK="${RESULTS_ROOT}/.${RUN_ID}.lock"
CONFIG="${PROJECT_ROOT}/configs/experiments/r1_aircraft_view_consistency_v1.yaml"

if ! mkdir "${RECOVERY_LOCK}" 2>/dev/null; then
  echo "R1-5 fold2 recovery already active" >&2
  exit 1
fi
cleanup_recovery_lock() { rmdir "${RECOVERY_LOCK}" 2>/dev/null || true; }
trap cleanup_recovery_lock EXIT
on_error() {
  local code=$?
  echo "recovery_failed_exit_${code}" > "${STATUS_PATH}"
  exit "${code}"
}
trap on_error ERR

[[ "$(cat "${STATUS_PATH}")" == "inferring_view_consistency_d4_fold_2" ]] || {
  echo "unexpected recovery status: $(cat "${STATUS_PATH}")" >&2
  exit 1
}
[[ -d "${STALE_LOCK}" ]] || { echo "original stale lock missing" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}/bundles/view_consistency/fold_2_aircraft_bundle.npz" ]] || {
  echo "fold2 bundle already exists; refuse inference replay" >&2
  exit 1
}
for fold in 0 1; do
  [[ -f "${RUN_ROOT}/bundles/view_consistency/fold_${fold}_aircraft_bundle.npz" ]]
  [[ -f "${RUN_ROOT}/bundles/view_consistency/fold_${fold}_runtime.json" ]]
done
for fold in 0 1 2; do
  [[ -f "${RUN_ROOT}/train/view_consistency/fold_${fold}/final_checkpoint.pt" ]]
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
echo "recovery_inferring_view_consistency_d4_fold_2" > "${STATUS_PATH}"
"${PYTHON_BIN}" scripts/r1_aircraft_refinement.py --config "${CONFIG}" infer \
  --inference-manifest "${INFERENCE_MANIFEST}" --data-root "${DATA_ROOT}" \
  --weights "${CONVNEXT_WEIGHTS}" \
  --checkpoint "${RUN_ROOT}/train/view_consistency/fold_2/final_checkpoint.pt" \
  --fold 2 --method view_consistency \
  --output-dir "${RUN_ROOT}/bundles/view_consistency" --device cuda

echo "recovery_running_crossfit_evaluation" > "${STATUS_PATH}"
"${PYTHON_BIN}" scripts/r1_aircraft_refinement.py --config "${CONFIG}" evaluate \
  --inference-manifest "${INFERENCE_MANIFEST}" \
  --base-logits-dir "${BASE_LOGITS_DIR}" \
  --p03-bundle-dir "${R11_ROOT}/bundles/p03" \
  --ce-bundle-dir "${R11_ROOT}/bundles/ce" \
  --kd-bundle-dir "${R11_ROOT}/bundles/selective_anchor_kd" \
  --consistency-bundle-dir "${RUN_ROOT}/bundles/view_consistency" \
  --aggregate-dir "${AGGREGATE_DIR}" \
  --formal-crop-manifest "${FORMAL_CROP_MANIFEST}" \
  --y1-calibration-result "${Y1_CALIBRATION_RESULT}" \
  --output-dir "${RUN_ROOT}/evaluation"

"${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
audit = json.loads((root / "audit" / "aircraft_data_audit.json").read_text())
decision = json.loads((root / "evaluation" / "decision.json").read_text())
assert audit["status"] == "pass"
assert audit["training"]["row_count"] == 17948
assert audit["inference"]["aircraft_proposals"] == 32062
assert decision["status"] == "complete"
assert decision["reference_condition"] == "ce_identity"
assert decision["primary_condition"] == "view_consistency_identity"
assert decision["secondary_reference_condition"] == "ce_d4"
assert decision["secondary_condition"] == "view_consistency_d4"
assert decision["maximum_bypass_delta"] <= 1e-12
for fold in range(3):
    runtime = json.loads(
        (root / "bundles" / "view_consistency" / f"fold_{fold}_runtime.json").read_text()
    )
    assert runtime["status"] == "complete"
(root / "FINAL_GATE_PASS").write_text("R1_5_TASK_PASS_AFTER_FOLD2_RECOVERY\n")
PY

find "${RUN_ROOT}/train" -type f -name final_checkpoint.pt -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/CHECKPOINTS_SHA256.txt"
RETURN_PACKAGE="${RESULTS_ROOT}/${RUN_ID}-return-no-checkpoints.tar.gz"
tar -C "${RESULTS_ROOT}" --exclude='final_checkpoint.pt' \
  -czf "${RETURN_PACKAGE}" "${RUN_ID}"
(
  cd "${RESULTS_ROOT}"
  sha256sum "$(basename "${RETURN_PACKAGE}")" \
    > "$(basename "${RETURN_PACKAGE}").sha256"
)
rmdir "${STALE_LOCK}"
echo "complete" > "${STATUS_PATH}"
echo "R1-5 fold2 recovery complete: ${RETURN_PACKAGE}"
