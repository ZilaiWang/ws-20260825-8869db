#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/xh-202625}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
TRAINING_MANIFEST="${TRAINING_MANIFEST:?set TRAINING_MANIFEST}"
INFERENCE_MANIFEST="${INFERENCE_MANIFEST:?set INFERENCE_MANIFEST}"
BASE_LOGITS_DIR="${BASE_LOGITS_DIR:?set BASE_LOGITS_DIR}"
AGGREGATE_DIR="${AGGREGATE_DIR:?set AGGREGATE_DIR}"
FORMAL_CROP_MANIFEST="${FORMAL_CROP_MANIFEST:?set FORMAL_CROP_MANIFEST}"
Y1_CALIBRATION_RESULT="${Y1_CALIBRATION_RESULT:?set Y1_CALIBRATION_RESULT}"
CONVNEXT_WEIGHTS="${CONVNEXT_WEIGHTS:?set CONVNEXT_WEIGHTS}"
P03_FOLD0_CHECKPOINT="${P03_FOLD0_CHECKPOINT:?set P03_FOLD0_CHECKPOINT}"
P03_FOLD1_CHECKPOINT="${P03_FOLD1_CHECKPOINT:?set P03_FOLD1_CHECKPOINT}"
P03_FOLD2_CHECKPOINT="${P03_FOLD2_CHECKPOINT:?set P03_FOLD2_CHECKPOINT}"
R11_ROOT="${R11_ROOT:-/workspace/results/R1-1-AIRCRAFT-PROPOSAL-REFINEMENT}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results}"
RUN_ID="R1-2-AIRCRAFT-CLASS-CENTER"
RUN_ROOT="${RESULTS_ROOT}/${RUN_ID}"
STATUS_PATH="${RESULTS_ROOT}/${RUN_ID}.status"
LOCK_DIR="${RESULTS_ROOT}/.${RUN_ID}.lock"
CONFIG="${PROJECT_ROOT}/configs/experiments/r1_aircraft_class_center_v1.yaml"
CODE_LOCK="${PROJECT_ROOT}/docs/server/R1_AIRCRAFT_CLASS_CENTER_TASK_01_CODE_SHA256.txt"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "R1-2 lock already exists: ${LOCK_DIR}" >&2
  exit 1
fi
cleanup_lock() { rmdir "${LOCK_DIR}" 2>/dev/null || true; }
trap cleanup_lock EXIT
on_error() {
  local code=$?
  echo "failed_exit_${code}" > "${STATUS_PATH}"
  exit "${code}"
}
trap on_error ERR

if [[ -e "${RUN_ROOT}" ]]; then
  echo "R1-2 run root exists; refuse overwrite: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}/audit" "${RUN_ROOT}/smoke" \
  "${RUN_ROOT}/train/class_center" "${RUN_ROOT}/bundles/class_center" \
  "${RUN_ROOT}/evaluation"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
echo "running_preflight" > "${STATUS_PATH}"
sha256sum --check "${CODE_LOCK}"
"${PYTHON_BIN}" -m pytest -q \
  tests/test_proposal_reranking.py tests/test_aircraft_refinement.py \
  tests/test_aircraft_class_center_torch.py
"${PYTHON_BIN}" -m ruff check \
  src/rsdet/analysis/aircraft_refinement.py scripts/r1_aircraft_refinement.py \
  tests/test_aircraft_refinement.py tests/test_aircraft_class_center_torch.py

for path in \
  "${DATA_ROOT}" "${TRAINING_MANIFEST}" "${INFERENCE_MANIFEST}" \
  "${BASE_LOGITS_DIR}" "${AGGREGATE_DIR}" "${FORMAL_CROP_MANIFEST}" \
  "${Y1_CALIBRATION_RESULT}" "${CONVNEXT_WEIGHTS}" \
  "${P03_FOLD0_CHECKPOINT}" "${P03_FOLD1_CHECKPOINT}" "${P03_FOLD2_CHECKPOINT}" \
  "${R11_ROOT}/FINAL_GATE_PASS" "${R11_ROOT}/bundles/p03" \
  "${R11_ROOT}/bundles/ce" "${R11_ROOT}/bundles/selective_anchor_kd"; do
  [[ -e "${path}" ]] || { echo "missing required asset: ${path}" >&2; exit 1; }
done

echo "running_data_audit" > "${STATUS_PATH}"
"${PYTHON_BIN}" scripts/r1_aircraft_refinement.py --config "${CONFIG}" audit \
  --training-manifest "${TRAINING_MANIFEST}" \
  --inference-manifest "${INFERENCE_MANIFEST}" \
  --output-dir "${RUN_ROOT}/audit"

checkpoints=("${P03_FOLD0_CHECKPOINT}" "${P03_FOLD1_CHECKPOINT}" "${P03_FOLD2_CHECKPOINT}")
echo "running_smoke_class_center" > "${STATUS_PATH}"
"${PYTHON_BIN}" scripts/r1_aircraft_refinement.py --config "${CONFIG}" train \
  --training-manifest "${TRAINING_MANIFEST}" --data-root "${DATA_ROOT}" \
  --weights "${CONVNEXT_WEIGHTS}" --p03-checkpoint "${checkpoints[0]}" \
  --fold 0 --method class_center --output-dir "${RUN_ROOT}/smoke/class_center" \
  --device cuda --batch-size 16 --num-workers 2 --smoke

for fold in 0 1 2; do
  echo "training_class_center_fold_${fold}" > "${STATUS_PATH}"
  "${PYTHON_BIN}" scripts/r1_aircraft_refinement.py --config "${CONFIG}" train \
    --training-manifest "${TRAINING_MANIFEST}" --data-root "${DATA_ROOT}" \
    --weights "${CONVNEXT_WEIGHTS}" --p03-checkpoint "${checkpoints[$fold]}" \
    --fold "${fold}" --method class_center \
    --output-dir "${RUN_ROOT}/train/class_center/fold_${fold}" --device cuda

  echo "inferring_class_center_d4_fold_${fold}" > "${STATUS_PATH}"
  "${PYTHON_BIN}" scripts/r1_aircraft_refinement.py --config "${CONFIG}" infer \
    --inference-manifest "${INFERENCE_MANIFEST}" --data-root "${DATA_ROOT}" \
    --weights "${CONVNEXT_WEIGHTS}" \
    --checkpoint "${RUN_ROOT}/train/class_center/fold_${fold}/final_checkpoint.pt" \
    --fold "${fold}" --method class_center \
    --output-dir "${RUN_ROOT}/bundles/class_center" --device cuda
done

echo "running_crossfit_evaluation" > "${STATUS_PATH}"
"${PYTHON_BIN}" scripts/r1_aircraft_refinement.py --config "${CONFIG}" evaluate \
  --inference-manifest "${INFERENCE_MANIFEST}" --base-logits-dir "${BASE_LOGITS_DIR}" \
  --p03-bundle-dir "${R11_ROOT}/bundles/p03" \
  --ce-bundle-dir "${R11_ROOT}/bundles/ce" \
  --kd-bundle-dir "${R11_ROOT}/bundles/selective_anchor_kd" \
  --center-bundle-dir "${RUN_ROOT}/bundles/class_center" \
  --aggregate-dir "${AGGREGATE_DIR}" --formal-crop-manifest "${FORMAL_CROP_MANIFEST}" \
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
assert decision["primary_condition"] == "class_center_d4"
assert decision["maximum_bypass_delta"] <= 1e-12
for fold in range(3):
    runtime = json.loads(
        (root / "bundles" / "class_center" / f"fold_{fold}_runtime.json").read_text()
    )
    assert runtime["status"] == "complete"
(root / "FINAL_GATE_PASS").write_text("R1_2_TASK_PASS\n", encoding="utf-8")
PY

find "${RUN_ROOT}/train" -type f -name final_checkpoint.pt -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/CHECKPOINTS_SHA256.txt"
RETURN_PACKAGE="${RESULTS_ROOT}/${RUN_ID}-return-no-checkpoints.tar.gz"
tar -C "${RESULTS_ROOT}" --exclude='final_checkpoint.pt' -czf "${RETURN_PACKAGE}" "${RUN_ID}"
sha256sum "${RETURN_PACKAGE}" > "${RETURN_PACKAGE}.sha256"
echo "complete" > "${STATUS_PATH}"
echo "R1-2 complete: ${RETURN_PACKAGE}"
