#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/xh-202625}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the D00 image root containing images/train}"
AGGREGATE_DIR="${AGGREGATE_DIR:?set AGGREGATE_DIR to M1-CV3-OOF-aggregate}"
FORMAL_CROP_MANIFEST="${FORMAL_CROP_MANIFEST:?set FORMAL_CROP_MANIFEST}"
Y1_CALIBRATION_RESULT="${Y1_CALIBRATION_RESULT:?set Y1_CALIBRATION_RESULT}"
CONVNEXT_WEIGHTS="${CONVNEXT_WEIGHTS:?set CONVNEXT_WEIGHTS}"
P03_FOLD0_CHECKPOINT="${P03_FOLD0_CHECKPOINT:?set P03_FOLD0_CHECKPOINT}"
P03_FOLD1_CHECKPOINT="${P03_FOLD1_CHECKPOINT:?set P03_FOLD1_CHECKPOINT}"
P03_FOLD2_CHECKPOINT="${P03_FOLD2_CHECKPOINT:?set P03_FOLD2_CHECKPOINT}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results}"
RUN_ROOT="${RESULTS_ROOT}/R1-0-P03-TEACHER-M1-OOF"
STATUS_PATH="${RESULTS_ROOT}/R1-0-P03-TEACHER-M1-OOF.status"
LOCK_DIR="${RESULTS_ROOT}/.R1-0-P03-TEACHER-M1-OOF.lock"
CONFIG="${PROJECT_ROOT}/configs/experiments/r1_proposal_reranking_v1.yaml"
CODE_LOCK="${PROJECT_ROOT}/docs/server/R1_PROPOSAL_RERANKING_TASK_00_CODE_SHA256.txt"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "R1-0 lock already exists: ${LOCK_DIR}" >&2
  exit 1
fi
cleanup_lock() { rmdir "${LOCK_DIR}" 2>/dev/null || true; }
trap cleanup_lock EXIT
on_error() {
  local exit_code=$?
  echo "failed_exit_${exit_code}" > "${STATUS_PATH}"
  exit "${exit_code}"
}
trap on_error ERR

mkdir -p "${RESULTS_ROOT}"
if [[ -e "${RUN_ROOT}" ]]; then
  echo "R1-0 run root already exists; refuse overwrite: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}/prepare" "${RUN_ROOT}/smoke" "${RUN_ROOT}/logits" "${RUN_ROOT}/evaluation"
echo "running_preflight" > "${STATUS_PATH}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

sha256sum --check "${CODE_LOCK}"
"${PYTHON_BIN}" -m pytest -q tests/test_proposal_reranking.py tests/test_yolo_calibration.py
"${PYTHON_BIN}" -m ruff check \
  src/rsdet/analysis/proposal_reranking.py \
  src/rsdet/postprocess/yolo_calibration.py \
  scripts/r1_proposal_reranking.py \
  tests/test_proposal_reranking.py

for required in \
  "${DATA_ROOT}" "${AGGREGATE_DIR}" "${FORMAL_CROP_MANIFEST}" \
  "${Y1_CALIBRATION_RESULT}" "${CONVNEXT_WEIGHTS}" \
  "${P03_FOLD0_CHECKPOINT}" "${P03_FOLD1_CHECKPOINT}" "${P03_FOLD2_CHECKPOINT}"; do
  [[ -e "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 1; }
done

echo "running_prepare" > "${STATUS_PATH}"
"${PYTHON_BIN}" scripts/r1_proposal_reranking.py --config "${CONFIG}" prepare \
  --aggregate-dir "${AGGREGATE_DIR}" \
  --output-dir "${RUN_ROOT}/prepare"
MANIFEST="${RUN_ROOT}/prepare/proposal_inference_manifest.csv"

checkpoints=("${P03_FOLD0_CHECKPOINT}" "${P03_FOLD1_CHECKPOINT}" "${P03_FOLD2_CHECKPOINT}")
for fold in 0 1 2; do
  echo "running_smoke_fold_${fold}" > "${STATUS_PATH}"
  "${PYTHON_BIN}" scripts/r1_proposal_reranking.py --config "${CONFIG}" infer \
    --manifest "${MANIFEST}" \
    --data-root "${DATA_ROOT}" \
    --weights "${CONVNEXT_WEIGHTS}" \
    --checkpoint "${checkpoints[$fold]}" \
    --fold "${fold}" \
    --output-dir "${RUN_ROOT}/smoke" \
    --device cuda --batch-size 32 --num-workers 2 --smoke
done

for fold in 0 1 2; do
  echo "running_inference_fold_${fold}" > "${STATUS_PATH}"
  "${PYTHON_BIN}" scripts/r1_proposal_reranking.py --config "${CONFIG}" infer \
    --manifest "${MANIFEST}" \
    --data-root "${DATA_ROOT}" \
    --weights "${CONVNEXT_WEIGHTS}" \
    --checkpoint "${checkpoints[$fold]}" \
    --fold "${fold}" \
    --output-dir "${RUN_ROOT}/logits" \
    --device cuda
done

echo "running_crossfit_evaluation" > "${STATUS_PATH}"
"${PYTHON_BIN}" scripts/r1_proposal_reranking.py --config "${CONFIG}" evaluate \
  --manifest "${MANIFEST}" \
  --logits-dir "${RUN_ROOT}/logits" \
  --aggregate-dir "${AGGREGATE_DIR}" \
  --formal-crop-manifest "${FORMAL_CROP_MANIFEST}" \
  --y1-calibration-result "${Y1_CALIBRATION_RESULT}" \
  --output-dir "${RUN_ROOT}/evaluation"

"${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
prepare = json.loads((root / "prepare" / "prepare_audit.json").read_text())
decision = json.loads((root / "evaluation" / "decision.json").read_text())
for fold in range(3):
    runtime = json.loads((root / "logits" / f"fold_{fold}_runtime.json").read_text())
    assert runtime["status"] == "complete"
assert prepare["proposal_count"] == 55548
assert decision["status"] == "complete"
(root / "FINAL_GATE_PASS").write_text("R1_0_TASK_PASS\n", encoding="utf-8")
PY

RETURN_PACKAGE="${RESULTS_ROOT}/R1-0-P03-TEACHER-M1-OOF-return.tar.gz"
tar -C "${RESULTS_ROOT}" -czf "${RETURN_PACKAGE}" "$(basename "${RUN_ROOT}")"
sha256sum "${RETURN_PACKAGE}" > "${RETURN_PACKAGE}.sha256"
echo "complete" > "${STATUS_PATH}"
echo "R1-0 complete: ${RETURN_PACKAGE}"
