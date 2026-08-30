#!/usr/bin/env bash
set -Eeuo pipefail

# PAV-V2 fold0 fast screen.  This is a controlled supervision-contract repair,
# not an epoch/architecture scale-up: V1 hyperparameters remain unchanged.
REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/xh-202625}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data}"
WEIGHT_PATH="${WEIGHT_PATH:-/root/autodl-tmp/pretrained/convnext_tiny-983f1562.pth}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/results}"
RUN_ID="HERA-PAV-V2-OBJECTNESS-RISK-FOLD0"
OUTPUT_DIR="${RESULT_ROOT}/${RUN_ID}"
STATUS_PATH="${RESULT_ROOT}/${RUN_ID}.status"
LOCK_PATH="${RESULT_ROOT}/${RUN_ID}.lock"

MANIFEST="${REPO_ROOT}/outputs/HERA-GUARD-PRECHECK/pav-manifest-oer-v2/hera_pav_manifest.csv"
PREDICTIONS="${REPO_ROOT}/outputs/HERA-GUARD-PRECHECK/corrected-oer-oof-v1/corrected_oer_oof_predictions.json"
FORMAL_MANIFEST="${REPO_ROOT}/outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv"
EXPECTED_PAV_SHA="165cb51428cb2f721a9516ccd9f7eb459d09a1e96e80ec738c530642a35690ad"
EXPECTED_FORMAL_SHA="a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128"
EXPECTED_WEIGHT_SHA="983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"

mkdir -p "${OUTPUT_DIR}"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "A PAV-V2 process already holds ${LOCK_PATH}" >&2
  exit 73
fi
on_error() {
  local code=$?
  printf 'failed:%s\n' "${code}" >"${STATUS_PATH}"
  exit "${code}"
}
trap on_error ERR INT TERM

export PATH="$(dirname "${PYTHON_BIN}"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
cd "${REPO_ROOT}"

check_sha() {
  local expected=$1 path=$2 actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ "${actual}" == "${expected}" ]] || {
    echo "SHA256 mismatch for ${path}: expected=${expected}, actual=${actual}" >&2
    return 1
  }
}

printf 'preflight\n' >"${STATUS_PATH}"
check_sha "${EXPECTED_PAV_SHA}" "${MANIFEST}"
check_sha "${EXPECTED_FORMAL_SHA}" "${FORMAL_MANIFEST}"
check_sha "${EXPECTED_WEIGHT_SHA}" "${WEIGHT_PATH}"

printf 'training\n' >"${STATUS_PATH}"
"${PYTHON_BIN}" scripts/train_hera_pav.py \
  --manifest "${MANIFEST}" \
  --data-root "${DATA_ROOT}" \
  --convnext-weights "${WEIGHT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --held-out-fold 0 \
  --freeze freeze_first_stages \
  --resolution 224 \
  --hidden-dim 512 \
  --epochs 4 \
  --batch-size 48 \
  --samples-per-epoch 24000 \
  --head-learning-rate 0.0003 \
  --backbone-learning-rate 0.00001 \
  --weight-decay 0.05 \
  --num-workers 6 \
  --seed 202625 \
  --device cuda \
  --verify-weight-sha256 \
  >"${OUTPUT_DIR}/train.log" 2>&1

printf 'evaluating\n' >"${STATUS_PATH}"
"${PYTHON_BIN}" scripts/evaluate_hera_pav_fast_screen.py \
  --predictions "${PREDICTIONS}" \
  --manifest "${MANIFEST}" \
  --pav-logits "${OUTPUT_DIR}/pav_fold0_oof_logits.npz" \
  --formal-crop-manifest "${FORMAL_MANIFEST}" \
  --project-config configs/project.yaml \
  --held-out-fold 0 \
  --output "${OUTPUT_DIR}/fast_screen_result.json" \
  >"${OUTPUT_DIR}/evaluate.log" 2>&1

printf 'packaging\n' >"${STATUS_PATH}"
RETURN_PACKAGE="${RESULT_ROOT}/${RUN_ID}-return-no-checkpoint.tar.gz"
tar --exclude='*.pt' -C "${RESULT_ROOT}" -czf "${RETURN_PACKAGE}" "${RUN_ID}"
sha256sum "${RETURN_PACKAGE}" >"${RETURN_PACKAGE}.sha256"
printf 'complete\n' >"${STATUS_PATH}"
