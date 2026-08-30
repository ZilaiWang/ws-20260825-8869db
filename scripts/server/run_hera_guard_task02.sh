#!/usr/bin/env bash
set -Eeuo pipefail

# Expand the admitted PAV screen to folds 1/2, then merge the three frozen
# outer-fold predictions.  The fold0-selected resolver is only confirmed here;
# no fusion weights, relabel thresholds, epochs, or checkpoints are searched.
REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/xh-202625}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data}"
WEIGHT_PATH="${WEIGHT_PATH:-/root/autodl-tmp/pretrained/convnext_tiny-983f1562.pth}"
RESULT_ROOT="${RESULT_ROOT:-/workspace/results}"
STATUS_PATH="${RESULT_ROOT}/HERA-PAV-THREEFOLD-V1.status"
LOCK_PATH="${RESULT_ROOT}/HERA-PAV-THREEFOLD-V1.lock"
MERGED_DIR="${RESULT_ROOT}/HERA-PAV-THREEFOLD-V1"

MANIFEST="${REPO_ROOT}/outputs/HERA-GUARD-PRECHECK/pav-manifest-oer-v1/hera_pav_manifest.csv"
PREDICTIONS="${REPO_ROOT}/outputs/HERA-GUARD-PRECHECK/corrected-oer-oof-v1/corrected_oer_oof_predictions.json"
FORMAL_MANIFEST="${REPO_ROOT}/outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv"
EXPECTED_PAV_SHA="d259ed6f5b88d96d890de0d2843c66b256d28dc7ebab5767df3f60cb0acc9156"
EXPECTED_FORMAL_SHA="a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128"
EXPECTED_WEIGHT_SHA="983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"

mkdir -p "${MERGED_DIR}"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "A HERA TASK-02 process already holds ${LOCK_PATH}" >&2
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
test -s "${RESULT_ROOT}/HERA-PAV-FAST-FOLD0-V1/pav_fold0_oof_logits.npz"
test -s "${RESULT_ROOT}/HERA-PAV-FAST-FOLD0-V1/pav_fold0_result.json"

for fold in 1 2; do
  run_id="HERA-PAV-FAST-FOLD${fold}-V1"
  output_dir="${RESULT_ROOT}/${run_id}"
  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing result directory: ${output_dir}" >&2
    exit 74
  fi
  mkdir -p "${output_dir}"
  printf 'training_fold%s\n' "${fold}" >"${STATUS_PATH}"
  "${PYTHON_BIN}" scripts/train_hera_pav.py \
    --manifest "${MANIFEST}" \
    --data-root "${DATA_ROOT}" \
    --convnext-weights "${WEIGHT_PATH}" \
    --output-dir "${output_dir}" \
    --held-out-fold "${fold}" \
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
    >"${output_dir}/train.log" 2>&1

  printf 'evaluating_fold%s\n' "${fold}" >"${STATUS_PATH}"
  "${PYTHON_BIN}" scripts/evaluate_hera_pav_fast_screen.py \
    --predictions "${PREDICTIONS}" \
    --manifest "${MANIFEST}" \
    --pav-logits "${output_dir}/pav_fold${fold}_oof_logits.npz" \
    --formal-crop-manifest "${FORMAL_MANIFEST}" \
    --project-config configs/project.yaml \
    --held-out-fold "${fold}" \
    --output "${output_dir}/fast_screen_result.json" \
    >"${output_dir}/evaluate.log" 2>&1
done

printf 'merging_oof\n' >"${STATUS_PATH}"
"${PYTHON_BIN}" scripts/merge_hera_pav_oof.py \
  --inputs \
    "${RESULT_ROOT}/HERA-PAV-FAST-FOLD0-V1/pav_fold0_oof_logits.npz" \
    "${RESULT_ROOT}/HERA-PAV-FAST-FOLD1-V1/pav_fold1_oof_logits.npz" \
    "${RESULT_ROOT}/HERA-PAV-FAST-FOLD2-V1/pav_fold2_oof_logits.npz" \
  --manifest "${MANIFEST}" \
  --output "${MERGED_DIR}/pav_threefold_oof_logits.npz" \
  --summary "${MERGED_DIR}/merge_summary.json" \
  >"${MERGED_DIR}/merge.log" 2>&1

printf 'evaluating_threefold_oof\n' >"${STATUS_PATH}"
"${PYTHON_BIN}" scripts/evaluate_hera_pav_fast_screen.py \
  --predictions "${PREDICTIONS}" \
  --manifest "${MANIFEST}" \
  --pav-logits "${MERGED_DIR}/pav_threefold_oof_logits.npz" \
  --formal-crop-manifest "${FORMAL_MANIFEST}" \
  --project-config configs/project.yaml \
  --held-out-fold all \
  --output "${MERGED_DIR}/threefold_oof_confirmation.json" \
  >"${MERGED_DIR}/evaluate_threefold.log" 2>&1

printf 'packaging\n' >"${STATUS_PATH}"
RETURN_PACKAGE="${RESULT_ROOT}/HERA-PAV-THREEFOLD-V1-return-no-checkpoint.tar.gz"
tar --exclude='*.pt' -C "${RESULT_ROOT}" -czf "${RETURN_PACKAGE}" \
  HERA-PAV-FAST-FOLD0-V1 HERA-PAV-FAST-FOLD1-V1 HERA-PAV-FAST-FOLD2-V1 \
  HERA-PAV-THREEFOLD-V1
sha256sum "${RETURN_PACKAGE}" >"${RETURN_PACKAGE}.sha256"
printf 'complete\n' >"${STATUS_PATH}"
