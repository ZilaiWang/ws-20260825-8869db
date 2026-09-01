#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO to the frozen training snapshot}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${OUT:?set OUT to the existing EXT-V experiment root}"
: "${CONTROL_DATASET:?set CONTROL_DATASET}"

STATUS="${OUT}/ext-v-omit-followup-status.txt"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed_exit_%s\n" "$code" >"${STATUS}"; fi' EXIT
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

COARSE_RESULT="${OUT}/external/ext-v/training_result.json"
COARSE_WEIGHT="${OUT}/external/ext-v/runs/foundation/weights/last.pt"
TARGET="${OUT}/fine/ext-v-omit-fold0"
if [[ -f "${TARGET}/training_result.json" ]]; then
  printf '%s\n' complete_already_present >"${STATUS}"
  exit 0
fi
if [[ -f "${TARGET}/runs/foundation/weights/last.pt" ]]; then
  printf '%s\n' blocked_incomplete_existing_target >"${STATUS}"
  exit 3
fi

printf '%s\n' waiting_for_coarse >"${STATUS}"
until [[ -f "${COARSE_RESULT}" && -f "${COARSE_WEIGHT}" ]]; do sleep 30; done
EXT_SHA="$(sha256sum "${COARSE_WEIGHT}" | awk '{print $1}')"

declare -a CELLS=(
  "0:${OUT}/fine/ext-v-patch-fold0/training_result.json"
  "1:${OUT}/fine/control-patch-fold0/training_result.json"
  "2:${OUT}/fine/control-omit-fold0/training_result.json"
)
printf '%s\n' waiting_for_first_free_fine_gpu >"${STATUS}"
FREED_GPU=""
while [[ -z "${FREED_GPU}" ]]; do
  for cell in "${CELLS[@]}"; do
    gpu="${cell%%:*}"
    result="${cell#*:}"
    if [[ -f "${result}" ]]; then FREED_GPU="${gpu}"; break; fi
  done
  [[ -n "${FREED_GPU}" ]] || sleep 15
done
sleep 10

printf 'training_on_gpu_%s\n' "${FREED_GPU}" >"${STATUS}"
CUDA_VISIBLE_DEVICES="${FREED_GPU}" "${PYTHON_BIN}" \
  scripts/train_external_initialized_y5_fine.py \
  --dataset "${CONTROL_DATASET}" --external-weights "${COARSE_WEIGHT}" \
  --expected-weight-sha256 "${EXT_SHA}" --output-dir "${TARGET}" \
  --epochs 40 --head-warmup-epochs 8 --freeze-layers 10 \
  --imgsz 1024 --batch 12 --workers 8 --seed 20260831 --device 0 \
  >"${OUT}/fine-ext-v-omit.log" 2>&1
[[ -f "${TARGET}/training_result.json" ]] || exit 4
sha256sum "${TARGET}/training_result.json" "${TARGET}/head_transfer_audit.json" \
  >"${OUT}/EXT_V_OMIT_SHA256.txt"
printf '%s\n' complete >"${STATUS}"
