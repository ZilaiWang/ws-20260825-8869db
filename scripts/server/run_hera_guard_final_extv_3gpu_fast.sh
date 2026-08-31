#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${OUT:?set OUT}"
: "${DOTA_PREP_STATUS:?set DOTA_PREP_STATUS}"
: "${EXT_V_DATASET:?set EXT_V_DATASET}"
: "${PATCH_DATASET:?set PATCH_DATASET}"
: "${CONTROL_DATASET:?set CONTROL_DATASET}"
: "${Y5_INITIAL:?set Y5_INITIAL}"
: "${Y5_INITIAL_SHA256:?set Y5_INITIAL_SHA256}"

mkdir -p "${OUT}"
STATUS="${OUT}/status.txt"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed_exit_%s\n" "$code" >"${STATUS}"; fi' EXIT
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

[[ "$(cat "${DOTA_PREP_STATUS}")" = ready_for_external_pretraining ]] || {
  printf '%s\n' blocked_dota_not_ready >"${STATUS}"
  exit 2
}
printf '%s  %s\n' "${Y5_INITIAL_SHA256}" "${Y5_INITIAL}" | sha256sum -c -
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "${GPU_COUNT}" -eq 3 ]] || { echo "requires exactly three visible GPUs" >&2; exit 2; }
sha256sum \
  scripts/train_external_y5_coarse.py \
  scripts/train_external_initialized_y5_fine.py \
  src/rsdet/external/transfer.py >"${OUT}/CODE_SHA256.txt"

guard_new_run() {
  local directory=$1 result=$2 checkpoint=$3
  [[ -f "${directory}/${result}" ]] && return 1
  if [[ -f "${directory}/${checkpoint}" ]]; then
    echo "incomplete existing run requires forensic review: ${directory}" >&2
    exit 3
  fi
  return 0
}

COARSE="${OUT}/external/ext-v"
printf '%s\n' extv_coarse_ddp_3gpu >"${STATUS}"
if guard_new_run "${COARSE}" training_result.json runs/foundation/weights/last.pt; then
  # Total batch 30 gives 10 images/GPU. Ultralytics accumulates two steps,
  # preserving the original effective optimization batch of 60 (12 x 5).
  "${PYTHON_BIN}" scripts/train_external_y5_coarse.py \
    --dataset "${EXT_V_DATASET}" --weights "${Y5_INITIAL}" \
    --expected-weight-sha256 "${Y5_INITIAL_SHA256}" --output-dir "${COARSE}" \
    --epochs 80 --imgsz 1024 --batch 30 --workers 8 --seed 20260831 \
    --device 0,1,2 >"${OUT}/coarse-ext-v.log" 2>&1
fi

EXT_WEIGHT="${COARSE}/runs/foundation/weights/last.pt"
EXT_SHA="$(sha256sum "${EXT_WEIGHT}" | awk '{print $1}')"
printf '%s\n' paired_fine_three_single_gpu >"${STATUS}"
run_fine() {
  local gpu=$1 dataset=$2 weights=$3 expected_sha=$4 run=$5 log=$6
  if guard_new_run "${run}" training_result.json runs/foundation/weights/last.pt; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_external_initialized_y5_fine.py \
      --dataset "${dataset}" --external-weights "${weights}" \
      --expected-weight-sha256 "${expected_sha}" --output-dir "${run}" \
      --epochs 40 --head-warmup-epochs 8 --freeze-layers 10 \
      --imgsz 1024 --batch 12 --workers 8 --seed 20260831 --device 0 \
      >"${log}" 2>&1
  fi
}
run_fine 0 "${PATCH_DATASET}" "${EXT_WEIGHT}" "${EXT_SHA}" \
  "${OUT}/fine/ext-v-patch-fold0" "${OUT}/fine-ext-v.log" & p0=$!
run_fine 1 "${PATCH_DATASET}" "${Y5_INITIAL}" "${Y5_INITIAL_SHA256}" \
  "${OUT}/fine/control-patch-fold0" "${OUT}/fine-control-patch.log" & p1=$!
run_fine 2 "${CONTROL_DATASET}" "${Y5_INITIAL}" "${Y5_INITIAL_SHA256}" \
  "${OUT}/fine/control-omit-fold0" "${OUT}/fine-control-omit.log" & p2=$!
failure=0
for pid in "${p0}" "${p1}" "${p2}"; do wait "${pid}" || failure=1; done
[[ "${failure}" = 0 ]] || exit 4

find "${OUT}" -type f \( -name training_result.json -o -name head_transfer_audit.json \) \
  -print0 | sort -z | xargs -0 sha256sum >"${OUT}/RESULT_SHA256.txt"
printf '%s\n' complete_ready_for_frozen_evaluation >"${STATUS}"
