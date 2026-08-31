#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${OUT:?set OUT to the final screen output directory}"
: "${TEACHER_CACHE:?set TEACHER_CACHE}"
: "${CV3_MANIFEST:?set CV3_MANIFEST}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${BASE_WEIGHT_0:?set BASE_WEIGHT_0}"
: "${BASE_WEIGHT_2:?set BASE_WEIGHT_2}"
: "${BASE_SHA_0:?set BASE_SHA_0}"
: "${BASE_SHA_2:?set BASE_SHA_2}"

HAD_GPU="${HAD_GPU:-2}"
STATUS="${OUT}/had-early-status.txt"
mkdir -p "${OUT}"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed_exit_%s\n" "$code" >"${STATUS}"; fi' EXIT
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

printf '%s  %s\n' "${BASE_SHA_0}" "${BASE_WEIGHT_0}" | sha256sum -c -
printf '%s  %s\n' "${BASE_SHA_2}" "${BASE_WEIGHT_2}" | sha256sum -c -

run_fold() {
  local fold=$1 base_weight=$2 base_sha=$3
  printf 'running_had_fold%s\n' "${fold}" >"${STATUS}"
  for mode in branch_only terminal_fpn; do
    local run="${OUT}/had/fold${fold}-${mode}"
    if [[ -f "${run}/training_result.json" ]]; then
      continue
    fi
    if [[ -f "${run}/adapter_last.pt" || -f "${run}/adapted_detector.pt" ]]; then
      echo "incomplete existing HAD run requires forensic review: ${run}" >&2
      exit 3
    fi
    CUDA_VISIBLE_DEVICES="${HAD_GPU}" "${PYTHON_BIN}" \
      scripts/train_in_model_dfine_agreement.py \
      --teacher-cache "${TEACHER_CACHE}" --split-manifest "${CV3_MANIFEST}" \
      --data-root "${DATA_ROOT}" --base-checkpoint "${base_weight}" \
      --expected-base-sha256 "${base_sha}" --output-dir "${run}" \
      --held-out-fold "${fold}" --mode "${mode}" --epochs 8 --imgsz 1024 \
      --max-proposals-per-image 64 --projection-dim 64 --hidden-dim 128 \
      --seed 20260831 --device cuda:0 \
      >"${OUT}/had-fold${fold}-${mode}.log" 2>&1
  done
}

run_fold 0 "${BASE_WEIGHT_0}" "${BASE_SHA_0}"
run_fold 2 "${BASE_WEIGHT_2}" "${BASE_SHA_2}"
printf '%s\n' complete >"${STATUS}"
