#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${OUT:?set OUT}"
: "${DOTA_DATASET:?set DOTA_DATASET to yolo-dota-v1}"
: "${DOTA_PREP_STATUS:?set DOTA_PREP_STATUS}"
: "${Y5_INITIAL:?set Y5_INITIAL}"
: "${Y5_INITIAL_SHA256:?set Y5_INITIAL_SHA256}"
: "${CV3_MANIFEST:?set CV3_MANIFEST}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${CONFIRMED_MISSING:?set CONFIRMED_MISSING}"
: "${IGNORED_AMBIGUOUS:?set IGNORED_AMBIGUOUS}"
: "${TEACHER_CACHE:?set TEACHER_CACHE}"
: "${BASE_WEIGHT_0:?set BASE_WEIGHT_0}"
: "${BASE_WEIGHT_2:?set BASE_WEIGHT_2}"
: "${BASE_SHA_0:?set BASE_SHA_0}"
: "${BASE_SHA_2:?set BASE_SHA_2}"

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
printf '%s  %s\n' "${BASE_SHA_0}" "${BASE_WEIGHT_0}" | sha256sum -c -
printf '%s  %s\n' "${BASE_SHA_2}" "${BASE_WEIGHT_2}" | sha256sum -c -
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "${GPU_COUNT}" -ge 3 ]] || {
  printf '%s\n' blocked_requires_three_gpus >"${STATUS}"
  exit 2
}
printf '%s\n' "${GPU_COUNT}" >"${OUT}/visible_gpu_count.txt"

sha256sum \
  scripts/train_external_y5_coarse.py \
  scripts/train_external_initialized_y5_fine.py \
  scripts/materialize_external_role_view.py \
  scripts/train_in_model_dfine_agreement.py \
  scripts/materialize_partial_label_safe_dataset.py \
  src/rsdet/external/transfer.py \
  src/rsdet/innovation/in_model_agreement.py \
  src/rsdet/innovation/yolo_feature_quality.py >"${OUT}/CODE_SHA256.txt"

printf '%s\n' materializing_paired_partial_label_datasets >"${STATUS}"
for fold in 0 2; do
  for policy in patch control; do
    target="${OUT}/datasets/fold${fold}-${policy}"
    if [[ ! -f "${target}/dataset_audit.json" ]]; then
      args=(
        --manifest "${CV3_MANIFEST}" --data-root "${DATA_ROOT}"
        --confirmed "${CONFIRMED_MISSING}" --ignored "${IGNORED_AMBIGUOUS}"
        --output-dir "${target}" --held-out-fold "${fold}"
        --ambiguous-policy exclude_image
      )
      [[ "${policy}" = patch ]] && args+=(--confirmed-policy add) || args+=(--confirmed-policy omit)
      "${PYTHON_BIN}" scripts/materialize_partial_label_safe_dataset.py "${args[@]}" \
        >"${OUT}/materialize-fold${fold}-${policy}.log" 2>&1
    fi
  done
done

# EXT-G and EXT-V must not create/overwrite the same Ultralytics labels cache while
# training concurrently.  Both views share immutable image bytes but own labels and
# labels/train.cache independently.
for role in ext-g ext-v; do
  role_view="${OUT}/datasets/dota-${role}-role-view"
  if [[ ! -f "${role_view}/role_view_audit.json" ]]; then
    "${PYTHON_BIN}" scripts/materialize_external_role_view.py \
      --role-yaml "${DOTA_DATASET}/dataset-${role}.yaml" \
      --output-dir "${role_view}" >"${OUT}/materialize-${role}-role-view.log" 2>&1
  fi
done
EXT_G_DATASET="${OUT}/datasets/dota-ext-g-role-view/dataset.yaml"
EXT_V_DATASET="${OUT}/datasets/dota-ext-v-role-view/dataset.yaml"

guard_new_run() {
  local directory=$1 result=$2 checkpoint=$3
  [[ -f "${directory}/${result}" ]] && return 1
  if [[ -f "${directory}/${checkpoint}" ]]; then
    echo "incomplete existing run requires forensic review: ${directory}" >&2
    exit 3
  fi
  return 0
}

printf '%s\n' stage1_external_pretrain_and_had >"${STATUS}"
(
  run="${OUT}/external/ext-g"
  if guard_new_run "${run}" training_result.json runs/foundation/weights/last.pt; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" scripts/train_external_y5_coarse.py \
      --dataset "${EXT_G_DATASET}" --weights "${Y5_INITIAL}" \
      --expected-weight-sha256 "${Y5_INITIAL_SHA256}" --output-dir "${run}" \
      --epochs 80 --imgsz 1024 --batch 12 --workers 8 --seed 20260831 --device 0
  fi
) >"${OUT}/stage1-ext-g.log" 2>&1 & p0=$!
(
  run="${OUT}/external/ext-v"
  if guard_new_run "${run}" training_result.json runs/foundation/weights/last.pt; then
    CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" scripts/train_external_y5_coarse.py \
      --dataset "${EXT_V_DATASET}" --weights "${Y5_INITIAL}" \
      --expected-weight-sha256 "${Y5_INITIAL_SHA256}" --output-dir "${run}" \
      --epochs 80 --imgsz 1024 --batch 12 --workers 8 --seed 20260831 --device 0
  fi
) >"${OUT}/stage1-ext-v.log" 2>&1 & p1=$!
run_had_fold() {
  local fold=$1 gpu=$2 base_weight=$3 base_sha=$4 log=$5
  (
    for mode in branch_only terminal_fpn; do
      run="${OUT}/had/fold${fold}-${mode}"
      if guard_new_run "${run}" training_result.json adapter_last.pt; then
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_in_model_dfine_agreement.py \
          --teacher-cache "${TEACHER_CACHE}" --split-manifest "${CV3_MANIFEST}" \
          --data-root "${DATA_ROOT}" --base-checkpoint "${base_weight}" \
          --expected-base-sha256 "${base_sha}" --output-dir "${run}" \
          --held-out-fold "${fold}" --mode "${mode}" --epochs 8 --imgsz 1024 \
          --max-proposals-per-image 64 --projection-dim 64 --hidden-dim 128 \
          --seed 20260831 --device cuda:0
      fi
    done
  ) >"${log}" 2>&1
}
if [[ "${GPU_COUNT}" -ge 4 ]]; then
  run_had_fold 0 2 "${BASE_WEIGHT_0}" "${BASE_SHA_0}" "${OUT}/stage1-had-fold0.log" & p2=$!
  run_had_fold 2 3 "${BASE_WEIGHT_2}" "${BASE_SHA_2}" "${OUT}/stage1-had-fold2.log" & p3=$!
else
  (
    run_had_fold 0 2 "${BASE_WEIGHT_0}" "${BASE_SHA_0}" "${OUT}/stage1-had-fold0.log"
    run_had_fold 2 2 "${BASE_WEIGHT_2}" "${BASE_SHA_2}" "${OUT}/stage1-had-fold2.log"
  ) & p2=$!
  p3=""
fi
failure=0
for pid in "$p0" "$p1" "$p2" ${p3:+"$p3"}; do wait "$pid" || failure=1; done
[[ "$failure" = 0 ]] || exit 4

printf '%s\n' stage2_paired_fine_transfer >"${STATUS}"
EXT_G_WEIGHT="${OUT}/external/ext-g/runs/foundation/weights/last.pt"
EXT_V_WEIGHT="${OUT}/external/ext-v/runs/foundation/weights/last.pt"
EXT_G_SHA=$(sha256sum "${EXT_G_WEIGHT}" | awk '{print $1}')
EXT_V_SHA=$(sha256sum "${EXT_V_WEIGHT}" | awk '{print $1}')
(
  run="${OUT}/fine/ext-g-patch-fold0"
  if guard_new_run "${run}" training_result.json runs/foundation/weights/last.pt; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" scripts/train_external_initialized_y5_fine.py \
      --dataset "${OUT}/datasets/fold0-patch/dataset.yaml" \
      --external-weights "${EXT_G_WEIGHT}" --expected-weight-sha256 "${EXT_G_SHA}" \
      --output-dir "${run}" --epochs 40 --head-warmup-epochs 8 --freeze-layers 10 \
      --imgsz 1024 --batch 12 --workers 8 --seed 20260831 --device 0
  fi
) >"${OUT}/stage2-ext-g.log" 2>&1 & p0=$!
(
  run="${OUT}/fine/ext-v-patch-fold0"
  if guard_new_run "${run}" training_result.json runs/foundation/weights/last.pt; then
    CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" scripts/train_external_initialized_y5_fine.py \
      --dataset "${OUT}/datasets/fold0-patch/dataset.yaml" \
      --external-weights "${EXT_V_WEIGHT}" --expected-weight-sha256 "${EXT_V_SHA}" \
      --output-dir "${run}" --epochs 40 --head-warmup-epochs 8 --freeze-layers 10 \
      --imgsz 1024 --batch 12 --workers 8 --seed 20260831 --device 0
  fi
) >"${OUT}/stage2-ext-v.log" 2>&1 & p1=$!
run_control_patch() {
  local gpu=$1
  run="${OUT}/fine/control-patch-fold0"
  if guard_new_run "${run}" training_result.json runs/foundation/weights/last.pt; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_external_initialized_y5_fine.py \
      --dataset "${OUT}/datasets/fold0-patch/dataset.yaml" \
      --external-weights "${Y5_INITIAL}" --expected-weight-sha256 "${Y5_INITIAL_SHA256}" \
      --output-dir "${run}" --epochs 40 --head-warmup-epochs 8 --freeze-layers 10 \
      --imgsz 1024 --batch 12 --workers 8 --seed 20260831 --device 0
  fi
}
run_control_omit() {
  local gpu=$1
  run="${OUT}/fine/control-omit-fold0"
  if guard_new_run "${run}" training_result.json runs/foundation/weights/last.pt; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/train_external_initialized_y5_fine.py \
      --dataset "${OUT}/datasets/fold0-control/dataset.yaml" \
      --external-weights "${Y5_INITIAL}" --expected-weight-sha256 "${Y5_INITIAL_SHA256}" \
      --output-dir "${run}" --epochs 40 --head-warmup-epochs 8 --freeze-layers 10 \
      --imgsz 1024 --batch 12 --workers 8 --seed 20260831 --device 0
  fi
}
if [[ "${GPU_COUNT}" -ge 4 ]]; then
  run_control_patch 2 >"${OUT}/stage2-control-patch.log" 2>&1 & p2=$!
  run_control_omit 3 >"${OUT}/stage2-control-omit.log" 2>&1 & p3=$!
else
  (
    run_control_patch 2 >"${OUT}/stage2-control-patch.log" 2>&1
    run_control_omit 2 >"${OUT}/stage2-control-omit.log" 2>&1
  ) & p2=$!
  p3=""
fi
failure=0
for pid in "$p0" "$p1" "$p2" ${p3:+"$p3"}; do wait "$pid" || failure=1; done
[[ "$failure" = 0 ]] || exit 5

find "${OUT}" -type f \( -name training_result.json -o -name dataset_audit.json \) -print0 | sort -z | xargs -0 sha256sum >"${OUT}/RESULT_SHA256.txt"
printf '%s\n' screen_training_complete_ready_for_frozen_evaluation >"${STATUS}"
