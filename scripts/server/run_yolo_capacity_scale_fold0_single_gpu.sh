#!/usr/bin/env bash
set -Eeuo pipefail

# Single-GPU controller for the frozen S/M x 1024/1280 fold0 screen.
# A complete cell is immutable and may be skipped after an external shutdown.
# An incomplete pre-existing cell is never resumed or overwritten automatically.

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625}
RESULTS_ROOT=${RESULTS_ROOT:-/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
export PYTHONPATH="${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"
CONTROLLER_STATUS=${RESULTS_ROOT}/controller_status.txt
CONTROLLER_LOG=${RESULTS_ROOT}/controller_events.log
LOCK_FILE=${RESULTS_ROOT}/controller.lock

mkdir -p "${RESULTS_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf 'another controller already holds %s\n' "${LOCK_FILE}" >&2
  exit 4
fi

current=preflight
failed() {
  code=$?
  printf 'failed_%s_exit_%s\n' "${current}" "${code}" >"${CONTROLLER_STATUS}"
  printf '%s failed condition=%s exit=%s\n' "$(date -Is)" "${current}" "${code}" >>"${CONTROLLER_LOG}"
  exit "${code}"
}
trap failed ERR

printf 'running\n' >"${CONTROLLER_STATUS}"
printf '%s controller_start\n' "$(date -Is)" >>"${CONTROLLER_LOG}"

for condition in s1024 m1024 s1280 m1280; do
  current=${condition}
  cell_status=${RESULTS_ROOT}/${condition}/status.txt
  if [[ -f "${cell_status}" ]] && [[ "$(cat "${cell_status}")" == complete ]]; then
    printf '%s skip_complete condition=%s\n' "$(date -Is)" "${condition}" >>"${CONTROLLER_LOG}"
    continue
  fi
  if [[ -e "${RESULTS_ROOT}/${condition}" ]]; then
    printf 'refusing incomplete pre-existing cell: %s\n' "${RESULTS_ROOT}/${condition}" >&2
    exit 5
  fi
  printf '%s start condition=%s\n' "$(date -Is)" "${condition}" >>"${CONTROLLER_LOG}"
  CUDA_VISIBLE_DEVICES=0 CONDITION=${condition} PROJECT=${PROJECT} \
    RESULTS_ROOT=${RESULTS_ROOT} PY=${PY} \
    bash "${PROJECT}/scripts/server/run_yolo_capacity_scale_fold0_condition.sh"
  test "$(cat "${cell_status}")" = complete
  printf '%s complete condition=%s\n' "$(date -Is)" "${condition}" >>"${CONTROLLER_LOG}"
done

current=finalize
PROJECT=${PROJECT} ROOT=${RESULTS_ROOT} PY=${PY} \
  bash "${PROJECT}/scripts/server/finalize_yolo_capacity_scale_fold0.sh"

trap - ERR
printf 'complete\n' >"${CONTROLLER_STATUS}"
printf '%s controller_complete\n' "$(date -Is)" >>"${CONTROLLER_LOG}"
