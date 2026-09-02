#!/usr/bin/env bash
set -Eeuo pipefail

# Three-GPU controller for the frozen S/M x 1024/1280 fold0 screen.
# GPU0 executes s1024 then m1280; GPU1 executes m1024; GPU2 executes s1280.

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625}
RESULTS_ROOT=${RESULTS_ROOT:-/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
CONTROLLER_STATUS=${RESULTS_ROOT}/controller_status.txt
CONTROLLER_LOG=${RESULTS_ROOT}/controller_events.log
LOCK_FILE=${RESULTS_ROOT}/controller.lock
export PYTHONPATH="${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RESULTS_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf 'another controller already holds %s\n' "${LOCK_FILE}" >&2
  exit 4
fi

failed() {
  code=$?
  printf 'failed_exit_%s\n' "${code}" >"${CONTROLLER_STATUS}"
  printf '%s controller_failed exit=%s\n' "$(date -Is)" "${code}" >>"${CONTROLLER_LOG}"
  jobs -pr | xargs -r kill 2>/dev/null || true
  exit "${code}"
}
trap failed ERR INT TERM

run_cell() {
  gpu=$1
  condition=$2
  cell_status=${RESULTS_ROOT}/${condition}/status.txt
  if [[ -f "${cell_status}" ]] && [[ "$(cat "${cell_status}")" == complete ]]; then
    printf '%s skip_complete condition=%s gpu=%s\n' \
      "$(date -Is)" "${condition}" "${gpu}" >>"${CONTROLLER_LOG}"
    return 0
  fi
  if [[ -e "${RESULTS_ROOT}/${condition}" ]]; then
    printf 'refusing incomplete pre-existing cell: %s\n' "${RESULTS_ROOT}/${condition}" >&2
    return 5
  fi
  printf '%s start condition=%s gpu=%s\n' \
    "$(date -Is)" "${condition}" "${gpu}" >>"${CONTROLLER_LOG}"
  CUDA_VISIBLE_DEVICES=${gpu} CONDITION=${condition} PROJECT=${PROJECT} \
    RESULTS_ROOT=${RESULTS_ROOT} PY=${PY} \
    bash "${PROJECT}/scripts/server/run_yolo_capacity_scale_fold0_condition.sh"
  test "$(cat "${cell_status}")" = complete
  printf '%s complete condition=%s gpu=%s\n' \
    "$(date -Is)" "${condition}" "${gpu}" >>"${CONTROLLER_LOG}"
}

printf 'running_parallel\n' >"${CONTROLLER_STATUS}"
printf '%s controller_start mode=three_gpu\n' "$(date -Is)" >>"${CONTROLLER_LOG}"

run_cell 0 s1024 & pid_s1024=$!
run_cell 1 m1024 & pid_m1024=$!
run_cell 2 s1280 & pid_s1280=$!

wait "${pid_s1024}"
run_cell 0 m1280
wait "${pid_m1024}"
wait "${pid_s1280}"

PROJECT=${PROJECT} ROOT=${RESULTS_ROOT} PY=${PY} \
  bash "${PROJECT}/scripts/server/finalize_yolo_capacity_scale_fold0.sh"

trap - ERR INT TERM
printf 'complete\n' >"${CONTROLLER_STATUS}"
printf '%s controller_complete mode=three_gpu\n' "$(date -Is)" >>"${CONTROLLER_LOG}"
