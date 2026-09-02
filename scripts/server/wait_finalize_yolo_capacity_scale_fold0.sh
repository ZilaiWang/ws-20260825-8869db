#!/usr/bin/env bash
set -Eeuo pipefail

# Recovery-safe finalizer for the frozen YOLO S/M x 1024/1280 fold0 screen.
# It never launches or resumes training. It only waits until every condition
# has independently reached `complete`, then runs the deterministic summary.

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625}
ROOT=${ROOT:-/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
POLL_SECONDS=${POLL_SECONDS:-30}
STATUS=${ROOT}/recovery_finalizer_status.txt
LOG=${ROOT}/recovery_finalizer.log
LOCK=${ROOT}/recovery_finalizer.lock
CONDITIONS=(s1024 s1280 m1024 m1280)

mkdir -p "${ROOT}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  printf 'another recovery finalizer already holds %s\n' "${LOCK}" >&2
  exit 4
fi

failed() {
  code=$?
  printf 'failed_exit_%s\n' "${code}" >"${STATUS}"
  printf '%s finalizer_failed exit=%s\n' "$(date -Is)" "${code}" >>"${LOG}"
  exit "${code}"
}
trap failed ERR INT TERM

printf 'waiting_for_cells\n' >"${STATUS}"
printf '%s finalizer_start\n' "$(date -Is)" >>"${LOG}"
while true; do
  complete=0
  states=()
  for condition in "${CONDITIONS[@]}"; do
    cell_status=${ROOT}/${condition}/status.txt
    state=missing
    if [[ -f "${cell_status}" ]]; then
      state=$(<"${cell_status}")
    fi
    states+=("${condition}=${state}")
    if [[ "${state}" == complete ]]; then
      complete=$((complete + 1))
    fi
  done
  printf '%s %s\n' "$(date -Is)" "${states[*]}" >>"${LOG}"
  if [[ "${complete}" -eq "${#CONDITIONS[@]}" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

printf 'finalizing\n' >"${STATUS}"
PROJECT="${PROJECT}" ROOT="${ROOT}" PY="${PY}" \
  bash "${PROJECT}/scripts/server/finalize_yolo_capacity_scale_fold0.sh"

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
printf '%s finalizer_complete\n' "$(date -Is)" >>"${LOG}"
