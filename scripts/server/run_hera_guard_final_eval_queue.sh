#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO to the frozen evaluation snapshot}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${GPU:?set one physical GPU index}"
: "${BASE_WEIGHT_0:?set BASE_WEIGHT_0}"
: "${BASE_WEIGHT_1:?set BASE_WEIGHT_1}"
: "${BASE_WEIGHT_2:?set BASE_WEIGHT_2}"
: "${NORMAL_ROOT:?set NORMAL_ROOT}"
: "${HARD_ROOT:?set HARD_ROOT}"
: "${SENTINEL_ROOT:?set SENTINEL_ROOT}"
: "${BASELINE_EVAL_ROOT:?set BASELINE_EVAL_ROOT}"
: "${CANDIDATE_A:?set CANDIDATE_A}"
: "${CANDIDATE_A_READY:?set CANDIDATE_A_READY}"
: "${OUT_A:?set OUT_A}"

run_candidate() {
  local candidate=$1 ready=$2 output=$3
  until [[ -f "${ready}" && -f "${candidate}" ]]; do sleep 15; done
  if [[ -f "${output}/decision.json" && "$(cat "${output}/status.txt" 2>/dev/null)" = complete ]]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" env \
    REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" OUT="${output}" CANDIDATE_FOLD=0 \
    CANDIDATE_WEIGHT="${candidate}" \
    BASE_WEIGHT_0="${BASE_WEIGHT_0}" BASE_WEIGHT_1="${BASE_WEIGHT_1}" \
    BASE_WEIGHT_2="${BASE_WEIGHT_2}" NORMAL_ROOT="${NORMAL_ROOT}" \
    HARD_ROOT="${HARD_ROOT}" SENTINEL_ROOT="${SENTINEL_ROOT}" \
    BASELINE_EVAL_ROOT="${BASELINE_EVAL_ROOT}" \
    bash "${REPO}/scripts/server/run_hera_guard_final_candidate_eval.sh" \
    >"${output}.driver.log" 2>&1
}

run_candidate "${CANDIDATE_A}" "${CANDIDATE_A_READY}" "${OUT_A}"
if [[ -n "${CANDIDATE_B:-}" ]]; then
  : "${CANDIDATE_B_READY:?set CANDIDATE_B_READY with CANDIDATE_B}"
  : "${OUT_B:?set OUT_B with CANDIDATE_B}"
  run_candidate "${CANDIDATE_B}" "${CANDIDATE_B_READY}" "${OUT_B}"
fi
