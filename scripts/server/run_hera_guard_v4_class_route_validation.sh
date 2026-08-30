#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/xh-202625}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}"
RESULTS="${RESULTS:-/workspace/results}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data}"
NORMAL_GT="${NORMAL_GT:-/workspace/inputs/HERA-GUARD-V3-FORMAL-DUAL-VIEW-V1/ground_truth.json}"
SENTINEL_ROOT="${SENTINEL_ROOT:-/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1}"
OUT="${RESULTS}/HERA-GUARD-V4-CLASS-ROUTE-VALIDATION"
NORMAL_ROOT="${OUT}/normal_root"
NORMAL="${OUT}/normal"
SENTINEL="${OUT}/sentinel"
IDENTITY_CONFIG="submission/docker/configs/y5_oof_safe_1024_floor0001.json"
DUAL_CONFIG="submission/docker/configs/y5_oof_safe_1024_rot90cwtta_floor0001.json"
HARD_FRONTIER="${RESULTS}/HERA-GUARD-V4-CACHED-TTA/strategies/identity_aircraft_rot_ship_vehicle_frontier.json"
WEIGHTS=(
  /workspace/xh-pre-eval-ab/artifacts/M1-CV3-OOF/last.pt/fold0_last.pt
  /workspace/xh-pre-eval-ab/artifacts/M1-CV3-OOF/last.pt/fold1_last.pt
  /workspace/xh-pre-eval-ab/artifacts/M1-CV3-OOF/last.pt/fold2_last.pt
)

cd "${REPO}"
mkdir -p "${OUT}" "${NORMAL}" "${SENTINEL}"
printf 'materialize_normal\n' >"${OUT}/status.txt"
"${PYTHON_BIN}" scripts/materialize_normal_cv3_pseudo_root.py \
  --gt "${NORMAL_GT}" --data-root "${DATA_ROOT}" --output-root "${NORMAL_ROOT}" \
  >"${OUT}/materialize_normal.log" 2>&1

run_views() {
  local root="$1"
  local target="$2"
  for view in identity dual; do
    local config="${IDENTITY_CONFIG}"
    if [[ "${view}" == dual ]]; then config="${DUAL_CONFIG}"; fi
    printf 'inference:%s:%s\n' "$(basename "${target}")" "${view}" >"${OUT}/status.txt"
    "${PYTHON_BIN}" scripts/run_cv3_oof_pseudo_eval.py \
      --pseudo-root "${root}" --config "${config}" --weights "${WEIGHTS[@]}" \
      --output-dir "${target}/${view}" >"${target}/${view}.log" 2>&1
  done
  printf 'route:%s\n' "$(basename "${target}")" >"${OUT}/status.txt"
  "${PYTHON_BIN}" scripts/build_cached_class_routed_tta.py \
    --identity "${target}/identity/predictions.json" \
    --rot90 "${target}/dual/predictions.json" \
    --output-dir "${target}/strategies" --support-iou 0.25 --nms-iou 0.50 \
    >"${target}/route.log" 2>&1
}

run_views "${NORMAL_ROOT}" "${NORMAL}"
for name in identity identity_aircraft_rot_ship_vehicle; do
  printf 'normal_frontier:%s\n' "${name}" >"${OUT}/status.txt"
  "${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_coarse_thresholds.py \
    --gt "${NORMAL_ROOT}/ground_truth.json" \
    --pred "${NORMAL}/strategies/${name}.json" \
    --output "${NORMAL}/strategies/${name}_frontier.json" --threshold-step 0.005 \
    >"${NORMAL}/strategies/${name}_frontier.log" 2>&1
done

run_views "${SENTINEL_ROOT}" "${SENTINEL}"
for name in identity identity_aircraft_rot_ship_vehicle; do
  printf 'sentinel_frozen:%s\n' "${name}" >"${OUT}/status.txt"
  source_frontier="${HARD_FRONTIER}"
  if [[ "${name}" == identity ]]; then
    source_frontier="${RESULTS}/HERA-GUARD-V4-CACHED-TTA/strategies/identity_frontier.json"
  fi
  "${PYTHON_BIN}" scripts/evaluate_pseudo_with_frozen_coarse_thresholds.py \
    --gt "${SENTINEL_ROOT}/ground_truth.json" \
    --pred "${SENTINEL}/strategies/${name}.json" \
    --source-frontier "${source_frontier}" --fdr-level 0.15 \
    --output "${SENTINEL}/strategies/${name}_frozen_fdr015.json" \
    >"${SENTINEL}/strategies/${name}_frozen_fdr015.log" 2>&1
done

printf 'complete\n' >"${OUT}/status.txt"
