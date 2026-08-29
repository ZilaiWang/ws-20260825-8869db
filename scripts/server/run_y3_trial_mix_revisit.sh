#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${PSEUDO_ROOT:?set PSEUDO_ROOT}"
: "${GT:?set GT}"
: "${WEIGHT_ROOT:?set WEIGHT_ROOT}"
: "${BASE_FOUR_SOURCE:?set BASE_FOUR_SOURCE}"
: "${WAIT_STATUS:?set WAIT_STATUS}"
: "${OUT:?set OUT}"

mkdir -p "${OUT}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
while [[ ! -f "${WAIT_STATUS}" ]] || [[ "$(cat "${WAIT_STATUS}")" != complete ]]; do
  printf '%s\n' waiting_for_gpu_slot > "${OUT}/status.txt"
  sleep 30
done

printf '%s\n' inference > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/run_cv3_oof_pseudo_eval.py \
  --pseudo-root "${PSEUDO_ROOT}" \
  --config "${REPO}/submission/docker/configs/y5_oof_safe_1024_floor0001.json" \
  --weights \
    "${WEIGHT_ROOT}/fold_0/training/runs/foundation/weights/last.pt" \
    "${WEIGHT_ROOT}/fold_1/training/runs/foundation/weights/last.pt" \
    "${WEIGHT_ROOT}/fold_2/training/runs/foundation/weights/last.pt" \
  --output-dir "${OUT}/inference" > "${OUT}/inference.log" 2>&1

printf '%s\n' analysis > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/analyze_cv3_pseudo_candidate_ceiling.py \
  --gt "${GT}" --pred "${OUT}/inference/predictions.json" \
  --output "${OUT}/single_candidate_ceiling.json" > "${OUT}/single_candidate.log" 2>&1
"${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "${GT}" --pred "${OUT}/inference/predictions.json" \
  --output "${OUT}/single_frontier.json" \
  --threshold-start 0.0 --threshold-stop 1.0 --threshold-step 0.001 \
  --fdr-levels 0.10 0.12 0.15 0.20 > "${OUT}/single_frontier.log" 2>&1

"${PYTHON_BIN}" scripts/merge_pseudo_candidate_sources.py \
  --source "BASE=${BASE_FOUR_SOURCE}" \
  --source "Y3=${OUT}/inference/predictions.json" \
  --nms-iou 0.60 --nms-backend torchvision \
  --output "${OUT}/base_plus_y3_nms060.json" \
  --summary "${OUT}/merge_summary.json" > "${OUT}/merge.log" 2>&1
"${PYTHON_BIN}" scripts/analyze_cv3_pseudo_candidate_ceiling.py \
  --gt "${GT}" --pred "${OUT}/base_plus_y3_nms060.json" \
  --output "${OUT}/merged_candidate_ceiling.json" > "${OUT}/merged_candidate.log" 2>&1
"${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "${GT}" --pred "${OUT}/base_plus_y3_nms060.json" \
  --output "${OUT}/merged_frontier.json" \
  --threshold-start 0.0 --threshold-stop 1.0 --threshold-step 0.001 \
  --fdr-levels 0.10 0.12 0.15 0.20 > "${OUT}/merged_frontier.log" 2>&1
(
  cd "${OUT}"
  sha256sum inference/predictions.json single_candidate_ceiling.json single_frontier.json \
    base_plus_y3_nms060.json merge_summary.json merged_candidate_ceiling.json \
    merged_frontier.json > SHA256SUMS
)
printf '%s\n' complete > "${OUT}/status.txt"
