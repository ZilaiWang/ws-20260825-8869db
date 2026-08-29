#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${PSEUDO_ROOT:?set PSEUDO_ROOT}"
: "${Y5_ROOT:?set Y5_ROOT}"
: "${M3_ROOT:?set M3_ROOT}"
: "${COPH_ROOT:?set COPH_ROOT}"
: "${DEV_RAW_FRONTIER:?set DEV_RAW_FRONTIER}"
: "${OUT:?set OUT}"

mkdir -p "${OUT}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
GT="${PSEUDO_ROOT}/ground_truth.json"
Y5_WEIGHTS=(
  "${Y5_ROOT}/fold_0/training/runs/foundation/weights/last.pt"
  "${Y5_ROOT}/fold_1/training/runs/foundation/weights/last.pt"
  "${Y5_ROOT}/fold_2/training/runs/foundation/weights/last.pt"
)
M3_WEIGHTS=(
  "${M3_ROOT}/fold_0/training/runs/foundation/weights/last.pt"
  "${M3_ROOT}/fold_1/training/runs/foundation/weights/last.pt"
  "${M3_ROOT}/fold_2/training/runs/foundation/weights/last.pt"
)
COPH_WEIGHTS=("${COPH_ROOT}/fold_0.pt" "${COPH_ROOT}/fold_1.pt" "${COPH_ROOT}/fold_2.pt")

printf '%s\n' y5rot > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/run_cv3_oof_pseudo_eval.py \
  --pseudo-root "${PSEUDO_ROOT}" \
  --config submission/docker/configs/y5_oof_safe_1024_floor0001.json \
  --weights "${Y5_WEIGHTS[@]}" --output-dir "${OUT}/y5rot" \
  > "${OUT}/y5rot.log" 2>&1

printf '%s\n' y5800 > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/run_cv3_oof_pseudo_eval.py \
  --pseudo-root "${PSEUDO_ROOT}" \
  --config submission/docker/configs/y5_oof_safe_800_floor0001.json \
  --weights "${Y5_WEIGHTS[@]}" --output-dir "${OUT}/y5800" \
  > "${OUT}/y5800.log" 2>&1

printf '%s\n' m3 > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/run_multifamily_cv3_pseudo_eval.py \
  --pseudo-root "${PSEUDO_ROOT}" --family rtdetr \
  --weights "${M3_WEIGHTS[@]}" --output-dir "${OUT}/m3" \
  --score-floor 0.03 --batch-size 2 --device cuda:0 > "${OUT}/m3.log" 2>&1

printf '%s\n' coph > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/run_cv3_oof_pseudo_eval.py \
  --pseudo-root "${PSEUDO_ROOT}" \
  --config submission/docker/configs/y5_oof_safe_1024_rot90tta_floor0001.json \
  --weights "${COPH_WEIGHTS[@]}" --output-dir "${OUT}/coph" \
  > "${OUT}/coph.log" 2>&1

printf '%s\n' merge > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/merge_pseudo_candidate_sources.py \
  --source "Y5ROT=${OUT}/y5rot/predictions.json" \
  --source "Y5800=${OUT}/y5800/predictions.json" \
  --source "M3=${OUT}/m3/predictions.json" \
  --source "COPH=${OUT}/coph/predictions.json" \
  --nms-iou 0.60 --nms-backend torchvision \
  --output "${OUT}/four_source_nms060.json" \
  --summary "${OUT}/merge_summary.json" > "${OUT}/merge.log" 2>&1

printf '%s\n' analysis > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/analyze_cv3_pseudo_candidate_ceiling.py \
  --gt "${GT}" --pred "${OUT}/four_source_nms060.json" \
  --output "${OUT}/candidate_ceiling.json" > "${OUT}/candidate.log" 2>&1
"${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "${GT}" --pred "${OUT}/four_source_nms060.json" \
  --output "${OUT}/sentinel_crossfit_frontier_diagnostic.json" \
  --threshold-start 0.0 --threshold-stop 1.0 --threshold-step 0.001 \
  --fdr-levels 0.10 0.12 0.15 0.20 > "${OUT}/frontier.log" 2>&1
for level in 0.10 0.12 0.15 0.20; do
  "${PYTHON_BIN}" scripts/evaluate_pseudo_with_frozen_thresholds.py \
    --gt "${GT}" --pred "${OUT}/four_source_nms060.json" \
    --source-frontier "${DEV_RAW_FRONTIER}" --fdr-level "${level}" \
    --output "${OUT}/frozen_dev_threshold_fdr${level}.json" \
    > "${OUT}/frozen_fdr${level}.log" 2>&1
done
(
  cd "${OUT}"
  sha256sum y5rot/predictions.json y5800/predictions.json m3/predictions.json \
    coph/predictions.json four_source_nms060.json merge_summary.json \
    candidate_ceiling.json sentinel_crossfit_frontier_diagnostic.json \
    frozen_dev_threshold_fdr*.json > SHA256SUMS
)
printf '%s\n' complete > "${OUT}/status.txt"
