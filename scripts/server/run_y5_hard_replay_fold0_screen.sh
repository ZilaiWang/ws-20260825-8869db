#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${MANIFEST:?set MANIFEST}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${TILE_ROOT:?set TILE_ROOT}"
: "${PSEUDO_ROOT:?set PSEUDO_ROOT}"
: "${WEIGHT_ROOT:?set WEIGHT_ROOT}"
: "${OUT:?set OUT}"

mkdir -p "${OUT}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

declare -a WEIGHT_SHA=(
  "47d98fab29cbc4b6836a907a77cda33294affbd891e90f9e3aab0b05578e7c96"
  "3b175e1471ae139dd4415cac094487e5a4be369cb0f1af6094bd3b2f1f25a9d4"
  "8cec3f91cd0421c328f1fa430de2b9b4def4f28fdab41b1d90d56e76bd413304"
)

printf '%s\n' training_fold0 > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/train_y5_hard_replay.py \
  --manifest "${MANIFEST}" \
  --data-root "${DATA_ROOT}" \
  --hard-negative-summary "${TILE_ROOT}/fold_0/summary.json" \
  --held-out-fold 0 \
  --weights "${WEIGHT_ROOT}/fold_0/training/runs/foundation/weights/last.pt" \
  --expected-weight-sha256 "${WEIGHT_SHA[0]}" \
  --output-dir "${OUT}/train_fold0" \
  --maximum-hard-tiles 320 \
  --epochs 6 \
  --imgsz 1024 \
  --batch 8 \
  --workers 8 \
  --seed 20260830 > "${OUT}/train_fold0.log" 2>&1

printf '%s\n' hard10k_inference > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/run_multifamily_cv3_pseudo_eval.py \
  --pseudo-root "${PSEUDO_ROOT}" \
  --family yolo \
  --weights \
    "${OUT}/train_fold0/runs/foundation/weights/last.pt" \
    "${WEIGHT_ROOT}/fold_1/training/runs/foundation/weights/last.pt" \
    "${WEIGHT_ROOT}/fold_2/training/runs/foundation/weights/last.pt" \
  --output-dir "${OUT}/inference" \
  --score-floor 0.03 \
  --batch-size 4 \
  --device cuda:0 > "${OUT}/inference.log" 2>&1

printf '%s\n' analysis > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "${PSEUDO_ROOT}/ground_truth.json" \
  --pred "${PSEUDO_ROOT}/../Y5-ROT90CW-CV3-PSEUDO10K-TRIAL-MIX-V1/predictions.json" \
  --output "${OUT}/baseline_frontier.json" \
  --threshold-start 0.001 --threshold-stop 0.996 --threshold-step 0.005 \
  --fdr-levels 0.10 0.12 0.15 0.20 > "${OUT}/baseline_frontier.log" 2>&1
"${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "${PSEUDO_ROOT}/ground_truth.json" \
  --pred "${OUT}/inference/predictions.json" \
  --output "${OUT}/candidate_frontier.json" \
  --threshold-start 0.001 --threshold-stop 0.996 --threshold-step 0.005 \
  --fdr-levels 0.10 0.12 0.15 0.20 > "${OUT}/candidate_frontier.log" 2>&1
"${PYTHON_BIN}" scripts/compare_hard_replay_screen.py \
  --baseline "${OUT}/baseline_frontier.json" \
  --candidate "${OUT}/candidate_frontier.json" \
  --output "${OUT}/decision.json" > "${OUT}/decision.log" 2>&1

(
  cd "${OUT}"
  sha256sum train_fold0/training_contract.json train_fold0/training_result.json \
    baseline_frontier.json candidate_frontier.json decision.json > SHA256SUMS
)
printf '%s\n' complete > "${OUT}/status.txt"
