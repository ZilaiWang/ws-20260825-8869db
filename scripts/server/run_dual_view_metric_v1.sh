#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${GT:?set GT}"
: "${EVIDENCE_PRED:?set EVIDENCE_PRED}"
: "${ANCHOR_PRED:?set ANCHOR_PRED}"
: "${PSEUDO_ROOT:?set PSEUDO_ROOT}"
: "${IMAGENET_WEIGHT:?set IMAGENET_WEIGHT}"
: "${BASELINE_FRONTIER:?set BASELINE_FRONTIER}"
: "${OUT:?set OUT}"

mkdir -p "${OUT}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf '%s\n' training > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/train_dual_view_metric_verifier.py \
  --gt "${GT}" --evidence-pred "${EVIDENCE_PRED}" --anchor-pred "${ANCHOR_PRED}" \
  --pseudo-root "${PSEUDO_ROOT}" --imagenet-weight "${IMAGENET_WEIGHT}" \
  --output-dir "${OUT}/model" --epochs 3 --batches-per-epoch 100 \
  --batch-size 32 --inference-batch-size 96 --resolution 224 \
  --context-ratio 1.75 --residual-limit 2.5 --device cuda:0 \
  > "${OUT}/train.log" 2>&1
printf '%s\n' frontier > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "${GT}" --pred "${OUT}/model/dual_view_oof_predictions.json" \
  --output "${OUT}/frontier.json" --threshold-step 0.001 > "${OUT}/frontier.log" 2>&1
"${PYTHON_BIN}" scripts/compare_metric_risk_stages.py \
  --baseline "${BASELINE_FRONTIER}" --stage "dual_view=${OUT}/frontier.json" \
  --output "${OUT}/decision.json" > "${OUT}/decision.log" 2>&1
(
  cd "${OUT}"
  sha256sum model/summary.json model/dual_view_oof_predictions.json \
    frontier.json decision.json > SHA256SUMS
)
printf '%s\n' complete > "${OUT}/status.txt"
