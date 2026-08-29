#!/usr/bin/env bash
set -euo pipefail

# Required environment variables keep every asset explicit and auditable.
: "${REPO:?set REPO}"
: "${GT:?set GT}"
: "${BASE_PRED:?set BASE_PRED}"
: "${PIXEL_RISK_PRED:?set PIXEL_RISK_PRED}"
: "${PIXEL_INFERENCE_SUMMARY:?set PIXEL_INFERENCE_SUMMARY}"
: "${OUT:?set OUT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUT}"
printf '%s\n' running > "${OUT}/status.txt"
cd "${REPO}"

"${PYTHON_BIN}" scripts/simulate_budgeted_pixel_verifier.py \
  --ground-truth "${GT}" \
  --base-predictions "${BASE_PRED}" \
  --pixel-risk-predictions "${PIXEL_RISK_PRED}" \
  --pixel-inference-summary "${PIXEL_INFERENCE_SUMMARY}" \
  --output-dir "${OUT}" \
  --budgets 0 64 128 256 512 1024 2048 \
  > "${OUT}/router.log" 2>&1

for budget in 0 64 128 256 512 1024 2048; do
  tag=$(printf '%04d' "${budget}")
  "${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
    --gt "${GT}" \
    --pred "${OUT}/budget_${tag}_predictions.json" \
    --output "${OUT}/budget_${tag}_frontier.json" \
    > "${OUT}/budget_${tag}_frontier.log" 2>&1
done

(
  cd "${OUT}"
  sha256sum summary.json budget_*_frontier.json budget_*_predictions.json > SHA256SUMS
)
printf '%s\n' complete > "${OUT}/status.txt"
