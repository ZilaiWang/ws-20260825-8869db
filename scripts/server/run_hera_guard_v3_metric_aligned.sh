#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${GT:?set GT}"
: "${EVIDENCE_PRED:?set EVIDENCE_PRED}"
: "${ANCHOR_PRED:?set ANCHOR_PRED}"
: "${OUT:?set OUT}"

mkdir -p "${OUT}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
printf '%s\n' audit > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/audit_metric_aligned_pseudo_labels.py \
  --gt "${GT}" --pred "${ANCHOR_PRED}" --output-dir "${OUT}/e0" \
  > "${OUT}/e0.log" 2>&1

printf '%s\n' train > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/train_metric_aligned_risk.py \
  --gt "${GT}" --evidence-pred "${EVIDENCE_PRED}" \
  --anchor-pred "${ANCHOR_PRED}" --output-dir "${OUT}/risk" \
  --epochs-per-stage 60 --residual-limit 2.5 --target-fdr 0.15 \
  --device cuda:0 > "${OUT}/train.log" 2>&1

printf '%s\n' frontier > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "${GT}" --pred "${ANCHOR_PRED}" --output "${OUT}/baseline_frontier.json" \
  > "${OUT}/baseline_frontier.log" 2>&1
for stage in bce rank soft_fdr one_winner; do
  "${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
    --gt "${GT}" --pred "${OUT}/risk/${stage}_oof_predictions.json" \
    --output "${OUT}/${stage}_frontier.json" > "${OUT}/${stage}_frontier.log" 2>&1
done

printf '%s\n' decision > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/compare_metric_risk_stages.py \
  --baseline "${OUT}/baseline_frontier.json" \
  --stage "bce=${OUT}/bce_frontier.json" \
  --stage "rank=${OUT}/rank_frontier.json" \
  --stage "soft_fdr=${OUT}/soft_fdr_frontier.json" \
  --stage "one_winner=${OUT}/one_winner_frontier.json" \
  --output "${OUT}/decision.json" > "${OUT}/decision.log" 2>&1
(
  cd "${OUT}"
  sha256sum e0/audit_summary.json risk/summary.json *_frontier.json decision.json \
    > SHA256SUMS
)
printf '%s\n' complete > "${OUT}/status.txt"
