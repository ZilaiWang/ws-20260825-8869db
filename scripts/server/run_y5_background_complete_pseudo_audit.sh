#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${PSEUDO_ROOT:?set PSEUDO_ROOT}"
: "${GT:?set GT}"
: "${TRAIN_ROOT:?set TRAIN_ROOT}"
: "${OUT:?set OUT}"

mkdir -p "${OUT}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
while [[ ! -f "${TRAIN_ROOT}/status.txt" ]] || [[ "$(cat "${TRAIN_ROOT}/status.txt")" != complete ]]; do
  if [[ -f "${TRAIN_ROOT}/status.txt" ]] && grep -q '^failed' "${TRAIN_ROOT}/status.txt"; then
    printf '%s\n' blocked_by_training_failure > "${OUT}/status.txt"
    exit 2
  fi
  printf '%s\n' waiting_for_training > "${OUT}/status.txt"
  sleep 30
done

printf '%s\n' inference > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/run_multifamily_cv3_pseudo_eval.py \
  --pseudo-root "${PSEUDO_ROOT}" \
  --family yolo \
  --weights \
    "${TRAIN_ROOT}/fold_0/runs/foundation/weights/last.pt" \
    "${TRAIN_ROOT}/fold_1/runs/foundation/weights/last.pt" \
    "${TRAIN_ROOT}/fold_2/runs/foundation/weights/last.pt" \
  --output-dir "${OUT}/inference" \
  --score-floor 0.03 \
  --batch-size 4 \
  --device cuda:0 > "${OUT}/inference.log" 2>&1

printf '%s\n' analysis > "${OUT}/status.txt"
"${PYTHON_BIN}" scripts/analyze_cv3_pseudo_candidate_ceiling.py \
  --gt "${GT}" --pred "${OUT}/inference/predictions.json" \
  --output "${OUT}/candidate_ceiling.json" > "${OUT}/candidate.log" 2>&1
"${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "${GT}" --pred "${OUT}/inference/predictions.json" \
  --output "${OUT}/frontier.json" \
  --threshold-start 0.0 --threshold-stop 1.0 --threshold-step 0.001 \
  --fdr-levels 0.10 0.12 0.15 0.20 > "${OUT}/frontier.log" 2>&1
(
  cd "${OUT}"
  sha256sum inference/predictions.json inference/run_summary.json \
    candidate_ceiling.json frontier.json > SHA256SUMS
)
printf '%s\n' complete > "${OUT}/status.txt"
