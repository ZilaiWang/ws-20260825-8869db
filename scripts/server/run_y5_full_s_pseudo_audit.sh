#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/workspace/xh-202625}
PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}
WEIGHT=${WEIGHT:-/workspace/results/Y5-FULL-S-20260829-R1/y5_full_s_sanitized.pt}
CONFIG=${CONFIG:-$REPO/submission/docker/configs/y5_oof_safe_1024_floor0001.json}
RESULT_ROOT=${RESULT_ROOT:-/workspace/results}
NATURAL_ROOT=${NATURAL_ROOT:-}
TRIAL_ROOT=${TRIAL_ROOT:-/root/autodl-tmp/pseudo10k-trial-mix-local}
OUT=${OUT:-$RESULT_ROOT/Y5-FULL-S-PSEUDO-AUDIT-V1}

mkdir -p "$OUT"
STATUS="$OUT/status.txt"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
for path in "$PYTHON_BIN" "$WEIGHT" "$CONFIG"; do
  test -e "$path"
done

run_domain() {
  local name=$1
  local root=$2
  local domain_out="$OUT/$name"
  mkdir -p "$domain_out"
  printf 'inference:%s\n' "$name" >"$STATUS"
  "$PYTHON_BIN" scripts/run_cv3_oof_pseudo_eval.py \
    --pseudo-root "$root" \
    --config "$CONFIG" \
    --weights "$WEIGHT" "$WEIGHT" "$WEIGHT" \
    --output-dir "$domain_out/inference" \
    >"$domain_out/inference.log" 2>&1
  printf 'frontier:%s\n' "$name" >"$STATUS"
  "$PYTHON_BIN" scripts/analyze_cv3_oof_pseudo_frontier.py \
    --gt "$root/ground_truth.json" \
    --pred "$domain_out/inference/predictions.json" \
    --output "$domain_out/frontier.json" \
    --threshold-step 0.001 \
    >"$domain_out/frontier.log" 2>&1
  "$PYTHON_BIN" scripts/analyze_cv3_oof_pseudo_coarse_thresholds.py \
    --gt "$root/ground_truth.json" \
    --pred "$domain_out/inference/predictions.json" \
    --output "$domain_out/coarse.json" \
    >"$domain_out/coarse.log" 2>&1
}

if [[ -n "$NATURAL_ROOT" ]]; then
  run_domain natural "$NATURAL_ROOT"
fi
run_domain trial "$TRIAL_ROOT"
(
  cd "$OUT"
  sha256sum trial/inference/predictions.json trial/frontier.json \
    trial/coarse.json >SHA256SUMS.txt
  if [[ -n "$NATURAL_ROOT" ]]; then
    sha256sum natural/inference/predictions.json natural/frontier.json \
      natural/coarse.json >>SHA256SUMS.txt
  fi
)
printf '%s\n' complete >"$STATUS"
