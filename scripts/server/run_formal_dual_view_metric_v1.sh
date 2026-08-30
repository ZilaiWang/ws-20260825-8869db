#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/workspace/xh-202625}
PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}
INPUT_ROOT=${INPUT_ROOT:-/workspace/inputs/HERA-GUARD-V3-FORMAL-DUAL-VIEW-V1}
IMAGE_ROOT=${IMAGE_ROOT:-/root/autodl-tmp/data}
IMAGENET_WEIGHT=${IMAGENET_WEIGHT:-/workspace/pretrained/convnext_tiny-983f1562.pth}
OUT=${OUT:-/workspace/results/HERA-GUARD-V3-FORMAL-DUAL-VIEW-V1}

mkdir -p "$OUT"
STATUS="$OUT/status.txt"
cd "$REPO"

printf '%s\n' preflight >"$STATUS"
for path in \
  "$INPUT_ROOT/ground_truth.json" \
  "$INPUT_ROOT/evidence_predictions.json" \
  "$INPUT_ROOT/anchor_predictions.json" \
  "$INPUT_ROOT/summary.json" \
  "$IMAGENET_WEIGHT"; do
  test -s "$path"
done

printf '%s\n' baseline_frontier >"$STATUS"
"$PYTHON_BIN" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "$INPUT_ROOT/ground_truth.json" \
  --pred "$INPUT_ROOT/anchor_predictions.json" \
  --output "$OUT/baseline_frontier.json" \
  --threshold-step 0.001 \
  >"$OUT/baseline_frontier.log" 2>&1

printf '%s\n' training >"$STATUS"
"$PYTHON_BIN" scripts/train_dual_view_metric_verifier.py \
  --gt "$INPUT_ROOT/ground_truth.json" \
  --evidence-pred "$INPUT_ROOT/evidence_predictions.json" \
  --anchor-pred "$INPUT_ROOT/anchor_predictions.json" \
  --pseudo-root "$IMAGE_ROOT" \
  --image-layout relative \
  --imagenet-weight "$IMAGENET_WEIGHT" \
  --output-dir "$OUT/model" \
  --epochs 3 \
  --batches-per-epoch 100 \
  --batch-size 32 \
  --inference-batch-size 96 \
  --resolution 224 \
  --context-ratio 1.75 \
  --residual-limit 2.5 \
  --render-workers 24 \
  --precache-views \
  --device cuda:0 \
  >"$OUT/train.log" 2>&1

printf '%s\n' frontier >"$STATUS"
"$PYTHON_BIN" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "$INPUT_ROOT/ground_truth.json" \
  --pred "$OUT/model/dual_view_oof_predictions.json" \
  --output "$OUT/dual_view_frontier.json" \
  --threshold-step 0.001 \
  >"$OUT/dual_view_frontier.log" 2>&1

"$PYTHON_BIN" scripts/compare_metric_risk_stages.py \
  --baseline "$OUT/baseline_frontier.json" \
  --stage dual_view="$OUT/dual_view_frontier.json" \
  --output "$OUT/decision.json" \
  >"$OUT/decision.log" 2>&1

sha256sum \
  "$INPUT_ROOT/ground_truth.json" \
  "$INPUT_ROOT/evidence_predictions.json" \
  "$INPUT_ROOT/anchor_predictions.json" \
  "$OUT/model/dual_view_oof_predictions.json" \
  "$OUT/baseline_frontier.json" \
  "$OUT/dual_view_frontier.json" \
  "$OUT/decision.json" \
  >"$OUT/SHA256SUMS.txt"
printf '%s\n' complete >"$STATUS"
