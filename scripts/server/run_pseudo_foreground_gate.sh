#!/usr/bin/env bash
# Fold-heldout proposal-domain foreground gate for the trial-mix pseudo-10K.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 PSEUDO_ROOT BASE_PREDICTIONS CONVNEXT_WEIGHT OUTPUT_DIR" >&2
  exit 2
fi

PSEUDO_ROOT="$(realpath "$1")"
BASE_PREDICTIONS="$(realpath "$2")"
CONVNEXT_WEIGHT="$(realpath "$3")"
OUTPUT_DIR="$4"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
FREEZE="${FREEZE:-freeze_backbone}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BATCHES_PER_EPOCH="${BATCHES_PER_EPOCH:-120}"
GT="$PSEUDO_ROOT/ground_truth.json"
STATUS="$OUTPUT_DIR/status.txt"
MANIFEST="$OUTPUT_DIR/foreground_manifest.csv"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
GATE_PREDICTIONS="$OUTPUT_DIR/foreground_blend_predictions.json"
OER_DIR="$OUTPUT_DIR/foreground_oer"

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$OER_DIR"
trap 'printf "failed\n" > "$STATUS"' ERR
printf "build_manifest\n" > "$STATUS"

"$PYTHON" scripts/build_cv3_pseudo_foreground_manifest.py \
  --gt "$GT" \
  --pred "$BASE_PREDICTIONS" \
  --pseudo-root "$PSEUDO_ROOT" \
  --output "$MANIFEST" \
  --summary "$OUTPUT_DIR/manifest_summary.json" \
  --negative-iou 0.05 \
  --context-ratio 1.25 \
  > "$OUTPUT_DIR/manifest.log" 2>&1

for fold in 0 1 2; do
  printf "train_fold_%s\n" "$fold" > "$STATUS"
  "$PYTHON" scripts/train_bg_gate.py \
    --manifest "$MANIFEST" \
    --data-root "$PSEUDO_ROOT" \
    --convnext-weights "$CONVNEXT_WEIGHT" \
    --output-dir "$CHECKPOINT_DIR" \
    --held-out-fold "$fold" \
    --freeze "$FREEZE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --batches-per-epoch "$BATCHES_PER_EPOCH" \
    --learning-rate 0.001 \
    --seed 202625 \
    --resolution 224 \
    --device "$DEVICE" \
    --verify-weight-sha256 \
    > "$OUTPUT_DIR/train_fold${fold}.log" 2>&1
done

printf "infer_foreground\n" > "$STATUS"
"$PYTHON" scripts/rerank_cv3_pseudo_with_foreground_gate.py \
  --gt "$GT" \
  --pred "$BASE_PREDICTIONS" \
  --pseudo-root "$PSEUDO_ROOT" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --imagenet-weight "$CONVNEXT_WEIGHT" \
  --output "$GATE_PREDICTIONS" \
  --summary "$OUTPUT_DIR/foreground_inference_summary.json" \
  --alpha 0.50 \
  --context-ratio 1.25 \
  --resolution 224 \
  --batch-size 256 \
  --device "$DEVICE" \
  > "$OUTPUT_DIR/foreground_inference.log" 2>&1

printf "fit_foreground_oer\n" > "$STATUS"
"$PYTHON" scripts/train_pseudo_foreground_oer.py \
  --ground-truth "$GT" \
  --foreground-predictions "$GATE_PREDICTIONS" \
  --output-dir "$OER_DIR" \
  --nms-iou 0.70 \
  > "$OUTPUT_DIR/foreground_oer.log" 2>&1

printf "evaluate\n" > "$STATUS"
"$PYTHON" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "$GT" \
  --pred "$GATE_PREDICTIONS" \
  --output "$OUTPUT_DIR/foreground_blend_frontier.json" \
  > "$OUTPUT_DIR/foreground_blend_frontier.log" 2>&1
"$PYTHON" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "$GT" \
  --pred "$OER_DIR/foreground_oer_predictions.json" \
  --output "$OUTPUT_DIR/foreground_oer_frontier.json" \
  > "$OUTPUT_DIR/foreground_oer_frontier.log" 2>&1

printf "complete\n" > "$STATUS"
trap - ERR
