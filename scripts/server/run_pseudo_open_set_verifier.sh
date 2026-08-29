#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 9 ]]; then
  echo "usage: $0 REPO PYTHON GT PRED PSEUDO_ROOT MANIFEST IMAGENET P03_DIR OUTPUT" >&2
  exit 2
fi

REPO=$1
PYTHON=$2
GT=$3
PRED=$4
PSEUDO_ROOT=$5
MANIFEST=$6
IMAGENET=$7
P03_DIR=$8
OUTPUT=$9
EPOCHS=${OPENSET_EPOCHS:-3}
BATCHES_PER_EPOCH=${OPENSET_BATCHES_PER_EPOCH:-40}
CONTEXT_RATIO=${OPENSET_CONTEXT_RATIO:-1.25}

cd "$REPO"
mkdir -p "$OUTPUT/checkpoints"
printf 'training\n' >"$OUTPUT/status.txt"

for fold in 0 1 2; do
  printf 'train_fold_%s\n' "$fold" >"$OUTPUT/status.txt"
  "$PYTHON" scripts/train_pseudo_open_set_verifier.py \
    --manifest "$MANIFEST" \
    --pseudo-root "$PSEUDO_ROOT" \
    --imagenet-weight "$IMAGENET" \
    --p03-checkpoint "$P03_DIR/p03_fold${fold}.pt" \
    --output "$OUTPUT/checkpoints/open_set_fold${fold}.pt" \
    --held-out-fold "$fold" \
    --regime last_stage \
    --epochs "$EPOCHS" \
    --batch-size 64 \
    --batches-per-epoch "$BATCHES_PER_EPOCH" \
    --backbone-lr 0.00002 \
    --head-lr 0.0002 \
    --resolution 224 \
    --device cuda:0 \
    >"$OUTPUT/train_fold${fold}.log" 2>&1
done

printf 'inference\n' >"$OUTPUT/status.txt"
"$PYTHON" scripts/rerank_cv3_pseudo_with_open_set_verifier.py \
  --gt "$GT" \
  --pred "$PRED" \
  --pseudo-root "$PSEUDO_ROOT" \
  --checkpoint-dir "$OUTPUT/checkpoints" \
  --imagenet-weight "$IMAGENET" \
  --output "$OUTPUT/open_set_blend_predictions.json" \
  --summary "$OUTPUT/inference_summary.json" \
  --alpha 0.50 \
  --context-ratio "$CONTEXT_RATIO" \
  --resolution 224 \
  --batch-size 192 \
  --device cuda:0 \
  >"$OUTPUT/inference.log" 2>&1

"$PYTHON" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "$GT" --pred "$OUTPUT/open_set_blend_predictions.json" \
  --output "$OUTPUT/open_set_blend_frontier.json" \
  >"$OUTPUT/open_set_blend_frontier.log" 2>&1

printf 'pixel_oer\n' >"$OUTPUT/status.txt"
"$PYTHON" scripts/train_pseudo_pixel_oer.py \
  --ground-truth "$GT" \
  --crop-predictions "$OUTPUT/open_set_blend_predictions.json" \
  --output-dir "$OUTPUT/pixel_oer" \
  --nms-iou 0.60 \
  >"$OUTPUT/pixel_oer.log" 2>&1
for variant in identity dual_hypothesis; do
  "$PYTHON" scripts/analyze_cv3_oof_pseudo_frontier.py \
    --gt "$GT" --pred "$OUTPUT/pixel_oer/${variant}_predictions.json" \
    --output "$OUTPUT/pixel_oer/${variant}_frontier.json" \
    >"$OUTPUT/pixel_oer/${variant}_frontier.log" 2>&1
done

printf 'complete\n' >"$OUTPUT/status.txt"
