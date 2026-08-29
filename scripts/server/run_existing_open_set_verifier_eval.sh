#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 9 ]]; then
  echo "usage: $0 REPO PYTHON GT PRED PSEUDO_ROOT CHECKPOINT_DIR IMAGENET OUTPUT CONTEXT_RATIO" >&2
  exit 2
fi

REPO=$1
PYTHON=$2
GT=$3
PRED=$4
PSEUDO_ROOT=$5
CHECKPOINT_DIR=$6
IMAGENET=$7
OUTPUT=$8
CONTEXT_RATIO=$9

mkdir -p "$OUTPUT"
cd "$REPO"

for fold in 0 1 2; do
  test -s "$CHECKPOINT_DIR/open_set_fold${fold}.pt"
done

printf 'inference\n' >"$OUTPUT/status.txt"
"$PYTHON" scripts/rerank_cv3_pseudo_with_open_set_verifier.py \
  --gt "$GT" \
  --pred "$PRED" \
  --pseudo-root "$PSEUDO_ROOT" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
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
  --gt "$GT" \
  --pred "$OUTPUT/open_set_blend_predictions.json" \
  --output "$OUTPUT/open_set_blend_frontier.json" \
  >"$OUTPUT/open_set_blend_frontier.log" 2>&1

printf 'pixel_oer\n' >"$OUTPUT/status.txt"
"$PYTHON" scripts/train_pseudo_pixel_oer.py \
  --ground-truth "$GT" \
  --crop-predictions "$OUTPUT/open_set_blend_predictions.json" \
  --output-dir "$OUTPUT/pixel_oer" \
  --nms-iou 0.60 \
  >"$OUTPUT/pixel_oer.log" 2>&1

for variant in identity direct dual_hypothesis; do
  pred_path="$OUTPUT/pixel_oer/${variant}_predictions.json"
  if [[ -s "$pred_path" ]]; then
    "$PYTHON" scripts/analyze_cv3_oof_pseudo_frontier.py \
      --gt "$GT" \
      --pred "$pred_path" \
      --output "$OUTPUT/pixel_oer/${variant}_frontier.json" \
      >"$OUTPUT/pixel_oer/${variant}_frontier.log" 2>&1
  fi
done

(
  cd "$OUTPUT"
  find . -type f ! -name SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum >SHA256SUMS.txt
)
printf 'complete\n' >"$OUTPUT/status.txt"
