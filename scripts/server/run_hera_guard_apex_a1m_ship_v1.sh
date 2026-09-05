#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/xh-apex-v1}"
PYTHON="${PYTHON:-/workspace/venvs/p06-cu121/bin/python}"
OUT="${OUT:-/root/autodl-tmp/results/HERA-GUARD-APEX-A1M-SHIP-ONLY-P40-CV3-V1}"
SOURCE="/root/autodl-tmp/results/HERA-GUARD-APEX-A0-A1-P40-CV3-V1"
P40="/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1"
PROXY="/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-FIXED-BENCHMARKS-V1"
DATA="/root/autodl-tmp/data"
IMAGES="/workspace/N1A/M1-CV3-OOF-aggregate/oof_images.csv"
P03="/root/autodl-tmp/results/P03-FORMAL-CV3-V2"
IMAGENET="/root/autodl-tmp/pretrained/convnext_tiny-983f1562.pth"

[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.log") 2>&1
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed exit=%s\n" "$code" > "$OUT/status.txt"; fi' EXIT
cd "$REPO"

printf "refitting_frozen_a1m_ship_normal\n" > "$OUT/status.txt"
"$PYTHON" scripts/run_apex_boundary_screen.py refit-normal \
  --manifest "$SOURCE/manifest/apex_boundary_manifest.jsonl" \
  --images-csv "$IMAGES" --predictions "$P40/aggregate/predictions_low.json" \
  --data-root "$DATA" --source-feature-root "$SOURCE/train-normal" \
  --frontier "$P40/aggregate/crossfit_frontier.json" \
  --p03-root "$P03" --imagenet "$IMAGENET" --arms a1m --target-coarse ship \
  --output "$OUT/train-normal" --batch-size 96

printf "evaluating_hard\n" > "$OUT/status.txt"
"$PYTHON" scripts/run_apex_boundary_screen.py evaluate-proxy \
  --condition hard --proxy-root /root/autodl-tmp/pseudo10k-trial-mix-local \
  --proxy-predictions "$PROXY/hard/progressive_i1280/predictions.json" \
  --model-root "$OUT/train-normal/models" \
  --frontier "$P40/aggregate/crossfit_frontier.json" \
  --p03-root "$P03" --imagenet "$IMAGENET" --arms a1m --target-coarse ship \
  --output "$OUT/hard" --batch-size 96

printf "evaluating_sentinel\n" > "$OUT/status.txt"
"$PYTHON" scripts/run_apex_boundary_screen.py evaluate-proxy \
  --condition sentinel --proxy-root /root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1 \
  --proxy-predictions "$PROXY/sentinel/progressive_i1280/predictions.json" \
  --model-root "$OUT/train-normal/models" \
  --frontier "$P40/aggregate/crossfit_frontier.json" \
  --p03-root "$P03" --imagenet "$IMAGENET" --arms a1m --target-coarse ship \
  --output "$OUT/sentinel" --batch-size 96
printf "complete\n" > "$OUT/status.txt"
