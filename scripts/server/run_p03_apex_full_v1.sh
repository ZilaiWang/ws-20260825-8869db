#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/xh-apex-v1}"
PYTHON="${PYTHON:-/workspace/venvs/p06-cu121/bin/python}"
OUT="${OUT:-/root/autodl-tmp/results/P03-APEX-FULL-V1}"
DINO="/root/autodl-tmp/results/HERA-GUARD-APEX-DINOV2B-P40-CV3-V1"

[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
mkdir -p "$OUT"
printf "waiting_for_dinov2_screen\n" > "$OUT/status.txt"
while [[ "$(cat "$DINO/status.txt" 2>/dev/null || true)" != "complete" ]]; do
  sleep 20
done

# The trainer itself refuses an existing output, so stage logs outside it.
rm "$OUT/status.txt"
rmdir "$OUT"
cd "$REPO"
exec "$PYTHON" scripts/train_crop_classifier_full.py \
  --config configs/experiments/p03_apex_full_v1.yaml \
  --manifest /workspace/N1A/formal_crop_manifest.csv \
  --data-root /root/autodl-tmp/data \
  --weights /root/autodl-tmp/pretrained/convnext_tiny-983f1562.pth \
  --output "$OUT"
