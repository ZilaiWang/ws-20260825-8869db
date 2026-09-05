#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/xh-apex-v1}"
PYTHON="${PYTHON:-/workspace/venvs/p06-cu121/bin/python}"
OUT="${OUT:-/root/autodl-tmp/results/HERA-GUARD-APEX-DINOV2B-P40-CV3-V1}"
SOURCE="/root/autodl-tmp/results/HERA-GUARD-APEX-A0-A1-P40-CV3-V1"
P40="/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1"
PROXY="/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-FIXED-BENCHMARKS-V1"
DATA="/root/autodl-tmp/data"
IMAGES="/workspace/N1A/M1-CV3-OOF-aggregate/oof_images.csv"
P03="/root/autodl-tmp/results/P03-FORMAL-CV3-V2"
IMAGENET="/root/autodl-tmp/pretrained/convnext_tiny-983f1562.pth"
DINO_REPO="/root/autodl-tmp/p04-reuse/repos/dinov2"
DINO_WEIGHTS="/root/autodl-tmp/p04-reuse/weights/dinov2_vitb14_pretrain.pth"
DINO_SHA="0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"

[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.log") 2>&1
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed exit=%s\n" "$code" > "$OUT/status.txt"; fi' EXIT
cd "$REPO"

COMMON=(
  --frontier "$P40/aggregate/crossfit_frontier.json"
  --p03-root "$P03" --imagenet "$IMAGENET" --arms a1m
  --feature-backbone dinov2b --dinov2-repo "$DINO_REPO"
  --dinov2-weights "$DINO_WEIGHTS" --dinov2-weight-sha256 "$DINO_SHA"
  --batch-size 64
)

printf "training_dinov2b_a1m_normal\n" > "$OUT/status.txt"
"$PYTHON" scripts/run_apex_boundary_screen.py train-normal \
  --manifest "$SOURCE/manifest/apex_boundary_manifest.jsonl" \
  --images-csv "$IMAGES" --predictions "$P40/aggregate/predictions_low.json" \
  --data-root "$DATA" "${COMMON[@]}" --output "$OUT/train-normal"

NORMAL_POSITIVE="$($PYTHON - "$OUT/train-normal/train_normal_summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))["normal_oof"]["a1m"]
print(int(d["delta_quality_vs_nms_control"] > 0))
PY
)"
if [[ "$NORMAL_POSITIVE" != "1" ]]; then
  echo "stopped: DINOv2-B A1M was not positive on Normal" > "$OUT/hard_skipped.txt"
  printf "complete\n" > "$OUT/status.txt"
  exit 0
fi

for CONDITION in hard sentinel; do
  printf "evaluating_%s\n" "$CONDITION" > "$OUT/status.txt"
  if [[ "$CONDITION" == "hard" ]]; then
    ROOT=/root/autodl-tmp/pseudo10k-trial-mix-local
  else
    ROOT=/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1
  fi
  "$PYTHON" scripts/run_apex_boundary_screen.py evaluate-proxy \
    --condition "$CONDITION" --proxy-root "$ROOT" \
    --proxy-predictions "$PROXY/$CONDITION/progressive_i1280/predictions.json" \
    --model-root "$OUT/train-normal/models" "${COMMON[@]}" \
    --output "$OUT/$CONDITION"
  POSITIVE="$($PYTHON - "$OUT/$CONDITION/comparison.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))["comparison"]["a1m"]
print(int(d["delta_quality_vs_nms_control"] > 0))
PY
)"
  if [[ "$POSITIVE" != "1" ]]; then
    echo "stopped: DINOv2-B A1M was not positive on $CONDITION" > "$OUT/${CONDITION}_gate_failed.txt"
    break
  fi
done
printf "complete\n" > "$OUT/status.txt"
