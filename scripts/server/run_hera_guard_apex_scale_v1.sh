#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/xh-apex-v1}"
PYTHON="${PYTHON:-/workspace/venvs/p06-cu121/bin/python}"
OUT="${OUT:-/root/autodl-tmp/results/HERA-GUARD-APEX-A2-SCALE-P40-CV3-V1}"
BASE="/root/autodl-tmp/results/HERA-GUARD-APEX-A0-A1-P40-CV3-V1"
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

printf "building_scale_manifest\n" > "$OUT/status.txt"
"$PYTHON" scripts/build_apex_scale_manifest.py \
  --images-csv "$IMAGES" --data-root "$DATA" \
  --output "$OUT/scale_manifest.jsonl" --max-scales-per-object 2
cp "$BASE/manifest/apex_boundary_manifest.jsonl" "$OUT/combined_manifest.jsonl"
cat "$OUT/scale_manifest.jsonl" >> "$OUT/combined_manifest.jsonl"

printf "training_a0_a2_normal\n" > "$OUT/status.txt"
"$PYTHON" scripts/run_apex_boundary_screen.py train-normal \
  --manifest "$OUT/combined_manifest.jsonl" --images-csv "$IMAGES" \
  --predictions "$P40/aggregate/predictions_low.json" --data-root "$DATA" \
  --frontier "$P40/aggregate/crossfit_frontier.json" \
  --p03-root "$P03" --imagenet "$IMAGENET" --arms a0 a2 \
  --output "$OUT/train-normal" --batch-size 96

NORMAL_POSITIVE="$($PYTHON - "$OUT/train-normal/train_normal_summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))["normal_oof"]
print(int(d["a2"]["delta_quality_vs_nms_control"] > 0))
PY
)"
if [[ "$NORMAL_POSITIVE" == "1" ]]; then
  printf "evaluating_hard\n" > "$OUT/status.txt"
  "$PYTHON" scripts/run_apex_boundary_screen.py evaluate-proxy \
    --condition hard --proxy-root /root/autodl-tmp/pseudo10k-trial-mix-local \
    --proxy-predictions "$PROXY/hard/progressive_i1280/predictions.json" \
    --model-root "$OUT/train-normal/models" \
    --frontier "$P40/aggregate/crossfit_frontier.json" \
    --p03-root "$P03" --imagenet "$IMAGENET" --arms a0 a2 \
    --output "$OUT/hard" --batch-size 96
else
  echo "stopped: A2 was not positive on Normal OOF" > "$OUT/hard_skipped.txt"
fi
printf "complete\n" > "$OUT/status.txt"
