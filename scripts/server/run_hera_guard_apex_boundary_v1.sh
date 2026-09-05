#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/xh-apex-v1}"
PYTHON="${PYTHON:-/workspace/venvs/p06-cu121/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-/root/autodl-tmp/results/HERA-GUARD-APEX-A0-A1-P40-CV3-V1}"
P40="/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1"
P40_PROXY="/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-FIXED-BENCHMARKS-V1"
DATA="/root/autodl-tmp/data"
IMAGES_CSV="/workspace/N1A/M1-CV3-OOF-aggregate/oof_images.csv"
P03="/root/autodl-tmp/results/P03-FORMAL-CV3-V2"
IMAGENET="/root/autodl-tmp/pretrained/convnext_tiny-983f1562.pth"
MANIFEST_DIR="$RESULT_ROOT/manifest"
TRAIN_DIR="$RESULT_ROOT/train-normal"

if [[ -e "$RESULT_ROOT" ]]; then
  echo "refusing to overwrite $RESULT_ROOT" >&2
  exit 2
fi
mkdir -p "$RESULT_ROOT"
exec > >(tee -a "$RESULT_ROOT/driver.log") 2>&1
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed exit=%s\n" "$code" > "$RESULT_ROOT/status.txt"; fi' EXIT
printf "preflight\n" > "$RESULT_ROOT/status.txt"
cd "$REPO"

for path in "$PYTHON" "$DATA" "$IMAGES_CSV" "$P03" "$IMAGENET" \
  "$P40/aggregate/predictions_low.json" "$P40/aggregate/crossfit_frontier.json" \
  "$P40_PROXY/hard/progressive_i1280/predictions.json" \
  "$P40_PROXY/sentinel/progressive_i1280/predictions.json"; do
  [[ -e "$path" ]] || { echo "missing asset: $path" >&2; exit 3; }
done

mkdir -p "$RESULT_ROOT/audit"
sha256sum \
  configs/experiments/hera_guard_apex_boundary_v1.json \
  scripts/build_apex_boundary_manifest.py \
  scripts/run_apex_boundary_screen.py \
  src/rsdet/augmentation/apex_boundary.py \
  src/rsdet/augmentation/jitter_hard_negative.py \
  src/rsdet/models/apex_boundary.py \
  "$IMAGES_CSV" "$P40/aggregate/predictions_low.json" \
  "$P40/aggregate/crossfit_frontier.json" "$IMAGENET" \
  > "$RESULT_ROOT/audit/input_and_code_sha256.txt"
nvidia-smi -q > "$RESULT_ROOT/audit/nvidia_smi_preflight.txt"

printf "building_manifest\n" > "$RESULT_ROOT/status.txt"
"$PYTHON" scripts/build_apex_boundary_manifest.py \
  --images-csv "$IMAGES_CSV" \
  --predictions "$P40/aggregate/predictions_low.json" \
  --data-root "$DATA" \
  --output "$MANIFEST_DIR" \
  --ship-floor 0.003 --vehicle-floor 0.001 --jitter-per-object 3

printf "training_and_normal_oof\n" > "$RESULT_ROOT/status.txt"
"$PYTHON" scripts/run_apex_boundary_screen.py train-normal \
  --manifest "$MANIFEST_DIR/apex_boundary_manifest.jsonl" \
  --images-csv "$IMAGES_CSV" \
  --predictions "$P40/aggregate/predictions_low.json" \
  --data-root "$DATA" \
  --frontier "$P40/aggregate/crossfit_frontier.json" \
  --p03-root "$P03" --imagenet "$IMAGENET" \
  --output "$TRAIN_DIR" --batch-size 96

printf "evaluating_hard\n" > "$RESULT_ROOT/status.txt"
"$PYTHON" scripts/run_apex_boundary_screen.py evaluate-proxy \
  --condition hard \
  --proxy-root /root/autodl-tmp/pseudo10k-trial-mix-local \
  --proxy-predictions "$P40_PROXY/hard/progressive_i1280/predictions.json" \
  --model-root "$TRAIN_DIR/models" \
  --frontier "$P40/aggregate/crossfit_frontier.json" \
  --p03-root "$P03" --imagenet "$IMAGENET" \
  --output "$RESULT_ROOT/hard" --batch-size 96

HARD_POSITIVE="$($PYTHON - "$RESULT_ROOT/hard/comparison.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))["comparison"]
print(int(any(d[a]["delta_quality_vs_nms_control"] > 0 for a in ("a0", "a1"))))
PY
)"
if [[ "$HARD_POSITIVE" == "1" ]]; then
  printf "evaluating_sentinel\n" > "$RESULT_ROOT/status.txt"
  "$PYTHON" scripts/run_apex_boundary_screen.py evaluate-proxy \
    --condition sentinel \
    --proxy-root /root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1 \
    --proxy-predictions "$P40_PROXY/sentinel/progressive_i1280/predictions.json" \
    --model-root "$TRAIN_DIR/models" \
    --frontier "$P40/aggregate/crossfit_frontier.json" \
    --p03-root "$P03" --imagenet "$IMAGENET" \
    --output "$RESULT_ROOT/sentinel" --batch-size 96
else
  printf "skipped_no_positive_hard_direction\n" > "$RESULT_ROOT/sentinel_skipped.txt"
fi

find "$RESULT_ROOT" -type f ! -name '*.npy' ! -name '*.joblib' -print0 \
  | sort -z | xargs -0 sha256sum > "$RESULT_ROOT/audit/result_sha256.txt"
printf "complete\n" > "$RESULT_ROOT/status.txt"
echo "APEX screen complete"
