#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/xh-apex-v1}"
PYTHON="${PYTHON:-/workspace/venvs/p06-cu121/bin/python}"
OUT="${OUT:-/root/autodl-tmp/results/HERA-GUARD-APEX-A1M-SHIP-D4-P40-CV3-V1}"
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

printf "waiting_for_identity_ship_gate\n" > "$OUT/status.txt"
while [[ "$(cat /root/autodl-tmp/results/HERA-GUARD-APEX-A1M-SHIP-ONLY-P40-CV3-V1/status.txt 2>/dev/null || true)" != "complete" ]]; do
  sleep 20
done

"$PYTHON" - "$SOURCE/manifest/apex_boundary_manifest.jsonl" "$OUT/ship_manifest.jsonl" <<'PY'
import json, sys
source, output = sys.argv[1:]
with open(source) as reader, open(output, "x") as writer:
    for line in reader:
        row = json.loads(line)
        if row["coarse"] == "ship":
            writer.write(json.dumps(row, separators=(",", ":")) + "\n")
PY

printf "training_ship_d4_normal\n" > "$OUT/status.txt"
"$PYTHON" scripts/run_apex_boundary_screen.py train-normal \
  --manifest "$OUT/ship_manifest.jsonl" --images-csv "$IMAGES" \
  --predictions "$P40/aggregate/predictions_low.json" --data-root "$DATA" \
  --frontier "$P40/aggregate/crossfit_frontier.json" \
  --p03-root "$P03" --imagenet "$IMAGENET" --arms a0 a1m \
  --target-coarse ship --view-mode d4 \
  --output "$OUT/train-normal" --batch-size 48

NORMAL_POSITIVE="$($PYTHON - "$OUT/train-normal/train_normal_summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))["normal_oof"]
print(int(d["a1m"]["delta_quality_vs_nms_control"] > 0))
PY
)"
if [[ "$NORMAL_POSITIVE" != "1" ]]; then
  echo "stopped: D4 A1M Ship was not positive on Normal OOF" > "$OUT/hard_skipped.txt"
  printf "complete\n" > "$OUT/status.txt"
  exit 0
fi

printf "evaluating_hard\n" > "$OUT/status.txt"
"$PYTHON" scripts/run_apex_boundary_screen.py evaluate-proxy \
  --condition hard --proxy-root /root/autodl-tmp/pseudo10k-trial-mix-local \
  --proxy-predictions "$PROXY/hard/progressive_i1280/predictions.json" \
  --model-root "$OUT/train-normal/models" \
  --frontier "$P40/aggregate/crossfit_frontier.json" \
  --p03-root "$P03" --imagenet "$IMAGENET" --arms a0 a1m \
  --target-coarse ship --view-mode d4 \
  --output "$OUT/hard" --batch-size 48

HARD_POSITIVE="$($PYTHON - "$OUT/hard/comparison.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))["comparison"]
print(int(d["a1m"]["delta_quality_vs_nms_control"] > 0))
PY
)"
if [[ "$HARD_POSITIVE" != "1" ]]; then
  echo "skipped: D4 A1M Ship was not positive on Hard" > "$OUT/sentinel_skipped.txt"
  printf "complete\n" > "$OUT/status.txt"
  exit 0
fi

printf "evaluating_sentinel\n" > "$OUT/status.txt"
"$PYTHON" scripts/run_apex_boundary_screen.py evaluate-proxy \
  --condition sentinel --proxy-root /root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1 \
  --proxy-predictions "$PROXY/sentinel/progressive_i1280/predictions.json" \
  --model-root "$OUT/train-normal/models" \
  --frontier "$P40/aggregate/crossfit_frontier.json" \
  --p03-root "$P03" --imagenet "$IMAGENET" --arms a0 a1m \
  --target-coarse ship --view-mode d4 \
  --output "$OUT/sentinel" --batch-size 48
printf "complete\n" > "$OUT/status.txt"
