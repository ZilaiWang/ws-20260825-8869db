#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/xh-apex-v1}"
PYTHON="${PYTHON:-/workspace/venvs/p06-cu121/bin/python}"
OUT="${OUT:-/root/autodl-tmp/results/HERA-GUARD-APEX-A3-REVIEWED-BACKGROUND-P40-CV3-V1}"
SOURCE="/root/autodl-tmp/results/HERA-GUARD-APEX-A0-A1-P40-CV3-V1"
P40="/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1"
PROXY="/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-FIXED-BENCHMARKS-V1"
BACKGROUND="/root/autodl-tmp/assets/MACROSHIFT-BACKGROUND-100MP-FROZEN"
BACKGROUND_PRED="/root/autodl-tmp/results/SCALEROUTE-PLAN15-BACKGROUND-100MP-V1/expert_i1280.json"
DATA="/root/autodl-tmp/data"
IMAGES="/workspace/N1A/M1-CV3-OOF-aggregate/oof_images.csv"
P03="/root/autodl-tmp/results/P03-FORMAL-CV3-V2"
IMAGENET="/root/autodl-tmp/pretrained/convnext_tiny-983f1562.pth"

[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.log") 2>&1
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed exit=%s\n" "$code" > "$OUT/status.txt"; fi' EXIT
cd "$REPO"

printf "waiting_for_ship_d4\n" > "$OUT/status.txt"
while [[ "$(cat /root/autodl-tmp/results/HERA-GUARD-APEX-A1M-SHIP-D4-P40-CV3-V1/status.txt 2>/dev/null || true)" != "complete" ]]; do
  sleep 20
done

printf "building_reviewed_background_manifest\n" > "$OUT/status.txt"
"$PYTHON" scripts/build_apex_background_manifest.py \
  --background-root "$BACKGROUND" \
  --background-manifest "$BACKGROUND/background_100mp_manifest.jsonl" \
  --review-decision "$BACKGROUND/background_visual_review_decision.json" \
  --predictions "$BACKGROUND_PRED" --images-csv "$IMAGES" \
  --output "$OUT/background_negative_manifest.jsonl"
"$PYTHON" - "$SOURCE/manifest/apex_boundary_manifest.jsonl" \
  "$OUT/background_negative_manifest.jsonl" "$OUT/combined_manifest.jsonl" <<'PY'
import json, sys
base, background, output = sys.argv[1:]
with open(output, "x") as writer:
    for path in (base, background):
        with open(path) as reader:
            for line in reader:
                row = json.loads(line)
                if row["role"] != "jitter_hard_negative":
                    writer.write(json.dumps(row, separators=(",", ":")) + "\n")
PY

printf "training_a0m_a3m_normal\n" > "$OUT/status.txt"
"$PYTHON" scripts/run_apex_boundary_screen.py train-normal \
  --manifest "$OUT/combined_manifest.jsonl" --images-csv "$IMAGES" \
  --predictions "$P40/aggregate/predictions_low.json" --data-root "$DATA" \
  --frontier "$P40/aggregate/crossfit_frontier.json" \
  --p03-root "$P03" --imagenet "$IMAGENET" --arms a0m a3m \
  --output "$OUT/train-normal" --batch-size 96

NORMAL_POSITIVE="$($PYTHON - "$OUT/train-normal/train_normal_summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))["normal_oof"]
print(int(d["a3m"]["delta_quality_vs_nms_control"] > 0))
PY
)"
if [[ "$NORMAL_POSITIVE" != "1" ]]; then
  echo "stopped: A3M was not positive on Normal OOF" > "$OUT/hard_skipped.txt"
  printf "complete\n" > "$OUT/status.txt"
  exit 0
fi

printf "evaluating_hard\n" > "$OUT/status.txt"
"$PYTHON" scripts/run_apex_boundary_screen.py evaluate-proxy \
  --condition hard --proxy-root /root/autodl-tmp/pseudo10k-trial-mix-local \
  --proxy-predictions "$PROXY/hard/progressive_i1280/predictions.json" \
  --model-root "$OUT/train-normal/models" \
  --frontier "$P40/aggregate/crossfit_frontier.json" \
  --p03-root "$P03" --imagenet "$IMAGENET" --arms a0m a3m \
  --output "$OUT/hard" --batch-size 96
printf "complete\n" > "$OUT/status.txt"
