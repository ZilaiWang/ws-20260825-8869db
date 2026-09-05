#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=/root/autodl-tmp/xh-p40-aircraft-consistency-v1
PY=/root/autodl-tmp/venvs/cv3-model-cu121/bin/python
OUT=/root/autodl-tmp/results/P40-AIRCRAFT-VIEW-CONSISTENCY-V1
P40=/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1
BASE=/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-FIXED-BENCHMARKS-V1
CE=/root/autodl-tmp/results/P40-AIRCRAFT-CE-D4-V1
exec 9>"$OUT.lock"
flock -n 9 || exit 4
test ! -e "$OUT"
mkdir "$OUT"
trap 'code=$?; printf "failed_exit_%s\n" "$code" >"$OUT/status.txt"; exit "$code"' ERR INT TERM
cd "$PROJECT"
export PYTHONPATH=src OMP_NUM_THREADS=4
for condition in hard sentinel; do
  if [[ "$condition" = hard ]]; then
    ROOT=/root/autodl-tmp/pseudo10k-trial-mix-local
    REF="$CE/comparison.json"
  else
    ROOT=/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1
    REF="$CE/sentinel/comparison.json"
  fi
  printf 'running_%s\n' "$condition" >"$OUT/status.txt"
  "$PY" -u scripts/run_p40_aircraft_ce_d4.py --condition "$condition" \
    --config configs/experiments/p40_aircraft_view_consistency_v1.json \
    --reference-comparison "$REF" \
    --pseudo-root "$ROOT" --pred "$BASE/$condition/progressive_i1280/predictions.json" \
    --frontier "$P40/aggregate/crossfit_frontier.json" \
    --classifier-root /workspace/results/R1-5-AIRCRAFT-VIEW-CONSISTENCY/train/view_consistency \
    --imagenet /root/autodl-tmp/pretrained/convnext_tiny-983f1562.pth \
    --output "$OUT/$condition" >"$OUT/$condition.log" 2>&1
  pass=$("$PY" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["review"]["direction_pass"]))' "$OUT/$condition/comparison.json")
  if [[ "$pass" != 1 ]]; then
    printf 'complete_rejected_by_%s_vs_ce_d4\n' "$condition" >"$OUT/status.txt"
    exit 0
  fi
done
printf 'complete_positive_increment_vs_ce_d4_review_required\n' >"$OUT/status.txt"
trap - ERR INT TERM
