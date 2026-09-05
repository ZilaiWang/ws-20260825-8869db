#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT=/root/autodl-tmp/xh-p40-aircraft-identity-v1
PY=/root/autodl-tmp/venvs/cv3-model-cu121/bin/python
OUT=/root/autodl-tmp/results/P40-AIRCRAFT-CE-IDENTITY-V1
NORMAL=/root/autodl-tmp/results/P40-AIRCRAFT-CE-D4-NORMAL-V1
P40=/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1
BASE=/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-FIXED-BENCHMARKS-V1
exec 9>"$OUT.lock"
flock -n 9 || exit 4
test ! -e "$OUT"
mkdir "$OUT"
trap 'code=$?; printf "failed_exit_%s\n" "$code" >"$OUT/status.txt"; exit "$code"' ERR INT TERM
printf 'waiting_existing_normal_diagnostic\n' >"$OUT/status.txt"
# Wait only for the existing owned diagnostic, with a bounded failure exit.
for attempt in $(seq 1 120); do
  if [[ -s "$NORMAL/comparison.json" ]]; then break; fi
  if ! pgrep -f 'python -u scripts/run_p40_aircraft_ce_d4.py --condition normal_diagnostic' >/dev/null; then
    printf 'normal diagnostic exited without result\n' >&2
    exit 5
  fi
  sleep 10
done
test -s "$NORMAL/comparison.json"
cd "$PROJECT"
export PYTHONPATH=src
export OMP_NUM_THREADS=4
for condition in hard sentinel; do
  if [[ "$condition" = hard ]]; then ROOT=/root/autodl-tmp/pseudo10k-trial-mix-local
  else ROOT=/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1; fi
  printf 'running_%s\n' "$condition" >"$OUT/status.txt"
  "$PY" -u scripts/run_p40_aircraft_ce_d4.py --condition "$condition" \
    --config configs/experiments/p40_aircraft_ce_identity_v1.json \
    --pseudo-root "$ROOT" --pred "$BASE/$condition/progressive_i1280/predictions.json" \
    --frontier "$P40/aggregate/crossfit_frontier.json" \
    --classifier-root /workspace/results/R1-1-AIRCRAFT-PROPOSAL-REFINEMENT/train/ce \
    --imagenet /root/autodl-tmp/pretrained/convnext_tiny-983f1562.pth \
    --output "$OUT/$condition" >"$OUT/$condition.log" 2>&1
  pass=$("$PY" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["review"]["direction_pass"]))' "$OUT/$condition/comparison.json")
  if [[ "$pass" != 1 ]]; then
    printf 'complete_rejected_by_%s\n' "$condition" >"$OUT/status.txt"
    exit 0
  fi
done
printf 'complete_positive_cost_ablation_review_required\n' >"$OUT/status.txt"
trap - ERR INT TERM
