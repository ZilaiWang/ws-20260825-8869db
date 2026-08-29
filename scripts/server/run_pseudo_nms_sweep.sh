#!/usr/bin/env bash
set -euo pipefail

# Deterministic class-aware NMS sensitivity audit over a frozen pseudo-10K
# candidate ledger.  This is a diagnostic only: it never changes detector
# weights or uses GT-derived features at inference.

if [[ $# -lt 6 ]]; then
  echo "usage: $0 REPO PYTHON GT OUTPUT_ROOT LABEL=JSON [LABEL=JSON ...]" >&2
  exit 2
fi

REPO=$1
PYTHON=$2
GT=$3
OUTPUT_ROOT=$4
shift 4
SOURCES=("$@")

cd "$REPO"
mkdir -p "$OUTPUT_ROOT"

for threshold in 0.40 0.50 0.60 0.65 0.70; do
  tag=${threshold/./p}
  output="$OUTPUT_ROOT/nms_${tag}"
  mkdir -p "$output"
  source_args=()
  for source in "${SOURCES[@]}"; do
    source_args+=(--source "$source")
  done
  "$PYTHON" scripts/merge_pseudo_candidate_sources.py \
    "${source_args[@]}" \
    --nms-iou "$threshold" \
    --output "$output/predictions.json" \
    --summary "$output/merge_summary.json" \
    >"$output/merge.log" 2>&1
  "$PYTHON" scripts/analyze_cv3_oof_pseudo_frontier.py \
    --gt "$GT" \
    --pred "$output/predictions.json" \
    --output "$output/frontier.json" \
    >"$output/frontier.log" 2>&1
  printf 'complete\n' >"$output/status.txt"
done

printf 'complete\n' >"$OUTPUT_ROOT/status.txt"
