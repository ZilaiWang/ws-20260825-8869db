#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 8 ]]; then
  echo "usage: $0 REPO PYTHON PSEUDO_ROOT CONFIG FOLD0_WEIGHT FOLD1_WEIGHT FOLD2_WEIGHT OUTPUT" >&2
  exit 2
fi

REPO=$1
PYTHON=$2
PSEUDO_ROOT=$3
CONFIG=$4
FOLD0_WEIGHT=$5
FOLD1_WEIGHT=$6
FOLD2_WEIGHT=$7
OUTPUT=$8
STATUS="$OUTPUT/status.txt"

cd "$REPO"
mkdir -p "$OUTPUT"
trap 'printf "failed\n" >"$STATUS"' ERR
printf 'audit_inputs\n' >"$STATUS"
for path in "$PSEUDO_ROOT/ground_truth.json" "$CONFIG" \
  "$FOLD0_WEIGHT" "$FOLD1_WEIGHT" "$FOLD2_WEIGHT"; do
  test -f "$path"
done
sha256sum "$CONFIG" "$FOLD0_WEIGHT" "$FOLD1_WEIGHT" "$FOLD2_WEIGHT" \
  >"$OUTPUT/input_sha256.txt"

printf 'inference\n' >"$STATUS"
"$PYTHON" scripts/run_cv3_oof_pseudo_eval.py \
  --pseudo-root "$PSEUDO_ROOT" \
  --config "$CONFIG" \
  --weights "$FOLD0_WEIGHT" "$FOLD1_WEIGHT" "$FOLD2_WEIGHT" \
  --output-dir "$OUTPUT/inference" \
  >"$OUTPUT/inference.log" 2>&1

printf 'frontier\n' >"$STATUS"
"$PYTHON" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "$PSEUDO_ROOT/ground_truth.json" \
  --pred "$OUTPUT/inference/predictions.json" \
  --output "$OUTPUT/frontier.json" \
  >"$OUTPUT/frontier.log" 2>&1

printf 'complete\n' >"$STATUS"
trap - ERR
