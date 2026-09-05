#!/usr/bin/env bash
# GPU-free wait on the existing reference lock, then exactly one anchor audit.
set -Eeuo pipefail
PROJECT=${PROJECT:-/root/autodl-tmp/xh-paired-trend-v1}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
BASE=${BASE:-/root/autodl-tmp/results/PAIRED-TREND-BASELINE-V1}
DATA=${DATA:-/root/autodl-tmp/data}
REFERENCE_OPS=${REFERENCE_OPS:-/root/autodl-tmp/results/PAIRED-TREND-REFERENCE-OPS-V1}
AUDIT=${AUDIT:-/root/autodl-tmp/results/PAIRED-TREND-ANCHOR-AUDIT-V1}
OPS=${OPS:-/root/autodl-tmp/results/PAIRED-TREND-ANCHOR-OPS-V1}
test -f "$AUDIT/contract.json"
test -f "$REFERENCE_OPS/chain.lock"
test ! -e "$OPS"
mkdir -p "$OPS"
exec 8>"$OPS/chain.lock"
flock -n 8
exec > >(tee -a "$OPS/main.log") 2>&1
status() { printf '%s\n' "$1" > "$OPS/status.txt"; date -Is; printf '%s\n' "$1"; }
failed() { code=$?; status "failed_exit_${code}"; exit "$code"; }
trap failed ERR
trap 'status interrupted; exit 130' INT TERM
status waiting_for_reference_and_regression
exec 9<"$REFERENCE_OPS/chain.lock"
flock -x 9
if [[ "$(<"$REFERENCE_OPS/status.txt")" != complete_reference_and_regression ]]; then
  status dependency_not_complete_preserved
  exit 1
fi
cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
status evaluating_declared_foundation_anchor
"$PY" scripts/run_paired_anchor_audit.py run --execute \
  --base "$BASE" --data-root "$DATA" --output "$AUDIT" --device cuda:0
status complete_anchor_audit
