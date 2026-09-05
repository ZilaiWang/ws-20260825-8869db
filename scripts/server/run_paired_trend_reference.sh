#!/usr/bin/env bash
# One-time single-GPU reference bootstrap. No full-data fit, resume or submission.
set -Eeuo pipefail
PROJECT=${PROJECT:-/root/autodl-tmp/xh-paired-trend-v1}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
DATA=${DATA:-/root/autodl-tmp/data}
WEIGHTS=${WEIGHTS:-/workspace/cv3-model-assets/yolo26s.pt}
BACKGROUND=${BACKGROUND:-/root/autodl-tmp/assets/MACROSHIFT-BACKGROUND-100MP-FROZEN}
BASE=${BASE:-/root/autodl-tmp/results/PAIRED-TREND-BASELINE-V1}
OPS=${OPS:-/root/autodl-tmp/results/PAIRED-TREND-REFERENCE-OPS-V1}
DEPLOY=${DEPLOY:-/root/autodl-tmp/results/PAIRED-TREND-BASELINE-DEPLOYMENT-V1}
test ! -e "$OPS"
mkdir -p "$OPS"
exec 9>"$OPS/chain.lock"
flock -n 9
exec > >(tee -a "$OPS/main.log") 2>&1
status() { printf '%s\n' "$1" > "$OPS/status.txt"; date -Is; printf '%s\n' "$1"; }
failed() { code=$?; status "failed_exit_${code}"; exit "$code"; }
trap failed ERR
trap 'status interrupted; exit 130' INT TERM
cd "$PROJECT"
export PYTHONPATH="$PROJECT:$PROJECT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
status baseline_160_plus_40_and_quality_review
"$PY" scripts/run_paired_trend.py baseline --execute \
  --data-root "$DATA" --weights "$WEIGHTS" --output "$BASE" --device cuda:0
status background_and_100mp_regression
"$PY" scripts/run_paired_deployment_regression.py run \
  --data-root "$DATA" --background "$BACKGROUND" \
  --checkpoint "$BASE/adaptation/run/weights/last.pt" --review "$BASE/evaluation" \
  --output "$DEPLOY" --device cuda:0
status complete_reference_and_regression
