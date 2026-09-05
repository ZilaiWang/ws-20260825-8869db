#!/usr/bin/env bash
# One frozen BN candidate on a separate GPU. Never train/full/submit/resume.
set -Eeuo pipefail
PROJECT=${PROJECT:-/root/autodl-tmp/xh-paired-bn-v1}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
DATA=${DATA:-/root/autodl-tmp/data}
BACKGROUND=${BACKGROUND:-/root/autodl-tmp/assets/MACROSHIFT-BACKGROUND-100MP-FROZEN}
BASE=${BASE:-/root/autodl-tmp/results/PAIRED-TREND-REFERENCE-COPY-V1}
OUT=${OUT:-/root/autodl-tmp/results/PAIRED-P40-BN-TRAINONLY-V1}
OPS=${OPS:-/root/autodl-tmp/results/PAIRED-P40-BN-OPS-V1}
BASE_DEPLOY=${BASE_DEPLOY:-/root/autodl-tmp/results/PAIRED-BN-CONTROL-DEPLOYMENT-V1}
BN_DEPLOY=${BN_DEPLOY:-/root/autodl-tmp/results/PAIRED-BN-CANDIDATE-DEPLOYMENT-V1}
test ! -e "$OPS"
test -f "$OUT/plan.json"
test -f "$BASE/evaluation/review.json"
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1

status baseline_development_diagnosis
"$PY" scripts/analyze_paired_development_capacity.py \
  --evaluation "$BASE/evaluation" --output "$OPS/baseline_development_capacity.json"
status bn_training_images_one_pass
"$PY" scripts/run_paired_bn_recalibration.py calibrate --execute \
  --baseline "$BASE" --output "$OUT" --data-root "$DATA" --device cuda:0
status paired_quality_review
"$PY" scripts/run_paired_bn_recalibration.py evaluate --execute \
  --baseline "$BASE" --output "$OUT" --data-root "$DATA" --device cuda:0
status candidate_development_diagnosis
"$PY" scripts/analyze_paired_development_capacity.py \
  --evaluation "$OUT/evaluation" --output "$OPS/candidate_development_capacity.json"
NEXT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["next_action"])' "$OUT/evaluation/review.json")
if [[ "$NEXT" == deployment_regression ]]; then
  # Both endpoints on this same GPU; no cross-server timing comparison shortcut.
  status paired_deployment_control
  "$PY" scripts/run_paired_deployment_regression.py run \
    --data-root "$DATA" --background "$BACKGROUND" \
    --checkpoint "$BASE/adaptation/run/weights/last.pt" --review "$BASE/evaluation" \
    --output "$BASE_DEPLOY" --device cuda:0
  status paired_deployment_candidate
  "$PY" scripts/run_paired_deployment_regression.py run \
    --data-root "$DATA" --background "$BACKGROUND" \
    --checkpoint "$OUT/calibration/bn_last.pt" --review "$OUT/evaluation" \
    --output "$BN_DEPLOY" --device cuda:0
  status complete_candidate_pending_manual_result_analysis
elif [[ "$NEXT" == stop_or_analyze ]]; then
  status complete_candidate_rejected_by_A
else
  status unexpected_quality_action
  exit 2
fi
