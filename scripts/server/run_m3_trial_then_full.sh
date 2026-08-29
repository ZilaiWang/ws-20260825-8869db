#!/usr/bin/env bash
# After the admitted Y5-S full fit releases the GPU, first run the short,
# scientifically independent M3 trial-mix test, then start the full M3 fit.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/xh-pre-eval-ab}
PYTHON_BIN=${PYTHON_BIN:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
RESULT_ROOT=${RESULT_ROOT:-/workspace/results}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data}
Y5_STATUS=${Y5_STATUS:-${RESULT_ROOT}/Y5-FULL-S-20260829-R1/status.txt}
PSEUDO_ROOT=${PSEUDO_ROOT:-${RESULT_ROOT}/CV3-OOF-PSEUDO10K-TRIAL-MIX-V1}
M3_OOF_ROOT=${M3_OOF_ROOT:-${RESULT_ROOT}/M3-CV3-OOF}
PSEUDO_OUTPUT=${PSEUDO_OUTPUT:-${RESULT_ROOT}/M3-CV3-PSEUDO10K-TRIAL-MIX-V1}
PSEUDO_STATUS=${PSEUDO_STATUS:-${PSEUDO_OUTPUT}.status.txt}
Y5_PSEUDO_PRED=${Y5_PSEUDO_PRED:-${RESULT_ROOT}/CV3-OOF-PSEUDO-EVAL/y5-safe1024-trial-mix-v1/predictions.json}
MULTI_OER_MODELS=${MULTI_OER_MODELS:-${RESULT_ROOT}/Y5-M3-MULTI-OER-CV3-V3}
MULTI_OER_OUTPUT=${MULTI_OER_OUTPUT:-${RESULT_ROOT}/Y5-M3-MULTI-OER-PSEUDO-TRIAL-V1}
M3_FULL_ROOT=${M3_FULL_ROOT:-${RESULT_ROOT}/M3-FULL-RTDETR-L-20260829-R1}
M3_FULL_STATUS=${M3_FULL_STATUS:-${M3_FULL_ROOT}.status.txt}
M3_ASSET=${M3_ASSET:-/workspace/cv3-model-assets/rtdetr-l.pt}
M3_ASSET_SHA=${M3_ASSET_SHA:-6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f}

while true; do
  state=$(cat "$Y5_STATUS" 2>/dev/null || true)
  if [[ "$state" == complete ]]; then
    break
  fi
  if [[ "$state" == failed:* ]]; then
    printf 'blocked_y5:%s\n' "$state" > "$M3_FULL_STATUS"
    exit 2
  fi
  printf 'waiting_for_y5s:%s\n' "$state" > "$M3_FULL_STATUS"
  sleep 60
done

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src"
if [[ ! -f "${PSEUDO_OUTPUT}/run_summary.json" ]]; then
  printf 'inference\n' > "$PSEUDO_STATUS"
  "$PYTHON_BIN" scripts/run_multifamily_cv3_pseudo_eval.py \
    --pseudo-root "$PSEUDO_ROOT" \
    --family rtdetr \
    --weights \
      "${M3_OOF_ROOT}/fold_0/training/runs/foundation/weights/last.pt" \
      "${M3_OOF_ROOT}/fold_1/training/runs/foundation/weights/last.pt" \
      "${M3_OOF_ROOT}/fold_2/training/runs/foundation/weights/last.pt" \
    --score-floor 0.03 \
    --batch-size 4 \
    --output-dir "$PSEUDO_OUTPUT" \
    > "${PSEUDO_OUTPUT}.inference.log" 2>&1
fi

if [[ ! -f "${PSEUDO_OUTPUT}/frontier.json" ]]; then
  printf 'frontier\n' > "$PSEUDO_STATUS"
  "$PYTHON_BIN" scripts/analyze_cv3_oof_pseudo_frontier.py \
    --gt "${PSEUDO_ROOT}/ground_truth.json" \
    --pred "${PSEUDO_OUTPUT}/predictions.json" \
    --output "${PSEUDO_OUTPUT}/frontier.json" \
    > "${PSEUDO_OUTPUT}.frontier.log" 2>&1
fi

if [[ ! -f "${MULTI_OER_OUTPUT}/frontier.json" ]]; then
  printf 'multi_oer\n' > "$PSEUDO_STATUS"
  "$PYTHON_BIN" scripts/rerank_cv3_pseudo_with_multi_detector_oer.py \
    --y5-predictions "$Y5_PSEUDO_PRED" \
    --m3-predictions "${PSEUDO_OUTPUT}/predictions.json" \
    --model-dir "$MULTI_OER_MODELS" \
    --formal-summary "${MULTI_OER_MODELS}/summary.json" \
    --output-dir "$MULTI_OER_OUTPUT" \
    > "${MULTI_OER_OUTPUT}.rerank.log" 2>&1
  "$PYTHON_BIN" scripts/analyze_cv3_oof_pseudo_frontier.py \
    --gt "${PSEUDO_ROOT}/ground_truth.json" \
    --pred "${MULTI_OER_OUTPUT}/predictions.json" \
    --output "${MULTI_OER_OUTPUT}/frontier.json" \
    > "${MULTI_OER_OUTPUT}.frontier.log" 2>&1
  for level in 0p100 0p120 0p150 0p170 0p200; do
    "$PYTHON_BIN" scripts/evaluate.py \
      --gt "${PSEUDO_ROOT}/ground_truth.json" \
      --pred "${MULTI_OER_OUTPUT}/formal_threshold_${level}.json" \
      --output "${MULTI_OER_OUTPUT}/formal_threshold_${level}.metrics.json" \
      > "${MULTI_OER_OUTPUT}.formal_threshold_${level}.log" 2>&1
  done
fi
printf 'complete\n' > "$PSEUDO_STATUS"

if [[ -e "$M3_FULL_ROOT" ]]; then
  printf 'blocked_existing_output\n' > "$M3_FULL_STATUS"
  exit 3
fi
printf 'training\n' > "$M3_FULL_STATUS"
if "$PYTHON_BIN" scripts/train_full_m3.py \
  --manifest data/splits/cv3_airport_proxy_k60_v2.json \
  --data-root "$DATA_ROOT" \
  --weights "$M3_ASSET" \
  --expected-weight-sha256 "$M3_ASSET_SHA" \
  --output-dir "$M3_FULL_ROOT" > "${M3_FULL_ROOT}.train.log" 2>&1; then
  printf 'complete\n' > "$M3_FULL_STATUS"
else
  code=$?
  printf 'failed:%s\n' "$code" > "$M3_FULL_STATUS"
  exit "$code"
fi
