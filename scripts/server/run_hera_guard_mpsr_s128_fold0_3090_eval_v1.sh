#!/usr/bin/env bash
set -Eeuo pipefail

# Fixed 3090 evaluation for the train-only three-GPU S128 candidate.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
EXPECTED_GPU_REGEX=${EXPECTED_GPU_REGEX:-RTX 3090}
TRAIN_ROOT=${TRAIN_ROOT:-/root/autodl-tmp/results/HERA-GUARD-MPSR-S128-FOLD0-40EP-3X4080-B60-V1}
P40=${P40:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1/fold_0}
GT=${GT:-/root/autodl-tmp/capscale-assets/instances_val.json}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-MPSR-S128-FOLD0-3090-EVAL-V1}
STATUS=${OUT}/status.txt
CHECKPOINT=${TRAIN_ROOT}/training/runs/resolution_adaptation/weights/last.pt

if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}/logs"
exec 9>"${OUT}.lock"
flock -n 9 || exit 4
failed() { code=$?; printf 'failed_exit_%s\n' "${code}" >"${STATUS}"; exit "${code}"; }
trap failed ERR INT TERM
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for path in "${CHECKPOINT}" "${P40}/resolved_infer.yaml" \
  "${P40}/predictions_low.json" "${GT}"; do
  test -f "${path}"
done
test "$("${PY}" -c 'import torch; print(torch.cuda.device_count())')" -ge 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -Eq "${EXPECTED_GPU_REGEX}"

printf 'infer_fold0_on_3090\n' >"${STATUS}"
"${PY}" scripts/materialize_yolo_cross_resolution_infer_config.py \
  --source "${P40}/resolved_infer.yaml" --output "${OUT}/resolved_infer.yaml" \
  --predictions "${OUT}/predictions_low.json" --imgsz 1280 --checkpoint "${CHECKPOINT}"
"${PY}" scripts/infer_cv3_oof.py --config "${OUT}/resolved_infer.yaml" \
  >"${OUT}/logs/infer.log" 2>&1

printf 'paired_fixed_evaluation\n' >"${STATUS}"
"${PY}" scripts/evaluate_fixed_score_threshold.py --gt "${GT}" \
  --pred "${P40}/predictions_low.json" --threshold 0.546 \
  --output "${OUT}/baseline_fixed_0546.json"
"${PY}" scripts/evaluate_fixed_score_threshold.py --gt "${GT}" \
  --pred "${OUT}/predictions_low.json" --threshold 0.546 \
  --output "${OUT}/candidate_fixed_0546.json"
"${PY}" scripts/compare_candidate_trend.py \
  --baseline "${OUT}/baseline_fixed_0546.json" \
  --candidate "${OUT}/candidate_fixed_0546.json" \
  --output "${OUT}/paired_comparison.json"
"${PY}" scripts/analyze_single_split_official_frontier.py \
  --gt "${GT}" --pred "${OUT}/predictions_low.json" \
  --output "${OUT}/candidate_frontier.json" --step 0.005 \
  >"${OUT}/logs/frontier.log" 2>&1
"${PY}" scripts/analyze_single_split_official_frontier.py \
  --gt "${GT}" --pred "${P40}/predictions_low.json" \
  --output "${OUT}/baseline_frontier.json" --step 0.005 \
  >"${OUT}/logs/baseline_frontier.log" 2>&1
"${PY}" scripts/triage_detector_candidate.py \
  --baseline-fixed "${OUT}/baseline_fixed_0546.json" \
  --candidate-fixed "${OUT}/candidate_fixed_0546.json" \
  --baseline-frontier "${OUT}/baseline_frontier.json" \
  --candidate-frontier "${OUT}/candidate_frontier.json" \
  --output "${OUT}/candidate_triage.json"

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
