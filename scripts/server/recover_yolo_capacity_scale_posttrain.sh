#!/usr/bin/env bash
set -Eeuo pipefail

# Recover only inference/evaluation after a cell has already completed all 40
# training epochs. This script never invokes training and never resumes weights.

CONDITION=${CONDITION:?set CONDITION to s1024, s1280, m1024, or m1280}
PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625}
RESULTS_ROOT=${RESULTS_ROOT:-/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data}
SPLIT_VIEW=${SPLIT_VIEW:-/workspace/results/Y5-ROT90-CV3-OOF/fold_0/split_view.json}
BASE_INFER=${BASE_INFER:-/workspace/results/Y5-ROT90-CV3-OOF/fold_0/resolved_infer.yaml}
FOLD_GT=${FOLD_GT:-/workspace/results/DFINE-M-FOLD0-40EP-V1-R2/coco/instances_val.json}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
export PYTHONPATH="${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"

case "${CONDITION}" in
  s1024|m1024) IMGSZ=1024 ;;
  s1280|m1280) IMGSZ=1280 ;;
  *) printf 'unsupported CONDITION=%s\n' "${CONDITION}" >&2; exit 2 ;;
esac

OUT=${RESULTS_ROOT}/${CONDITION}
STATUS=${OUT}/status.txt
CHECKPOINT=${OUT}/runs/foundation/weights/last.pt
RESULTS=${OUT}/runs/foundation/results.csv
RECOVERY_CONFIG=${OUT}/resolved_infer_recovery.yaml

test -d "${OUT}"
test -f "${CHECKPOINT}"
test -f "${RESULTS}"
test "$(($(wc -l <"${RESULTS}") - 1))" -eq 40
test ! -e "${RECOVERY_CONFIG}"
test ! -e "${OUT}/predictions_low.json"
test ! -e "${OUT}/frontier.json"

failed() {
  code=$?
  printf 'failed_posttrain_recovery_exit_%s\n' "${code}" >"${STATUS}"
  exit "${code}"
}
trap failed ERR

cd "${PROJECT}"
printf 'posttrain_recovery_inference\n' >"${STATUS}"
"${PY}" scripts/materialize_standard_yolo_infer_config.py \
  --base "${BASE_INFER}" --checkpoint "${CHECKPOINT}" \
  --predictions "${OUT}/predictions_low.json" \
  --data-root "${DATA_ROOT}" --split-view "${SPLIT_VIEW}" \
  --output "${RECOVERY_CONFIG}" --imgsz "${IMGSZ}" \
  --batch-size 4 --device cuda:0
"${PY}" scripts/infer_cv3_oof.py --config "${RECOVERY_CONFIG}" \
  >"${OUT}/logs/infer_recovery.log" 2>&1

printf 'posttrain_recovery_evaluating\n' >"${STATUS}"
"${PY}" scripts/analyze_single_split_official_frontier.py \
  --gt "${FOLD_GT}" --pred "${OUT}/predictions_low.json" \
  --output "${OUT}/frontier.json" --step 0.005 \
  >"${OUT}/logs/frontier_recovery.log" 2>&1

trap - ERR
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
