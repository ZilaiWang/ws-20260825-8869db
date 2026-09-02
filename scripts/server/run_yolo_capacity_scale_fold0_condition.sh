#!/usr/bin/env bash
set -Eeuo pipefail

# One frozen cell of the 2x2 S/M x 1024/1280 Y5 fold0 screen.
# Launch each condition exactly once, optionally on separate physical GPUs by
# setting CUDA_VISIBLE_DEVICES before this script. Inside the isolated process
# the contract always addresses cuda:0.

CONDITION=${CONDITION:?set CONDITION to s1024, s1280, m1024, or m1280}
PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625}
RESULTS_ROOT=${RESULTS_ROOT:-/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data}
SPLIT_VIEW=${SPLIT_VIEW:-/workspace/results/Y5-ROT90-CV3-OOF/fold_0/split_view.json}
BASE_INFER=${BASE_INFER:-/workspace/results/Y5-ROT90-CV3-OOF/fold_0/resolved_infer.yaml}
FOLD_GT=${FOLD_GT:-/workspace/results/DFINE-M-FOLD0-40EP-V1-R2/coco/instances_val.json}
WEIGHTS_ROOT=${WEIGHTS_ROOT:-/root/autodl-tmp/pretrained}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
export PYTHONPATH="${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"

SPLIT_SHA=a647ce030fa832aadc6a6c286a3f6464ac1783f71797a52cc598ec340f128943
GT_SHA=2641d3bb15388b9a19812ab514b993d5f68ef90d7a59fb02834bf7903e585977
S_WEIGHT_SHA=646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
M_WEIGHT_SHA=401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7

case "${CONDITION}" in
  s1024)
    TEMPLATE=${PROJECT}/configs/experiments/s25_yolo26s_1024_fold0_40ep.yaml
    WEIGHTS=${WEIGHTS_ROOT}/yolo26s.pt
    WEIGHT_SHA=${S_WEIGHT_SHA}
    IMGSZ=1024
    ;;
  s1280)
    TEMPLATE=${PROJECT}/configs/experiments/s25_yolo26s_1280_fold0_40ep.yaml
    WEIGHTS=${WEIGHTS_ROOT}/yolo26s.pt
    WEIGHT_SHA=${S_WEIGHT_SHA}
    IMGSZ=1280
    ;;
  m1024)
    TEMPLATE=${PROJECT}/configs/experiments/m25_yolo26m_1024_fold0_40ep.yaml
    WEIGHTS=${WEIGHTS_ROOT}/yolo26m.pt
    WEIGHT_SHA=${M_WEIGHT_SHA}
    IMGSZ=1024
    ;;
  m1280)
    TEMPLATE=${PROJECT}/configs/experiments/m25_yolo26m_1280_fold0_40ep.yaml
    WEIGHTS=${WEIGHTS_ROOT}/yolo26m.pt
    WEIGHT_SHA=${M_WEIGHT_SHA}
    IMGSZ=1280
    ;;
  *)
    printf 'unsupported CONDITION=%s\n' "${CONDITION}" >&2
    exit 2
    ;;
esac

OUT=${RESULTS_ROOT}/${CONDITION}
STATUS=${OUT}/status.txt
if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}/logs"
failed() {
  code=$?
  printf 'failed_exit_%s\n' "${code}" >"${STATUS}"
  exit "${code}"
}
trap failed ERR

printf 'preflight\n' >"${STATUS}"
for path in "${TEMPLATE}" "${WEIGHTS}" "${SPLIT_VIEW}" "${BASE_INFER}" "${FOLD_GT}"; do
  test -f "${path}"
done
test -d "${DATA_ROOT}"
test "$(sha256sum "${WEIGHTS}" | awk '{print $1}')" = "${WEIGHT_SHA}"
test "$(sha256sum "${SPLIT_VIEW}" | awk '{print $1}')" = "${SPLIT_SHA}"
test "$(sha256sum "${FOLD_GT}" | awk '{print $1}')" = "${GT_SHA}"
"${PY}" - <<'PY'
import torch
assert torch.cuda.is_available()
print({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
PY

cd "${PROJECT}"
"${PY}" scripts/materialize_yolo_capacity_scale_config.py \
  --template "${TEMPLATE}" --output "${OUT}/train_contract.yaml" \
  --output-dir "${OUT}" --data-root "${DATA_ROOT}" \
  --split-view "${SPLIT_VIEW}" --weights "${WEIGHTS}" --device cuda:0

printf 'training\n' >"${STATUS}"
"${PY}" scripts/train_cv3_oof.py \
  --config "${OUT}/train_contract.yaml" --innovation y5 --rotate90-p 1.0 \
  >"${OUT}/logs/train.log" 2>&1
CHECKPOINT=${OUT}/runs/foundation/weights/last.pt
RESULTS=${OUT}/runs/foundation/results.csv
test -f "${CHECKPOINT}"
test "$(($(wc -l <"${RESULTS}") - 1))" -eq 40

printf 'inference\n' >"${STATUS}"
"${PY}" scripts/materialize_standard_yolo_infer_config.py \
  --base "${BASE_INFER}" --checkpoint "${CHECKPOINT}" \
  --predictions "${OUT}/predictions_low.json" \
  --data-root "${DATA_ROOT}" --split-view "${SPLIT_VIEW}" \
  --output "${OUT}/resolved_infer.yaml" --imgsz "${IMGSZ}" \
  --batch-size 4 --device cuda:0
"${PY}" scripts/infer_cv3_oof.py --config "${OUT}/resolved_infer.yaml" \
  >"${OUT}/logs/infer.log" 2>&1

printf 'evaluating\n' >"${STATUS}"
"${PY}" scripts/analyze_single_split_official_frontier.py \
  --gt "${FOLD_GT}" --pred "${OUT}/predictions_low.json" \
  --output "${OUT}/frontier.json" --step 0.005 \
  >"${OUT}/logs/frontier.log" 2>&1

trap - ERR
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
