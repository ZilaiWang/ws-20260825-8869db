#!/usr/bin/env bash
set -Eeuo pipefail

CONDITION=${CONDITION:?set CONDITION=s1024 or s1280}
FOLD=${FOLD:?set FOLD=1 or 2}
PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625-capscale}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
ASSET_ROOT=${ASSET_ROOT:-/root/autodl-tmp/capscale-cv3-assets}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/capscale-data-${CONDITION}-fold${FOLD}}
RESULTS_ROOT=${RESULTS_ROOT:-/root/autodl-tmp/results/YOLO-SCALE-CV3-CONFIRM-V1}
WEIGHTS=${WEIGHTS:-/root/autodl-tmp/capscale-assets/yolo26s.pt}
OUT=${RESULTS_ROOT}/${CONDITION}/fold_${FOLD}
STATUS=${OUT}/status.txt

WEIGHT_SHA=646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
LABEL_TREE_SHA=929d73218df463629eca8d0f66becd032a742029df310b4b8c38c167b64a2128
case "${FOLD}" in
  1)
    SPLIT_SHA=eb9353b0fc68d5175df1aac140e4a012f6938adac9b1a02a464501110c188670
    GT_SHA=89833d8746a9614ba9cd6cec47efe453ba2125f6b8843256d46d2828d476f169
    ;;
  2)
    SPLIT_SHA=99445fdd81d7f25a33b5642744e6804e664850ed6b0f516ce7c9d1a2adfe003f
    GT_SHA=921689a78feb1e5c8a0453ad09155a37038c39abe41bece494af5122b7251892
    ;;
  *) printf 'FOLD must be 1 or 2\n' >&2; exit 2 ;;
esac
case "${CONDITION}" in
  s1024) TEMPLATE=${PROJECT}/configs/experiments/s25_yolo26s_1024_fold0_40ep.yaml; IMGSZ=1024 ;;
  s1280) TEMPLATE=${PROJECT}/configs/experiments/s25_yolo26s_1280_fold0_40ep.yaml; IMGSZ=1280 ;;
  *) printf 'CONDITION must be s1024 or s1280\n' >&2; exit 2 ;;
esac
SPLIT_VIEW=${ASSET_ROOT}/fold_${FOLD}/split_view.json
FOLD_GT=${ASSET_ROOT}/fold_${FOLD}/instances_val.json
BASE_INFER=${ASSET_ROOT}/base_infer.yaml

if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}/logs"
failed() { code=$?; printf 'failed_exit_%s\n' "${code}" >"${STATUS}"; exit "${code}"; }
trap failed ERR INT TERM
for path in "${PY}" "${WEIGHTS}" "${SPLIT_VIEW}" "${FOLD_GT}" "${BASE_INFER}" "${DATA_ROOT}/images" "${DATA_ROOT}/labels"; do
  test -e "${path}"
done
test "$(sha256sum "${WEIGHTS}" | awk '{print $1}')" = "${WEIGHT_SHA}"
test "$(sha256sum "${SPLIT_VIEW}" | awk '{print $1}')" = "${SPLIT_SHA}"
test "$(sha256sum "${FOLD_GT}" | awk '{print $1}')" = "${GT_SHA}"

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"
printf 'training\n' >"${STATUS}"
"${PY}" scripts/materialize_yolo_capacity_scale_config.py \
  --template "${TEMPLATE}" --output "${OUT}/train_contract.yaml" \
  --output-dir "${OUT}" --data-root "${DATA_ROOT}" \
  --split-view "${SPLIT_VIEW}" --weights "${WEIGHTS}" --device cuda:0
"${PY}" scripts/train_cv3_oof.py --config "${OUT}/train_contract.yaml" \
  --innovation y5 --rotate90-p 1.0 >"${OUT}/logs/train.log" 2>&1
test "$(($(wc -l <"${OUT}/runs/foundation/results.csv") - 1))" -eq 40

printf 'inference\n' >"${STATUS}"
"${PY}" scripts/materialize_standard_yolo_infer_config.py \
  --base "${BASE_INFER}" --checkpoint "${OUT}/runs/foundation/weights/last.pt" \
  --predictions "${OUT}/predictions_low.json" --data-root "${DATA_ROOT}" \
  --split-view "${SPLIT_VIEW}" --output "${OUT}/resolved_infer.yaml" \
  --imgsz "${IMGSZ}" --batch-size 4 --device cuda:0
"${PY}" scripts/infer_cv3_oof.py --config "${OUT}/resolved_infer.yaml" \
  >"${OUT}/logs/infer.log" 2>&1
"${PY}" scripts/analyze_single_split_official_frontier.py \
  --gt "${FOLD_GT}" --pred "${OUT}/predictions_low.json" \
  --output "${OUT}/frontier.json" --step 0.005 >"${OUT}/logs/frontier.log" 2>&1

sha256sum "${OUT}/runs/foundation/weights/last.pt" "${OUT}/predictions_low.json" \
  "${OUT}/frontier.json" >"${OUT}/RESULT_SHA256.txt"
trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
