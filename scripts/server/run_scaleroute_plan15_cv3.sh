#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625-capscale}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
FOLD0_ROOT=${FOLD0_ROOT:-/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1}
CONFIRM_ROOT=${CONFIRM_ROOT:-/root/autodl-tmp/results/YOLO-SCALE-CV3-CONFIRM-V1}
ASSET0=${ASSET0:-/root/autodl-tmp/capscale-assets}
ASSET12=${ASSET12:-/root/autodl-tmp/capscale-cv3-assets}
OUT=${OUT:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-CV3-V1}
STATUS=${OUT}/status.txt

if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}/audits" "${OUT}/route" "${OUT}/cross"
exec 9>"${OUT}.lock"
flock -n 9 || exit 4
failed() { code=$?; printf 'failed_exit_%s\n' "${code}" >"${STATUS}"; exit "${code}"; }
trap failed ERR INT TERM
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for path in "${PY}" "${ASSET0}/instances_val.json" \
  "${ASSET12}/fold_1/instances_val.json" "${ASSET12}/fold_2/instances_val.json" \
  "${CONFIRM_ROOT}/s1024/aggregate/ground_truth.json" \
  "${CONFIRM_ROOT}/s1024/aggregate/predictions_low.json" \
  "${CONFIRM_ROOT}/s1280/aggregate/predictions_low.json" \
  "${CONFIRM_ROOT}/s1024/aggregate/crossfit_frontier.json" \
  "${CONFIRM_ROOT}/s1280/aggregate/crossfit_frontier.json"; do
  test -f "${path}"
done

source_dir() {
  local condition=$1 fold=$2
  if [[ "${fold}" = 0 ]]; then
    printf '%s/%s\n' "${FOLD0_ROOT}" "${condition}"
  else
    printf '%s/%s/fold_%s\n' "${CONFIRM_ROOT}" "${condition}" "${fold}"
  fi
}

fold_gt() {
  local fold=$1
  if [[ "${fold}" = 0 ]]; then
    printf '%s/instances_val.json\n' "${ASSET0}"
  else
    printf '%s/fold_%s/instances_val.json\n' "${ASSET12}" "${fold}"
  fi
}

printf 'audit_and_route\n' >"${STATUS}"
for condition in s1024 s1280; do
  for fold in 0 1 2; do
    source=$(source_dir "${condition}" "${fold}")
    for path in "${source}/predictions_low.json" "${source}/resolved_infer.yaml" \
      "${source}/runs/foundation/weights/last.pt"; do
      test -f "${path}"
    done
    "${PY}" scripts/audit_max_det_saturation.py \
      --predictions "${source}/predictions_low.json" --max-det 500 \
      --output "${OUT}/audits/${condition}_fold${fold}_maxdet500.json" >/dev/null
  done
done

"${PY}" scripts/analyze_cv3_class_resolution_route.py \
  --gt "${CONFIRM_ROOT}/s1024/aggregate/ground_truth.json" \
  --primary-pred "${CONFIRM_ROOT}/s1024/aggregate/predictions_low.json" \
  --highres-pred "${CONFIRM_ROOT}/s1280/aggregate/predictions_low.json" \
  --primary-frontier "${CONFIRM_ROOT}/s1024/aggregate/crossfit_frontier.json" \
  --highres-frontier "${CONFIRM_ROOT}/s1280/aggregate/crossfit_frontier.json" \
  --primary-labels 0-23 --highres-labels 24 --fdr-level 0.150 \
  --output "${OUT}/route/s1024_sa_s1280_vehicle.json" \
  >"${OUT}/route/s1024_sa_s1280_vehicle.log" 2>&1

run_cross() {
  local id=$1 source_condition=$2 infer_imgsz=$3 fold=$4
  local source gt out
  source=$(source_dir "${source_condition}" "${fold}")
  gt=$(fold_gt "${fold}")
  out="${OUT}/cross/${id}/fold_${fold}"
  mkdir -p "${out}/logs"
  printf 'cross_%s_fold_%s\n' "${id}" "${fold}" >"${STATUS}"
  "${PY}" scripts/materialize_yolo_cross_resolution_infer_config.py \
    --source "${source}/resolved_infer.yaml" \
    --output "${out}/resolved_infer.yaml" \
    --predictions "${out}/predictions_low.json" --imgsz "${infer_imgsz}"
  "${PY}" scripts/infer_cv3_oof.py --config "${out}/resolved_infer.yaml" \
    >"${out}/logs/infer.log" 2>&1
  "${PY}" scripts/analyze_single_split_official_frontier.py \
    --gt "${gt}" --pred "${out}/predictions_low.json" \
    --output "${out}/frontier.json" --step 0.005 >"${out}/logs/frontier.log" 2>&1
  "${PY}" scripts/audit_max_det_saturation.py \
    --predictions "${out}/predictions_low.json" --max-det 500 \
    --output "${out}/maxdet500.json" >/dev/null
  sha256sum "${source}/runs/foundation/weights/last.pt" \
    "${out}/resolved_infer.yaml" "${out}/predictions_low.json" \
    "${out}/frontier.json" "${out}/maxdet500.json" >"${out}/RESULT_SHA256.txt"
}

for fold in 0 1 2; do run_cross x01 s1024 1280 "${fold}"; done
for fold in 0 1 2; do run_cross x10 s1280 1024 "${fold}"; done

printf 'aggregate\n' >"${STATUS}"
for id in x01 x10; do
  aggregate="${OUT}/cross/${id}/aggregate"
  mkdir -p "${aggregate}"
  "${PY}" scripts/merge_cv3_coco_ledgers.py \
    --gt "${ASSET0}/instances_val.json" "${ASSET12}/fold_1/instances_val.json" \
      "${ASSET12}/fold_2/instances_val.json" \
    --pred "${OUT}/cross/${id}/fold_0/predictions_low.json" \
      "${OUT}/cross/${id}/fold_1/predictions_low.json" \
      "${OUT}/cross/${id}/fold_2/predictions_low.json" \
    --output-gt "${aggregate}/ground_truth.json" \
    --output-pred "${aggregate}/predictions_low.json"
  "${PY}" scripts/analyze_cv3_oof_pseudo_frontier.py \
    --gt "${aggregate}/ground_truth.json" --pred "${aggregate}/predictions_low.json" \
    --output "${aggregate}/crossfit_frontier.json" --threshold-step 0.005 \
    >"${aggregate}/frontier.log" 2>&1
done

"${PY}" scripts/analyze_resolution_cross_matrix.py \
  --t1024-i1024 "${CONFIRM_ROOT}/s1024/aggregate/crossfit_frontier.json" \
  --t1024-i1280 "${OUT}/cross/x01/aggregate/crossfit_frontier.json" \
  --t1280-i1024 "${OUT}/cross/x10/aggregate/crossfit_frontier.json" \
  --t1280-i1280 "${CONFIRM_ROOT}/s1280/aggregate/crossfit_frontier.json" \
  --fdr-level 0.150 --output "${OUT}/resolution_cross_matrix.json" \
  >"${OUT}/resolution_cross_matrix.log" 2>&1

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
