#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625-capscale}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
FOLD0_ROOT=${FOLD0_ROOT:-/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1/s1024}
CONFIRM_ROOT=${CONFIRM_ROOT:-/root/autodl-tmp/results/YOLO-SCALE-CV3-CONFIRM-V1/s1024}
ASSET0=${ASSET0:-/root/autodl-tmp/capscale-assets}
ASSET12=${ASSET12:-/root/autodl-tmp/capscale-cv3-assets}
OUT=${OUT:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE-CV3-V1}
STATUS=${OUT}/status.txt

if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}"
exec 9>"${OUT}.lock"
flock -n 9 || exit 4
failed() { code=$?; printf 'failed_exit_%s\n' "${code}" >"${STATUS}"; exit "${code}"; }
trap failed ERR INT TERM
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"

source_dir() {
  local fold=$1
  if [[ "${fold}" = 0 ]]; then printf '%s\n' "${FOLD0_ROOT}"
  else printf '%s/fold_%s\n' "${CONFIRM_ROOT}" "${fold}"
  fi
}

fold_gt() {
  local fold=$1
  if [[ "${fold}" = 0 ]]; then printf '%s/instances_val.json\n' "${ASSET0}"
  else printf '%s/fold_%s/instances_val.json\n' "${ASSET12}" "${fold}"
  fi
}

for fold in 0 1 2; do
  source=$(source_dir "${fold}")
  for path in "${source}/runs/foundation/weights/last.pt" \
    "${source}/dataset.yaml" "${source}/resolved_infer.yaml" "$(fold_gt "${fold}")"; do
    test -f "${path}"
  done
done

for fold in 0 1 2; do
  source=$(source_dir "${fold}")
  fold_out="${OUT}/fold_${fold}"
  train_out="${fold_out}/adaptation"
  mkdir -p "${fold_out}/logs"
  printf 'train_fold_%s\n' "${fold}" >"${STATUS}"
  "${PY}" scripts/train_progressive_resolution_adaptation.py \
    --weights "${source}/runs/foundation/weights/last.pt" \
    --data "${source}/dataset.yaml" --output "${train_out}" \
    --epochs 20 --imgsz 1280 --batch 8 --workers 4 --device cuda:0 \
    --seed 42 --lr0 0.0002 --lrf 0.10 --rotate90-p 1.0 \
    >"${fold_out}/logs/train.log" 2>&1

  checkpoint="${train_out}/runs/resolution_adaptation/weights/last.pt"
  test -f "${checkpoint}"
  printf 'infer_fold_%s\n' "${fold}" >"${STATUS}"
  "${PY}" scripts/materialize_yolo_cross_resolution_infer_config.py \
    --source "${source}/resolved_infer.yaml" \
    --output "${fold_out}/resolved_infer.yaml" \
    --predictions "${fold_out}/predictions_low.json" --imgsz 1280 \
    --checkpoint "${checkpoint}"
  "${PY}" scripts/infer_cv3_oof.py --config "${fold_out}/resolved_infer.yaml" \
    >"${fold_out}/logs/infer.log" 2>&1
  "${PY}" scripts/analyze_single_split_official_frontier.py \
    --gt "$(fold_gt "${fold}")" --pred "${fold_out}/predictions_low.json" \
    --output "${fold_out}/frontier.json" --step 0.005 \
    >"${fold_out}/logs/frontier.log" 2>&1
done

printf 'aggregate\n' >"${STATUS}"
mkdir -p "${OUT}/aggregate"
"${PY}" scripts/merge_cv3_coco_ledgers.py \
  --gt "${ASSET0}/instances_val.json" "${ASSET12}/fold_1/instances_val.json" \
    "${ASSET12}/fold_2/instances_val.json" \
  --pred "${OUT}/fold_0/predictions_low.json" "${OUT}/fold_1/predictions_low.json" \
    "${OUT}/fold_2/predictions_low.json" \
  --output-gt "${OUT}/aggregate/ground_truth.json" \
  --output-pred "${OUT}/aggregate/predictions_low.json"
"${PY}" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "${OUT}/aggregate/ground_truth.json" \
  --pred "${OUT}/aggregate/predictions_low.json" \
  --output "${OUT}/aggregate/crossfit_frontier.json" --threshold-step 0.005 \
  >"${OUT}/aggregate/frontier.log" 2>&1

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
