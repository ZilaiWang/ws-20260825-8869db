#!/usr/bin/env bash
set -Eeuo pipefail

# Plan 17: one fold per visible GPU, same-architecture P40 -> hierarchy
# class-24 output-row interpolation, frozen alpha grid and threshold.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
P40_ROOT=${P40_ROOT:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1}
DONOR0=${DONOR0:-/root/autodl-tmp/results/HERA-GUARD-MPSR-HIER-S256V128-FOLD0-40EP-3X4080-B60-V1/training/runs/resolution_adaptation/weights/last.pt}
DONOR12=${DONOR12:-/root/autodl-tmp/results/HERA-GUARD-MPSR-HIER-S256V128-CV3-REMAINING-3X4080-B60-V1}
ASSET0=${ASSET0:-/root/autodl-tmp/capscale-assets}
ASSET12=${ASSET12:-/root/autodl-tmp/capscale-cv3-assets}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-ATTEMPT4-VEHICLE-TASK-VECTOR-CV3-V1}
STATUS=${OUT}/status.txt
THRESHOLD=${THRESHOLD:-0.546}
MODULE_REGEX='model\.23\.(one2one_)?cv3\.[012]\.2'

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
test "$("${PY}" -c 'import torch; print(torch.cuda.device_count())')" -eq 3

fold_gt() {
  if [[ "$1" = 0 ]]; then printf '%s/instances_val.json\n' "${ASSET0}"
  else printf '%s/fold_%s/instances_val.json\n' "${ASSET12}" "$1"; fi
}
donor_checkpoint() {
  if [[ "$1" = 0 ]]; then printf '%s\n' "${DONOR0}"
  else printf '%s/fold_%s/training/runs/resolution_adaptation/weights/last.pt\n' "${DONOR12}" "$1"; fi
}

for fold in 0 1 2; do
  test -f "${P40_ROOT}/fold_${fold}/adaptation/runs/resolution_adaptation/weights/last.pt"
  test -f "${P40_ROOT}/fold_${fold}/resolved_infer.yaml"
  test -f "${P40_ROOT}/fold_${fold}/predictions_low.json"
  test -f "$(fold_gt "${fold}")"
  test -f "$(donor_checkpoint "${fold}")"
done

worker() {
  local fold=$1
  local fold_out="${OUT}/fold_${fold}"
  local base="${P40_ROOT}/fold_${fold}/adaptation/runs/resolution_adaptation/weights/last.pt"
  local donor
  donor=$(donor_checkpoint "${fold}")
  mkdir -p "${fold_out}/logs"
  printf 'merge fold=%s alpha=0\n' "${fold}" >"${fold_out}/progress.txt"
  "${PY}" scripts/merge_yolo_class_task_vector.py \
    --base "${base}" --donor "${donor}" --output "${fold_out}/alpha_0000.pt" \
    --alpha 0 --class-ids 24 --module-regex "${MODULE_REGEX}" \
    >"${fold_out}/logs/merge_alpha_0000.log" 2>&1
  for spec in '0125 0.125' '0250 0.25' '0500 0.5'; do
    set -- ${spec}
    local token=$1 alpha=$2
    printf 'merge fold=%s alpha=%s\n' "${fold}" "${alpha}" >"${fold_out}/progress.txt"
    "${PY}" scripts/merge_yolo_class_task_vector.py \
      --base "${base}" --donor "${donor}" --output "${fold_out}/alpha_${token}.pt" \
      --alpha "${alpha}" --class-ids 24 --module-regex "${MODULE_REGEX}" \
      >"${fold_out}/logs/merge_alpha_${token}.log" 2>&1
    printf 'infer fold=%s alpha=%s\n' "${fold}" "${alpha}" >"${fold_out}/progress.txt"
    "${PY}" scripts/materialize_yolo_cross_resolution_infer_config.py \
      --source "${P40_ROOT}/fold_${fold}/resolved_infer.yaml" \
      --output "${fold_out}/infer_alpha_${token}.yaml" \
      --predictions "${fold_out}/pred_alpha_${token}.json" --imgsz 1280 \
      --checkpoint "${fold_out}/alpha_${token}.pt" \
      >"${fold_out}/logs/config_alpha_${token}.log" 2>&1
    "${PY}" scripts/infer_cv3_oof.py --config "${fold_out}/infer_alpha_${token}.yaml" \
      >"${fold_out}/logs/infer_alpha_${token}.log" 2>&1
  done
  printf 'complete\n' >"${fold_out}/progress.txt"
}

printf 'three_fold_inference\n' >"${STATUS}"
for fold in 0 1 2; do
  CUDA_VISIBLE_DEVICES=${fold} worker "${fold}" &
done
wait

printf 'outer_policy_analysis\n' >"${STATUS}"
"${PY}" - "${OUT}/policy_inputs.json" "${OUT}" "${P40_ROOT}" \
  "${ASSET0}" "${ASSET12}" <<'PY'
import json, sys
from pathlib import Path
output, root, p40, asset0, asset12 = map(Path, sys.argv[1:])
folds=[]
for fold in range(3):
    gt = asset0/'instances_val.json' if fold == 0 else asset12/f'fold_{fold}'/'instances_val.json'
    folds.append({
        'fold': fold,
        'gt': str(gt),
        'baseline': str(p40/f'fold_{fold}'/'predictions_low.json'),
        'candidates': {
            '0': str(p40/f'fold_{fold}'/'predictions_low.json'),
            '0.125': str(root/f'fold_{fold}'/'pred_alpha_0125.json'),
            '0.25': str(root/f'fold_{fold}'/'pred_alpha_0250.json'),
            '0.5': str(root/f'fold_{fold}'/'pred_alpha_0500.json'),
        },
    })
output.write_text(json.dumps({'alphas':[0,0.125,0.25,0.5],'folds':folds},indent=2)+'\n')
PY
"${PY}" scripts/analyze_vehicle_task_vector_policy.py \
  --manifest "${OUT}/policy_inputs.json" \
  --image-groups data/splits/paired_trend_v1/manifest.json \
  --threshold "${THRESHOLD}" --output "${OUT}/policy_result.json" \
  >"${OUT}/policy_analysis.log" 2>&1

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
