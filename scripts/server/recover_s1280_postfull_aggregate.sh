#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625-capscale}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
ROOT=${ROOT:-/root/autodl-tmp/results/YOLO-SCALE-CV3-CONFIRM-V1}
ASSETS=${ASSETS:-/root/autodl-tmp/capscale-cv3-assets}
PURITY_OUT=${PURITY_OUT:-/root/autodl-tmp/results/S1280-COARSE-PURITY-FOLD0-V1-R2}
STATUS=${ROOT}/recovery_status.txt
mkdir -p "${ROOT}"
exec 9>"${ROOT}.recovery.lock"
flock -n 9 || exit 4
failed() { code=$?; printf 'failed_exit_%s\n' "${code}" >"${STATUS}"; exit "${code}"; }
trap failed ERR INT TERM
cd "${PROJECT}"

test "$(cat /root/autodl-tmp/results/Y5-FULL-S1280-3GPU-R1/status.txt)" = complete
for condition in s1024 s1280; do
  for fold in 1 2; do
    test "$(cat "${ROOT}/${condition}/fold_${fold}/status.txt")" = complete
  done
done

printf 'purity_r2\n' >"${STATUS}"
CUDA_VISIBLE_DEVICES=0 OUT="${PURITY_OUT}" \
  bash scripts/server/run_s1280_purity_fold0_eval.sh

printf 'aggregate\n' >"${STATUS}"
for condition in s1024 s1280; do
  out="${ROOT}/${condition}/aggregate"
  if [[ -e "${out}" ]]; then
    printf 'refusing pre-existing aggregate: %s\n' "${out}" >&2
    exit 3
  fi
  mkdir -p "${out}"
  fold0=/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1/${condition}
  "${PY}" scripts/merge_cv3_coco_ledgers.py \
    --gt /root/autodl-tmp/capscale-assets/instances_val.json \
      "${ASSETS}/fold_1/instances_val.json" "${ASSETS}/fold_2/instances_val.json" \
    --pred "${fold0}/predictions_low.json" \
      "${ROOT}/${condition}/fold_1/predictions_low.json" \
      "${ROOT}/${condition}/fold_2/predictions_low.json" \
    --output-gt "${out}/ground_truth.json" --output-pred "${out}/predictions_low.json"
  "${PY}" scripts/analyze_cv3_oof_pseudo_frontier.py \
    --gt "${out}/ground_truth.json" --pred "${out}/predictions_low.json" \
    --output "${out}/crossfit_frontier.json" --threshold-step 0.005 \
    >"${out}/frontier.log" 2>&1
done
"${PY}" scripts/analyze_paired_fine_error_surface.py \
  --gt "${ROOT}/s1280/aggregate/ground_truth.json" \
  --baseline "${ROOT}/s1024/aggregate/predictions_low.json" \
  --candidate "${ROOT}/s1280/aggregate/predictions_low.json" \
  --baseline-frontier "${ROOT}/s1024/aggregate/crossfit_frontier.json" \
  --candidate-frontier "${ROOT}/s1280/aggregate/crossfit_frontier.json" \
  --output-json "${ROOT}/paired_cv3_diagnosis.json" \
  --output-csv "${ROOT}/paired_cv3_fine.csv" >"${ROOT}/paired_cv3.log" 2>&1
trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${ROOT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${ROOT}/SHA256SUMS.txt"
