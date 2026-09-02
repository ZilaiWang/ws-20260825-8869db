#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625-capscale}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
FULL=${FULL:-/root/autodl-tmp/results/Y5-FULL-S1280-3GPU-R1}
ROOT=${ROOT:-/root/autodl-tmp/results/YOLO-SCALE-CV3-CONFIRM-V1}
ASSETS=${ASSETS:-/root/autodl-tmp/capscale-cv3-assets}
STATUS=${ROOT}/controller_status.txt
mkdir -p "${ROOT}"
exec 9>"${ROOT}.lock"
flock -n 9 || exit 4
failed() { code=$?; printf 'failed_exit_%s\n' "${code}" >"${STATUS}"; exit "${code}"; }
trap failed ERR INT TERM
cd "${PROJECT}"

printf 'waiting_for_full\n' >"${STATUS}"
while [[ "$(cat "${FULL}/status.txt" 2>/dev/null || true)" == training* ]]; do
  sleep 30
done
test "$(cat "${FULL}/status.txt")" = complete

printf 'running_confirmatory_folds\n' >"${STATUS}"
CUDA_VISIBLE_DEVICES=0 CONDITION=s1280 FOLD=1 \
  DATA_ROOT=/root/autodl-tmp/capscale-data-s1280-fold1 \
  bash "${PROJECT}/scripts/server/run_yolo_scale_cv3_confirm_condition.sh" &
pid_a=$!
CUDA_VISIBLE_DEVICES=1 CONDITION=s1280 FOLD=2 \
  DATA_ROOT=/root/autodl-tmp/capscale-data-s1280-fold2 \
  bash "${PROJECT}/scripts/server/run_yolo_scale_cv3_confirm_condition.sh" &
pid_b=$!
(
  CUDA_VISIBLE_DEVICES=2 CONDITION=s1024 FOLD=1 \
    DATA_ROOT=/root/autodl-tmp/capscale-data-s1024-fold1 \
    bash "${PROJECT}/scripts/server/run_yolo_scale_cv3_confirm_condition.sh"
  CUDA_VISIBLE_DEVICES=2 CONDITION=s1024 FOLD=2 \
    DATA_ROOT=/root/autodl-tmp/capscale-data-s1024-fold2 \
    bash "${PROJECT}/scripts/server/run_yolo_scale_cv3_confirm_condition.sh"
) &
pid_c=$!
wait "${pid_a}" "${pid_b}" "${pid_c}"

printf 'running_purity_and_aggregate\n' >"${STATUS}"
CUDA_VISIBLE_DEVICES=0 bash "${PROJECT}/scripts/server/run_s1280_purity_fold0_eval.sh"

for condition in s1024 s1280; do
  out="${ROOT}/${condition}/aggregate"
  mkdir -p "${out}"
  fold0=/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1/${condition}
  "${PY}" "${PROJECT}/scripts/merge_cv3_coco_ledgers.py" \
    --gt /root/autodl-tmp/capscale-assets/instances_val.json \
      "${ASSETS}/fold_1/instances_val.json" "${ASSETS}/fold_2/instances_val.json" \
    --pred "${fold0}/predictions_low.json" \
      "${ROOT}/${condition}/fold_1/predictions_low.json" \
      "${ROOT}/${condition}/fold_2/predictions_low.json" \
    --output-gt "${out}/ground_truth.json" --output-pred "${out}/predictions_low.json"
  "${PY}" "${PROJECT}/scripts/analyze_cv3_oof_pseudo_frontier.py" \
    --gt "${out}/ground_truth.json" --pred "${out}/predictions_low.json" \
    --output "${out}/crossfit_frontier.json" --threshold-step 0.005 \
    >"${out}/frontier.log" 2>&1
done
"${PY}" "${PROJECT}/scripts/analyze_paired_fine_error_surface.py" \
  --gt "${ROOT}/s1280/aggregate/ground_truth.json" \
  --baseline "${ROOT}/s1024/aggregate/predictions_low.json" \
  --candidate "${ROOT}/s1280/aggregate/predictions_low.json" \
  --baseline-frontier "${ROOT}/s1024/aggregate/crossfit_frontier.json" \
  --candidate-frontier "${ROOT}/s1280/aggregate/crossfit_frontier.json" \
  --output-json "${ROOT}/paired_cv3_diagnosis.json" \
  --output-csv "${ROOT}/paired_cv3_fine.csv" >"${ROOT}/paired_cv3.log" 2>&1
find "${ROOT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${ROOT}/SHA256SUMS.txt"
trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
