#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/root/autodl-tmp/xh-202625-macroexpert}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}"
ROOT="${ROOT:-/root/autodl-tmp/results/MACROEXPERT-M-V1}"
BASELINE_ROOT="${BASELINE_ROOT:-/workspace/results/Y5-ROT90-CV3-OOF}"
NORMAL_GT="${NORMAL_GT:-/workspace/inputs/HERA-GUARD-V3-FORMAL-DUAL-VIEW-V1/ground_truth.json}"
HARD_ROOT="${HARD_ROOT:-/root/autodl-tmp/pseudo10k-trial-mix-local}"
SENTINEL_ROOT="${SENTINEL_ROOT:-/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1}"
GROUP_MAP="${GROUP_MAP:-${REPO}/tmp/macroshift_group_map.csv}"
CHECKPOINT="${ROOT}/fold0-40ep/runs/foundation/weights/last.pt"
STATUS="${ROOT}/status.txt"

fail() {
  code=$?
  printf 'evaluation_failed:%s\n' "${code}" >"${STATUS}"
  exit "${code}"
}
trap fail ERR

while [[ "$(cat "${STATUS}" 2>/dev/null || true)" = training ]]; do sleep 20; done
test "$(cat "${STATUS}")" = trained
test -f "${CHECKPOINT}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}"
OUT="${ROOT}/evaluation"
mkdir -p "${OUT}/normal/specialist"
printf 'evaluating_normal\n' >"${STATUS}"

"${PYTHON_BIN}" scripts/materialize_macroexpert_infer_config.py \
  --base "${BASELINE_ROOT}/fold_0/resolved_infer.yaml" \
  --checkpoint "${CHECKPOINT}" \
  --predictions "${OUT}/normal/specialist/predictions.json" \
  --output "${OUT}/normal/specialist/resolved_infer.yaml"
"${PYTHON_BIN}" scripts/infer_cv3_oof.py \
  --config "${OUT}/normal/specialist/resolved_infer.yaml" \
  >"${OUT}/normal/specialist/infer.log" 2>&1
"${PYTHON_BIN}" scripts/merge_coco_predictions.py \
  --input "${BASELINE_ROOT}/fold_0/predictions_low.json" \
  --input "${BASELINE_ROOT}/fold_1/predictions_low.json" \
  --input "${BASELINE_ROOT}/fold_2/predictions_low.json" \
  --output "${OUT}/normal/baseline_predictions.json"
"${PYTHON_BIN}" scripts/compose_macroexpert_predictions.py \
  --primary "${OUT}/normal/baseline_predictions.json" \
  --specialist "${OUT}/normal/specialist/predictions.json" \
  --manifest "${REPO}/data/splits/cv3_airport_proxy_k60_v2.json" --fold 0 \
  --output "${OUT}/normal/candidate_predictions.json" \
  >"${OUT}/normal/compose.log"

for side in baseline candidate; do
  "${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
    --gt "${NORMAL_GT}" --pred "${OUT}/normal/${side}_predictions.json" \
    --output "${OUT}/normal/${side}_frontier.json" \
    --threshold-start 0.001 --threshold-stop 0.996 --threshold-step 0.005 \
    --fdr-levels 0.10 0.12 0.15 0.20 \
    >"${OUT}/normal/${side}_frontier.log" 2>&1
  if [[ -f "${GROUP_MAP}" ]]; then
    "${PYTHON_BIN}" scripts/analyze_macro_risk_v2.py \
      --gt "${NORMAL_GT}" --pred "${OUT}/normal/${side}_predictions.json" \
      --group-map "${GROUP_MAP}" --output "${OUT}/normal/${side}_macro_risk_v2.json" \
      --bootstrap-iterations 1000 >"${OUT}/normal/${side}_macro_risk_v2.log" 2>&1
  fi
done

run_proxy() {
  local name=$1 pseudo_root=$2
  local run="${OUT}/${name}"
  mkdir -p "${run}/baseline" "${run}/specialist"
  "${PYTHON_BIN}" scripts/run_multifamily_cv3_pseudo_eval.py \
    --pseudo-root "${pseudo_root}" --family yolo \
    --weights "${BASELINE_ROOT}/fold_0/training/runs/foundation/weights/last.pt" \
      "${BASELINE_ROOT}/fold_1/training/runs/foundation/weights/last.pt" \
      "${BASELINE_ROOT}/fold_2/training/runs/foundation/weights/last.pt" \
    --output-dir "${run}/baseline" --score-floor 0.03 --batch-size 4 --device cuda:0 \
    >"${run}/baseline.log" 2>&1
  "${PYTHON_BIN}" scripts/run_multifamily_cv3_pseudo_eval.py \
    --pseudo-root "${pseudo_root}" --family yolo \
    --weights "${CHECKPOINT}" "${CHECKPOINT}" "${CHECKPOINT}" --folds 0 \
    --output-dir "${run}/specialist" --score-floor 0.03 --batch-size 2 --device cuda:0 \
    --imgsz 1280 --tile-size 1248 --overlap 256 --macroexpert-label-space \
    >"${run}/specialist.log" 2>&1
  "${PYTHON_BIN}" scripts/compose_macroexpert_predictions.py \
    --primary "${run}/baseline/predictions.json" \
    --specialist "${run}/specialist/predictions.json" \
    --ground-truth "${pseudo_root}/ground_truth.json" --fold 0 \
    --output "${run}/candidate_predictions.json" >"${run}/compose.log"
  for side in baseline candidate; do
    local pred="${run}/${side}/predictions.json"
    [[ "${side}" = candidate ]] && pred="${run}/candidate_predictions.json"
    "${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
      --gt "${pseudo_root}/ground_truth.json" --pred "${pred}" \
      --output "${run}/${side}_frontier.json" \
      --threshold-start 0.001 --threshold-stop 0.996 --threshold-step 0.005 \
      --fdr-levels 0.10 0.12 0.15 0.20 >"${run}/${side}_frontier.log" 2>&1
  done
}

printf 'evaluating_hard\n' >"${STATUS}"
run_proxy hard "${HARD_ROOT}"
printf 'evaluating_sentinel\n' >"${STATUS}"
run_proxy sentinel "${SENTINEL_ROOT}"
sha256sum \
  "${OUT}/normal/baseline_frontier.json" "${OUT}/normal/candidate_frontier.json" \
  "${OUT}/hard/baseline_frontier.json" "${OUT}/hard/candidate_frontier.json" \
  "${OUT}/sentinel/baseline_frontier.json" "${OUT}/sentinel/candidate_frontier.json" \
  >"${OUT}/RESULT_SHA256.txt"
printf 'complete\n' >"${STATUS}"
