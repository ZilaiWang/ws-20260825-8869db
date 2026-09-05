#!/usr/bin/env bash
set -Eeuo pipefail

# RTX 3090 inference and CV3 OOF aggregation for the class-disjoint
# Ship256/Vehicle128 specialist. Training is performed elsewhere; this driver
# only consumes three completed 40-epoch checkpoints. P40 owns fine 0..23 and
# the candidate owns fine 24 exactly, so Ship/Aircraft remain bitwise stable.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
P40_ROOT=${P40_ROOT:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1}
FOLD0_EVAL=${FOLD0_EVAL:-/root/autodl-tmp/results/HERA-GUARD-MPSR-HIER-S256V128-FOLD0-3090-EVAL-V1}
CAND0=${CAND0:-/root/autodl-tmp/results/HERA-GUARD-MPSR-HIER-S256V128-FOLD0-40EP-3X4080-B60-V1/training/runs/resolution_adaptation/weights/last.pt}
CAND12=${CAND12:-/root/autodl-tmp/results/HERA-GUARD-MPSR-HIER-S256V128-CV3-REMAINING-3X4080-B60-V1}
ASSET0=${ASSET0:-/root/autodl-tmp/capscale-assets}
ASSET12=${ASSET12:-/root/autodl-tmp/capscale-cv3-assets}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-MPSR-HIER-VEHICLE-ROUTE-CV3-OOF-3090-V1}
STATUS=${OUT}/status.txt
THRESHOLD=${THRESHOLD:-0.546}

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

test "$("${PY}" -c 'import torch; print(torch.cuda.device_count())')" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'RTX 3090'

fold_gt() {
  local fold=$1
  if [[ "${fold}" = 0 ]]; then printf '%s/instances_val.json\n' "${ASSET0}"
  else printf '%s/fold_%s/instances_val.json\n' "${ASSET12}" "${fold}"; fi
}

candidate_checkpoint() {
  local fold=$1
  if [[ "${fold}" = 0 ]]; then printf '%s\n' "${CAND0}"
  else printf '%s/fold_%s/training/runs/resolution_adaptation/weights/last.pt\n' "${CAND12}" "${fold}"; fi
}

for fold in 0 1 2; do
  p40="${P40_ROOT}/fold_${fold}"
  checkpoint=$(candidate_checkpoint "${fold}")
  for path in "${p40}/resolved_infer.yaml" "${p40}/predictions_low.json" \
    "$(fold_gt "${fold}")" "${checkpoint}"; do
    test -f "${path}"
  done
  if [[ "${fold}" != 0 ]]; then
    results="${CAND12}/fold_${fold}/training/runs/resolution_adaptation/results.csv"
    test -f "${results}"
    test "$(( $(wc -l <"${results}") - 1 ))" -eq 40
  fi
done
test -f "${FOLD0_EVAL}/predictions_low.json"

for fold in 0 1 2; do
  p40="${P40_ROOT}/fold_${fold}"
  fold_out="${OUT}/fold_${fold}"
  mkdir -p "${fold_out}/logs"
  if [[ "${fold}" = 0 ]]; then
    cp "${FOLD0_EVAL}/predictions_low.json" "${fold_out}/candidate_predictions_low.json"
  else
    printf 'infer_fold_%s\n' "${fold}" >"${STATUS}"
    "${PY}" scripts/materialize_yolo_cross_resolution_infer_config.py \
      --source "${p40}/resolved_infer.yaml" --output "${fold_out}/resolved_infer.yaml" \
      --predictions "${fold_out}/candidate_predictions_low.json" --imgsz 1280 \
      --checkpoint "$(candidate_checkpoint "${fold}")"
    "${PY}" scripts/infer_cv3_oof.py --config "${fold_out}/resolved_infer.yaml" \
      >"${fold_out}/logs/infer.log" 2>&1
  fi

  printf 'compose_and_evaluate_fold_%s\n' "${fold}" >"${STATUS}"
  "${PY}" scripts/compose_class_disjoint_predictions.py \
    --primary "${p40}/predictions_low.json" \
    --expert "${fold_out}/candidate_predictions_low.json" \
    --primary-labels 0-23 --expert-labels 24 \
    --output "${fold_out}/vehicle_route_predictions.json"
  "${PY}" scripts/evaluate_fixed_score_threshold.py --gt "$(fold_gt "${fold}")" \
    --pred "${p40}/predictions_low.json" --threshold "${THRESHOLD}" \
    --output "${fold_out}/baseline_fixed_0546.json"
  "${PY}" scripts/evaluate_fixed_score_threshold.py --gt "$(fold_gt "${fold}")" \
    --pred "${fold_out}/vehicle_route_predictions.json" --threshold "${THRESHOLD}" \
    --output "${fold_out}/candidate_fixed_0546.json"
  "${PY}" scripts/compare_candidate_trend.py \
    --baseline "${fold_out}/baseline_fixed_0546.json" \
    --candidate "${fold_out}/candidate_fixed_0546.json" \
    --output "${fold_out}/paired_comparison.json"
done

printf 'aggregate_cv3_oof\n' >"${STATUS}"
mkdir -p "${OUT}/aggregate"
"${PY}" scripts/merge_cv3_coco_ledgers.py \
  --gt "$(fold_gt 0)" "$(fold_gt 1)" "$(fold_gt 2)" \
  --pred "${P40_ROOT}/fold_0/predictions_low.json" \
    "${P40_ROOT}/fold_1/predictions_low.json" "${P40_ROOT}/fold_2/predictions_low.json" \
  --output-gt "${OUT}/aggregate/ground_truth.json" \
  --output-pred "${OUT}/aggregate/baseline_predictions_low.json"
"${PY}" scripts/merge_cv3_coco_ledgers.py \
  --gt "$(fold_gt 0)" "$(fold_gt 1)" "$(fold_gt 2)" \
  --pred "${OUT}/fold_0/vehicle_route_predictions.json" \
    "${OUT}/fold_1/vehicle_route_predictions.json" "${OUT}/fold_2/vehicle_route_predictions.json" \
  --output-gt "${OUT}/aggregate/candidate_ground_truth.json" \
  --output-pred "${OUT}/aggregate/candidate_predictions_low.json"
"${PY}" - "${OUT}/aggregate/ground_truth.json" \
  "${OUT}/aggregate/candidate_ground_truth.json" <<'PY'
import json
import sys
from pathlib import Path

baseline = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
candidate = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
for field in ("images", "annotations", "categories"):
    if baseline.get(field) != candidate.get(field):
        raise SystemExit(f"merged GT semantic mismatch: {field}")
print("MERGED_GT_SEMANTIC_PARITY_PASS")
PY

"${PY}" scripts/evaluate_fixed_score_threshold.py \
  --gt "${OUT}/aggregate/ground_truth.json" \
  --pred "${OUT}/aggregate/baseline_predictions_low.json" --threshold "${THRESHOLD}" \
  --output "${OUT}/aggregate/baseline_fixed_0546.json"
"${PY}" scripts/evaluate_fixed_score_threshold.py \
  --gt "${OUT}/aggregate/ground_truth.json" \
  --pred "${OUT}/aggregate/candidate_predictions_low.json" --threshold "${THRESHOLD}" \
  --output "${OUT}/aggregate/candidate_fixed_0546.json"
"${PY}" scripts/compare_candidate_trend.py \
  --baseline "${OUT}/aggregate/baseline_fixed_0546.json" \
  --candidate "${OUT}/aggregate/candidate_fixed_0546.json" \
  --output "${OUT}/aggregate/paired_comparison.json"
for method in baseline candidate; do
  "${PY}" scripts/analyze_single_split_official_frontier.py \
    --gt "${OUT}/aggregate/ground_truth.json" \
    --pred "${OUT}/aggregate/${method}_predictions_low.json" \
    --output "${OUT}/aggregate/${method}_frontier.json" --step 0.005 \
    >"${OUT}/aggregate/${method}_frontier.log" 2>&1
done
"${PY}" scripts/triage_detector_candidate.py \
  --baseline-fixed "${OUT}/aggregate/baseline_fixed_0546.json" \
  --candidate-fixed "${OUT}/aggregate/candidate_fixed_0546.json" \
  --baseline-frontier "${OUT}/aggregate/baseline_frontier.json" \
  --candidate-frontier "${OUT}/aggregate/candidate_frontier.json" \
  --output "${OUT}/aggregate/candidate_triage.json"

for fold in 0 1 2; do
  sha256sum "$(candidate_checkpoint "${fold}")"
done >"${OUT}/checkpoint_sha256.txt"
trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
