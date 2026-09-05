#!/usr/bin/env bash
set -Eeuo pipefail

# Independent fold confirmation for the Vehicle-only S96 route on one RTX
# 3090. Batch 8 is accumulated by Ultralytics against nbs=64, matching the
# effective batch scale of the 3-GPU fold0 run (global batch 60).
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
FOLD=${FOLD:-1}
SOURCE=${SOURCE:-/root/autodl-tmp/results/YOLO-SCALE-CV3-CONFIRM-V1/s1024/fold_${FOLD}}
P40=${P40:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1/fold_${FOLD}}
GT=${GT:-/root/autodl-tmp/capscale-cv3-assets/fold_${FOLD}/instances_val.json}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-MPSR-VEHICLE-S96-FOLD${FOLD}-40EP-3090-EVAL-V1}
STATUS=${OUT}/status.txt

if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}/logs"
exec 9>"${OUT}.lock"
flock -n 9 || exit 4
failed() { code=$?; printf 'failed_exit_%s\n' "${code}" >"${STATUS}"; exit "${code}"; }
trap failed ERR INT TERM
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for path in "${SOURCE}/train.txt" "${SOURCE}/val.txt" \
  "${SOURCE}/runs/foundation/weights/last.pt" "${P40}/resolved_infer.yaml" \
  "${P40}/predictions_low.json" "${GT}"; do
  test -f "${path}"
done
test "$("${PY}" -c 'import torch; print(torch.cuda.device_count())')" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'RTX 3090'

printf 'materialize_vehicle_s96_fold%s\n' "${FOLD}" >"${STATUS}"
"${PY}" scripts/materialize_object_scale_detector_scenes.py \
  --train-list "${SOURCE}/train.txt" --val-list "${SOURCE}/val.txt" \
  --output "${OUT}/data" --network-size 1280 --target-network-side 96 \
  --vehicle-target-network-side 96 --target-classes 24 --output-pixels 800 \
  --max-extra-images 744 --max-extra-fraction 0.25 --seed 42 \
  >"${OUT}/logs/materialize.log" 2>&1
"${PY}" - "${OUT}/data/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
assert summary["train_validation_overlap"] == 0
assert summary["selected_count"] > 0
assert summary["selected_by_class"] == {"24": summary["selected_count"]}
assert summary["policies"]["vehicle"]["target_network_side"] == 96
assert summary["validation_unchanged"] is True
PY

printf 'train_fold%s_40ep_3090\n' "${FOLD}" >"${STATUS}"
"${PY}" scripts/train_progressive_resolution_adaptation.py \
  --weights "${SOURCE}/runs/foundation/weights/last.pt" \
  --data "${OUT}/data/dataset.yaml" --output "${OUT}/training" \
  --epochs 40 --imgsz 1280 --batch 8 --workers 4 --device cuda:0 \
  --seed 42 --lr0 0.0002 --lrf 0.10 --rotate90-p 1.0 \
  >"${OUT}/logs/train.log" 2>&1

checkpoint="${OUT}/training/runs/resolution_adaptation/weights/last.pt"
results="${OUT}/training/runs/resolution_adaptation/results.csv"
test -f "${checkpoint}"
test "$(( $(wc -l <"${results}") - 1 ))" -eq 40

printf 'infer_fold%s_3090\n' "${FOLD}" >"${STATUS}"
"${PY}" scripts/materialize_yolo_cross_resolution_infer_config.py \
  --source "${P40}/resolved_infer.yaml" --output "${OUT}/resolved_infer.yaml" \
  --predictions "${OUT}/candidate_predictions_low.json" --imgsz 1280 \
  --checkpoint "${checkpoint}"
"${PY}" scripts/infer_cv3_oof.py --config "${OUT}/resolved_infer.yaml" \
  >"${OUT}/logs/infer.log" 2>&1

printf 'compose_vehicle_route_and_evaluate\n' >"${STATUS}"
"${PY}" scripts/compose_class_disjoint_predictions.py \
  --primary "${P40}/predictions_low.json" --expert "${OUT}/candidate_predictions_low.json" \
  --primary-labels 0-23 --expert-labels 24 --output "${OUT}/vehicle_route_predictions.json"
"${PY}" scripts/evaluate_fixed_score_threshold.py --gt "${GT}" \
  --pred "${P40}/predictions_low.json" --threshold 0.546 \
  --output "${OUT}/baseline_fixed_0546.json"
"${PY}" scripts/evaluate_fixed_score_threshold.py --gt "${GT}" \
  --pred "${OUT}/vehicle_route_predictions.json" --threshold 0.546 \
  --output "${OUT}/candidate_fixed_0546.json"
"${PY}" scripts/compare_candidate_trend.py \
  --baseline "${OUT}/baseline_fixed_0546.json" \
  --candidate "${OUT}/candidate_fixed_0546.json" \
  --output "${OUT}/paired_comparison.json"
for method in baseline candidate; do
  pred="${P40}/predictions_low.json"
  [[ "${method}" = candidate ]] && pred="${OUT}/vehicle_route_predictions.json"
  "${PY}" scripts/analyze_single_split_official_frontier.py --gt "${GT}" --pred "${pred}" \
    --output "${OUT}/${method}_frontier.json" --step 0.005 \
    >"${OUT}/logs/${method}_frontier.log" 2>&1
done
"${PY}" scripts/triage_detector_candidate.py \
  --baseline-fixed "${OUT}/baseline_fixed_0546.json" \
  --candidate-fixed "${OUT}/candidate_fixed_0546.json" \
  --baseline-frontier "${OUT}/baseline_frontier.json" \
  --candidate-frontier "${OUT}/candidate_frontier.json" \
  --output "${OUT}/candidate_triage.json"

sha256sum "${checkpoint}" >"${OUT}/checkpoint.sha256"
trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
