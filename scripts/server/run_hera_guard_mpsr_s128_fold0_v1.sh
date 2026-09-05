#!/usr/bin/env bash
set -Eeuo pipefail

# Plan-16 detector-level object-scale scene-crop paired screen. This queues
# behind the full P03 job so a single-GPU host is never oversubscribed.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
SOURCE=${SOURCE:-/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1/s1024}
P40=${P40:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1/fold_0}
GT=${GT:-/root/autodl-tmp/capscale-assets/instances_val.json}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-MPSR-S128-FOLD0-40EP-V1}
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

printf 'queued_behind_p03_apex_full\n' >"${STATUS}"
while screen -list 2>/dev/null | grep -q '[.]p03-apex-full'; do sleep 30; done
if [[ -f /root/autodl-tmp/results/P03-APEX-FULL-V1/status.txt ]]; then
  grep -qx complete /root/autodl-tmp/results/P03-APEX-FULL-V1/status.txt
fi

printf 'materialize_s128_scene_supplement\n' >"${STATUS}"
"${PY}" scripts/materialize_object_scale_detector_scenes.py \
  --train-list "${SOURCE}/train.txt" --val-list "${SOURCE}/val.txt" \
  --output "${OUT}/data" --network-size 1280 --target-network-side 128 \
  --output-pixels 800 --max-extra-images 744 --max-extra-fraction 0.25 --seed 42 \
  >"${OUT}/logs/materialize.log" 2>&1

printf 'train_fold0_40ep\n' >"${STATUS}"
"${PY}" scripts/train_progressive_resolution_adaptation.py \
  --weights "${SOURCE}/runs/foundation/weights/last.pt" \
  --data "${OUT}/data/dataset.yaml" --output "${OUT}/training" \
  --epochs 40 --imgsz 1280 --batch 8 --workers 4 --device cuda:0 \
  --seed 42 --lr0 0.0002 --lrf 0.10 --rotate90-p 1.0 \
  >"${OUT}/logs/train.log" 2>&1

CHECKPOINT="${OUT}/training/runs/resolution_adaptation/weights/last.pt"
test -f "${CHECKPOINT}"
printf 'infer_fold0\n' >"${STATUS}"
"${PY}" scripts/materialize_yolo_cross_resolution_infer_config.py \
  --source "${P40}/resolved_infer.yaml" --output "${OUT}/resolved_infer.yaml" \
  --predictions "${OUT}/predictions_low.json" --imgsz 1280 --checkpoint "${CHECKPOINT}"
"${PY}" scripts/infer_cv3_oof.py --config "${OUT}/resolved_infer.yaml" \
  >"${OUT}/logs/infer.log" 2>&1

printf 'paired_fixed_evaluation\n' >"${STATUS}"
"${PY}" scripts/evaluate_fixed_score_threshold.py --gt "${GT}" \
  --pred "${P40}/predictions_low.json" --threshold 0.546 \
  --output "${OUT}/baseline_fixed_0546.json"
"${PY}" scripts/evaluate_fixed_score_threshold.py --gt "${GT}" \
  --pred "${OUT}/predictions_low.json" --threshold 0.546 \
  --output "${OUT}/candidate_fixed_0546.json"
"${PY}" scripts/compare_candidate_trend.py \
  --baseline "${OUT}/baseline_fixed_0546.json" \
  --candidate "${OUT}/candidate_fixed_0546.json" \
  --output "${OUT}/paired_comparison.json"
"${PY}" scripts/analyze_single_split_official_frontier.py \
  --gt "${GT}" --pred "${OUT}/predictions_low.json" \
  --output "${OUT}/candidate_frontier.json" --step 0.005 \
  >"${OUT}/logs/frontier.log" 2>&1

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
