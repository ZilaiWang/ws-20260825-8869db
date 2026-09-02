#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625}
PEER_ROOT=${PEER_ROOT:-/root/autodl-tmp/peer-methods/star-xh25-hcl}
DEIM_ROOT=${DEIM_ROOT:-${PEER_ROOT}/.third_party/DEIM}
PY=${PY:-/root/autodl-tmp/miniconda3/bin/python}
NORMAL=${NORMAL:-/root/autodl-tmp/results/PEER-DEIM-HCL-M-FOLD0-40EP-V1}
BASELINE=${BASELINE:-/workspace/results/DEIM-M-FOLD0-40EP-V1-R2}
OUT=${OUT:-/root/autodl-tmp/results/PEER-DEIM-HCL-M-FOLD0-FIXED-BENCHMARKS-V1}
HARD=${HARD:-/root/autodl-tmp/pseudo10k-trial-mix-local/fold_0}
SENTINEL=${SENTINEL:-/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1/fold_0}
BASE_CONFIG=${PROJECT}/configs/experiments/deim_dfine_m_fold0_40ep.yml
CAND_CONFIG=${PROJECT}/configs/experiments/deim_hcl_m_fold0_40ep.yml
STATUS=${OUT}/status.txt

mkdir -p "${OUT}/logs"
failed() {
  code=$?
  printf 'failed_exit_%s\n' "${code}" >"${STATUS}"
  exit "${code}"
}
trap failed ERR
printf 'waiting_for_normal_screen\n' >"${STATUS}"

while true; do
  state=$(cat "${NORMAL}/status.txt" 2>/dev/null || true)
  case "${state}" in
    complete) break ;;
    failed*) printf 'stopped_upstream_%s\n' "${state}" >"${STATUS}"; trap - ERR; exit 0 ;;
  esac
  sleep 30
done

next_action=$("${PY}" - "${NORMAL}/paired_decision.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["next_action"])
PY
)
if [[ "${next_action}" != run_fixed_hard_sentinel_tiled_screen ]]; then
  printf 'stopped_normal_gate\n' >"${STATUS}"
  trap - ERR
  exit 0
fi

for root in "${HARD}" "${SENTINEL}"; do
  test -f "${root}/ground_truth.json"
  test -d "${root}/images"
  test "$(find "${root}/images" -type f | wc -l)" -eq 2
done
test -f "${BASELINE}/training/last.pth"
test -f "${NORMAL}/training/last.pth"

export PYTHONPATH="${PROJECT}/research/peer_runtime:${PEER_ROOT}/src:${DEIM_ROOT}:/workspace/venvs/deim-cu121/lib/python3.10/site-packages:/root/autodl-tmp/venvs/cv3-model-cu121/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT}"

infer_one() {
  name=$1
  root=$2
  config=$3
  checkpoint=$4
  printf 'infer_%s\n' "${name}" >"${STATUS}"
  "${PY}" scripts/infer_deim_tiled_coco.py \
    --deim-root "${DEIM_ROOT}" \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --coco "${root}/ground_truth.json" \
    --image-root "${root}/images" \
    --output "${OUT}/${name}_predictions.json" \
    --summary "${OUT}/${name}_inference_summary.json" \
    --expected-checkpoint-epoch 39 \
    --imgsz 1024 --tile-size 1024 --overlap 256 --batch-size 4 \
    --score-floor 0.001 --fine-nms-iou 0.70 --coarse-nms-iou 0.85 \
    --device cuda:0 \
    >"${OUT}/logs/${name}_infer.log" 2>&1
  "${PY}" scripts/analyze_single_split_official_frontier.py \
    --gt "${root}/ground_truth.json" \
    --pred "${OUT}/${name}_predictions.json" \
    --output "${OUT}/${name}_frontier.json" \
    >"${OUT}/logs/${name}_frontier.log" 2>&1
}

infer_one hard_baseline "${HARD}" "${BASE_CONFIG}" "${BASELINE}/training/last.pth"
infer_one hard_candidate "${HARD}" "${CAND_CONFIG}" "${NORMAL}/training/last.pth"
infer_one sentinel_baseline "${SENTINEL}" "${BASE_CONFIG}" "${BASELINE}/training/last.pth"
infer_one sentinel_candidate "${SENTINEL}" "${CAND_CONFIG}" "${NORMAL}/training/last.pth"

baseline_threshold=$("${PY}" - "${OUT}/hard_baseline_frontier.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["frontiers"]["0.150"]["threshold"])
PY
)
candidate_threshold=$("${PY}" - "${OUT}/hard_candidate_frontier.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["frontiers"]["0.150"]["threshold"])
PY
)

printf 'fixed_sentinel_transfer\n' >"${STATUS}"
"${PY}" scripts/evaluate_fixed_score_threshold.py \
  --gt "${SENTINEL}/ground_truth.json" \
  --pred "${OUT}/sentinel_baseline_predictions.json" \
  --threshold "${baseline_threshold}" \
  --output "${OUT}/sentinel_baseline_fixed.json" \
  >"${OUT}/logs/sentinel_baseline_fixed.log" 2>&1
"${PY}" scripts/evaluate_fixed_score_threshold.py \
  --gt "${SENTINEL}/ground_truth.json" \
  --pred "${OUT}/sentinel_candidate_predictions.json" \
  --threshold "${candidate_threshold}" \
  --output "${OUT}/sentinel_candidate_fixed.json" \
  >"${OUT}/logs/sentinel_candidate_fixed.log" 2>&1

printf 'decision\n' >"${STATUS}"
"${PY}" scripts/decide_peer_fixed_benchmarks.py \
  --hard-baseline "${OUT}/hard_baseline_frontier.json" \
  --hard-candidate "${OUT}/hard_candidate_frontier.json" \
  --sentinel-baseline "${OUT}/sentinel_baseline_fixed.json" \
  --sentinel-candidate "${OUT}/sentinel_candidate_fixed.json" \
  --output "${OUT}/fixed_benchmark_decision.json" \
  >"${OUT}/logs/decision.log" 2>&1

trap - ERR
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
