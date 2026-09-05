#!/usr/bin/env bash
set -Eeuo pipefail

# Post-training validation of the exact deployable composition.  This waits for
# the already-running full hierarchy fit; it never starts or resumes training.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
TRAIN=${TRAIN:-/root/autodl-tmp/results/HERA-GUARD-MPSR-HIER-S256V128-FULL-40EP-3X4080-B60-V1}
OUT=${OUT:-/root/autodl-tmp/results/P40-HIER-S256V128-AIRCRAFT-D4-FULL-VALIDATION-V1}
BASE_CONFIG=${BASE_CONFIG:-${PROJECT}/configs/experiments/p40_aircraft_d4_full_runtime_candidate_v1.json}
HARD_ROOT=${HARD_ROOT:-/root/autodl-tmp/pseudo10k-trial-mix-local}
SENTINEL_ROOT=${SENTINEL_ROOT:-/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1}
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

printf 'waiting_for_full_hierarchy_checkpoint\n' >"${STATUS}"
while :; do
  train_status=$(cat "${TRAIN}/status.txt" 2>/dev/null || true)
  [[ "${train_status}" != failed_* ]] || exit 5
  [[ "${train_status}" == full_checkpoint_ready_waiting_for_3090_validation ]] && break
  sleep 15
done

checkpoint="${TRAIN}/training/runs/resolution_adaptation/weights/last.pt"
results="${TRAIN}/training/runs/resolution_adaptation/results.csv"
test -f "${checkpoint}"
test "$(( $(wc -l <"${results}") - 1 ))" -eq 40
for path in "${BASE_CONFIG}" "${HARD_ROOT}/ground_truth.json" \
  "${SENTINEL_ROOT}/ground_truth.json"; do
  test -f "${path}"
done

printf 'sanitize_and_build_exact_runtime\n' >"${STATUS}"
"${PY}" scripts/sanitize_yolo_checkpoint.py \
  --input "${checkpoint}" --output "${OUT}/hier_full_sanitized.pt" --imgsz 1280 \
  --require-tensor-identical --allow-index-names \
  >"${OUT}/sanitize.json"
"${PY}" scripts/build_p40_hier_d4_runtime_config.py \
  --base-config "${BASE_CONFIG}" \
  --expert-weight "${OUT}/hier_full_sanitized.pt" \
  --expert-threshold 0.546 \
  --workpoint-id p40_hier_s256v128_t0546_aircraft_d4_full_20260904 \
  --output "${OUT}/candidate_runtime_config.json" \
  >"${OUT}/runtime_config_build.json"

run_runtime() {
  local role=$1 condition=$2 device=$3 root=$4 config=$5
  local destination="${OUT}/${condition}/${role}"
  mkdir -p "${destination}"
  CUDA_VISIBLE_DEVICES="${device}" "${PY}" -u scripts/run_competition_runtime_coco.py \
    --config "${config}" --pseudo-root "${root}" --device cuda:0 \
    --predictions "${destination}/predictions.json" \
    --summary "${destination}/runtime_summary.json" \
    >"${OUT}/logs/${condition}_${role}.log" 2>&1
}

printf 'exact_runtime_proxy_inference\n' >"${STATUS}"
run_runtime baseline hard 0 "${HARD_ROOT}" "${BASE_CONFIG}" & p0=$!
run_runtime candidate hard 1 "${HARD_ROOT}" "${OUT}/candidate_runtime_config.json" & p1=$!
run_runtime baseline sentinel 2 "${SENTINEL_ROOT}" "${BASE_CONFIG}" & p2=$!
wait "${p0}" "${p1}" "${p2}"
run_runtime candidate sentinel 1 "${SENTINEL_ROOT}" "${OUT}/candidate_runtime_config.json"

printf 'evaluate_exact_runtime_proxy\n' >"${STATUS}"
for condition in hard sentinel; do
  if [[ "${condition}" == hard ]]; then root="${HARD_ROOT}"; else root="${SENTINEL_ROOT}"; fi
  for role in baseline candidate; do
    "${PY}" scripts/evaluate_fixed_score_threshold.py \
      --gt "${root}/ground_truth.json" \
      --pred "${OUT}/${condition}/${role}/predictions.json" --threshold 0 \
      --output "${OUT}/${condition}/${role}/metrics.json"
  done
  "${PY}" scripts/compare_candidate_trend.py \
    --baseline "${OUT}/${condition}/baseline/metrics.json" \
    --candidate "${OUT}/${condition}/candidate/metrics.json" \
    --output "${OUT}/${condition}/paired_comparison.json"
done

sha256sum "${checkpoint}" "${OUT}/hier_full_sanitized.pt" \
  "${OUT}/candidate_runtime_config.json" >"${OUT}/asset_sha256.txt"
trap - ERR INT TERM
printf 'complete_waiting_for_3090_latency_and_submission_decision\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
