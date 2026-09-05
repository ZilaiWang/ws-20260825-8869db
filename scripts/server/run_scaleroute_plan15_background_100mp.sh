#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625-capscale}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
BACKGROUND=${BACKGROUND:-/root/autodl-tmp/assets/MACROSHIFT-BACKGROUND-100MP-FROZEN}
PRIMARY=${PRIMARY:-/root/autodl-tmp/results/Y5-FULL-S-20260829-R1/y5_full_s_sanitized.pt}
EXPERT=${EXPERT:-/root/autodl-tmp/results/Y5-FULL-S1280-3GPU-R1/runs/foundation/weights/last.pt}
OUT=${OUT:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-BACKGROUND-100MP-V1}
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

MANIFEST="${BACKGROUND}/background_100mp_manifest.jsonl"
for path in "${PY}" "${MANIFEST}" "${PRIMARY}" "${EXPERT}"; do test -f "${path}"; done
test "$(sha256sum "${MANIFEST}" | cut -d' ' -f1)" = \
  ed3cbbe6952ea5a7792821a316bd3b0ed93888f74a50eda2630f630c9c9020e7
test "$(sha256sum "${PRIMARY}" | cut -d' ' -f1)" = \
  f7e30fac3391d7048314d1c41df0e878a4d5f0423e5c55b68ee7df5917418229
test "$(sha256sum "${EXPERT}" | cut -d' ' -f1)" = \
  539ab9c9b9a0e4ed925037496f1ca2b6882e4fb6fa5dd0dacb93408438ed0460

run_infer() {
  local id=$1 checkpoint=$2 imgsz=$3
  printf 'infer_%s\n' "${id}" >"${STATUS}"
  "${PY}" scripts/infer_yolo_background_100mp.py \
    --manifest "${MANIFEST}" --root "${BACKGROUND}" --checkpoint "${checkpoint}" \
    --output "${OUT}/${id}.json" --imgsz "${imgsz}" --batch-size 16 \
    --confidence 0.001 --iou 0.7 --max-detections 500 --device cuda:0 \
    >"${OUT}/logs/${id}.log" 2>&1
}

run_infer primary_i1024 "${PRIMARY}" 1024
run_infer expert_i1280 "${EXPERT}" 1280
run_infer expert_i1024 "${EXPERT}" 1024

printf 'analyze_routes\n' >"${STATUS}"
"${PY}" scripts/analyze_background_resolution_route.py \
  --manifest "${MANIFEST}" --primary-pred "${OUT}/primary_i1024.json" \
  --expert-pred "${OUT}/expert_i1280.json" --primary-labels 0-23 \
  --expert-labels 24 --primary-threshold 0.646 --expert-threshold 0.646 \
  --output "${OUT}/r2_s1024_sa_s1280_vehicle.json"
"${PY}" scripts/analyze_background_resolution_route.py \
  --manifest "${MANIFEST}" --primary-pred "${OUT}/primary_i1024.json" \
  --expert-pred "${OUT}/expert_i1024.json" --primary-labels 4-23 \
  --expert-labels 0-3,24 --primary-threshold 0.646 --expert-threshold 0.611 \
  --output "${OUT}/r3_s1024_aircraft_s1280train_ship_vehicle.json"

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
