#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/workspace/xh-202625}
DFINE=${DFINE:-/workspace/third_party/D-FINE-codex-20260830}
OUT=${OUT:-/workspace/results/DFINE-M-FULL-40EP-AGREEMENT-V1}
PY=${PY:-/workspace/venvs/dfine-cu121/bin/python}
CONFIG=${PROJECT}/configs/experiments/dfine_m_full_40ep.yml
SOURCE=/workspace/results/DFINE-M-CV3-VEHICLE-V1/fold_2
STATUS=${OUT}/status.txt
mkdir -p "${OUT}/logs" "${OUT}/coco" "${OUT}/assets"

failed() {
  code=$?
  printf 'failed_exit_%s\n' "${code}" > "${STATUS}"
  exit "${code}"
}
trap failed ERR
printf 'preflight\n' > "${STATUS}"
test -x "${PY}"
test -d "${DFINE}"
test -f "${CONFIG}"
test -f "${SOURCE}/coco/instances_train.json"
test -f "${SOURCE}/coco/instances_val.json"

"${PY}" "${PROJECT}/scripts/build_full_coco_from_train_val.py" \
  --train "${SOURCE}/coco/instances_train.json" \
  --val "${SOURCE}/coco/instances_val.json" \
  --output "${OUT}/coco/instances_full.json" \
  --audit "${OUT}/coco/full_audit.json" \
  --expected-images 4481 > "${OUT}/logs/coco.log"

WEIGHT=${OUT}/assets/dfine_m_coco.pth
cp "${SOURCE}/assets/dfine_m_coco.pth" "${WEIGHT}"
EXPECTED=b44a7586bf490858c7b8bce9e44bd025cb88724df9a07a8deb3ae1c12e608195
test "$(sha256sum "${WEIGHT}" | awk '{print $1}')" = "${EXPECTED}"
sha256sum "${WEIGHT}" > "${OUT}/assets/dfine_m_coco.pth.sha256"

printf 'training\n' > "${STATUS}"
cd "${DFINE}"
CUDA_VISIBLE_DEVICES=0 "${PY}" train.py \
  -c "${CONFIG}" -t "${WEIGHT}" --use-amp --seed 42 \
  --output-dir "${OUT}/training" > "${OUT}/logs/train.log" 2>&1

test -f "${OUT}/training/last.pth"
test "$(wc -l < "${OUT}/training/log.txt")" -eq 40
sha256sum \
  "${OUT}/coco/instances_full.json" \
  "${OUT}/coco/full_audit.json" \
  "${OUT}/training/last.pth" > "${OUT}/SHA256SUMS.txt"
trap - ERR
printf 'complete\n' > "${STATUS}"
