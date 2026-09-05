#!/usr/bin/env bash
set -Eeuo pipefail

# Three-GPU train-only execution of the frozen S128 object-scale scene-crop
# screen. Inference is intentionally deferred to the official-comparable 3090.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
SOURCE=${SOURCE:-/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1/s1024}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-MPSR-S128-FOLD0-40EP-3X4080-B60-V1}
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
  "${SOURCE}/runs/foundation/weights/last.pt"; do
  test -f "${path}"
done
test "$("${PY}" -c 'import torch; print(torch.cuda.device_count())')" -ge 3

printf 'materialize_s128_scene_supplement\n' >"${STATUS}"
"${PY}" scripts/materialize_object_scale_detector_scenes.py \
  --train-list "${SOURCE}/train.txt" --val-list "${SOURCE}/val.txt" \
  --output "${OUT}/data" --network-size 1280 --target-network-side 128 \
  --output-pixels 800 --max-extra-images 744 --max-extra-fraction 0.25 --seed 42 \
  >"${OUT}/logs/materialize.log" 2>&1

printf 'train_fold0_40ep_3gpu\n' >"${STATUS}"
"${PY}" scripts/train_progressive_resolution_adaptation.py \
  --weights "${SOURCE}/runs/foundation/weights/last.pt" \
  --data "${OUT}/data/dataset.yaml" --output "${OUT}/training" \
  --epochs 40 --imgsz 1280 --batch 60 --workers 4 --device 0,1,2 \
  --seed 42 --lr0 0.0002 --lrf 0.10 --rotate90-p 1.0 \
  >"${OUT}/logs/train.log" 2>&1

CHECKPOINT="${OUT}/training/runs/resolution_adaptation/weights/last.pt"
test -f "${CHECKPOINT}"
sha256sum "${CHECKPOINT}" >"${OUT}/checkpoint.sha256"
trap - ERR INT TERM
printf 'trained_waiting_for_3090_inference\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
