#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
OUT=${OUT:-/root/autodl-tmp/results/R1-5-AIRCRAFT-VIEW-CONSISTENCY-FULL-V1}
MANIFEST=${MANIFEST:-/workspace/inputs/R1-1/proposal_crop_manifest.csv}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data}
WEIGHTS=${WEIGHTS:-/root/autodl-tmp/pretrained/convnext_tiny-983f1562.pth}
P03=${P03:-/root/autodl-tmp/results/P03-APEX-FULL-V1/final_checkpoint.pt}

if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}/logs"
exec 9>"${OUT}.lock"
flock -n 9 || exit 4
failed() { code=$?; printf 'failed_exit_%s\n' "${code}" >"${OUT}/status.txt"; exit "${code}"; }
trap failed ERR INT TERM
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"
printf 'train_full_aircraft_view_consistency_5ep\n' >"${OUT}/status.txt"
CUDA_VISIBLE_DEVICES=0 "${PY}" scripts/train_aircraft_view_consistency_full.py \
  --config configs/experiments/r1_aircraft_view_consistency_full_v1.yaml \
  --training-manifest "${MANIFEST}" --data-root "${DATA_ROOT}" \
  --weights "${WEIGHTS}" --p03-checkpoint "${P03}" \
  --output-dir "${OUT}/training" --device cuda:0 \
  >"${OUT}/logs/train.log" 2>&1
sha256sum "${OUT}/training/final_checkpoint.pt" >"${OUT}/checkpoint.sha256"
trap - ERR INT TERM
printf 'full_checkpoint_ready_for_runtime_integration\n' >"${OUT}/status.txt"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
