#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/root/autodl-tmp/xh-202625-macroexpert}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}"
ROOT="${ROOT:-/root/autodl-tmp/results/MACROEXPERT-M-V1}"
WEIGHTS="${WEIGHTS:-/root/autodl-tmp/pretrained/yolo26m.pt}"
DATA="${DATA:-${ROOT}/fold0-view/dataset.yaml}"
OUT="${OUT:-${ROOT}/fold0-40ep}"
STATUS="${ROOT}/status.txt"
LOG="${ROOT}/fold0-40ep.log"

fail() {
  code=$?
  printf 'failed:%s\n' "${code}" >"${STATUS}"
  exit "${code}"
}
trap fail ERR

test -f "${WEIGHTS}"
test -f "${DATA}"
test ! -e "${OUT}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}"
printf 'training\n' >"${STATUS}"
"${PYTHON_BIN}" scripts/train_yolo_fixed_dataset.py \
  --weights "${WEIGHTS}" --data "${DATA}" --output "${OUT}" \
  --epochs 40 --imgsz 1280 --batch 4 --workers 4 --device 0 \
  >"${LOG}" 2>&1
test -f "${OUT}/result.json"
printf 'trained\n' >"${STATUS}"
