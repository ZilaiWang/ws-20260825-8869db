#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/autodl-tmp/xh-202625}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}"
CACHE="${CACHE:-/workspace/results/HERA-GUARD-V4-OMQ-FACTORIAL-GPU/base_crop/cache.npz}"
RESULT_ROOT="${RESULT_ROOT:-/root/autodl-tmp/results/MACROSHIFT-SHIP-QUALITY-V1}"
HELD_OUT_FOLD="${HELD_OUT_FOLD:-0}"
DEVICE="${DEVICE:-cuda:0}"

if [[ ! "${HELD_OUT_FOLD}" =~ ^[012]$ ]]; then
  echo "HELD_OUT_FOLD must be 0, 1, or 2" >&2
  exit 2
fi

mkdir -p "${RESULT_ROOT}"
STATUS="${RESULT_ROOT}/fold_${HELD_OUT_FOLD}.status"
LOG="${RESULT_ROOT}/fold_${HELD_OUT_FOLD}.log"
LOCK="${RESULT_ROOT}/fold_${HELD_OUT_FOLD}.lock"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "fold ${HELD_OUT_FOLD} is already owned by another process" >&2
  exit 3
fi

if [[ -f "${RESULT_ROOT}/fold_${HELD_OUT_FOLD}/quality_fold${HELD_OUT_FOLD}.json" ]]; then
  echo complete >"${STATUS}"
  exit 0
fi

on_error() {
  local rc=$?
  echo "failed:${rc}" >"${STATUS}"
  exit "${rc}"
}
trap on_error ERR INT TERM

cd "${ROOT}"
test -x "${PYTHON_BIN}"
test -s "${CACHE}"
export CACHE
echo preflight >"${STATUS}"
{
  date -Is
  sha256sum "${CACHE}" \
    scripts/train_official_quality_head.py \
    configs/experiments/macroshift_ship_objectness_quality_v1.json
  "${PYTHON_BIN}" - <<'PY'
import numpy as np
import torch
import os
from pathlib import Path

cache = Path(os.environ["CACHE"])
with np.load(cache, allow_pickle=False) as payload:
    required = {
        "features", "detector_score", "best_same_fine_iou", "coarse_id",
        "protected_tp", "active_fp", "active_mask", "group_id", "fold",
    }
    missing = sorted(required - set(payload.files))
    if missing:
        raise RuntimeError(f"cache missing arrays: {missing}")
    n = int(payload["features"].shape[0])
    print({"rows": n, "dim": int(payload["features"].shape[1]),
           "ship_rows": int((payload["coarse_id"] == 0).sum()),
           "folds": {str(i): int((payload["fold"] == i).sum()) for i in range(3)}})
print({"torch": torch.__version__, "cuda": torch.cuda.is_available(),
       "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
PY
} >>"${LOG}" 2>&1

echo training >"${STATUS}"
"${PYTHON_BIN}" scripts/train_official_quality_head.py \
  --data "${CACHE}" \
  --output-dir "${RESULT_ROOT}/fold_${HELD_OUT_FOLD}" \
  --held-out-fold "${HELD_OUT_FOLD}" \
  --coarse-filter ship \
  --sampling group_balanced \
  --robustness group_dro \
  --rank-enabled \
  --epochs 20 \
  --batch-size 512 \
  --hidden-dim 192 \
  --residual-limit 1.75 \
  --device "${DEVICE}" >>"${LOG}" 2>&1

test -s "${RESULT_ROOT}/fold_${HELD_OUT_FOLD}/quality_fold${HELD_OUT_FOLD}.pt"
test -s "${RESULT_ROOT}/fold_${HELD_OUT_FOLD}/quality_fold${HELD_OUT_FOLD}.torchscript.pt"
test -s "${RESULT_ROOT}/fold_${HELD_OUT_FOLD}/quality_fold${HELD_OUT_FOLD}.scores.npz"
echo complete >"${STATUS}"
trap - ERR INT TERM
