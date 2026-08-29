#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${MANIFEST:?set MANIFEST}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${WEIGHT_ROOT:?set WEIGHT_ROOT}"
: "${OUT:?set OUT}"

mkdir -p "${OUT}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

declare -a WEIGHT_SHA=(
  "47d98fab29cbc4b6836a907a77cda33294affbd891e90f9e3aab0b05578e7c96"
  "3b175e1471ae139dd4415cac094487e5a4be369cb0f1af6094bd3b2f1f25a9d4"
  "8cec3f91cd0421c328f1fa430de2b9b4def4f28fdab41b1d90d56e76bd413304"
)

printf '%s\n' training > "${OUT}/status.txt"
for fold in 0 1 2; do
  "${PYTHON_BIN}" scripts/train_y5_coarse_detector.py \
    --manifest "${MANIFEST}" \
    --data-root "${DATA_ROOT}" \
    --held-out-fold "${fold}" \
    --weights "${WEIGHT_ROOT}/fold_${fold}/training/runs/foundation/weights/last.pt" \
    --expected-weight-sha256 "${WEIGHT_SHA[$fold]}" \
    --output-dir "${OUT}/fold_${fold}" \
    --epochs 30 \
    --imgsz 1024 \
    --batch 8 \
    --workers 8 \
    --seed 20260829 \
    > "${OUT}/fold_${fold}.log" 2>&1
done
(
  cd "${OUT}"
  sha256sum fold_*/training_contract.json fold_*/training_result.json \
    fold_*/runs/foundation/weights/last.pt > SHA256SUMS
)
printf '%s\n' complete > "${OUT}/status.txt"
