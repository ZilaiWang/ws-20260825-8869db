#!/usr/bin/env bash
set -Eeuo pipefail

# Confirm the positive fold0 hierarchical scale candidate on folds 1 and 2.
# Each fold uses its own mature S1024 checkpoint, source-safe train list and
# untouched validation list. Folds run serially with all three GPUs so the
# optimization contract matches fold0 (global batch 60).
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-/root/autodl-tmp/results/YOLO-SCALE-CV3-CONFIRM-V1/s1024}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-MPSR-HIER-S256V128-CV3-REMAINING-3X4080-B60-V1}
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
test "$("${PY}" -c 'import torch; print(torch.cuda.device_count())')" -ge 3

for fold in 1 2; do
  source="${SOURCE_ROOT}/fold_${fold}"
  fold_out="${OUT}/fold_${fold}"
  for path in "${source}/train.txt" "${source}/val.txt" \
    "${source}/runs/foundation/weights/last.pt"; do
    test -f "${path}"
  done
  mkdir -p "${fold_out}/logs"
  printf 'fold_%s_materialize\n' "${fold}" >"${STATUS}"
  "${PY}" scripts/materialize_object_scale_detector_scenes.py \
    --train-list "${source}/train.txt" --val-list "${source}/val.txt" \
    --output "${fold_out}/data" --network-size 1280 \
    --ship-target-network-side 256 --vehicle-target-network-side 128 \
    --output-pixels 800 --max-extra-images 744 --max-extra-fraction 0.25 --seed 42 \
    >"${fold_out}/logs/materialize.log" 2>&1

  expected=412
  [[ "${fold}" = 2 ]] && expected=379
  "${PY}" - "${fold_out}/data/summary.json" "${expected}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
assert summary["train_validation_overlap"] == 0
assert summary["selected_count"] == int(sys.argv[2])
assert summary["policies"]["ship"]["target_network_side"] == 256
assert summary["policies"]["vehicle"]["target_network_side"] == 128
assert summary["validation_unchanged"] is True
PY

  printf 'fold_%s_train_40ep_3gpu\n' "${fold}" >"${STATUS}"
  "${PY}" scripts/train_progressive_resolution_adaptation.py \
    --weights "${source}/runs/foundation/weights/last.pt" \
    --data "${fold_out}/data/dataset.yaml" --output "${fold_out}/training" \
    --epochs 40 --imgsz 1280 --batch 60 --workers 4 --device 0,1,2 \
    --seed 42 --lr0 0.0002 --lrf 0.10 --rotate90-p 1.0 \
    >"${fold_out}/logs/train.log" 2>&1

  checkpoint="${fold_out}/training/runs/resolution_adaptation/weights/last.pt"
  results="${fold_out}/training/runs/resolution_adaptation/results.csv"
  test -f "${checkpoint}"
  test "$(($(wc -l <"${results}") - 1))" -eq 40
  sha256sum "${checkpoint}" >"${fold_out}/checkpoint.sha256"
done

trap - ERR INT TERM
printf 'trained_folds_1_2_waiting_for_fixed_inference\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
