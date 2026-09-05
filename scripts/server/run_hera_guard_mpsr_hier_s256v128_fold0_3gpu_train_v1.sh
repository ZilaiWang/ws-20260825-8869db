#!/usr/bin/env bash
set -Eeuo pipefail

# Train-only paired screen for hierarchical object-scale scene crops:
# Ship targets are rendered near 256 px and Vehicle targets near 128 px at
# network size 1280. Evaluation is intentionally deferred to an RTX 3090.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
SOURCE=${SOURCE:-/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1/s1024}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-MPSR-HIER-S256V128-FOLD0-40EP-3X4080-B60-V1}
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

printf 'materialize_hier_s256v128_scene_supplement\n' >"${STATUS}"
"${PY}" scripts/materialize_object_scale_detector_scenes.py \
  --train-list "${SOURCE}/train.txt" --val-list "${SOURCE}/val.txt" \
  --output "${OUT}/data" --network-size 1280 \
  --ship-target-network-side 256 --vehicle-target-network-side 128 \
  --output-pixels 800 --max-extra-images 744 --max-extra-fraction 0.25 --seed 42 \
  >"${OUT}/logs/materialize.log" 2>&1

"${PY}" - "${OUT}/data/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
assert summary["train_validation_overlap"] == 0
assert summary["selected_count"] == 359
assert summary["policies"]["ship"]["target_network_side"] == 256
assert summary["policies"]["vehicle"]["target_network_side"] == 128
assert summary["validation_unchanged"] is True
PY

printf 'train_fold0_40ep_3gpu\n' >"${STATUS}"
"${PY}" scripts/train_progressive_resolution_adaptation.py \
  --weights "${SOURCE}/runs/foundation/weights/last.pt" \
  --data "${OUT}/data/dataset.yaml" --output "${OUT}/training" \
  --epochs 40 --imgsz 1280 --batch 60 --workers 4 --device 0,1,2 \
  --seed 42 --lr0 0.0002 --lrf 0.10 --rotate90-p 1.0 \
  >"${OUT}/logs/train.log" 2>&1

CHECKPOINT="${OUT}/training/runs/resolution_adaptation/weights/last.pt"
RESULTS="${OUT}/training/runs/resolution_adaptation/results.csv"
test -f "${CHECKPOINT}"
test "$(($(wc -l <"${RESULTS}") - 1))" -eq 40
sha256sum "${CHECKPOINT}" >"${OUT}/checkpoint.sha256"
trap - ERR INT TERM
printf 'trained_waiting_for_3090_inference\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
