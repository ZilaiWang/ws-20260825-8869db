#!/usr/bin/env bash
set -Eeuo pipefail

# Full-data Vehicle S96 expert. This is paired with the official P40 recipe:
# the same mature S1024/160e initializer is adapted for 40 epochs at 1280,
# changing only the source-safe Vehicle scene-crop supplement. No validation
# labels are used and fixed-last is the only checkpoint.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-apex-v1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
INITIAL=${INITIAL:-/root/autodl-tmp/results/Y5-FULL-S-20260829-R1/y5_full_s_sanitized.pt}
ALL_IMAGES=${ALL_IMAGES:-/root/autodl-tmp/results/Y5-FULL-S1280-3GPU-R1/all_train_images.txt}
OUT=${OUT:-/root/autodl-tmp/results/HERA-GUARD-MPSR-VEHICLE-S96-FULL-40EP-3X4080-B60-V1}
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

for path in "${INITIAL}" "${ALL_IMAGES}"; do test -f "${path}"; done
test "$("${PY}" -c 'import torch; print(torch.cuda.device_count())')" -ge 3
test "$(grep -cve '^$' "${ALL_IMAGES}")" -eq 4481

printf 'materialize_full_vehicle_s96\n' >"${STATUS}"
"${PY}" scripts/materialize_object_scale_detector_scenes.py \
  --train-list "${ALL_IMAGES}" --val-list "${ALL_IMAGES}" --full-training \
  --output "${OUT}/data" --network-size 1280 --target-network-side 96 \
  --vehicle-target-network-side 96 --target-classes 24 --output-pixels 800 \
  --max-extra-images 1120 --max-extra-fraction 0.25 --seed 42 \
  >"${OUT}/logs/materialize.log" 2>&1

"${PY}" - "${OUT}/data/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
assert summary["train_source_count"] == 4481
assert summary["validation_source_count"] == 4481
assert summary["train_validation_overlap"] == 4481
assert summary["training_mode"] == "full_no_validation"
assert summary["validation_used_for_training_or_selection"] is False
assert summary["selected_count"] > 0
assert summary["selected_by_class"] == {"24": summary["selected_count"]}
assert summary["policies"]["vehicle"]["target_network_side"] == 96
PY

printf 'train_full_40ep_3gpu\n' >"${STATUS}"
"${PY}" scripts/train_progressive_resolution_adaptation.py \
  --weights "${INITIAL}" --data "${OUT}/data/dataset.yaml" \
  --output "${OUT}/training" --epochs 40 --imgsz 1280 --batch 60 \
  --workers 4 --device 0,1,2 --seed 42 --lr0 0.0002 --lrf 0.10 \
  --rotate90-p 1.0 >"${OUT}/logs/train.log" 2>&1

checkpoint="${OUT}/training/runs/resolution_adaptation/weights/last.pt"
results="${OUT}/training/runs/resolution_adaptation/results.csv"
test -f "${checkpoint}"
test "$(( $(wc -l <"${results}") - 1 ))" -eq 40
sha256sum "${INITIAL}" >"${OUT}/initial_checkpoint.sha256"
sha256sum "${checkpoint}" >"${OUT}/checkpoint.sha256"

trap - ERR INT TERM
printf 'full_checkpoint_ready_waiting_for_deployment_validation\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
