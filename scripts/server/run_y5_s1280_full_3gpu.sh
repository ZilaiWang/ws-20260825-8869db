#!/usr/bin/env bash
set -Eeuo pipefail

# Unique full-data candidate authorized by the S/M x 1024/1280 fold0 screen.
# Scientific contract: YOLO26-s, 25 classes, 1280, Y5 RandomRotate90, all
# 4,481 official images, 160 fixed epochs, seed42, fixed last checkpoint.

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625-capscale}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/capscale-data-s1280-full}
MANIFEST=${MANIFEST:-/root/autodl-tmp/capscale-assets/split_view.json}
WEIGHTS=${WEIGHTS:-/root/autodl-tmp/capscale-assets/yolo26s.pt}
OUT=${OUT:-/root/autodl-tmp/results/Y5-FULL-S1280-3GPU-R1}
PREFLIGHT_OUT=${PREFLIGHT_OUT:-${OUT}-preflight}
STATUS=${OUT}/status.txt
LOG=${OUT}/train.log
LOCK=${OUT}.lock

WEIGHT_SHA=646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
MANIFEST_SHA=a647ce030fa832aadc6a6c286a3f6464ac1783f71797a52cc598ec340f128943
LABEL_TREE_SHA=929d73218df463629eca8d0f66becd032a742029df310b4b8c38c167b64a2128

mkdir -p "${OUT}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  printf 'another full-training driver holds %s\n' "${LOCK}" >&2
  exit 4
fi
failed() {
  code=$?
  printf 'failed_exit_%s\n' "${code}" >"${STATUS}"
  exit "${code}"
}
trap failed ERR INT TERM

if [[ -e "${OUT}/runs" || -e "${OUT}/training_result.json" ]]; then
  printf 'refusing non-fresh full run: %s\n' "${OUT}" >&2
  exit 3
fi
for path in "${PY}" "${MANIFEST}" "${WEIGHTS}" "${DATA_ROOT}/images" "${DATA_ROOT}/labels"; do
  test -e "${path}"
done
test "$(sha256sum "${WEIGHTS}" | awk '{print $1}')" = "${WEIGHT_SHA}"
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = "${MANIFEST_SHA}"
test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')" -eq 3

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PY}" - "${DATA_ROOT}/labels" "${LABEL_TREE_SHA}" <<'PY'
from pathlib import Path
import hashlib
import sys

root = Path(sys.argv[1])
expected = sys.argv[2]
files = sorted(root.rglob("*.txt"), key=lambda p: p.relative_to(root).as_posix())
if len(files) != 8962:
    raise SystemExit(f"expected 8962 labels, found {len(files)}")
digest = hashlib.sha256()
for path in files:
    digest.update(path.relative_to(root).as_posix().encode())
    digest.update(b"\0")
    digest.update(path.read_bytes())
actual = digest.hexdigest()
if actual != expected:
    raise SystemExit(f"label tree SHA mismatch: {actual}")
print({"label_count": len(files), "label_tree_sha256": actual})
PY

sha256sum \
  scripts/train_full_y5.py \
  src/rsdet/innovation/trainers.py \
  >"${OUT}/CODE_SHA256.txt"
"${PY}" - <<'PY' >"${OUT}/ENVIRONMENT.txt"
import platform
import sys
import albumentations
import torch
import ultralytics
print("python", sys.version.replace("\n", " "))
print("platform", platform.platform())
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("ultralytics", ultralytics.__version__)
print("albumentations", albumentations.__version__)
PY
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader \
  >"${OUT}/GPU_ENVIRONMENT.csv"

if [[ ! -f "${PREFLIGHT_OUT}/training_contract.json" ]]; then
  "${PY}" scripts/train_full_y5.py \
    --manifest "${MANIFEST}" --data-root "${DATA_ROOT}" \
    --weights "${WEIGHTS}" --expected-weight-sha256 "${WEIGHT_SHA}" \
    --output-dir "${PREFLIGHT_OUT}" --model-key Y5-FULL-S1280-3GPU-R1 \
    --epochs 160 --imgsz 1280 --batch 12 --workers 8 --seed 42 \
    --rotate90-p 1.0 --device 0,1,2 --dry-run \
    >"${OUT}/preflight.log" 2>&1
fi
"${PY}" - "${PREFLIGHT_OUT}/training_contract.json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
args = payload["train_args"]
assert payload["status"] == "dry_run"
assert payload["dataset_audit"]["image_count"] == 4481
assert payload["uses_validation_for_selection"] is False
assert (args["epochs"], args["imgsz"], args["batch"], args["seed"]) == (160, 1280, 12, 42)
assert args["device"] == "0,1,2"
assert payload["rotate90_p"] == 1.0
assert payload["checkpoint_selection"] == "fixed_epoch_last"
PY

printf 'training_ddp_3gpu\n' >"${STATUS}"
"${PY}" scripts/train_full_y5.py \
  --manifest "${MANIFEST}" --data-root "${DATA_ROOT}" \
  --weights "${WEIGHTS}" --expected-weight-sha256 "${WEIGHT_SHA}" \
  --output-dir "${OUT}" --model-key Y5-FULL-S1280-3GPU-R1 \
  --epochs 160 --imgsz 1280 --batch 12 --workers 8 --seed 42 \
  --rotate90-p 1.0 --device 0,1,2 \
  >"${LOG}" 2>&1

test -f "${OUT}/training_result.json"
test -f "${OUT}/runs/foundation/weights/last.pt"
test "$(($(wc -l <"${OUT}/runs/foundation/results.csv") - 1))" -eq 160
sha256sum \
  "${OUT}/training_contract.json" \
  "${OUT}/training_result.json" \
  "${OUT}/runs/foundation/results.csv" \
  "${OUT}/runs/foundation/weights/last.pt" \
  >"${OUT}/RESULT_SHA256.txt"

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
