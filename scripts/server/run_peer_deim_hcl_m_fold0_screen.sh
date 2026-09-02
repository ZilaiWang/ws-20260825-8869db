#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625}
PEER_ROOT=${PEER_ROOT:-/root/autodl-tmp/peer-methods/star-xh25-hcl}
DEIM_ROOT=${DEIM_ROOT:-${PEER_ROOT}/.third_party/DEIM}
OUT=${OUT:-/root/autodl-tmp/results/PEER-DEIM-HCL-M-FOLD0-40EP-V1}
PY=${PY:-/root/autodl-tmp/miniconda3/bin/python}
BASELINE=${BASELINE:-/workspace/results/DEIM-M-FOLD0-40EP-V1-R2}
CONFIG=${CONFIG:-${PROJECT}/configs/experiments/deim_hcl_m_fold0_40ep.yml}
STATUS=${OUT}/status.txt

PEER_REPOSITORY=https://github.com/star1sakura/XH-202625-remote-sensing-detection.git
PEER_COMMIT=d23ef57ea5e3ea80ec71e883776718a8c3c1510a
DEIM_REPOSITORY=https://github.com/ShihuaHuang95/DEIM.git
DEIM_COMMIT=09d35d53d39ee3145a1e61e3a989b28b9468d1dd
PRETRAINED_SHA=2b6cd0582a4aa711f583982057b7fb0f3daebdd98e4dc168824714014c3219bc
TRAIN_COCO_SHA=41e93416083ad39cd8b665b53be6613f81d9d9d6c1d052da1809b7e71d5686ef
VAL_COCO_SHA=2641d3bb15388b9a19812ab514b993d5f68ef90d7a59fb02834bf7903e585977

mkdir -p "${OUT}/logs" "${OUT}/coco" "${OUT}/assets" "$(dirname "${PEER_ROOT}")"
failed() {
  code=$?
  printf 'failed_exit_%s\n' "${code}" >"${STATUS}"
  exit "${code}"
}
trap failed ERR
printf 'preflight\n' >"${STATUS}"

if [[ ! -d "${PEER_ROOT}/.git" ]]; then
  git clone --filter=blob:none "${PEER_REPOSITORY}" "${PEER_ROOT}"
fi
git -C "${PEER_ROOT}" fetch origin "${PEER_COMMIT}"
test -z "$(git -C "${PEER_ROOT}" status --porcelain)"
git -C "${PEER_ROOT}" switch --detach "${PEER_COMMIT}"
test "$(git -C "${PEER_ROOT}" rev-parse HEAD)" = "${PEER_COMMIT}"

if [[ ! -d "${DEIM_ROOT}/.git" ]]; then
  git clone --filter=blob:none "${DEIM_REPOSITORY}" "${DEIM_ROOT}"
fi
git -C "${DEIM_ROOT}" fetch origin "${DEIM_COMMIT}"
if [[ "$(git -C "${DEIM_ROOT}" rev-parse HEAD)" != "${DEIM_COMMIT}" ]]; then
  test -z "$(git -C "${DEIM_ROOT}" status --porcelain)"
  git -C "${DEIM_ROOT}" switch --detach "${DEIM_COMMIT}"
fi
if ! grep -q 'xh_detect.deim_bhcl_adapter' "${DEIM_ROOT}/engine/__init__.py"; then
  for patch in \
    deim-preserve-epoch-checkpoints.patch \
    deim-torchvision-v2-compat.patch \
    deim-selective-class-row-finetune.patch \
    deim-bhcl.patch; do
    patch_path=${PEER_ROOT}/patches/${patch}
    git -C "${DEIM_ROOT}" apply --check "${patch_path}"
    git -C "${DEIM_ROOT}" apply "${patch_path}"
  done
fi
grep -q 'xh_detect.deim_bhcl_adapter' "${DEIM_ROOT}/engine/__init__.py"
grep -q 'initialize_after_tuning' "${DEIM_ROOT}/engine/solver/_solver.py"

export PYTHONPATH="${PROJECT}/research/peer_runtime:${PEER_ROOT}/src:${DEIM_ROOT}:/workspace/venvs/deim-cu121/lib/python3.10/site-packages:/root/autodl-tmp/venvs/cv3-model-cu121/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
"${PY}" - <<'PY'
import torch
from xh_detect.deim_bhcl_adapter import BHCLDEIMCriterion, BHCLDFINETransformer
assert torch.cuda.is_available()
assert BHCLDFINETransformer.__name__ == "BHCLDFINETransformer"
assert BHCLDEIMCriterion.__name__ == "BHCLDEIMCriterion"
print({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
PY

cp "${BASELINE}/coco/instances_train.json" "${OUT}/coco/instances_train.json"
cp "${BASELINE}/coco/instances_val.json" "${OUT}/coco/instances_val.json"
cp "${BASELINE}/assets/deim_m_coco.pth" "${OUT}/assets/deim_m_coco.pth"
test "$(sha256sum "${OUT}/coco/instances_train.json" | awk '{print $1}')" = "${TRAIN_COCO_SHA}"
test "$(sha256sum "${OUT}/coco/instances_val.json" | awk '{print $1}')" = "${VAL_COCO_SHA}"
test "$(sha256sum "${OUT}/assets/deim_m_coco.pth" | awk '{print $1}')" = "${PRETRAINED_SHA}"

printf 'smoke\n' >"${STATUS}"
cd "${PROJECT}"
"${PY}" - "${CONFIG}" <<'PY'
import sys
import torch
from engine.core import YAMLConfig

cfg = YAMLConfig(sys.argv[1])
cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
model = cfg.model.cuda().train()
decoder = model.decoder
decoder.initialize_after_tuning()
assert decoder.decoder.decoupled_ready
assert len(decoder.decoder.layers) == 4
assert decoder.bhcl_mode == "hcl"
assert sum(p.numel() for p in model.parameters()) > 1_000_000
with torch.no_grad():
    features = [
        torch.randn(1, 256, 128, 128, device="cuda"),
        torch.randn(1, 256, 64, 64, device="cuda"),
        torch.randn(1, 256, 32, 32, device="cuda"),
    ]
    targets = [{
        "labels": torch.tensor([0, 4, 24], device="cuda"),
        "boxes": torch.tensor(
            [[0.2, 0.2, 0.05, 0.05], [0.5, 0.5, 0.1, 0.1], [0.8, 0.8, 0.04, 0.04]],
            device="cuda",
        ),
    }]
    outputs = decoder(features, targets)
assert outputs["bhcl_features"].shape[-1] == 128
assert torch.isfinite(outputs["bhcl_features"]).all()
print({"bhcl_features": list(outputs["bhcl_features"].shape)})
PY

printf 'training\n' >"${STATUS}"
cd "${DEIM_ROOT}"
CUDA_VISIBLE_DEVICES=0 "${PY}" train.py \
  -c "${CONFIG}" \
  -t "${OUT}/assets/deim_m_coco.pth" \
  --use-amp --seed 42 --output-dir "${OUT}/training" \
  >"${OUT}/logs/train.log" 2>&1

test -f "${OUT}/training/last.pth"
test "$(wc -l <"${OUT}/training/log.txt")" -eq 40
printf 'normal_inference\n' >"${STATUS}"
cd "${PROJECT}"
"${PY}" scripts/infer_deim_coco.py \
  --deim-root "${DEIM_ROOT}" \
  --config "${CONFIG}" \
  --checkpoint "${OUT}/training/last.pth" \
  --coco "${OUT}/coco/instances_val.json" \
  --image-root /root/autodl-tmp/data \
  --imgsz 1024 --batch-size 4 --score-floor 0.001 \
  --output "${OUT}/predictions.json" \
  --summary "${OUT}/inference_summary.json" \
  >"${OUT}/logs/infer.log" 2>&1

"${PY}" scripts/analyze_single_split_official_frontier.py \
  --gt "${OUT}/coco/instances_val.json" \
  --pred "${OUT}/predictions.json" \
  --output "${OUT}/frontier.json" \
  >"${OUT}/logs/frontier.log" 2>&1

"${PY}" scripts/decide_peer_normal_screen.py \
  --baseline "${BASELINE}/frontier.json" \
  --candidate "${OUT}/frontier.json" \
  --output "${OUT}/paired_decision.json" \
  --research-only-unlicensed-reference

trap - ERR
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
