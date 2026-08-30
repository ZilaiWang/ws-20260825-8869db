#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/workspace/xh-202625}
DFINE=${DFINE:-/workspace/third_party/D-FINE-codex-20260830}
OUT=${OUT:-/workspace/results/DFINE-M-FOLD0-40EP-V1-R2}
BASE_PY=${BASE_PY:-/workspace/venvs/p06-cu121/bin/python}
VENV=${VENV:-/workspace/venvs/dfine-cu121}
PY=${VENV}/bin/python
STATUS=${OUT}/status.txt
mkdir -p "${OUT}/logs" "${OUT}/coco" "${OUT}/assets"

failed() {
  code=$?
  printf 'failed_exit_%s\n' "${code}" > "${STATUS}"
  exit "${code}"
}
trap failed ERR
printf 'preflight\n' > "${STATUS}"

if [[ ! -x "${PY}" ]]; then
  "${BASE_PY}" -m venv --system-site-packages "${VENV}"
fi
if ! "${PY}" -c 'import pycocotools, faster_coco_eval, yaml, tensorboard, scipy, numpy, calflops, transformers, loguru, matplotlib' >/dev/null 2>&1; then
  "${PY}" -m pip install -q \
    pycocotools 'faster-coco-eval>=1.6.6' PyYAML tensorboard \
    calflops transformers loguru matplotlib
fi
if ! "${PY}" -c 'import numpy, scipy; assert numpy.__version__ == "1.26.4"' 2>/dev/null; then
  "${PY}" -m pip install -q --ignore-installed numpy==1.26.4 scipy==1.15.3
fi

"${PY}" "${PROJECT}/scripts/build_dfine_coco_fold.py" \
  --split-view /workspace/results/Y5-ROT90-CV3-OOF/fold_0/split_view.json \
  --data-root /root/autodl-tmp/data \
  --output-dir "${OUT}/coco" > "${OUT}/logs/coco.log"

WEIGHT=${OUT}/assets/dfine_m_coco.pth
if [[ ! -f "${WEIGHT}" ]]; then
  curl -L --retry 5 --retry-delay 3 \
    -o "${WEIGHT}.partial" \
    https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth
  mv "${WEIGHT}.partial" "${WEIGHT}"
fi
EXPECTED=b44a7586bf490858c7b8bce9e44bd025cb88724df9a07a8deb3ae1c12e608195
ACTUAL=$(sha256sum "${WEIGHT}" | awk '{print $1}')
test "${ACTUAL}" = "${EXPECTED}"
sha256sum "${WEIGHT}" > "${OUT}/assets/dfine_m_coco.pth.sha256"

printf 'training\n' > "${STATUS}"
cd "${DFINE}"
CUDA_VISIBLE_DEVICES=0 "${PY}" train.py \
  -c "${PROJECT}/configs/experiments/dfine_m_fold0_40ep.yml" \
  -t "${WEIGHT}" --use-amp --seed 42 --output-dir "${OUT}/training" \
  > "${OUT}/logs/train.log" 2>&1

test -f "${OUT}/training/last.pth"
printf 'infer\n' > "${STATUS}"
cd "${PROJECT}"
"${PY}" scripts/infer_dfine_coco.py \
  --dfine-root "${DFINE}" \
  --config configs/experiments/dfine_m_fold0_40ep.yml \
  --checkpoint "${OUT}/training/last.pth" \
  --coco "${OUT}/coco/instances_val.json" \
  --image-root /root/autodl-tmp/data \
  --imgsz 1024 --batch-size 4 --score-floor 0.001 \
  --output "${OUT}/predictions.json" \
  --summary "${OUT}/inference_summary.json" \
  > "${OUT}/logs/infer.log" 2>&1

"${PY}" scripts/analyze_single_split_official_frontier.py \
  --gt "${OUT}/coco/instances_val.json" \
  --pred "${OUT}/predictions.json" \
  --output "${OUT}/frontier.json" \
  > "${OUT}/logs/frontier.log" 2>&1

sha256sum "${OUT}"/*.json "${OUT}/training/last.pth" > "${OUT}/SHA256SUMS.txt"
trap - ERR
printf 'complete\n' > "${STATUS}"
