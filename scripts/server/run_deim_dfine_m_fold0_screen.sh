#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625}
DEIM=${DEIM:-/workspace/third_party/DEIM-codex-20260830}
OUT=${OUT:-/workspace/results/DEIM-M-FOLD0-40EP-V1-R2}
PY=${PY:-/root/autodl-tmp/miniconda3/bin/python}
STATUS=${OUT}/status.txt
export PYTHONPATH="/workspace/venvs/deim-cu121/lib/python3.10/site-packages:/root/autodl-tmp/venvs/cv3-model-cu121/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUT}/logs" "${OUT}/coco" "${OUT}/assets"

failed() {
  code=$?
  printf 'failed_exit_%s\n' "${code}" > "${STATUS}"
  exit "${code}"
}
trap failed ERR
printf 'preflight\n' > "${STATUS}"

if ! "${PY}" -c 'import pycocotools, yaml, tensorboard, scipy, numpy, calflops, transformers, loguru, matplotlib' >/dev/null 2>&1; then
  "${PY}" -m pip install -q pycocotools PyYAML tensorboard scipy==1.15.3 \
    numpy==1.26.4 calflops transformers loguru matplotlib
fi

"${PY}" "${PROJECT}/scripts/build_dfine_coco_fold.py" \
  --split-view /workspace/results/Y5-ROT90-CV3-OOF/fold_0/split_view.json \
  --data-root /root/autodl-tmp/data \
  --output-dir "${OUT}/coco" > "${OUT}/logs/coco.log"

WEIGHT=${OUT}/assets/deim_m_coco.pth
EXPECTED=2b6cd0582a4aa711f583982057b7fb0f3daebdd98e4dc168824714014c3219bc
ACTUAL=$(sha256sum "${WEIGHT}" | awk '{print $1}')
test "${ACTUAL}" = "${EXPECTED}"
sha256sum "${WEIGHT}" > "${OUT}/assets/deim_m_coco.pth.sha256"

printf 'training\n' > "${STATUS}"
cd "${DEIM}"
CUDA_VISIBLE_DEVICES=0 "${PY}" train.py \
  -c "${PROJECT}/configs/experiments/deim_dfine_m_fold0_40ep.yml" \
  -t "${WEIGHT}" --use-amp --seed 42 --output-dir "${OUT}/training" \
  > "${OUT}/logs/train.log" 2>&1

test -f "${OUT}/training/last.pth"
printf 'infer\n' > "${STATUS}"
cd "${PROJECT}"
"${PY}" scripts/infer_deim_coco.py \
  --deim-root "${DEIM}" \
  --config configs/experiments/deim_dfine_m_fold0_40ep.yml \
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
