#!/usr/bin/env bash
set -euo pipefail

# Resume only the frozen post-training evaluation after the original inference
# adapter failed to materialize the trained BHCL decoder. Never retrain here.
PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625}
PEER_ROOT=${PEER_ROOT:-/root/autodl-tmp/peer-methods/star-xh25-hcl}
DEIM_ROOT=${DEIM_ROOT:-${PEER_ROOT}/.third_party/DEIM}
OUT=${OUT:-/root/autodl-tmp/results/PEER-DEIM-HCL-M-FOLD0-40EP-V1}
BASELINE=${BASELINE:-/workspace/results/DEIM-M-FOLD0-40EP-V1-R2}
CONFIG=${CONFIG:-${PROJECT}/configs/experiments/deim_hcl_m_fold0_40ep.yml}
PY=${PY:-/root/autodl-tmp/miniconda3/bin/python}
STATUS=${OUT}/status.txt

failed() {
  code=$?
  printf 'failed_posttrain_recovery_exit_%s\n' "${code}" >"${STATUS}"
  exit "${code}"
}
trap failed ERR

case "$(cat "${STATUS}")" in
  failed_exit_1 | failed_posttrain_recovery_exit_1) ;;
  *) echo 'recovery requires a preserved post-training failure state' >&2; exit 2 ;;
esac
test -f "${OUT}/training/last.pth"
test "$(wc -l <"${OUT}/training/log.txt")" -eq 40
if pgrep -f '^/root/autodl-tmp/miniconda3/bin/python train.py .*deim_hcl_m_fold0_40ep.yml' >/dev/null; then
  echo 'refusing recovery while a training process is still running' >&2
  exit 2
fi

export PYTHONPATH="${PROJECT}/research/peer_runtime:${PEER_ROOT}/src:${DEIM_ROOT}:/workspace/venvs/deim-cu121/lib/python3.10/site-packages:/root/autodl-tmp/venvs/cv3-model-cu121/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT}"

"${PY}" - "${OUT}/training/last.pth" "${OUT}/posttrain_recovery_manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

checkpoint_path, output_path = map(Path, sys.argv[1:])
checkpoint = torch.load(checkpoint_path, map_location="cpu")
if int(checkpoint.get("last_epoch", -1)) != 39:
    raise RuntimeError("recovery requires the frozen epoch-39 checkpoint")
payload = {
    "status": "approved_posttrain_only_recovery",
    "original_failure": "missing_bhcl_decoder_materialization_before_load_state_dict",
    "training_restarted": False,
    "checkpoint_epoch": 39,
    "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
}
output_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

printf 'normal_inference_recovery\n' >"${STATUS}"
"${PY}" scripts/infer_deim_coco.py \
  --deim-root "${DEIM_ROOT}" \
  --config "${CONFIG}" \
  --checkpoint "${OUT}/training/last.pth" \
  --coco "${OUT}/coco/instances_val.json" \
  --image-root /root/autodl-tmp/data \
  --imgsz 1024 --batch-size 4 --score-floor 0.001 \
  --output "${OUT}/predictions.json" \
  --summary "${OUT}/inference_summary.json" \
  >"${OUT}/logs/infer_recovery.log" 2>&1

"${PY}" scripts/analyze_single_split_official_frontier.py \
  --gt "${OUT}/coco/instances_val.json" \
  --pred "${OUT}/predictions.json" \
  --output "${OUT}/frontier.json" \
  >"${OUT}/logs/frontier_recovery.log" 2>&1

"${PY}" scripts/decide_peer_normal_screen.py \
  --baseline "${BASELINE}/frontier.json" \
  --candidate "${OUT}/frontier.json" \
  --output "${OUT}/paired_decision.json" \
  --research-only-unlicensed-reference

trap - ERR
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"

# Preserve the original gate: the fixed Hard/Sentinel chain self-stops when
# Normal fails and otherwise runs the already frozen benchmark contract.
exec bash "${PROJECT}/scripts/server/run_peer_deim_hcl_fixed_benchmarks.sh"
