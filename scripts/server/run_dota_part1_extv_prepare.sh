#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${SOURCE_ROOT:?set SOURCE_ROOT containing images and the frozen COCO file}"
: "${ASSET_ROOT:?set ASSET_ROOT on the data disk}"
: "${OUT:?set OUT}"

STATUS="${OUT}/status.txt"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed_exit_%s\n" "$code" >"${STATUS}"; fi' EXIT
mkdir -p "${OUT}" "${ASSET_ROOT}/derived" "${ASSET_ROOT}/audit"
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

COCO="${SOURCE_ROOT}/derived/train-part1-coarse-v2.json"
IMAGES="${SOURCE_ROOT}/prepared/train-part1/images"
SLICED_COCO="${ASSET_ROOT}/derived/train-part1-sliced-1024.json"
DATASET="${ASSET_ROOT}/yolo-dota-part1-v2"

[[ -f "${COCO}" ]] || { echo "missing frozen DOTA COCO: ${COCO}" >&2; exit 2; }
[[ "$(find "${IMAGES}" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')" = 469 ]] || {
  echo "DOTA part1 must contain exactly 469 PNG images" >&2
  exit 2
}

sha256sum \
  scripts/slice_external_coarse_coco.py \
  scripts/export_coarse_coco_to_yolo.py \
  scripts/build_external_role_sampler.py \
  scripts/render_external_coarse_audit.py \
  src/rsdet/external/slicing.py >"${OUT}/CODE_SHA256.txt"

printf '%s\n' slicing_scale_preserving_all_visible_1024 >"${STATUS}"
if [[ ! -f "${SLICED_COCO}" ]]; then
  "${PYTHON_BIN}" scripts/slice_external_coarse_coco.py \
    --input-coco "${COCO}" --image-root "${IMAGES}" \
    --output-coco "${SLICED_COCO}" \
    --output-image-root "${DATASET}/images/train" \
    --audit "${ASSET_ROOT}/audit/slicing.json" \
    --tile-size 1024 --overlap 256 --min-visibility 0.7 \
    --empty-tiles-per-image 2 --workers 12
fi

printf '%s\n' exporting_yolo_and_extv_role >"${STATUS}"
if [[ ! -f "${DATASET}/dataset.yaml" ]]; then
  "${PYTHON_BIN}" scripts/export_coarse_coco_to_yolo.py \
    --coco "${SLICED_COCO}" --dataset-root "${DATASET}" --split train \
    --audit "${ASSET_ROOT}/audit/yolo-export.json"
fi
if [[ ! -f "${DATASET}/dataset-ext-v.yaml" ]]; then
  "${PYTHON_BIN}" scripts/build_external_role_sampler.py \
    --coco "${SLICED_COCO}" --dataset-root "${DATASET}" --role EXT-V \
    --audit "${ASSET_ROOT}/audit/ext-v.json"
fi

printf '%s\n' rendering_visual_audit >"${STATUS}"
if [[ ! -f "${OUT}/visual/render_summary.json" ]]; then
  "${PYTHON_BIN}" scripts/render_external_coarse_audit.py \
    --coco "${SLICED_COCO}" --image-root "${DATASET}/images/train" \
    --output-dir "${OUT}/visual" --maximum-images 96 --columns 4
fi

if [[ ! -f "${OUT}/visual_decision.json" ]]; then
  printf '%s\n' waiting_for_agent_visual_review >"${STATUS}"
  exit 0
fi

"${PYTHON_BIN}" - "${OUT}/visual_decision.json" "${ASSET_ROOT}" "${DATASET}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

decision_path = Path(sys.argv[1])
asset_root = Path(sys.argv[2])
dataset = Path(sys.argv[3])
decision = json.loads(decision_path.read_text())
if decision.get("status") != "pass":
    raise SystemExit("visual decision is not pass")
required = [
    asset_root / "audit" / name
    for name in ("slicing.json", "yolo-export.json", "ext-v.json")
]
for path in required:
    if not path.is_file():
        raise SystemExit(f"missing audit: {path}")
payload = {
    "status": "ready_for_external_pretraining",
    "protocol": "dota_part1_469_extv_scale_preserving_all_visible_v2",
    "dataset_yaml": str((dataset / "dataset.yaml").resolve()),
    "ext_v_yaml": str((dataset / "dataset-ext-v.yaml").resolve()),
    "audit_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in required
    },
    "visual_decision": decision,
}
(decision_path.parent / "prepare_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
)
PY

sha256sum "${OUT}/prepare_summary.json" "${OUT}/visual_decision.json" >"${OUT}/SHA256SUMS"
printf '%s\n' ready_for_external_pretraining >"${STATUS}"
