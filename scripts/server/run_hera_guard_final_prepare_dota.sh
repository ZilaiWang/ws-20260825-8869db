#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO to the xh-202625 checkout}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${ASSET_ROOT:?set ASSET_ROOT; at least 60 GiB free is recommended}"
: "${OUT:?set OUT}"

STATUS="${OUT}/status.txt"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed_exit_%s\n" "$code" >"${STATUS}"; fi' EXIT
mkdir -p "${OUT}" "${ASSET_ROOT}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

RAW="${ASSET_ROOT}/raw"
PREPARED="${ASSET_ROOT}/prepared"
DERIVED="${ASSET_ROOT}/derived"
DATASET="${ASSET_ROOT}/yolo-dota-v1"
AUDIT="${OUT}/audit"
mkdir -p "${PREPARED}/train" "${PREPARED}/val" "${DERIVED}" "${AUDIT}"

printf '%s\n' code_and_disk_gate >"${STATUS}"
sha256sum \
  configs/external/dota_v1_coarse.json \
  scripts/download_external_dota_v1.py \
  scripts/extract_external_archives.py \
  scripts/import_dota_to_coarse_coco.py \
  scripts/merge_external_coarse_coco.py \
  scripts/slice_external_coarse_coco.py \
  scripts/export_coarse_coco_to_yolo.py \
  scripts/build_external_role_sampler.py \
  scripts/render_external_coarse_audit.py \
  src/rsdet/external/dota.py src/rsdet/external/slicing.py >"${OUT}/CODE_SHA256.txt"

if [[ ! -f "${ASSET_ROOT}/ASSET_LOCK.json" ]]; then
  printf '%s\n' downloading_official_dota >"${STATUS}"
  "${PYTHON_BIN}" scripts/download_external_dota_v1.py \
    --config configs/external/dota_v1_coarse.json \
    --output-root "${ASSET_ROOT}" --minimum-free-gib 60
fi

if [[ ! -f "${AUDIT}/extract-train-images.json" ]]; then
  printf '%s\n' extracting_train_images >"${STATUS}"
  "${PYTHON_BIN}" scripts/extract_external_archives.py \
    --archive "${RAW}/train/images/part1.zip" \
    --archive "${RAW}/train/images/part2.zip" \
    --archive "${RAW}/train/images/part3.zip" \
    --output-dir "${PREPARED}/train" --manifest "${AUDIT}/extract-train-images.json"
fi
if [[ ! -f "${AUDIT}/extract-train-labels.json" ]]; then
  "${PYTHON_BIN}" scripts/extract_external_archives.py \
    --archive "${RAW}/train/labelTxt-v1.0/labelTxt.zip" \
    --output-dir "${PREPARED}/train" --manifest "${AUDIT}/extract-train-labels.json"
fi
if [[ ! -f "${AUDIT}/extract-val-images.json" ]]; then
  printf '%s\n' extracting_val >"${STATUS}"
  "${PYTHON_BIN}" scripts/extract_external_archives.py \
    --archive "${RAW}/val/images/part1.zip" \
    --output-dir "${PREPARED}/val" --manifest "${AUDIT}/extract-val-images.json"
fi
if [[ ! -f "${AUDIT}/extract-val-labels.json" ]]; then
  "${PYTHON_BIN}" scripts/extract_external_archives.py \
    --archive "${RAW}/val/labelTxt-v1.0/labelTxt.zip" \
    --output-dir "${PREPARED}/val" --manifest "${AUDIT}/extract-val-labels.json"
fi

if [[ ! -f "${DERIVED}/train-coarse.json" ]]; then
  printf '%s\n' importing_train >"${STATUS}"
  "${PYTHON_BIN}" scripts/import_dota_to_coarse_coco.py \
    --image-root "${PREPARED}/train" --label-root "${PREPARED}/train" \
    --output "${DERIVED}/train-coarse.json" --audit "${AUDIT}/import-train.json"
fi
if [[ ! -f "${DERIVED}/val-coarse.json" ]]; then
  printf '%s\n' importing_val >"${STATUS}"
  "${PYTHON_BIN}" scripts/import_dota_to_coarse_coco.py \
    --image-root "${PREPARED}/val" --label-root "${PREPARED}/val" \
    --output "${DERIVED}/val-coarse.json" --audit "${AUDIT}/import-val.json"
fi
if [[ ! -f "${DERIVED}/train-val-coarse.json" ]]; then
  printf '%s\n' merging_train_val >"${STATUS}"
  "${PYTHON_BIN}" scripts/merge_external_coarse_coco.py \
    --source "train=${DERIVED}/train-coarse.json" \
    --source "val=${DERIVED}/val-coarse.json" \
    --output "${DERIVED}/train-val-coarse.json" --audit "${AUDIT}/merge.json"
fi
if [[ ! -f "${DERIVED}/train-val-sliced.json" ]]; then
  printf '%s\n' slicing_1024_overlap256 >"${STATUS}"
  "${PYTHON_BIN}" scripts/slice_external_coarse_coco.py \
    --input-coco "${DERIVED}/train-val-coarse.json" --image-root "${PREPARED}" \
    --output-coco "${DERIVED}/train-val-sliced.json" \
    --output-image-root "${DATASET}/images/train" --audit "${AUDIT}/slicing.json" \
    --tile-size 1024 --overlap 256 --min-visibility 0.7 --empty-tiles-per-image 2 \
    --workers 16
fi
if [[ ! -f "${DATASET}/dataset.yaml" ]]; then
  printf '%s\n' exporting_yolo >"${STATUS}"
  "${PYTHON_BIN}" scripts/export_coarse_coco_to_yolo.py \
    --coco "${DERIVED}/train-val-sliced.json" --dataset-root "${DATASET}" \
    --split train --audit "${AUDIT}/yolo-export.json"
fi
if [[ ! -f "${DATASET}/dataset-ext-g.yaml" ]]; then
  "${PYTHON_BIN}" scripts/build_external_role_sampler.py \
    --coco "${DERIVED}/train-val-sliced.json" --dataset-root "${DATASET}" \
    --role EXT-G --audit "${AUDIT}/ext-g.json"
fi
if [[ ! -f "${DATASET}/dataset-ext-v.yaml" ]]; then
  "${PYTHON_BIN}" scripts/build_external_role_sampler.py \
    --coco "${DERIVED}/train-val-sliced.json" --dataset-root "${DATASET}" \
    --role EXT-V --audit "${AUDIT}/ext-v.json"
fi
if [[ ! -f "${OUT}/visual/render_summary.json" ]]; then
  printf '%s\n' rendering_deterministic_visual_audit >"${STATUS}"
  "${PYTHON_BIN}" scripts/render_external_coarse_audit.py \
    --coco "${DERIVED}/train-val-sliced.json" --image-root "${DATASET}/images/train" \
    --output-dir "${OUT}/visual" --maximum-images 96 --columns 4
fi

if [[ ! -f "${OUT}/visual_decision.json" ]]; then
  printf '%s\n' waiting_for_agent_visual_review >"${STATUS}"
  printf '%s\n' "Inspect OUT/visual contact sheets; write visual_decision.json with status=pass."
  exit 0
fi
"${PYTHON_BIN}" - "${OUT}/visual_decision.json" "${AUDIT}" "${DATASET}" <<'PY'
import hashlib,json,sys
from pathlib import Path
decision=json.load(open(sys.argv[1]))
if decision.get("status") != "pass": raise SystemExit("visual decision is not pass")
audit=Path(sys.argv[2]); dataset=Path(sys.argv[3])
required=[audit/x for x in ("import-train.json","import-val.json","merge.json","slicing.json","yolo-export.json","ext-g.json","ext-v.json")]
for path in required:
    if not path.is_file(): raise SystemExit(f"missing audit: {path}")
rows={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in required}
summary={"status":"ready_for_external_pretraining","protocol":"hera_guard_final_dota_full_prepare_v1","dataset_yaml":str((dataset/"dataset.yaml").resolve()),"ext_g_yaml":str((dataset/"dataset-ext-g.yaml").resolve()),"ext_v_yaml":str((dataset/"dataset-ext-v.yaml").resolve()),"audit_sha256":rows,"visual_decision":decision}
Path(sys.argv[1]).with_name("prepare_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
PY
sha256sum "${OUT}/prepare_summary.json" "${OUT}/visual_decision.json" >"${OUT}/SHA256SUMS"
printf '%s\n' ready_for_external_pretraining >"${STATUS}"
