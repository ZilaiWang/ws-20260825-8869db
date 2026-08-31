#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${ASSET_ROOT:?set ASSET_ROOT; at least 50 GiB free is recommended}"
: "${OUT:?set OUT}"

mkdir -p "${ASSET_ROOT}" "${OUT}/audit"
STATUS="${OUT}/status.txt"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed_exit_%s\n" "$code" >"${STATUS}"; fi' EXIT
cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
RAW="${ASSET_ROOT}/raw"
PREPARED="${ASSET_ROOT}/prepared"
DERIVED="${ASSET_ROOT}/derived"
DATASET="${ASSET_ROOT}/yolo-dior"
AUDIT="${OUT}/audit"
mkdir -p "${RAW}" "${PREPARED}" "${DERIVED}" "${AUDIT}"

printf '%s\n' code_and_asset_gate >"${STATUS}"
sha256sum \
  configs/external/dior_coarse.json \
  scripts/download_external_gdrive_folder.py \
  scripts/extract_external_archives.py \
  scripts/discover_dior_layout.py \
  scripts/import_dior_to_coarse_coco.py \
  scripts/slice_external_coarse_coco.py \
  scripts/export_coarse_coco_to_yolo.py \
  scripts/build_external_role_sampler.py \
  scripts/render_external_coarse_audit.py \
  src/rsdet/external/dior.py src/rsdet/external/slicing.py >"${OUT}/CODE_SHA256.txt"

if [[ ! -f "${RAW}/ASSET_LOCK.json" ]]; then
  printf '%s\n' downloading_official_dior >"${STATUS}"
  "${PYTHON_BIN}" scripts/download_external_gdrive_folder.py \
    --config configs/external/dior_coarse.json --output-root "${RAW}" \
    --minimum-free-gib 50 >"${OUT}/download.log" 2>&1
fi
if [[ ! -f "${AUDIT}/extract.json" ]]; then
  printf '%s\n' extracting_official_dior >"${STATUS}"
  mapfile -d '' archives < <(find "${RAW}" -type f -iname '*.zip' -print0 | sort -z)
  [[ "${#archives[@]}" -gt 0 ]] || { echo 'official folder contains no ZIP archives' >&2; exit 2; }
  command=("${PYTHON_BIN}" scripts/extract_external_archives.py)
  for archive in "${archives[@]}"; do command+=(--archive "${archive}"); done
  command+=(--output-dir "${PREPARED}" --manifest "${AUDIT}/extract.json")
  "${command[@]}" >"${OUT}/extract.log" 2>&1
fi
if [[ ! -f "${AUDIT}/layout.json" ]]; then
  "${PYTHON_BIN}" scripts/discover_dior_layout.py \
    --root "${PREPARED}" --output "${AUDIT}/layout.json" >"${OUT}/layout.log" 2>&1
fi
mapfile -t layout < <("${PYTHON_BIN}" - "${AUDIT}/layout.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print(x["image_root"]); print(x["annotation_root"]); print(x["split_file"])
PY
)
IMAGE_ROOT="${layout[0]}"
ANNOTATION_ROOT="${layout[1]}"
SPLIT_FILE="${layout[2]}"

if [[ ! -f "${DERIVED}/trainval-coarse.json" ]]; then
  printf '%s\n' importing_trainval >"${STATUS}"
  "${PYTHON_BIN}" scripts/import_dior_to_coarse_coco.py \
    --image-root "${IMAGE_ROOT}" --annotation-root "${ANNOTATION_ROOT}" \
    --split-file "${SPLIT_FILE}" --output "${DERIVED}/trainval-coarse.json" \
    --audit "${AUDIT}/import.json" >"${OUT}/import.log" 2>&1
fi
if [[ ! -f "${DERIVED}/trainval-sliced.json" ]]; then
  printf '%s\n' slicing_1024_overlap256 >"${STATUS}"
  "${PYTHON_BIN}" scripts/slice_external_coarse_coco.py \
    --input-coco "${DERIVED}/trainval-coarse.json" --image-root "${IMAGE_ROOT}" \
    --output-coco "${DERIVED}/trainval-sliced.json" \
    --output-image-root "${DATASET}/images/train" --audit "${AUDIT}/slicing.json" \
    --tile-size 1024 --overlap 256 --min-visibility 0.7 --empty-tiles-per-image 2 \
    >"${OUT}/slice.log" 2>&1
fi
if [[ ! -f "${DATASET}/dataset.yaml" ]]; then
  "${PYTHON_BIN}" scripts/export_coarse_coco_to_yolo.py \
    --coco "${DERIVED}/trainval-sliced.json" --dataset-root "${DATASET}" \
    --split train --audit "${AUDIT}/yolo-export.json" >"${OUT}/export.log" 2>&1
fi
for role in EXT-G EXT-V; do
  lower=$(printf '%s' "${role}" | tr '[:upper:]' '[:lower:]')
  if [[ ! -f "${DATASET}/dataset-${lower}.yaml" ]]; then
    "${PYTHON_BIN}" scripts/build_external_role_sampler.py \
      --coco "${DERIVED}/trainval-sliced.json" --dataset-root "${DATASET}" \
      --role "${role}" --audit "${AUDIT}/${lower}.json" >"${OUT}/${lower}.log" 2>&1
  fi
done
if [[ ! -f "${OUT}/visual/render_summary.json" ]]; then
  printf '%s\n' rendering_visual_audit >"${STATUS}"
  "${PYTHON_BIN}" scripts/render_external_coarse_audit.py \
    --coco "${DERIVED}/trainval-sliced.json" --image-root "${DATASET}/images/train" \
    --output-dir "${OUT}/visual" --maximum-images 96 --columns 4
fi
if [[ ! -f "${OUT}/visual_decision.json" ]]; then
  printf '%s\n' waiting_for_agent_visual_review >"${STATUS}"
  exit 0
fi
"${PYTHON_BIN}" - "${OUT}/visual_decision.json" "${AUDIT}" "${DATASET}" <<'PY'
import hashlib,json,sys
from pathlib import Path
d=json.load(open(sys.argv[1]))
if d.get("status") != "pass": raise SystemExit("visual decision is not pass")
a=Path(sys.argv[2]); ds=Path(sys.argv[3])
required=[a/x for x in ("layout.json","import.json","slicing.json","yolo-export.json","ext-g.json","ext-v.json")]
for p in required:
    if not p.is_file(): raise SystemExit(f"missing audit: {p}")
out={"status":"ready_for_external_pretraining","protocol":"hera_guard_final_dior_full_prepare_v1","dataset_yaml":str((ds/"dataset.yaml").resolve()),"ext_g_yaml":str((ds/"dataset-ext-g.yaml").resolve()),"ext_v_yaml":str((ds/"dataset-ext-v.yaml").resolve()),"audit_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in required},"visual_decision":d}
Path(sys.argv[1]).with_name("prepare_summary.json").write_text(json.dumps(out,indent=2)+"\n")
PY
sha256sum "${OUT}/prepare_summary.json" "${OUT}/visual_decision.json" >"${OUT}/SHA256SUMS"
printf '%s\n' ready_for_external_pretraining >"${STATUS}"
