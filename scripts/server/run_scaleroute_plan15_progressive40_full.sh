#!/usr/bin/env bash
set -Eeuo pipefail

# Unique full-data candidate admitted by the Plan-15 paired CV3 experiment:
# mature S1024 full checkpoint -> 40 epoch, low-LR, no-mosaic S1280 adaptation.
# The driver is intentionally fresh-output-only and does not build a Docker image.

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625-capscale}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
SOURCE=${SOURCE:-/root/autodl-tmp/results/Y5-FULL-S-20260829-R1/y5_full_s_sanitized.pt}
DATA=${DATA:-/root/autodl-tmp/results/Y5-FULL-S1280-3GPU-R1/dataset_full.yaml}
TRAIN_LIST=${TRAIN_LIST:-/root/autodl-tmp/results/Y5-FULL-S1280-3GPU-R1/all_train_images.txt}
SOURCE_MANIFEST=${SOURCE_MANIFEST:-/root/autodl-tmp/capscale-assets/split_view.json}
OOF_FRONTIER=${OOF_FRONTIER:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1/aggregate/crossfit_frontier.json}
BACKGROUND=${BACKGROUND:-/root/autodl-tmp/assets/MACROSHIFT-BACKGROUND-100MP-FROZEN}
HARD=${HARD:-/root/autodl-tmp/pseudo10k-trial-mix-local}
OUT=${OUT:-/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-V1}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
RESUME_SHA=${RESUME_SHA:-}
ORIGINAL_CONTRACT=${ORIGINAL_CONTRACT:-}
STATUS=${OUT}/status.txt
DEPLOYMENT_THRESHOLD=0.536

SOURCE_SHA=f7e30fac3391d7048314d1c41df0e878a4d5f0423e5c55b68ee7df5917418229
DATA_SHA=f25a9c8448235b42ac805781c191f99be432f3fdce96f684857abd38a3382579
TRAIN_LIST_SHA=30d7bac4bbdc5069becf5b54d9cc3cf89f348459584f3713799dc072572dcb19
SOURCE_MANIFEST_SHA=a647ce030fa832aadc6a6c286a3f6464ac1783f71797a52cc598ec340f128943
OOF_FRONTIER_SHA=545e02b2d252909400ff5cf8f9ea7768bb8438dd99c6e26269fc0807132c81be
BACKGROUND_MANIFEST_SHA=ed3cbbe6952ea5a7792821a316bd3b0ed93888f74a50eda2630f630c9c9020e7

if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}"
exec 9>"${OUT}.lock"
flock -n 9 || exit 4
failed() {
  code=$?
  printf 'failed_exit_%s\n' "${code}" >"${STATUS}"
  exit "${code}"
}
trap failed ERR INT TERM

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

require_sha() {
  local path=$1
  local expected=$2
  test -f "${path}"
  actual=$(sha256sum "${path}" | awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    printf 'SHA mismatch: %s expected=%s actual=%s\n' "${path}" "${expected}" "${actual}" >&2
    return 1
  fi
}

printf 'validating_frozen_inputs\n' >"${STATUS}"
require_sha "${SOURCE}" "${SOURCE_SHA}"
require_sha "${DATA}" "${DATA_SHA}"
require_sha "${TRAIN_LIST}" "${TRAIN_LIST_SHA}"
require_sha "${SOURCE_MANIFEST}" "${SOURCE_MANIFEST_SHA}"
require_sha "${OOF_FRONTIER}" "${OOF_FRONTIER_SHA}"
require_sha "${BACKGROUND}/background_100mp_manifest.jsonl" "${BACKGROUND_MANIFEST_SHA}"
test -f "${HARD}/ground_truth.json"

"${PY}" - "${OOF_FRONTIER}" "${DEPLOYMENT_THRESHOLD}" "${OUT}/frozen_input_contract.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

frontier_path = Path(sys.argv[1])
expected_threshold = float(sys.argv[2])
output_path = Path(sys.argv[3])
frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
actual_threshold = float(frontier["frontiers"]["0.150"]["pooled_oracle"]["threshold"])
if actual_threshold != expected_threshold:
    raise SystemExit(
        f"deployment threshold mismatch: expected={expected_threshold} actual={actual_threshold}"
    )
payload = {
    "schema_version": "scaleroute_plan15_progressive40_full_contract_v1",
    "status": "frozen_inputs_validated",
    "source_recipe": "S1024_full_160e_to_S1280_progressive_40e",
    "external_training_data": False,
    "training_images": 4481,
    "deployment_threshold_source": "P40_CV3_pooled_oracle_at_platform_fdr_0.150",
    "deployment_threshold": actual_threshold,
    "checkpoint_selection": "fixed_last",
    "docker_packaging_authorized": False,
    "official_submission_authorized": False,
}
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

printf 'training_progressive40_full\n' >"${STATUS}"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  require_sha "${RESUME_CHECKPOINT}" "${RESUME_SHA}"
  test -f "${ORIGINAL_CONTRACT}"
  printf 'training_progressive40_full_3gpu_resume_epoch3\n' >"${STATUS}"
  "${PY}" scripts/resume_progressive_resolution_ddp.py \
    --checkpoint "${RESUME_CHECKPOINT}" --expected-sha256 "${RESUME_SHA}" \
    --original-contract "${ORIGINAL_CONTRACT}" --output "${OUT}/adaptation" \
    >"${OUT}/train.log" 2>&1
else
"${PY}" scripts/train_progressive_resolution_adaptation.py \
  --weights "${SOURCE}" --data "${DATA}" --output "${OUT}/adaptation" \
  --epochs 40 --imgsz 1280 --batch 8 --workers 8 --device cuda:0 \
  --seed 42 --lr0 0.0002 --lrf 0.10 --rotate90-p 1.0 \
  >"${OUT}/train.log" 2>&1
fi

CHECKPOINT=${OUT}/adaptation/runs/resolution_adaptation/weights/last.pt
RESULTS=${OUT}/adaptation/runs/resolution_adaptation/results.csv
test -f "${CHECKPOINT}"
test -f "${OUT}/adaptation/training_result.json"
test "$(wc -l <"${RESULTS}")" -eq 41
sha256sum "${CHECKPOINT}" >"${OUT}/checkpoint.sha256"

printf 'background_100mp_low_floor_inference\n' >"${STATUS}"
mkdir -p "${OUT}/background_100mp"
"${PY}" scripts/infer_yolo_background_100mp.py \
  --manifest "${BACKGROUND}/background_100mp_manifest.jsonl" \
  --root "${BACKGROUND}" --checkpoint "${CHECKPOINT}" \
  --output "${OUT}/background_100mp/predictions_low.json" \
  --imgsz 1280 --device cuda:0 --batch-size 16 \
  --confidence 0.001 --iou 0.70 --max-detections 500 \
  >"${OUT}/background_100mp/inference.log" 2>&1

printf 'background_100mp_frozen_threshold_evaluation\n' >"${STATUS}"
"${PY}" scripts/analyze_background_resolution_route.py \
  --manifest "${BACKGROUND}/background_100mp_manifest.jsonl" \
  --primary-pred "${OUT}/background_100mp/predictions_low.json" \
  --expert-pred "${OUT}/background_100mp/predictions_low.json" \
  --primary-labels 0-23 --expert-labels 24 \
  --primary-threshold "${DEPLOYMENT_THRESHOLD}" \
  --expert-threshold "${DEPLOYMENT_THRESHOLD}" \
  --project-config configs/project.yaml \
  --output "${OUT}/background_100mp/frozen_threshold_result.json" \
  >"${OUT}/background_100mp/evaluation.log" 2>&1

# This is timing-only. Passing the same full checkpoint for all fold slots is
# deliberate; predictions from a full-data model are not used as validation.
printf 'timing_only_hard_pseudo10k\n' >"${STATUS}"
"${PY}" scripts/run_multifamily_cv3_pseudo_eval.py \
  --pseudo-root "${HARD}" --family yolo \
  --weights "${CHECKPOINT}" "${CHECKPOINT}" "${CHECKPOINT}" \
  --output-dir "${OUT}/timing_only_hard" --score-floor 0.001 \
  --batch-size 4 --device cuda:0 --imgsz 1280 --tile-size 1024 --overlap 256 \
  >"${OUT}/timing_only_hard.log" 2>&1

printf 'finalizing_validation_artifacts\n' >"${STATUS}"
"${PY}" scripts/finalize_scaleroute_progressive_full.py \
  --training-result "${OUT}/adaptation/training_result.json" \
  --results-csv "${RESULTS}" \
  --background-result "${OUT}/background_100mp/frozen_threshold_result.json" \
  --background-runtime "${OUT}/background_100mp/predictions_low.runtime.json" \
  --timing-summary "${OUT}/timing_only_hard/run_summary.json" \
  --input-contract "${OUT}/frozen_input_contract.json" \
  --output "${OUT}/validation_summary.json"

trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum >"${OUT}/SHA256SUMS.txt"
