#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/workspace/xh-202625}
DFINE=${DFINE:-/workspace/third_party/D-FINE-codex-20260830}
TRAIN=${TRAIN:-/workspace/results/DFINE-M-FULL-40EP-AGREEMENT-V1}
OUT=${OUT:-/workspace/results/DFINE-M-FULL-FIXED-BENCHMARKS-V1}
PY=${PY:-/workspace/venvs/p06-cu121/bin/python}
DFINE_SITE=${DFINE_SITE:-/workspace/venvs/dfine-cu121/lib/python3.10/site-packages}
PRIMARY=${PRIMARY:-/workspace/y5_full_s_sanitized.pt}
LARGE=${LARGE:-/root/autodl-tmp/results/Y5-FULL-L-20260829-R1/runs/foundation/weights/last.pt}
HARD=${HARD:-/root/autodl-tmp/pseudo10k-trial-mix-local}
SENTINEL=${SENTINEL:-/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1}
STATUS=${OUT}/status.txt
mkdir -p "${OUT}/logs" "${OUT}/configs"

failed() {
  code=$?
  printf 'failed_exit_%s\n' "${code}" > "${STATUS}"
  exit "${code}"
}
trap failed ERR
printf 'waiting_for_full_checkpoint\n' > "${STATUS}"
while test "$(cat "${TRAIN}/status.txt" 2>/dev/null || true)" != complete; do
  current=$(cat "${TRAIN}/status.txt" 2>/dev/null || true)
  case "${current}" in
    failed*) printf 'blocked_upstream_%s\n' "${current}" > "${STATUS}"; exit 2 ;;
  esac
  sleep 30
done

CHECKPOINT=${TRAIN}/training/last.pth
test -f "${CHECKPOINT}"
test "$(wc -l < "${TRAIN}/training/log.txt")" -eq 40
test -f "${PRIMARY}"
test -f "${LARGE}"
PRIMARY_SHA=$(sha256sum "${PRIMARY}" | awk '{print $1}')
test "${PRIMARY_SHA}" = f7e30fac3391d7048314d1c41df0e878a4d5f0423e5c55b68ee7df5917418229
LARGE_SHA=$(sha256sum "${LARGE}" | awk '{print $1}')
test "${LARGE_SHA}" = 5124b4070b8b847e8385aaafea69ccbaa227ce744525eeb440cb2beb88e2d348
DFINE_SHA=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')

cp "${PROJECT}/submission/docker/configs/y5_full_s_safe_1024_thr015.json" \
  "${OUT}/configs/base.json"
cp "${PROJECT}/submission/docker/configs/y5_full_s_safe_1024_thr015.json" \
  "${OUT}/configs/large.json"
cp "${PROJECT}/submission/docker/configs/y5_dfine_vehicle_agreement_0055.json" \
  "${OUT}/configs/dual_vehicle.json"
cp "${PROJECT}/submission/docker/configs/y5_dfine_aircraft_vehicle_agreement_v1.json" \
  "${OUT}/configs/dual_airvehicle.json"
sed "s#/app/vendor/dfine#${DFINE}#g" \
  "${PROJECT}/configs/experiments/dfine_m_deploy.yml" \
  > "${OUT}/configs/dfine.yml"

"${PY}" - "${OUT}/configs/base.json" "${OUT}/configs/large.json" \
  "${OUT}/configs/dual_vehicle.json" \
  "${OUT}/configs/dual_airvehicle.json" \
  "${PRIMARY}" "${PRIMARY_SHA}" "${DFINE}" "${OUT}/configs/dfine.yml" \
  "${CHECKPOINT}" "${DFINE_SHA}" <<'PY'
import json,sys
base_path,large_path,vehicle_path,airvehicle_path,primary,primary_sha,root,cfg,weight,weight_sha=sys.argv[1:]
for path in (base_path,large_path,vehicle_path,airvehicle_path):
    doc=json.load(open(path))
    doc["model"]["weight_path"]=primary
    doc["model"]["expected_sha256"]=primary_sha
    if "agreement_model" in doc:
        doc["agreement_model"].update({
            "root_path":root,
            "config_path":cfg,
            "weight_path":weight,
            "expected_sha256":weight_sha,
        })
    with open(path,"w") as f: json.dump(doc,f,indent=2); f.write("\n")
PY

export PYTHONPATH="${PROJECT}/src:${DFINE_SITE}"
# Keep the repository root out of ``sys.path[0]``.  Long-lived training servers may
# contain an old editable/install copy or a stale top-level ``rsdet`` directory;
# every benchmark subprocess must import the frozen checkout under PROJECT/src.
cd /tmp
"${PY}" - "${PROJECT}/src" <<'PY'
import inspect,sys
expected=sys.argv[1]
from rsdet.pipeline.large_image import PipelineConfig
from rsdet.submission import competition
pipeline_path=inspect.getfile(PipelineConfig)
competition_path=competition.__file__
if not pipeline_path.startswith(expected) or not competition_path.startswith(expected):
    raise SystemExit(f"mixed rsdet import: pipeline={pipeline_path}, competition={competition_path}")
if "score_threshold_by_coarse" not in inspect.signature(PipelineConfig).parameters:
    raise SystemExit("PipelineConfig is missing score_threshold_by_coarse")
print(f"RSDET_IMPORT_GATE_PASS pipeline={pipeline_path} competition={competition_path}")
PY
printf 'hard10k_inference\n' > "${STATUS}"
for condition in hard10k sentinel; do
  if test "${condition}" = hard10k; then ROOT=${HARD}; else ROOT=${SENTINEL}; fi
  # The historical Y5-L artifact was saved under an incompatible NumPy RNG
  # pickle environment.  It is already a stopped negative route and is not an
  # input to the frozen D-FINE decision, so it must not block the teacher audit.
  for route in base dual_vehicle dual_airvehicle; do
    RUN=${OUT}/${condition}/${route}
    ROUTE_WEIGHT=${PRIMARY}
    if test ! -f "${RUN}/official_metrics.json"; then
      "${PY}" "${PROJECT}/scripts/run_cv3_oof_pseudo_eval.py" \
        --pseudo-root "${ROOT}" \
        --config "${OUT}/configs/${route}.json" \
        --weights "${ROUTE_WEIGHT}" "${ROUTE_WEIGHT}" "${ROUTE_WEIGHT}" \
        --output-dir "${RUN}" > "${OUT}/logs/${condition}-${route}-infer.log" 2>&1
      LATENCY=$("${PY}" - "${RUN}/run_summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(sum(i["wall_seconds"] for i in x["folds"])/sum(i["images"] for i in x["folds"]))
PY
      )
      "${PY}" "${PROJECT}/scripts/evaluate.py" \
        --gt "${ROOT}/ground_truth.json" \
        --pred "${RUN}/predictions.json" \
        --project-config "${PROJECT}/configs/project.yaml" \
        --latency-seconds "${LATENCY}" \
        --output "${RUN}/official_metrics.json" \
        > "${OUT}/logs/${condition}-${route}-eval.log" 2>&1
    fi
  done
  printf '%s_complete\n' "${condition}" > "${STATUS}"
done

"${PY}" "${PROJECT}/scripts/compare_dfine_fixed_benchmarks.py" \
  --hard-base "${OUT}/hard10k/base/official_metrics.json" \
  --hard-dual "${OUT}/hard10k/dual_vehicle/official_metrics.json" \
  --sentinel-base "${OUT}/sentinel/base/official_metrics.json" \
  --sentinel-dual "${OUT}/sentinel/dual_vehicle/official_metrics.json" \
  --hard-airvehicle "${OUT}/hard10k/dual_airvehicle/official_metrics.json" \
  --sentinel-airvehicle "${OUT}/sentinel/dual_airvehicle/official_metrics.json" \
  --output "${OUT}/decision.json" > "${OUT}/logs/decision.log"

find "${OUT}" -type f ! -name SHA256SUMS.txt -print0 | sort -z | \
  xargs -0 sha256sum > "${OUT}/SHA256SUMS.txt"
trap - ERR
printf 'complete\n' > "${STATUS}"
