#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-/root/autodl-tmp/xh-202625-capscale}
PY=${PY:-/root/autodl-tmp/venvs/cv3-model-cu121/bin/python}
SOURCE=${SOURCE:-/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1/s1280}
GT=${GT:-/root/autodl-tmp/capscale-assets/instances_val.json}
OUT=${OUT:-/root/autodl-tmp/results/S1280-COARSE-PURITY-FOLD0-V1}
STATUS=${OUT}/status.txt

if [[ -e "${OUT}" ]]; then
  printf 'refusing non-fresh output: %s\n' "${OUT}" >&2
  exit 3
fi
mkdir -p "${OUT}/logs"
failed() { code=$?; printf 'failed_exit_%s\n' "${code}" >"${STATUS}"; exit "${code}"; }
trap failed ERR INT TERM
for path in "${SOURCE}/resolved_infer.yaml" "${SOURCE}/predictions_low.json" \
  "${SOURCE}/frontier.json" "${SOURCE}/runs/foundation/weights/last.pt" "${GT}"; do
  test -f "${path}"
done
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}"
printf 'inference\n' >"${STATUS}"
"${PY}" scripts/materialize_yolo_score_transform_config.py \
  --base "${SOURCE}/resolved_infer.yaml" --output "${OUT}/resolved_infer.yaml" \
  --predictions "${OUT}/predictions_low.json" --transform coarse_purity_sqrt
"${PY}" scripts/infer_cv3_oof.py --config "${OUT}/resolved_infer.yaml" \
  >"${OUT}/logs/infer.log" 2>&1
printf 'evaluation\n' >"${STATUS}"
"${PY}" scripts/analyze_single_split_official_frontier.py \
  --gt "${GT}" --pred "${OUT}/predictions_low.json" \
  --output "${OUT}/frontier.json" --step 0.005 >"${OUT}/logs/frontier.log" 2>&1
"${PY}" scripts/analyze_single_split_paired_scale.py \
  --gt "${GT}" --baseline "${SOURCE}/predictions_low.json" \
  --candidate "${OUT}/predictions_low.json" \
  --baseline-frontier "${SOURCE}/frontier.json" \
  --candidate-frontier "${OUT}/frontier.json" \
  --output-json "${OUT}/paired_diagnosis.json" \
  --output-csv "${OUT}/paired_fine.csv" >"${OUT}/logs/paired.log" 2>&1
sha256sum "${OUT}/predictions_low.json" "${OUT}/frontier.json" \
  "${OUT}/paired_diagnosis.json" "${OUT}/paired_fine.csv" >"${OUT}/RESULT_SHA256.txt"
trap - ERR INT TERM
printf 'complete\n' >"${STATUS}"
