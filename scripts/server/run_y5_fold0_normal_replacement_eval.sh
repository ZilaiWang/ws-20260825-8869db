#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${WEIGHT:?set WEIGHT}"
: "${OUT:?set OUT}"
: "${BASELINE_ROOT:?set BASELINE_ROOT}"
: "${GROUND_TRUTH:?set GROUND_TRUTH}"

cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUT}/fold0"

BASE_CONFIG="${BASELINE_ROOT}/fold_0/resolved_infer.yaml"
RESOLVED="${OUT}/fold0/resolved_infer.yaml"
PREDICTIONS="${OUT}/fold0/predictions_low.json"

if [[ ! -f "${OUT}/fold0/run_summary.json" ]]; then
  "${PYTHON_BIN}" - "${BASE_CONFIG}" "${WEIGHT}" "${PREDICTIONS}" "${RESOLVED}" <<'PY'
import sys
from pathlib import Path

import yaml

source, weight, predictions, output = map(Path, sys.argv[1:])
payload = yaml.safe_load(source.read_text(encoding="utf-8"))
payload["model"]["checkpoint"] = str(weight.resolve())
payload["model"]["confidence"] = 0.001
payload["output_json"] = str(predictions.resolve())
output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY
  "${PYTHON_BIN}" scripts/infer_cv3_oof.py --config "${RESOLVED}" \
    >"${OUT}/fold0/infer.log" 2>&1
  "${PYTHON_BIN}" - "${WEIGHT}" "${RESOLVED}" "${PREDICTIONS}" \
    "${OUT}/fold0/run_summary.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

weight, config, predictions, output = map(Path, sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
rows = json.loads(predictions.read_text(encoding="utf-8"))
payload = {
    "status": "complete",
    "protocol": "normal_cv3_fold0_replacement_inference_v1",
    "weight": str(weight.resolve()),
    "weight_sha256": sha(weight),
    "config_sha256": sha(config),
    "predictions_sha256": sha(predictions),
    "prediction_count": len(rows),
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
fi

"${PYTHON_BIN}" scripts/merge_coco_predictions.py \
  --input "${PREDICTIONS}" \
  --input "${BASELINE_ROOT}/fold_1/predictions_low.json" \
  --input "${BASELINE_ROOT}/fold_2/predictions_low.json" \
  --output "${OUT}/predictions.json"
"${PYTHON_BIN}" scripts/analyze_cv3_oof_pseudo_frontier.py \
  --gt "${GROUND_TRUTH}" --pred "${OUT}/predictions.json" \
  --output "${OUT}/frontier.json" --threshold-start 0.001 \
  --threshold-stop 0.996 --threshold-step 0.005 \
  --fdr-levels 0.10 0.12 0.15 0.20 >"${OUT}/frontier.log" 2>&1

sha256sum \
  "${OUT}/fold0/run_summary.json" "${OUT}/predictions.json" "${OUT}/frontier.json" \
  >"${OUT}/RESULT_SHA256.txt"
printf '%s\n' complete >"${OUT}/status.txt"
