#!/usr/bin/env bash
set -euo pipefail

# Formal GPU driver for Y2 or its gated Y3 successor.
# Required environment variables are intentionally explicit; no server path is guessed.

MODE="${1:-}"
if [[ "$MODE" != "y2" && "$MODE" != "y3" ]]; then
  echo "usage: $0 y2|y3" >&2
  exit 2
fi

STATUS_FILE=""
on_exit() {
  local exit_code=$?
  if [[ -n "$STATUS_FILE" && "$exit_code" -ne 0 ]]; then
    {
      echo "status=failed"
      echo "mode=$MODE"
      echo "exit_code=$exit_code"
      echo "failed_at=$(date -Is)"
    } > "$STATUS_FILE"
  fi
}
trap on_exit EXIT

required=(
  PROJECT_ROOT DATA_ROOT RESULTS_ROOT PYTHON_BIN PRETRAINED_WEIGHT DATA_LOCK
  P02_MANIFEST FORMAL_CROP
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done

CV3_MANIFEST="$PROJECT_ROOT/data/splits/cv3_airport_proxy_k60_v2.json"
CV3_SHA="27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331"
PRETRAINED_SHA="646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
DATA_LOCK_SHA="03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a"
FORMAL_CROP_SHA="a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128"
P02_SHA="f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e"

require_sha() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA mismatch: $path expected=$expected actual=$actual" >&2
    exit 3
  fi
}

require_sha "$CV3_SHA" "$CV3_MANIFEST"
require_sha "$PRETRAINED_SHA" "$PRETRAINED_WEIGHT"
require_sha "$DATA_LOCK_SHA" "$DATA_LOCK"
require_sha "$FORMAL_CROP_SHA" "$FORMAL_CROP"
require_sha "$P02_SHA" "$P02_MANIFEST"

actual_ultralytics="$($PYTHON_BIN -c 'import importlib.metadata; print(importlib.metadata.version("ultralytics"))')"
if [[ "$actual_ultralytics" != "8.4.103" ]]; then
  echo "ultralytics must be 8.4.103, actual=$actual_ultralytics" >&2
  exit 3
fi

if [[ "$MODE" == "y2" ]]; then
  if [[ -z "${M1_CALIBRATION_RESULT:-}" && -z "${M1_AGGREGATE_ROOT:-}" ]]; then
    echo "Y2 requires M1_CALIBRATION_RESULT or M1_AGGREGATE_ROOT" >&2
    exit 2
  fi
  MODEL_KEY="P2"
  MODEL_NAME="yolo26s-p2"
  TEMPLATE="$PROJECT_ROOT/configs/experiments/y2_yolo26s_p2_1024_cv3_oof.template.yaml"
  INFER_TEMPLATE="$PROJECT_ROOT/configs/experiments/y2_yolo26s_p2_1024_cv3_oof_infer.template.yaml"
  RUN_ROOT="$RESULTS_ROOT/Y2-P2-CV3-OOF"
  AGG_ROOT="$RESULTS_ROOT/Y2-P2-CV3-OOF-aggregate"
  CAL_ROOT="$RESULTS_ROOT/Y2-P2-CALIBRATION"
  GATE_ARGS=()
else
  : "${Y2_DECISION:?Y3 requires Y2_DECISION}"
  : "${P2_CALIBRATION_RESULT:?Y3 requires P2_CALIBRATION_RESULT}"
  MODEL_KEY="Y3"
  MODEL_NAME="yolo26s-p2-ibs-pair"
  TEMPLATE="$PROJECT_ROOT/configs/experiments/y3_yolo26_p2_ibs_1024_cv3_oof.template.yaml"
  INFER_TEMPLATE="$PROJECT_ROOT/configs/experiments/y3_yolo26_p2_ibs_1024_cv3_oof_infer.template.yaml"
  RUN_ROOT="$RESULTS_ROOT/Y3-P2-IBS-CV3-OOF"
  AGG_ROOT="$RESULTS_ROOT/Y3-P2-IBS-CV3-OOF-aggregate"
  CAL_ROOT="$RESULTS_ROOT/Y3-P2-IBS-CALIBRATION"
  GATE_ARGS=(--gate-decision "$Y2_DECISION")
fi
STATUS_FILE="$RESULTS_ROOT/${MODEL_KEY}-FORMAL-CV3.status"
{
  echo "status=running"
  echo "mode=$MODE"
  echo "started_at=$(date -Is)"
} > "$STATUS_FILE"

for path in "$RUN_ROOT" "$AGG_ROOT" "$CAL_ROOT"; do
  if [[ -e "$path" ]]; then
    echo "formal output already exists; refuse overwrite/resume: $path" >&2
    exit 4
  fi
done

cd "$PROJECT_ROOT"
PYTHONPATH=src "$PYTHON_BIN" scripts/prepare_cv3_oof.py \
  --manifest "$CV3_MANIFEST" \
  --manifest-sha256 "$CV3_SHA" \
  --output-dir "$RUN_ROOT" \
  --model-key "$MODEL_KEY" \
  --model-family yolo \
  --model-name "$MODEL_NAME" \
  --seed 42 \
  --input-size 1024 \
  --foundation-epochs 160 \
  --low-score-threshold 0.001 \
  --max-detections 500 \
  --pretrained-weight "$PRETRAINED_WEIGHT" \
  --pretrained-weight-sha256 "$PRETRAINED_SHA" \
  --detection-data-lock "$DATA_LOCK" \
  --detection-data-lock-sha256 "$DATA_LOCK_SHA"

code_lock="$RUN_ROOT/CODE_SHA256.txt"
sha256sum \
  src/rsdet/experiments/cv3_oof.py \
  src/rsdet/models/ibs_sampling.py \
  src/rsdet/postprocess/yolo_calibration.py \
  scripts/prepare_cv3_oof.py \
  scripts/materialize_cv3_oof_config.py \
  scripts/y2_p2_runtime.py \
  scripts/y1_crossfit_calibration.py \
  scripts/y2_decide_p2.py \
  scripts/y3_decide_ibs.py \
  scripts/finalize_cv3_oof_fold.py \
  scripts/audit_cv3_oof.py \
  "$TEMPLATE" "$INFER_TEMPLATE" > "$code_lock"
git rev-parse HEAD > "$RUN_ROOT/GIT_COMMIT.txt"
git status --short > "$RUN_ROOT/GIT_STATUS.txt"

for fold in 0 1 2; do
  fold_dir="$RUN_ROOT/fold_$fold"
  mkdir -p "$fold_dir/input-gates"
  PYTHONPATH=src "$PYTHON_BIN" scripts/lock_formal_detection_data.py verify \
    --config configs/experiments/formal_detection_data_lock.json \
    --data-root "$DATA_ROOT" \
    --cv3-manifest "$CV3_MANIFEST" \
    --p02-manifest "$P02_MANIFEST" \
    --formal-crop-manifest "$FORMAL_CROP" \
    --lock "$DATA_LOCK" \
    --expected-lock-sha256 "$DATA_LOCK_SHA" \
    --report "$fold_dir/input-gates/detection_data_lock_verification.json"

  PYTHONPATH=src "$PYTHON_BIN" scripts/materialize_cv3_oof_config.py \
    --template "$TEMPLATE" \
    --output "$fold_dir/resolved_config.yaml" \
    --fold "$fold" \
    --data-root "$DATA_ROOT" \
    --split-view "$fold_dir/split_view.json" \
    --fold-output-dir "$fold_dir" \
    --pretrained-weight "$PRETRAINED_WEIGHT"

  PYTHONPATH=src "$PYTHON_BIN" scripts/y2_p2_runtime.py train \
    --config "$fold_dir/resolved_config.yaml" "${GATE_ARGS[@]}"

  last="$fold_dir/runs/foundation/weights/last.pt"
  PYTHONPATH=src "$PYTHON_BIN" scripts/materialize_cv3_oof_config.py \
    --template "$INFER_TEMPLATE" \
    --output "$fold_dir/resolved_infer.yaml" \
    --fold "$fold" \
    --data-root "$DATA_ROOT" \
    --split-view "$fold_dir/split_view.json" \
    --fold-output-dir "$fold_dir" \
    --checkpoint "$last"

  PYTHONPATH=src "$PYTHON_BIN" scripts/y2_p2_runtime.py infer \
    --config "$fold_dir/resolved_infer.yaml"

  PYTHONPATH=src "$PYTHON_BIN" scripts/finalize_cv3_oof_fold.py \
    --plan "$RUN_ROOT/oof_run_plan.json" \
    --fold "$fold" \
    --train-config "$fold_dir/resolved_config.yaml" \
    --train-summary "$fold_dir/train_summary.json" \
    --infer-config "$fold_dir/resolved_infer.yaml" \
    --environment "$fold_dir/environment.txt" \
    --checkpoint "$last" \
    --predictions "$fold_dir/predictions_low.json" \
    --runtime "$fold_dir/predictions_low.runtime.json" \
    --data-lock-verification "$fold_dir/input-gates/detection_data_lock_verification.json" \
    --output "$fold_dir/fold_metadata.json"
done

PYTHONPATH=src "$PYTHON_BIN" scripts/audit_cv3_oof.py \
  --manifest "$CV3_MANIFEST" \
  --manifest-sha256 "$CV3_SHA" \
  --plan "$RUN_ROOT/oof_run_plan.json" \
  --run-root "$RUN_ROOT" \
  --output-dir "$AGG_ROOT" \
  --formal-crop-manifest "$FORMAL_CROP"

PYTHONPATH=src "$PYTHON_BIN" scripts/y1_crossfit_calibration.py \
  --aggregate-dir "$AGG_ROOT" \
  --formal-crop-manifest "$FORMAL_CROP" \
  --output-dir "$CAL_ROOT"

if [[ "$MODE" == "y2" ]]; then
  if [[ -z "${M1_CALIBRATION_RESULT:-}" ]]; then
    m1_cal_root="$RESULTS_ROOT/Y1-M1-CALIBRATION-REPLAY"
    if [[ ! -e "$m1_cal_root" ]]; then
      PYTHONPATH=src "$PYTHON_BIN" scripts/y1_crossfit_calibration.py \
        --aggregate-dir "$M1_AGGREGATE_ROOT" \
        --formal-crop-manifest "$FORMAL_CROP" \
        --output-dir "$m1_cal_root"
    fi
    M1_CALIBRATION_RESULT="$m1_cal_root/calibration_result.json"
  fi
  if [[ ! -f "$M1_CALIBRATION_RESULT" ]]; then
    echo "M1 calibration result not found: $M1_CALIBRATION_RESULT" >&2
    exit 5
  fi
  PYTHONPATH=src "$PYTHON_BIN" scripts/y2_decide_p2.py \
    --m1-calibration "$M1_CALIBRATION_RESULT" \
    --p2-calibration "$CAL_ROOT/calibration_result.json" \
    --p2-aggregate "$AGG_ROOT" \
    --output "$RESULTS_ROOT/Y2-P2-DECISION.json"
else
  PYTHONPATH=src "$PYTHON_BIN" scripts/y3_decide_ibs.py \
    --p2-calibration "$P2_CALIBRATION_RESULT" \
    --y3-calibration "$CAL_ROOT/calibration_result.json" \
    --y3-aggregate "$AGG_ROOT" \
    --output "$RESULTS_ROOT/Y3-P2-IBS-DECISION.json"
fi

return_archive="$RESULTS_ROOT/${MODEL_KEY}-FORMAL-CV3-return-no-checkpoints.tar.gz"
tar --exclude='*.pt' --exclude='prepared_data' -czf "$return_archive" \
  -C "$RESULTS_ROOT" "$(basename "$RUN_ROOT")" "$(basename "$AGG_ROOT")" \
  "$(basename "$CAL_ROOT")"
sha256sum "$return_archive" > "$return_archive.sha256"
{
  echo "status=complete"
  echo "mode=$MODE"
  echo "completed_at=$(date -Is)"
  echo "aggregate=$AGG_ROOT/oof_metadata.json"
  echo "calibration=$CAL_ROOT/calibration_result.json"
  echo "return_archive=$return_archive"
} > "$STATUS_FILE"
echo "$MODE complete: $return_archive"
