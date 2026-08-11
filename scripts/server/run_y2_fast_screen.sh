#!/usr/bin/env bash
set -euo pipefail

# Paired YOLO26-s versus YOLO26-s-P2 screen.  This is an exploratory gate,
# never a substitute for the formal three-fold/fixed-160 protocol.

FOLD="${1:-0}"
if [[ "$FOLD" != "0" && "$FOLD" != "1" ]]; then
  echo "usage: $0 [0|1]" >&2
  exit 2
fi

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

CV3="$PROJECT_ROOT/data/splits/cv3_airport_proxy_k60_v2.json"
RUN_ROOT="$RESULTS_ROOT/Y2-FAST-SCREEN-FOLD${FOLD}"
STATUS="$RESULTS_ROOT/Y2-FAST-SCREEN-FOLD${FOLD}.status"
ARCHIVE="$RESULTS_ROOT/Y2-FAST-SCREEN-FOLD${FOLD}-return-no-checkpoints.tar.gz"

on_exit() {
  code=$?
  if [[ "$code" -ne 0 ]]; then
    {
      echo "status=failed"
      echo "fold=$FOLD"
      echo "exit_code=$code"
      echo "failed_at=$(date -Is)"
    } > "$STATUS"
  fi
}
trap on_exit EXIT

if [[ -e "$RUN_ROOT" || -e "$ARCHIVE" ]]; then
  echo "screen output exists; refuse overwrite/resume" >&2
  exit 4
fi

require_sha() {
  expected="$1"
  path="$2"
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA mismatch: $path expected=$expected actual=$actual" >&2
    exit 3
  }
}

require_sha 27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331 "$CV3"
require_sha 646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b "$PRETRAINED_WEIGHT"
require_sha 03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a "$DATA_LOCK"
require_sha f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e "$P02_MANIFEST"
require_sha a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128 "$FORMAL_CROP"

mkdir -p "$RUN_ROOT/input-gates"
{
  echo "status=running"
  echo "fold=$FOLD"
  echo "started_at=$(date -Is)"
} > "$STATUS"

cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH=src "$PYTHON_BIN" scripts/lock_formal_detection_data.py verify \
  --config configs/experiments/formal_detection_data_lock.json \
  --data-root "$DATA_ROOT" \
  --cv3-manifest "$CV3" \
  --p02-manifest "$P02_MANIFEST" \
  --formal-crop-manifest "$FORMAL_CROP" \
  --lock "$DATA_LOCK" \
  --expected-lock-sha256 03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a \
  --report "$RUN_ROOT/input-gates/data_lock_verification.json"

PYTHONPATH=src "$PYTHON_BIN" scripts/yolo_fast_screen.py prepare \
  --cv3-manifest "$CV3" \
  --fold "$FOLD" \
  --output "$RUN_ROOT/split_view.json"

sha256sum \
  scripts/yolo_fast_screen.py \
  scripts/materialize_cv3_oof_config.py \
  scripts/server/run_y2_fast_screen.sh \
  configs/experiments/yolo_fast_screen_m1_fold.template.yaml \
  configs/experiments/yolo_fast_screen_y2_p2_fold.template.yaml \
  configs/experiments/yolo_fast_screen_infer.template.yaml \
  src/rsdet/analysis/oof_detection.py \
  src/rsdet/evaluation/official_metric.py > "$RUN_ROOT/CODE_SHA256.txt"
git rev-parse HEAD > "$RUN_ROOT/GIT_COMMIT.txt"
git status --short > "$RUN_ROOT/GIT_STATUS.txt"
"$PYTHON_BIN" -m pip freeze > "$RUN_ROOT/PIP_FREEZE.txt"
nvidia-smi > "$RUN_ROOT/NVIDIA_SMI.txt"

for key in M1S Y2S; do
  if [[ "$key" == "M1S" ]]; then
    template="configs/experiments/yolo_fast_screen_m1_fold.template.yaml"
  else
    template="configs/experiments/yolo_fast_screen_y2_p2_fold.template.yaml"
  fi
  candidate_root="$RUN_ROOT/$key"
  mkdir -p "$candidate_root"
  PYTHONPATH=src "$PYTHON_BIN" scripts/materialize_cv3_oof_config.py \
    --template "$template" \
    --output "$candidate_root/resolved_train.yaml" \
    --fold "$FOLD" \
    --data-root "$DATA_ROOT" \
    --split-view "$RUN_ROOT/split_view.json" \
    --fold-output-dir "$candidate_root" \
    --pretrained-weight "$PRETRAINED_WEIGHT"
  PYTHONPATH=src "$PYTHON_BIN" scripts/yolo_fast_screen.py train \
    --config "$candidate_root/resolved_train.yaml"

  last="$candidate_root/runs/screen/weights/last.pt"
  PYTHONPATH=src "$PYTHON_BIN" scripts/materialize_cv3_oof_config.py \
    --template configs/experiments/yolo_fast_screen_infer.template.yaml \
    --output "$candidate_root/resolved_infer.yaml" \
    --fold "$FOLD" \
    --data-root "$DATA_ROOT" \
    --split-view "$RUN_ROOT/split_view.json" \
    --fold-output-dir "$candidate_root" \
    --checkpoint "$last" \
    --candidate-key "$key"
  PYTHONPATH=src "$PYTHON_BIN" scripts/yolo_fast_screen.py infer \
    --config "$candidate_root/resolved_infer.yaml"
done

PYTHONPATH=src "$PYTHON_BIN" scripts/yolo_fast_screen.py evaluate \
  --project-config configs/project.yaml \
  --cv3-manifest "$CV3" \
  --formal-crop "$FORMAL_CROP" \
  --fold "$FOLD" \
  --baseline-predictions "$RUN_ROOT/M1S/predictions_low.json" \
  --candidate-predictions "$RUN_ROOT/Y2S/predictions_low.json" \
  --output "$RUN_ROOT/screening_result.json"

tar --exclude='*.pt' --exclude='prepared_data' -czf "$ARCHIVE" \
  -C "$RESULTS_ROOT" "$(basename "$RUN_ROOT")"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
{
  echo "status=complete"
  echo "fold=$FOLD"
  echo "completed_at=$(date -Is)"
  echo "decision=$RUN_ROOT/screening_result.json"
  echo "archive=$ARCHIVE"
} > "$STATUS"
echo "Y2 fast screen complete: $ARCHIVE"
