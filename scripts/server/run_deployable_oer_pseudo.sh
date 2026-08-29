#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/xh-cv3-pseudo}
RESULT_ROOT=${RESULT_ROOT:-/root/autodl-tmp/results}
PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/p06-cu121/bin/python}
P03_ROOT=${P03_ROOT:-/workspace/results/P03-FORMAL-CV3-V2}
IMAGENET_WEIGHT=${IMAGENET_WEIGHT:-/root/autodl-tmp/pretrained/convnext_tiny-983f1562.pth}
MODEL_ROOT=${MODEL_ROOT:-$RESULT_ROOT/DEPLOYABLE-OER-METADATA12-V1}
OUTPUT_ROOT=${OUTPUT_ROOT:-$RESULT_ROOT/CV3-OOF-PSEUDO-DEPLOYABLE-OER-V1}
TWO_VIEW_ROOT=${TWO_VIEW_ROOT:-$RESULT_ROOT/CV3-OOF-PSEUDO-TWO-VIEW-V1/y5_oof_safe1024_rot90cwtta_floor0p001}

mkdir -p "$OUTPUT_ROOT"
STATUS="$OUTPUT_ROOT/status.txt"
printf '%s\n' running > "$STATUS"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" - <<'PY'
import joblib
import sklearn

print(f"dependency_check: sklearn={sklearn.__version__} joblib={joblib.__version__}")
PY

run_domain() {
  local domain=$1
  local pseudo_root gt pred output
  if [[ "$domain" == natural ]]; then
    pseudo_root="$RESULT_ROOT/CV3-OOF-PSEUDO10K-V1"
  else
    pseudo_root="$RESULT_ROOT/CV3-OOF-PSEUDO10K-TRIAL-MIX-V1"
  fi
  gt="$pseudo_root/ground_truth.json"
  pred="$TWO_VIEW_ROOT/$domain/predictions.json"
  output="$OUTPUT_ROOT/$domain"
  mkdir -p "$output"

  if [[ -s "$output/p03-evidence/predictions_r3_fused.json" && \
        -s "$output/p03-evidence/proposal_scores.csv" && \
        -s "$output/p03-evidence/summary.json" ]]; then
    printf 'p03-evidence-skip:%s\n' "$domain" > "$STATUS"
  else
    printf 'p03-evidence:%s\n' "$domain" > "$STATUS"
    "$PYTHON_BIN" scripts/rerank_cv3_pseudo_with_crop.py \
      --gt "$gt" \
      --pred "$pred" \
      --pseudo-root "$pseudo_root" \
      --checkpoint-pattern "$P03_ROOT/ft-tight-224-fold{fold}/final_checkpoint.pt" \
      --imagenet-weight "$IMAGENET_WEIGHT" \
      --output-dir "$output/p03-evidence" \
      --device cuda \
      > "$output/p03-evidence.log"
  fi

  printf 'oer:%s\n' "$domain" > "$STATUS"
  "$PYTHON_BIN" scripts/rerank_cv3_pseudo_with_deployable_oer.py \
    --gt "$gt" \
    --pred "$output/p03-evidence/predictions_r3_fused.json" \
    --model-pattern "$MODEL_ROOT/oer_heldout_fold{fold}.joblib" \
    --output-dir "$output/oer" \
    > "$output/oer.log"

  printf 'frontier:%s\n' "$domain" > "$STATUS"
  "$PYTHON_BIN" scripts/analyze_cv3_oof_pseudo_frontier.py \
    --gt "$gt" \
    --pred "$output/oer/predictions_oer_aircraft_nms.json" \
    --output "$output/frontier.json" \
    > "$output/frontier.log"
  "$PYTHON_BIN" scripts/analyze_cv3_oof_pseudo_coarse_thresholds.py \
    --gt "$gt" \
    --pred "$output/oer/predictions_oer_aircraft_nms.json" \
    --output "$output/coarse.json" \
    > "$output/coarse.log"
}

run_domain natural
run_domain trial
printf '%s\n' complete > "$STATUS"
