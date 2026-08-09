#!/usr/bin/env bash
# P03-FORMAL-CV3-V2：tight-224 ConvNeXt 正式三折复验
# 来源：docs/server/P03_FORMAL_CV3_V2_REPLAY.md（2026-08-01 冻结）
# 用法：cd /workspace/xh-202625 && bash docs/server/run_p03_formal.sh
set -euo pipefail

cd /workspace/xh-202625
source /workspace/venvs/p03-cu121/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

ROOT=/workspace/results/P03-FORMAL-CV3-V2
P02_PATH_REGISTER=/workspace/results/FORMAL-CV3-CROP-TASK-01/p02_manifest_path.txt
EXP="$(cat "$P02_PATH_REGISTER")"
FORMAL_DIR=/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop
FORMAL="$FORMAL_DIR/formal_crop_manifest.csv"
EXPECTED_FORMAL_SHA=a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128
EXPECTED_EXP_SHA=f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e
mkdir -p "$ROOT/logs"

echo "=== [1/8] 代码 SHA 门禁 ==="
sha256sum -c docs/server/CV3_FORMAL_EXPERIMENT_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/formal-stage-code-sha256.log"

echo "=== [2/8] 环境 preflight + pytest + ruff ==="
{
  date -Is
  git rev-parse HEAD
  git status --short
  nvidia-smi
  python --version
  python -c "import torch,torchvision; print(torch.__version__,torchvision.__version__,torch.version.cuda)"
  df -h /workspace
} 2>&1 | tee "$ROOT/system_preflight.txt"

PYTHONPATH=src pytest -q \
  tests/test_formal_cv3.py tests/test_formal_crop.py \
  tests/test_p03_p04_formal_replay.py \
  tests/test_p03_training_utils.py tests/test_p03_summary_cli.py \
  2>&1 | tee "$ROOT/logs/pytest.log"

ruff check \
  src/rsdet/data/formal_cv3.py src/rsdet/analysis/formal_crop.py \
  src/rsdet/analysis/formal_replay.py \
  scripts/audit_p03_p04_formal_inputs.py \
  scripts/freeze_p03_formal_config.py scripts/train_crop_classifier.py \
  scripts/summarize_p03_p04_formal.py \
  tests/test_formal_cv3.py tests/test_formal_crop.py \
  tests/test_p03_p04_formal_replay.py \
  2>&1 | tee "$ROOT/logs/ruff.log"

echo "=== [3/8] 上游 formal manifest 门禁 ==="
test "$FORMAL" = \
  "/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv"
test -f "$FORMAL" && test -f "$FORMAL_DIR/formal_crop_audit.json" \
  && test -f "$FORMAL_DIR/resolved_config.yaml" || {
  echo "waiting_for_FORMAL_CV3_CROP_TASK_01: run-a formal artifact 缺失或不完整" \
    | tee "$ROOT/logs/formal-upstream-waiting.log" >&2
  exit 2
}
test "$(sha256sum "$FORMAL" | cut -d' ' -f1)" = "$EXPECTED_FORMAL_SHA"
test "$(sha256sum "$EXP" | cut -d' ' -f1)" = "$EXPECTED_EXP_SHA"
printf 'F00_RUN_A_VERIFIED_AND_CONSUMED\n' \
  | tee "$ROOT/logs/formal-upstream-accepted.log"

echo "=== [4/8] formal 输入独立审计 + 配置冻结 + 环境检查 ==="
PYTHONPATH=src python scripts/audit_p03_p04_formal_inputs.py \
  --formal-manifest "$FORMAL" \
  --exploratory-manifest "$EXP" \
  --cv3-manifest data/splits/cv3_airport_proxy_k60_v2.json \
  --output "$ROOT/formal_input_audit.json" \
  2>&1 | tee "$ROOT/logs/formal-input-audit.log"

python3 - "$ROOT/formal_input_audit.json" <<'PY'
import json, sys
audit = json.load(open(sys.argv[1]))
assert audit.get("status") == "formal_replay_inputs_ready", audit
print("FORMAL_INPUT_AUDIT_OK")
PY

PYTHONPATH=src python scripts/freeze_p03_formal_config.py \
  --template configs/experiments/p03_formal_cv3_v2.yaml \
  --input-audit "$ROOT/formal_input_audit.json" \
  --output "$ROOT/p03_formal_resolved.yaml" \
  2>&1 | tee "$ROOT/logs/freeze-config.log"

PYTHONPATH=src python scripts/check_p03_environment.py \
  --manifest "$FORMAL" \
  --data-root /workspace/data \
  --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
  --expected-manifest-sha256 "$EXPECTED_FORMAL_SHA" \
  --verify-source-count 4481 \
  --output "$ROOT/environment_check.json" \
  2>&1 | tee "$ROOT/logs/environment-check.log"

echo "=== [5/8] fold0 smoke ==="
PYTHONPATH=src python scripts/train_crop_classifier.py \
  --config "$ROOT/p03_formal_resolved.yaml" \
  --manifest "$FORMAL" \
  --data-root /workspace/data \
  --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
  --output-dir "$ROOT/smoke-fold0" \
  --fold 0 --policy tight --resolution 224 \
  --regime fine_tune --sampler natural --seed 42 \
  --smoke --overwrite \
  2>&1 | tee "$ROOT/logs/smoke.log"

echo "=== [6/8] 三个正式 run ==="
for FOLD in 0 1 2; do
  RUN="$ROOT/ft-tight-224-fold${FOLD}"
  test ! -e "$RUN" || {
    test -f "$RUN/run_summary.json" && continue
    echo "发现不完整 run，停止而非覆盖: $RUN" >&2
    exit 2
  }
  PYTHONPATH=src python scripts/train_crop_classifier.py \
    --config "$ROOT/p03_formal_resolved.yaml" \
    --manifest "$FORMAL" \
    --data-root /workspace/data \
    --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
    --output-dir "$RUN" \
    --fold "$FOLD" --policy tight --resolution 224 \
    --regime fine_tune --sampler natural --seed 42 \
    2>&1 | tee "$ROOT/logs/fold${FOLD}.log"
  test "${PIPESTATUS[0]}" -eq 0 || exit 1
done

echo "=== [7/8] 正式汇总 ==="
PYTHONPATH=src python scripts/summarize_p03_p04_formal.py \
  --stage p03 \
  --runs-root "$ROOT" \
  --input-audit "$ROOT/formal_input_audit.json" \
  --output "$ROOT/formal_summary.json" \
  2>&1 | tee "$ROOT/logs/formal-summary.log"

echo "=== [8/8] 回传包 ==="
find "$ROOT"/ft-tight-224-fold*/final_checkpoint.pt -type f -print0 \
  | sort -z | xargs -0 sha256sum > "$ROOT/CHECKPOINTS_SHA256.txt"

cd /workspace/results
tar --exclude='final_checkpoint.pt' \
  -czf P03-FORMAL-CV3-V2-results-no-checkpoints.tar.gz \
  P03-FORMAL-CV3-V2
sha256sum P03-FORMAL-CV3-V2-results-no-checkpoints.tar.gz
echo "P03_FORMAL_COMPLETE"
