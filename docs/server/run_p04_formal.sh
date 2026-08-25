#!/usr/bin/env bash
# P04-FORMAL-CV3-V2：三教师正式特征复验（18 个 probe）
# 来源：docs/server/P04_FORMAL_CV3_V2_REPLAY.md（2026-08-01 冻结）
# 用法：cd /workspace/xh-202625 && bash docs/server/run_p04_formal.sh
set -euo pipefail

cd /workspace/xh-202625
source /workspace/venvs/p04-cu121/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

ROOT=/workspace/results/P04-FORMAL-CV3-V2
P02_PATH_REGISTER=/workspace/results/FORMAL-CV3-CROP-TASK-01/p02_manifest_path.txt
EXP="$(cat "$P02_PATH_REGISTER")"
FORMAL=/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv
CONV=/workspace/p04-cache/convnext-tight224-d4-v1
DINO=/workspace/p04-cache/dinov2-vitb14-tight224-d4-v1
CLEAN=/workspace/p04-cache/cleandift-tight224-d4-v1
ASSET_LOCK=/workspace/p04-assets/ASSET_LOCK.json
EXPECTED_FORMAL_SHA=a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128
EXPECTED_EXP_SHA=f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e
mkdir -p "$ROOT/logs"

echo "=== [1/6] 代码 SHA 门禁 ==="
sha256sum -c docs/server/CV3_FORMAL_EXPERIMENT_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/formal-stage-code-sha256.log"

echo "=== [2/6] 环境 preflight + pytest ==="
{
  date -Is
  git rev-parse HEAD
  git status --short
  nvidia-smi
  python --version
  python -c "import torch; print(torch.__version__,torch.version.cuda)"
  df -h /workspace
} 2>&1 | tee "$ROOT/system_preflight.txt"

PYTHONPATH=src pytest -q \
  tests/test_formal_cv3.py tests/test_formal_crop.py \
  tests/test_p03_p04_formal_replay.py \
  tests/test_p04_feature_pipeline.py tests/test_p04_feature_cli.py \
  2>&1 | tee "$ROOT/logs/pytest.log"

ruff check \
  src/rsdet/analysis/formal_replay.py src/rsdet/features \
  scripts/audit_p03_p04_formal_inputs.py \
  scripts/train_p04_feature_probe.py scripts/summarize_p03_p04_formal.py \
  tests/test_p03_p04_formal_replay.py tests/test_p04_feature_*.py \
  2>&1 | tee "$ROOT/logs/ruff.log"

echo "=== [3/6] 上游 formal manifest 门禁 ==="
test "$FORMAL" = \
  "/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv"
test -f "$FORMAL" || {
  echo "waiting_for_FORMAL_CV3_CROP_TASK_01: run-a formal manifest 缺失" \
    | tee "$ROOT/logs/formal-upstream-waiting.log" >&2
  exit 2
}
test "$(sha256sum "$FORMAL" | cut -d' ' -f1)" = "$EXPECTED_FORMAL_SHA"
test "$(sha256sum "$EXP" | cut -d' ' -f1)" = "$EXPECTED_EXP_SHA"
test -s "$ASSET_LOCK"

PYTHONPATH=src python scripts/check_p04_environment.py \
  --asset-lock "$ASSET_LOCK" \
  --manifest "$EXP" \
  --expected-manifest-sha256 "$EXPECTED_EXP_SHA" \
  --data-root /workspace/data \
  --expected-gpu "NVIDIA GeForce RTX 4080 SUPER" \
  --verify-source-count 32 \
  --verify-sd-inventory \
  --output "$ROOT/p04_environment_check.json" \
  2>&1 | tee "$ROOT/logs/p04-environment-check.log"

echo "=== [4/6] cache 复用审计（硬门禁） ==="
PYTHONPATH=src python scripts/audit_p03_p04_formal_inputs.py \
  --formal-manifest "$FORMAL" \
  --exploratory-manifest "$EXP" \
  --cv3-manifest data/splits/cv3_airport_proxy_k60_v2.json \
  --data-root /workspace/data \
  --cache "convnext=$CONV" \
  --cache "dinov2b=$DINO" \
  --cache "cleandift=$CLEAN" \
  --asset-lock "$ASSET_LOCK" \
  --cache-identity "convnext=a01c6a127=convnext_tiny_imagenet1k_v1" \
  --cache-identity "dinov2b=d5a1c283=dinov2_vitb14" \
  --cache-identity "cleandift=2d50def4=cleandift_sd15" \
  --output "$ROOT/formal_input_and_cache_reuse_audit.json" \
  2>&1 | tee "$ROOT/logs/formal-cache-audit.log"

python3 - "$ROOT/formal_input_and_cache_reuse_audit.json" <<'PY'
import json, sys
audit = json.load(open(sys.argv[1]))
assert audit.get("status") == "formal_replay_inputs_ready", audit
assert audit.get("cache_count") == 3, audit
for c in audit.get("caches", []):
    assert c.get("object_count") == 20933, c
    assert c.get("row_count") == 167464, c
print("CACHE_REUSE_AUDIT_OK")
PY

echo "=== [5/6] 18 个正式 probe ==="
run_probe () {
  NAME="$1"; CACHE="$2"; FEATURE="$3"; PCA="$4"; FOLD="$5"
  RUN="$ROOT/${NAME}/fold${FOLD}"
  test ! -e "$RUN" || {
    test -f "$RUN/run_summary.json" && return 0
    echo "发现不完整 run，停止而非覆盖: $RUN" >&2
    return 2
  }
  EXTRA=()
  test "$PCA" = "native" || EXTRA=(--pca-dim 384)
  PYTHONPATH=src python scripts/train_p04_feature_probe.py \
    --cache-dir "$CACHE" \
    --feature-name "$FEATURE" \
    --manifest "$FORMAL" \
    --output-dir "$RUN" \
    --fold "$FOLD" \
    --batch-size 96 \
    --epochs 15 \
    --minimum-epochs 15 \
    --patience 0 \
    --min-delta 0 \
    --normalization train_rms \
    --head-init p04_default \
    --checkpoint-selection fixed_epoch_last \
    --seed 42 \
    --reuse-audit "$ROOT/formal_input_and_cache_reuse_audit.json" \
    "${EXTRA[@]}" \
    2>&1 | tee "$ROOT/logs/${NAME}-fold${FOLD}.log"
  test "${PIPESTATUS[0]}" -eq 0
}

for FOLD in 0 1 2; do
  run_probe convnext-native "$CONV" convnext_gap native "$FOLD" || exit 1
  run_probe convnext-pca384 "$CONV" convnext_gap pca384 "$FOLD" || exit 1
  run_probe dino-b-clspatch-native "$DINO" dino_cls_patchmean native "$FOLD" || exit 1
  run_probe dino-b-clspatch-pca384 "$DINO" dino_cls_patchmean pca384 "$FOLD" || exit 1
  run_probe cleandift-map0-native "$CLEAN" clean_map0 native "$FOLD" || exit 1
  run_probe cleandift-map0-pca384 "$CLEAN" clean_map0 pca384 "$FOLD" || exit 1
done

echo "=== [6/6] 正式汇总 + 回传包 ==="
PYTHONPATH=src python scripts/summarize_p03_p04_formal.py \
  --stage p04 \
  --runs-root "$ROOT" \
  --input-audit "$ROOT/formal_input_and_cache_reuse_audit.json" \
  --output "$ROOT/formal_summary.json" \
  2>&1 | tee "$ROOT/logs/formal-summary.log"

find "$ROOT" -name final_checkpoint.pt -print0 \
  | sort -z | xargs -0 sha256sum > "$ROOT/CHECKPOINTS_SHA256.txt"

cd /workspace/results
tar --exclude='final_checkpoint.pt' \
  -czf P04-FORMAL-CV3-V2-results-no-checkpoints.tar.gz \
  P04-FORMAL-CV3-V2
sha256sum P04-FORMAL-CV3-V2-results-no-checkpoints.tar.gz
echo "P04_FORMAL_COMPLETE"
