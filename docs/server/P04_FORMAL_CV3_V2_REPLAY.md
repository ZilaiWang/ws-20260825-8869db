# P04-FORMAL-CV3-V2：三教师正式特征复验

## 0. 唯一目标

复用同一服务器的三个 fold-independent D4 cache，只训练正式 CV3 三折读出：

```text
ConvNeXt       convnext_gap          native 768 + PCA384
DINOv2-B       dino_cls_patchmean    native 1536 + PCA384
CleanDIFT      clean_map0            native 1280 + PCA384
```

每项三折，共 18 个 probe。统一 `train_rms`、`p04_default` head、seed42。
不提取新教师，不跑 DINO-S/CLS、Clean map6/map9、融合、微调或新 seed。

## 1. 固定路径

```bash
set -euo pipefail
cd /workspace/xh-202625
source /workspace/venvs/p04-cu121/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

ROOT=/workspace/results/P04-FORMAL-CV3-V2
P02_PATH_REGISTER=/workspace/results/FORMAL-CV3-CROP-TASK-01/p02_manifest_path.txt
test -s "$P02_PATH_REGISTER" || {
  echo "waiting_for_FORMAL_CV3_CROP_TASK_01: P0-2 路径登记缺失" >&2
  exit 2
}
EXP="$(cat "$P02_PATH_REGISTER")"
FORMAL=/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv
CONV=/workspace/p04-cache/convnext-tight224-d4-v1
DINO=/workspace/p04-cache/dinov2-vitb14-tight224-d4-v1
CLEAN=/workspace/p04-cache/cleandift-tight224-d4-v1
ASSET_LOCK=/workspace/p04-assets/ASSET_LOCK.json
mkdir -p "$ROOT/logs"

sha256sum -c docs/server/CV3_FORMAL_EXPERIMENT_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/formal-stage-code-sha256.log"
```

如实际 cache 路径不同，只允许根据服务器已有 `cache_meta.json/index.json`
定位同 fingerprint 缓存；不得重新提取后冒充复用。

## 2. 环境、代码和上游门禁

```bash
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

test "$FORMAL" = \
  "/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv"
test -f "$FORMAL" || {
  echo "waiting_for_FORMAL_CV3_CROP_TASK_01: run-a formal manifest 缺失" \
    | tee "$ROOT/logs/formal-upstream-waiting.log" >&2
  exit 2
}
test "$(sha256sum "$FORMAL" | cut -d' ' -f1)" = \
  "a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128"
test "$(sha256sum "$EXP" | cut -d' ' -f1)" = \
  "f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e"
test -s "$ASSET_LOCK"

PYTHONPATH=src python scripts/check_p04_environment.py \
  --asset-lock "$ASSET_LOCK" \
  --manifest "$EXP" \
  --expected-manifest-sha256 f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e \
  --data-root /workspace/data \
  --expected-gpu "NVIDIA GeForce RTX 4080 SUPER" \
  --verify-source-count 32 \
  --verify-sd-inventory \
  --output "$ROOT/p04_environment_check.json" \
  2>&1 | tee "$ROOT/logs/p04-environment-check.log"
```

`FORMAL` 必须精确指向公共 F00 的 `run-a`。若文件不存在，停止并报告
`waiting_for_FORMAL_CV3_CROP_TASK_01`；不得生成替代副本，也不得消费
`run-b`。

三个 cache 必须各有 `cache_meta.json/index.json`，原 TASK-01/02/04 cache
audit 必须可读取。任一缺失先停止并报告，不自动新建同名空目录。

## 3. UID/crop/canonical224 全量复用门禁

此步骤会读取原图并重渲染 20,933 个 tight canonical224；三 cache 共用同一
次渲染结果。

```bash
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
```

必须同时满足：

```text
status=formal_replay_inputs_ready
formal SHA=a3bed44f...4128
cache_count=3
每 cache object_count=20933
每 cache row_count=167464
canonical/crop/missing/unknown/incomplete-view mismatch 全为0
每 cache fingerprint 以登记前缀开头，teacher_id 与登记值一致
cache metadata 绑定的 asset-lock SHA 与当前 ASSET_LOCK.json 完全一致
```

审计 JSON 会把实际完整 fingerprint、`cache_meta.json`/`index.json` SHA 和
asset-lock SHA 写入正式产物；任务单中的 8 位前缀只用于选择已审核 cache，
不能替代审计产出的完整身份。

只要任一 cache 有一个 mismatch，整个 18-run 正式矩阵停止；不能只跑剩余
教师，也不能通过忽略对象或 `--allow-cache-subset` 绕过。诊断与重提取必须
另开任务 ID。

## 4. 18 个正式 probe

定义唯一运行函数：

```bash
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
```

训练脚本会为每个 fold 保存：

- `pca_train_only.npz`（仅 PCA 行）；
- `train_rms_train_only.json`；
- checkpoint 中的 manifest/cache/audit SHA；
- logits、predictions、per-class、confusion、history 和 run summary。

PCA、RMS 和线性头都只能读取两个训练 fold。验证仅使用 r0。

## 5. 正式汇总

```bash
PYTHONPATH=src python scripts/summarize_p03_p04_formal.py \
  --stage p04 \
  --runs-root "$ROOT" \
  --input-audit "$ROOT/formal_input_and_cache_reuse_audit.json" \
  --output "$ROOT/formal_summary.json" \
  2>&1 | tee "$ROOT/logs/formal-summary.log"
```

汇总硬门禁：

- 条件集合必须恰为三个教师 × native/PCA384；
- 每条件恰有 folds 0/1/2，总计 18 个；
- 所有 run manifest 和 reuse-audit SHA 一致；
- 每个 feature 必须来自已准入的唯一 cache fingerprint，且原生维度为
  768/1536/1280，PCA 行输出必须为 384D；
- resolved runtime 必须仍是固定 15 epoch、batch 96、seed42、
  `checkpoint_selection=fixed_epoch_last`、train-RMS 和 `p04_default`
  head；held-out fold 不参与逐 epoch 选 checkpoint 或 early stop，只在
  训练完成后评估一次；
- 每折 RMS、PCA（若有）和 head 均标记 train-only；
- 三折 `predictions.csv` 的 UID/crop/class 必须逐行对齐 formal manifest，
  并与 logits 的 labels/argmax 一致；
- 保存指标必须由 logits 独立复算一致；每条件 pooled OOF 恰有 20,933
  个不同对象；
- 输出逐类、三大类、固定 9/8/8 tier，以及 TU-160 压力折。

native 与 PCA384 分开排名。服务器不自行宣布 CleanDIFT 入选；只回报数据。

## 6. 回传与保留

```bash
find "$ROOT" -name final_checkpoint.pt -print0 \
  | sort -z | xargs -0 sha256sum > "$ROOT/CHECKPOINTS_SHA256.txt"

cd /workspace/results
tar --exclude='final_checkpoint.pt' \
  -czf P04-FORMAL-CV3-V2-results-no-checkpoints.tar.gz \
  P04-FORMAL-CV3-V2
sha256sum P04-FORMAL-CV3-V2-results-no-checkpoints.tar.gz
```

保留三个旧 cache 和 18 个小 checkpoint，直到本地验收。最终回报包括：
3 cache fingerprint/行数/SHA、canonical 门禁、18/18 run、六组逐折与
mean±std、pooled OOF、native/PCA384 排名、TU-160、耗时、显存和回传包 SHA。
