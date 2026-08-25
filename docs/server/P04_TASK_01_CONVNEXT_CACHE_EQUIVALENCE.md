# P04 服务器任务 01：ConvNeXt D4 缓存与 P03 等价门禁

## 0. 任务边界

前置：P04-TASK-00 `complete`，`ASSET_LOCK.json` 、真实模型 smoke 和 Clean repeat 全部通过。

本任务只回答：“离线 D4 特征缓存是否保留 P03 frozen ConvNeXt 的训练语义？”它分成两组：

1. **P03 等价组**：不做 L2，复制 P03 head 初始化随机数消耗，只用于工程门禁；
2. **P04 主协议组**：L2 normalization + P04 统一 head init，作为后续 DINO/Clean 的公平 R0 基线。

两组不得混合汇总。P03 等价组不通过时立即停止，不运行 DINO/Clean。

## 1. 冻结输入与路径

```text
repo       /workspace/xh-202625
venv       /workspace/venvs/p04-cu121
manifest   /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv
data       /workspace/data
asset lock /workspace/p04-assets/ASSET_LOCK.json
cache      /workspace/p04-cache/convnext-tight224-d4-v1
results    /workspace/results/P04-TASK-01
```

冻结条件：`tight-224` / 20,933 对象 / D4 8 视图 / natural / seed=42 / identity validation / fp16 cache。

P03 tight-224 frozen 三折 macro recall：

```text
fold0 0.8651274625503599
fold1 0.8450608179269498
fold2 0.8810256037629665
mean  0.8637379614134254
```

## 2. 预检与代码门禁

```bash
cd /workspace/xh-202625
source /workspace/venvs/p04-cu121/bin/activate
set -o pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 XFORMERS_DISABLED=1
mkdir -p /workspace/results/P04-TASK-01/logs /workspace/p04-cache

sha256sum -c docs/server/P04_CODE_SHA256.txt \
  2>&1 | tee /workspace/results/P04-TASK-01/logs/code-sha256.log

PYTHONPATH=src pytest -q \
  tests/test_p04_feature_pipeline.py tests/test_p04_feature_cli.py \
  2>&1 | tee /workspace/results/P04-TASK-01/logs/pytest.log

ruff check src/rsdet/features scripts/*p04*.py tests/test_p04_feature_*.py \
  2>&1 | tee /workspace/results/P04-TASK-01/logs/ruff.log

python scripts/check_p04_environment.py \
  --asset-lock /workspace/p04-assets/ASSET_LOCK.json \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --expected-manifest-sha256 f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e \
  --data-root /workspace/data \
  --expected-gpu "NVIDIA GeForce RTX 4080 SUPER" \
  --verify-source-count 32 \
  --output /workspace/results/P04-TASK-01/environment_check.json
```

另存 `nvidia-smi`、`df -h /workspace`、Git commit/dirty 到 `system_preflight.txt`。任一门禁失败立即停止。

## 3. 全量 D4 特征缓存

```bash
python scripts/extract_p04_features.py \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --data-root /workspace/data \
  --asset-lock /workspace/p04-assets/ASSET_LOCK.json \
  --output-dir /workspace/p04-cache/convnext-tight224-d4-v1 \
  --teacher convnext_tiny \
  --views d4 \
  --batch-size 96 \
  --shard-size 2048 \
  --compute-dtype float16 \
  --storage-dtype float16 \
  2>&1 | tee /workspace/results/P04-TASK-01/logs/extract-convnext.log
```

预期 `20,933×8=167,464` 行、`convnext_gap=768D`。执行：

```bash
python scripts/audit_p04_feature_cache.py \
  --cache-dir /workspace/p04-cache/convnext-tight224-d4-v1 \
  --expected-objects 20933 \
  --expected-rows 167464 \
  --expected-feature convnext_gap=768 \
  --output /workspace/results/P04-TASK-01/convnext-cache-audit.json
```

审计必须是 `pass`，唯一行 key 必须等于 167,464。再原命令重跑一次，必须全部显示 `SKIP shard`；否则 resume 门禁失败。

## 4. P03 等价组（硬门禁）

```bash
for FOLD in 0 1 2; do
  python scripts/train_p04_feature_probe.py \
    --cache-dir /workspace/p04-cache/convnext-tight224-d4-v1 \
    --feature-name convnext_gap \
    --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
    --output-dir "/workspace/results/P04-TASK-01/equivalence/fold${FOLD}" \
    --fold "$FOLD" \
    --batch-size 96 \
    --normalization none \
    --head-init p03_convnext_compat \
    --seed 42 \
    2>&1 | tee "/workspace/results/P04-TASK-01/logs/equivalence-fold${FOLD}.log"
  test "${PIPESTATUS[0]}" -eq 0 || exit 1
done

python scripts/summarize_p04_probes.py \
  --runs-root /workspace/results/P04-TASK-01/equivalence \
  --output /workspace/results/P04-TASK-01/equivalence_summary.json \
  --expected-fold-macro-recall 0.8651274625503599,0.8450608179269498,0.8810256037629665 \
  --equivalence-tolerance 0.003 \
  2>&1 | tee /workspace/results/P04-TASK-01/logs/equivalence-summary.log
```

硬门禁是三折均值绝对差 `<=0.003`。同时报每折 delta；即使均值通过，单折绝对差 `>0.01` 也必须单独标记并排查，不得隐藏。

如失败，优先检查：D4 选样、autocast、fp16 cache、ConvNeXt `classifier[:-1]`、head init、scheduler。不得放宽阈值或调学习率。

## 5. P04 R0 主协议基线

只在第 4 节通过后执行：

```bash
for FOLD in 0 1 2; do
  python scripts/train_p04_feature_probe.py \
    --cache-dir /workspace/p04-cache/convnext-tight224-d4-v1 \
    --feature-name convnext_gap \
    --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
    --output-dir "/workspace/results/P04-TASK-01/p04-r0-l2/fold${FOLD}" \
    --fold "$FOLD" \
    --batch-size 96 \
    --normalization l2 \
    --head-init p04_default \
    --seed 42 \
    2>&1 | tee "/workspace/results/P04-TASK-01/logs/p04-r0-l2-fold${FOLD}.log"
  test "${PIPESTATUS[0]}" -eq 0 || exit 1
done

python scripts/summarize_p04_probes.py \
  --runs-root /workspace/results/P04-TASK-01/p04-r0-l2 \
  --output /workspace/results/P04-TASK-01/p04_r0_l2_summary.json
```

R0 得分可与 P03 不同；这不是失败，因为它是为所有教师统一的 L2 主协议。不在当前 exploratory split 上宣告教师入选。

## 6. 产物、保留与回报

服务器保留整个 `/workspace/p04-cache/convnext-tight224-d4-v1`。回传任务结果目录，不在小型回传包中加入 cache shard；需回传 cache 的 `cache_meta.json/index.json/*-audit.json`。

回报：

1. 状态是 `complete` 还是 `blocked_at_equivalence`；
2. 环境、资产锁、代码 SHA、pytest/ruff；
3. cache 对象数、行数、shard、fingerprint、大小、速度、VRAM；
4. P03 等价三折指标、每折 delta、均值 delta 和门禁结果；
5. P04 R0 L2 三折 mean±sample std；
6. resume 重跑是否全部 skip；
7. 所有日志和任何异常，不自行改参补跑。
