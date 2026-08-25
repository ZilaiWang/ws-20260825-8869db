# P04 服务器任务 02：DINOv2-S/B 全量特征缓存

## 0. 任务边界

前置：P04-TASK-01 的 P03 等价门禁通过。

本任务建立判别式教师阶梯：

- DINOv2 ViT-S/14（无 registers）：`dino_cls=384D`；
- DINOv2 ViT-B/14（无 registers）：`dino_cls=768D`；
- B 的预注册诊断读出：`CLS + mean(patch)=1536D`。

本任务可在当前 exploratory fold 运行三折线性头作为通路与成本诊断，但不得宣告 DINO 正式优于 R0；正式结论等 B 同源隔离划分。不跑 DINO-L/g、DINOv3、register 版或 fine-tune。除原生维度外，按预注册协议运行 ConvNeXt、DINO-B 两种读出的 train-fold-only PCA384 维度控制；DINO-S 原生就是 384D，不重复做等价旋转。

## 1. 路径与冻结条件

```text
asset lock  /workspace/p04-assets/ASSET_LOCK.json
manifest    /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv
DINO-S      /workspace/p04-cache/dinov2-vits14-tight224-d4-v1
DINO-B      /workspace/p04-cache/dinov2-vitb14-tight224-d4-v1
results     /workspace/results/P04-TASK-02
```

两个 cache 均是`tight-224` / D4 / fp16 storage / empty metadata / fold-independent。DINO-B 的 CLS 和 patchmean 必须在同一次前向中缓存，不重跑骨干。

## 2. 预检

```bash
cd /workspace/xh-202625
source /workspace/venvs/p04-cu121/bin/activate
set -o pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 XFORMERS_DISABLED=1
mkdir -p /workspace/results/P04-TASK-02/logs /workspace/p04-cache

sha256sum -c docs/server/P04_CODE_SHA256.txt \
  2>&1 | tee /workspace/results/P04-TASK-02/logs/code-sha256.log

python scripts/check_p04_environment.py \
  --asset-lock /workspace/p04-assets/ASSET_LOCK.json \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --expected-manifest-sha256 f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e \
  --data-root /workspace/data \
  --expected-gpu "NVIDIA GeForce RTX 4080 SUPER" \
  --verify-source-count 32 \
  --output /workspace/results/P04-TASK-02/environment_check.json
```

再运行 P04 专用 pytest/ruff，并确认 TASK-01 `equivalence_gate.status=pass`。如服务器上不存在 TASK-01 结果，不凭口头说明跳过。

## 3. DINOv2-S 全量缓存

```bash
python scripts/extract_p04_features.py \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --data-root /workspace/data \
  --asset-lock /workspace/p04-assets/ASSET_LOCK.json \
  --output-dir /workspace/p04-cache/dinov2-vits14-tight224-d4-v1 \
  --teacher dinov2_vits14 \
  --views d4 \
  --batch-size 64 \
  --shard-size 2048 \
  --compute-dtype float16 \
  --storage-dtype float16 \
  2>&1 | tee /workspace/results/P04-TASK-02/logs/extract-dino-s.log

python scripts/audit_p04_feature_cache.py \
  --cache-dir /workspace/p04-cache/dinov2-vits14-tight224-d4-v1 \
  --expected-objects 20933 \
  --expected-rows 167464 \
  --expected-feature dino_cls=384 \
  --output /workspace/results/P04-TASK-02/dino-s-cache-audit.json
```

必须是 167,464 行、384D、无 NaN/Inf、无重复 key。

## 4. DINOv2-B 全量缓存

```bash
python scripts/extract_p04_features.py \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --data-root /workspace/data \
  --asset-lock /workspace/p04-assets/ASSET_LOCK.json \
  --output-dir /workspace/p04-cache/dinov2-vitb14-tight224-d4-v1 \
  --teacher dinov2_vitb14 \
  --include-patch-mean \
  --views d4 \
  --batch-size 32 \
  --shard-size 2048 \
  --compute-dtype float16 \
  --storage-dtype float16 \
  2>&1 | tee /workspace/results/P04-TASK-02/logs/extract-dino-b.log

python scripts/audit_p04_feature_cache.py \
  --cache-dir /workspace/p04-cache/dinov2-vitb14-tight224-d4-v1 \
  --expected-objects 20933 \
  --expected-rows 167464 \
  --expected-feature dino_cls=768 \
  --expected-feature dino_cls_patchmean=1536 \
  --output /workspace/results/P04-TASK-02/dino-b-cache-audit.json
```

必须同时存在 `dino_cls=768D` 和 `dino_cls_patchmean=1536D`，且行 key 完全相同。

如已通过 TASK-00 的固定 batch 仍 OOM，可将 S/B batch 各自减半一次，必须保留失败日志并重新从未完成 shard 续跑。不得改输入、dtype、权重或 feature location。

## 5. exploratory 三折通路与维度控制诊断

只在两个 cache audit 通过后执行。六组分开目录：

```bash
for FOLD in 0 1 2; do
  python scripts/train_p04_feature_probe.py \
    --cache-dir /workspace/p04-cache/dinov2-vits14-tight224-d4-v1 \
    --feature-name dino_cls \
    --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
    --output-dir "/workspace/results/P04-TASK-02/probe-dino-s-cls/fold${FOLD}" \
    --fold "$FOLD" --batch-size 96 --normalization l2 --head-init p04_default --seed 42

  python scripts/train_p04_feature_probe.py \
    --cache-dir /workspace/p04-cache/dinov2-vitb14-tight224-d4-v1 \
    --feature-name dino_cls \
    --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
    --output-dir "/workspace/results/P04-TASK-02/probe-dino-b-cls/fold${FOLD}" \
    --fold "$FOLD" --batch-size 96 --normalization l2 --head-init p04_default --seed 42

  python scripts/train_p04_feature_probe.py \
    --cache-dir /workspace/p04-cache/dinov2-vitb14-tight224-d4-v1 \
    --feature-name dino_cls_patchmean \
    --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
    --output-dir "/workspace/results/P04-TASK-02/probe-dino-b-cls-patch/fold${FOLD}" \
    --fold "$FOLD" --batch-size 96 --normalization l2 --head-init p04_default --seed 42

  python scripts/train_p04_feature_probe.py \
    --cache-dir /workspace/p04-cache/convnext-tight224-d4-v1 \
    --feature-name convnext_gap \
    --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
    --output-dir "/workspace/results/P04-TASK-02/probe-convnext-pca384/fold${FOLD}" \
    --fold "$FOLD" --batch-size 96 --normalization l2 --pca-dim 384 \
    --head-init p04_default --seed 42

  python scripts/train_p04_feature_probe.py \
    --cache-dir /workspace/p04-cache/dinov2-vitb14-tight224-d4-v1 \
    --feature-name dino_cls \
    --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
    --output-dir "/workspace/results/P04-TASK-02/probe-dino-b-cls-pca384/fold${FOLD}" \
    --fold "$FOLD" --batch-size 96 --normalization l2 --pca-dim 384 \
    --head-init p04_default --seed 42

  python scripts/train_p04_feature_probe.py \
    --cache-dir /workspace/p04-cache/dinov2-vitb14-tight224-d4-v1 \
    --feature-name dino_cls_patchmean \
    --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
    --output-dir "/workspace/results/P04-TASK-02/probe-dino-b-cls-patch-pca384/fold${FOLD}" \
    --fold "$FOLD" --batch-size 96 --normalization l2 --pca-dim 384 \
    --head-init p04_default --seed 42
done

for NAME in \
  dino-s-cls dino-b-cls dino-b-cls-patch \
  convnext-pca384 dino-b-cls-pca384 dino-b-cls-patch-pca384; do
  python scripts/summarize_p04_probes.py \
    --runs-root "/workspace/results/P04-TASK-02/probe-${NAME}" \
    --output "/workspace/results/P04-TASK-02/${NAME}-summary.json"
done
```

这些分数只用于发现读出失效、检查大教师成本、检查“原生维度收益是否在 384D 后消失”，并预测 B split 后实验范围。不根据 exploratory 高低删除 S/B，不搜索中间层或重跑 seed。PCA 必须由每个 fold 的训练对象及其 D4 视图单独拟合，禁止使用验证 fold 或全数据拟合。

## 6. 验收与回报

服务器保留两个全量 cache。小型回传包包含两份 meta/index/audit、probe 产物和日志，不包含 `.npz` shards。

回报：

1. 代码/环境/资产锁及 TASK-01 前置门禁；
2. S/B 权重 SHA、官方 repo commit、offline 运行证明；
3. 每个 cache 行数、维度、shard、fingerprint、磁盘、速度、VRAM、总耗时；
4. 18/18 probe 是否成功，六组 mean±sample std；
5. 任何 OOM 及唯一允许的 batch 减半记录；
6. 明确标注“当前仅 exploratory，未形成正式模型选择”。
