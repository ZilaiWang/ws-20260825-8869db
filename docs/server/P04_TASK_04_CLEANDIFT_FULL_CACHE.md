# P04 服务器任务 04：CleanDIFT 全量 D4 缓存与探索性 probe

## 0. 任务边界

前置：P04-TASK-03 的 CleanDIFT 双重提取、cache audit 和 repeat cosine 门禁通过。raw DIFT 的 `ensemble4_gate` 不影响本任务。

本任务：

1. 用 TASK-03 冻结配置提取 20,933 对象的 CleanDIFT D4 特征；
2. 用 calibration overlap 证明全量 cache 与 TASK-03 同对象特征一致；
3. 在当前 exploratory fold 上运行预注册的 native 与 PCA384 probe，验证完整训练通路和估算教师价值；
4. 为 B 的正式同源隔离 split 准备可复用、fold-independent 的无标签 cache。

本任务不是 P04 正式教师选择。B 的正式 manifest 到达并验收前，任何分数只能标为 `exploratory`。不得根据这些分数新增 layer、搜索 timestep 或把 map #9 事后提升为主行。

## 1. 路径与冻结配置

```text
manifest       /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv
asset lock      /workspace/p04-assets/ASSET_LOCK.json
calibration A   /workspace/p04-cache/cleandift-cal256-d4-a-v1
full cache      /workspace/p04-cache/cleandift-tight224-d4-v1
results         /workspace/results/P04-TASK-04
```

冻结：canonical tight-224、canonical224→512（bicubic、`align_corners=False`、`antialias=True`、clamp `[0,1]`）、D4、empty prompt、VAE `mode()`、CleanDIFT SD1.5、`t=261`、map0/map6/map9 同一次前向、fp16 compute/storage、shard 2048、fold-independent。

## 2. 预检

```bash
cd /workspace/xh-202625
source /workspace/venvs/p04-cu121/bin/activate
set -o pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 XFORMERS_DISABLED=1
ROOT=/workspace/results/P04-TASK-04
MANIFEST=/workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv
ASSETS=/workspace/p04-assets/ASSET_LOCK.json
CACHE=/workspace/p04-cache/cleandift-tight224-d4-v1
CAL=/workspace/p04-cache/cleandift-cal256-d4-a-v1
mkdir -p "$ROOT/logs" /workspace/p04-cache

sha256sum -c docs/server/P04_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/code-sha256.log"

PYTHONPATH=src pytest -q \
  tests/test_p04_feature_pipeline.py tests/test_p04_feature_cli.py \
  2>&1 | tee "$ROOT/logs/pytest.log"

ruff check src/rsdet/features src/rsdet/analysis/p04_features.py \
  scripts/*p04*.py tests/test_p04_feature_*.py \
  2>&1 | tee "$ROOT/logs/ruff.log"

python scripts/check_p04_environment.py \
  --asset-lock "$ASSETS" \
  --manifest "$MANIFEST" \
  --expected-manifest-sha256 f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e \
  --data-root /workspace/data \
  --expected-gpu "NVIDIA GeForce RTX 4080 SUPER" \
  --verify-source-count 32 \
  --verify-sd-inventory \
  --output "$ROOT/environment_check.json" \
  2>&1 | tee "$ROOT/logs/environment-check.log"
```

还必须机器读取并确认：

- TASK-01 `equivalence_gate.status=pass`；
- TASK-03 `clean_repeat.status=pass`；
- calibration A audit 为 pass；
- calibration A 的 teacher/preprocessing/asset lock SHA 与本任务完全一致；
- 至少 12 GB 可用磁盘。

任一不满足立即停止，不通过口头回报跳过。

## 3. 全量 CleanDIFT D4 缓存

```bash
python scripts/extract_p04_features.py \
  --manifest "$MANIFEST" \
  --data-root /workspace/data \
  --asset-lock "$ASSETS" \
  --output-dir "$CACHE" \
  --teacher cleandift_sd15 \
  --views d4 \
  --latent-policy mode \
  --batch-size 4 \
  --shard-size 2048 \
  --compute-dtype float16 \
  --storage-dtype float16 \
  2>&1 | tee "$ROOT/logs/extract-clean-full.log"
```

优先使用 TASK-03 已验证的最终 batch；若 TASK-03 被批准降到2或1，本命令同步采用该值并在回报中说明。运行中只允许在真实 OOM 后将 batch 减半一次，保留失败日志并依靠 shard resume；不得改变 dtype、视图、latent、层、输入或权重。

预期：

```text
object_count             20,933
row_count               167,464
clean_map0 dimension       1,280
clean_map6 dimension       1,280
clean_map9 dimension         640
storage dtype             float16
```

## 4. 完整性、resume 与 calibration overlap 门禁

```bash
python scripts/audit_p04_feature_cache.py \
  --cache-dir "$CACHE" \
  --expected-objects 20933 \
  --expected-rows 167464 \
  --expected-feature clean_map0=1280 \
  --expected-feature clean_map6=1280 \
  --expected-feature clean_map9=640 \
  --output "$ROOT/clean-full-cache-audit.json" \
  2>&1 | tee "$ROOT/logs/audit-clean-full.log"

python scripts/audit_p04_feature_cache.py \
  --cache-dir "$CACHE" \
  --compare-overlap-cache "$CAL" \
  --expected-common-rows 2048 \
  --output "$ROOT/clean-calibration-overlap-audit.json" \
  2>&1 | tee "$ROOT/logs/audit-clean-overlap.log"
```

overlap 必须恰好覆盖 calibration 的 2,048 行；三个特征 cosine p05 均须 `>=0.999`。这同时检查 canonical crop、D4 view、权重、latent、batch 和存储协议没有在 TASK-03→04 间漂移。

将第 3 节原命令重跑，必须全部 `SKIP shard`。如 overlap 或 resume 失败，状态为 blocked，不运行 probe。

## 5. exploratory native probe

三层均用 L2、同一个 25 类线性头、seed=42、natural、identity validation；map0 是预注册分类主行，map6 是结构主行，map9 只是局部几何探索行。

```bash
for FEATURE in clean_map0 clean_map6 clean_map9; do
  for FOLD in 0 1 2; do
    python scripts/train_p04_feature_probe.py \
      --cache-dir "$CACHE" \
      --feature-name "$FEATURE" \
      --manifest "$MANIFEST" \
      --output-dir "$ROOT/probe-${FEATURE}-native/fold${FOLD}" \
      --fold "$FOLD" --batch-size 96 \
      --normalization l2 --head-init p04_default --seed 42 \
      2>&1 | tee "$ROOT/logs/probe-${FEATURE}-native-fold${FOLD}.log"
    test "${PIPESTATUS[0]}" -eq 0 || exit 1
  done
  python scripts/summarize_p04_probes.py \
    --runs-root "$ROOT/probe-${FEATURE}-native" \
    --output "$ROOT/${FEATURE}-native-summary.json"
done
```

## 6. exploratory PCA384 维度控制

只对预注册主行 map0/map6 执行。PCA 必须每 fold 只在训练对象的 D4 特征上拟合，`whiten=False`，transform 后再次 L2；不得使用验证 fold。

```bash
for FEATURE in clean_map0 clean_map6; do
  for FOLD in 0 1 2; do
    python scripts/train_p04_feature_probe.py \
      --cache-dir "$CACHE" \
      --feature-name "$FEATURE" \
      --manifest "$MANIFEST" \
      --output-dir "$ROOT/probe-${FEATURE}-pca384/fold${FOLD}" \
      --fold "$FOLD" --batch-size 96 --normalization l2 --pca-dim 384 \
      --head-init p04_default --seed 42 \
      2>&1 | tee "$ROOT/logs/probe-${FEATURE}-pca384-fold${FOLD}.log"
    test "${PIPESTATUS[0]}" -eq 0 || exit 1
  done
  python scripts/summarize_p04_probes.py \
    --runs-root "$ROOT/probe-${FEATURE}-pca384" \
    --output "$ROOT/${FEATURE}-pca384-summary.json"
done
```

共 15/15 probe。不得补 seed、改 lr 或选择最高 epoch 之外的 checkpoint。与 TASK-02 的 ConvNeXt/DINO exploratory 指标只能做同 split 配对诊断，不能写成最终模型结论。

## 7. B 正式划分到达后的 cache 复用合同

本任务完成后不要重命名、合并或删除 cache。B manifest 到达时另立 P04-TASK-05：

1. 以 B manifest 为唯一 label/fold 真值；
2. 校验 `annotation_uid + crop_id + canonical_input_sha256`；
3. 若只改 fold/group，100% 复用 cache；
4. 新增/修正 crop 只做 delta cache；
5. 正式 native/PCA384 三折重新训练线性头；
6. 当前 exploratory score 不参与入选阈值或 seed 选择。

## 8. 验收、保留与回报

服务器保留全量 cache 与 15 个 probe checkpoint，至少到本地确认收到索引/报告。小型回传包不含 cache shards 和 checkpoint，只含：

- `cache_meta.json`、`index.json`、全部 sidecar 与 extraction summary；
- full audit、calibration overlap audit；
- 15 个 run 的 metrics/predictions/config/history/summary；
- 5 份三折 summary；
- 代码、环境、资产、preflight 和日志。

回报：

1. 状态及任何停止门禁；
2. 代码/环境/资产/TASK-01/TASK-03 前置状态；
3. cache 对象、行、特征维度、shard、fingerprint、磁盘、吞吐、峰值显存、总耗时；
4. full audit、2,048 行 overlap 与 repeat/resume 结果；
5. 15/15 probe 完成情况及五组 mean±sample std；
6. native→PCA384 差异，但不据此正式选择教师；
7. OOM、batch 变更、失败和重试的完整记录；
8. 明确标记 `exploratory_until_B_formal_split`。
