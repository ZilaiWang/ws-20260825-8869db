# P04 服务器任务 03：raw DIFT 与 CleanDIFT 无标签校准

## 0. 任务边界

前置：P04-TASK-00 技术门禁通过，P04-TASK-01 等价门禁通过，P04-TASK-02 两份 DINO cache 审计通过。

本任务不做模型选择，不利用验证标签搜索 timestep/layer，不生成合成数据。它只回答：

1. 项目对 raw DIFT 的审计实现能否稳定输出论文锚点特征；
2. raw DIFT 的 ensemble-4 是否足以逼近 ensemble-8；
3. CleanDIFT 在项目 canonical224→512 合同下是否确定、有限且可断点续跑；
4. CleanDIFT 的三个预注册 map 对 D4 几何变换呈现什么稳定性；
5. 全量 TASK-04 应冻结哪套技术配置。

raw DIFT 的 `ensemble4_gate` 是科学决策信号：失败只表示“不把 raw DIFT 扩展到全量”，不能据此阻止 CleanDIFT 进入 TASK-04。CleanDIFT 重复提取、缓存完整性或资产一致性失败才是技术阻断。

## 1. 冻结路径与配置

```text
repo              /workspace/xh-202625
venv              /workspace/venvs/p04-cu121
manifest          /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv
asset lock         /workspace/p04-assets/ASSET_LOCK.json
calibration list   /workspace/results/P04-TASK-03/calibration_uids_256.txt
raw cache          /workspace/p04-cache/raw-dift-cal256-identity-v1
Clean cache A      /workspace/p04-cache/cleandift-cal256-d4-a-v1
Clean cache B      /workspace/p04-cache/cleandift-cal256-d4-b-v1
results            /workspace/results/P04-TASK-03
```

冻结条件：

- 子集：确定性、类别与风险桶轮询的 256 对象；
- raw DIFT：identity；map #0 / `t=100`，map #6 / `t=261`；嵌套 ensemble `1/4/8`；
- CleanDIFT：D4 八视图；map #0/#6/#9 同一次前向；固定 `t=261`；
- 输入：项目 loader 唯一 canonical 224，再固定 bicubic、`align_corners=False`、`antialias=True` 上采样到 512，并 clamp 到 `[0,1]` 后映射至 VAE 的 `[-1,1]`；
- prompt：空字符串；
- VAE latent：`mode()`，不采样；
- compute/storage：fp16/fp16；
- seed：42，只影响 raw DIFT 的确定性噪声序列；
- 不读取类别 prompt、地理信息或文件名语义。

使用 `latent_dist.mode()` 是项目的确定性表征抽取约定；它不宣称复现 CleanDIFT 训练时的随机 VAE 采样。后续若研究 VAE posterior 采样，应另立敏感性实验，不能混写本 cache。

## 2. 预检

```bash
cd /workspace/xh-202625
source /workspace/venvs/p04-cu121/bin/activate
set -o pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 XFORMERS_DISABLED=1
ROOT=/workspace/results/P04-TASK-03
MANIFEST=/workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv
ASSETS=/workspace/p04-assets/ASSET_LOCK.json
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

另外把 GPU、driver、磁盘、Git commit/dirty、TASK-01 `equivalence_gate.status`、TASK-02 两份 cache audit 状态写入 `system_preflight.txt`。任何技术前置失败立即停止。

## 3. 生成并冻结 calibration 子集

```bash
python scripts/select_p04_calibration_subset.py \
  --manifest "$MANIFEST" \
  --policy tight \
  --count 256 \
  --output "$ROOT/calibration_uids_256.txt" \
  2>&1 | tee "$ROOT/logs/select-calibration.log"
```

必须得到 256 个唯一 UID，且列表 SHA-256 必须为：

```text
d945ba9084b273373a3e9161cdf3500ffc151147ee11675e1832daae3ddf8e20
```

已在本地对冻结 manifest 独立生成并核对：25 类均有覆盖，三折为 84/89/83，三大类为 aircraft 201、ship 49、vehicle 6。该列表的选择只基于 class/risk 元数据，不读取任何模型预测或验证分数。TASK-03 的所有 cache 必须引用同一列表 SHA；不匹配即停止，不在服务器重新定义选样规则。

## 4. raw DIFT identity 校准

```bash
python scripts/extract_p04_features.py \
  --manifest "$MANIFEST" \
  --data-root /workspace/data \
  --asset-lock "$ASSETS" \
  --output-dir /workspace/p04-cache/raw-dift-cal256-identity-v1 \
  --teacher raw_dift_sd15 \
  --annotation-list "$ROOT/calibration_uids_256.txt" \
  --views identity \
  --raw-ensemble-sizes 1,4,8 \
  --global-seed 42 \
  --latent-policy mode \
  --batch-size 1 \
  --shard-size 64 \
  --compute-dtype float16 \
  --storage-dtype float16 \
  2>&1 | tee "$ROOT/logs/extract-raw.log"

python scripts/audit_p04_feature_cache.py \
  --cache-dir /workspace/p04-cache/raw-dift-cal256-identity-v1 \
  --expected-objects 256 \
  --expected-rows 256 \
  --expected-feature raw_map0_t100_e1=1280 \
  --expected-feature raw_map0_t100_e4=1280 \
  --expected-feature raw_map0_t100_e8=1280 \
  --expected-feature raw_map6_t261_e1=1280 \
  --expected-feature raw_map6_t261_e4=1280 \
  --expected-feature raw_map6_t261_e8=1280 \
  --output "$ROOT/raw-cache-audit.json" \
  2>&1 | tee "$ROOT/logs/audit-raw.log"
```

预期 256 行并同时包含：

```text
raw_map0_t100_e1/e4/e8     1280D
raw_map6_t261_e1/e4/e8     1280D
```

原提取命令再执行一次，必须全部 `SKIP shard`。raw 运行只允许 `batch-size=1`，确保每个对象的嵌套噪声序列与 batch 划分无关。

## 5. CleanDIFT D4 双重提取

第一次：

```bash
python scripts/extract_p04_features.py \
  --manifest "$MANIFEST" \
  --data-root /workspace/data \
  --asset-lock "$ASSETS" \
  --output-dir /workspace/p04-cache/cleandift-cal256-d4-a-v1 \
  --teacher cleandift_sd15 \
  --annotation-list "$ROOT/calibration_uids_256.txt" \
  --views d4 \
  --latent-policy mode \
  --batch-size 4 \
  --shard-size 512 \
  --compute-dtype float16 \
  --storage-dtype float16 \
  2>&1 | tee "$ROOT/logs/extract-clean-a.log"
```

第二次写入独立 cache：

```bash
python scripts/extract_p04_features.py \
  --manifest "$MANIFEST" \
  --data-root /workspace/data \
  --asset-lock "$ASSETS" \
  --output-dir /workspace/p04-cache/cleandift-cal256-d4-b-v1 \
  --teacher cleandift_sd15 \
  --annotation-list "$ROOT/calibration_uids_256.txt" \
  --views d4 \
  --latent-policy mode \
  --batch-size 4 \
  --shard-size 512 \
  --compute-dtype float16 \
  --storage-dtype float16 \
  2>&1 | tee "$ROOT/logs/extract-clean-b.log"
```

若 TASK-00 的真实 smoke 已证明 batch=4 OOM，或本节第一次实际 OOM，只允许降为 batch=2，再次 OOM可降为1；必须保留失败日志。A/B 必须使用相同最终 batch，禁止改变其他配置。

分别审计，并执行严格逐 key 重复比较：

```bash
python scripts/audit_p04_feature_cache.py \
  --cache-dir /workspace/p04-cache/cleandift-cal256-d4-a-v1 \
  --expected-objects 256 \
  --expected-rows 2048 \
  --expected-feature clean_map0=1280 \
  --expected-feature clean_map6=1280 \
  --expected-feature clean_map9=640 \
  --output "$ROOT/clean-a-cache-audit.json"

python scripts/audit_p04_feature_cache.py \
  --cache-dir /workspace/p04-cache/cleandift-cal256-d4-b-v1 \
  --expected-objects 256 \
  --expected-rows 2048 \
  --expected-feature clean_map0=1280 \
  --expected-feature clean_map6=1280 \
  --expected-feature clean_map9=640 \
  --output "$ROOT/clean-b-cache-audit.json"

python scripts/audit_p04_feature_cache.py \
  --cache-dir /workspace/p04-cache/cleandift-cal256-d4-a-v1 \
  --compare-cache /workspace/p04-cache/cleandift-cal256-d4-b-v1 \
  --expected-common-rows 2048 \
  --output "$ROOT/clean-repeat-audit.json" \
  2>&1 | tee "$ROOT/logs/audit-clean-repeat.log"
```

预期每份 2,048 行，`clean_map0=1280D`、`clean_map6=1280D`、`clean_map9=640D`。每个特征 repeat cosine p05 必须 `>=0.999`。

## 6. 无标签稳定性报告与决策

```bash
python scripts/analyze_p04_feature_stability.py \
  --raw-cache /workspace/p04-cache/raw-dift-cal256-identity-v1 \
  --clean-cache /workspace/p04-cache/cleandift-cal256-d4-a-v1 \
  --clean-repeat-cache /workspace/p04-cache/cleandift-cal256-d4-b-v1 \
  --output "$ROOT/feature_stability.json" \
  2>&1 | tee "$ROOT/logs/feature-stability.log"
```

解释规则：

- raw `e4 vs e8` 在 map0 与 map6 均满足 median `>=0.99`、p05 `>=0.97`：raw ensemble-4 通过技术成本门槛，但仍不自动进入全量；
- raw gate 失败：冻结为 `raw_not_scaled`，保留全部报告，TASK-04 仍继续 CleanDIFT；
- Clean repeat 失败：状态为 `blocked_at_clean_repeat`，不得进入 TASK-04；
- D4 非 identity 与 r0 的 cosine 仅作层级/模态诊断，不设入选阈值，不把“旋转不变”当作教师优劣；
- 本任务禁止训练 probe 或根据 25 类标签选择 map/timestep。

## 7. 验收与回报

三个 cache 均保留服务器，TASK-04 需要 Clean A 做 overlap 门禁。小型回传包应包含 cache meta/index/sidecar/audit、子集列表、稳定性报告和日志，不含 `.npz` shard；如果总量很小也不要擅自删除服务器 cache。

回报：

1. 技术状态与科学状态分开报告；
2. calibration 列表数量、SHA 和 class/major/risk 覆盖摘要；
3. 三个 cache 行数、维度、fingerprint、大小、速度、VRAM、耗时；
4. raw e1/e4 相对 e8 的 map0/map6 cosine 分布及 gate；
5. Clean A/B repeat 每层 cosine、最大绝对差和 gate；
6. Clean 三层 D4 vs r0 的 overall/by-view 分布；
7. resume 是否全部 skip、任何 OOM 和唯一允许的 batch 变更；
8. 明确说明“未使用验证标签，未形成教师入选结论”。
