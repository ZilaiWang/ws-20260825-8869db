# P04 服务器任务 00：资产锁、独立环境与真实模型 smoke

## 0. 任务边界

你是服务器执行 AI。本任务不训练、不比较模型分数，只负责：

1. 建立 P04 独立 Python/CUDA 环境；
2. 从冻结官方地址下载 ConvNeXt、DINOv2-S/B、CleanDIFT-SD1.5 和 SD1.5 必需组件；
3. checkout DINOv2、CleanDIFT 与 raw DIFT 参考实现的冻结 commit；
4. 生成唯一 `ASSET_LOCK.json`，后续 01—04 只读该锁；
5. 用真实权重对五种 adapter 做 smoke（Conv/DINO/Clean 各 8 对象，raw DIFT 2 对象）；
6. 验证输出维度、NaN/Inf、确定性、显存和断点续跑。

禁止：使用 `latest/main`、在特征提取时联网、使用类别 prompt、改为 DINOv3/registers、直接读原图 512 crop、因 smoke 分数调参。

## 1. 冻结路径

```text
/workspace/xh-202625/
/workspace/venvs/p04-cu121/
/workspace/data/
/workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv
/workspace/p04-assets/weights/
/workspace/p04-assets/repos/
/workspace/p04-assets/models/stable-diffusion-v1-5/
/workspace/p04-assets/ASSET_LOCK.json
/workspace/results/P04-TASK-00/
```

manifest SHA-256：

```text
f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e
```

需要至少 25 GB 可用磁盘。GPU 为已批准的 RTX 4080 SUPER 32GB；如更换 GPU，停止并报告。

## 2. 代码门禁

同步委托方最新工作树，然后：

```bash
cd /workspace/xh-202625
set -o pipefail
mkdir -p /workspace/results/P04-TASK-00/logs
sha256sum -c docs/server/P04_CODE_SHA256.txt \
  2>&1 | tee /workspace/results/P04-TASK-00/logs/code-sha256.log
```

必须全部 `OK`。不得在服务器临时修代码绕过哈希。

## 3. 建立独立环境

优先复制已验证 P03 CUDA 12.1 环境；如重建：

```bash
python3.10 -m venv /workspace/venvs/p04-cu121
source /workspace/venvs/p04-cu121/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-p04.txt
python -m pip install -e .
python -m pip freeze > /workspace/results/P04-TASK-00/pip-freeze.txt
```

如官方 PyTorch 源中断，可使用之前已验证镜像或复制环境，但版本必须一致。不允许通过降级 diffusers/transformers 临时修复。

执行：

```bash
PYTHONPATH=src pytest -q \
  tests/test_p04_feature_pipeline.py \
  tests/test_p04_feature_cli.py \
  tests/test_imports.py \
  2>&1 | tee /workspace/results/P04-TASK-00/logs/pytest.log

ruff check \
  src/rsdet/features \
  src/rsdet/analysis/p04_features.py \
  scripts/*p04*.py \
  tests/test_p04_feature_pipeline.py \
  tests/test_p04_feature_cli.py \
  2>&1 | tee /workspace/results/P04-TASK-00/logs/ruff.log
```

## 4. 下载和冻结资产

```bash
mkdir -p \
  /workspace/p04-assets/weights \
  /workspace/p04-assets/repos \
  /workspace/p04-assets/models/stable-diffusion-v1-5
```

### 4.1 官方 Git 仓库

```bash
git clone https://github.com/facebookresearch/dinov2.git \
  /workspace/p04-assets/repos/dinov2
git -C /workspace/p04-assets/repos/dinov2 \
  checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8

git clone https://github.com/CompVis/cleandift.git \
  /workspace/p04-assets/repos/cleandift
git -C /workspace/p04-assets/repos/cleandift \
  checkout b070976b22b125167384eed5c96be3a694468763

git clone https://github.com/Tsingularity/dift.git \
  /workspace/p04-assets/repos/dift_reference
git -C /workspace/p04-assets/repos/dift_reference \
  checkout 9421eb2034396c5b66f1aff37f03e540c264e52f
```

若目录已存在，不要重复 clone，只核对 `git rev-parse HEAD`。将三份源码仓库的 `LICENSE` 复制到任务结果目录。DIFT 仓库只作为项目 raw DIFT 审计实现的来源快照，正式运行不导入其旧依赖环境。

### 4.2 ConvNeXt 和 DINOv2

```bash
curl -L --fail --retry 5 \
  -o /workspace/p04-assets/weights/convnext_tiny-983f1562.pth \
  https://download.pytorch.org/models/convnext_tiny-983f1562.pth

curl -L --fail --retry 5 \
  -o /workspace/p04-assets/weights/dinov2_vits14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth

curl -L --fail --retry 5 \
  -o /workspace/p04-assets/weights/dinov2_vitb14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
```

DINO 权重必须分别为 `88,283,115` 和 `346,378,731` bytes。不能使用文件名相同的镜像代替；下载后由资产锁记录 SHA-256。

### 4.3 CleanDIFT 和 SD1.5 snapshot

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

weights = Path('/workspace/p04-assets/weights')
model = Path('/workspace/p04-assets/models/stable-diffusion-v1-5')
hf_hub_download(
    repo_id='CompVis/cleandift',
    filename='cleandift_sd15_unet.safetensors',
    revision='bf3a8d841ebdce7e212b61e42877f8fdaed81d58',
    local_dir=weights,
)
snapshot_download(
    repo_id='stable-diffusion-v1-5/stable-diffusion-v1-5',
    revision='451f4fe16113bff5a5d2269ed5ad43b0592e9a14',
    local_dir=model,
    allow_patterns=[
        'model_index.json', 'README.md', 'LICENSE*',
        'scheduler/*', 'tokenizer/*',
        'text_encoder/*', 'unet/*', 'vae/*',
    ],
)
(model / '.p04_hf_revision').write_text(
    '451f4fe16113bff5a5d2269ed5ad43b0592e9a14\n', encoding='utf-8'
)
PY
```

CleanDIFT U-Net SHA-256 必须为：

```text
56697cc83cef762ac7ca0c8b9e749ee0abacfb426da92dc7fd5d7025ec727516
```

SD1.5 snapshot 必须包含其 `README.md` 与至少一个 `LICENSE*` 文件；把许可证复制到 TASK-00 结果的 `licenses/stable-diffusion-v1-5/`。不得为了减少下载体积删除许可证或模型卡。

## 5. 生成资产锁与环境门禁

```bash
python scripts/lock_p04_assets.py \
  --asset-root /workspace/p04-assets \
  --output /workspace/p04-assets/ASSET_LOCK.json \
  2>&1 | tee /workspace/results/P04-TASK-00/logs/asset-lock.log

python scripts/check_p04_environment.py \
  --asset-lock /workspace/p04-assets/ASSET_LOCK.json \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --expected-manifest-sha256 f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e \
  --data-root /workspace/data \
  --expected-gpu "NVIDIA GeForce RTX 4080 SUPER" \
  --verify-source-count 0 \
  --verify-sd-inventory \
  --output /workspace/results/P04-TASK-00/environment_check.json \
  2>&1 | tee /workspace/results/P04-TASK-00/logs/environment-check.log
```

`status` 必须为 `pass`。生成后禁止编辑 `ASSET_LOCK.json`；后续任务重新现算文件 SHA。

## 6. 离线真实权重 smoke

下载结束后先开启离线环境：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export XFORMERS_DISABLED=1
MANIFEST=/workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv
ASSETS=/workspace/p04-assets/ASSET_LOCK.json
ROOT=/workspace/results/P04-TASK-00
```

分别执行：

```bash
python scripts/extract_p04_features.py --manifest "$MANIFEST" --data-root /workspace/data \
  --asset-lock "$ASSETS" --output-dir "$ROOT/smoke-convnext" \
  --teacher convnext_tiny --views identity --max-samples 8 --batch-size 8 --shard-size 8

python scripts/extract_p04_features.py --manifest "$MANIFEST" --data-root /workspace/data \
  --asset-lock "$ASSETS" --output-dir "$ROOT/smoke-dino-s" \
  --teacher dinov2_vits14 --views identity --max-samples 8 --batch-size 8 --shard-size 8

python scripts/extract_p04_features.py --manifest "$MANIFEST" --data-root /workspace/data \
  --asset-lock "$ASSETS" --output-dir "$ROOT/smoke-dino-b" \
  --teacher dinov2_vitb14 --include-patch-mean --views identity \
  --max-samples 8 --batch-size 4 --shard-size 8

python scripts/extract_p04_features.py --manifest "$MANIFEST" --data-root /workspace/data \
  --asset-lock "$ASSETS" --output-dir "$ROOT/smoke-clean-a" \
  --teacher cleandift_sd15 --latent-policy mode --views identity \
  --max-samples 8 --batch-size 1 --shard-size 8

python scripts/extract_p04_features.py --manifest "$MANIFEST" --data-root /workspace/data \
  --asset-lock "$ASSETS" --output-dir "$ROOT/smoke-clean-b" \
  --teacher cleandift_sd15 --latent-policy mode --views identity \
  --max-samples 8 --batch-size 1 --shard-size 8

python scripts/extract_p04_features.py --manifest "$MANIFEST" --data-root /workspace/data \
  --asset-lock "$ASSETS" --output-dir "$ROOT/smoke-raw-a" \
  --teacher raw_dift_sd15 --raw-ensemble-sizes 1 --views identity \
  --max-samples 2 --batch-size 1 --shard-size 2

python scripts/extract_p04_features.py --manifest "$MANIFEST" --data-root /workspace/data \
  --asset-lock "$ASSETS" --output-dir "$ROOT/smoke-raw-b" \
  --teacher raw_dift_sd15 --raw-ensemble-sizes 1 --views identity \
  --max-samples 2 --batch-size 1 --shard-size 2
```

每个目录执行带行数/维度预期的 cache audit；Clean 两次提取还必须比较：

```bash
python scripts/audit_p04_feature_cache.py --cache-dir "$ROOT/smoke-convnext" \
  --expected-objects 8 --expected-rows 8 --expected-feature convnext_gap=768 \
  --output "$ROOT/smoke-convnext-audit.json"

python scripts/audit_p04_feature_cache.py --cache-dir "$ROOT/smoke-dino-s" \
  --expected-objects 8 --expected-rows 8 --expected-feature dino_cls=384 \
  --output "$ROOT/smoke-dino-s-audit.json"

python scripts/audit_p04_feature_cache.py --cache-dir "$ROOT/smoke-dino-b" \
  --expected-objects 8 --expected-rows 8 --expected-feature dino_cls=768 \
  --expected-feature dino_cls_patchmean=1536 \
  --output "$ROOT/smoke-dino-b-audit.json"

for CACHE in smoke-clean-a smoke-clean-b; do
  python scripts/audit_p04_feature_cache.py --cache-dir "$ROOT/$CACHE" \
    --expected-objects 8 --expected-rows 8 --expected-feature clean_map0=1280 \
    --expected-feature clean_map6=1280 --expected-feature clean_map9=640 \
    --output "$ROOT/${CACHE}-audit.json"
done

for CACHE in smoke-raw-a smoke-raw-b; do
  python scripts/audit_p04_feature_cache.py --cache-dir "$ROOT/$CACHE" \
    --expected-objects 2 --expected-rows 2 \
    --expected-feature raw_map0_t100_e1=1280 \
    --expected-feature raw_map6_t261_e1=1280 \
    --output "$ROOT/${CACHE}-audit.json"
done

python scripts/audit_p04_feature_cache.py \
  --cache-dir "$ROOT/smoke-clean-a" \
  --compare-cache "$ROOT/smoke-clean-b" \
  --expected-common-rows 8 \
  --output "$ROOT/clean-repeat-audit.json"

python scripts/audit_p04_feature_cache.py \
  --cache-dir "$ROOT/smoke-raw-a" \
  --compare-cache "$ROOT/smoke-raw-b" \
  --expected-common-rows 2 \
  --output "$ROOT/raw-repeat-audit.json"
```

必须得到：

| teacher | feature | 维度 |
| --- | --- | ---: |
| ConvNeXt | `convnext_gap` | 768 |
| DINO-S | `dino_cls` | 384 |
| DINO-B | `dino_cls` | 768 |
| DINO-B | `dino_cls_patchmean` | 1536 |
| CleanDIFT | `clean_map0` | 1280 |
| CleanDIFT | `clean_map6` | 1280 |
| CleanDIFT | `clean_map9` | 640 |
| raw DIFT | `raw_map0_t100_e1` | 1280 |
| raw DIFT | `raw_map6_t261_e1` | 1280 |

Clean 与 raw repeat 每个共同特征的 cosine p05 必须不低于 `0.999`。所有缓存无 NaN/Inf，且重复执行同一 extractor 命令应全部 `SKIP shard`。

## 7. 停止条件

任一发生即停止，不进入 TASK-01：

- 权重大小/SHA、Git commit、SD revision 不匹配；
- 环境版本不符或提取期间尝试联网；
- 输出 shape 错误、NaN/Inf、单图 OOM；
- canonical224→512 之外的图像输入；
- Clean 或 raw repeat 确定性门禁失败；
- cache resume 覆盖或重复 key。

## 8. 回传与最终回报

回传 TASK-00 整个结果目录和 `ASSET_LOCK.json`，不回传 SD1.5/DINO/Clean 大权重。权重和官方 repo 保留在服务器，不得删除。

回报：

1. 状态和停止门禁；
2. GPU/driver/Python/torch/torchvision/CUDA/cuDNN/磁盘；
3. `P04_CODE_SHA256.txt`、pytest、ruff；
4. 每个资产的路径、大小、SHA、revision/commit；
5. `ASSET_LOCK.json` 自身 SHA 和 fingerprint；
6. 七次真实提取（五个 adapter，Clean/raw 各一组 repeat）的 shape、速度、峰值显存；
7. Clean/raw repeat 的 cosine p05/median/max absolute difference；
8. 任何失败的完整命令、traceback 和日志路径。
