# P03 服务器任务 01：环境、smoke 与三折 linear probe

## 0. 任务边界

你是服务器执行 AI。本任务只做以下三件事：

1. 建立并验证冻结的 RTX 3090 / PyTorch 环境；
2. 跑通 P03 smoke test；
3. 在 smoke 通过后运行 `tight/context_1p25 × 224/336 × 3 folds` 的 12 个 natural-sampler linear probe。

不要擅自修改模型、学习率、epoch、增强、sampler、fold、数据或代码。不要运行 fine-tune、jitter 训练、扩散模型或其他骨干。如发现问题，保留完整日志并停止在当前门禁，不自行绕过。

## 1. 期望资源和目录

最低资源：

- Ubuntu 22.04 LTS；
- NVIDIA RTX 3090 24 GB；
- 至少 32 GB RAM、8 vCPU、40 GB 可用磁盘；
- NVIDIA driver 支持 CUDA 12.1；
- Python 3.10。

固定目录：

```text
/workspace/xh-202625/                         # Git 仓库
/workspace/data/                              # 官方数据根目录
/workspace/data/images/train/                 # 原图
/workspace/data/images/val/                   # 可为空，但目录应保留
/workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv
/workspace/pretrained/convnext_tiny-983f1562.pth
/workspace/results/P03-TASK-01/
```

必须由委托方上传的非 Git 资产：

- 完整 `/workspace/data`，约 1.23 GB；
- P0-2 `crop_manifest.csv`，约 49 MB；
- ConvNeXt-Tiny 权重，约 109 MB。

期望 SHA-256：

```text
crop_manifest.csv
f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e

convnext_tiny-983f1562.pth
983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d
```

权重的官方直链：

```text
https://download.pytorch.org/models/convnext_tiny-983f1562.pth
```

若委托方未上传权重且服务器可访问该链接，可下载到指定路径；下载后必须校验完整 SHA-256。不得通过 torchvision 训练时隐式下载。

## 2. GPU 和驱动门禁

先建立任务和日志目录，并要求 pipeline 中任一命令失败都返回非零状态：

```bash
mkdir -p /workspace/results/P03-TASK-01/logs
set -o pipefail
{
  date -Is
  nvidia-smi
  python3.10 --version
  df -h /workspace
  free -h
} 2>&1 | tee /workspace/results/P03-TASK-01/system_preflight.txt
```

停止条件：

- GPU 不是 RTX 3090；
- GPU 可用显存明显小于 22 GB 且有不明进程占用；
- Python 3.10 不可用；
- `/workspace` 剩余磁盘小于 40 GB；
- NVIDIA driver 无法正常识别 GPU。

保存 `nvidia-smi` 和资源输出到：

```text
/workspace/results/P03-TASK-01/system_preflight.txt
```

## 3. 建立冻结环境

进入仓库：

```bash
cd /workspace/xh-202625
```

建立独立环境：

```bash
python3.10 -m venv /workspace/venvs/p03-cu121
source /workspace/venvs/p03-cu121/bin/activate
python -m pip install --upgrade pip==24.3.1 setuptools==75.6.0 wheel==0.45.1
```

先安装官方 cu121 版本：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

再安装项目的固定辅助依赖和本仓库：

```bash
python -m pip install -r requirements-p03.txt
python -m pip install -e . --no-deps
```

不要安装 nightly、cu124/cu126、最新版 torch，不要在运行中自动升级任何包。

## 4. 静态质量检查

在仓库根目录执行并保留日志：

```bash
python -m pytest tests/test_crop_classification_data.py tests/test_classification_metrics.py tests/test_imports.py -q \
  2>&1 | tee /workspace/results/P03-TASK-01/logs/static-pytest.log
python -m ruff check src/rsdet/data/crop_classification.py src/rsdet/evaluation/classification.py src/rsdet/models/crop_classifier.py scripts/train_crop_classifier.py scripts/check_p03_environment.py scripts/summarize_p03_runs.py tests/test_crop_classification_data.py tests/test_classification_metrics.py \
  2>&1 | tee /workspace/results/P03-TASK-01/logs/static-ruff.log
```

任一项失败就停止，保留版本、完整 traceback 和 Git commit。不要在服务器上修代码后继续跑。

## 5. 环境、权重和数据门禁

执行：

```bash
python scripts/check_p03_environment.py \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --data-root /workspace/data \
  --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
  --output /workspace/results/P03-TASK-01/environment_check.json \
  --verify-source-count 0 \
  2>&1 | tee /workspace/results/P03-TASK-01/logs/environment-check.log
```

确认 `environment_check.json` 中：

- `status` 为 `pass`；
- `torch` 为 2.5.1；
- `torchvision` 为 0.20.1；
- `torch_cuda_runtime` 为 12.1；
- `gpu_name` 为 NVIDIA GeForce RTX 3090；
- 权重与 manifest checksum 正确；
- 3 个 fold 都有非空 train/val；
- P0-3 引用的全部唯一源图都可打开，尺寸与 SHA-256 均与 manifest 一致。

任一项不符合就停止。

## 6. Smoke test

执行：

```bash
python scripts/train_crop_classifier.py \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --data-root /workspace/data \
  --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
  --output-dir /workspace/results/P03-TASK-01/smoke-fold0-tight-224 \
  --fold 0 \
  --policy tight \
  --resolution 224 \
  --regime linear_probe \
  --sampler natural \
  --seed 42 \
  --smoke \
  --overwrite \
  2>&1 | tee /workspace/results/P03-TASK-01/logs/smoke-fold0-tight-224.log
```

该命令应只训练 1 epoch，最多使用 256 train / 128 val。它只测通路，不得将分数写入正式比较。

检查：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('/workspace/results/P03-TASK-01/smoke-fold0-tight-224')
required = {
    'resolved_config.yaml', 'meta.json', 'metrics.json', 'per_class_metrics.csv',
    'confusion_matrix.csv', 'predictions.csv', 'validation_logits.npz',
    'history.csv', 'best_checkpoint.pt', 'run_summary.json',
}
missing = sorted(name for name in required if not (root / name).is_file())
if missing:
    raise SystemExit(f'missing smoke artifacts: {missing}')
summary = json.loads((root / 'run_summary.json').read_text())
meta = json.loads((root / 'meta.json').read_text())
assert summary['condition']['smoke'] is True
assert summary['n_train'] <= 256 and summary['n_val'] <= 128
assert meta['gpu']['name'].endswith('RTX 3090')
print('SMOKE_GATE_PASS')
PY
```

只有输出 `SMOKE_GATE_PASS` 后才能继续。

## 7. 运行 12 个 linear probe

对每个条件单独执行下面命令，将 `POLICY`、`RES`、`FOLD` 替换为组合值：

```bash
POLICY=tight
RES=224
FOLD=0
RUN=/workspace/results/P03-TASK-01/lp-${POLICY}-${RES}-fold${FOLD}

python scripts/train_crop_classifier.py \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --data-root /workspace/data \
  --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
  --output-dir "${RUN}" \
  --fold "${FOLD}" \
  --policy "${POLICY}" \
  --resolution "${RES}" \
  --regime linear_probe \
  --sampler natural \
  --seed 42 \
  2>&1 | tee "/workspace/results/P03-TASK-01/logs/lp-${POLICY}-${RES}-fold${FOLD}.log"
```

必须完成的组合：

```text
tight        224  fold0 fold1 fold2
tight        336  fold0 fold1 fold2
context_1p25 224  fold0 fold1 fold2
context_1p25 336  fold0 fold1 fold2
```

可串行运行，不要在单张 3090 上同时启动多个训练进程。一个 run 失败时：

1. 保留原输出目录和完整 stdout/stderr；
2. 记录最后一个成功 run；
3. 停止后续组合；
4. 不加 batch size、不降分辨率、不改 epoch 来绕过失败。

若确认是 CUDA OOM，只报告当时 batch、显存占用、traceback 和 GPU 进程；由本地负责人决定是否下调固定 batch size。

## 8. 三折汇总门禁

12 个 run 全部成功后执行：

```bash
python scripts/summarize_p03_runs.py \
  --runs-root /workspace/results/P03-TASK-01 \
  --output-dir /workspace/results/P03-TASK-01/aggregate \
  --regime linear_probe \
  --sampler natural \
  --seed 42 \
  2>&1 | tee /workspace/results/P03-TASK-01/logs/aggregate.log
```

应生成：

```text
/workspace/results/P03-TASK-01/aggregate/aggregate.csv
/workspace/results/P03-TASK-01/aggregate/selection.json
```

`aggregate.csv` 必须恰有 4 行，每行 `n_folds=3`。不要手工改 `selection.json`；它只是按预注册规则生成的 P03-2 候选，最终是否补第 3 名由本地配对分析决定。

## 9. 返回产物

必须返回：

- `system_preflight.txt`；
- `environment_check.json`；
- smoke 的全部文件；
- 12 个 run 中除 `best_checkpoint.pt` 外的全部文件；
- `aggregate.csv` 和 `selection.json`；
- 所有 stdout/stderr 日志；
- 仓库 Git commit 和 `git status --short`；
- 服务器上 12 个 checkpoint 的路径、大小和 SHA-256 清单。

线性探针 checkpoint 可暂不打包回传，但不得在本地确认收到结果前删除。

打包前生成文件清单：

```bash
cd /workspace/results
find P03-TASK-01 -type f -not -name 'best_checkpoint.pt' -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > P03-TASK-01/RETURN_FILES_SHA256.txt
```

只打包小型结果：

```bash
cd /workspace/results
tar --exclude='best_checkpoint.pt' -czf P03-TASK-01-results-no-checkpoints.tar.gz P03-TASK-01
sha256sum P03-TASK-01-results-no-checkpoints.tar.gz
```

## 10. 最终执行回报格式

只在任务完成或触发停止条件后回报，包含：

1. 状态：`complete` 或 `blocked_at_<gate>`；
2. Git commit 和是否 dirty；
3. GPU/driver/PyTorch/torchvision/CUDA runtime；
4. environment check 结果；
5. smoke 是否通过；
6. 12 个 run 成功数；
7. 4 个条件的三折 macro recall mean±std 和 macro F1 mean±std；
8. `selection.json` 选出的两个条件；
9. 每个分辨率的峰值显存范围和实测总耗时；
10. 返回压缩包路径、大小和 SHA-256；
11. 如失败，给出完整失败命令、traceback 和日志路径。
