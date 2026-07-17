# P03 服务器任务 02：tight-224/336 三折全量微调

## 0. 任务边界

你是服务器执行 AI。P03-TASK-01 已通过完整性和本地配对分析，本任务只执行：

1. 确认原服务器冻结环境和输入未变；
2. 分别对 tight-224 和 tight-336 做 fine-tune smoke；
3. 串行运行 `2 resolutions × 3 folds = 6` 个 natural-sampler 全量微调；
4. 生成三折汇总、完整回传包和 checkpoint 清单。

禁止擅自执行：

- context_1p25 微调；
- `sqrt_inverse` sampler；
- `jitter_light` 训练或评估；
- 更换模型、分辨率、增强、优化器、学习率、epoch、seed 或 batch size；
- 从 P03-1 linear-probe checkpoint 继续训练；
- 并行启动多个 GPU 训练进程。

全量微调必须由同一份官方 ImageNet-1K V1 权重重新初始化 ConvNeXt-Tiny，不使用 P03-1 学到的线性头，以保证两个分辨率的起点一致。

## 1. 固定环境和路径

继续使用 P03-TASK-01 服务器：

```text
GPU: NVIDIA GeForce RTX 4080 SUPER
reported device memory: 33,794,359,296 bytes
Python: 3.10.12
torch: 2.5.1+cu121
torchvision: 0.20.1+cu121
CUDA runtime: 12.1
cuDNN: 9.1.0
```

固定路径：

```text
/workspace/xh-202625/
/workspace/venvs/p03-cu121/
/workspace/data/
/workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv
/workspace/pretrained/convnext_tiny-983f1562.pth
/workspace/results/P03-TASK-02/
```

期望 SHA-256：

```text
crop_manifest.csv
f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e

convnext_tiny-983f1562.pth
983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d
```

P03-1 的 12 个 checkpoint 不是本任务输入。在委托方明确同意删除前，仍保留在原路径。

## 2. 任务预检

执行：

```bash
cd /workspace/xh-202625
source /workspace/venvs/p03-cu121/bin/activate
set -o pipefail
mkdir -p /workspace/results/P03-TASK-02/logs
```

保留环境记录：

```bash
{
  date -Is
  git rev-parse HEAD
  git status --short
  nvidia-smi
  python --version
  python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); p=torch.cuda.get_device_properties(0); print(p.name, p.total_memory)"
  df -h /workspace
} 2>&1 | tee /workspace/results/P03-TASK-02/system_preflight.txt
```

停止条件：

- GPU 不是上述同一设备，或有不明 GPU 进程占用；
- PyTorch/torchvision/CUDA runtime 版本变化；
- 可用磁盘小于 20 GB；
- Git 工作树中的 P0-3 训练代码与 TASK-01 不一致；
- P03-TASK-01 的 12 个 `run_summary.json` 不完整。

重跑环境/输入门禁；本次随机复核 32 张源图，因为 TASK-01 已全量核对 4,481 张：

```bash
python scripts/check_p03_environment.py \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --data-root /workspace/data \
  --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
  --output /workspace/results/P03-TASK-02/environment_check.json \
  --verify-source-count 32 \
  2>&1 | tee /workspace/results/P03-TASK-02/logs/environment-check.log
```

必须确认 `status: pass` 后继续。

## 3. Fine-tune smoke 门禁

分别用 224 和 336 运行一个 1 epoch 小样本 fine-tune smoke。两者都要完成 forward、全模型 backward、AMP、checkpoint 和评估：

```bash
RES=224
python scripts/train_crop_classifier.py \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --data-root /workspace/data \
  --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
  --output-dir "/workspace/results/P03-TASK-02/smoke-tight-${RES}" \
  --fold 0 \
  --policy tight \
  --resolution "${RES}" \
  --regime fine_tune \
  --sampler natural \
  --seed 42 \
  --smoke \
  --overwrite \
  2>&1 | tee "/workspace/results/P03-TASK-02/logs/smoke-tight-${RES}.log"
```

将 `RES=336` 再执行一次。

对两个 smoke 检查：

- `run_summary.json`、`meta.json`、`best_checkpoint.pt`、`validation_logits.npz` 等 10 项产物齐全；
- `n_train <= 256`、`n_val <= 128`、`smoke=true`；
- `model_parameters.trainable_parameters == model_parameters.total_parameters`；
- 无 NaN/Inf；
- 记录 224/336 的峰值显存；
- 336 无 OOM。

两个 smoke 任一失败就停止。smoke 分数不进入实验比较。

## 4. 全量微调配置

使用仓库已冻结配置 `configs/experiments/p03_convnext_tiny.yaml`：

```text
architecture: ConvNeXt-Tiny
initialization: official ImageNet-1K V1
regime: fine_tune
sampler: natural
seed: 42
epochs upper bound: 30
minimum epochs before early stop: 12
early-stopping patience: 8
backbone LR: 1e-4
head LR: 5e-4
AdamW weight decay: 0.05
warmup: 2 epochs
label smoothing: 0.1
batch size 224: 96
batch size 336: 48
```

不要传 `--epochs`、`--batch-size`、`--max-*-samples` 或 `--checkpoint`，避免覆盖冻结配置。

## 5. 串行运行 6 个 run

每个组合使用：

```bash
RES=224
FOLD=0
RUN="/workspace/results/P03-TASK-02/ft-tight-${RES}-fold${FOLD}"

python scripts/train_crop_classifier.py \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --data-root /workspace/data \
  --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
  --output-dir "${RUN}" \
  --fold "${FOLD}" \
  --policy tight \
  --resolution "${RES}" \
  --regime fine_tune \
  --sampler natural \
  --seed 42 \
  2>&1 | tee "/workspace/results/P03-TASK-02/logs/ft-tight-${RES}-fold${FOLD}.log"
```

必须串行完成：

```text
tight 224 fold0
tight 224 fold1
tight 224 fold2
tight 336 fold0
tight 336 fold1
tight 336 fold2
```

所谓“断点续跑”只指 run 级：已有完整且条件匹配的 `run_summary.json` 可跳过。训练脚本不支持单个 run 在某 epoch 中断后继续。如某 run 失败：

1. 立即停止后续 run；
2. 保留完整日志和不完整输出；
3. 不擅自降 batch、改分辨率或改学习率；
4. 报告失败命令、traceback、当时 GPU 显存和磁盘；
5. 得到指示后如需重跑，先将失败目录改名归档，再从头运行该 run。

## 6. 三折汇总门禁

6/6 run 全部成功后执行：

```bash
python scripts/summarize_p03_runs.py \
  --runs-root /workspace/results/P03-TASK-02 \
  --output-dir /workspace/results/P03-TASK-02/aggregate \
  --regime fine_tune \
  --sampler natural \
  --seed 42 \
  2>&1 | tee /workspace/results/P03-TASK-02/logs/aggregate.log
```

`aggregate.csv` 必须恰有 2 行，两行均 `n_folds=3`。必须保留完整数值，不只回报四舍五入后表格。

本任务不根据服务器汇总自行选最终分辨率。本地将按以下预注册规则决定：

1. 三折 mean macro recall 为主指标；
2. 差值 `<=0.005` 视为工程并列；
3. 并列时结合 macro F1、aircraft20、头中尾、折间方向和同对象配对；
4. 仍不能分出稳定收益时选 224。

## 7. 返回产物和 checkpoint

必须返回：

- `system_preflight.txt`、`environment_check.json` 和全部 logs；
- 两个 fine-tune smoke 的全部小型产物；
- 6 个正式 run 中除 `best_checkpoint.pt` 外的全部文件；
- `aggregate.csv`、`selection.json`；
- 6 个 checkpoint 的路径、大小和 SHA-256；
- Git commit/status；
- 总耗时、每个 run 耗时、峰值 VRAM 和是否触发早停。

本次的 fine-tune checkpoint 是下一阶段 `jitter_light` 配对评估的输入，因此服务器必须保留 6 个 checkpoint，直到本地分析选出唯一工作点并确认需要下载哪 3 个。

生成 checkpoint 清单：

```bash
find /workspace/results/P03-TASK-02/ft-*/best_checkpoint.pt -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > /workspace/results/P03-TASK-02/CHECKPOINTS_SHA256.txt

find /workspace/results/P03-TASK-02/ft-*/best_checkpoint.pt -type f -printf '%s %p\n' \
  | sort \
  > /workspace/results/P03-TASK-02/CHECKPOINTS_SIZES.txt
```

修正 TASK-01 的清单自包含问题：不将 `RETURN_FILES_SHA256.txt` 自身写入其内。

```bash
cd /workspace/results
find P03-TASK-02 -type f \
  -not -name 'best_checkpoint.pt' \
  -not -name 'RETURN_FILES_SHA256.txt' \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > /tmp/P03-TASK-02_RETURN_FILES_SHA256.txt
mv /tmp/P03-TASK-02_RETURN_FILES_SHA256.txt P03-TASK-02/RETURN_FILES_SHA256.txt

tar --exclude='best_checkpoint.pt' \
  -czf P03-TASK-02-results-no-checkpoints.tar.gz \
  P03-TASK-02
sha256sum P03-TASK-02-results-no-checkpoints.tar.gz
```

## 8. 最终回报格式

1. 状态：`complete` 或 `blocked_at_<gate>`；
2. Git commit、dirty 状态及与 TASK-01 代码是否一致；
3. GPU/driver/PyTorch/torchvision/CUDA runtime；
4. 环境复检与 224/336 smoke 结果；
5. 6 个正式 run 成功数；
6. tight-224 与 tight-336 的三折 macro recall/F1/accuracy/aircraft20 mean±std；
7. 每折最佳 epoch，是否早停；
8. 224/336 峰值 VRAM、samples/s、单 run 和总耗时；
9. checkpoint 清单路径；
10. 回传包路径、大小和 SHA-256；
11. 失败时给出完整失败命令、traceback、日志和当时资源状态。
