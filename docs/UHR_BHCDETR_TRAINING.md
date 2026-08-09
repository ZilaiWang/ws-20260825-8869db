# UHR-BHC-DETR 小目标训练与 AutoDL 运行手册

本文说明如何在现有 BHC-DETR + BHCL 模型上启用论文 [UHR-DETR: Efficient End-to-End Small Object Detection for Ultra-High-Resolution Remote Sensing Imagery](https://arxiv.org/abs/2604.21435) 中可由当前数据可靠监督的 Gain Map、Local Peak Margin Loss、ISSGA 和 Global-Local 解码思想。

本文的所有 Linux 命令均以以下目录为起点：

```bash
cd ~/autodl-tmp/xh-202625-master
```

本文不创建独立 Conda 环境或 Python 虚拟环境。不要执行 `conda create`、`python -m venv` 或 `virtualenv`；直接复用 AutoDL PyTorch 镜像自带的 `base` 环境。

## 1. 先明确实现边界

### 1.1 当前实现是什么

当前实现是面向现有裁剪训练数据的 UHR 论文适配，仍然属于 `bhcdetr` 模型：

- 保留上一阶段实现的分类查询/定位查询解耦、共享 self-attention、独立任务分支和每层 BHCL 原型库；
- 保留 25 个细类的 Hungarian 匹配、focal、L1、水平 GIoU 和 `0.6 × BHCL`；
- ResNet-50 的 C5/stride-32 特征继续充当全局 memory；
- 新增 C4/stride-16 局部特征，为小目标提供比 C5 更密的空间表征；
- 在 C5 投影特征上预测离散 Gain Map；
- 按论文公式使用 IoF-sum Gain Map target、平方支持集上的 DFL 和四邻域 LPM；
- 使用线性 Iterative Soft-Subtraction Greedy Algorithm（ISSGA）选择同一输入图或同一 tile 内的高价值 C4 token 窗口；
- 分类和定位分支先读取 C5 全局 memory，再读取 ISSGA 选出的 C4 局部 memory；
- 局部 token 总数受 `uhr_max_local_tokens` 限制，避免 dense attention 的显存和时延失控；
- 推理、阈值筛选、跨 tile 融合和 COCO JSON 输出继续沿用现有工程链路。

对应代码主要位于：

```text
src/rsdet/models/uhr_small_object.py
src/rsdet/models/bhcdetr.py
src/rsdet/models/detection_loss.py
src/rsdet/engine/trainer.py
```

### 1.2 当前实现不是什么

它不是论文完整的 10K UHR-DETR 复现。当前训练集包含 4,481 张已经裁剪的图像，尺寸约为 423–1,384 像素，中位数约 800；没有带完整标注的 10,000×10,000 连续场景。因而当前代码没有假装实现无法由现有数据监督的部分：

- 没有 ResNet-18 的整幅 10K 全局支路；
- 没有在 10K 原始像素图上直接预测并裁取论文默认的 40 个 512×512 patch；
- 没有把外层 10K 滑窗完全替换为论文的整图稀疏 patch router；
- 没有论文的动态 300–3,000 queries、DAB 全局 anchor 路由和 quality-aware query selection；
- 没有作者 MMDetection 实现中的 CUDA Multi-Scale Deformable Attention；当前使用有上限的 C4 token 和 PyTorch `MultiheadAttention`；
- 没有把 BHC-DETR 的 focal 分类损失替换成 UHR-DETR 的 Varifocal Loss；这是为保留上一阶段分类/BHCL 基线而采用的混合训练目标；
- 没有论文整幅图统一解码后完全取消跨 tile 融合的行为。

因此，10K 推理仍是：

```text
10K 图像
→ 现有外层滑窗
→ 每个 tile 内执行 Gain Map + ISSGA + C5/C4 Global-Local 解码
→ 恢复全图坐标
→ 跨 tile 融合
```

不要在报告中把它写成“完整 UHR-DETR”或引用论文的 10× 加速、0.357 秒和 mAP 数字作为本项目结果。本项目只报告自己在 CV3 OOF 和指定硬件上实测得到的 Recall、FDR、时延和显存。

### 1.3 为什么采用这个边界

直接在约 800 像素的裁剪图上训练一个整图 router，再让它在 10K 图上决定哪些区域永远不进入检测器，会产生严重的训练/测试尺度与背景分布偏移。一旦 router 漏掉目标，后续检测器无法恢复该目标，路由覆盖率就会成为 Recall 的硬上限。

当前适配让训练和推理看到一致的 1024 输入几何：训练图 letterbox 到 1024，10K 推理 tile 也 letterbox 到 1024；`uhr_patch_size=512`、Gain Map target、ISSGA 和局部 token 选择均在同一输入内部执行。它不能制造新的像素细节，但能够把有限的 decoder attention 更集中地分配给 C4 小目标区域，并用 C5 提供较大范围的上下文。

## 2. AutoDL base 环境

### 2.1 每次登录后的初始化

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip
export TORCH_HOME=/root/autodl-tmp/cache/torch
mkdir -p "$PIP_CACHE_DIR" "$TORCH_HOME"

cd ~/autodl-tmp/xh-202625-master
```

这里的 `conda activate base` 是启用镜像已有环境，不是创建独立环境。

首次上传代码后安装项目：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[model,dev]"
```

AutoDL 镜像应预先提供与 CUDA 匹配的 PyTorch 和 torchvision。如果下面检查失败，应更换合适的 AutoDL PyTorch 镜像，不要在容器里创建新环境，也不要盲目重装 NVIDIA 驱动或完整 CUDA Toolkit：

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python - <<'PY'
import scipy
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("scipy:", scipy.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), "当前 base 环境无法使用 CUDA"
PY
```

### 2.2 数据和旧 checkpoint

本文默认目录为：

```text
/root/autodl-tmp/
├── data/
│   ├── images/train/
│   └── labels/train/
└── xh-202625-master/
    └── outputs/BHCDETR-R50-1024-devv2-seed42/best.pt
```

检查数据和旧模型：

```bash
cd ~/autodl-tmp/xh-202625-master
test -d ../data/images/train
test -d ../data/labels/train
test -f outputs/BHCDETR-R50-1024-devv2-seed42/best.pt

python scripts/check_dataset.py \
  --data-root /root/autodl-tmp/data \
  --split train \
  --official-train
```

如果 `best.pt` 位于其他目录，在后续本地 YAML 中修改 `init_checkpoint`，不要复制到模板声明以外的模糊路径。

## 3. 配置说明

正式模板：

```text
configs/models/bhcdetr_uhr_r50_1024.yaml
```

软件冒烟模板：

```text
configs/bhcdetr_uhr.smoke.yaml
```

先复制正式模板，所有服务器路径和实验改动只写入 local 文件：

```bash
cd ~/autodl-tmp/xh-202625-master
cp configs/models/bhcdetr_uhr_r50_1024.yaml configs/bhcdetr_uhr.local.yaml
```

正式配置的冻结实验名和输出目录为：

```yaml
experiment_id: "BHCDETR-UHR-R50-1024-devv2-seed42"
output_dir: "outputs/BHCDETR-UHR-R50-1024-devv2-seed42"
```

### 3.1 UHR 适配参数

本项目的主要新增参数为：

```yaml
architecture:
  image_size: 1024
  uhr_enabled: true
  uhr_gain_bin_limit: 6
  uhr_patch_size: 512
  uhr_patch_budget: 4
  uhr_max_local_tokens: 1024
  uhr_gain_head_groups: 32

loss:
  gain_map_weight: 1.0
  gain_lpm_weight: 1.0
  gain_lpm_margin: 0.05
```

含义如下：

| 参数 | 含义 |
|---|---|
| `uhr_gain_bin_limit` | Gain Map 离散上限 M，输出 M+1 个 bin，期望支持为 `b²` |
| `uhr_patch_size` | 相对于 1024 输入的论文式局部窗口边长；不是从 10K 原图直接裁图 |
| `uhr_patch_budget` | 每个输入/tile 执行 ISSGA 的窗口预算；4 是 1024 tile 的工程值，不是论文全图的 K=40/80 |
| `uhr_max_local_tokens` | 送入每个 decoder 局部 attention 的 C4 token 上限 |
| `uhr_gain_head_groups` | Gain Head 的 GroupNorm 分组；论文未公布该值，32 是工程默认 |
| `gain_map_weight` | Gain Map DFL 权重；论文未公布该权重，当前值是可审计工程默认 |
| `gain_lpm_weight` | LPM 权重；论文未公布该权重，当前值是可审计工程默认 |
| `gain_lpm_margin` | 论文消融采用的局部峰值间隔，默认 0.05 |

`M=6` 和 512 patch 来自论文公开设置，但论文未给出平方 bins 的具体 DFL 插值代码、
相等 Gain plateau 的取峰规则和两个辅助损失权重。当前代码分别采用相邻平方支持点线性
插值、确定性 plateau tie-break、`1.0/1.0` 权重，并给每个 ISSGA 窗口预留局部 token
配额；这些都是可审计的项目选择。在完成稳定基线前不要同时修改 M、patch size、
patch budget 和 loss 权重。

### 3.2 从旧 BHC-DETR warm-start

正式 local 配置应包含：

```yaml
resume: null
init_checkpoint: "outputs/BHCDETR-R50-1024-devv2-seed42/best.pt"
```

`init_checkpoint` 的语义是“兼容初始化”，不是恢复旧训练：

- 按参数名和 tensor shape 加载旧模型中兼容的权重；
- 兼容的 BHCL prototype bank 也会加载；
- 新增的 C4 projection、Gain Map Head 和 local cross-attention 保持新初始化；
- epoch、global step、旧 optimizer、旧 scheduler 和旧 AMP scaler 不恢复；
- 新实验从 epoch 0 开始，并创建自己的 `best.pt` 和 `last.pt`。

训练启动后检查：

```text
outputs/BHCDETR-UHR-R50-1024-devv2-seed42/init_checkpoint_report.json
```

报告中的 `loaded_tensors` 应大于 0；出现新模块相关的 `missing_preview` 是预期行为。若大部分旧 backbone、query、head 都没有加载，应先停止训练并检查 checkpoint 和配置，不要只看程序没有报错。

`resume` 和 `init_checkpoint` 互斥：

- 第一次启动 UHR 实验：使用旧模型 `init_checkpoint`，`resume: null`；
- UHR 训练中断后：把 local YAML 中 `init_checkpoint` 改成 `null`，再用 `--resume` 指向同一 UHR 实验的 `last.pt`。

不要用旧 BHC `best.pt` 作为 `--resume`；旧模型缺少新模块和新 optimizer 状态，只能通过 `init_checkpoint` warm-start。

### 3.3 优化器和调度

正式模板使用论文给出的优化设置作为新阶段初值：

```yaml
train:
  learning_rate: 0.0001
  backbone_learning_rate: 0.00001
  weight_decay: 0.0001
  scheduler: "warmup_multistep"
  warmup_steps: 500
  lr_milestones: [8, 11]
  lr_gamma: 0.1
  clip_max_norm: 0.1
```

backbone 使用主学习率的 0.1 倍；warmup 按 optimizer update 计数，里程碑按 epoch 转换为 update。`metrics.jsonl` 同时记录 `main` 和 `backbone` 学习率。

## 4. 代码与数据冒烟检查

### 4.1 单元测试

先运行 UHR 专项测试：

```bash
cd ~/autodl-tmp/xh-202625-master
python -m pytest tests/test_uhr_small_object.py -q
```

再运行与 BHC-DETR 直接相关的回归测试：

```bash
python -m pytest \
  tests/test_bhcdetr_architecture.py \
  tests/test_bhcdetr_detection_loss.py \
  tests/test_bhcl.py \
  tests/test_bhcdetr_dataset.py \
  -q
```

专项测试覆盖 Gain Map head/expectation、IoF target、平方支持 DFL、四邻域 LPM、线性 ISSGA，以及稀疏局部 token 的形状、掩码和确定性。通过单元测试只说明公式与软件路径成立，不代表精度或 10K 时延通过。

### 4.2 正式配置 dry-run

```bash
python scripts/train.py \
  --config configs/bhcdetr_uhr.local.yaml \
  --dry-run
```

应检查日志或 `data_audit.json` 中至少包含：

```text
uhr_small_object_enabled = true
train_source_images      = 3548
val_source_images        = 933
fine_classes             = 25
init_checkpoint          = outputs/BHCDETR-R50-1024-devv2-seed42/best.pt
```

dry-run 不构建模型，因此不会生成 `init_checkpoint_report.json`。

### 4.3 CUDA 一步冒烟

```bash
python scripts/train.py \
  --config configs/bhcdetr_uhr.smoke.yaml \
  --device cuda:0 \
  --max-steps 1
```

冒烟配置是缩小图像、查询数和层数的软件检查，不用于指标评估，也不替代旧正式 checkpoint warm-start 审计。成功时应完成一次 forward、Gain Map/LPM/BHC/BHCL loss、backward、验证以及 `best.pt`/`last.pt` 写入。

## 5. 正式训练

建议使用 tmux，防止 SSH 断开终止训练：

```bash
cd ~/autodl-tmp/xh-202625-master
tmux new -s bhcdetr_uhr
```

进入 tmux 后重新激活已有 base 环境并启动训练：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip
export TORCH_HOME=/root/autodl-tmp/cache/torch
cd ~/autodl-tmp/xh-202625-master

mkdir -p outputs/BHCDETR-UHR-R50-1024-devv2-seed42
set -o pipefail
python -u scripts/train.py \
  --config configs/bhcdetr_uhr.local.yaml \
  --device cuda:0 2>&1 | \
  tee outputs/BHCDETR-UHR-R50-1024-devv2-seed42/train_console.log
```

按 `Ctrl+B`，再按 `D` 可退出 tmux 而不停止任务。重新连接后：

```bash
cd ~/autodl-tmp/xh-202625-master
tmux attach -t bhcdetr_uhr
```

另开终端监控：

```bash
cd ~/autodl-tmp/xh-202625-master
nvidia-smi -l 2
```

```bash
cd ~/autodl-tmp/xh-202625-master
tail -f outputs/BHCDETR-UHR-R50-1024-devv2-seed42/train_console.log
```

warm-start 完成后立即检查加载报告：

```bash
python -m json.tool \
  outputs/BHCDETR-UHR-R50-1024-devv2-seed42/init_checkpoint_report.json
```

每轮指标写入：

```text
outputs/BHCDETR-UHR-R50-1024-devv2-seed42/metrics.jsonl
```

重点检查这些 loss 是否有限：

```text
loss_total
loss_class
loss_bbox
loss_giou
loss_bhcl
loss_gain_map
loss_gain_lpm
```

正式模板已经使用 `train.batch_size: 1`、每个 source image 的 `views_per_image: 2` 和 `gradient_accumulation_steps: 4`。如果仍出现 CUDA OOM，`batch_size` 已无法继续降低；应先把 `architecture.uhr_max_local_tokens` 从 1024 降到 512，并为这个改动使用新的实验名和输出目录。需要明确：梯度累积能近似保持 optimizer 的有效 batch，但不能降低单次 forward 的峰值显存，也不能扩大 BHCL 单次 forward 能看到的对比样本集合。

### 5.1 RTX 5090 高速配置（AutoDL）

如果 RTX 5090 在标准 `batch_size: 1` 配置下只占用约 5 GB 显存、每轮耗时很长，优先使用专门的吞吐配置：

```text
configs/models/bhcdetr_uhr_r50_1024_5090.yaml
```

该配置保持 UHR 网络结构、损失权重、学习率和训练轮数不变，主要把训练改为 `batch_size: 4`、`gradient_accumulation_steps: 1`，并启用 8 workers、预取、channels-last、TF32/high matmul precision、fused AdamW（运行时不支持会自动回退）以及每 2 轮验证一次。它使用独立实验名和输出目录：

```yaml
experiment_id: "BHCDETR-UHR-R50-1024-devv2-seed42-5090"
output_dir: "outputs/BHCDETR-UHR-R50-1024-devv2-seed42-5090"
```

代码级优化没有用近似算法换取速度。trainer 在 GPU 上累计 loss 指标，每个训练/验证阶段只在 epoch 末用一次批量 D2H 完成汇总，不再对每个 batch 的每个 loss 调用 `.item()`。6 层 decoder × batch 的 Hungarian cost 先合并为一次 D2H，求解后的索引也打包为一次 H2D；但每张图、每层的矩阵仍分别调用 SciPy `linear_sum_assignment` 做精确匹配，不会混合不同图或 decoder 层。BHCL 类别项和 prototype EMA、UHR token selector 改为固定形状的批量张量运算，仍保留 BHCL 公式、ISSGA route quota、有效 token 与 padding mask 语义。

以下命令都直接复用第 2.1 节已经启用的 AutoDL `base` 环境，不创建任何新环境。

#### 5.1.1 复制配置并执行 dry-run

只在 local 副本中做服务器侧修改：

```bash
cd ~/autodl-tmp/xh-202625-master
test -f configs/models/bhcdetr_uhr_r50_1024_5090.yaml
test -f outputs/BHCDETR-R50-1024-devv2-seed42/best.pt

cp configs/models/bhcdetr_uhr_r50_1024_5090.yaml \
  configs/bhcdetr_uhr_5090.local.yaml

python scripts/train.py \
  --config configs/bhcdetr_uhr_5090.local.yaml \
  --dry-run
```

dry-run 应显示 5090 独立输出目录、`uhr_small_object_enabled = true`、训练/验证样本数和旧 BHC `best.pt` 的 warm-start 路径。也可以直接查看审计文件：

```bash
python -m json.tool \
  outputs/BHCDETR-UHR-R50-1024-devv2-seed42-5090/data_audit.json
```

#### 5.1.2 运行 20 个 optimizer step 的显存/吞吐冒烟测试

先运行与上述精确语义直接相关的回归测试；四组测试全部通过后再开始 GPU smoke：

```bash
cd ~/autodl-tmp/xh-202625-master
python -m pytest \
  tests/test_trainer_metrics.py \
  tests/test_bhcl.py \
  tests/test_uhr_small_object.py \
  tests/test_bhcdetr_detection_loss.py \
  -q
```

冒烟测试必须使用单独输出目录，不能把它生成的 `last.pt`、`best.pt` 和 `metrics.jsonl` 写入正式实验。下面先从 local 配置复制一个 smoke 副本，再只替换实验名和输出目录：

```bash
cd ~/autodl-tmp/xh-202625-master
cp configs/bhcdetr_uhr_5090.local.yaml \
  configs/bhcdetr_uhr_5090.smoke.local.yaml

sed -i \
  's/BHCDETR-UHR-R50-1024-devv2-seed42-5090/BHCDETR-UHR-R50-1024-devv2-seed42-5090-smoke/g' \
  configs/bhcdetr_uhr_5090.smoke.local.yaml

mkdir -p outputs/BHCDETR-UHR-R50-1024-devv2-seed42-5090-smoke
set -o pipefail
python -u scripts/train.py \
  --config configs/bhcdetr_uhr_5090.smoke.local.yaml \
  --device cuda:0 \
  --max-steps 20 2>&1 | \
  tee outputs/BHCDETR-UHR-R50-1024-devv2-seed42-5090-smoke/smoke_console.log
```

训练启动日志应包含类似下列吞吐配置摘要：

```text
throughput profile: source_batch=4 model_views=8 effective_views_per_update=8 ... val_interval=2
```

`--max-steps 20` 停止时会额外运行 1 个 validation batch 并写入 smoke checkpoint，这是预期行为。完成后检查 loss、实际 step 和错误日志：

```bash
grep -E \
  'throughput profile|epoch=|CUDA out of memory|训练失败' \
  outputs/BHCDETR-UHR-R50-1024-devv2-seed42-5090-smoke/smoke_console.log

tail -n 1 \
  outputs/BHCDETR-UHR-R50-1024-devv2-seed42-5090-smoke/metrics.jsonl | \
  python -m json.tool
```

建议在另一个 AutoDL 终端持续观察利用率、显存、功耗和时钟。任选一个命令即可；按 `Ctrl+C` 停止监控：

```bash
nvidia-smi dmon -s pucm -d 1
```

或者：

```bash
watch -n 1 \
  'nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits'
```

不要只看初始化后的瞬时显存；至少观察若干稳定训练 step。冒烟通过的最低条件是：20 个 optimizer step 完成、所有 loss 有限、没有 OOM，且 GPU 利用率和显存占用相比 `batch_size: 1` 明显提高。

#### 5.1.3 OOM 时回退到 batch 2 / accumulation 2

如果 `batch_size: 4` 冒烟测试发生 CUDA OOM，先只调整吞吐批量，不改模型结构或 `uhr_max_local_tokens`：

```bash
cd ~/autodl-tmp/xh-202625-master
sed -i 's/^  batch_size: 4$/  batch_size: 2/' \
  configs/bhcdetr_uhr_5090.local.yaml
sed -i 's/^  gradient_accumulation_steps: 1$/  gradient_accumulation_steps: 2/' \
  configs/bhcdetr_uhr_5090.local.yaml

cp configs/bhcdetr_uhr_5090.local.yaml \
  configs/bhcdetr_uhr_5090.smoke.local.yaml
sed -i \
  's/BHCDETR-UHR-R50-1024-devv2-seed42-5090/BHCDETR-UHR-R50-1024-devv2-seed42-5090-smoke/g' \
  configs/bhcdetr_uhr_5090.smoke.local.yaml

set -o pipefail
python -u scripts/train.py \
  --config configs/bhcdetr_uhr_5090.smoke.local.yaml \
  --device cuda:0 \
  --max-steps 20 2>&1 | \
  tee outputs/BHCDETR-UHR-R50-1024-devv2-seed42-5090-smoke/smoke_console.log
```

`batch_size: 2`、`views_per_image: 2`、`gradient_accumulation_steps: 2` 仍是名义上的每次更新 4 张 source image / 8 个 view。若这个回退仍 OOM，再降低 `uhr_max_local_tokens` 就属于模型容量消融，必须换新的实验名和输出目录，不能静默混入 5090 吞吐实验。

#### 5.1.4 批量语义与严谨对照

标准配置与 5090 配置的名义更新规模为：

| 配置 | source batch | 每张 source 的 view | 累积步数 | 每次 optimizer update 的 source/view |
|---|---:|---:|---:|---:|
| 标准 | 1 | 2 | 4 | 4 / 8 |
| 5090 | 4 | 2 | 1 | 4 / 8 |
| OOM 回退 | 2 | 2 | 2 | 4 / 8 |

在当前 3,548 张训练图划分下，三者名义上都是每轮 887 次 optimizer update，因此 warmup、epoch 8/11 里程碑和整个 LR schedule 不变。不要因为单次 forward 的 batch 变大而线性放大学习率；继续使用主干外 `1e-4`、backbone `1e-5`。

但是，`4 × 2 × 1` **不等价于** `1 × 2 × 4`：检测损失会在包含更多 source/view 的单次 forward 上聚合，BHCL 单次能看到的对比样本集合以及 prototype 更新频率也会变化。`2 × 2 × 2` 同样改变这两项 batch semantics。因此 5090 配置属于独立训练实验，而不是只改变硬件吞吐的位级等价复现。

为了得到可解释的对照，建议按模板保留：

```yaml
resume: null
init_checkpoint: "outputs/BHCDETR-R50-1024-devv2-seed42/best.pt"
```

也就是从同一个旧 BHC `best.pt` 重新 warm-start 5090 实验。技术上可以把慢配置的 UHR `last.pt` 恢复后再改变 batch，但这会在一次训练中混合两套 detection loss/BHCL batch semantics，不应作为严谨消融或公平对照，也不应继续写入原慢配置输出目录。

#### 5.1.5 正式训练

20 步 smoke 通过后，使用未改名的 `configs/bhcdetr_uhr_5090.local.yaml` 启动新的正式实验：

```bash
cd ~/autodl-tmp/xh-202625-master
tmux new -s bhcdetr_uhr_5090
```

进入 tmux 后执行：

```bash
cd ~/autodl-tmp/xh-202625-master
mkdir -p outputs/BHCDETR-UHR-R50-1024-devv2-seed42-5090
set -o pipefail
python -u scripts/train.py \
  --config configs/bhcdetr_uhr_5090.local.yaml \
  --device cuda:0 2>&1 | \
  tee outputs/BHCDETR-UHR-R50-1024-devv2-seed42-5090/train_console.log
```

正式配置的 `validation_interval: 2` 表示在人类计数的第 2、4、6、8、10、12 轮运行完整验证。未验证的轮次在 `metrics.jsonl` 中会出现：

```json
{"validation_ran": false, "val": {}}
```

相应控制台中的 `val_loss=nan` 只是“本轮跳过验证”的占位，不代表训练 loss 变成 NaN；应同时检查该轮 `train.loss_total` 是否有限。`best.pt` 只会在实际运行验证的轮次更新，`last.pt` 仍然每轮更新；第 12 轮是最终轮，必定验证。

### 5.2 中断恢复

先复制当前 local 配置：

```bash
cp configs/bhcdetr_uhr.local.yaml configs/bhcdetr_uhr.resume.yaml
```

编辑 `configs/bhcdetr_uhr.resume.yaml`：

```yaml
init_checkpoint: null
resume: null
```

然后严格恢复同一 UHR checkpoint：

```bash
cd ~/autodl-tmp/xh-202625-master
set -o pipefail
python -u scripts/train.py \
  --config configs/bhcdetr_uhr.resume.yaml \
  --resume outputs/BHCDETR-UHR-R50-1024-devv2-seed42/last.pt \
  --device cuda:0 2>&1 | \
  tee -a outputs/BHCDETR-UHR-R50-1024-devv2-seed42/train_console.log
```

恢复会严格加载 UHR 模型、BHCL 原型、optimizer、scheduler、scaler、epoch 和 global step。不要同时保留非空 `init_checkpoint`。

## 6. 推理配置

UHR 适配仍使用现有 `bhcdetr` adapter，不新建 adapter 名称。复制现有推理模板：

```bash
cd ~/autodl-tmp/xh-202625-master
cp configs/infer.example.yaml configs/infer_uhr.local.yaml
```

至少修改以下字段：

```yaml
model:
  adapter: "bhcdetr"
  checkpoint: "outputs/BHCDETR-UHR-R50-1024-devv2-seed42/best.pt"
  image_size: 1024
  confidence: 0.001
  max_detections: 300
  half: true

device: "cuda:0"
# 先用 1 验证正确性和显存；完成实测后可把 2/4 作为独立吞吐消融。
batch_size: 1

input:
  data_root: "../data"
  manifest: "data/splits/dev_v2_airport_proxy_k60.json"
  split: "val"

tiling:
  enabled: true
  force: false
  tile_size: 1024
  overlap: 200
  fine_nms_iou: 0.55
  coarse_nms_iou: 0.85
  max_detections: 2000

score_thresholds:
  ship: 0.001
  aircraft: 0.001
  vehicle: 0.001
fine_score_thresholds: {}

output_json: "outputs/BHCDETR-UHR-R50-1024-devv2-seed42/val_predictions_low.json"
benchmark_output: "outputs/BHCDETR-UHR-R50-1024-devv2-seed42/benchmark_10k.json"
```

不要因为名称中有 UHR 就关闭外层 tiling。当前模型的 UHR 适配发生在每个输入/tile 内；对 10K 图关闭 tiling 会把整幅图 letterbox 到 1024，极小目标会严重丢失。

首次推理保持 `confidence=0.001` 和三个粗类阈值均为 0.001，用于保存完整的低阈值候选。阈值必须在跨 tile 融合后的低阈值结果上选择。

## 7. 验证集 GT、推理和评估

如果尚未导出 `dev_v2` 验证 GT：

```bash
cd ~/autodl-tmp/xh-202625-master
python scripts/export_coco.py \
  --data-root /root/autodl-tmp/data \
  --split train \
  --manifest data/splits/dev_v2_airport_proxy_k60.json \
  --manifest-split val \
  --output outputs/dev_v2_val_gt.json
```

运行低阈值推理：

```bash
python scripts/infer.py --config configs/infer_uhr.local.yaml
```

检查预测格式：

```bash
python scripts/validate_predictions.py \
  --pred outputs/BHCDETR-UHR-R50-1024-devv2-seed42/val_predictions_low.json \
  --gt outputs/dev_v2_val_gt.json
```

评估未经阈值优化的候选：

```bash
python scripts/evaluate.py \
  --gt outputs/dev_v2_val_gt.json \
  --pred outputs/BHCDETR-UHR-R50-1024-devv2-seed42/val_predictions_low.json \
  --project-config configs/project.yaml \
  --output outputs/BHCDETR-UHR-R50-1024-devv2-seed42/val_metrics_low.json
```

硬门槛是 pooled：

```text
Overall Recall >= 0.85
Overall FDR    <= 0.20
```

同时必须查看 ship、aircraft、vehicle 的 macro 指标及 25 个细类，不能让样本量最大的飞机掩盖 FSC、QHS 或 MS 的退化。

## 8. 阈值调优

### 8.1 全局阈值扫描

```bash
python scripts/sweep_thresholds.py \
  --gt outputs/dev_v2_val_gt.json \
  --pred outputs/BHCDETR-UHR-R50-1024-devv2-seed42/val_predictions_low.json \
  --output-dir outputs/BHCDETR-UHR-R50-1024-devv2-seed42/threshold_sweep \
  --project-config configs/project.yaml \
  --threshold-stage post_fusion
```

如果 0.01 网格接近门槛，再在边界附近使用 0.001 网格，不要先假设旧 BHC 模型的 0.23/0.27/0.305 阈值仍适用于 UHR checkpoint。新局部 attention 会改变置信度分布，必须重新校准。

### 8.2 三个粗类独立阈值

```bash
python scripts/tune_thresholds.py \
  --gt outputs/dev_v2_val_gt.json \
  --pred outputs/BHCDETR-UHR-R50-1024-devv2-seed42/val_predictions_low.json \
  --project-config configs/project.yaml \
  --min-threshold 0.0 \
  --max-threshold 0.50 \
  --step 0.005 \
  --matching-policy fine \
  --output outputs/BHCDETR-UHR-R50-1024-devv2-seed42/coarse_threshold_tuning_step005.json \
  --best-pred outputs/BHCDETR-UHR-R50-1024-devv2-seed42/val_predictions_coarse_tuned_step005.json \
  --best-metrics outputs/BHCDETR-UHR-R50-1024-devv2-seed42/val_metrics_coarse_tuned_step005.json
```

正式规则要求 25 个细类一致，因此保留 `--matching-policy fine`。`coarse` 只能用于定位/粗类正确时的理论诊断，不能作为正式成绩。

评估最佳粗类阈值结果：

```bash
python scripts/evaluate.py \
  --gt outputs/dev_v2_val_gt.json \
  --pred outputs/BHCDETR-UHR-R50-1024-devv2-seed42/val_predictions_coarse_tuned_step005.json \
  --project-config configs/project.yaml \
  --output outputs/BHCDETR-UHR-R50-1024-devv2-seed42/val_evaluation_coarse_tuned_step005.json
```

选择粗类阈值时，`fine_score_thresholds` 应保持空映射，否则细类阈值会覆盖粗类阈值：

```yaml
fine_score_thresholds: {}
```

### 8.3 细类阈值仅作诊断

```bash
python scripts/tune_fine_thresholds.py \
  --gt outputs/dev_v2_val_gt.json \
  --pred outputs/BHCDETR-UHR-R50-1024-devv2-seed42/val_predictions_low.json \
  --project-config configs/project.yaml \
  --min-threshold 0.001 \
  --max-threshold 0.50 \
  --step 0.005 \
  --output outputs/BHCDETR-UHR-R50-1024-devv2-seed42/fine_threshold_tuning_step005.json \
  --best-pred outputs/BHCDETR-UHR-R50-1024-devv2-seed42/val_predictions_fine_tuned_step005.json \
  --best-metrics outputs/BHCDETR-UHR-R50-1024-devv2-seed42/val_metrics_fine_tuned_step005.json
```

25 类阈值在同一份 933 图验证集上搜索和报告会产生明显乐观偏差，尤其 HM、LQS 和 FSC 样本很少。该结果用于判断模型校准上限，最终阈值必须通过 CV3 OOF 或独立校准集冻结。

## 9. 当前可执行的消融

所有消融都从正式模板复制新的 local YAML，使用不同的 `experiment_id` 和 `output_dir`，严禁覆盖正式结果。固定数据划分、seed、旧 `init_checkpoint`、训练轮数和后处理，只改变一个因素。

### 9.1 基础对照

建议至少比较：

| 实验 | 修改 | 目的 |
|---|---|---|
| A0 | 上一阶段旧 BHC `best.pt`，不继续训练 | 当前真实基线 |
| A1 | `uhr_enabled: true`，完整默认配置 | 判断整体适配收益 |
| A2 | A1，但 `gain_lpm_weight: 0.0` | 判断 LPM 的增益 |
| A3 | A1，但 `uhr_patch_budget: 1` | 判断更强稀疏路由的 Recall 风险 |
| A4 | A1，但 `uhr_patch_budget: 2` | patch budget 中间点 |
| A5 | A1，但 `uhr_max_local_tokens: 512` | 相对 1024-token 基线的精度、显存和时延权衡 |

还可以在前述实验稳定后单独比较 `uhr_patch_size: 256/512/768`。不要同时修改 patch size 和 budget，否则无法解释差异。

当前配置没有独立关闭 Global Cross-Attention、把 ISSGA 替换成 rigid NMS 或切换到 CUDA MSDA 的开关。因此不能用配置文件伪造论文完整的 `Rigid → ISSGA → +LPM → +GCA` 消融。若以后新增了经过测试的明确开关，再补做这组消融。

### 9.2 每个消融必须记录

- `init_checkpoint_report.json`；
- `metrics.jsonl` 和训练日志；
- `best.pt`、`last.pt` 及配置文件；
- 低阈值预测；
- pooled Recall/FDR；
- ship、aircraft、vehicle macro Recall/FDR；
- FSC、QHS、MS 等弱类的 TP/FP/FN；
- 真实 10K 的模型时延、总时延和峰值显存。

单次 `dev_v2` 上通过门槛不能证明稳定泛化。确定候选后，使用 `data/splits/cv3_airport_proxy_k60_v2.json` 完成三个 held-out fold，并在 OOF 预测上统一选择阈值。

## 10. 10K 时延测试

复制出的 `configs/infer_uhr.local.yaml` 同时用于验证推理和 10K 测速，确保 checkpoint、tile、batch、half precision、融合和阈值一致。

在 RTX 3090 或评分方案认可的等效硬件上，使用真实 10,000×10,000 图像：

```bash
cd ~/autodl-tmp/xh-202625-master
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/benchmark.py \
  --config configs/infer_uhr.local.yaml \
  --image /root/autodl-tmp/benchmark/real_10000x10000.png \
  --warmup 3 \
  --runs 10 \
  --device cuda:0 \
  --official-hardware-claim \
  --output outputs/BHCDETR-UHR-R50-1024-devv2-seed42/benchmark_10k.json
```

只有真实图像、合格硬件和冻结正式配置同时满足时，才可使用 `--official-hardware-claim`。未提供 `--image` 时脚本生成的零像素代理只能检查工程路径，不能作为正式时延证据。

当前适配仍执行外层滑窗，因此不能预先承诺论文所报告的 10× 加速。`uhr_max_local_tokens` 主要减少 decoder 局部 attention 的 token 数，ResNet-50 对所有外层 tile 的卷积仍然存在。最终必须实测整条流水线是否满足每幅 10K 图不超过 20 秒。

## 11. 常见错误

### `resume and init_checkpoint are mutually exclusive`

恢复 UHR `last.pt` 前，把恢复配置中的 `init_checkpoint` 改为 `null`。旧 BHC `best.pt` 只能用于第一次 warm-start。

### `init_checkpoint does not exist`

从项目根目录执行命令，并核对：

```bash
test -f outputs/BHCDETR-R50-1024-devv2-seed42/best.pt
```

### OOM

正式模板的 `batch_size` 已经是 1，不能再降；梯度累积也不会降低单次 forward 的峰值显存。可先将 `uhr_max_local_tokens` 从 1024 降至 512，仍不足时再降至 256。每次改动都属于模型消融，必须使用新的实验名和输出目录，不能在正式训练中途静默修改。

### `loss_gain_map` 或 `loss_gain_lpm` 非有限

立即停止并保留日志/checkpoint。先运行 `tests/test_uhr_small_object.py`，再检查 AMP、输入 boxes、padding mask 和配置值；不要用跳过非有限 loss 的方式继续训练。

### UHR 模型推理结果与旧模型阈值差异很大

这是正常可能性。新 local attention 会改变分类 logits 的分布。必须从 `confidence=0.001` 的新低阈值预测重新做 post-fusion 阈值扫描，不能直接复用旧阈值。

### 把 `tiling.enabled` 关闭后 10K 速度很快但 Recall 崩溃

这不是有效优化。当前实现不是整幅 10K raw-patch router；关闭 tiling 会把 10K 图缩到 1024。保持冻结滑窗几何，并通过真实测速评估 tile 内稀疏 token 的收益。

## 12. 结果判断

一个可报告的 UHR-BHC-DETR 候选至少应同时满足：

1. `init_checkpoint_report.json` 证明旧 BHC 权重被正确 warm-start；
2. Gain Map、LPM、BHC detection 和 BHCL loss 全程有限；
3. `dev_v2` 上完整评估优于或不劣于旧 BHC 基线；
4. CV3 OOF 上 pooled Recall/FDR 及三个粗类 macro 指标稳定；
5. 阈值只使用训练/OOF 数据冻结，未查看正式测试标签；
6. 真实 10K 图在认可硬件上的端到端时延不超过 20 秒；
7. 报告明确称其为“裁剪数据兼容的 UHR 论文适配”，不冒充论文完整 UHR-DETR。

这套适配重点改善的是 C4 小目标表征和全局/局部 attention 分配。它不等同于超分辨率，也不能仅凭新增模块保证 FSC 的 Recall/FDR 提升；最终结论必须由逐类 OOF 指标和真实 10K 测速共同决定。
