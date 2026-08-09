# BHC-DETR 论文方法移植与运行手册

本文对应论文 [Balanced Hierarchical Contrastive Learning with Decoupled Queries for Fine-grained Object Detection in Remote Sensing Images](https://arxiv.org/abs/2512.24074)（CVPR 2026，arXiv:2512.24074v1）。

若要在本模型上启用 arXiv:2604.21435v1 派生的小目标 Gain Map、LPM、ISSGA 和
C4 局部细节路径，请改用
[`UHR_BHCDETR_TRAINING.md`](UHR_BHCDETR_TRAINING.md) 中的配置与 AutoDL 步骤。
原文档继续作为不启用 UHR 扩展时的 BHC-DETR 基线说明。

## 1. 当前交付状态

项目的活动训练、推理和测速入口已经切换为 `bhcdetr`：

- `scripts/train.py` 只接受 `model: bhcdetr`；
- `scripts/infer.py` 和 `scripts/benchmark.py` 只接受 `model.adapter: bhcdetr`；
- 活动依赖不再需要 Ultralytics；
- 旧 YOLO、RT-DETR 和 HPR 文件仅作为历史实验资产保留，不会被当前入口加载；
- 已在真实冻结划分上完成 dry-run：训练 3548 张、验证 933 张，共 25 个细类；
- 已完成小模型的训练一步、验证一步、checkpoint、恢复所需状态、推理及测速流水线冒烟验证。

这里的“完成”是指代码链路可执行，不表示模型已经完成正式训练，更不表示已经达到 Recall、FDR 或 20 秒门槛。正式精度和速度必须在训练服务器及指定硬件上实测。

## 2. 论文内容与代码的对应关系

| 论文方法 | 项目实现 |
|---|---|
| 分类查询与定位查询解耦 | `src/rsdet/models/bhcdetr.py` 的 `DecoupledDecoderLayer` |
| 两种查询拼接后做共享自注意力，再拆分 | 同一层的 `joint self_attn` 与 `split` |
| 分类、定位使用独立 cross-attention 和 FFN | `classification_branch`、`localization_branch` |
| 分类头只读取分类查询，回归头只读取定位查询 | `classifier`、`box_regressor` |
| 层级标签树 | `src/rsdet/models/hierarchy.py` |
| Eq. (7) 层级权重 | `HierarchySpec.level_weights` |
| Eq. (8) 类别平衡分母 | `BalancedHierarchicalContrastiveLoss._balanced_level_loss` |
| Eq. (9) 同祖先正样本与类别原型 | 同上 |
| Eq. (10) 分层 EMA 原型更新 | `update_prototype_bank` |
| 每个 decoder 层独立 BHCL 原型库 | `BHCDetrCriterion.bhcl_layers` |
| 两个随机增强视图 | `BHCDetrDataset` 与 `bhcdetr_collate` |
| Hungarian 一对一匹配 | `HungarianMatcher` |
| focal 分类、L1 与 IoU 类损失 | `src/rsdet/models/detection_loss.py` |
| BHCL 总权重 0.6、温度 0.1、epsilon 0.1 | 训练 YAML 的 `loss` 配置 |
| AdamW、学习率 5e-5 | `src/rsdet/engine/trainer.py` 与训练 YAML |

本项目标签层级为：

```text
root
├── ship       -> HM, LQS, QHS, MS
├── aircraft   -> A1 ... A20
└── vehicle    -> FSC
```

root 不建立原型。粗粒度 3 个节点、细粒度 25 个节点，共维护 28 个原型；每个 decoder 层拥有独立的 28 节点原型库。

## 3. 必须明确的复现边界

这是一份“论文核心方法在当前比赛数据上的可执行移植”，不是论文表格结果所用实验系统的逐组件复刻，原因如下：

1. 论文把方法接入 OrientedFormer，并用 RHINO 做了泛化验证；当前仓库没有这两个框架及其算子，因此使用自包含的 ResNet-50、单尺度 C5、vanilla DETR 编码器作为承载网络。
2. 论文数据和检测头使用旋转框；比赛训练标签只有水平 YOLO 框，没有角度监督，因此本实现输出 normalized `cxcywh`，匹配与训练使用水平 GIoU，而不是 Rotated IoU。
3. 论文未公开或未明确给出 projection dimension、训练 epoch、weight decay、随机平移幅度和标准 DETR 匹配权重。本项目为这些项给出了可审计的工程默认值。
4. 论文实验使用 4 张 RTX 4090；当前 trainer 是单设备实现，没有 DDP 或跨卡对比特征聚合。
5. 单尺度 C5 的 stride 为 32，对极小目标有能力风险。它适合作为第一步论文方法基线，不能据此声称已经完整复现 OrientedFormer/RHINO，也不能预先承诺比赛指标。

论文未明确参数的当前默认值：

| 参数 | 当前值 | 说明 |
|---|---:|---|
| projection dimension | 128 | 工程默认 |
| epochs | 120 | 工程默认 |
| weight decay | 1e-4 | 工程默认 |
| random shift ratio | 0.1 | 工程默认 |
| match class/L1/GIoU | 2/5/2 | 标准 DETR 风格默认 |
| decoder/encoder layers | 6/6 | 工程默认 |
| object queries | 300 | 工程默认 |

这些值均写在 YAML 和 checkpoint 中，后续实验修改时可以追溯。

## 4. AutoDL 实例与环境安装

### 4.1 创建实例

推荐租用 NVIDIA RTX 3090 或 RTX 4090 实例，直接选择同时满足 Python 3.10、PyTorch >= 2.1 和 CUDA 可用的 AutoDL PyTorch 镜像，并确保数据盘容量足够保存数据集、checkpoint 和预测结果。正式 20 秒门槛测速应使用评分方案认可的 RTX 3090 或等效硬件；4090 上的结果只能作为工程参考。

AutoDL 的数据盘挂载在 `/root/autodl-tmp`，适合保存数据集、项目、模型权重和高频读写输出；系统盘与数据盘的区别见 [AutoDL 目录说明](https://www.autodl.com/docs/env/)。本文统一采用以下布局：

```text
/root/autodl-tmp/
├── xh-202625-master/       # 项目代码
├── data/                   # images/train 与 labels/train
├── model_assets/           # 可选的离线 ResNet-50 权重
├── benchmark/              # 真实 10K 测速图
└── cache/                  # pip/torch 下载缓存
```

### 4.2 上传项目和数据

可以使用 JupyterLab、FileZilla、AutoDL 文件存储或 SCP。AutoDL 官方的上传方法见 [上传数据文档](https://www.autodl.com/docs/scp/)。如使用 SCP，下面两条命令在本地 Windows PowerShell 中执行，并把端口和主机替换为控制台显示的 SSH 信息：

```powershell
scp -rP <SSH端口> "F:\study\揭榜挂帅\xh-202625-master" root@<AutoDL主机>:/root/autodl-tmp/
scp -rP <SSH端口> "F:\study\揭榜挂帅\data" root@<AutoDL主机>:/root/autodl-tmp/
```

如果先上传到 AutoDL 文件存储 `/root/autodl-fs`，训练前应复制到本地数据盘，避免直接从网络盘读取大量小文件：

```bash
cp -a /root/autodl-fs/xh-202625-master /root/autodl-tmp/
cp -a /root/autodl-fs/data /root/autodl-tmp/
```

登录实例后检查目录：

```bash
source /root/.bashrc
cd /root/autodl-tmp/xh-202625-master
test -f pyproject.toml
test -d /root/autodl-tmp/data/images/train
test -d /root/autodl-tmp/data/labels/train
df -h /root/autodl-tmp
```

### 4.3 使用镜像自带环境

不创建新的 Conda 环境，直接使用 AutoDL PyTorch 镜像自带的 `base` 环境。首次配置实例时执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip
export TORCH_HOME=/root/autodl-tmp/cache/torch
mkdir -p "$PIP_CACHE_DIR" "$TORCH_HOME"

python -m pip install --upgrade pip setuptools wheel
cd /root/autodl-tmp/xh-202625-master
python -m pip install -e ".[model,dev]"
```

执行项目安装前，镜像本身必须已经提供 Python 3.10、PyTorch >= 2.1、匹配的 torchvision 和可用 CUDA。如果不满足，应在 AutoDL 控制台更换合适的 PyTorch 镜像，而不是在当前实例中创建新环境或自行替换 CUDA/Torch。

每次重新打开 SSH 终端或 tmux 会话时，先执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip
export TORCH_HOME=/root/autodl-tmp/cache/torch
cd /root/autodl-tmp/xh-202625-master
```

验证 GPU、Torch、torchvision 与 SciPy：

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch, torchvision, scipy; print('torch=', torch.__version__); print('torchvision=', torchvision.__version__); print('scipy=', scipy.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -c "import torch; assert torch.cuda.is_available(); x=torch.randn(1024,1024,device='cuda'); print(x.square().mean().item())"
```

必须看到 `cuda=True` 且设备名与租用实例一致。若为 `False`，先核对实例镜像、`nvidia-smi` 和所装 wheel，不要在容器中自行重装 NVIDIA 驱动或盲目安装完整 CUDA Toolkit。

## 5. 数据与配置检查

本文约定项目位于 `/root/autodl-tmp/xh-202625-master`，数据位于 `/root/autodl-tmp/data`。因此配置中的 `data.root: ../data` 无需修改。先在项目根目录检查原始数据：

```bash
cd /root/autodl-tmp/xh-202625-master
python scripts/check_dataset.py \
  --data-root /root/autodl-tmp/data \
  --split train \
  --official-train
```

复制正式配置，不要直接修改模板：

```bash
cp configs/models/bhcdetr_r50_1024.yaml configs/bhcdetr.local.yaml
```

检查 `configs/bhcdetr.local.yaml` 中至少这些字段：

```yaml
output_dir: "outputs/BHCDETR-R50-1024-devv2-seed42"
data:
  root: "../data"
  manifest: "data/splits/dev_v2_airport_proxy_k60.json"
architecture:
  backbone_pretrained: true
  backbone_weights: null
train:
  device: "cuda:0"
```

`backbone_pretrained: true` 且 `backbone_weights: null` 时，torchvision 第一次构建模型会下载 ResNet-50 权重，并写入前面设置的 `/root/autodl-tmp/cache/torch`。如果实例不能访问下载地址，应上传完整的 torchvision ResNet-50 `state_dict`，并改为：

```yaml
architecture:
  backbone_pretrained: false
  backbone_weights: "/root/autodl-tmp/model_assets/resnet50_state_dict.pth"
```

仅检查配置、文件和冻结划分，不构建模型：

```bash
python scripts/train.py --config configs/bhcdetr.local.yaml --dry-run
```

预期审计结果为：

```text
train_source_images = 3548
val_source_images   = 933
fine_classes        = 25
views_per_source    = 2
```

快速验证 forward、loss、backward、验证和 checkpoint 链路：

```bash
python scripts/train.py \
  --config configs/bhcdetr.smoke.yaml \
  --device cuda:0 \
  --max-steps 1
```

`bhcdetr.smoke.yaml` 是 64 像素、1 层 decoder 的软件测试配置，仅用于验证 AutoDL 的 CUDA forward、loss、backward、验证和 checkpoint 链路，其权重绝不能用于精度评估。

## 6. 正式训练

默认配置每次 forward 读取 4 张源图，每张生成 2 个独立增强视图，因此 BHCL 在同一次 forward 中能看到 8 个样本：

通过 SSH 训练时必须使用 tmux 或 screen，避免 SSH 断开后训练进程退出。AutoDL 也明确建议 SSH 任务使用守护会话，参见 [AutoDL SSH 文档](https://www.autodl.com/docs/ssh/)。如果镜像没有 tmux，先安装：

```bash
apt-get update
apt-get install -y tmux
```

创建训练会话：

```bash
tmux new -s bhcdetr
```

进入 tmux 后执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip
export TORCH_HOME=/root/autodl-tmp/cache/torch
cd /root/autodl-tmp/xh-202625-master

mkdir -p outputs/BHCDETR-R50-1024-devv2-seed42
set -o pipefail
python -u scripts/train.py --config configs/bhcdetr.local.yaml 2>&1 | \
  tee outputs/BHCDETR-R50-1024-devv2-seed42/train_console.log
```

按 `Ctrl+B`，再按 `D` 可退出但不停止训练；重新连接 SSH 后使用 `tmux attach -t bhcdetr` 返回。另开终端可监控：

```bash
# 终端 A：持续查看 GPU
nvidia-smi -l 2

# 终端 B：持续查看训练日志
tail -f /root/autodl-tmp/xh-202625-master/outputs/BHCDETR-R50-1024-devv2-seed42/train_console.log
```

主要输出：

```text
outputs/BHCDETR-R50-1024-devv2-seed42/
├── data_audit.json
├── metrics.jsonl
├── last.pt
└── best.pt
```

checkpoint 同时保存模型、每层 BHCL 原型库、优化器、scheduler、AMP scaler、epoch、global step 和配置。

中断后恢复：

```bash
cd /root/autodl-tmp/xh-202625-master
set -o pipefail
python -u scripts/train.py \
  --config configs/bhcdetr.local.yaml \
  --resume outputs/BHCDETR-R50-1024-devv2-seed42/last.pt 2>&1 | \
  tee -a outputs/BHCDETR-R50-1024-devv2-seed42/train_console.log
```

如果单张 24 GB GPU 显存不足，可以把 `batch_size` 从 4 降为 2，并把 `gradient_accumulation_steps` 改为 2。这样优化器的有效样本数接近不变，但 BHCL 每次 forward 只能看到 4 个视图；梯度累积不会扩大对比学习的正负样本集合，这是一个明确的复现折中。

`/root/autodl-tmp` 在实例关机后仍会保留，但本地数据盘没有冗余副本。每个重要阶段应把 checkpoint、配置和指标复制到 AutoDL 文件存储或下载到本地，参见 [AutoDL 数据可靠性说明](https://www.autodl.com/docs/maintenance/)。已挂载 `/root/autodl-fs` 时执行：

```bash
mkdir -p /root/autodl-fs/BHCDETR-R50-1024-devv2-seed42
cp -av configs/bhcdetr.local.yaml \
  outputs/BHCDETR-R50-1024-devv2-seed42/best.pt \
  outputs/BHCDETR-R50-1024-devv2-seed42/last.pt \
  outputs/BHCDETR-R50-1024-devv2-seed42/metrics.jsonl \
  /root/autodl-fs/BHCDETR-R50-1024-devv2-seed42/
```

## 7. 导出验证集 GT

冻结开发划分的图像物理上仍位于 `images/train`，所以 `--split` 使用 `train`，再由 manifest 选择逻辑 `val`：

```bash
cd /root/autodl-tmp/xh-202625-master
python scripts/export_coco.py \
  --data-root /root/autodl-tmp/data \
  --split train \
  --manifest data/splits/dev_v2_airport_proxy_k60.json \
  --manifest-split val \
  --output outputs/dev_v2_val_gt.json
```

## 8. 推理、校验与官方指标

复制推理配置：

```bash
cd /root/autodl-tmp/xh-202625-master
cp configs/infer.example.yaml configs/infer.local.yaml
```

在 `configs/infer.local.yaml` 中确认：

- `model.checkpoint` 指向正式训练的 `best.pt`；
- `input.data_root` 和 `input.manifest` 正确；
- 验证集使用 `input.split: val`；
- 初次推理保持 0.001 的低候选阈值；
- 可取消 `evaluation` 段注释并填写验证 GT。

运行：

```bash
python scripts/infer.py --config configs/infer.local.yaml
```

输出包括 COCO detection JSON、逐图阶段耗时 JSON，以及启用 `evaluation` 时的官方指标 JSON。

单独校验和评估：

```bash
python scripts/validate_predictions.py \
  --pred outputs/BHCDETR-R50-1024-devv2-seed42/val_predictions_low.json \
  --gt outputs/dev_v2_val_gt.json

python scripts/evaluate.py \
  --gt outputs/dev_v2_val_gt.json \
  --pred outputs/BHCDETR-R50-1024-devv2-seed42/val_predictions_low.json \
  --project-config configs/project.yaml \
  --output outputs/BHCDETR-R50-1024-devv2-seed42/val_metrics_low.json
```

评估结果同时包含：

- pooled overall Recall/FDR，用于 Recall >= 0.85、FDR <= 0.20 硬门槛；
- `official_ranking`，按船舶 4 细类、飞机 20 细类、车辆 1 细类分别做 macro 平均。

## 9. 阈值选择

必须先用低阈值保存完整候选，再在跨 tile 融合后的结果上扫描：

```bash
python scripts/sweep_thresholds.py \
  --gt outputs/dev_v2_val_gt.json \
  --pred outputs/BHCDETR-R50-1024-devv2-seed42/val_predictions_low.json \
  --output-dir outputs/BHCDETR-R50-1024-devv2-seed42/threshold_sweep \
  --project-config configs/project.yaml \
  --threshold-stage post_fusion
```

根据扫描结果把冻结工作点写回 `score_thresholds` 或 `fine_score_thresholds`，重新推理并保存最终验证指标。不能在正式测试集上选择阈值。

## 10. 三折交叉验证

首轮先用 `dev_v2` 打通和调参；冻结方案后再做 CV3。每折复制一份训练配置：

```yaml
data:
  root: "../data"
  manifest: "data/splits/cv3_airport_proxy_k60_v2.json"
  held_out_fold: 0  # 依次改为 0、1、2
```

推理该折验证集时使用：

```yaml
input:
  data_root: "../data"
  manifest: "data/splits/cv3_airport_proxy_k60_v2.json"
  split: "val"
  held_out_fold: 0
```

导出该折 GT：

```bash
python scripts/export_coco.py \
  --data-root /root/autodl-tmp/data \
  --split train \
  --manifest data/splits/cv3_airport_proxy_k60_v2.json \
  --manifest-split val \
  --held-out-fold 0 \
  --output outputs/cv3_fold0_gt.json
```

分别训练、推理和评估 fold 0/1/2，记录均值和最差折；不要只报告单次 `dev_v2` 最优结果。

## 11. 10000 x 10000 端到端测速

使用与正式推理完全相同的 checkpoint、tile、batch、融合和序列化设置。评分方案排除磁盘读图时间，因此脚本从图像已读入内存后开始计时，并在结束前同步 CUDA，避免漏计异步 GPU 工作。

在 RTX 3090 或经确认的等效硬件上运行：

```bash
cd /root/autodl-tmp/xh-202625-master
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/benchmark.py \
  --config configs/infer.local.yaml \
  --image /root/autodl-tmp/benchmark/real_10000x10000.png \
  --warmup 3 \
  --runs 10 \
  --device cuda:0 \
  --official-hardware-claim \
  --output outputs/BHCDETR-R50-1024-devv2-seed42/benchmark_10k.json
```

只有同时满足以下条件，才能把结果写为正式通过：

- 输入确实是 10000 x 10000；
- GPU 是 RTX 3090 或评分方案认可的等效硬件；
- 使用正式 checkpoint 和冻结推理配置；
- `competition_gate.official_gate_passed` 为 `true`；
- p95 不超过 20 秒。

未提供 `--image` 时脚本创建零像素代理图，只能检查工程路径和粗略性能，不能作为正式测速证据。

## 12. 已执行的验证

本次实现已完成以下检查：

- 216 个项目 Python 文件通过 UTF-8 AST 解析；
- BHC-DETR、BHCL、hierarchy、dataset 和 box loss 测试：23 passed；
- 非 Torch 的推理融合、指标、COCO 导出和接口回归：75 passed；
- 小模型 ResNet-50 forward -> Hungarian -> detection loss + BHCL -> backward 成功；
- 分类查询、定位查询、分类头、框回归头和 projection head 梯度均存在且有限；
- 真实 `dev_v2` dry-run：3548 train / 933 val；
- CPU 软件冒烟：训练一步、验证一步、`best.pt`/`last.pt`、单图推理和 benchmark 全部完成。

这些验证证明代码路径可执行；正式模型尚未在 CUDA 服务器完成 120 epoch 训练，因此目前没有可报告的正式 Recall、FDR 和 3090 10K 时延。
