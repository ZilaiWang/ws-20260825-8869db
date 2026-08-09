# 模型技术路线：BHC-DETR 论文方法移植

当前活动模型仅为 `bhcdetr`：使用 ResNet-50 DETR 承载论文
arXiv:2512.24074v1 的 Decoupled Queries 与 Balanced Hierarchical
Contrastive Learning。活动训练、推理、融合和测速入口已全部接通。

完整的论文—代码映射、复现边界与逐步命令见
[`BHCDETR_IMPLEMENTATION.md`](BHCDETR_IMPLEMENTATION.md)。

小目标训练的可选升级使用 arXiv:2604.21435v1 中可由当前裁剪数据监督的 Gain
Map、LPM 与 ISSGA，并从 C4/stride-16 提取有上限的局部 token。准确实现边界、
AutoDL 命令及消融方案见
[`UHR_BHCDETR_TRAINING.md`](UHR_BHCDETR_TRAINING.md)。

```text
RGB -> ResNet-50 C5 -> Transformer encoder
    -> Q_cls/Q_loc 拼接共享 self-attention 后拆分
    -> 两条独立 cross-attention + FFN
    -> 分类头(Q_cls) / 水平框头(Q_loc)
    -> focal + L1 + GIoU + 0.6 * BHCL

可选 UHR-inspired 小目标路径：
ResNet-50 C5 -> Gain Map -> DFL + LPM -> ISSGA 窗口
ResNet-50 C4 -> 每窗口配额 + 有上限的局部 token
decoder 每个任务分支：C5 global attention -> C4 local attention -> FFN
```

比赛层级为 `root -> ship/aircraft/vehicle -> 25 fine classes`。每个 decoder
层独立维护 3 个粗类和 25 个细类原型；每张源图生成两个增强视图。

重要边界：论文实验基线/最终系统采用 OrientedFormer（另验证 RHINO）与旋转框，
当前实现是单尺度 ResNet-C5 vanilla DETR 与水平框 GIoU。因此这是论文核心方法的
可执行移植，不是论文实验模型的完整复现；stride 32 对极小目标的风险必须由正式训练
与消融验证。当前没有正式训练指标，不能声称已达到 Recall/FDR/20 秒门槛。

启用小目标路径后，stride-32 单尺度瓶颈由 C4 局部 memory 缓解，但当前仍先在每个
1024 tile 上运行完整 ResNet-50，并使用 bounded dense MHA；它没有论文的 R18 整图
分支、raw-patch R50、动态 queries 或 MSDA，因而同样不能继承论文的 10K 加速结论。

以下 V1 路线仅为历史记录，已停用，不再作为运行指引。

<!--
# 不均衡小样本时敏目标检测技术路线（V1）

## 1. 问题判断

当前训练集有 4,481 张图、20,933 个框和 25 个细类。最大/最小细类实例数为
2,147/17，约 126.29:1；发射车只有 67 张正样本图。官方匹配又要求框和细类 ID
同时正确，所以把 25 类提前归并为舰船、飞机、车辆三类训练会直接破坏指标。

本题同时有三类约束：

1. **长尾与小样本**：HM、LQS 和 FSC 等类别容易欠拟合，简单全量过采样又会过拟合；
2. **细粒度识别**：20 个飞机型号外观接近，错分会同时产生一个 FP 和一个 FN；
3. **时敏大图**：10K 图必须切片检测，既要控制 tile 数量，也要处理重叠区重复框。

因此方法升级为 **HPR-TD（Hierarchical Prototype Refinement for Time-sensitive
Detection，层次化原型精修时敏检测）**：成熟实时检测器负责定位，项目自研 HPR
轻量网络负责少样本细类精修，HBR 两阶段再平衡负责稳定训练。各部分均可独立消融。

## 2. 总体路线

```text
冻结 dev_v1（已完成）/cv3（待完成）
  → COCO 预训练权重
  → 阶段 A：原始分布训练，学习稳定定位与通用表征
  → 阶段 B：有限重复采样 + 低学习率 + 冻结前部层 + 温和分类加权
  → 低阈值生成 25 类候选
  → 对不确定舰船/飞机框启动 HPR：放大裁剪 → 余弦分类 → EMA 原型融合
  → 10K 重叠切片坐标恢复
  → 细类 NMS → 同大类极高 IoU 去重
  → 舰船/飞机/车辆独立工作点
  → 官方 Recall/FDR + pipeline 时延联合选型
```

主线对比固定为：

| 编号 | 模型 | 输入 | 要回答的问题 |
|---|---|---:|---|
| M1 | YOLO26-s | 1024 | 速度基线是否已有足够候选召回 |
| M2 | YOLO26-m | 1280 | 更高容量/分辨率能否显著改善稀有类和细类混淆 |
| M3 | RT-DETR-L | 1024 | NMS-free 的不同候选机制是否值得作为备线 |

M1/M2 使用同一模型族、相同 split、相同两阶段预算和同一评测器，避免把多项变化
混在一个实验里。M3 只用于结构对照，不承担大规模调参。

## 3. 已落地的关键方法

### 3.1 有上限的类别感知重复采样

对训练 split 中类别 `c` 的正样本图频率记为 `f(c)`，阈值为 `t`，则图像 `i` 的
重复因子为：

```text
r(i) = min(r_max, max(1, max_c∈i sqrt(t / f(c))))
```

默认 `t=0.05`、`r_max=4`。小数部分用固定 seed 随机舍入。该策略只生成派生路径
清单，不复制图像、不修改标签；上限用于抑制极少类在几十轮内被机械记忆。

代码同时导出 effective-number 权重和每张图的 repeat factor，便于审计。V1 不直接
把极端 effective-number 权重硬塞进损失，而在再平衡阶段采用较温和的
`cls_pw=0.25`，避免“重复采样 + 强逆频率权重”双重补偿造成高 FDR。

### 3.2 表征学习与分类再平衡解耦

- 阶段 A 用原始分布训练，使定位分支和骨干网络看到真实场景先验；
- 阶段 B 从阶段 A 的 `best.pt` 开始，只用均衡清单，降低学习率并冻结前 10 层；
- 阶段 B 降低 Mosaic、关闭 MixUp，避免少量稀有目标被过强合成增强破坏细粒度外观。

这比从第 1 个 epoch 就强行全量过采样更适合 17/30 个实例的极小类，也可通过移除
阶段 B 做直接消融。

### 3.3 自研 HPR 少样本细粒度分支

YOLO/RT-DETR 仍作为可靠、快速的定位基线，但不再让其单一检测头独自承担 20 个相近
飞机型号和 4 个舰船细类的识别。HPR 对检测框加入 15% 上下文并 letterbox 到
`128×128`，使用项目实现的 72,329 参数轻量网络：

- 三级深度可分离卷积控制计算量；
- 局部 `3×3` 和空洞 `3×3` 双支路同时编码局部构型与较大尺度轮廓；
- 通道门控突出机翼、尾翼、舰艏等有判别力的结构；
- L2 归一化余弦分类器消除高频类分类权重范数天然更大的偏置；
- 三大类辅助头保证细类特征不破坏舰船/飞机/车辆的层次结构；
- 每个细类维护 EMA 原型，分类 logit 与原型相似度按权重 `α` 融合。

设归一化目标特征为 `z`、细类分类器权重为 `w_c`、EMA 原型为 `p_c`，则：

```text
l_cls(c) = s · cos(z, w_c)
l_hpr(c) = (1 - α) · l_cls(c) + α · s · cos(z, p_c)
```

训练使用框位置/尺度扰动模拟检测误差，并结合平方根逆频率采样、class-balanced focal、
LDAM、大类辅助损失以及批内/原型紧致约束。推理只重判基础置信度低于阈值的舰船和飞机，并把新标签限制在原大类内；
车辆只有一个细类，不进入 HPR。默认 `score_blend=0`，只改细类、不擅自改变检测分数。

这一设计直接针对官方规则：细类错分原本会同时产生 FP 和 FN；HPR 在不重复运行完整
检测骨干的情况下，把小目标放大后单独辨型。未训练网络在 RTX 4060 Laptop 上的
FP32 冒烟测速为 batch 1/32/128 约 2.96/3.44/11.15 ms，批量时约 0.09–0.11 ms/框；这只是
HPR 前向开销，不是 3090 官方 pipeline 时延结论。

### 3.4 分层跨切片抑制

大图重叠切片会让同一目标出现多次。系统先在相同细类内做 `IoU=0.55` 的 NMS，
再可选地在同一大类内用 `IoU=0.85` 去除“位置几乎相同但细类不同”的重复框。
第二级只处理极高重叠框，目的在于减少一个目标因跨 tile 细类抖动产生的额外 FP，
且不会把舰船、飞机和车辆互相抑制。该开关必须在 dev_v1 上做开/关消融后再冻结。

### 3.5 指标和时延共同驱动的工作点

模型内部使用低候选阈值（默认 0.03）保留召回，融合后对舰船、飞机、车辆分别设
工作阈值。正式选择不以 mAP 单独决定，而以：

```text
硬门槛：Recall ≥ 0.85，FDR ≤ 0.20，10K pipeline ≤ 20 s
内部目标：Recall ≥ 0.88，FDR ≤ 0.17，10K pipeline ≤ 17 s
```

M1 首先争取满足时延；只有 M2 对稀有类/飞机细类的收益超过时延代价时才升级主干。

## 4. 代码与配置

- `src/rsdet/data/imbalance.py`：冻结 manifest 校验、重复采样、权重统计和派生 YAML；
- `src/rsdet/engine/trainer.py`：基础训练/稀有类再平衡两阶段编排；
- `src/rsdet/models/ultralytics_adapter.py`：YOLO 与 RT-DETR 统一推理适配；
- `src/rsdet/models/prototype_refiner.py`：自研 HPR 网络、原型更新和组合损失；
- `src/rsdet/data/object_crops.py`：严格跟随 manifest 的 GT 目标裁剪数据；
- `scripts/train_refiner.py`：HPR 独立训练、验证和 checkpoint 入口；
- `src/rsdet/postprocess/tile_fusion.py`：坐标恢复与分层跨 tile 去重；
- `configs/models/`：M1、M2、M3 的冻结起始配置。

训练前先做不会占用 GPU 的完整审计：

```bash
cp configs/models/m1_yolo26s_1024.yaml configs/m1.local.yaml
# 编辑 configs/m1.local.yaml，填写 data.root
python scripts/train.py --config configs/m1.local.yaml --dry-run
```

审计会检查 manifest、图像/标签配对、25 类训练覆盖和路径越界，并在实验目录生成：

```text
prepared_data/
├── train.txt
├── train_balanced.txt
├── val.txt
├── dataset_base.yaml
├── dataset_balanced.yaml
└── imbalance_statistics.json
```

审计通过后去掉 `--dry-run` 开训。推理使用 `configs/infer.example.yaml`，正式测试必须
提供含稳定 `image_id` 的 manifest；`image_dir` 模式只用于本地调试。

基础检测器得到首个 `best.pt` 后，训练 HPR：

```bash
cp configs/refiner.example.yaml configs/refiner.local.yaml
# 编辑 data.root 和 data.manifest
python scripts/train_refiner.py --config configs/refiner.local.yaml
```

然后在推理配置的 `model.refiner` 中填写 HPR `best.pt`。HPR 与检测器使用同一份
`dev_v1`，但只用 train GT 裁剪训练，val 裁剪只用于选择 checkpoint。

10K 测速复用完全相同的切片、模型和融合代码，并输出 p50/p95 与峰值显存：

```bash
python scripts/benchmark.py --config configs/infer.example.yaml --image /path/to/10k.png
```

## 5. 必做消融与决策表

| 实验 | 改变量 | 结论用途 |
|---|---|---|
| A0 | M1 仅阶段 A | 原始长尾基线 |
| A1 | A0 + 有限 RFS | 判断稀有类收益及过拟合 |
| A2 | A1 + `cls_pw=0.25` | 判断残余分类加权是否降低细类漏检 |
| A3 | A2 + HPR（全部候选） | 判断细类混淆改善上限 |
| A4 | A3 + 置信度门控 | 测量大部分 HPR 收益能否以更低时延取得 |
| A5 | A4 + 分层跨 tile NMS | 判断 FDR 收益和 Recall 代价 |
| B0 | 完整 M2 | 判断容量/分辨率是否值得时延成本 |
| C0 | 完整 M3 | 判断不同结构候选质量 |

每项至少记录总体和三大类 Recall/FDR、25 类 Recall、混淆、p50/p95 pipeline 时延、
显存峰值。HM/LQS/FSC 单次 dev 波动大，最终方向需同时看 cv3 均值和最差 fold。

## 6. 当前边界与下一步

当前 HPR 已在 PyTorch 2.5.1/CUDA 12.1/RTX 4060 Laptop 上通过前向、反向、原型更新
和推理标签融合冒烟测试，仓库已包含冻结的 `dev_v1`，但 `cv3` 尚未生成，模型环境
与正式训练仍待完成，因此**没有真实模型精度或 3090 时延可报告**。另外，训练数据没有
纯背景图，FDR 的外推风险仍然很高。开始正式实验前必须完成：

1. 使用已冻结且无同源泄漏的 `dev_v1` 跑首轮实验，并补充 `cv3`；
2. 在匹配 CUDA 的环境安装 PyTorch/Ultralytics 并记录确切版本；
3. 先跑 M1 A0/A1/A2，再跑 HPR A3/A4，最后决定是否投入 M2；
4. 从 M1 低阈值预测中挖掘难负样本，但所有补充数据单独记录版本和许可；
5. 用真实或可追溯合成 10K 图完成 3090 预热后 p50/p95 测速。
-->
