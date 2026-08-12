# R1-1 飞机候选框提议域适配与旋转一致性实验计划

日期：2026-08-12  
状态：`implemented_ready_for_server`  
主实验：`R1-1-AIRCRAFT-PROPOSAL-REFINEMENT`

## 1. 为什么下一步不再改 YOLO 主干

当前正式 M1 OOF、Y1-C2 和 R1-0 已经给出足够明确的错误定位：

1. R1-0 在冻结 C2 后净增 168 TP，说明真实 proposal 中存在可恢复的细类信息；
2. 事后 aircraft-only 路由净增 170 TP、FP 减少 282，aircraft macro Recall
   `0.90548→0.92335`、macro FDR `0.13881→0.12278`；
3. 同一个全类别策略却令 ship Recall 下降 `0.05268`，vehicle macro FDR 上升
   `0.07535`；
4. P03 已证明 GT crop 上普通 ConvNeXt-T 能达到约 `0.97` macro Recall，R1-0 的
   剩余问题主要是 proposal 偏移、旋转波动和微调遗忘，而不是缺少更大的检测主干。

因此，当前最高性价比路径是把已验证的飞机候选框分类信号做稳，而不是同时替换 neck、
loss、attention 和检测头。LSKNet/P2/专家路由等检测结构方法留到定位或小目标召回重新成为
主瓶颈时再启动。

## 2. 主假设与 2×2 因子设计

主假设：

> 从对应 fold 的 P03 fixed-epoch checkpoint 初始化，只在另外两折的真实 aircraft
> proposal 上做 5 个固定 epoch 的低学习率适配，并在教师原本预测正确的样本上施加
> soft-logit 锚定，可在纠正 proposal 域错误的同时减少 broken TP；D4 八视图概率平均
> 可进一步压低任意朝向造成的预测方差。

实验不是把多个模块混成一个黑盒，而是完整比较：

| 条件 | 提议域训练 | 推理视图 | 作用 |
|---|---|---|---|
| `p03_identity` | 无 | 单视图 | 正式 aircraft-only 零训练参考 |
| `p03_d4` | 无 | D4 | 单独测旋转集成 |
| `ce_identity` | 5 epoch CE | 单视图 | 单独测普通提议域适配 |
| `ce_d4` | 5 epoch CE | D4 | CE 与旋转集成组合 |
| `selective_anchor_kd_identity` | CE + 选择性锚定 KD | 单视图 | 单独测防遗忘 |
| `selective_anchor_kd_d4` | CE + 选择性锚定 KD | D4 | 预注册主条件 |

主条件只有一个；其余五项是因子消融，不根据 held-out 结果反复改训练参数。

## 3. 数据合同与防泄漏

### 3.1 训练数据

来源：`outputs/N2-PROPO-CROP-v2/proposal_crop_manifest.csv`。

筛选必须同时满足：

- GT `class_id∈[4,23]`；
- detector 原始 `detector_category_id∈[4,23]`；
- view 为 `deployable_positive` 或 `oracle_positive`；
- 标签来自 v2 修复后的 matched/oracle GT，而不是 detector 预测类。

冻结后为 17,948 行，fold 0/1/2 为 6,325/6,220/5,403；20 个飞机类最少 267、
最多 2,126。跨大类 oracle 样本被排除，因为它们在正式 aircraft router 中不可达。

### 3.2 外层三折

对 held-out fold `h`：

- P03 初始化使用 fold `h` 的 checkpoint；
- proposal 微调只读取另外两个 fold；
- held-out rows 不加载、不验证、不选 epoch；
- checkpoint 固定为 epoch 5 的最后状态；
- 只在训练完成后对 held-out 的 M1 aircraft proposal 推理一次。

每种条件最终形成完整 OOF 概率。门控在另外两折各自的 OOF 输出上选择，再应用到 held-out
折，避免使用当前模型在自己训练 proposal 上的预测调阈值。

### 3.3 结构性安全边界

- 只有 detector 原大类为 aircraft 的 32,062 个 proposal 进入对象头；
- ship/vehicle 的类别、score、bbox 逐条原样旁路；
- aircraft 只能在 20 个飞机细类内部变化；
- bbox 永不修改；
- 不引入 background 类，避免把重分类和背景拒识混成一个变量。

## 4. 模型与损失

学生和教师均为 P03-F ConvNeXt-T 25 类模型。训练只对 logits `[4:24]` 计算条件飞机
分类损失：

\[
L_{CE}=CE(z^S_{4:24},y;\epsilon=0.05)
\]

选择性锚定蒸馏只作用于教师 top-1 与真值一致的样本：

\[
L=L_{CE}+0.5\,\mathbf 1[\arg\max z^T=y]\,
T^2 KL(p^T_T\Vert p^S_T),\quad T=2.
\]

这样教师正确时抑制灾难性漂移；教师错误时不强迫学生复现旧错误，只由真值 CE 纠正。
该设计由 Learning without Forgetting/KD 的能力保持思想导出，但“仅锚定教师正确样本”是
本项目针对 new TP/broken TP 矛盾的实现，不冒充某篇论文的原样复现。

冻结训练参数：natural sampling、随机 D4、epoch 5、backbone LR `2e-5`、head LR
`1e-4`、AdamW、weight decay `0.05`、无 held-out early stopping。

## 5. D4 推理

每个 proposal 生成 `r0/r90/r180/r270` 与其水平翻转后的八个精确视图，不做任意角插值。
对每个视图先计算 20 类条件 softmax，再平均概率。遥感俯视飞机没有固定“朝上”语义，
离散旋转/反射是本任务真实对称性；这也是 G-CNN 中利用群对称性降低样本复杂度的基本
出发点。当前先用无需改网络的推理集成验证价值，只有收益稳定才考虑旋转等变骨干。

## 6. 比较与门禁

所有条件经过同一 frozen Y1-C2 后比较。预注册主条件相对 `p03_identity` 必须：

- Overall Recall ≥ 0.85、FDR ≤ 0.20；
- aircraft macro Recall 下降不超过 0.002；
- aircraft macro FDR 上升不超过 0.002；
- 配对 net TP ≥ 0；
- ship/vehicle 的 macro/pooled Recall/FDR 最大绝对变化 ≤ `1e-12`。

同时报告 new/broken/retained TP、FP/FN delta、三折方向，以及 SU-24、FA-18、C-5、
SU-35 的流入流出。即使主门禁通过，也只进入最终系统候选，不把同一 OOF 语料上的迭代
开发误写成独立测试集正式结论。

## 7. 后续论文方法优先级

### 7.1 本轮执行

1. **Learning without Forgetting / logit KD**：直接约束 broken TP；
2. **D4 rotation/reflection probability ensemble**：直接处理遥感任意朝向；
3. 普通 CE proposal adaptation：作为防遗忘损失的必要对照。

### 7.2 若本轮只改善头类、尾类仍不稳

下一轮只选一种长尾分类方法，不再重采样：

- **Balanced Softmax**：处理 proposal 训练分布与均衡 macro 目标之间的先验偏置；
- **Decoupling Representation and Classifier / cRT**：先适配表征，再冻结表征重训平衡分类器；
- **Logit Adjustment**：作为无需重训或轻量校准的对照。

P03 的 sqrt-inverse 已经没有带来稳定收益，因此不重复“换一个 sampler”作为创新。

### 7.3 若剩余错误集中在相似机型的局部部件

再启动一个模型级方向：

- **Progressive Multi-Granularity Training (PMG)**：tight crop 与局部颗粒特征；
- **class-center fine-grained loss**：增大 FA-18/SU-24 等相似类间隔；
- **DINOv2-B CLS+patch teacher**：P04 已显示其冻结特征强于 ConvNeXt/CleanDIFT，
  可作为局部结构教师，而不是先上扩散模型。

CleanDIFT 在 P04 frozen probe 中低于 DINOv2-B；扩散增强尚未有正式真实 OOF 证据。因此
它们不是当前第一顺位。先把低成本、与错误机制一一对应的方法跑完，再决定是否增加大教师。

## 8. 参考论文

- Learning without Forgetting
- Distilling the Knowledge in a Neural Network
- Group Equivariant Convolutional Networks
- Balanced Meta-Softmax for Long-Tailed Visual Recognition
- Decoupling Representation and Classifier for Long-Tailed Recognition
- Fine-Grained Visual Classification via Progressive Multi-Granularity Training of Jigsaw Patches
- Exploration of Class Center for Fine-Grained Visual Classification

## 9. 实现索引

- `configs/experiments/r1_aircraft_refinement_v1.yaml`
- `src/rsdet/analysis/aircraft_refinement.py`
- `scripts/r1_aircraft_refinement.py`
- `scripts/server/run_r1_aircraft_refinement.sh`
- `tests/test_aircraft_refinement.py`
- `docs/server/R1_AIRCRAFT_REFINEMENT_TASK_01.md`

