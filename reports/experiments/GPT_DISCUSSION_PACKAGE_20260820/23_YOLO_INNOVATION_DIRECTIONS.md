# YOLO26-s 下一阶段改进方向：问题驱动的论文筛选与实验顺序

日期：2026-08-11
状态：`planning_authority`
基线：M1 YOLO26-s / 1024 / CV3 v2 / seed42 / 160 fixed epochs / `last.pt`

> 本报告只回答“基于当前可信证据，YOLO 下一步值得改什么、按什么顺序改”。
> 它不把论文中的公开数据集增益外推为本项目增益，也不以堆叠模块代替消融。

## 1. 先明确真正需要解决的问题

当前 cross-fit held-out pooled 基线为 Recall `0.9176`、FDR `0.1990`。它仅以
约 `0.001` 的余量通过 FDR 门槛，而且 fold0/2 的 FDR 仍为 `0.2259/0.2294`。
完整 V1.6 macro 更清楚地暴露了短板：

| 问题 | 当前证据 | 对 YOLO 改进的含义 |
|---|---|---|
| 车辆候选与质量 | vehicle Recall/FDR `0.6119/0.6239`；402 个车辆中 80 个无候选、71 个仅有低分候选 | 必须恢复浅层小目标信息，同时让新增候选具备可用分数与定位质量 |
| 舰船长尾虚警 | ship macro Recall/FDR `0.7162/0.5389`；LQS/HM 每类在船指标中各占 25% | 需要类别与空间校准、少样本稳定学习，不能只优化样本量占优的大类 |
| 背景虚警 | 正式分解 `FP_BG=3303`，占 FP 70.7% | 需要分数校准、困难样本学习；若做独立背景拒识，必须先完成人工背景确认 |
| 飞机细粒度混淆 | aircraft macro 较强，但 TU-160、F-22 是结构性短板；`FP_CLS=1115` | 可用层次/原型辅助监督，最终输出仍保持官方 25 细类 |
| 域间分数漂移 | 同一阈值仅 fold1 过 FDR 门槛 | 所有校准必须 cross-fit；不能在同一 OOF 上选阈值并回评 |
| P2 转化不足 | 历史 P2 三折均减少无候选，合计 `-43%`，但新增候选多落在低分桶 | P2 方向有机制证据，但“加一个 P2 头”不是最终答案 |

普通框回归不是当前主矛盾：`R_loc@oracle-class=0.9705`，正式 `FP_LOC=66`，
优先做 bbox diffusion 或复杂回归头的性价比很低。

## 2. 论文筛选原则

论文来源为项目外层目录
`论文资料_arXiv新找_20260806/README.md` 和 `INDEX.md`。本轮重点阅读了
P2、小目标特征融合、长尾校准、困难样本采样、旋转等变、遥感骨干、细粒度
层次监督与 proposal 重排相关原文。

筛选标准：

1. 直接对应上述已量化错误，而非只在别的数据集提高 mAP；
2. 能在 YOLO26-s 上形成单变量对照；
3. 优先无推理成本或低推理成本；
4. 有开源实现或足够明确的方法定义；
5. 必须接受当前 CV3、fixed-last、官方 pooled + macro 双口径验收。

## 3. 优先级结论

### 3.1 第一优先：cross-fit 类别—空间校准（已完成）

基线：M1 原始低阈值 OOF 与当前 cross-fit 阈值选择代码。
参考论文：**Fractal Calibration for Long-Tailed Object Detection (FRACAL)**。

FRACAL 是推理期 logit 后校准：除类别频率外，还用训练集中各类在图像空间的
分布统计修正分数。它比换骨干更直接对应当前的三个事实：长尾类权重被官方
macro 放大、舰船/车辆 FDR 很高、不同 fold 的分数分布发生漂移。论文同时给出
代码，并说明可用于 sigmoid 单阶段检测器。

建议按四级消融执行：

1. `C0`：现有 cross-fit 全局阈值；
2. `C1`：三大类共享阈值；
3. `C2`：只做有收缩约束的 class-prior logit adjustment；
4. `C3`：FRACAL-inspired 类别 + 空间分形代理校准。

当前 OOF 只保留 NMS 后的已选类别和单标量分数，因此 C2/C3
只是后 NMS 筛选，不能宣称为 FRACAL 完整复现。完整方法需要在
NMS 前对全部类别 logits 校准。

空间统计只允许从每个外层训练 fold 计算，阈值和超参只在内层校准数据拟合；
held-out fold 仅评一次。稀有类不能独立自由拟合大量参数，必须向船/飞机大类
统计收缩。

准入信号：pooled Recall 不下降超过 `0.005`；pooled FDR 明显低于当前
`0.1990`；ship/vehicle macro FDR 至少一项稳定下降，且 2/3 folds 同方向。
正式结果：C2 相对 C0 的 pooled Recall 变化为 `-0.0041`，
pooled FDR 变化为 `-0.0397`，macro FDR 变化为 `-0.0297`，
三折 FDR 均改善，因此 C2 准入。C1 跌破 FDR 硬门槛；C3 相对
C2 无增量，不准入，也不启动完整 FRACAL 工程改造。

### 3.2 第二优先：正式 P2 基线，再解决“候选有了但质量不足”

基线：官方 `yolo26s-p2.yaml`，只增加 stride-4 的
P2/P3/P4/P5 检测路径。注意 Ultralytics 的尺度字母必须位于
`yolo26` 之后；历史脚本中的 `yolo26-p2s.yaml` 会回退为 n 级，
不能与 M1 构成容量对齐的正式对照。
参考论文：

- **Edge-Constrained UAV Small-Object Detection with P2 Enhancement and
  Quantum-Inspired Lightweight Structure Search**；
- **FRFDet: Efficient UAV Small Object Detection with Symmetric Sampling and
  Scalable Fusion**。

现有历史结果已经说明 P2 可在三个 fold 一致减少车辆“完全无候选”，但该实验
使用 60 epoch / best.pt，不能与 M1 的 160 epoch / last.pt 做正式比较。因此
正确顺序不是继续加注意力，而是：

1. `S0`：正式重跑完整 P2，严格使用 160 epoch、fixed-last、三折低阈值 OOF；
2. `S1`：只有 S0 在 unique vehicle recovery 上有稳定净收益，才处理新增候选
   的分数/质量；
3. `S2`：在 P2 路径单独尝试 FRFDet 的一个思想，不同时搬两项：
   - 先测对称上下采样，减少浅层细节在融合中的空间错位；或
   - 再测尺度—特征关系融合，让 P2 与深层语义交互时抑制背景冗余；
4. `S3`：S2 有效后才裁剪成 P2-Lite，比较收益保留和开销下降。

S0/S1 的关键读数不是 Ultralytics mAP，而是：新增 unique vehicle TP、被破坏
TP、低分候选跨阈值数、`FP_BG/FP_CLS` 增量、vehicle Recall/FDR 以及每折结果。
若完整 P2 仍只增加低分候选而不改善官方指标，停止 P2-Lite，不继续堆模块。

### 3.3 第三优先：YOLO 内部的层次细粒度辅助监督

基线：YOLO26-s 原 25 类输出头。
参考论文：**Hierarchical Fine-Grained Aerial Object Detection**。

该论文的核心不是更大的检测器，而是把类别层次和视觉原型加入细粒度学习。
本项目没有论文所需的完整部件属性标注，因此不宜直接复现其属性重建模块；
可先做一个可审计的简化版本：

- 保留官方 25 类主损失；
- 增加 ship/aircraft/vehicle 粗类辅助损失；
- 在正样本分类特征上增加分层原型或监督对比约束；
- 只在训练期存在，正式输出接口和推理成本不变。

它主要针对 TU-160/F-22、LQS/HM 等类间混淆。准入要求是 `FP_CLS` 净下降，
目标细类 macro Recall/FDR 改善，且飞机总体与 pooled Recall 不退化。若只提高
训练集或 GT crop 分类而不改善 Pred-OOF 端到端指标，不进入最终模型。

### 3.4 第四优先：错误感知的反遗忘采样

基线：M1 natural sampler；P03 的 sqrt-inverse 结果已经证明“按频率静态过采样”
没有稳定优势。
参考论文：**Does YOLO Really Need to See Every Training Image in Every Epoch?**。

AFSS 用每图 precision/recall 的较小值表示学习充分度：困难图持续参与，中等图
保证短期覆盖，容易图低频回看防止遗忘。它比简单按类别频率重复采样更符合
当前错误分布，因为困难主要集中在小目标、特定机场代理组和易混淆类别，而非
只由类频决定。

建议先实现日志回放版，不立即改变训练：记录每 epoch 的图像级匹配充分度，
验证其与 held-out 错误、车辆 near-miss、尾类和困难 source group 的相关性。
相关性成立后再运行 `natural vs AFSS` 三折单因素实验。该方向无额外推理成本，
但需防止把漏标/模糊样本当成困难样本反复放大。

### 3.5 第五优先：离散旋转一致性训练

基线：M1 默认增强；当前正式模板没有显式冻结 90° 旋转策略。
参考论文：**Measuring the Impact of Rotation Equivariance on Aerial Object Detection**。

由于项目输出 HBB，不建议首先改造成完整旋转等变骨干或 OBB 检测器。低风险
实验是 0/90/180/270° 的离散旋转增强，或同图旋转前后分类/质量的一致性损失。
90° 变换后的 HBB 可精确变换，适合飞机和舰船的任意朝向。

只在来源隔离 CV3 上评估，并单独检查小目标插值损失。若旋转一致性改善尾类
但伤害车辆 Recall，则只用于对象 crop 学生或飞机/舰船训练，而不进入共享
YOLO 主干。

## 4. 暂不优先的结构替换

| 方向 | 参考论文 | 当前判断 |
|---|---|---|
| 整体替换 LSKNet 骨干 | LSKNet: A Foundation Lightweight Backbone for Remote Sensing | 有遥感长程上下文依据，但会同时改变预训练、容量和主干；应晚于校准/P2 单因素 |
| LSKA + Gold-YOLO 组合 | Dual-Strategy Improvement of YOLOv11n for Multi-Scale Object Detection in Remote Sensing Images | 论文一次组合多个变化，难解释增益来源；仅可拆成单模块对照 |
| 完整旋转等变网络 | Measuring the Impact of Rotation Equivariance on Aerial Object Detection | 当前是 HBB 任务，工程成本高；先做离散旋转一致性 |
| 不确定性 EDL 全头 | Disentangling Hardness from Noise | 原文是分类任务；可借鉴“困难与噪声分开”，不直接把分类结果当检测证据 |
| 端到端背景专家 | Decoupled Pipeline with Proposal Reranking and Score Fusion for Positive-Unlabeled Marine Species Detection | 与 N2 背景拒识相近；需要 N0-4 v2 人工背景标签，当前不作为 YOLO 首轮修改 |
| SAT/复杂合成增强 | YOLO-SAT | 当前 P07 已证明 SD1.5 融合不合格；真实结构保护的传统增强可留作独立数据消融 |

## 5. 推荐执行顺序

```text
Y0  冻结 M1 与官方 V1.6 评估
 ├─ Y1  cross-fit class-prior / FRACAL 校准（CPU，先做）
 ├─ Y2  P2 正式 160-epoch fixed-last 三折
 │    └─ 通过后：P2 单一采样/融合模块 → P2-Lite
 ├─ Y3  层次粗细类辅助损失（训练期）
 ├─ Y4  AFSS 日志诊断 → 相关性成立后训练消融
 └─ Y5  90° 离散旋转一致性
```

资源顺序建议：先执行 Y1，因为无需重训；GPU 首轮只并行 Y2 与 Y3，不同时把
P2、LSK、Gold-YOLO、旋转和新 loss 堆进同一模型。Y4/Y5 都是无推理成本方向，
在前两条结论明确后再进入正式三折。

## 6. 每个实验的统一输出

每个 YOLO 变体必须交付：

1. 相同 CV3 v2、初始权重、160 fixed epochs、三折 `last.pt` 和低阈值 OOF；
2. pooled Recall/FDR 与完整 4/20/1 macro；
3. fold0/1/2 和 source-group 分层；
4. `TP_new / TP_broken / FP_BG / FP_CLS / FP_DUP / FP_LOC`；
5. vehicle tiny/small、LQS/HM、TU-160/F-22；
6. cross-fit 工作点，不允许同 OOF 选点回评；
7. 只有模型收益成立后才进入完整 10K 时延，不把 4080S model-only 速度当官方时延。

## 7. 参考论文与开源入口

- [Fractal Calibration for Long-Tailed Object Detection](https://arxiv.org/abs/2410.11774)，代码：<https://github.com/kostas1515/FRACAL>
- [Edge-Constrained UAV Small-Object Detection with P2 Enhancement and Quantum-Inspired Lightweight Structure Search](https://arxiv.org/abs/2606.09081)，代码：<https://github.com/Ming23233/UAV-QIEA-Edge-Detection>
- [FRFDet: Efficient UAV Small Object Detection with Symmetric Sampling and Scalable Fusion](https://arxiv.org/abs/2607.04125)，代码：<https://github.com/HZAI-ZJNU/FRFDet>
- [Does YOLO Really Need to See Every Training Image in Every Epoch?](https://arxiv.org/abs/2603.17684)
- [Hierarchical Fine-Grained Aerial Object Detection](https://arxiv.org/abs/2606.16448)，项目页：<https://nnnnerd.github.io/PSP-Benchmark/>
- [Measuring the Impact of Rotation Equivariance on Aerial Object Detection](https://arxiv.org/abs/2507.09896)，代码：<https://github.com/Nu1sance/MessDet>
- [LSKNet: A Foundation Lightweight Backbone for Remote Sensing](https://arxiv.org/abs/2403.11735)，代码：<https://github.com/zcablii/LSKNet>
- [Disentangling Hardness from Noise: An Uncertainty-Driven Model-Agnostic Framework for Long-Tailed Remote Sensing Classification](https://arxiv.org/abs/2601.00278)
- [Decoupled Pipeline with Proposal Reranking and Score Fusion for Positive-Unlabeled Marine Species Detection](https://arxiv.org/abs/2607.18700)，代码：<https://github.com/dsgt-arc/fathomnetclef-2026>

## 8. 与现有文档的关系

- 当前可信数字与停止项：`PRE_INNOVATION_CLOSURE_20260810.md`；
- M1 正式血缘：`M1_CV3_OOF_FORMAL_RESULT_AND_RECOVERY_AUDIT_v2.md`；
- 车辆 near-miss：`A_MAINLINE1_NEARMISS_AUDIT_RESULT_20260810.md`；
- P2 历史机制结果：`A_MAINLINE1_P2_TRIPLE_VERIFICATION_20260810.md`；
- 统一实验合同：`docs/EXPERIMENT_PROTOCOL.md`；
- 大文件可用性：`ARTIFACT_RELEASE_REGISTER.csv`。

本报告不覆盖下一阶段 A—E 分工总纲，而是把其中“改 YOLO”部分收敛为可按
证据排序执行的实验队列。
