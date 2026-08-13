# R1-2 飞机类中心约束快速实验计划

日期：2026-08-14  
状态：`implemented_waiting_for_r1_1_completion`  
实验 ID：`R1-2-AIRCRAFT-CLASS-CENTER`

## 1. 问题与进入依据

R1-1 已经证明对象级飞机重排具有明确收益，但训练集准确率在 5 epoch 内接近
`99.9%`，说明问题不是训练样本拟合不足，而是来源变化下的类内差异和易混淆类间
边界。现有错误集中在 TU-160、SU-24、F-22、SU-34/E-8/F-16 等细类；普通
CE 和“教师正确样本同视图 KD”都可能继续强化容易样本。

本实验借鉴 **Exploration of Class Center for Fine-Grained Visual
Classification** 的核心思想：用类别中心同时收紧类内分布、推离最相似错误类。
它不是论文的完整复现；只采用可审计的特征中心约束，不采用其动态软标签表。

## 2. 唯一变量

基线与 R1-1 CE 完全相同：

- fold-specific P03 fixed-epoch-30 初始化；
- proposal-domain 飞机正样本；
- natural sampling、uniform random D4；
- 5 fixed epochs、相同优化器和学习率；
- 舰船、车辆、跨粗类路径完全旁路。

新增训练期辅助项：

\[
L=L_{CE}+\lambda\frac{e}{E}
\left[(1-\cos(f,c_y))+\max(0,m+s_{neg}-s_y)\right]
\]

其中类别中心以 P03 分类器权重初始化，随后用 detached batch feature 做 EMA；
`s_neg` 是当前样本与最相似非目标中心的余弦相似度。冻结参数为：

- `loss_weight=0.10`；
- `momentum=0.90`；
- `margin=0.90`（使预训练特征中真正靠近边界的样本产生 push 梯度）；
- `negative_weight=1.0`。

中心只在训练期存在，checkpoint 和部署模型仍是原 ConvNeXt-T 25 类状态字典。

## 3. 快筛协议

1. 三折各训练一次，不做 held-out early stopping；
2. 每折输出 identity 与 D4 概率；
3. cross-fit 只评 `ce_identity / class_center_identity / class_center_d4`；
4. 主比较为 `class_center_identity vs ce_identity`，严格隔离类中心损失的增量；
   D4 只作为正交附加读数；
5. 全部结果仍属于同一正式 OOF 上的迭代开发，不授予最终正式准入。

## 4. 验收与停止

技术门禁：

- 三折 checkpoint、bundle 和 runtime 完整；
- 32,062 个飞机候选严格 OOF 覆盖；
- ship/vehicle 指标差异为 0；
- feature-path logits 与模型原 forward 完全一致；
- 中心 loss/梯度有限，中心更新后单位归一。

科学判据：

- aircraft macro Recall 不低于参考超过 `0.002`；
- aircraft macro FDR 不恶化超过 `0.002`；
- paired net TP 非负；
- 若不优于既有 CE/D4，则停止类中心权重搜索，不做多参数网格；
- 若只改善总体而恶化 TU-160 压力折，不进入最终候选。

## 5. 与后续方法的关系

R1-2 回答“结构化类中心是否改善来源泛化”。D4 教师蒸馏回答“能否把多视图收益
压回单视图”。两者不能在第一轮组合；只有各自单因素通过后，才允许组合实验。

参考：<https://arxiv.org/abs/2407.04243>。
