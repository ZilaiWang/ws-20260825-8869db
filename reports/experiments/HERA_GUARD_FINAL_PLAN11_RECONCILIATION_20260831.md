# HERA-Guard Final：方案 11 与最新实验的合并决策（2026-08-31）

状态：`implemented_and_smoke_validated / full_teacher_complete / waiting_for_4x3090`

> 后续实现、真实 smoke、外部资产与 4 GPU 合同统一记录在
> `reports/experiments/HERA_GUARD_FINAL_PREFLIGHT_EXECUTION_20260831.md`。本文保留最初的
> 证据修正与路线选择，不再代表当前执行进度。

本文回答一个问题：在《改进方案11》提出外部遥感预训练、漏标安全训练和
D-FINE→Y5 蒸馏之后，结合 2026-08-31 已完成的正式实验，下一步唯一值得投入
算力的路线是什么。

## 1. 先修正方案 11 的证据基础

方案 11 的总体判断成立：当前系统接近的是“固定 Y5 proposal + 离线轻量重排”
的方法族上限，而不是检测任务本身的上限。但文档中引用的 D-FINE
`+22 TP / +6 FP` 不能继续作为正式依据。

复核发现，早期 product rerank 实现曾在构造 baseline 前覆盖 Y5 原始分数，导致
baseline 与 candidate 实际评估了同一产品分路线。该错误已经修正。当前可信证据为：

|证据合同|结果|用途|
|---|---:|---|
|严格 cross-fit 增量风险 0.15|+7 TP / +1 FP|无偏外层证据|
|严格 cross-fit 增量风险 0.20|+11 TP / +2 FP|无偏外层证据|
|每训练折 Recall 零损失 guard|+4 TP / -22 FP|对齐 v2 工作点的稳健性诊断|
|全 OOF 固定 vehicle 产品阈值 0.059|+9 TP / 0 FP，Recall +2.239pp|最终 full 超参数候选，不是无偏估计|
|全 OOF aircraft 产品阈值 0.05|+80 TP / -387 FP，Recall +0.448pp|需 Hard/Sentinel 验证的第二 Attack 候选|

因此可陈述的结论是：**D-FINE 提供了真实、跨三折可见的车辆互补信号，但可期待
收益约为 1.7--2.7pp 的量级，不是已经证明的 4--5.5pp。**

## 2. 已经被最新实验否定的实施方式

方案 11 中“先训练一个小 HAD proposal head”的设想，若仍采用冻结 metadata、
crop 或高维离线 FPN 特征，已经没有继续价值：

- 34-D metadata 学生在增量风险 0.20 下只有 fold2 的 +2 TP；
- metadata+crop 63-D 学生为 0 TP / 0 FP；
- 冻结 ConvNeXt tight/context 学生虽然能拟合教师分数，但跨域阈值漂移；
- support 学生的可见收益只来自 fold2，pooled +15 TP 同时 +16 FP；
- Q1/Q2 已证明冻结的 3,200-D FPN 证据不能稳定改进外层排序。

所以停止以下路线：更多 MLP、hidden dim、crop 尺寸、离线 FPN 拼接和教师分数回归。
若继续蒸馏，必须让监督进入 Y5 的最后 neck block / detection head，使底层车辆表示
本身改变；否则只是重复已失败方法。

## 3. 当前正在完成的零机会成本决策

服务器上的 D-FINE-M full 训练使用完整 4,481 图、20,933 框、固定 40 epoch、
seed 42、1024、batch 8，最终只能使用 epoch 40 的 last checkpoint。训练已经完成，
last SHA256 为 `a6d21b7a...fb49d`；冻结矩阵已按修复后的单一代码树完成：

1. Y5-S v2 identity；
2. 已存在的 Y5-L full checkpoint（因历史 NumPy RNG pickle 与当前冻结环境不兼容而跳过；
   该路线已有负向证据且不参与 D-FINE 决策）；
3. Y5-S + D-FINE vehicle-only，固定 0.059；
4. Y5-S + D-FINE aircraft+vehicle，固定 0.05/0.059；
5. 同时在 Hard10K 与 source-disjoint Sentinel 上评估，不重新调阈值。

结果为两条 D-FINE 路线均不准入：vehicle-only 在 Hard/Sentinel 的 vehicle Recall 分别
下降 9.239pp/8.025pp，时延由约 4.3s 增至约 9.4s；aircraft+vehicle 同样失败。短期 Attack
分支关闭，D-FINE 只作为训练期 HAD 教师，不进入 Docker。

full 训练内部 AP 只用于确认训练健康，因为 full 数据同时作为训练/内部评估输入，
不能用于方法选择。

## 4. 下一阶段只保留三条有新信息的主线

### 4.1 主线 A：vehicle-first 严格 OOF 漏标审计

目标不是批量伪标签，而是消除官方 fine-tune 中“真实车辆被当成背景”的错误梯度。

候选只允许来自对应图像的 OOF 模型：

- Y5 OOF vehicle；
- D-FINE OOF vehicle；
- 与全部已有 GT 的 IoU < 0.05；
- 两模型 vehicle 几何一致，IoU >= 0.35；
- 排除 tile seam、重复框和明显截断框；
- 按产品证据、两模型原始分和稳定性排序。

代理逐图审核后已经形成三类版本化资产：

```text
data/annotation_patches/
  confirmed_missing_v1.json
  ignored_ambiguous_v1.json
  rejected_candidates_v1.json
  audit_manifest.json
```

关键修正：**未人工审核的疑似漏标不得自动设为 ignore。** 模型幻觉若被自动 ignore，
会取消最需要的背景负梯度并恶化 FDR。只有人工确认“确有目标但 fine label 无法可靠
确定”的区域才可进入 `ignored_ambiguous_v1`。vehicle 只有一个正式细类，确认后可直接
补 class 24；ship/aircraft 无法确认型号时只能 ignore，不能硬造 fine label。

先做 vehicle，只有审核发现率和规模足够再扩 ship。该审计可与外部 Stage-A 预训练
并行，但 annotation patch 必须在官方 25 类 fine-tune 前冻结。

该流水线已于 2026-08-31 落地并在三折 OOF 全量运行：

- 4,481 张图全部唯一覆盖；
- 固定 vehicle 产品阈值 0.059、支持 IoU 0.35、与全部 GT IoU < 0.05；
- 图内 IoU 0.50 贪心去重后得到 67 个候选，fold0/1/2 分别为 29/13/25；
- 产品证据中位数 0.156，支持 IoU 中位数 0.773；
- 10/67 候选紧贴图像边界，逐张审核时单独判断截断/边界结构；
- 67 张 full+zoom 卡片和 12 张 contact sheet 已全部检查；
- 最终为 32 confirmed、20 ambiguous ignore、15 rejected，证明“两模型一致”不能自动
  等同于漏标。

资产：

- `scripts/build_missing_label_consensus_review.py`；
- `scripts/render_missing_label_consensus_review.py`；
- `scripts/compile_missing_label_consensus_review.py`；
- `outputs/HERA-GUARD-FINAL-MISSING-LABEL-VEHICLE-REVIEW-V1/summary.json`；
- `outputs/HERA-GUARD-FINAL-MISSING-LABEL-VEHICLE-REVIEW-V1/manual_missing_label_review.csv`。

编译器已经在 67 行完整后冻结 annotation/ignore 准入和逐文件 SHA；partial-label-safe
物化器及 paired control 已通过测试与全量审计。

### 4.2 主线 B：外部遥感 coarse/objectness 预训练

这是当前最可能改变 ship/vehicle 表示上限的方向，值得申请 4 张 3090。当前已完成
DOTA/DIOR 官方下载、转换、切片、role sampler、粗类训练和 fresh 25 类 head 迁移实现；
DOTA 官方 train part1 真实数据 smoke、代理视觉审核与两阶段训练 smoke 均通过。完整
train+val 因本地磁盘门禁留待 4 GPU 服务器下载，不再属于代码或方案缺口。

开训前必须补齐：

1. dataset/license manifest 与文件 SHA；
2. 每个数据集显式的原类→四粗类映射；
3. OBB/polygon→HBB 的可审计转换；
4. scale-preserving 大图切片，记录截断、保留比例和来源；
5. dataset-aware sampler，避免百万级单一数据源淹没其余域；
6. 外部四类 head→官方 25 类 head 的确定性替换与 backbone/neck 迁移；
7. 外部数据与 Normal/Hard/Sentinel 严格互斥的指纹检查。

第一轮只做两个不同能力的初始化，不能一次混成不可解释的大杂烩：

- `EXT-G`：ship、港口/水域、通用遥感 objectness 与 other-object 背景；
- `EXT-V`：tiny vehicle、道路/停车区/城市结构化背景。

AI-TOD 等带非商业或 share-alike 条款的数据必须先完成比赛使用许可审计；许可不明确时
不进入正式权重。外部数据只提供粗类/objectness 表示，绝不映射成比赛军机/舰船型号。

### 4.3 主线 C：Y5 内部的车辆异构蒸馏

蒸馏继续，但实施位置必须前移：

```text
D-FINE 只在训练时生成严格 OOF teacher target
        ↓
Y5 最后一个 neck block + vehicle quality channel 共同学习
        ↓
推理时只运行单个 Y5-S
```

首轮结构只增加一个 vehicle-specific bounded residual/quality channel：

- 输入为 Q0 metadata + 低维 P3/P4 ROI 投影；
- 投影层和最后 neck block 低学习率更新；
- residual 零初始化，初始输出严格等价 incumbent；
- teacher target 使用同细类 IoU、校准 support 和官方 match target；
- rank pair 只在相近原始分、同 source 风格下的 protected TP 与 active FP 间构造；
- ship/aircraft 默认旁路。

先跑 fold0 与 fold2：fold0 是历史困难域，fold2 是早期 specialist 反向风险域。两折任一
Recall 下降即停止，不通过后不补 epoch、不扫十组蒸馏权重。

## 5. 四张 3090 的正确分配

在外部数据资产与 annotation patch 就绪后，第一轮只允许四个配对实验：

|GPU|实验|配对基线|
|---|---|---|
|0|EXT-G → official fold0 快筛|相同 schedule 的官方初始化 fold0|
|1|EXT-V → official fold0 快筛|相同 schedule 的官方初始化 fold0|
|2|in-model HAD fold0|无 HAD、相同初始化与 schedule 的 fold0|
|3|in-model HAD fold2|无 HAD、相同初始化与 schedule 的 fold2|

准入门：

- Normal candidate floor 下降 <=0.3pp；
- Normal 25 类 macro 下降 <=0.3pp；
- 任一粗类 Recall 下降 <=0.5pp；
- EXT 在 Hard 的 ship 或 vehicle fixed-risk 至少 +0.5pp，Sentinel 同方向；
- HAD 两折 vehicle Recall 均不下降，两折 pooled 增益 >=2pp，vehicle FDR 恶化 <=1pp；
- 不在 Hard/Sentinel 上选择阈值、checkpoint 或融合权重。

通过后才扩 CV3。最终只保留一个 external initialization 和一个 HAD 结构，再跑唯一
full-data 配方。任何失败路线都写入总账，不产生“稍微改一下再试”的分支爆炸。

## 6. Safe、Attack 与 Final 的关系

### Safe

保持官方 `trial-v2.0`：Y5-S full、identity、统一 0.15、safe fusion。它仍是唯一正式
incumbent，官方 86.2274。

### 近期 Attack

仅由当前 full D-FINE 冻结矩阵决定。若 vehicle-only 或 aircraft+vehicle 在 Hard 与
Sentinel 同时满足 Recall guard，且双模型真实 3090 时延的质量收益足以覆盖第七项排名，
才允许构建一次官方候选。否则不提交。

### Final

```text
external-pretrained Y5-S
+ reviewed partial-label-safe official fine-tune
+ in-model vehicle HAD
+ 已通过的 Q0 OMQ
+ safe fusion
```

Final 的关键特征是单视图、单主检测器；D-FINE 和外部 detector 仅为训练期教师。

## 7. 结论与唯一执行顺序

方案 11 给出的“新遥感域 + 可靠标注 + 异构架构知识”方向是目前最合理的突破口，
但应按以下顺序执行：

1. 完成正在运行的 full D-FINE 和两套冻结基准，决定近期 Attack；
2. 立即建立 vehicle-first OOF 漏标人工审核包；
3. 并行补齐外部数据、许可和统一预训练流水线；
4. 资产门禁通过后申请 4 张 3090，运行 EXT-G、EXT-V、HAD-fold0、HAD-fold2；
5. 只把通过门禁的一个 external init 与一个 HAD 结构扩成 CV3；
6. 唯一配方 full 训练，最后执行 Normal/Hard/Sentinel、细类 macro、真实时延和 Docker 验收。

当前方法可以称为完整、严谨的比赛研究框架，但还不能把“外部预训练”和“异构蒸馏”
写成已经实现的算法贡献。只有主线 B/C 通过并进入单模型后，`HERA-Guard Final` 才是
具有实证支撑的最终创新算法，而不是一份愿景。
