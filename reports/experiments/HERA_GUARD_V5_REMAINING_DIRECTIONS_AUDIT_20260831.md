# HERA-Guard V5：剩余方向审计与 crop 主线决策（2026-08-31）

## 1. 决策摘要

正式阶段当前唯一 incumbent 仍为 `trial-v2.0`：综合分 86.2274，舰船
Recall/FDR 为 0.942287/0.126937，飞机为 0.999246/0.024300，车辆为
0.946309/0.237838，平均时延 2.704833 秒。飞机已接近饱和；主要分数缺口是车辆高
FDR，其次是舰船 Recall/FDR。任何只继续提高飞机本地 OOF 的方案，都不太可能把正式
综合分从 86 推到 93。

本轮对《改进方案 1--10》和已有结果重新做了“实现状态—独立增益—部署成本”审计，
并补做三组严格 Normal-CV3 消融。结论是：

1. **当前唯一仍有明确独立正增益的证据是固定候选框的 RGB crop 语义证据。**
   crop-only 质量头把 FDR15 Recall 从 0.916830 提高到 0.923613（+0.678pp）。
2. OTO 支持、D4 支持、Y3 层次模型支持和 Y4-AFSS 支持均不能单独达到准入线；把
   D4/OTO 加回 crop 还会降低 crop-only 结果。
3. 简单按粗类把 crop 质量分数与原 detector 分数拼接，可以保留对应粗类的小幅增益，
   但证实完整增益主要来自飞机；舰船/车辆合计只增加约 0.048pp。下一步不能再靠规则
   路由或融合权重扫描追分。
4. 项目曾记录“F1 三分类开放拒绝尚未探索”，而现有 `f1_fg_rejector_manifest.py`
   实际仍把结构化背景和普通背景合并成二分类标签。这不是已经完成的负向实验，而是
   一条真实未执行的路线。
5. 因此 V5 的第一主线冻结为：**Y5 proposal-domain、显式三类开放拒绝、tight +
   context 固定框复核，并把输出作为质量证据，不创建新框、不直接硬改类。**

## 2. 本轮新增审计

### 2.1 Y5 one-to-many / one-to-one 血缘核验

当前 full Y5 checkpoint 同时包含 one-to-many（OTM）和 one-to-one（OTO）分支，标准
Ultralytics 端到端推理默认返回 OTO；历史正式 OOF 适配器则曾明确使用 OTM + NMS。
同一 Normal-CV3 上按当前官方匹配器重放后：

| 分支 | FDR15 Recall | 实际 FDR | macro Recall | ship R/FDR | aircraft R/FDR | vehicle R/FDR |
|---|---:|---:|---:|---:|---:|---:|
| OTM | 0.916830 | 0.151022 | 0.860783 | 0.760626/0.147870 | 0.951762/0.151364 | 0.407960/0.154639 |
| OTO | 0.916687 | 0.151117 | 0.860758 | 0.759508/0.148767 | 0.951706/0.151329 | 0.410448/0.158163 |

二者几乎等价。正式 Docker 仍应显式冻结分支语义，避免 Ultralytics 版本漂移，但“切换
分支”不是提升分数的方向。

原始前沿：

- `outputs/HERA-GUARD-V5-REMAINING-DIRECTIONS-20260831/branch/otm-frontier.json`
- `outputs/HERA-GUARD-V5-REMAINING-DIRECTIONS-20260831/branch/oto-frontier.json`

### 2.2 Q0 证据归因消融

统一使用相同标签、相同三折、相同质量头训练合同和 0.005 阈值网格，仅改变输入证据：

| 变体 | 维度 | FDR15 Recall | 实际 FDR | macro Recall | 相对 detector |
|---|---:|---:|---:|---:|---:|
| detector/base | 34 | 0.916830 | 0.151022 | 0.860783 | 0 |
| base + OTO | 35 | 0.916591 | 0.152817 | 0.856592 | -0.024pp |
| base + D4 | 35 | 0.916209 | 0.153507 | 0.856130 | -0.062pp |
| base + D4 + OTO | 36 | 0.916639 | 0.153446 | 0.856446 | -0.019pp |
| **base + crop** | **63** | **0.923613** | **0.152686** | **0.868859** | **+0.678pp** |
| base + crop + OTO | 64 | 0.922658 | 0.153266 | 0.868043 | +0.583pp |
| base + crop + D4 | 64 | 0.922849 | 0.152608 | 0.866935 | +0.602pp |

crop-only 的三粗类 Recall 为舰船 0.763609、飞机 0.959157、车辆 0.412935；相对
detector 分别约 +0.298pp、+0.740pp、+0.498pp。它也是唯一三粗类均未下降的变体。

这组结果修正了“完整 Q0 65D 整体有效”的宽泛说法：真正有效的是 crop
top-1 probability、margin、entropy、detector/crop agreement 和 crop class one-hot；
D4/OTO 不是独立增益来源。

完整前沿位于：
`outputs/HERA-GUARD-V5-REMAINING-DIRECTIONS-20260831/factorial/`。

### 2.3 旧专家只作 support 的复验

Y3/Y4 没有以候选并集方式重启，而是只对 incumbent 候选提供同细类 IoU/score 支持：

| 支持证据 | FDR15 Recall | 实际 FDR | macro Recall | 相对 detector |
|---|---:|---:|---:|---:|
| Y3 | 0.916113 | 0.153632 | 0.856094 | -0.072pp |
| Y4 | 0.917976 | 0.151799 | 0.858407 | +0.115pp |
| Y3 + Y4 | 0.918741 | 0.154154 | 0.858984 | +0.191pp |

三者均低于 +0.5pp 准入线，也明显低于 crop-only；旧专家支持路线停止。完整前沿位于
`outputs/HERA-GUARD-V5-REMAINING-DIRECTIONS-20260831/expert_support/`。

### 2.4 粗类分数路由诊断

为了检验“只把 crop 用于舰船/车辆即可保护飞机”，补做四个有限、非扫描路由。结果
显示 vehicle-only 把车辆 Recall 从 0.407960 提高到 0.412935，但总体 Recall 仅从
0.916830 提高到 0.916925；ship-only 为 0.917212，ship+vehicle 为 0.917308；只有
aircraft 路由已达到 0.923136。完整 crop 的 0.923613 因而主要由飞机贡献，舰船和车辆
在现有闭集 crop 表征下只有很小的正式相关增量。

这些结果只是否决简单拼接，不是否决“按粗类训练独立开放集复核头”。完整前沿位于
`outputs/HERA-GUARD-V5-REMAINING-DIRECTIONS-20260831/class_routes/`。

## 3. 为什么当前分数仍然不够

### 3.1 本地 pooled 增益的主要贡献来自飞机

crop-only 的 +0.678pp 中，飞机贡献最大；但正式 `trial-v2.0` 飞机 Recall 已是
0.999246，几乎没有可转化空间。因此不能把 Normal-CV3 的 0.9236 直接解释为正式分数
会大幅上涨。真正需要的是把 crop 对车辆和舰船的 TP/FP 可分性单独学出来。

### 3.2 旧 crop 模型是闭集分类器，不会回答“这是不是目标”

旧 P03 ConvNeXt-T 在干净 GT crop 上细类能力很强，但面对真实 proposal 背景时必须在
25 个前景类中选一个。既有审计中 44,910 个背景候选的 crop 幻觉高度集中到
MS/QHS/FSC/SU-24。这解释了为什么 crop 可以作为弱证据改善质量头，却不适合直接改类
或 hard drop。

### 3.3 F1 的“三分类”此前没有真正实现

`reports/experiments/HERA_FIELD_F1_20260821/F1_RESULT.md` 已明确写出“未探索：真实目标 /
结构化背景 / 普通背景”。当前 manifest 构建脚本虽然识别了结构化背景，最终写盘前却
移除了该标记；训练器仍是二分类 BCE。因此二分类 F1 的边际结果不能证伪真正的三分类
开放拒绝。

## 4. 后续有限实验序列

### V5-A：crop-only 部署等价闭环（最高优先级）

目的：先证明 0.923613 的增益能由 Docker 可生成的证据严格复现。

1. 冻结 Y5 候选、OTM/OTO 语义、canonical box 和 NMS；
2. 仅在全局融合后的唯一对象上裁一次图，不在每个 tile 重复运行 crop；
3. 使用现有 P03 ConvNeXt-T，输入固定 `tight 1.0x / 224`；
4. 只保留本轮验证过的 63D `base + crop` schema；
5. 三折训练质量头，再训练 full crop classifier 和 full 质量头；
6. 在 Normal、Hard10K、source-disjoint sentinel 和 RTX3090 容器上成对验收。

晋级条件：Normal Recall 增益至少 +0.5pp；Hard 至少 +0.5pp；sentinel 同方向；舰船和
车辆任一 Recall 不下降超过 0.5pp；新增时延目标不超过 0.15 秒/10K 图。

### V5-B：真实三类 proposal-domain 开放拒绝

只有 V5-A 部署等价成立后进入。训练样本严格来自 held-out Y5 proposals：

- `foreground`：官方一目标一胜者 TP；
- `structured_background`：跑道线、岸线、码头设施、车辆状纹理等高混淆 FP；
- `ordinary_background`：其余 FP；
- 输入固定为 `tight 1.0x + context 1.25x`，不扫描尺度；
- 输出三类概率和共享 embedding；不直接 DROP，不直接 RELABEL；
- 三类概率作为 crop-only 质量头的新证据，仍由官方指标外层 CV3 选工作点。

该实验直接针对正式车辆 FDR=0.237838 和舰船 FDR=0.126937，是最可能产生正式收益的
一条线。二分类 F1 只作初始化对照。

### V5-C：选择性高分辨率复核

若 V5-B 显示像素证据可观测，但 224 crop 对小车辆不足，则只增加一个预注册变体：
`336 tight + 224 context`。强 teacher（DINOv2-B/遥感表征）只允许作离线可观测性探针；
若没有超过 ConvNeXt-T，不进入 Docker。若超过，再蒸馏到轻量 crop head。

选择性触发必须在全局对象融合后，仅覆盖 ship/vehicle 的 near-threshold 或高风险对象；
不得全图二次检测，不得由 verifier 创建新框。

### V5-D：crop-only 的域鲁棒训练

此前 GroupDRO 因高维 FPN Q2 失败而没有运行；现在 crop-only ERM 已独立通过，可以重新
开放有限的两个比较：group-balanced ERM 与 GroupDRO。两者只能使用相同 63D schema，
不改 hidden dim、epoch、residual limit。若 sentinel 没有超过 crop-only ERM，立即停止。

### V5-E：外部 coarse/objectness 预训练

这是中期候选形成路线，不应阻塞 V5-A/B。外部数据只学
`ship/aircraft/vehicle/other_remote_object`，随后重建官方 25 类头；先 fold0 40 epoch
快筛，再决定是否三折/full。它解决 no-candidate，而不是当前已观察到的 proposal FP。

### V5-F：车辆稀疏 zoom/seed recovery

只有 vehicle FN 审计证明大量 GT 在低阈值候选池中完全无候选时才进入。固定一种中心
种子和一种局部尺度，仅在疑似区域二次检测；若主要问题仍是已有候选排序，则不运行。

## 5. 明确停止项

不再投入到以下方向：阈值/NMS 网格、全量 rot90 TTA、OTO/OTM 分支切换、D4 特征继续
拼接、Y3/Y4 无条件候选并集或 support、M3、Y5-L、D-FINE/DEIM vehicle specialist、
高维 FPN-Q1/Q2、SAHI/P2、困难背景强微调、CleanDIFT/DINO 特征直接拼接、普通 25 类
crop 硬改标签。它们已有负向、边际或域不稳证据。

## 6. 正式提交策略含义

本轮新增结果仍是 Normal-CV3 归因，不足以替换 `trial-v2.0`。在 V5-A 完成 full
checkpoint、Hard/sentinel 复验和 Docker 3090 验收之前，不消耗正式提交机会。

若目标是 93 分，下一候选不能只是“总体 Recall +0.7pp”；它至少要在保持当前飞机
近饱和和 2.7 秒级时延的同时，显著降低 vehicle FDR，并让 ship Recall/FDR 同向改善。
V5-A 是部署可行性门，V5-B 才是正式分数主攻。

## 9. V5-B 实际执行结果（2026-08-31）

V5-B 已按真实三分类合同完成。全量 65,301 个 held-out proposals 中，训练标签为
20,391 个前景、29,310 个结构化背景、11,711 个普通背景；3,889 个“可匹配但不是唯一
获胜候选”标为 ignore，只生成 OOF 分数，不作为背景训练。tight/context 两路各 768D，
三折提取无非有限值，最终在 `base_crop` 63D 上只增加三类概率及前景 margin 四维。

三折三分类 balanced accuracy 分别为 0.5945、0.6422、0.6326。这个中间指标只证明
存在可学习信号，不作为科学准入结论。FDR 0.150 外层 official-matching 配对结果如下：

| 方法 | Recall | FDR | macro Recall | ship R | aircraft R | vehicle R |
|---|---:|---:|---:|---:|---:|---:|
| crop-only | 0.923613 | 0.152686 | 0.868859 | 0.763609 | 0.959157 | 0.412935 |
| crop + true 3-way open set | 0.923996 | 0.153449 | 0.868511 | 0.764728 | 0.958989 | 0.432836 |
| delta | **+0.000382** | +0.000762 | -0.000348 | +0.001119 | -0.000168 | **+0.019900** |

车辆 Recall 的 +1.99pp 说明双 crop 三分类证据对车辆不是完全无效；但车辆 FDR 同时从
0.153061 升到 0.171429，总体 Recall 只增加 0.038pp，未达到预注册 +0.5pp 门槛。
因此 `admitted=false`，不训练 full-data 部署头、不扫描融合权重，也不据此替换当前正式
镜像。下一步进入 V5-D 的 crop-only D1/D2/D3 鲁棒训练归因。

可核查产物：

- `outputs/HERA-GUARD-V5-OPEN-SET-V1/manifest_summary.json`；
- `outputs/HERA-GUARD-V5-OPEN-SET-V1/extraction_summary.json`；
- `outputs/HERA-GUARD-V5-OPEN-SET-V1/open_set_head/summary.json`；
- `outputs/HERA-GUARD-V5-OPEN-SET-V1/quality/eval/frontier.json`；
- `outputs/HERA-GUARD-V5-OPEN-SET-V1/decision.json`。

## 10. V5-D crop-only 鲁棒训练结果（2026-08-31）

在完全相同的 `base_crop` cache、质量头容量、epoch 和官方外层 CV3 下，补齐 D1/D2/D3：

| 训练方式 | Recall | FDR | macro Recall | ship R | aircraft R | vehicle R |
|---|---:|---:|---:|---:|---:|---:|
| D0 uniform ERM | **0.923613** | 0.152686 | **0.868859** | **0.763609** | **0.959157** | **0.412935** |
| D1 group-balanced ERM | 0.922085 | 0.151896 | 0.866230 | 0.761745 | 0.957701 | 0.410448 |
| D2 uniform GroupDRO | 0.920317 | 0.152330 | 0.865963 | 0.761372 | 0.955852 | 0.402985 |
| D3 group-balanced GroupDRO | 0.916018 | 0.156401 | 0.859199 | 0.754288 | 0.952995 | 0.353234 |

三种鲁棒化均低于 D0，且强度越高车辆下降越明显。现有 group_id 适合数据隔离与 CV，
但不是 proposal 真假排序的有效优化单位；V5-D 因此关闭，不进入 full-data 训练。完整前沿位于
`outputs/HERA-GUARD-V5-CROP-ROBUSTNESS-V1/`。

## 11. V5-C 车辆选择性高分辨率结果（2026-08-31）

为区分“车辆专属路由”与“分辨率”效应，严格执行 224 控制臂与 336 单因素臂；只有
category 24 的 5,131 个 proposals 增加复核，其他类别的增量特征全为零。

| 条件 | Recall | FDR | macro Recall | vehicle Recall | vehicle FDR |
|---|---:|---:|---:|---:|---:|
| crop-only | **0.923613** | **0.152686** | **0.868859** | **0.412935** | **0.153061** |
| vehicle-only 224 | 0.922228 | 0.153401 | 0.866389 | 0.355721 | 0.205556 |
| vehicle-only 336 | 0.922228 | 0.153772 | 0.866383 | 0.353234 | 0.244681 |

车辆三分类标签跨 fold 极不稳定，224 的 foreground recall 分别为 1.000、0.000、0.917；
336 没有修复这一数据问题，反而继续提高车辆 FDR。由此可排除“只需提升 crop 分辨率”
的解释，V5-C 关闭。可核查产物位于
`outputs/HERA-GUARD-V5-SELECTIVE-VEHICLE-RESOLUTION-V1/`。

## 12. 旧 F1 × 新 crop-only 漏测组合复验（2026-08-31）

旧 F1 端到端二分类 foreground gate 过去只接入旧 14D OER，未与本轮独立通过的
crop-only 63D 质量头组合。复验严格复用原三折 OOF logit，只增加 1D，不重训 F1、
不改变阈值网格：

| 条件 | Recall | FDR | macro Recall | ship R | aircraft R | vehicle R |
|---|---:|---:|---:|---:|---:|---:|
| crop-only | **0.923613** | **0.152686** | 0.868859 | **0.763609** | 0.959157 | **0.412935** |
| crop + F1 logit | 0.923518 | 0.153294 | **0.868970** | 0.760626 | **0.959550** | 0.410448 |

总体 Recall -0.0096pp、FDR +0.0607pp，仍为持平偏负。它排除了“旧 F1 只是接错了
质量头”这一解释；二分类端到端前景证据与 crop-only 也基本冗余。产物位于
`outputs/HERA-GUARD-V5-F1-CROP-REVISIT-V1/`。

## 13. 本轮收敛判断

本轮四条增量路线都用同一 Normal-CV3 official-matching 外层协议闭环：真实三分类、
域鲁棒、车辆选择性高分辨率、旧 F1 新组合。没有一条达到 +0.5pp；继续在相同
65,301 proposals 上追加轻量证据或采样技巧的期望收益已经很低。

下一阶段不再扩展 OMQ 特征维度，主问题转为：

1. 形成比当前 Y5 proposal 更可分的检测表示，而不是在固定候选上反复重排；
2. 为 ship/vehicle 引入来源合规的外部 coarse objectness/真实背景预训练；
3. 单独优化正式镜像的候选生成与时延 Pareto，不能用双视图换取很小的检测收益；
4. 在没有新候选形成证据前，`trial-v2.0` 仍是唯一正式 incumbent。
