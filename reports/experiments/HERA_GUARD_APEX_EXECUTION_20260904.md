# HERA-Guard APEX（方案16）执行记录

## 1. 目标与边界

本轮不改 P40 检测器，也不改已经准入的 Aircraft view-consistency D4 模块。目标是验证 Ship/Vehicle 的 proposal 边界学习能否在保持原框、原分数、原细类标签的前提下，从 P40 低分尾部高精度补回漏检。

本报告把快速冻结骨干筛选与正式模型明确分开。A0/A1 即使在代理集上通过，也只说明值得进入深度微调和部署时延复测，不等价于正式平台 84 分。

## 2. 配对实验

- A0：只使用真实 P40 proposal 的 official-match 正负样本。
- A1：A0 + LMP 抖动难负例 + 细类/尺度视觉原型边界特征。
- 视觉骨干：各外层 fold 对应的 P03 ConvNeXt-Tiny，冻结，仅提取 224 tight crop embedding。
- 校准：训练 fold 内按来源组稳定哈希隔离；Ship rescue precision 不低于 95%，Vehicle 不低于 90%；无法达到则 fail closed。
- 执行顺序：Normal OOF → Hard → 仅在 Hard 正向时进入 Sentinel-B。
- 比较基线：与候选采用完全相同的 same-fine NMS-only control，隔离 NMS 本身的影响。

## 3. 数据构建

逐图以官方分数降序、同细类、一对一匹配生成：`canonical_tp`、`fp_duplicate`、`fp_cls`、`fp_bg`、`ignore_geometry`。仅前三类可靠负例/正例进入训练，几何边界样本丢弃；另加入 GT 正例和仅用于 proposal 分类器的确定性 jitter 难负例，绝不回写检测训练标签。

待服务器产物：

- `manifest/manifest_summary.json`
- `train-normal/train_normal_summary.json`
- `hard/comparison.json`
- `sentinel/comparison.json`（仅 Hard 正向时存在）
- `audit/input_and_code_sha256.txt`
- `audit/result_sha256.txt`

## 4. 已完成结果

### 4.1 真实 proposal 与增强样本规模

OOF 共构建 41,059 个 Ship/Vehicle crop，覆盖 255 个来源组。低阈值 P40 尾部仍有
Ship 789 个、Vehicle 249 个 canonical TP，说明“有候选但排序不足”的容量判断成立。

|粗类|canonical TP|GT positive|active FP/duplicate/FP_CLS|LMP jitter|
|---|---:|---:|---:|---:|
|Ship|2,522|2,682|12,120|8,022|
|Vehicle|382|402|13,723|1,206|

Vehicle 的正负比例远低于 Ship；其部署问题不是简单缺少候选，而是从约 1.6 万个低分
候选中高精度识别不足 250 个潜在真框。

### 4.2 A0/A1 冻结骨干筛选

公平对照必须是与候选执行完全相同 same-fine NMS 的 `nms_control`。NMS 本身在
Normal 提升 `+0.2361`，在 Hard 却降低 `-0.1664`，因此不得把这部分变化归给 APEX。

|条件|方法|相对 NMS control 的质量分|Ship rescue|Vehicle rescue|结论|
|---|---|---:|---:|---:|---|
|Normal|A0|−0.0547|25|1|不准入|
|Normal|A1 原实现|−0.0923|4|2|不准入|
|Hard|A0|−0.1405|13|1|不准入|
|Hard|A1 原实现|−0.0956|6|1|不准入|
|Normal|A1 source-safe 修正|−0.0404|7|1|不准入|

原 A1 把负例原型混在同一桶中；source-safe 修正后改为 positive、jitter-negative、
active-FP 分桶，并对每个当前样本精确排除同来源原型。损失有所收窄，但方向仍为负，
因此不能进入 Hard/Sentinel，也不能靠放松质量阈值制造表面提升。

失败机制很明确：Ship 虽偶尔救回 TP，但新增 FP 足以抵消质量分；Vehicle 只放入
1–2 个框就令 FDR 上升，说明线性冻结头没有学到可迁移的高精度边界。

### 4.3 正在执行的剩余判别实验

- A2：已生成 4,066 个 Ship、573 个 Vehicle 有效 object-scale positive；拒绝了
  1,298/231 个不满足可见率约束的 crop。当前做 A0/A2 配对 Normal，只有 A2 正向才进 Hard。
- A1M：复用 A1 三份不可变特征缓存，实现 64 维 MLP、来源稳健加权 BCE、同来源且
  原分相近的 TP>FP rank loss。它用于区分“LMP/原型机制无效”和“线性头表达不足”；
  Normal 不正向即停止。
- A3 尚未开始。Background-100MP 可作为 active-FP/负原型池，但没有通过 mask、语义背景
  与 missing-label guard 的素材不能冒充完整 Domain-RAG-lite。

当前没有 APEX Ship/Vehicle 模块通过门禁；正式部署主线仍是 P40 加已经独立验证通过的
Aircraft D4 高置信重标模块。

### 4.4 A1M：非线性边界头与粗类拆分

A1M 把冻结 embedding 的线性逻辑回归替换为 64 维 MLP，使用来源稳健加权 BCE、
同来源且原始分数相近的 `TP > active FP/jitter` 排序损失。合并 Ship+Vehicle 时：

|条件|相对 NMS control|Ship Recall/FDR 变化|Vehicle Recall/FDR 变化|
|---|---:|---:|---:|
|Normal|+1.6778|+5.60pp / −3.66pp|+2.24pp / +1.48pp|
|Hard|+0.0295|+2.60pp / +0.25pp|+0.54pp / +2.02pp|
|Sentinel-B|−0.1858|+3.08pp / +0.58pp|0 / +3.08pp|

Vehicle 在确认集上主要增加 FP，明确停用。冻结同一权重只启用 Ship 后，质量分变为
Normal `+1.6104`、Hard `+0.1908`、Sentinel-B `+0.1439`，三套首次同向。
但逐框账本表明 Ship rescue 精度约为 52% / 67% / 44%，远低于预注册的 95%。
正增益来自宏平均对稀有 Ship 细类的较高价值，并不等于新增框足够可靠，故仍不准入。

### 4.5 D4 Ship A1M

D4 八视图平均 embedding 在 Normal 仅 `+0.0470`，29 个新增框中约 9 TP/7 FP
（其余被 NMS），没有修复纯度；Hard 为 `−0.0732`。按顺序门禁停止，Sentinel-B
不读取。结论：飞机分类器上的 D4 正收益不能直接外推到 Ship validity rescue。

### 4.6 后续已排队

- A3-lite：382 张、100.139MP 的 Background-100MP 已逐张视觉审核，确认剩余可见或
  模糊目标为 0。从 P40 背景响应构建 1,043 个来源限额 hard negatives；只作为
  classifier active-FP 负例，不冒充缺少 mask 的 Domain-RAG 合成。
- A1M robust：以 5-way source inner-OOF 覆盖全部训练来源选择阈值，Ship 至少要求
  10 个校准 TP、3 个正来源组，再在全部非 outer-fold 数据上重拟合。该实验用于修复
  单一 20% 校准子集仅凭 `2 TP / 0 FP` 选阈导致的乐观偏差。

### 4.7 A2 与 A3-lite 终态

|模块|Normal 相对 NMS control|关键现象|结论|
|---|---:|---|---|
|A2 object-scale positive|−0.0032|Ship 救回 23，Vehicle 0；Ship Recall +0.16pp、FDR +0.07pp|无独立收益，停止|
|A0M（A3 公平对照）|+0.0110|只在 fold2 Vehicle 找到阈值|近似无变化|
|A3M reviewed background|−0.0279|加入 1,043 个已审纯背景负例后仅救回 6 个 Vehicle|负向消融，停止|

A3M 相对同头 A0M 的直接差为 `−0.0388`。背景清单覆盖 Ship
754 个、Vehicle 289 个裁块，全部继承 Background-100MP 的逐图审核结果，因而本次失败
不能归因于明显漏标；它说明“增加真实纯背景负例”仍不能解决极稀疏低分 TP 的校准问题。

### 4.8 source inner-OOF 稳健校准

此前全 Ship A1M 的阈值由单一来源子集中的 `2 TP / 0 FP` 决定。改用五路来源
inner-OOF、要求至少 10 个 TP 且覆盖至少 3 个正来源组后，三个 outer fold 均无法在
95% 精度下选择阈值，全部 fail closed，Normal 增益严格回到 `0.0000`。

这项结果否决“全 Ship A1M 已可部署”的解释，但没有否决其 embedding 中存在局部可用
信号；后续只能收窄动作范围，不能放松精度门槛。

### 4.9 Ship class-safe 0/1 动作白名单

全 Ship 误救审计显示，新增 FP 全部来自 fine class 2；fine class 0/1 在 Normal 是
`4 TP / 0 FP`，且在不同外层代理上分别仍能补回 TP。由此冻结动作白名单 `{0, 1}`：
模型仍使用完整 Ship 训练样本学习 objectness 边界，但只允许原标签为 0/1 的低分框被加入。

|条件|质量分变化|有效新增 TP/FP|Ship macro Recall|Ship macro FDR|95% 精度门|
|---|---:|---:|---:|---:|---|
|Normal OOF|**+1.7194**|4 / 0|36.85% → **42.09%**|14.62% → **10.45%**|通过|
|Hard|**+0.2292**|1 / 0|41.48% → **43.75%**|7.26% → **7.26%**|通过|
|Sentinel-B|**+0.2801**|1 / 0|33.16% → **35.94%**|7.54% → **7.54%**|通过|

三套测试共实际检查 1,743 个 Ship 尾部框，只动作 8 次；经 official matcher 与最终
same-fine NMS 后形成 6 个净新增 TP、0 个净新增 FP。它是方案16中第一个完整通过
Normal → Hard → Sentinel-B、且满足 Ship 95% 增量精度门的模块。

准入范围必须保持窄：只准 `fine_id in {0,1}`、原框/原分/原标签不变；fine 2/3、Vehicle、
D4 Ship、A2、A3 均不得顺带加入。由于训练折校准仍只有一个 fold 找到 2 个 TP 的有效阈值，
本结果定义为“小样本准入候选”，下一步需做 full-data 拟合、冻结阈值来源说明、Docker
逐框一致性与时延验收，不能据此宣称已达到正式平台 84 分。

### 4.10 DINOv2-B 强教师上限

在不改变 proposal、A1M 头、LMP 样本、来源划分或精度门的情况下，把 P03 ConvNeXt
embedding 单因素替换为项目 P04 已证明更强的 DINOv2-B `CLS + patch-mean` 1536D
冻结表征。权重 SHA 为 `0b8b82f8…c73`，源码 commit 为 `7764ea0f…`。

结果是三个 outer fold 的 Ship 与 Vehicle 均无法从来源隔离校准集选出满足
95%/90% 精度的阈值，六个分类头全部 fail closed，Normal 质量变化为 `0.0000`。
因此不进入 Hard/Sentinel，也不启动蒸馏。结论不是 DINOv2 表征整体较弱，而是当前
proposal 尾部的可校准真阳性过少且与结构化背景混杂，单纯换更强通用教师不能解决。

### 4.11 class-safe 候选的来源稳健性复核

4.9 的 `{0,1}` 白名单在单一 20% 校准切分上只依赖一个 outer fold 的
`2 TP / 0 FP` 选择阈值，属于必须复核的小样本信号。保持 proposal、A1M、动作白名单、
95% 精度要求和最终 NMS 全部不变，仅把阈值选择替换为五路 source inner-OOF，并要求
至少 10 个校准 TP、3 个正来源组。复核后，三个 outer fold 的 positive source group
均为 0，全部 fail closed；Normal 相对 NMS control 的增益严格为 `0.0000`。

因此，4.9 是有价值的误差定位结果：Ship fine 0/1 的低分尾部确实存在少量纯净可救信号；
但它不是可复现的部署准入结果。单切分的 Normal/Hard/Sentinel 三套正增益保留为历史观测，
不得用于训练 full、打包或官方提交。APEX proposal classifier 路线到此停止。

## 5. 决策规则

只有模块相对同 NMS-only control 在 Normal、Hard、Sentinel-B 同方向改善，并满足
Ship 95% / Vehicle 90% 的有效增量精度门，并通过来源稳健阈值复核，才进入 full-data
与 Docker 阶段。当前没有 Ship/Vehicle APEX 模块满足完整条件：Ship class-safe 0/1
只通过了单切分观察，随后被 source inner-OOF 正式否决；A0/A1/A2/A3、全 Ship A1M、
Vehicle A1M、D4 Ship 和 DINOv2-B 也均为负向或 fail-closed 消融。不得扫描融合权重、
事后放宽阈值，或把 4.9 的探索性结果物化为部署模块。

下一条正式探索转到检测器级小目标提议学习：以 P40 fold0/40e 为成对基线，只增加
满足完整标注可见率约束的 Ship/Vehicle object-scale scene crops。它与已失败的孤立目标
拉伸、proposal 后分类不同；裁块保留上下文和所有完整可见标注，部分截断目标整块拒绝。

补充门禁解释：方案16中 Ship `FDR 增长≤0.5pp` 是 proposal rescue 的强准入建议，
不再作为所有检测器探索的一票否决。细类样本很少时单框即可造成超过 0.5pp 的跳变，
而官方按三粗类 macro 指标的均值判断总体 FDR 20% 硬门。后续检测器候选以相同工作点的
官方质量贡献净变化为主，逐类 Recall/FDR 全量报告；`+0.5` 质量分只作为是否进入 Hard
确认的工程带。只有总体 macro FDR 硬门风险、跨集反向或总分净损失才构成停止理由。

P03 全量 ConvNeXt-T 也已完成：20,933 个正式 tight crop、30 个固定 epoch、自然采样，
最终 checkpoint SHA `a2392b34…a3014db`；训练末 loss `0.62266`、accuracy `0.99971`。
它未使用验证集选权，只作为已独立准入 Aircraft D4 模块的 full-data 分类骨干资产。

### 5.1 检测器级尺度审计与 S128 执行

在 P40 冻结工作点对全部 1,218 个 Ship/Vehicle FN 计算其进入 1280 网络后的长边：

|粗类|FN|网络长边中位数|Q10 / Q90|结论|
|---|---:|---:|---:|---|
|Ship|949|235.29 px|132.80 / 443.00|漏检不以极小目标为主，需要较宽尺度上下文|
|Vehicle|269|66.84 px|51.20 / 118.04|全部低于 140 px，是明确的小目标瓶颈|

因此先运行 S128 单因素，它主要验证 Vehicle 小目标尺度假设。训练集 2,974 张与验证集
1,507 张路径零交集；仅从训练集物化 82 张场景裁块（fine 1/2/3/24 分别为
1/14/26/41），每张来源图最多一个。目标可见率须≥95%，其他标注可见率≥70%才保留，
落在 5%–70% 的部分截断标注会使整张裁块被拒绝；边界外采用 reflect padding。

单卡任务迁移到 3×RTX 4080 SUPER 后从同一 S1024 checkpoint 重新开始。DDP 总 batch 60
（每卡20）配合 `nbs=64` 的有效批量约60，接近原单卡 batch8、累积后的约64；首轮
batch24 在第2个 epoch 后中止并保留为 aborted，不参与结果。正式训练保持40 epoch、
imgsz1280、lr0=0.0002、seed42和Rot90不变。三卡训练已在 1,160.34 秒内完成40个
epoch，末轮 box/cls/dfl loss 为 `0.77931 / 0.33949 / 0.00535`；checkpoint SHA 为
`5e922ab2…1e626b8`，已下载并复核一致。训练过程中每卡约26.8GB，主体利用率接近100%。
随后在 RTX 3090 对原始 1,507 张 fold0 验证图完成低阈值推理，并在预先冻结的全局阈值
`0.546` 上与 P40 成对比较。S128 的质量贡献为 `47.2283`，P40 为 `50.3034`，差值
**−3.0752**，明确不准入。整体趋势不是“多找出 Vehicle”：Vehicle Recall 下降
`3.76pp`，同时 FDR 下降 `5.68pp`；Ship 宏 Recall 虽增加 `0.94pp`，宏 FDR 却增加
`16.76pp`。逐细类审计确认主要风险来自仅有 6 个 GT 的 Ship fine 0：候选未新增 TP，
却增加 2 FP，使其 FDR 从 0 跳到 66.67%。更常见的 Ship fine 2/3 则分别减少 8/17 TP。

为排除单纯置信度标定漂移，又读取同一验证标签上的诊断 oracle frontier。S128 的最优
阈值从 `0.546` 附近移至 `0.626`，但其最优质量仍只有 `60.0754`，低于 P40 的
`61.3468`（最优阈值 `0.616`），差值仍为 **−1.2713**。因此失败不是只需改阈值，
而是排序与识别边界本身退化；S128 不进入 Hard、Sentinel、full 或 Docker。

按原先已经 dry-run 的相邻假设，另行运行分级尺度 `Ship256/Vehicle128` 探索：它扩大
Ship 上下文并保持 Vehicle 目标尺度不变，共 359 张训练场景。该实验在 S128 负向后不再
具有自动准入资格，但能以一个受控实验回答“统一 128 尺度错误”还是“整个场景裁块方向
错误”。它保持 P40 初始化、40 epoch、imgsz1280、lr、seed、Rot90 与验证集不变；完成后
仍只在冻结阈值做一次成对评估，不扫描尺度或阈值。

S128 的逐粗类标签 oracle 进一步排除了“不同粗类只需不同阈值”的解释。候选相对基线的
最佳质量贡献：Ship `16.2543 < 16.6395`，Vehicle `14.4767 < 15.6083`；Aircraft
基本持平。也就是说 Ship/Vehicle 在允许各自选择最优阈值后仍退化。统一候选分流器
`scripts/triage_detector_candidate.py` 因此返回
`stop_ranking_and_fixed_workpoint_both_worse`；该 oracle 只用于诊断，绝不写入部署阈值。

`Ship256/Vehicle128` 已完成 40 epoch，checkpoint SHA 为
`de0a24b9…475511fd`，必要训练产物与固定评测均已回收，完整结果见第 7 节。为避免三卡空闲，另启动一个
独立窄对照 `Vehicle-only S96`：仅增加 36 张来源隔离的 Vehicle 场景裁块，Ship 与
Aircraft 完全不增加增强图；其目的是检验 S128 表现出的“Vehicle FDR 降、Recall 也降”
是否来自放大过强。它不是部署候选，仍须先通过同一 1,507 图固定工作点评测。

## 6. 方案16实现覆盖与剩余边界

|方案16方向|代码/产物|终态或下一步|
|---|---|---|
|LMP jitter hard negatives|`jitter_hard_negative.py`、`build_apex_boundary_manifest.py`|A1/A1M/稳健校准均拒绝，不重跑|
|PET-DINO式动态原型|`prototype_memory.py`、`apex_boundary.py`|线性、MLP、DINOv2-B 均未形成来源稳健阈值，停止|
|MPSR classifier positives|`build_apex_scale_manifest.py`|A2 Normal 负向，停止|
|Domain-RAG-lite背景负例|`background_retrieval.py`、`build_apex_background_manifest.py`|1,043个已审背景负例 A3M 负向；无可靠 mask 时不做伪合成|
|ETS式单因素选择|A0/A1/A2/A3 配对矩阵与顺序门禁|已完成；不组合独立失败模块|
|MPSR detector scene crops|`materialize_object_scale_detector_scenes.py`及三卡驱动|S128拒绝；分层尺度与 V96 均获 Vehicle 路由 CV3 资格|

补丁中的四个底层原语已逐项对照现有实现，补丁测试在现有代码上为 `7 passed`；项目级
APEX测试为 `11 passed, 2 skipped`。下载补丁里的 standalone COCO manifest 脚本没有
覆盖当前 YOLO/source-group 集成实现，因此不另加一份重复入口。当前仍缺少可审计实例
mask，Domain-RAG 图像合成按设计 fail closed；这不是遗漏代码，而是为了避免矩形粘贴和
隐藏目标污染。

## 7. 分层尺度与类别互斥路由结果

`Ship256/Vehicle128` 分层场景裁块候选已完成固定 fold0、Hard 和 Sentinel-B 复核。
完整候选在 Normal 的质量贡献变化为 `+2.3115`，但 Ship、Aircraft、Vehicle Recall
分别下降 `1.895pp / 1.270pp / 4.511pp`，正收益几乎全部来自 Vehicle FDR 下降
`10.490pp`。因此没有把整个检测器替换为候选，而是冻结类别互斥路由：P40 独占
fine 0--23，分层候选只提供 fine 24，合成前不改变任何框、分数或标签。

|测试|Vehicle Recall 基线→路由|Vehicle FDR 基线→路由|质量贡献变化|
|---|---:|---:|---:|
|Normal fold0|40.60% → 36.09%|18.18% → 7.69%|**+2.5421**|
|Hard|47.28% → 41.85%|20.18% → 7.23%|**+3.1205**|
|Sentinel-B|53.70% → 51.23%|25.00% → 2.35%|**+5.3287**|

Ship 与 Aircraft 在三套账本中逐框保持不变。Normal 单切分 label-oracle 诊断仍有
`+1.2734`，说明不是只靠固定阈值偶然获益；Hard/Sentinel-B 的同方向变化说明该模型
形成了可迁移的高精度 Vehicle 专家。代价是 Vehicle Recall 下降 `2.47--5.43pp`，故它
当前只获得 **CV3 确认资格**，尚未获得 full、Docker 或正式提交资格。checkpoint SHA 为
`de0a24b9f6057b1bd4eb229be90625ce9ad82facb3c1b1c33148c818475511fd`。

另一个独立单因素 `Vehicle-only S96` 只增加 36 张 Vehicle 场景裁块。在 Normal 中，
完整候选质量变化 `-0.4185`，但类别互斥 Vehicle 路由变化为 `+1.9598`：Vehicle Recall
下降 `2.256pp`、FDR 下降 `7.656pp`，单切分 oracle 仍为 `+0.5276`。它比层级候选保留
更多 Recall，但压低 FDR 的能力也更弱；checkpoint SHA 为
`7413855e952264bad0522758f2f7d8ff1ec97d89adf6107ccadec2a63f9a9bf5`。

V96 的独立复核随后给出比 fold0 更强的结果：Hard 的 Vehicle Recall 从 `47.28%`
升至 `49.46%`（`+2.174pp`），FDR 从 `20.18%` 降至 `6.19%`（`-13.998pp`），质量
贡献 `+4.1859`；Sentinel-B Recall 从 `53.70%` 升至 `54.32%`（`+0.617pp`），FDR
从 `25.00%` 降至 `2.22%`（`-22.778pp`），质量贡献 `+5.6773`。这不是单纯用 Recall
换 FDR：两个独立困难集上二者同时改善。V96 因而成为当前优先级最高的 Vehicle 路由，
随后在 RTX 3090 完成 fold1 独立确认。相对本折 P40，Vehicle Recall 从 `21.64%`
降至 `18.66%`（`-2.985pp`），FDR 从 `30.95%` 降至 `16.67%`
（`-14.286pp`），质量贡献仍为 **`+1.8248`**；单折 oracle 诊断也为 `+1.9850`。
checkpoint SHA 为
`9e63057876bbd399cbbec248b6534e69a8e007ce6e90e2a4264710683a2513ad`。

至此 V96 在 Normal fold0、Normal fold1、Hard、Sentinel-B 四份账本均为正向，其中两个
困难代理实现 Recall/FDR 同时改善，两个 Normal 折则以少量 Recall 换取更大的 FDR 改善。
它获得 full-data 拟合资格。full 训练严格复用正式 P40 的 S1024/160e 初始化，只改变
Vehicle S96 场景补充；4,481 张全量图共找到 53 张合格补充场景，固定训练 40 epoch、
三卡全局 batch 60、fixed-last，不使用验证集选权。hierarchy 的 fold1/fold2 也均完成
40 epoch，继续作为强度更高但 Recall 代价更大的 CV3 对照。两条路线都必须保持 fine 24
独占，不能把完整候选误当主检测器替换。

为避免把 fold0 偶然性误当成正式结论，三卡服务器已按完全相同合同训练分层候选的
fold1/fold2（每折 40 epoch、全局 batch 60、seed 42、imgsz 1280、Rot90、同源隔离）。
训练完成后将在 RTX 3090 上对每折原始 held-out 图做一次固定低阈值推理，并仅聚合
Vehicle 路由的 CV3 OOF。阈值不扫描、类别所有权不改变；CV3 未完成前不得物化 full。

## 8. V96 独立分支工作点校准

正式 v2.0 的三粗类平均 Recall 为 `86.2480%`，距离 `85%` 硬门仅
`1.2480pp`。其中 Vehicle 约有 95 个 GT；若 Ship/Aircraft 不变，多漏 3 个 Vehicle
后平均 Recall 约为 `85.1954%`，多漏 4 个则约为 `84.8445%`。因此 `0.546` 上 Normal
两折的 Recall 损失虽然质量分为正，但不能直接作为部署工作点。

已实现 `scripts/analyze_resolution_route_thresholds.py`，严格按 Docker 的类别互斥语义
评估：内部 CV 折的 P40 固定拥有 fine 0--23 且阈值保持 `0.546`，V96 只拥有 fine 24
并单独扫描分支阈值。fold0+fold1 作为校准集；约束为每折 Vehicle Recall 不得下降超过
`2.2pp`、每折总体 FDR 不得变差、平均质量贡献必须为正。该扫描不是正式隐藏标签
oracle，且 fold2 在阈值冻结前保持未打开。

校准得到唯一的受保护最优点 **`expert_threshold=0.505`**；它也恰好是校准区间内的
无约束质量最优点：

|校准折|Vehicle Recall 变化|Vehicle FDR 变化|质量贡献变化|
|---|---:|---:|---:|
|fold0|`+1.504pp`|`-8.504pp`|`+2.5815`|
|fold1|`+2.985pp`|`-17.794pp`|`+3.4294`|
|两折均值|`+2.244pp`|`-13.149pp`|`+3.0054`|

因此先前 `0.546` 上的 Recall/FDR 交换主要是 V96 分数标定变化，而不是提议排序能力
不足。`0.505` 同时提高两折 Vehicle Recall 并显著降低 FDR，消除了硬门风险。完整曲线
及输入 SHA 保存在
`outputs/HERA-GUARD-APEX-20260904/HERA-GUARD-MPSR-VEHICLE-S96-CAL-F01-V1/calibration.json`。
该阈值现已冻结，正在 RTX 3090 上训练 fold2；fold2 完成后只在 `0.505` 打开一次，
不再根据结果改阈值。V96 full 已完成 40/40 epoch：全量 4,481 图中加入 53 张仅含
Vehicle 目标的合格 S96 场景补充，初始化权重 SHA 为 `f7e30fac…17418229`，最终
checkpoint SHA 为 `6b4909f5c535c632dd1a36a7dc4c9dc6ed6cb26f02b74b4d934220113ddca5c6`。
训练固定使用三卡 global batch60、imgsz1280、seed42、Rot90、fixed-last，且不使用验证集
选权。正式 full 组合仍保持已提交 P40 的 `primary_threshold=0.536`，只给 V96 Vehicle
分支使用 `expert_threshold=0.505`；内部折的 `0.546` 不得误写进正式配置。只有 fold2
同方向且 full
权重完成独立推理、时延及离线/Docker逐框一致性后，才生成提交候选。

## 9. V96 full 困难代理复核与下一条全量链

V96 full 完成后立即以正式部署工作点做只读复核：P40 继续独占 fine 0--23 并使用
`0.536`，V96 full 只提供 fine 24 并使用校准冻结的 `0.505`。这两份评估使用的是 full
权重，且代理图像来自训练语料构造，因此只用于检查部署组合是否保持既定方向，不作为
新的准入证据。

|账本|Vehicle Recall P40→P40+V96|Vehicle FDR P40→P40+V96|三粗类平均 Recall|三粗类平均 FDR|
|---|---:|---:|---:|---:|
|Hard|24.46% → **80.43%**|19.64% → **1.99%**|52.76% → **71.42%**|19.72% → **13.84%**|
|Sentinel-B|46.30% → **88.89%**|10.71% → **0.69%**|64.39% → **78.58%**|14.99% → **11.65%**|

Ship 与 Aircraft 的预测逐框取自同一 P40 输入，组合时没有变化。该结果与四份先前独立
账本方向一致，说明 full 物化没有破坏 Vehicle 专家的高召回、低虚警特性；最终是否准入
仍以尚未打开的 fold2 `0.505` 固定工作点和 Docker 一致性为准。

三卡服务器的原 V96 driver 在 fixed-last 权重生成后按设计退出，所以一度显示空闲，
并非训练中断。后续已启动 `R1-5-AIRCRAFT-VIEW-CONSISTENCY-FULL-V1`：从冻结的 P03
全量 30e ConvNeXt-T checkpoint 初始化，使用全部 17,948 条已审计 Aircraft proposal
crop，训练 5e symmetric-D4 consistency，fixed-last、无验证集和无 checkpoint 选择。
该模块只允许改 fine 4--23，Ship/Vehicle 必须结构旁路；完成后再接入 P40+V96 组合并做
固定 Hard/Sentinel、3090 时延和离线/Docker逐框一致性。此阶段不生成镜像、不正式提交。

飞机全量模块随后完成：5/5 epoch、17,948 条训练 crop、fixed-last checkpoint SHA
`5f1b175f8b2c310a3c0583e652f5cf7cfd444d0b0d74d794a9ac59e4537832d5`。训练损失
从 `0.33674` 降至 `0.33228`，D4 consistency loss 从 `0.00654` 降至 `0.00175`，
所有数值有限。该 full 模型在 Hard/Sentinel-B 上相对已经通过的 CE-D4 参考，质量贡献
分别再增加 `+3.2466 / +3.5084`，方向与三折实验一致。因为 full 模型见过构造代理时
使用的正式训练来源，这两个数值只证明 full 物化和推理路径没有退化，不能代替 OOF 准入。

把 D4 full 只用于 fine 4--23、V96 full 只用于 fine 24 后，类别互斥组合保持 Ship
逐框不变；Hard 的 Aircraft/Vehicle macro Recall 为 `89.45% / 80.43%`、macro FDR
为 `1.79% / 1.99%`，Sentinel-B 分别为 `92.83% / 88.89%` 与 `3.06% / 0.69%`。
这再次确认两个模块能机械组合且不会互相覆盖。当前剩余关键项不是继续训练新模型，而是：
完成未参与校准的 V96 fold2 `0.505` 验收、把 Aircraft D4 后处理接入双分辨率 runtime、
再在 RTX 3090 做端到端时延和离线/Docker逐框一致性。

## 10. fold2 冻结确认与部署决策

fold2 在阈值冻结后只打开一次。在 P40 `0.546` 与 V96 专家
`0.505` 的预注册工作点上，Vehicle Recall 从 `33.333%` 降至 `30.370%`
（`-2.963pp`），FDR 从 `21.053%` 升至 `22.642%`（`+1.589pp`），质量贡献
变化为 **`-0.4690`**。该点同时违反 Recall 保护、FDR 不恶化与质量正增益
三项条件，因此 `admission=false`。

这一结果推翻了“V96 四份独立账本均稳定正向”的中间判断：V96 在
fold0/fold1 与 Hard/Sentinel-B 上存在强信号，但该信号没有跨到保留的第三
来源折。full V96 的 Hard/Sentinel 数值又使用了训练语料派生代理，不能覆盖这份
独立反证。因此：

- V96 full 保留为诊断资产，不进入 Docker，不占用官方提交次数；
- P40 继续独占 Ship/Vehicle，Aircraft D4 仍是唯一通过独立证据的新模块；
- 已训练的 `Ship256/Vehicle128` fold1/fold2 只补完一次固定 CV3，作为替代
  Vehicle 路由的收口对照，不再为 V96 扫阈值或训练新 full。

部署端已新增 Aircraft D4 后处理，且实现了与分辨率专家共存的结构；
P40+V96+D4 在一张真实图上完成端到端加载和推理，证明双分支工程链
可运行，但因 V96 被 fold2 否决，正式压力测改用
`p40_aircraft_d4_full_runtime_candidate_v1.json`：保持 P40 `0.536`，只增加高置信
Aircraft D4 重标与同细类 NMS。三张 4080 SUPER 仅分片执行 4,481 图端到端
稳定性测试，该测试不用于估计隐藏集分数。

## 11. 分层 Vehicle 专家 CV3 闭环与 full 准入

`Ship256/Vehicle128` 类别互斥路由完成三折固定 `0.546` 工作点评估。
P40 始终独占 fine 0--23，候选只提供 fine 24，因此 Ship/Aircraft 逐框不变。

|outer fold|质量贡献变化|Vehicle Recall 变化|Vehicle FDR 变化|
|---|---:|---:|---:|
|fold0|`+2.5421`|`-4.511pp`|`-10.490pp`|
|fold1|`+2.2034`|`-4.478pp`|`-16.138pp`|
|fold2|`+1.0483`|`+3.704pp`|`-3.020pp`|
|CV3 聚合|**`+1.9209`**|`-1.741pp`|**`-8.853pp`**|

三折质量贡献均为正，聚合 label-oracle 诊断仍为 `+1.5272`，说明该结果
不是只由固定阈值的标定差异造成。与 V96 相比，它的优势是跨折稳定的 FDR
改善；代价是 CV3 上 Vehicle Recall 平均下降 `1.741pp`。按当前正式 P40 约
`86.248%` 的三类平均 Recall 锚点粗略投影，若该变化原样迁移，总门禁约为
`85.668%`，仍高于 `85%`，但余量只约 `0.668pp`；因此 full 候选必须再做
3090 固定代理、时延和门禁风险计算，不得因 CV3 正向就自动提交。

三卡全量任务已准入：从正式 P40 S1024/160e 权重初始化，对 4,481 张全量
训练图物化完整标注可见的 Ship256/Vehicle128 场景裁块，固定 40 epoch、
imgsz 1280、global batch60、seed42、Rot90、fixed-last，不使用验证标签选权。
该任务在三分片 Aircraft D4 runtime soak 结束后自动接续，防止与三卡 DDP 争抢。

runtime soak 已通过：三分片共覆盖 4,481 张图、20,720 个最终输出框，无崩溃、
非有限值、非法类别或退化框。三分片平均逐图时间为 `0.3099 / 0.3106 /
0.2962s`，p95 为 `0.9138 / 0.9314 / 0.8440s`；这是 4080 SUPER 上的训练裁图
工程稳定性数据，不替代 3090 大图延迟门。压测结束后 full 训练已自动开始：
4,481 张原图加 575 张合格场景裁块，共 5,056 张训练输入；三卡 DDP 数据扫描、
权重迁移与 Rot90 加载均已通过。

## 12. P40 + hierarchy + Aircraft-D4 组合合同

V96 与 hierarchy 都不是主检测器替换。二者都从 P40 初始化并只允许在部署时拥有
fine 24；区别在于 V96 只补 53 张 Vehicle-S96 场景，而 hierarchy 在每个来源折内
同时物化 Ship-256 与 Vehicle-128 尺度场景，以更强的尺度课程约束检测器。V96 在
冻结 `0.505` 后的 fold0/fold1 为正、fold2 为负；hierarchy 使用事前固定 `0.546`
时三折质量贡献分别为 `+2.5421/+2.2034/+1.0483`，所以后者的核心价值是跨折稳定，
不是某一折峰值更高。聚合上 hierarchy 将 Vehicle FDR 降低 `8.853pp`，代价为
Vehicle Recall 降低 `1.741pp`，质量贡献增加 `1.9209`。

最终组合采用严格类别互斥：

- P40：fine 0--23，正式阈值 `0.536`；
- hierarchy：只提供 fine 24，固定阈值 `0.546`；
- Aircraft-D4：只对 P40 已输出的 fine 4--23 proposal 做高置信重标和同类 NMS，
  `relabel_min_probability=0.9`，不产生新框、不改 Ship/Vehicle；
- 两个检测分支各自完成大图融合后才按类别合并，分支之间不做几何融合。

因此三者机械可组合且互不覆盖。部署代码原有双分辨率路由已经支持在路由后调用
Aircraft-D4；新增 `build_p40_hier_d4_runtime_config.py` 在 full 权重净化后计算 SHA、
生成并校验唯一候选配置，`run_competition_runtime_coco.py` 则让固定代理直接经过与
Docker 相同的 `CompetitionDetector` 路径。相关单元与合同测试为 `21 passed`。

三卡已排队 `P40-HIER-S256V128-AIRCRAFT-D4-FULL-VALIDATION-V1`：等待当前 40 epoch
full 训练完成后，依次做 tensor-identical 权重净化、精确运行时配置生成，以及 Hard/
Sentinel-B 上 `P40+D4` 对 `P40+hierarchy+D4` 的逐框整链对照。该任务不扫阈值、
不打包 Docker、不发起官方提交；终态仍须补 RTX 3090 时延后才能给出提交决定。

把两项独立 OOF 增量机械平移到正式 P40 仅可作风险量级估计：hierarchy 单独约
`78.3`、D4 单独约 `78.7`、组合在 7 秒假设下约 `80.4`，相对正式 `76.6010`
有上行空间。该估计对 Aircraft FDR 触及零下限，明显偏乐观，不能作为官方分数承诺；
最终只看正在执行的整链方向、召回余量和 3090 实测时延。

full 分层模型随后完成 40/40 epoch；净化前 checkpoint SHA 为
`dcfd533993fe8112bc457f8c11da12c6ac0cdd5881f589060d1664ff21d87b1c`，
tensor-identical 净化后 SHA 为
`5e49e9bf532f04c69f8dc0735f1a66fd791e60e1602b6f06c6469d9a3aeb5cc9`。
精确运行时配置 SHA 为
`8d09e29830f9b269302b08c49880634728e0a1b5eefaa2794e25cc92ba6ec601`。
在 `expert_threshold=0.546` 下，整链对照结果并非两个代理同向：

|固定代理|Vehicle Recall P40+D4→加 hierarchy|Vehicle FDR P40+D4→加 hierarchy|质量贡献变化|
|---|---:|---:|---:|
|Hard|`80.978% → 81.522%`|`3.871% → 1.961%`|`+0.6006`|
|Sentinel-B|`89.506% → 87.037%`|`2.685% → 1.399%`|`-0.5732`|

Ship 和 Aircraft 在两套账本中保持不变，说明组合所有权正确；Sentinel-B 的负向完全
来自 Vehicle 少检 4 个 TP，不能归因于 D4 冲突。RTX 3090 上候选逐框指标与 4080
方向一致，Hard/Sentinel-B 的整张 100MP 压力图平均时延为 `20.320 / 18.246s`。
该时延包含超大图切片和 D4，不能直接替代官网普通图平均时延；它只说明整链在最重代理
上没有崩溃。相关回执已拉回
`outputs/HERA-GUARD-APEX-20260904/P40-HIER-S256V128-AIRCRAFT-D4-FULL-VALIDATION-V1`
和 `.../P40-HIER-S256V128-AIRCRAFT-D4-FULL-3090-V1`。

由于 `0.546` 时 Vehicle FDR 仅约 1.4--2.0%，但 Sentinel-B 存在 Recall 损失，新增
有界工作点复核：P40、D4、checkpoint、切片和融合全部冻结，只把 Vehicle 专家阈值
预先固定为 `0.480/0.510/0.535` 并在三卡并行复跑。三点中 `0.480` 最好：Hard
质量贡献为 `+0.8923`，Sentinel-B 从 `-0.5732` 收窄为 `-0.2403`；Vehicle Recall
分别为 `84.239% / 88.889%`，FDR 为 `1.899% / 2.703%`。它仍未满足双代理同向，
所以不能把均值正向包装成稳健准入；继续只做一次更低区间的有界收口，不做开放网格。

收口区间固定为 `0.420/0.450/0.465`，三点结果如下。这里的增量均以同一次精确
runtime 生成的 `P40+Aircraft-D4` 为基线，故 D4 的处理和开销已包含在两侧。

|Vehicle 阈值|Hard 质量变化|Sentinel-B 质量变化|两套最差变化|结论|
|---:|---:|---:|---:|---|
|0.420|**+1.8155**|**+1.0211**|**+1.0211**|双集同向，稳健最优|
|0.450|+1.9413|+0.7206|+0.7206|双集同向|
|0.465|+1.1001|0.0000|0.0000|无独立增益|

按事前规则“最大化两套代理的最差质量变化，并要求两套都为正”，唯一选择为
**`expert_threshold=0.420`**。在该点，Hard 的 Vehicle Recall/FDR 从
`80.978%/3.871%` 变为 `88.043%/2.994%`；Sentinel-B 从
`89.506%/2.685%` 变为 `92.593%/3.226%`。后一套 FDR 增加 `0.541pp`，但 Recall
增加 `3.086pp`，按正式分段计分后的净质量仍为 `+1.0211`；两套三粗类平均 Recall
分别为 `91.305%/92.568%`，平均 FDR 为 `2.711%/4.941%`，均有充分硬门余量。
该选择仍是代理集上的工作点选择，不是隐藏集分数预测；因此已启动两项只读收尾：
三卡覆盖 4,481 张正式训练图的整链稳定性压测，以及 RTX 3090 上的跨硬件逐框指标与
100MP 压力图时延复验。二者不再改阈值，也不自动生成 Docker。

RTX 3090 复验已完成，配置 SHA 为
`3b5f7f2d6a7cbd3d8d282fe5724ad6e0a7c06224889707401910c8c31d1215c5`。
Hard 的 Vehicle `R/FDR=88.043%/2.994%`，与 4080 完全一致；Sentinel-B 的 Recall
仍为 `92.593%`，FDR 为 `3.846%`，比 4080 多 1 个边界 FP，但按同一 P40+D4 基线
折算的质量方向仍为正。这一差异说明跨硬件 NMS 边界可能出现单框浮点抖动，不能要求
预测 JSON bitwise 一致，但门禁和改进方向一致。3090 上两张 100MP 代理集的平均逐图
时延为 `12.204 / 11.373s`，p95 为 `13.620 / 13.345s`，均低于 20 秒；该次文件缓存
已热，故只作为硬门压力复验，不与先前冷缓存耗时做小数级优劣比较。

三卡整链稳定性压测也已完成：4,481 张图全部唯一覆盖，输出 20,726 框，零崩溃、零非法
类别、零非有限值；三分片加权平均为 `0.1976s/图`，最差 shard p95 为 `0.5132s`。
这是 4080 SUPER 上的普通训练裁图工程指标，不替代 3090 的 100MP 压力时延，也不因
full 模型见过这些图而用于估分。两类测试合起来证明的是运行链稳定且时延门有余量。

固定代理上的模块贡献可以相加，因为 D4 只改 Aircraft 标签、hierarchy 只替换 Vehicle：
view-consistency D4 相对裸 P40 的质量贡献为 Hard `+2.7903`、Sentinel-B `+2.3850`；
hierarchy `t=0.420` 在 D4 基线上再增加约 `+1.8155/+1.0211`（3090 Sentinel 的单框
抖动后仍正向）。因此整套方案相对 P40 的质量方向约为 `+4.61/+3.41`，尚未扣除附加
时延。它只代表固定代理趋势，不可机械加到正式 `76.6010`；更合理的提交预期是“有希望
高于 P40，主要收益来自飞机细类纠错和车辆召回/FDR同时改善”，而不是承诺某个隐藏分数。

提交前代码配置已单独生成但尚未构建镜像：
`submission/docker/configs/p40_hier_s256v128_t042_aircraft_d4_v1.json`。四项本地资产及
目标容器路径记录在
`outputs/HERA-GUARD-APEX-20260904/P40-HIER-D4-DEPLOYMENT-ASSETS/manifest.json`；P40、
hierarchy、Aircraft-D4 checkpoint 和 ConvNeXt-Tiny 初始化权重的大小与 SHA 均已逐文件
复核通过。此处只完成候选 staging，不等同于 Docker/入口回归完成，也没有推送或提交。

## 13. P40 高置信 Vehicle 补漏与最终组合冻结

`t=0.420` 的 hierarchy 已同时提升两套固定代理，但 P40 与 hierarchy 的 Vehicle
预测并非完全重合。为利用已经执行的 P40 分支而不增加模型和时延，本轮只允许 P40
将高置信 fine-24 框补回 hierarchy：hierarchy 框绝对优先；P40 框按分数确定性降序处理，
只有与已保留 hierarchy/P40 Vehicle 框的 IoU 小于固定去重阈值时才追加。它不替换
hierarchy 几何，不改变 Ship/Aircraft，也不读取图像级或来源级条件。

搜索范围事前限制为 `P40 score={0.60,0.70,0.80}` 与
`dedup IoU={0.35,0.50,0.70}` 的 9 个离散组合。选择规则为：Hard 与 Sentinel-B
质量变化都必须严格为正，再最大化两者的最差变化；不会在结果后继续加密网格。

|P40 Vehicle 阈值|去重 IoU|Hard 质量变化|Sentinel-B 质量变化|两套最差变化|
|---:|---:|---:|---:|---:|
|0.60|0.35|-0.0594|+0.5490|-0.0594|
|0.60|0.50|+0.1553|+0.5490|+0.1553|
|0.60|0.70|**+0.2142**|**+0.5490**|**+0.2142**|
|0.70|0.35|-0.1140|+0.0649|-0.1140|
|0.70|0.50|-0.1140|+0.0649|-0.1140|
|0.70|0.70|-0.0594|+0.0649|-0.0594|
|0.80|0.35/0.50/0.70|-0.1650|-0.1772|-0.1772|

因此唯一冻结点为 **`primary rescue threshold=0.600, dedup IoU=0.700`**。
它在 Hard 追加 8 个 Vehicle 框，Vehicle Recall 从 `88.043%` 升至 `90.217%`，
FDR 从 `2.994%` 升至 `5.143%`；在 Sentinel-B 追加 4 个框，Recall 从
`92.593%` 升至 `94.444%`，FDR 从 `3.226%` 升至 `3.774%`。虽然 FDR 略升，
但仍远低于 20% 门槛，且按正式分段质量权重两套都净正向。完整 9 点结果和输入 SHA
保存在
`outputs/HERA-GUARD-APEX-20260904/P40-HIER-S256V128-T042-D4-PRIMARY-VEHICLE-RESCUE-V1/summary.json`。

部署端新增 `PrimaryLabelRescue` 合同，并由配置键 `resolution_primary_rescue` 明确限定
标签、阈值和去重 IoU。4080 exact runtime 的 Hard 2062 框与 Sentinel-B 1882 框，
和离线重放在 `(image_id, category_id, score, bbox)` 上逐框完全一致，证明分析逻辑没有
在接入 Docker 路径时漂移。两台硬件的完整复验如下：

|硬件|固定代理|相对 hierarchy `t=0.420` 质量变化|Vehicle Recall/FDR|均值/p95 秒每 100MP 图|
|---|---|---:|---:|---:|
|4080 SUPER|Hard|+0.2142|90.217% / 5.143%|13.642 / 14.595|
|4080 SUPER|Sentinel-B|+0.5490|94.444% / 3.774%|13.943 / 14.728|
|RTX 3090|Hard|+0.2142|90.217% / 5.143%|12.734 / 14.172|
|RTX 3090|Sentinel-B|+0.5544|94.444% / 4.375%|12.051 / 13.606|

RTX 3090 的 Sentinel-B 比 4080 多 1 个边界 FP，但方向、召回和时延门均保持一致。
若把这里偏重的 100MP 压力图均值直接代入官方七项公式，Hard/Sentinel-B 分别为
`86.09/85.95`；这只是一致口径的内部压力分，不能当作官网预测。相对正式 P40 的
`76.601`，更可信的信息仍是三个模块在固定代理上的增量方向，而不是内部绝对值。
该补漏复用已经运行的 P40 结果，计算量只有少量 IoU 比较，因此相对原组合没有可测的
模型推理增量。时延仍由 P40、hierarchy 和 Aircraft-D4 三部分主导。

最终冻结的提交候选为：P40 拥有 fine 0--23、阈值 `0.536`；hierarchy 拥有 fine 24、
阈值 `0.420`；P40 仅以 `score>=0.600`、`IoU<0.700` 的规则补回 hierarchy 未覆盖的
fine-24；Aircraft-D4 只重标 fine 4--23，概率阈值 `0.900`。对应配置为
`submission/docker/configs/p40_hier_s256v128_t042_p40v060_iou070_aircraft_d4_v1.json`，
SHA256 为 `3fbf3d69a4e4f9727d84efebe0cd4342076d97d581e6c37ba1547a9126d17759`。
资产权重没有变化，仍使用已验收的四个文件。

这一增量只由 8/4 个补漏框驱动，不能把代理上的 `+0.21/+0.55` 当作隐藏集保证；但它
没有引入新模型、没有碰已通过的 Ship/Aircraft、两套固定代理与两种 GPU 都同向，因而
在现有设计内比无补漏版更适合作为唯一提交候选。至此停止参数搜索，后续只允许进行
Docker 构建、资产 SHA、入口、输出合同和 3090 容器复测，不再改变算法工作点。

## 14. 正式 Attempt 3 结果：Aircraft 成功，Vehicle 与时延使总分回退

`v3.0` 已由官方平台完成：综合分 `75.9405`，低于 `v2.0` 的 `76.6010`，但三项
硬门全部通过。Ship 的 `418 TP / 35 FP / 202 FN` 与 v2 完全一致；Aircraft 净增
21 TP、减少 21 FP；Vehicle 只净增 1 TP，却增加 8 FP；平均时延从 `3.551833s`
增加到 `6.964667s`。

按七项等权总分拆解，Aircraft 贡献 `+0.3215`，Vehicle 贡献 `-0.4944`，时延贡献
`-0.4876`，合计 `-0.6605`。正式反馈因此确认 D4 的隐藏域正收益，同时否决当前
hierarchy `t=0.420` 加 P40 rescue 的 Vehicle 组合。下一候选必须撤掉 hierarchy 与
Vehicle rescue，只保留 P40 + Aircraft-D4，并以完整 Docker 配对时延验证 D4 单独增时
是否小于其官方质量原始收益 `2.2505s`。

完整接口字段、推断边界与代理失真复盘见
[正式 Attempt 3 复盘](FORMAL_ATTEMPT3_V3_PLATFORM_RESULT_AND_PROXY_POSTMORTEM_20260904.md)。
