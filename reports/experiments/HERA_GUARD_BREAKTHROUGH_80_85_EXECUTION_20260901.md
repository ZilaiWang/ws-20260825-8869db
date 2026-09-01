# HERA-Guard 80/85 分突破主线执行记录（2026-09-01）

## 1. 目标与不可变约束

目标不是在开发集上继续扫描一个好看的阈值，而是在固定的
`platform_observed_20260831` 计分口径下得到可迁移的新模型：

1. 正式隐藏分先超过 80；
2. 达到 80 后继续追 85；
3. Normal-CV3、Hard10K 和 source-disjoint Sentinel 必须同方向；
4. Aircraft 基本不退化；
5. 未经明确授权不打包、不构建 Docker、不推送、不提交。

当前正式锚点为 72.1331：

|粗类|Recall|FDR|
|---|---:|---:|
|Ship|0.874969|0.320177|
|Aircraft|0.967641|0.064691|
|Vehicle|0.852632|0.325000|

平均 Recall 为 0.898414，已经通过 0.85 门；平均 FDR 为 0.236623，未通过 0.20
门。正式耗时 2.473167 秒，不是当前主瓶颈。

## 2. 分数敏感度与真正目标

正式分数是三个 Recall 子分、三个 FDR 子分和一个耗时子分的算术平均。在当前区间，
单纯把 Ship 和 Vehicle FDR 从约 0.32 降到 0.20，其他指标完全不变，约只能增加 2.6
分，仍远低于 80。

若两类 FDR 都接近 0.10 且 Recall 不变，投影分约为 80.5。因此 80 分目标本质上要求：

```text
大约砍掉 Ship/Vehicle 一半以上的有效 FP
+ 基本保护现有 TP
```

85 分还需要在更低 FDR、Ship/Vehicle Recall 提升或两者的组合上继续取得明显收益。
这排除了“再调一次统一阈值”作为主突破口。

## 3. 已排除路线的统一解释

### 3.1 后处理/质量头已经饱和

- 固定阈值迁移无法同时满足 FDR 和 Recall 门；
- MacroRisk V2、Vehicle Reject+Rescue、Ship objectness-quality 均被外层门拒绝；
- base+crop quality 经标签无关分位数校准后只在 Sentinel 正向，在 Hard 上 FDR 反向；
- score-sqrt hard-negative 头在正确 Hard GT 上 Recall 仅 0.0908；
- D-FINE 乘法 agreement、HAD 蒸馏、全量双视图均已否决。

这些结果说明当前冻结 proposal/crop 表示不能稳定识别新 source/style 下的结构化背景。

### 3.2 困难背景全检测器微调失败的原因

旧 background-complete 每折加入 640 个困难 tile、微调 20 epoch，候选 Recall 下降约
10.5pp。后续 conservative hard replay 仍把 320 个困难 tile（训练图约 9.7%）加入
fold0，微调 6 epoch 后候选 Recall 下降 7.18pp。

这否定的是“高强度、同一 detector 误差来源驱动的短程负样本微调”，而不是背景监督
本身。其主要风险包括：

- hard tile 与当前 detector 错误高度相关，覆盖面窄；
- 空 tile 很少，且结构分布偏向六张 pseudo mosaic；
- 分类/目标性梯度直接压低低分真实目标；
- 训练图中 hard tile 占比过高，短程微调也发生严重分数漂移。

因此不再扫描旧 hard-replay 的 tile 数、epoch 或学习率。

## 4. 第一条新表示路线：DOTA part1 EXT-V

这是当前最高优先级、尚未正式完成的表示学习实验。它使用 469 张完整 DOTA
高分辨率场景、26,777 个经审计的 HBB 标注，只学习：

```text
aircraft / ship / vehicle / other_remote_object
```

外部类别不伪装成官方 25 个细类。原图按 1024、overlap=256 做尺度保持切片；每张图
最多保留两个确定性空背景 tile。EXT-V 冻结采样对 Vehicle 图重复 2 次、
`other_remote_object` 图重复 4 次，以减轻 Ship 数量主导。

### 4.1 配对快筛

|阶段|候选|对照|固定合同|
|---|---|---|---|
|外部预训练|DOTA EXT-V|无|40 epoch，Y5-S，1024|
|官方 fold0|EXT-V backbone/neck，重置25类头|官方初始 Y5-S，同样重置头|各40 epoch；8 epoch head warmup|
|评估|候选替换 fold0|对照替换 fold0；原160ep incumbent作第二锚点|Hard、Sentinel 只推 fold0，fold1/2复用冻结预测|

### 4.2 准入条件

至少满足：

```text
candidate 相对 paired control：
  Hard @FDR15 Recall >= +0.5pp
  Sentinel 同方向
  candidate floor 不下降超过0.3pp
  任一 coarse Recall 不下降超过0.5pp

candidate 相对 160ep incumbent：
  不要求40ep快筛绝对超过，但必须显示 Ship/Vehicle FDR/Recall 的明确新方向
```

若通过，扩展到正式 CV3 或直接采用更长官方 fine-tune 进行第二阶段确认；若失败，停止
DOTA part1 邻域，不用额外 epoch 掩盖负结果。

## 5. 后续突破口的冻结优先级

只有当前路线给出结果后，才按以下顺序推进，避免并行堆叠不可归因变量：

1. **同架构第二 seed 一致性专家**：配对控制训练本身也生成独立 seed；先测试
   same-fine support 能否压 Ship/Vehicle FP，不新增框；
2. **低强度检测级背景正则**：仅在外部表征正向但 FDR 仍偏高时，使用经过来源扩展的
   Background-100MP，加入蒸馏保护或只更新有限检测头；不复刻旧 9.7% hard replay；
3. **独立粗类 objectness 分支**：只有能在 Hard/Sentinel 学到稳定分数尺度时，才作为
   有界残差，不使用硬 DROP；
4. **唯一 full 配方**：只组合独立通过门禁的模块，使用全部 4,481 张官方图固定 epoch
   训练。full 完成不等于自动提交。

## 6. 当前执行状态

|项目|状态|
|---|---|
|方案10--13与负向结果复核|完成|
|80/85分敏感度分析|完成|
|DOTA part1完整资产|本机与服务器均已冻结|
|DOTA part1服务器完整性|469/469 逐图 SHA256 通过|
|center-owned v1切片|1816 tiles；人工视觉门禁失败，重叠 tile 出现完整对象无标签|
|all-visible v2切片|2066 tiles / 57,853 instances；96卡人工视觉门禁通过|
|paired control 40 epoch|8 epoch head warmup + 32 epoch foundation，完成|
|EXT-V 40 epoch + candidate 40 epoch|完成；三域门禁拒绝|
|Hard/Sentinel fold0替换评估|完成|
|Normal-CV3 fold0替换评估|完成|
|Varifocal质量感知分类|完成；跨域与尺度无关诊断均大幅退化，拒绝|
|Hard-negative focal分类|7项专项测试通过，执行中|
|同架构第二seed支持备用路线|完成回放；Hard/Sentinel Recall 大幅下降，拒绝|
|full训练|未准入|
|打包/提交|禁止，未执行|

## 7. 文件索引

- `scripts/server/run_dota_part1_extv_prepare.sh`
- `scripts/server/run_extv_part1_single_gpu_screen.sh`
- `scripts/server/finalize_extv_single_gpu_screen.sh`
- `scripts/server/run_y5_fold0_normal_replacement_eval.sh`
- `scripts/server/run_quality_aware_single_gpu_screen.sh`
- `scripts/train_external_y5_coarse.py`
- `scripts/train_external_initialized_y5_fine.py`
- `scripts/rescore_same_arch_support.py`
- `scripts/run_multifamily_cv3_pseudo_eval.py`
- `scripts/analyze_cv3_oof_pseudo_frontier.py`
- `src/rsdet/innovation/quality_aware_loss.py`
- `src/rsdet/submission/same_arch_support.py`
- `outputs/HERA-GUARD-BREAKTHROUGH-DOTA-PART1-PREP-V1/visual_review_failure.json`
- `reports/experiments/FORMAL_ATTEMPT1_ABSOLUTE_SCORE_FREEZE_20260831.md`
- `reports/experiments/IMPROVEMENT_PLAN13_AUDIT_AND_NEXT_ACTION_20260901.md`
- `reports/experiments/MACROSHIFT_FROZEN_7STEP_IMPLEMENTATION_AND_LOCAL_RESULTS_20260901.md`
- `reports/experiments/MACROSHIFT_ATTEMPT2_CANDIDATE_TOURNAMENT_20260901.md`

## 8. v1 视觉门禁发现与修正

第一次全量切片保持 1024 像素尺度并保留了全部 26,777 个源标注，但其
`center-owned` 去重规则与 256 像素重叠不兼容：一个对象只在唯一 owner tile 中有
标签，像素却会完整出现在相邻 tile。96 张审核卡中，多张码头图出现成排完整船只未
画框，因此这些 tile 会把真实目标作为背景监督。

该版本未进入训练。v2 改为：只要对象在某个 tile 中的保留面积比例达到 0.7，就在
该 tile 中复制相应的裁剪标注。源对象在重叠 tile 之间允许出现多个训练实例；审计
增加源标注覆盖数、重复源标注数和单对象最大 tile 实例数。这个修正消除的是标签
语义错误，不属于为了结果调参。

v2 最终得到 2,066 个 tile，其中 1,769 个含目标、297 个为空背景；57,853 个 tile
实例覆盖全部 26,777 个源标注，18,364 个源标注在充分可见的重叠 tile 中有多个训练
副本，单个源标注最多出现 9 次。对 8 张 contact sheet、共 96 个样本的全量人工检查
未再发现 v1 的系统性完整目标漏标，框几何与四个粗类语义均可接受，因此只准入当前
fold0 配对快筛，不自动准入 full 或提交。

## 9. EXT-V 配对结果：拒绝

外部粗类预训练本身正常收敛：40 epoch 最终 Recall 0.9474、mAP50 0.9638、
mAP50-95 0.7613。候选与 control 均使用同一 official fold0、同一 seed、8+32 epoch
训练，仅初始化不同。固定 @FDR15 / Hard 阈值冻结到 Sentinel 的结果如下：

|条件|Recall control|Recall EXT-V|Δ Recall|FDR control|FDR EXT-V|Δ FDR|
|---|---:|---:|---:|---:|---:|---:|
|Normal-CV3|0.914871|0.907562|-0.007309|0.164223|0.171623|+0.007400|
|Hard|0.731696|0.721965|-0.009731|0.150619|0.150027|-0.000591|
|Sentinel（Hard阈值）|0.746064|0.741493|-0.004571|0.108617|0.108669|+0.000053|

Normal 的 Vehicle Recall 下降 2.736pp、FDR 增加 2.552pp；Hard Vehicle FDR 虽下降
2.650pp，但 Recall 仍下降 1.087pp；Sentinel 只有 Ship Recall 增加 0.857pp，Aircraft
Recall 下降 1.831pp，Vehicle FDR 增加 5.068pp。五项统一准入门均未通过，正式结论为
`stop_without_parameter_scan`。因此停止 DOTA part1 / EXT-V 邻域，不用增加 epoch、改变
采样或融合掩盖负结果，也不进入 full。

这个结果把瓶颈进一步定位为官方 25 细类下的**分数排序与跨域校准**，而不是粗类目标
是否可见。下一条路线保持数据、初始化、seed 和训练长度不变，只将全阶段 BCE 替换为
质量感知 Varifocal，以直接检验高置信困难负样本能否获得更好的排序。

## 10. 同架构第二 seed 支持：拒绝

使用 40 epoch paired control 作为 fold0 specialist，对 160 epoch incumbent 的原框只做
同细类支持乘积；Ship IoU 固定 0.50、Vehicle 固定 0.35，Aircraft 完全 bypass，
fold1/2 也完全 bypass。没有新增框、改框、改类或融合权重扫描。

在 fold0 范围内，Hard 的 3,895 个 Ship/Vehicle proposal 只有 869 个获得支持，
Sentinel 的 2,960 个只有 730 个获得支持。Hard @FDR15 Recall 从 0.744671 降到
0.708990；Sentinel 自身 @FDR15 从 0.799391 降到 0.772981；Hard 阈值冻结到 Sentinel
后从 0.756221 降到 0.714068。虽然 Hard Vehicle FDR 从 0.2031 降到 0.1176，但
Vehicle Recall 同时从 0.2772 降到 0.2446，Ship 损失更大。再考虑双模型时延，这一路
没有准入可能，停止而不扫描 IoU 或融合权重。

## 11. Varifocal 全阶段分类损失：拒绝

在与 paired control 完全相同的数据、初始化、seed、8+32 epoch 和增强下，只将
one-to-many / one-to-one 的 BCE 替换为标准 Varifocal 权重：负样本权重为
`0.75 * sigmoid(logit)^2`，正样本权重为 task-aligned soft quality target。

三折替换评估首先暴露出明显分数尺度错位；为避免因 candidate 只替换 fold0 而使用
baseline fold1/2 阈值误判，随后补做了每个条件 fold0 内、使用 held-out 标签的
scale-invariant oracle frontier。该诊断不用于阈值选择，只判断排序是否有信号：

|条件|control Recall@FDR15|Varifocal Recall@FDR15|Δ Recall|
|---|---:|---:|---:|
|Normal fold0|0.870204|0.719864|-0.150340|
|Hard fold0|0.692308|0.458580|-0.233728|
|Sentinel fold0|0.740741|0.471510|-0.269231|

Normal macro Recall 从 0.808784 降至 0.642837，Hard 从 0.711642 降至 0.584585，
Sentinel 从 0.728647 降至 0.615092。Ship 与 Vehicle 损失最大。结论不是简单校准问题，
而是 soft-quality 正样本再次乘质量权重后，低质量真实小目标梯度被过度削弱。

下一项保持相同的困难负样本 focal 权重，但把所有正样本权重固定为 1，以隔离“困难
负样本聚焦”本身是否有效；这不是 Varifocal 参数扫描，而是由失败机理导出的预注册
正样本保护对照。

## 12. 训练长度补充诊断：40 epoch 不能替代 160 epoch

EXT-V 配对实验中的 official-init control 只训练 8+32 epoch。为排除“160 epoch 源域过拟合”
这一潜在捷径，补做 fold0 内尺度无关前沿，将该 40 epoch control 与正式 160 epoch
incumbent 直接比较。两者在各自 held-out 标签上独立选择 FDR15 工作点，因此不受分数绝对
尺度影响：

|条件|40ep Recall@FDR15|160ep Recall@FDR15|差值|
|---|---:|---:|---:|
|Normal fold0|0.870204|0.896327|-2.612pp|
|Hard fold0|0.692308|0.727811|-3.550pp|
|Sentinel fold0|0.740741|0.799145|-5.840pp|

三个域一致支持 160 epoch；缩短训练会损失成熟候选排序，不能作为低成本正式候选。结果位于
服务器 `HERA-GUARD-TRAIN-LENGTH-DIAGNOSTIC-V1`。这一结果也限定了当前 40 epoch 损失实验
的用途：它只能判断损失函数相对同长度 paired control 的方向，不能仅凭通过该对照就直接
替代 160 epoch incumbent。

## 13. 正样本保护 hard-negative focal 与后续保守路线

全 25 类 hard-negative focal 保持 Varifocal 的负样本权重
`0.75 * sigmoid(logit)^2`，但把所有正样本权重固定为 1。训练数据、初始化、seed 与
8+32 epoch paired control 完全一致。三域执行与尺度无关诊断均已完成：

|条件|BCE control Recall@FDR15|hard-negative focal|差值|
|---|---:|---:|---:|
|Normal fold0|0.870204|0.780136|-9.007pp|
|Hard fold0|0.692308|0.430473|-26.183pp|
|Sentinel fold0|0.740741|0.488604|-25.214pp|

组合三折门禁同样失败：Hard Recall 相对 paired control 下降 6.580pp；Hard 阈值冻结到
Sentinel 后下降 8.126pp。Normal 的分数尺度发生巨大漂移，但 scale-invariant 前沿已经
证明并非校准可以修复。结果说明即便保护正样本 BCE 强度，对全部类别削弱大量易负样本
仍会破坏真实小目标的细类排序。该路线正式结论为 `stop_without_parameter_scan`。

该全类版本失败后，下一条不是扫描 alpha/gamma，而是将机制收窄到正式短板：只对 Ship
四类与 Vehicle 一类应用 hard-negative focal，Aircraft 二十类保持原 BCE。更关键的保守
候选从成熟 160 epoch incumbent 出发，只训练两条 YOLO26 分类分支最后 1x1 卷积中的
Ship/Vehicle 五个输出行；backbone、neck、box、DFL、BatchNorm 与 Aircraft 行均冻结。
这样可以检验“对成熟定位结果做受限排序修复”，而不重新学习已经有效的定位表示。

对应实现：

- `scripts/train_selective_classifier_finetune.py`
- `scripts/server/run_selective_classifier_single_gpu_screen.sh`
- `src/rsdet/innovation/quality_aware_loss.py::selective_classifier_trainer`

该路线仍须三域门禁通过才允许扩展；未授权打包或正式提交。

## 14. 成熟权重 Ship/Vehicle 分类行 hard-negative 微调：拒绝

从 160 epoch incumbent fold0 出发训练 12 epoch，只开放 Detect 头中 one-to-many 与
one-to-one 三尺度最终 1x1 分类卷积的 Ship 0--3 和 Vehicle 24 输出行。精确 checkpoint
审计确认：708 个 state tensor 中仅 12 个允许的分类张量变化；每个张量只有五个目标行
变化，Aircraft、box、DFL、backbone、neck 和 BN 均逐字节不变。

固定三域外层结果：

|条件|Recall 差值|FDR 差值|关键粗类变化|
|---|---:|---:|---|
|Normal|-0.425pp|+4.213pp|Ship FDR +19.965pp；Vehicle FDR +13.740pp|
|Hard|-1.019pp|+0.312pp|Vehicle Recall -1.630pp、FDR +3.497pp|
|Sentinel（Hard阈值冻结）|+0.152pp|+1.339pp|Vehicle Recall +1.235pp，但三个粗类 FDR 均恶化|

所有主要准入门失败，结论为 `stop_without_parameter_scan`。该结果把失败定位得更清楚：
并非全网络遗忘或框回归漂移，而是当前 hard-negative focal 对成熟 logits 的更新方向本身
不能跨域泛化。

下一项只做必要的机制对照：完全相同的成熟权重、五行冻结范围、数据、seed、12 epoch 与
零 weight decay，恢复原始 BCE。它检验保守分类再适配本身是否有价值，而不是继续扫描 focal
超参数。

## 15. 成熟权重 Ship/Vehicle 分类行 BCE 微调：拒绝

机制对照保持第 14 节的成熟 160 epoch 输入权重、五个允许分类行、数据、seed、12 epoch、
优化器和零 weight decay 全部不变，仅把 hard-negative focal 恢复为原始 BCE。精确 checkpoint
审计再次通过：708 个 state tensor 中只有 12 个允许分类张量变化，无越界参数漂移。

|条件|Recall 差值|FDR 差值|关键粗类变化|
|---|---:|---:|---|
|Normal|-0.024pp|-0.042pp|Vehicle Recall -0.995pp、FDR -2.689pp|
|Hard|-0.788pp|+0.090pp|Vehicle Recall -2.717pp；Ship Recall -0.524pp|
|Sentinel（Hard阈值冻结）|+0.254pp|+0.681pp|Ship Recall +0.643pp；Vehicle Recall -1.235pp|

Normal 表面接近不变，但 Hard 的主要召回门、粗类 -0.5pp floor 和冻结 Sentinel 的稳健性门均
失败，正式决定仍为 `stop_without_parameter_scan`。这证明第 14 节失败不只是 focal 公式所致：
在没有新增信息的情况下，对成熟模型五个分类行继续做常规再适配也不能产生稳定跨域收益。
因此停止“相同数据 + 分类行微调”邻域，不扫描 epoch、学习率或 weight decay。

下一项只改变一个真正与正式短板相关的数据因素：从已冻结且逐图审计的
Background-100MP 中，按 fold0 训练来源筛选背景 crop，严格排除 validation 来源；每四个
512 crop 拼为一个 1024 原尺度空背景 mosaic。它们预计只占增强后训练清单约 2%，远低于已
失败的 9.7% hard replay。训练仍从成熟权重开始、仍只更新五个分类行、仍使用原始 BCE，
由此配对检验低强度、来源扩展的背景监督是否能降低 Ship/Vehicle FDR 而保护 TP。

## 16. fold0 无泄漏低强度 Background-100MP：未产生稳定收益

Background-100MP 冻结 manifest 的 382 张 crop 全部按来源归属审计。fold0 中 241 张来自
训练来源，141 张来自 validation 来源并被排除，未知来源为 0。240 张训练 crop 被组成 60
张 1024 mosaic，覆盖 197 个来源；每个 mosaic 的四个 crop 来源互异，标签严格为空，原始
512 像素尺度不变。增强后训练清单为 3,034 张，背景 mosaic 占 1.978%，验证来源泄漏为 0。

训练仍从成熟 160 epoch fold0 权重开始，仅更新五个 Ship/Vehicle 分类行，原始 BCE、seed、
12 epoch、优化器和评估合同与第 15 节一致。精确 checkpoint 审计通过，12 个允许张量变化，
其余参数无漂移。三域结果为：

|条件|Recall 差值|FDR 差值|关键粗类变化|
|---|---:|---:|---|
|Normal|-0.024pp|-0.068pp|Vehicle Recall -0.995pp、FDR -2.525pp|
|Hard|-0.741pp|+0.173pp|Vehicle Recall -2.717pp；Ship FDR +0.700pp|
|Sentinel（Hard阈值冻结）|+0.152pp|+0.802pp|Ship Recall +0.429pp；Vehicle Recall -1.235pp|

它相对无背景 BCE 对照只产生很小扰动，并没有把背景监督转化为 Hard/Sentinel 上可迁移的
排序能力。结论为 `stop_without_parameter_scan`，不扩大背景比例，也不回到旧 hard replay。

进一步审计显示，无背景 BCE 的六个分类 bias 最大绝对变化达到约 2.8--4.1，而成熟 bias
本身约为 -9 到 -14；这解释了为何即使只开放五行，常规再训练仍会重设整套分数尺度。下一
个单因素机制把相同低强度背景训练改为**有界残差更新**：每个分类权重行相对成熟权重的
L2 变化不超过原行范数的 5%，bias 变化不超过 0.25；每个 optimizer step 后同时投影 live
model 与 EMA。该边界约比已观察到的失败漂移保守一个数量级，不扫描边界值。

## 17. 最终分类行有界残差：仍拒绝

低强度 Background-100MP 数据、成熟权重、BCE、五个目标行和 12 epoch 全部保持不变，
只增加第 16 节预注册的逐步投影。精确审计确认 708 个 state tensor 中仍只有 12 个允许
张量发生变化；最终 bias 最大变化恰为 0.25，未发生越界漂移。

|条件|Recall 差值|FDR 差值|关键粗类变化|
|---|---:|---:|---|
|Normal|-0.024pp|-0.072pp|Vehicle Recall -0.995pp、FDR -2.689pp|
|Hard|-0.741pp|+0.173pp|Vehicle Recall -2.717pp；Ship FDR +0.700pp|
|Sentinel（Hard阈值冻结）|+0.152pp|+0.802pp|Ship Recall +0.429pp；Vehicle Recall -1.235pp|

结果几乎复现无界背景版本，说明失败并非仅由 bias 大幅漂移造成。至此，“相同成熟表示上
只修改 Ship/Vehicle 最终分类行”的 focal、BCE、低强度背景和有界残差四个机制均被三域
证据拒绝，正式关闭该邻域，不扫描边界、学习率或 epoch。

下一项改变真正被旧实验固定住的变量：允许 `cv3/one2one_cv3` 分类分支的空间卷积学习
背景上下文，但所有分支张量每步限制在成熟参数 1% 的相对 L2 邻域，最终五行仍使用
5%/0.25 边界，BN running statistics 冻结。它不修改 box/DFL、backbone、neck 或训练数据，
用于判断“缺少空间背景表示”是否是分类行路线无法降低 Hard/Sentinel 虚警的原因。

## 18. 有界空间分类分支：Normal 有信号，跨域拒绝

该模型开放两条分类分支的空间卷积，但把普通分支张量限制在成熟权重 1% 相对 L2 邻域，
最终五行保持 5%/0.25 边界。checkpoint 专项审计通过：84 个张量变化、0 个越界；实际
最大普通分支相对变化 1.0007%，最终权重行 5.0034%，bias 0.25（差异在 FP16 容差内）。

|条件|Recall 差值|FDR 差值|关键粗类变化|
|---|---:|---:|---|
|Normal|+0.139pp|-0.076pp|Vehicle FDR -3.416pp，但 Recall -1.244pp|
|Hard|-0.371pp|+0.109pp|Vehicle Recall -2.717pp；Ship FDR +1.223pp|
|Sentinel（Hard阈值冻结）|+0.559pp|+0.914pp|Ship Recall +0.965pp，但 Ship FDR +1.481pp|

相对最终行路线，它在 Normal 与 Sentinel 产生了真实召回信号，证明空间上下文确实是缺失
自由度；但 Hard 方向仍负，且 Sentinel 的召回由更多虚警换得，不能扩三折。停止投影边界
扫描。下一项保留后半段表示学习能力，改用成熟 160 epoch 权重同时作为 student 初始化与
冻结 teacher，通过 YOLO26 原生 score-weighted feature distillation 保护正常图像响应；背景
比例、数据、seed 与候选选择合同保持冻结。

## 19. 成熟 teacher 背景蒸馏：拒绝

该实验使用同一个成熟 160 epoch fold0 checkpoint 同时作为 student
初始化和冻结 teacher，训练数据仍为无泄漏、仅占 1.978% 的
Background-100MP 增强集。固定 6 epoch、前 10 层冻结、原生特征蒸馏
权重 6.0，lr0=5e-5；未做权重、epoch 或融合扫描。

| 条件 | Recall 差值 | FDR 差值 | 关键粗类变化 |
|---|---:|---:|---|
|Normal|-0.965pp|+0.638pp|Ship Recall -0.522pp/FDR +2.612pp；Vehicle Recall -5.224pp/FDR -4.914pp|
|Hard|-1.715pp|+0.115pp|Ship Recall -1.887pp/FDR +1.619pp；Vehicle Recall -3.261pp|
|Sentinel（Hard 阈值冻结）|-0.457pp|+0.657pp|Ship FDR +2.231pp；Vehicle Recall +0.617pp/FDR +1.069pp|

三域均未通过准入门。原生 score-weighted feature distillation 会保护
teacher 已有高响应位置，其中也包括需要压制的结构化背景虚警，因而与负背景
监督冲突。结论为 `stop_without_parameter_scan`，不继续扫蒸馏权重。

## 20. 本轮最终结论与未评估原型

本轮系统排除了最终分类行 focal/BCE、低强度背景监督、有界最终行、
有界空间分支和成熟 teacher 特征蒸馏。没有任何一项在 Normal/Hard/Sentinel
同时满足 Recall 保护与 FDR 改善，因此无新模块获得 full、Docker 或
正式提交资格。当前可部署主线仍是 full YOLO26-s/Y5 旋转增强、identity
分数链；预测评历史最高 86.2274 与正式隐藏集 72.1331 必须分开报告。

关停前还实现了 `coarse_purity_sqrt` 原始 logits 固定分数变换原型：飞机
严格旁路，Ship/Vehicle 使用细类分数与所属粗类概率纯度的几何平均。
该代码有数值与输出对齐防御，但尚未获得 Hard/Sentinel/Normal 实验结果，
故明确标记为 `implemented_not_evaluated_not_admitted`，不得写入主线配置。
