# 《改进方案13》全量核验、本地复算与下一阶段执行决策

日期：2026-09-01
状态：`complete_local_audit / no_official_submission_authorized`
正式锚点：Attempt 1，submission `3358`，score `72.1331`
机器可读结果：`reports/experiments/improvement_plan13_local_analysis_v1.json`

## 0. 结论先行

《改进方案13》的核心判断——**历史选择长期受 pooled 指标支配，正式优化必须转向三个粗类等权、粗类内细类宏平均和七子分**——成立，而且是当前最重要的纠偏。

但文档把若干“正确方向”写成了“已经足以进入正式提交的路线”，需要收紧：

1. 平台评分器和 Attempt 1 回归测试已经在当前主干完成，不再是缺口；真正未完成的是所有历史消费者的协议迁移。
2. 当前三折 OOF 的 Ship 错误并非以细类错分为主。Ship FP 中 `FP_BG=576/862`，FN 中 `FN_MISS=295/415`；仅做 Ship classification head 不能解决主要损失。
3. 25 细类阈值是必要能力，但本次本地交叉拟合证明，朴素 score-first MacroRisk 会用牺牲 Recall 换 FDR；带 Recall 地板的版本又未能跨折泛化。它目前是研发项目，不是可立即提交的 Attempt 2。
4. D-FINE 确有车辆互补信息，但 `+22 TP/+6 FP` 来自已确认的旧 baseline 覆盖错误，不能引用。可信无偏证据是 `+7TP/+1FP` 或 `+11TP/+2FP`，预期 Recall 增益约 1.7--2.7pp。
5. 文档提出的 Reject + Rescue 与已经失败的“全量乘法 agreement”是不同机制，值得优先做离线 replay；但完整双模型约增加 7 秒，七子分会损失约 1 分，必须做 selective D-FINE 或取得足够质量增益。
6. 85 分目标的量级判断正确：保持 Aircraft 和时延不变，Ship、Vehicle 同时接近 `R=0.96/FDR=0.15` 时复算为 `84.9466`。单纯刚过 FDR 硬门只有约 `73.31`。

下一阶段应采用“先修评价与部署，再做两个有独立因果含义的模块”的顺序：

```text
平台协议全链迁移
  -> 稳健的细类宏风险控制（不得牺牲 Recall）
  -> Vehicle 高阈值拒绝 + D-FINE 只救回
  -> 根据错误审计选择 Ship 背景排序修复或细类修复
  -> 只组合独立通过的模块
```

在这些门禁完成前，不授权新的官方提交。

## 1. 使用的证据和边界

### 1.1 正式隐藏集可确定事实

Attempt 1 的页面/API 聚合结果为：

|粗类|macro Recall|macro FDR|pooled TP/FP/FN|pooled Recall/FDR|
|---|---:|---:|---:|---:|
|Ship|0.874969|0.320177|466/94/154|0.751613 / 0.167857|
|Aircraft|0.967641|0.064691|4800/290/142|0.971267 / 0.056974|
|Vehicle|0.852632|0.325000|81/39/14|0.852632 / 0.325000|

三粗类算术平均硬门：

```text
Recall = 0.898414 -> pass
FDR    = 0.236623 -> fail
time   = 2.473167 -> pass
```

全部目标 pooled 为 `Recall=0.945201/FDR=0.073310`，但不决定平台硬门或总分。

### 1.2 正式隐藏集不可确定事实

页面没有返回 25 个细类的 TP/FP/FN，因此以下内容都不能从一次提交唯一反推：

- 哪一个 Ship 细类造成 0.320177 macro FDR；
- 隐藏集每个细类的数量；
- 某一具体阈值改变在隐藏集上会增加多少 TP/FP；
- Ship 的 macro/pooled 差异究竟由哪些源域或细类共同造成。

所以正式结果只能用于构造压力轴和验证假设，不能被当作隐藏标签拟合。

### 1.3 本地分析输入

本次重算使用：

- `outputs/HERA-GUARD-V4-DFINE-VEHICLE-CV3-20260831/fold_{0,1,2}/instances_val.json`；
- 同目录三折 `y5_predictions.json`；
- 4,481 张唯一 OOF 图、20,933 个 GT、65,301 个低阈值候选；
- 完整 25 类 taxonomy；
- Ship/Aircraft IoU 0.50，Vehicle IoU 0.35；
- 为便于与正式回传对照，评分时暂用正式时延 2.473167 秒。

这些 OOF 来自三折模型，正式提交来自全量训练模型，因此本地绝对分必然更保守；本地结果用于候选排序和错误归因，不作为正式分数预测。

## 2. 《改进方案13》哪些判断成立

### 2.1 72 分不是工程故障

成立。正式 pooled Recall 94.52%、FDR 7.33%，时延正常，说明 Docker、切片、坐标恢复和基本候选链没有崩。主要损失来自宏平均和隐藏域下的类别风险，而不是输出格式错误。

### 2.2 Ship 的 macro 与 pooled 指向不同子问题

成立。正式 Ship：

```text
macro Recall 0.874969 > pooled Recall 0.751613
macro FDR    0.320177 > pooled FDR    0.167857
```

这表示大 support 类更可能拖 pooled Recall，小 support 类更可能拖 macro FDR。统一 Ship 阈值不具备同时修复两个方向的自由度。

但这只能证明需要细类级观察和风险控制，不能直接推出“Ship cls-only 是第一训练路线”。本地错误分解见第 5 节。

### 2.3 Vehicle 是背景排序问题

成立。Vehicle 单细类没有同粗类错分，正式需要从 `81TP/39FP/14FN` 接近：

```text
TP >= 92
FP <= 16
```

即大约 `+11TP/-23FP`。单一阈值很难同时完成，Reject + Rescue 的结构假设合理。

### 2.4 只过 FDR 门槛远不足 85 分

成立，复算如下：

|情形|近似总分|硬门|
|---|---:|---|
|正式 Attempt 1|72.1331|Recall/Time 过，FDR 不过|
|保持 Recall，仅令 Ship/Vehicle FDR 约 0.27/0.265|73.3135|刚过 FDR|
|Ship 0.92/0.18，Vehicle 0.90/0.18|79.4228|通过|
|Ship 0.96/0.15，Vehicle 0.96/0.15|84.9466|通过|

当前分段内，任一粗类 Recall 提高 1pp，平台总分约增加 0.381；Ship/Vehicle 在 FDR>0.2 时，FDR 降低 1pp 只增加约 0.107 分。FDR 低于 0.2 后，1pp 约增加 0.286 分。硬门必须过，但高分仍主要依赖 Recall 和进入低 FDR 区间。

## 3. 文档中的仓库问题：逐项代码核验

### 3.1 已完成，不应重复实现

以下内容已在 commit `5b10e32` 完成：

- `src/rsdet/evaluation/absolute_score.py`；
- `platform_confirmed_score()`；
- Attempt 1 的 72.1331 和硬门回归测试；
- `configs/evaluation/formal_hidden_anchor_v1.json`；
- `scripts/analyze_formal_hidden_distribution.py`；
- `reports/experiments/FORMAL_HIDDEN_DISTRIBUTION_INFERENCE_AND_PROXY_V1_20260901.md`。

因此《改进方案13》所说“公开 main 没有 absolute_score.py”是基于旧快照，当前已经失效。

### 3.2 仍然真实存在的协议漂移

1. `configs/project.yaml` 的注释仍写硬门 pooled，需要改成命名协议，不能继续使用含糊的 `official`。
2. `RankingMetrics.overall_recall/fdr` 是 25 细类等权，即 Ship/Aircraft/Vehicle 权重 4/20/1，不等于平台三粗类 1/3 等权。
3. `src/rsdet/analysis/oof_detection.py::build_threshold_curve()` 的 `overall_*` 和 `official_gate_passed` 仍按所有 TP/FP/FN pooled。
4. `src/rsdet/evaluation/hierarchical_thresholds.py` 仍按单曲线目标 FDR 最大 Recall，且文件说明仍将硬门描述为 pooled。
5. 多个历史脚本直接把 `ranking.overall_*` 命名为 `official_macro_*`，容易继续误用。

结论：评分器已经有了，但训练选择、CV3 汇总、阈值选择和报告消费者尚未全部迁移。E0 不能只改一个函数，必须做调用图审计。

### 3.3 细类阈值尚不能部署

以下正式链只支持 global/coarse：

- `src/rsdet/pipeline/large_image.py::PipelineConfig`；
- `src/rsdet/postprocess/safe_tile_fusion.py`；
- `src/rsdet/submission/competition.py` 配置校验与实例化。

必须增加 `score_threshold_by_fine`，优先级固定为：

```text
fine > coarse > global
```

第一版仍保持“融合前过滤”，不同时改变阈值语义。只有离线、Docker 和正式入口逐框一致，细类阈值才有提交资格。

### 3.4 GitHub README 建议不采纳原文形式

用户已明确要求 GitHub 根 README 隐藏，因此不恢复公开 README。应更新：

- 仓库内部状态索引；
- 实验报告导航；
- 最终源码/报告提交包中的复现说明。

这能满足可复现性，又不违反当前公开仓库策略。

## 4. 本地基线与正式结果的对应关系

统一阈值 0.15 的三折 OOF 重算：

|指标|本地 OOF|正式 Attempt 1|解释|
|---|---:|---:|---|
|Ship macro Recall|0.715918|0.874969|OOF 明显更难|
|Ship macro FDR|0.318093|0.320177|高度接近|
|Aircraft macro Recall|0.909282|0.967641|OOF 更难|
|Aircraft macro FDR|0.119779|0.064691|OOF 更难|
|Vehicle Recall|0.549751|0.852632|OOF 明显更难|
|Vehicle FDR|0.348083|0.325000|接近|
|三粗类平均 Recall|0.724984|0.898414|绝对值不可直接预测|
|三粗类平均 FDR|0.261985|0.236623|风险方向一致|
|七子分代理|62.6797|72.1331|OOF 保守约 9.45 分|

可得出：

1. 当前 OOF 适合淘汰破坏 Recall 的路线；
2. Ship/Vehicle FDR 风险与正式结果方向一致；
3. 不能要求本地绝对 Recall/总分等于正式值；
4. 候选准入应看 paired delta、最坏折和压力集，而非“本地必须达到 85”。

## 5. 本地错误分解：对 Ship/Vehicle 路线的实际影响

统一 0.15 下全部 FP/FN 守恒分解：

|粗类|FP_DUP|FP_CLS|FP_LOC|FP_BG|FN_CLS|FN_LOC|FN_MISS|
|---|---:|---:|---:|---:|---:|---:|---:|
|Ship|166|83|37|576|83|37|295|
|Aircraft|147|868|4|951|868|4|360|
|Vehicle|7|0|9|102|0|9|172|

### 5.1 Ship

Ship FP 构成：

```text
FP_BG  66.8%
FP_DUP 19.3%
FP_CLS  9.6%
FP_LOC  4.3%
```

Ship FN 构成：

```text
FN_MISS 71.1%
FN_CLS  20.0%
FN_LOC   8.9%
```

因此 Ship classification-only 只能直接触及约 20% 的 FN_CLS 和约 10% 的 FP_CLS。它可能改善宏平均尾类，但不是当前 Ship 的唯一或首要表示问题。

Ship 第一阶段应拆成：

1. 细类宏风险控制，解决 HM/LQS 与 QHS/MS 阈值方向冲突；
2. 对 `FP_BG/FN_MISS` 做图像审核和背景域/候选分数审计；
3. 只有当 HM/LQS 的 macro 损失主要确认为同粗类错分，才训练 cls-only/EFL；
4. 如果主要是无候选/低分候选，应做 Ship objectness/quality 分支或 active-FP 背景训练，而不是只改分类头。

### 5.2 Vehicle

Vehicle 的 `FP_BG=102/118`、`FN_MISS=172/181`，完全支持“背景排序 + 低分救回”的问题定义。D-FINE Reject + Rescue 的优先级应高于 Ship cls-only。

### 5.3 Aircraft

Aircraft FP_CLS/FN_CLS 数量大，TU-160 等细类仍是本地宏平均短板；但正式 Aircraft 已达 `R=0.9676/FDR=0.0647`，当前不能为了本地 25 类 overall 主动改 Aircraft，除非模块具备严格零退化旁路。

## 6. MacroRisk 本地实测：方向成立，当前实现不准提交

本次做了两个三折泄漏安全诊断。每个 held-out fold 的 25 类阈值只由另外两折拟合，再应用到 held-out；未使用正式隐藏标签。

### 6.1 Score-first 细类阈值

结果：

|指标|统一 0.15|Score-first cross-fit|变化|
|---|---:|---:|---:|
|代理总分|62.6797|65.9620|+3.2823|
|三粗类平均 Recall|0.724984|0.593437|-13.155pp|
|三粗类平均 FDR|0.261985|0.122157|-13.983pp|
|Recall gate|fail|fail|无改善|
|FDR gate|fail|pass|改善|

它通过大幅升高 Ship/Vehicle 阈值换得 FDR，不能作为提交候选。

### 6.2 每粗类 Recall 训练侧最多下降 0.5pp

结果：

|指标|统一 0.15|Recall-floor cross-fit|变化|
|---|---:|---:|---:|
|代理总分|62.6797|62.1707|-0.5090|
|三粗类平均 Recall|0.724984|0.697893|-2.709pp|
|三粗类平均 FDR|0.261985|0.264048|+0.206pp|

训练侧约束没有跨折泛化，尤其 HM/LQS 阈值在三折间大幅漂移。

### 6.3 决策

《改进方案13》的 Pareto/beam search 只是组合优化的一部分；真正的缺口是统计稳健性。MacroRisk 必须补齐：

- logit 层级收缩到 coarse anchor；
- 最小 support 与阈值跨度限制；
- source-group bootstrap；
- 平均分减方差/最坏折风险；
- held-out coarse Recall 地板；
- 正式漂移压力族；
- fine 阈值 Docker 逐框 parity。

准入线应是 paired cross-fit `P10(Delta score)>0`，且任一粗类 Recall 中位数下降不超过 0.5pp；否则不能使用官方机会验证。

## 7. Ship Fine-Head Repair 的修订版本

### 7.1 保留的内容

- 冻结 backbone/neck/box/DFL，只训练受限分类分支；
- Equalized Focal/EQLv2 比直接套用 Balanced Softmax 更适合 sigmoid dense detector；
- incumbent logit distillation；
- same-coarse wrong-fine margin；
- rare positive 使用 sqrt inverse frequency，cap 4；
- source-group sampler；
- residual 零初始化、只改 Ship 四类。

### 7.2 调整的优先级

它从“最高优先级训练路线”调整为“错误审核后的条件路线”。启动条件：

```text
Ship 尾类 FP_CLS_PAIR 或 FN_CLS_PAIR 对宏平均损失占主导
并且 candidate floor/box quality 已足够
```

若视觉审核显示 HM/LQS 主要是背景误检，第一训练模块应改为 coarse-preserving Ship quality/objectness，而非细类 residual。

### 7.3 NorCal/FRACAL

只保留为历史对照，不重启。Y1-C3 的 FRACAL-inspired 空间分形校准相对 C2 没有独立收益；当前时间窗口不值得做完整 FRACAL 工程复现。NorCal 若复用现有预测只需 CPU，可随 MacroRisk 做一次冻结对照，但不得独立占用正式提交。

## 8. Vehicle D-FINE Reject + Rescue 的证据修正

### 8.1 文档对现有代码的批评成立

`apply_label_agreement()` 对被选类别执行：

```text
new_score = primary_score * specialist_support
```

无支持时 `support=0`，等于赋予 D-FINE 否决 incumbent TP 的权力。这解释了完整 agreement 在 Hard/Sentinel 上 Vehicle Recall 分别下降 9.239pp/8.025pp。

### 8.2 文档引用的收益需要修正

`+22TP/+6FP` 来自已确认的 baseline 覆盖错误。可信证据为：

|合同|结果|可信度|
|---|---:|---|
|严格 cross-fit，增量风险 0.15|+7TP/+1FP|无偏外层|
|严格 cross-fit，增量风险 0.20|+11TP/+2FP|无偏外层|
|训练折 Recall 零损失 guard|+4TP/-22FP|稳健性诊断|
|全 OOF 固定 0.059|+9TP/0FP|有偏超参数候选|

合理期望是 Vehicle Recall +1.7--2.7pp，不是已经证明的 +4--5.5pp。

### 8.3 仍然值得做的原因

Reject + Rescue 不会重评分 incumbent core：

```text
score >= t_high                 -> 原样保留
t_floor <= score < t_high       -> D-FINE 同类 IoU/支持达标才救回
unsupported low-score candidate -> 丢弃
```

这与失败的乘法 agreement 不同，具备新的因果含义。

### 8.4 时延门槛

完整双模型约 9.4 秒，相对正式 2.47 秒使 time 子分下降约 6.93，折算总分约 `-0.99`。在当前 Recall 分段，Vehicle Recall +2.7pp 约增加 1.03 分，几乎刚好抵消完整双模型时延；因此：

- 完整 D-FINE 只作为离线上限；
- 正式候选必须优先 selective tile；
- 或同时显著降低 Vehicle FDR；
- 绝不能只凭 +TP 提交。

### 8.5 第一轮离线实验

冻结 `t_floor/t_high/support_iou/max_tiles`，不扫描融合权重。报告：

- core 保留的 TP/FP；
- raise `t_high` 删除的 TP/FP；
- D-FINE 恢复的 TP/FP；
- selected tile ratio；
- 完整 10K latency；
- Normal/Hard/Sentinel-B 七子分。

准入要求至少：Vehicle Recall +3pp、FDR 不恶化、任一其他粗类 Recall 不下降、计入时延后总分 +1；否则停止。

## 9. 权重汤的实际位置

Greedy model soup 理论成立且不增加推理时延，但当前没有足够的同架构、同初始化、独立通过门禁的 checkpoint：EXT-V、HAD、patch 已被拒绝。

因此 soup 不属于当前第一轮。只有 Ship/Vehicle 新模块产生至少两个相邻低损失且单独通过的 seed/checkpoint 后，才允许：

```text
线性插值检查 -> greedy add -> BN 重校准 -> MacroMirror/Hard/Sentinel-B
```

不混合 EMA/非 EMA，不混不同 head 结构。

## 10. 验证体系的最终形态

### 10.1 Proxy A：MacroMirror-CV3

必须同时输出：

- 25 细类 TP/FP/FN/R/FDR；
- 3 粗类 macro R/FDR；
- 3 粗类等权硬门；
- 7 个绝对子分和总分；
- pooled 仅诊断；
- paired delta、折间方差、最坏折。

正式模式禁止 `require_complete_taxonomy=False`。

### 10.2 Proxy B：Background/Hard-100MP

本次数据核验确认 4,481 张图中零张无标注图，最大图约 1.51MP，无法模拟 10K 大面积背景累积。应建立：

- FP/100MP；
- FP/10K image；
- 每粗类/细类 FP；
- 港口、机场、道路、屋顶、水面等背景域；
- tile 数与 FP 增长曲线；
- 完整标注或人工审核，避免把未标目标当背景。

外部数据当前优先用于背景评价和 active-FP，而不是再做已经失败的普通 coarse pretraining。

### 10.3 Sentinel-B

现有 Sentinel 已多次参与选择，不再承担最终密封作用。Sentinel-B：

- 只比较最终两个候选；
- 不查看逐图预测；
- 不调阈值/epoch/融合；
- source group 与开发集互斥。

### 10.4 Formal-Anchor Shift Family

以 Attempt 1 的粗类聚合差异构造多种压力，而非唯一伪隐藏集：

- TP score 下移；
- FP score 上移；
- 背景面积增加；
- rare fine class 上权；
- source/style 替换。

候选必须在多数合理 shift 下优于 incumbent。

## 11. 现有方向的最终账本

### 11.1 继续

1. 平台协议全链迁移；
2. 稳健 MacroRisk + fine 部署；
3. Vehicle Reject + Rescue；
4. Ship 错误审核后选择 fine-tail 或 objectness；
5. Background-100MP；
6. Sentinel-B；
7. 通过模块的最终组合；
8. 有合格 checkpoint 后的 greedy soup。

### 11.2 条件继续

- Ship cls-only/EFL：仅在 FP_CLS/FN_CLS 尾类证据通过后；
- NorCal：仅 CPU 复用预测的冻结对照；
- conditional residual：cls-only 先通过后；
- full D-FINE：仅作离线上限，不直接部署。

### 11.3 明确停止

- EXT-V 原配方和补 epoch；
- HAD 补 epoch/扫权重；
- D-FINE 全量乘法 agreement；
- full rot90/D4、全量 TTA；
- Y5-L/更大 YOLO；
- pooled/global/coarse 大网格；
- crop/DINO/CleanDIFT/FPN MLP 堆叠；
- background-complete 全模型续训；
- FRACAL-inspired C3 重做；
- partial-label patch 与 EXT-V 2x2 邻域；
- 以 Aircraft 小幅改善宣称整体突破。

## 12. 下一步执行顺序

### Phase A：CPU/代码，立即执行

#### A0 平台协议调用图迁移

新增唯一活动名：

```text
platform_observed_20260831
```

保留 `legacy_pooled`、`v1_6_documented` 仅作历史诊断。修正 project config、threshold curve、hierarchical threshold、CV3 summary、admission、报告脚本的调用。所有正式消费者共享一个 scorer。

验收：Attempt 1 在所有入口均复现 72.1331、Recall pass、FDR fail。

#### A1 fine 阈值部署

实现 `score_threshold_by_fine`，验证：

- config schema；
- safe fusion；
- global/large-image；
- competition entry；
- fine > coarse > global；
- 离线输出与 Docker 逐框、逐分数、逐顺序一致。

#### A2 MacroRisk V2

实现 non-dominated fine curves、粗类联合组合、层级收缩和 group bootstrap。必须同时输出 raw、shrunk、cross-fit、shift-family，不允许只报同折 oracle。

停止条件：若 recalled-floor cross-fit 与 P10 仍不为正，fine threshold 路线不提交。

#### A3 Vehicle Reject + Rescue replay

使用现有 Y5/D-FINE OOF，不重新训练；先做完整双模型上限，再做 selected-tile 近似。禁止 specialist 否决 core。

#### A4 Ship 审核包

从 576 FP_BG、295 FN_MISS、83 FP/FN_CLS 中按 fine class/source/size/score 分层取样，直接判定：背景、漏检、错细类、截断、重复、标注问题。由审核结果选择 S-fine 或 S-objectness。

#### A5 Proxy B 与 Sentinel-B 冻结

完成数据来源、标注完整性、source-disjoint 和 SHA 门禁后再用于候选准入。

### Phase B：GPU，只跑由 Phase A 触发的路线

|GPU|任务|启动条件|
|---|---|---|
|0|Ship 模块 fold0|A4 证明明确瓶颈|
|1|Ship 模块 fold1|同上|
|2|Ship 模块 fold2|同上|
|3|Selective D-FINE 真实 10K 与离线 replay|A3 代码 parity 通过|

如果 A4 指向 objectness，则三张卡训练 bounded Ship quality；若指向细类错分，则训练 cls-only/EFL。不能预先固定为 cls-only。

### Phase C：组合与 full

只组合独立通过的 S 模块和 V 模块。组合后重新执行 MacroMirror、Hard-100MP、Sentinel-B 和 3090 latency；通过后才训练唯一 full 配方并构建 Docker。

## 13. 剩余四次正式机会

当前不能机械照搬原文的 Attempt 2--5，必须由离线门禁触发：

### Attempt 2：最小风险控制候选

候选仍优先是同权重 fine MacroRisk，但只有在：

```text
cross-fit paired score > 0
P10 delta > 0
任一粗类 Recall 中位数下降 <= 0.5pp
Docker parity 完整
```

时才提交。当前朴素版本不满足。

### Attempt 3：单模块最强候选

在 Ship 修复或 Vehicle Reject + Rescue 中选择一个独立证据更强的，不同时叠加两个未经正式验证的变量。

### Attempt 4：互补模块组合

只有 Attempt 3 的正式反馈与本地趋势一致，且另一个模块独立通过，才组合。

### Attempt 5：最终最强或稳健回退

只提交已获得正式/密封证据的最优组合；不用于探索。

## 14. 四卡资源安排的核验

《改进方案13》的总体资源安排合理，但顺序调整为：

```text
0 GPU: A0--A5 完成
GPU0--2: 由 Ship 审核决定的唯一三折训练
GPU3: Vehicle replay / selected-tile / 10K latency
```

MacroRisk、错误分解、D-FINE prediction replay 都不应占用三张训练卡。三个 Ship fold 同时开，只允许一个预注册配方，不同时跑 EFL/EQLv2/残差的参数森林。

## 15. 创新叙事的可保留与不可提前宣称部分

可保留名称 `HERA-Guard MacroShift`，逻辑链成立：

1. Platform-Mirror Macro Risk Evaluation；
2. Coarse-Preserving Fine/Quality Repair；
3. Asymmetric Cross-Architecture Vehicle Rescue；
4. Uncertainty-Shrunk Fine Calibration；
5. Scale-Preserving Safe Large-Image Inference。

但当前已实证的只有第 1 项基础评分器和第 5 项既有大图链；第 2--4 项必须通过本地/密封/正式门禁后才能写成最终方法贡献。否则只能写为探索路线。

## 16. 对《改进方案13》全章节覆盖检查

|原文章节|本报告处理位置|最终判断|
|---|---|---|
|最终判断|0、11、15|方向成立，执行证据需收紧|
|一：72 暴露什么|1、2、4、5|成立，补充本地错误结构|
|二：85 分目标|2.4|复算正确|
|三：评分主干|3|大部分成立，absolute scorer 已完成|
|四：P0 五改动|3、12.A0/A1|评分回归已完成，其余迁移待做；README 改内部索引|
|五：本地验证|4、5、10|全部保留并具体化|
|六：MacroRisk|6、12.A2|方向保留，当前交叉拟合未准入|
|七：Ship 修复|5.1、7|从必做改为错误证据触发|
|八：Vehicle D-FINE|8、12.A3|机制保留，收益证据下修|
|九：权重汤|9|延后至存在合格成员|
|十：4 GPU|14|保留框架，先完成 CPU 阶段|
|十一：E0--E8|12、17|重排为 A0--A5/S/V/C|
|十二：四次提交|13|保留单因素原则，取消“立即 MacroRisk”|
|十三：停止路线|11.3|完整保留并补充已拒绝项|
|十四：创新性|15|叙事可用，贡献需实验兑现|
|立即执行顺序|12|形成门禁化执行顺序|

没有遗漏原文提出的评分、部署、错误审计、bootstrap、背景压力、Sentinel、漂移族、Ship、Vehicle、soup、GPU、实验矩阵、提交机会、停止路线和创新叙事。

## 17. 最终实验矩阵

|ID|唯一变化|数据/算力|核心判据|结果动作|
|---|---|---|---|---|
|A0|平台协议全链迁移|CPU|Attempt1 全入口精确回归|失败即阻断全部|
|A1|fine 阈值部署|CPU/Docker|逐框 parity|失败不做 MacroRisk 提交|
|A2|shrunk MacroRisk|CPU|P10 delta>0、Recall 地板|通过才成为 Attempt2|
|A3|Vehicle Reject+Rescue|CPU replay|+TP/-FP 账本、core 零损失|通过转 selective|
|A4|Selective D-FINE|GPU latency|含时延总分 +>=1|通过成为单模块候选|
|A5|Ship error review|人工/CPU|确定 fine vs objectness 主因|只启动一个训练方向|
|S1|Ship bounded repair CV3|3 GPU|Ship R/FDR 同向、其他类零退化|通过扩 full|
|B1|Background-100MP|CPU/GPU infer|FP/100MP、域稳定|作为外层门禁|
|SB|Sentinel-B|密封|冻结阈值同向|只用于最后两候选|
|C1|S+V 组合|GPU|全部外层同向|通过训练唯一 full|
|C2|greedy soup|CPU/GPU infer|有>=2合格成员、P10>0|可选最后增益|

## 18. 当前唯一建议

立即做 A0--A5，不立即提交、不立即开 Ship cls-only 三折。最先可能形成真实跨档增益的模块是：

```text
Vehicle high-threshold reject
+ D-FINE low-score rescue
+ selective tile execution
```

最先可能形成低风险正式候选的模块是：

```text
完成层级收缩和组稳健验证后的 fine MacroRisk
```

但本次本地实测已经证明，当前朴素 fine threshold 版本不合格。下一次官方机会必须留给“跨折、压力集、Docker 三者同时通过”的候选，而不是把理论合理性当作实证收益。

## 19. 证据与实现索引

|用途|文件|
|---|---|
|正式回传、隐藏粗类分布与代理设计|`reports/experiments/FORMAL_HIDDEN_DISTRIBUTION_INFERENCE_AND_PROXY_V1_20260901.md`|
|正式锚点机器配置|`configs/evaluation/formal_hidden_anchor_v1.json`|
|正式分数复算脚本|`scripts/analyze_formal_hidden_distribution.py`|
|七子分与硬门实现|`src/rsdet/evaluation/absolute_score.py`|
|25 细类/粗类宏评估|`src/rsdet/evaluation/official_metric.py`|
|当前 pooled threshold curve|`src/rsdet/analysis/oof_detection.py`|
|现有层级阈值与 logit shrink|`src/rsdet/evaluation/hierarchical_thresholds.py`|
|错误分解实现|`src/rsdet/analysis/oof_detection.py::decompose_official_errors`|
|错误聚合入口|`scripts/evaluate_experiment.py`、`scripts/analyze_experiment_errors.py`|
|当前 D-FINE 乘法 agreement|`src/rsdet/submission/agreement.py`|
|D-FINE 可信证据修正|`reports/experiments/HERA_GUARD_FINAL_PLAN11_RECONCILIATION_20260831.md`|
|D-FINE/HAD/EXT-V/patch 最终拒绝|`reports/experiments/HERA_GUARD_FINAL_PREFLIGHT_EXECUTION_20260831.md`|
|大图阈值配置|`src/rsdet/pipeline/large_image.py`|
|Safe Fusion 阈值过滤|`src/rsdet/postprocess/safe_tile_fusion.py`|
|Docker 配置和执行入口|`src/rsdet/submission/competition.py`|
|本次 OOF GT/预测|`outputs/HERA-GUARD-V4-DFINE-VEHICLE-CV3-20260831/fold_{0,1,2}`|
|本报告机器结果|`reports/experiments/improvement_plan13_local_analysis_v1.json`|

本索引使用仓库内相对路径，不移动大型预测、checkpoint 或数据资产。
