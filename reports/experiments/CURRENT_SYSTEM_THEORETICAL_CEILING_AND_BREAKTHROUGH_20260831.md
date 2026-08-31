# 当前系统理论上限、差距与突破路线（2026-08-31）

状态：`evidence_synthesis_complete / next_experiment_not_yet_admitted`

本文不把“理论上限”伪造成一个可精确计算的综合分。官方综合分来自七项相对排名，取决于全部队伍，不是我们六个质量数和时延的公开函数。本文将上限分成指标上限、候选上限、排序上限、跨域上限和部署上限，只使用已完成实验的可核查证据。

## 1. 当前官方锚点及真正的短板

`trial-v2.0` 仍是唯一 incumbent：

| 项目 | Recall | FDR | 距离理想 Recall=1 | 距离理想 FDR=0 |
|---|---:|---:|---:|---:|
| ship | 0.942287 | 0.126937 | 5.7713pp | 12.6937pp |
| aircraft | 0.999246 | 0.024300 | 0.0754pp | 2.4300pp |
| vehicle | 0.946309 | 0.237838 | 5.3691pp | **23.7838pp** |

平均时延为 2.704833s，综合分 86.2274。这个表直接给出三个结论：

1. aircraft Recall 已经饱和，再增加视图或模型所能换取的排名收益极小；
2. vehicle FDR 是绝对数值上最大的缺口，但不能像 v3 那样用损失 Recall 来换；
3. ship 同时有 Recall 和 FDR 缺口，是第二个必须从表示层解决的粗类。

官方 V1.6 排名又不只看平台展示的粗类 pooled 数值：船 4 个细类、飞机 20 个细类在粗类内等权平均。因此 LQS/HM 之类小 support 船类以及 TU-160/F-22 等尾类，可以在 pooled 数值几乎不变时显著影响相对排名。所以“飞机平台 Recall 接近 1”不等于 20 个飞机细类都已解决。

## 2. 上限分解

### 2.1 候选形成上限并不低

在 Hard10K 的四源候选池中：

- GT 共 2,158；
- candidate oracle 为 2,100 / 2,158 = **97.31%**；
- source-disjoint sentinel 的 oracle 为 1,933 / 1,969 = **98.17%**；
- sentinel 中只有 6 个细类失败和 30 个定位失败。

这不证明当前单个 Y5 已有 98% 的部署召回上限；它证明现有异构模型/视图中已经存在大量互补真信号。真正未解的是：怎样在不引入海量背景框和时延的情况下，将这些信号压缩进单模型。

### 2.2 固定风险排序是首要瓶颈

同一 Hard10K 候选池中，在 FDR 约 0.15 时只保留了 1,863 个 TP：

- 237 个已经“可被某个候选命中”的 GT 因排序而丢失；
- 58 个 GT 是真正的候选/定位缺失。

换言之，该工作点的 295 个 FN 中约 **80.3% 是排序缺口**，约 19.7% 是候选缺口。另一份工作点错误分解也一致：

- 322 个 FP 中 FP_BG = 248，占 **77.0%**；
- 302 个 FN 中 FN_MISS = 275，占 **91.1%**；
- FN_CLS + FN_LOC 只有 27，即便全部修好，在 2,158 GT 上也只约等于 **1.25pp** Recall 上限。

这里的 `FN_MISS` 多数不是模型在 score floor 下完全没有框，而是在实用阈值上被混在背景候选之下。因此主要矛盾可以更准确地写成：

> 低分真实小/微小 ship、vehicle 与大量高分结构化背景相互穿插，导致降阈值恢复 Recall 时 FDR 更快爆炸。

### 2.3 当前候选到排序的可见缺口约为 11--13pp

四源 oracle 为 97.31%，source-disjoint oracle 为 98.17%；相应的低风险外层结果只有约 85--86% Recall。在完全相同的候选前提下，可见的“有框但不能安全排上去”缺口约为 **11--13pp**。这个数是当前方法家族最重要的理论改进空间，但它不能被解读为“再训一个 MLP 就能增长 12pp”。

### 2.4 域外迁移上限低于同域 oracle

D-FINE vehicle specialist 在折 1 上可以比 Y5 提高 29.10pp vehicle Recall，但折 2 反而下降 3.70pp。这说明异构表示有真信号，但它的原始分数刻度不能在域间平移。任何只在两折上学一个全局阈值、温度或融合权重的方法，上限都会被这种 source/style shift 限制。

### 2.5 部署上限受第七项排名约束

v3 把时延从 2.704833s 提高到 4.888833s（+80.74%），虽然四个展示质量项改善，综合分仍下降 1.2256。这已经实证：“候选 oracle 更高”不等于“比赛系统更强”。全量双视图、全量异构双检测器和原始无条件并集即使提升 Recall，也会同时损失 FDR 和时延排名。

## 3. 为什么已尝试的方法开始平台化

### 3.1 相同 proposal 上的轻量特征已接近边际上限

crop-only 相对 detector 有 +0.678pp Recall 的 Normal-CV3 增益，但主要来自官方已饱和的 aircraft。随后在同一 65,301 proposals 上的真三类 open-set、group-balanced ERM、GroupDRO、vehicle-only 224/336 和旧 F1 logit 组合均未超过 +0.5pp 门。其中三类 open-set 的 vehicle Recall 增长 1.99pp，但 vehicle FDR 同时恶化。

这不是训练轮数不够，而是同一个 ConvNeXt crop 表示和同一批 Y5 proposals 中的可分信息已经被提取得比较充分。继续叠特征、改 sampler 或扫 hidden dim 不会产生跨档结果。

### 3.2 监督目标与比赛目标不一致

YOLO 的标准训练优化分类/回归/AP，而比赛实际要求的是：

- 细类内 prediction-first 唯一匹配；
- ship/aircraft 的 IoU 0.50 与 vehicle 的 IoU 0.35；
- 每个粗类在低 FDR 条件下的排序；
- 25 细类内等权的 macro 排名；
- 同一目标多框不能重复计分。

已做的 BCE、RankNet、soft-FDR、one-winner 轻量头最多只有千分点级改善，原因是它们在固定表示上改目标，没有改变 TP/FP 的底层可分性。

### 3.3 训练数据不包含足够的结构化负样本分布

正式 4,481 图以目标图为主。Hard10K 的 31,003 个背景负候选和 FP_BG 集中于少数结构化场景，这种负分布在常规训练中没有充分覆盖。旧 hard replay 又因为数量少、来源偏、与当前 detector 错误高度相关，在压 FP 时破坏了候选形成。这解释了“背景问题很大”和“直接困难负样本微调反而变差”为什么能同时成立。

### 3.4 现有 group_id 解决泄漏，不等于解决风险优化

airport proxy group 对划分与泄漏控制是有价值的，但 crop-only GroupDRO 和 balanced sampling 的负向结果表明：机场组不是 proposal 真假难度的最佳优化组。需要区分：

- leakage group：防止同域跨折；
- style/risk group：描述传感器、背景结构、尺度与候选风险。

前者已经可用，后者尚未被建立成稳定的训练条件。

## 4. 新的异构一致性上限诊断

为区分“D-FINE 完全无用”与“D-FINE 不能直接替换”，本次在已完成的三折 Y5/D-FINE OOF 预测上做了只读诊断。规则不新增 D-FINE 框：

1. 只取 Y5 的 vehicle 候选；
2. 只看低于 Y5 外层 FDR15 阈值的候选；
3. 对每个 Y5 框找 IoU >= 0.35 的最高 D-FINE vehicle score；
4. 一致性分数固定为 `Y5 score * D-FINE support score`；
5. 用其他两折选增量风险阈值，只在留出折计数。

早期一次性诊断曾得到 16/4 与 22/6；在冻结实现、SHA 和两折选阈值合同后，
正式结果如下（早期数字不再作为正式结论引用）：

| held-out fold | 新增 TP | 新增 FP | 边际 FDR |
|---:|---:|---:|---:|
| 0 | 5 | 0 | 0.000 |
| 1 | 2 | 0 | 0.000 |
| 2 | 4 | 2 | 0.333 |
| pooled | **11** | **2** | **0.154** |

对 402 个 CV3 vehicle GT，11 TP 等于 **vehicle Recall +2.74pp**。在基线
164 TP / 30 FP 上，产品恢复得到 175 TP / 32 FP，vehicle FDR 由 15.46%
变为 15.46%（差异小于 0.01pp）。更保守的增量 FDR <= 0.15 为 7 TP / 1 FP，
vehicle Recall +1.74pp，FDR 反而下降 0.12pp。进一步的完整产品重排在同一
FDR15 下达到 198 TP / 36 FP，即 vehicle Recall +8.46pp、FDR -0.08pp；
详见 `HERA_GUARD_V6_DFINE_AGREEMENT_EXECUTION_20260831.md`。

同时必须记录反证：

- 单用 D-FINE support score 的低分增量 FDR 约 72%--92%，不可用；
- 全量 D-FINE vehicle 替换存在折 2 负增益，不可用；
- 同样的 same-fine 乘积规则用于 ship 时几乎无可交叉拟合的低风险增量，不应扩展到 ship。

因此新结论不是重启“D-FINE specialist 替换”，而是：

> D-FINE 提供了可以在三折出现的 vehicle 异构一致性信号；最终形态应该是将该信号蒸馏回单个 Y5-compatible 表示，而不是在 Docker 内长期运行两个检测器。

本节仍是诊断而不是科学准入结论；它使用了已有 OOF 标签进行外层计数，后续必须固化为独立脚本、SHA 和预注册合同后再执行正式试验。

## 5. 接下来应该在哪里提升

### P0：先把评估目标固定为“五个弱排名维度”

方法选择不再以 pooled Recall 单值为主，而是同时检查：

- ship Recall / FDR；
- vehicle Recall / FDR；
- latency；
- aircraft 两项作为不可退化的 guardrail；
- 25 细类 macro 必须与 pooled 并列。

一个方法即使 pooled +0.7pp，如果收益主要来自已饱和 aircraft，也不是强候选。

### P1：车辆异构教师蒸馏（当前最高信息密度）

目标不是全量并集，而是让单个 Y5 学到“哪些低分 vehicle 候选也被 D-FINE 以正确几何支持”：

1. 固化三折交叉拟合的 agreement audit，不扫融合权重；
2. 用 OOF D-FINE 产生 `same-fine IoU / support score / agreement target`；
3. 候选仍只由 Y5 产生，teacher 不创建新框；
4. 训练一个极小的 vehicle agreement/objectness 分支，或将 teacher 软排序蒸馏到 Y5 的 vehicle score；
5. ship/aircraft 严格旁路；
6. 正式门为三折 vehicle Recall 均不下降、pooled vehicle Recall 增益 >= 2pp、vehicle FDR 恶化 <= 1pp，然后再过 Hard/sentinel。

这条路与已失败的 FPN-Q1/Q2 不同：Q1/Q2 用当前 detector 自己的标签在高维离线特征上重排；新路线的监督来自异构 detector 中已被实证的 vehicle 互补信号，最终要蒸馏到单检测器。

### P2：遥感外部粗类/objectness 预训练（最可能改变表示上限）

使用来源和许可合规的 FAIR1M、RarePlanes、xView、AI-TOD、DOTA 等遥感数据时，只做
`ship / aircraft / vehicle / other_remote_object`粗类和 objectness 预训练，然后重建官方 25 类头并在正式数据上微调。

它解决的是两个根因：

- 见到更多小/微小遥感对象尺度与形态；
- 见到更多结构化背景与 `other_remote_object`，而不是只在本数据上追加与现 detector 强相关的错误。

先做 fold0 40 epoch 的候选地板 + Hard 风险快筛；只有候选地板不下降、ship/vehicle 固定风险明显改善，才扩 CV3/full。不允许把外部细类伪装成官方型号标签。

### P3：ship 的独立表示与尾类路线

D-FINE agreement 对 ship 的诊断为负，所以不应生搬 vehicle 路线。ship 优先从外部舰船数据的 objectness/形态预训练获取新表示，再用正式 4 细类微调。选择模型时必须看 4 细类 macro，特别是 LQS/HM，不能让 MS 样本量掩盖小类。

如果外部数据暂时不可用，次选是训练第二个同架构 Y5 随机种子，只用 same-fine 一致性作 ship 质量支持，先证明三折质量收益再考虑蒸馏；不直接部署双模型。

### P4：只在质量候选锁定后做时延优化

时延是七项之一，不是“小于 20 秒就不用管”。对最终单视图单模型候选依次验证：

- FP16 / 静态 shape / batch 与 I/O 流水；
- ONNX/TensorRT 等价输出；
- 可重参数化结构或 teacher-to-student 蒸馏；
- RTX 3090 等价硬件的真实 10K 测速。

必须先保证结果 JSON 和所有框在容差内等价，不能为了时延排名改变候选语义。

## 6. 不再建议消耗资源的路线

1. 阈值、NMS、融合权重的细网格搜索；
2. 全量 rot90/D4/高分辨率 TTA；
3. 在同一 65,301 proposals 上继续叠 crop/DINO/CleanDIFT/FPN 特征；
4. 直接部署 D-FINE 替换 vehicle 或无条件并集；
5. 用机场 leakage group 反复做 GroupDRO/sampler 扫描；
6. 继续放大 Y5-S 到 Y5-L，或回到 M3/DEIM 单模型；
7. 把主要资源放在细类改错或框坐标微调；
8. 以 pooled Recall 改善、飞机改善或单折改善宣布正向。

## 7. 现实目标与准入判定

因为综合分是相对排名，不存在“某组六指标数学上保证 93 分”的表。但要从 86.2 进入显著更高的排名区间，新系统至少应该朝下列 Pareto 目标靠近：

| 粗类 | Recall 目标 | FDR 目标 |
|---|---:|---:|
| ship | >= 0.96 | <= 0.08--0.10 |
| aircraft | >= 0.995 | <= 0.03 |
| vehicle | >= 0.96 | <= 0.12--0.15 |

时延应低于当前 2.70s 或至少不明显上升。这些是工程目标和参赛策略，不是对官方隐藏集的保证。

下一轮正式方法只有同时满足以下条件才能替换 v2：

- Normal-CV3 pooled Recall 不下降超过 0.3pp；
- 任一粗类 Recall 不下降超过 0.5pp；
- Hard10K@FDR15 Recall 提高至少 0.5pp；
- source-disjoint sentinel 同方向；
- ship 和 vehicle 不得以明显 Recall/FDR 交换伪装成升级；
- 25 细类 macro 和尾类不退化；
- Docker 形态不依赖第二个全量 detector，或其质量收益足以覆盖时延排名代价。

## 8. 最终判断

当前系统并没有接近“无信号可用”的理论上限；它接近的是“固定 Y5 proposals + 轻量离线重排”这一方法家族的上限。当前最大缺口不是再调一个阈值，而是：

1. 让单个部署模型学到异构 detector 中已经存在的 vehicle 互补信号；
2. 用更广泛遥感数据改变 ship/vehicle 对结构化背景的表示边界；
3. 在细类 macro 口径下专门保护小 support 类；
4. 把新信号压缩回单视图、单模型、2.7s 级或更快的部署链。

按信息价值排序，下一步是 **vehicle 异构一致性正式审计→蒸馏**，与 **外部遥感 coarse/objectness 预训练** 并行；ship 不应跟随 D-FINE 路线，而应跟随外部表示和细类 macro 路线。

## 9. 证据索引

- `reports/experiments/FORMAL_FIVE_SUBMISSION_AND_LOCAL_TOURNAMENT_20260831.md`
- `reports/experiments/OFFICIAL_TRIAL_V1_V3_DEEP_ANALYSIS_AND_GPT_HANDOFF_20260830.md`
- `reports/experiments/HERA_GUARD_V3_METRIC_ALIGNED_EXECUTION_20260830.md`
- `reports/experiments/HERA_GUARD_V4_CHAMPIONSHIP_EXECUTION_20260830.md`
- `reports/experiments/HERA_GUARD_V5_REMAINING_DIRECTIONS_AUDIT_20260831.md`
- `outputs/HERA-GUARD-V4-DFINE-VEHICLE-CV3-20260831/crossfit_route.json`
- `outputs/HERA-GUARD-V4-DFINE-VEHICLE-CV3-20260831/fold_*/{y5_predictions,dfine_predictions,instances_val}.json`
- `outputs/HERA-GUARD-V3-20260830/SENTINEL/`
