# 正式隐藏集分布反推与本地代理评测 V1

日期：2026-09-01
状态：`coarse_distribution_identified / fine_distribution_under-determined / proxy_v1_specified`

## 1. 结论先行

可以反推官方隐藏集的一部分分布，但不能从一次提交唯一恢复整个测试集。

能够确定的事实：

- 正式隐藏集有 **5,657 个 GT 对象**：舰船 620、飞机 4,942、车辆 95；
- 三类占比为 10.960% / 87.361% / 1.679%；
- 本次模型的三类 TP/FP/FN、pooled 指标和细类 macro 指标；
- 官方当前确定的七项计分法与三粗类硬门汇总方式；
- 当前隐藏域的主要难点为舰船细类 macro FDR 和车辆 Recall/FDR，而不是粗类数量比例。

无法唯一确定的信息：

- 4 个舰船细类、20 个飞机细类各自有多少 GT；
- 测试图数、每图密度、空背景图数和负样本总量；
- 来源域、机场/场景、目标尺寸、分辨率和难负样本类型的真实分布。

因此，本地不应伪造一个“唯一的官方复刻集”，而应冻结两套测试：

1. **Formal-Anchor Proxy**：用已知的官方数量、当前 incumbent 难度和多个细类先验构造的组级重采样压力测试；
2. **Frozen Source-Disjoint Sentinel**：不参与任何阈值或混合权重拟合，只防止对正式锚点过拟合。

Normal-CV3 仍保留为开发诊断，但不再单独决定正式提交。

## 2. 官方隐藏集可识别的数量分布

### 2.1 粗类 GT 和训练集对照

| 粗类 | 训练 GT | 训练占比 | 正式 GT | 正式占比 | 正式/训练占比比值 |
|---|---:|---:|---:|---:|---:|
| ship | 2,682 | 12.812% | 620 | 10.960% | 0.855 |
| aircraft | 17,849 | 85.267% | 4,942 | 87.361% | 1.025 |
| vehicle | 402 | 1.920% | 95 | 1.679% | 0.874 |
| 合计 | 20,933 | 100% | 5,657 | 100% | - |

这个结果否决了一个容易产生的误判：正式分数大幅下降不是因为官方突然将车辆或舰船占比放大。三粗类比例与训练集很接近，甚至与按训练分布进行约 1/4 至 1/3 抽样是相容的。这只是“相容”，不能证明官方真的从训练集同分布抽样。

以训练集每张正图平均对象数做弱先验，对应约 310 张舰船正图、851 张飞机正图和 16 张车辆正图。这个估计不包含空背景图，也不是可识别真值，不用作硬门。

### 2.2 细类 macro 与 pooled 计数的差异

| 粗类 | 官方 macro R | pooled R | 差值 | 官方 macro FDR | pooled FDR | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| ship | 0.874969 | 0.751613 | +0.123356 | 0.320177 | 0.167857 | +0.152320 |
| aircraft | 0.967641 | 0.971267 | -0.003626 | 0.064691 | 0.056974 | +0.007717 |
| vehicle | 0.852632 | 0.852632 | 0 | 0.325000 | 0.325000 | 0 |

舰船是最重要的证据。其 pooled Recall 只有 75.16%，但 4 细类等权 macro Recall 为 87.50%；pooled FDR 为 16.79%，macro FDR 却为 32.02%。这与“大 support 类承担了大量 FN，小 support 类的少量 FP 又被等权放大”高度一致。所以：

- 不能再用 pooled TP/FP 优先的前沿选正式候选；
- HM/LQS 即使只有少量预测，也必须在每个本地正式代理集中出现；
- 阈值或复核器应直接优化 7 子分，而不是对 5,657 个对象做一次 pooled 计分。

飞机的 macro/pooled 差异很小，说明 20 细类在这次模型上的总体难度更均匀。车辆只有 FSC 一个细类，macro 与 pooled 天然相同，正式下降是真实域移而非汇总幻象。

## 3. 为什么不能继续“反解”出唯一的 25 类分布

对一个含 `K` 个细类的粗类，未知量至少包含每细类 `GT_i/TP_i/FP_i`，共 `3K` 个数。官方只给出：

- `sum(GT_i)`；
- `sum(TP_i)`；
- `sum(FP_i)`；
- `mean(TP_i / GT_i)`；
- `mean(FP_i / (TP_i + FP_i))`。

舰船是 12 个未知量、5 类约束；飞机是 60 个未知量、5 类约束。即使再加入整数和上下界，仍有大量解。任何一套精确到 25 类的“官方分布”都只能是先验猜测。

为便于构造代理集，机器产物给出了一套训练比例最大余数分配。其中舰船为 HM 4、LQS 7、QHS 148、MS 461；这些数只是 `train_proportional` 场景，绝不标记为隐藏真值。完整 25 类分配见机器产物。

## 4. 本地评测为什么会虚高

### 4.1 聚合目标曾经不完全对齐

`official_metric.py` 能够正确输出舰船4类、飞机20类、车辆1类的 coarse macro，但其历史 `overall_recall/overall_fdr` 是对全部 25 细类再等权。这不等于当前平台的三粗类算术平均，更不等于七子分。因为飞机有 20 个细类，历史 overall 实际上会把飞机放大到 80% 权重，而官方七项中飞机只占 2/7。

已新增 `platform_confirmed_score()` 固化官方七项、三粗类硬门和 pooled-count 诊断隔离。正式 Attempt 1 回归测试精确复现 72.1331，并得到 Recall 门通过、FDR 门失败、时延门通过。

### 4.2 Normal-CV3 与正式域的负样本难度不同

粗类数量比例已经很接近，但从预测试 Safe 到正式 Attempt 1，舰船 Recall/FDR 变化为 -6.73pp/+19.32pp，车辆为 -9.37pp/+8.72pp。时延反而改善。这更符合来源域、分辨率、背景结构、密度与细类不平衡导致的域移，不是 Docker 或类别数量问题。

### 4.3 Hard10K 不是独立隐藏集

项目文档已明确标记 Pseudo-10K 是 deployment proxy，不是独立 benchmark。它可以检验切片、融合和大图路径，但不能单独预测新来源的 FDR。

## 5. 冻结的两套本地测试

### 5.1 Test A：Formal-Anchor Proxy V1（主排名预测）

#### 数据单位

- 先按 source/group 采样，再使用组内整图；
- 飞机使用 `data/splits/cv3_airport_proxy_k60_v2_groups.json`；
- 舰船和车辆使用现有 CV3 source/group 字段，没有可信组时至少按原图而非框采样；
- 不允许独立抽框，否则会破坏每图密度、FP 共现和背景相关性。

#### 硬数量目标

- 每个 bootstrap 目标舰船 620、飞机 4,942、车辆 95 个 GT；
- 因为采样单位是整图/整组，允许小范围超出，但计分前不丢框；
- 每个粗类的细类 taxonomy 必须完整，稀有类不得因某次采样消失。

#### 三个细类混合场景

1. `train_proportional`：训练分布先验；
2. `fine_balanced`：细类接近等量，直接压力测官方 macro；
3. `rare_upweighted`：确保 HM/LQS/TU-160/KC-10 等小 support 类不被 pooled 数量淹没。

三个场景同时报告，候选按最差场景准入，不选择对自己最有利的场景。

#### 难度锚定与防泄漏

Attempt 1 incumbent 可以一次性用来冻结代理集难度：在组级候选中选择一个混合，使 incumbent 的舰船、飞机、车辆 R/FDR 接近正式锚点。这不是对新模型调阈值，而是用一个已结束的基线测量代理集的域难度。

冻结后必须记录 group/image ID、标签、预测、随机种子和 SHA；后续候选不允许改变代理集组成。所有阈值只在 nested development groups 上选择，在 untouched confirmation groups 上报告。

#### 重复与准入

- 至少 200 次 group bootstrap；
- 报告分数中位数、P10、三门全通过比例和每粗类区间；
- 准入安全裕量：三粗类平均 Recall >= 0.87、FDR <= 0.18、时延 <= 15 s；
- 三门全通过 bootstrap 比例 >= 90%；
- 新候选对 incumbent 的 P10 绝对分不降，任一粗类 Recall 下降不超过 0.5pp。

### 5.2 Test B：Frozen Source-Disjoint Sentinel（泛化保底）

- 使用现有 source-disjoint 组和 Hard 选定的冻结阈值；
- Sentinel 自身不允许选阈值、混合权重、NMS 或路由规则；
- 按相同七项官方绝对分和三粗类硬门评估；
- 它不需要数值完全拟合 Attempt 1，职责是拒绝“只对锚点有效”的方法。

## 6. 本地评测的最终决策顺序

1. 所有方法先在 Normal-CV3 做快速配对消融；
2. 有方向的候选进入 Formal-Anchor Proxy，阈值只在 nested dev 选择；
3. 在 untouched anchor confirmation 和 Frozen Sentinel 上双重验收；
4. 用 3090 实测时延代入第七子分；
5. 只提交三门有安全裕量、P10 分数正增益且 Sentinel 同方向的候选。

这会将“本地 94，线上 72”的问题拆成两部分：一部分是计分口径偏差，已经用精确七项回归消除；另一部分是真实域移，由 anchor-calibrated 压力集和 source-disjoint Sentinel 联合约束。

## 7. 产物索引

- 正式锚点：`configs/evaluation/formal_hidden_anchor_v1.json`；
- 粗类/细类先验分析器：`scripts/analyze_formal_hidden_distribution.py`；
- 机器分析结果：`reports/experiments/formal_hidden_distribution_inference_v1.json`；
- 官方七项评分器：`src/rsdet/evaluation/absolute_score.py::platform_confirmed_score`；
- 正式 72.1331 回归测试：`tests/test_absolute_score.py`；
- 训练集类统计：`../dataset_audit/shareable/tables/class_statistics.csv`；
- CV3 分组：`data/splits/cv3_airport_proxy_k60_v2.json`；
- 飞机机场代理组：`data/splits/cv3_airport_proxy_k60_v2_groups.json`；
- 正式 Attempt 1 原结论：`reports/experiments/FORMAL_ATTEMPT1_ABSOLUTE_SCORE_FREEZE_20260831.md`。

## 8. 对后续官方提交的更新规则

每获得一次新的正式结果，只做三件事：

1. 用冻结的代理集事先预测与官方实测的差值评估代理误差；
2. 更新不确定性范围和方法间排序相关性；
3. 只在有系统性偏差时发布新的 proxy 版本，保留旧版本，不追每一次提交改集合。

当前只有一个正式锚点，无法估计“本地方法排序—官方方法排序”的稳定相关性。在第二个真正不同的正式候选返回前，任何精确的线下到线上分数映射都是过度拟合。
