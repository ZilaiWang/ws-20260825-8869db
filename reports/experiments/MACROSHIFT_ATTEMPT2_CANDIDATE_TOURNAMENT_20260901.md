# MacroShift Attempt-2 候选锦标赛与提交前结论（2026-09-01）

## 1. 任务边界

本轮目标是完成方案 13 之后仍未读出的低成本候选，并用唯一正式协议
`platform_observed_20260831` 判断是否存在值得占用第二次正式提交机会的模块。本轮未构建
Docker、未打包权重、未 push、未触发官方提交。只有在用户另行明确授权后才允许进入这些
步骤。

不可变正式锚点为第一次正式提交：总分 `72.1331`，Ship R/FDR
`0.874969/0.320177`，Aircraft `0.967641/0.064691`，Vehicle
`0.852632/0.325000`，3090 平均时延 `2.473167s`。三个 Recall 的宏均值通过
`0.85` 门槛，三个 FDR 的宏均值约 `0.236623`，未通过 `0.20` 门槛。因此第二次提交
首先必须降低 Ship/Vehicle FDR，同时不能用明显 Recall 损失换取表面分数。

## 2. 协议和实现收口

- 正式协议注册表扩展到 17 个入口，审计状态 `pass`；
- 新增严格 coarse identity-bypass 的 Normal nested-CV3 选择器；
- 新增冻结路由在 Hard/Sentinel 上的无标签推理与一次性评估器；
- 新增 Normal 训练折、每折/粗类、257 固定分位点的标签无关单调校准；
- 相关单元测试与评估测试共 17 项通过，ruff 通过；
- 所有压力集标签只用于最终计分，不参与质量模型、路由或校准器拟合。

一个重要资产血缘问题已被发现并纠正：当前 46,566 行 Hard 候选对应
`pseudo10k-trial-mix-local/ground_truth.json`（6 图、2,158 GT、SHA 前缀
`e02e139e5b7f`），不能与早期 `CV3-OOF-PSEUDO10K-V1` 的 2,875-GT 账本混用；
Sentinel 对应 6 图、1,969 GT、SHA 前缀 `eb1f8850624d`。错配账本产生的零召回结果
全部标为无效技术诊断，不进入科学结论。

## 3. 有限阈值迁移：正式拒绝

只评估了 7 个预注册固定策略：identity、Vehicle `0.18/0.20/0.22`，以及三组
Ship+Vehicle 联合阈值。每折 OOF 增量先加到不可变正式锚点，再用官方七子分公式重算。

- Vehicle `0.18`：中位分数增量约 `+0.341`，最坏折 `-0.145`，Recall 最大下降
  `2.26pp`；
- Vehicle `0.20`：中位 `+0.443`、最坏 `+0.118`，Recall 最大下降 `3.76pp`；
- Vehicle `0.22`：中位 `+0.468`、最坏 `+0.413`，Recall 最大下降 `5.26pp`；
- 所有带 Ship 提阈值策略的中位/最坏增量均为负，且 Recall 损失超过 `5pp`；
- 没有策略能把正式投影 FDR 宏门槛压到 `<=0.20` 并同时满足 Recall 保护。

结论：阈值微调不是第二次提交候选。

## 4. coarse score-sqrt：正式拒绝

遗留的 9 个 coarse-binary hard-score checkpoint 已完整复跑。最初读出还发现了错误 GT
引用，纠正到 2,158-GT Hard 账本后，FDR15 结果仍只有 Recall `0.090825`，而冻结基线
为 `0.849861`。它把粗类 hard-negative 概率直接传播到细类候选后，严重破坏细类内部
排序。该路线不是轻微负向消融，而是数量级失败，永久停止。

## 5. 原始 base+crop 质量头

### 5.1 Normal 严格 nested-CV3

每个 outer fold、每个粗类仅用另外两折选择 identity 或 quality，并要求两个选择折分别
满足 Recall/FDR 保护。结果：Ship 三折均 identity；Aircraft、Vehicle 三折均 quality。

| 指标 | baseline | candidate | 增量 |
|---|---:|---:|---:|
| gate Recall | 0.719449 | 0.720181 | +0.073pp |
| gate FDR | 0.237447 | 0.226170 | -1.128pp |
| 绝对分数 | 63.4368 | 64.1844 | +0.7476 |
| 最大粗类 Recall 降幅 | — | — | 0.029pp |

### 5.2 压力集

未校准质量分数在 Sentinel FDR15 上 pooled Recall `+1.879pp`、FDR `-0.074pp`，说明
视觉质量证据确有跨源信息；但在 Hard 上 pooled Recall `-12.697pp`、FDR `+0.704pp`。
原因是质量头残差输出的跨域绝对尺度漂移，而不是单纯缺乏判别信息。原始版本因此拒绝。

## 6. 标签无关分位数校准

对 outer fold `k` 的质量模型，只在 Normal 的另外两折、每个粗类分别计算 257 个质量分数
与 incumbent 分数分位点，形成单调 CDF 映射；不读取 GT 标签，也不读取 Hard/Sentinel
分布。映射后重新执行完整路由选择。

### 6.1 Normal

| 指标 | baseline | calibrated candidate | 增量 |
|---|---:|---:|---:|
| gate Recall | 0.719449 | 0.721772 | +0.232pp |
| gate FDR | 0.237447 | 0.231856 | -0.559pp |
| 绝对分数 | 63.4368 | 64.0500 | +0.6132 |
| 最大粗类 Recall 降幅 | — | — | 0.049pp |

冻结路由为：Ship 三折 identity；Aircraft 三折 quality；Vehicle fold0/1 quality、fold2
identity。该候选通过 Normal 门。

### 6.2 旧全局 FDR15 压力诊断

| 数据集 | baseline R/FDR | candidate R/FDR | ΔR | ΔFDR |
|---|---:|---:|---:|---:|
| Hard-2158 | 0.849861 / 0.146977 | 0.848471 / 0.147976 | -0.139pp | +0.100pp |
| Sentinel-1969 | 0.788217 / 0.151913 | 0.801422 / 0.146566 | +1.320pp | -0.535pp |

Hard 的最坏粗类 Recall 降幅为 Ship `0.505pp`，刚好超过预注册 `0.5pp` 上限，同时总体
方向也轻微为负；因此按旧全局风险门，该模块仍不能正式准入。Sentinel 三个粗类 Recall
均上升，说明校准确实消除了原始版本的大部分跨域失配。

### 6.3 Platform-aligned 每粗类阈值压力读出

正式平台分别报告三个粗类，部署也支持逐类阈值，因此补充执行两折选阈值、一折评估的
每粗类 FDR15 前沿。它比旧的单一全局阈值更接近正式平台，但仍不读取压力折来修改模型。

| 数据集 | baseline R/FDR | candidate R/FDR | ΔR | ΔFDR |
|---|---:|---:|---:|---:|
| Hard-2158 | 0.765987 / 0.116987 | 0.767377 / 0.129795 | +0.139pp | +1.281pp |
| Sentinel-1969 | 0.772981 / 0.149721 | 0.780599 / 0.144209 | +0.762pp | -0.551pp |

Hard 分解为：Ship 完全不变；Aircraft Recall `+0.294pp` 但 FDR `+2.363pp`；Vehicle
Recall 不变、FDR `-3.182pp`。Sentinel 分解为：Ship 完全不变；Aircraft Recall
`+1.716pp`、FDR `-1.240pp`；Vehicle Recall 不变、FDR `+1.679pp`。

该候选在 Sentinel 上稳健为正，但 Hard Recall 增益仅 `0.139pp`，低于预注册
`+0.5pp`，且 Hard FDR 明显恶化；因此每粗类协议下仍拒绝。由于三重门禁未通过，按停止
条件不运行 Background-100MP，也不训练 full 部署头。

把 Normal 三折增量机械平移到第一次正式提交，只作风险估计时，分数约从 `72.1331` 变为
`72.9553`，但 FDR 宏均值仍约 `0.2310`，无法越过 `0.20` 硬门；再结合 Hard 的负向结果，
这不足以值得消耗正式提交次数。

## 7. 当前决策

截至本报告当前版本，没有任何新增模块完成 Normal、Hard、Sentinel 三重准入，因此：

1. 不打包、不构建镜像、不提交；
2. 不把阈值迁移、score-sqrt 或原始 quality 写入正式配方；
3. 分位数校准 quality 是唯一“接近但最终未通过”的候选；
4. 第二次正式提交应等待真正的新权重/候选形成能力，而不是继续调后处理；
5. Background-100MP 和 full 训练只对三重门禁通过者执行。

## 8. 结果索引

- 本地：`outputs/MACROSHIFT-ATTEMPT2-CANDIDATE-TOURNAMENT-20260901/`；
- Normal 原始质量路由：`base_crop_normal/analysis_strict.json`；
- hard-score：`hardscore/decision.json` 与 `hardscore/identity_frontier.json`；
- 正式锚点阈值迁移：`threshold_transfer/analysis.json`；
- 协议审计：`protocol_audit/audit.json`；
- 服务器校准 Normal：`/root/autodl-tmp/results/MACROSHIFT-BASE-CROP-QUANTILE-CAL-V1/`；
- 服务器校准 Hard：`/root/autodl-tmp/results/MACROSHIFT-BASE-CROP-QUANTILE-PRESSURE-HARD-V1/`；
- 服务器校准 Sentinel：`/root/autodl-tmp/results/MACROSHIFT-BASE-CROP-QUANTILE-PRESSURE-SENTINEL-V1/`。
