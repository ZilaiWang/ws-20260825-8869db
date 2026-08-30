# HERA-Guard 前置修正与 PAV 快筛报告（2026-08-26）

## 1. 本轮目标

本轮根据项目根目录《改进方案7.md》收尾结论，先修复会污染后续创新实验的评估、标签与验证协议，再实现 HERA-Guard 第一阶段：

1. Y5-ROT90/D4 固定候选；
2. 官方 prediction-first 标签与 fixed-risk 工作点；
3. Proposal-Aligned Verifier（PAV）：tight + context 双视图、共享 ConvNeXt-T 主干、五任务头；
4. Metric-Aligned Asymmetric Resolver（MAR）的单调有界实现；
5. fold0 快筛，达门禁后才扩展三折和 nested resolver。

SCOPE 保留为离线诊断与反事实分析工具，不再直接视为可部署主链。

## 2. 已确认并修复的问题

### 2.1 匹配方向错误

旧 `scope/official_scorer.py`、增量 scorer 和对象图标签使用 GT-first 近似；官方规则是预测按分数降序，逐预测选择最佳未匹配同细类 GT。现统一到：

- `src/rsdet/evaluation/official_metric.py::evaluate_predictions_with_trace`
- `src/rsdet/evaluation/official_frontier.py`
- `src/rsdet/analysis/oer_labels.py`

测试加入了 GT-first 会得到 2 TP、官方 prediction-first 只能得到 1 TP 的确定性反例，并验证全量 scorer、增量 scorer、对象标签与官方 trace 一致。

### 2.2 工作点计数错误

旧反事实 `delta_fp` 使用完整低分尾部 FP 总数，和 FDR=0.12 的实际工作点不是同一对象。新实现明确拆分：

- `protected_tp`：工作点选中且官方 TP；
- `active_fp`：工作点选中且官方 FP；
- `inactive_tail`：工作点未选中，不能当活跃 FP 训练。

分数相同的候选以完整 tie block 扫描，禁止只挑同分块内的 TP。

### 2.3 OOF 泄漏

旧 Gate1 对动作行使用随机 `StratifiedKFold/KFold`，同一候选的 DROP/RELABEL 可以跨折。新 `grouped_oof.py` 强制：

- 外层只允许 frozen formal CV3 fold；
- source group 不跨训练/验证；
- 同候选的所有动作不跨折；
- 内层只能按 source group 做确定性分组；
- 每次实验写出 split audit。

### 2.4 Sentinel 泄漏和 GT oracle

旧 A5 在 sentinel 段先构建非 sentinel mask，随后却对 `Xall/yall` 拟合最终模型；同时旧改类只改“GT 已知错误”的候选。现已改为：

- 最终 sentinel 模型只拟合 non-sentinel 行；
- 概率按 candidate ID 严格对齐；
- 改类不得查询 GT；
- sentinel 与 A3/A5 frontier 均复用官方 scorer。

## 3. 全量重建的实测结果

输入：65,301 条 Y5 D4 OOF 候选、20,933 GT、4,481 图 formal CV3。

### 3.1 对象标签变化

修正前后 TP 总数都为 20,391，但逐候选身份发生显著变化：

| 项目 | 数量 |
|---|---:|
| 旧 TP → 新 FP | 924 |
| 旧 FP → 新 TP | 924 |
| `is_valid` 改变 | 1,848 |
| `matched_uid` 改变 | 1,857 |

结论：总量相同不能证明标签正确；约 9% 的 TP 身份被分给了不同候选，足以污染 OER/路由器监督。

重建产物：

- `outputs/HERA-GUARD-PRECHECK/object-graph-official-v1/nodes.csv`
- `outputs/HERA-GUARD-PRECHECK/object-graph-official-v1/edges.csv`
- `outputs/HERA-GUARD-PRECHECK/object-graph-official-v1/label_contract.json`

### 3.2 修正后的 OER 基线

使用正式 CV3 外层折、255 个 source group 零交叉、14 个固定特征（含 D4/OTO），得到：

| FDR 上限 | Recall | TP | FP |
|---:|---:|---:|---:|
| 0.15 | 0.948932 | 19,864 | 3,499 |
| 0.12 | **0.943104** | **19,742** | **2,687** |
| 0.11 | 0.940620 | 19,690 | 2,431 |
| 0.10 | 0.936655 | 19,607 | 2,176 |

旧报告 `+D4+has_oto ≈ 0.9620` 不能继续作为可靠正式基线。新的 0.943104 是 HERA 快筛的真实起点。

在该 FDR=0.12 工作点按 V1.6 细类等权 macro 口径重新汇总：

| 大类 | macro Recall | macro FDR | pooled Recall | pooled FDR |
|---|---:|---:|---:|---:|
| Ship | 0.703636 | 0.343634 | 0.844892 | 0.266429 |
| Aircraft | 0.956434 | 0.094643 | 0.964648 | 0.089958 |
| Vehicle | 0.641791 | 0.385714 | 0.641791 | 0.385714 |

这进一步确认后续优先级：PAV 首先必须改善车辆与舰船的背景拒识，同时保护飞机；单看 pooled Recall 会掩盖 HM/LQS 和 FSC 的排名短板。当前六项最差值为车辆 `1-FDR=0.614286`。

产物：

- `outputs/HERA-GUARD-PRECHECK/corrected-oer-oof-v1/summary.json`
- `outputs/HERA-GUARD-PRECHECK/corrected-oer-oof-v1/corrected_oer_oof_predictions.json`
- `outputs/HERA-GUARD-PRECHECK/corrected-oer-oof-v1/corrected_oer_oof_scores.csv`

### 3.3 PAV 工作点监督

以修正 OER 分数定义 FDR=0.12 工作点，以冻结 Y5 原始分数定义非循环 foreground 标签：

| 角色 | 数量 |
|---|---:|
| protected_tp | 19,742 |
| active_fp | 2,687 |
| inactive_tail | 42,872 |
| official foreground | 20,391 |

这种双合同避免“用 OER 自己改变匹配标签，再拿改变后的标签重训 OER”的循环定义。

PAV manifest：`outputs/HERA-GUARD-PRECHECK/pav-manifest-oer-v1/hera_pav_manifest.csv`，SHA256 `d259ed6f5b88d96d890de0d2843c66b256d28dc7ebab5767df3f60cb0acc9156`。

## 4. HERA-Guard 当前实现

### 4.1 PAV

`src/rsdet/hera_guard/verifier.py`：

- tight 1.10× 与 context 1.60× 两个 224 crop；
- 拼成 2B，一次共享 ConvNeXt-T 主干前向；
- `[tight, context, difference, abs-difference]` 融合；
- 12 维可部署 metadata；
- foreground / coarse-3 / fine-25 / IoU-quality / TP-protect 五头。

`losses.py` 使用 Balanced Softmax fine loss、前景掩码身份监督、quality BCE 和 protected-TP 非对称权重；背景-only batch 返回连接计算图的零损失，不产生 NaN。

### 4.2 MAR

`resolver.py` 使用 softplus 非负权重与有界 `rho*tanh(delta)` 残差：证据越可靠不能反向降低分数，且不会硬删除候选。细类改写只允许同粗类、高概率、高 margin，并受 protect 概率 veto。

### 4.3 快筛策略

首轮只跑 formal fold0：4 epoch、每 epoch 24,000 个均衡样本、最后 stage + heads 微调。held-out 标签不选 checkpoint、不调阈值；只评预注册的四个固定融合和一个保守改类变体。

探索门禁：

- `ΔRecall@FDR0.12 ≥ +0.002`；
- `ΔRecall@FDR0.10 ≥ -0.001`；
- FDR=0.12 工作点的六项官方 coarse-macro 最差项下降不超过 0.005；
- 通过才扩三折；不通过停止或重做 PAV，不用长训练掩盖结构问题。

配置：`configs/experiments/hera_guard_pav_fast_screen_v1.yaml`。

## 5. 验证状态（历史起点）

本节记录进入 GPU 快筛前的状态；最终状态见第 11 节。

- 当时全仓 pytest：654 passed，5 skipped；
- ruff：本轮文件全绿；
- 官方匹配、tie block、工作点标签、group OOF、PAV empty-positive loss、MAR 单调性均有单元测试；
- 当时科学状态为 `ready_for_pav_fast_screen`，尚未形成正式入选结论。

## 6. 后续分支

1. fold0 PAV 通过：补 fold1/2，形成完整 PAV OOF；
2. 用外层训练域内的 inner-group OOF 拟合 MAR，不在外层 held-out 选融合权重/阈值；
3. 三折聚合评估官方 ranking macro-fine Recall/FDR、pooled 刚性门槛和错误分解；
4. 只在 PAV+MAR 稳定收益后做困难对象门控与 10K 时延；
5. 未通过：优先检查 crop 尺度、active-FP 采样和 protect loss，不直接增加 epoch。

## 7. RTX 3090 执行记录与 fold0 快筛结果

服务器接管前先让已运行到末段的 D3/D4 自然结束，避免破坏既有实验。D3-WorstGroup
与 D4-WorstGroupLoss 的 fold0/fold1 均有 40 个 epoch（`results.csv` 41 行含表头）
及完整 `last.pt`；随后 GPU 释放。HERA 使用同一台 RTX 3090，但使用独立结果目录和
锁文件，不改写 D3/D4 资产。

后续追溯实际驱动确认：D3 使用了 `--hard-curriculum`，D4 使用了
`--innovation worstgroup --hard-curriculum --wg-gain 1.5`，因此实验身份不是普通
续训；但 hard-curriculum 文件仍由全量 formal GT/三折错误统计共同生成，同一份列表
用于不同外层 fold，held-out 信息参与了 worst-group 选择。此外 D4 同时包含过采样和
loss 加权，不是预注册要求的 loss-only 单因素实验。故这四个 checkpoint 只能归档为
泄漏诊断资产，不能称为 `D3-clean`、不能补 fold2、不能写入正式成绩。若未来重启该
方向，必须在每个外层训练域内部重新生成 worst-group，并拆分 sampler-only、loss-only
和 both；当前 PAV/MAR 未晋级后，不再为该条件分支追加 GPU。

执行前复核：

- PAV manifest SHA256：`d259ed6f5b88d96d890de0d2843c66b256d28dc7ebab5767df3f60cb0acc9156`；
- formal manifest SHA256：`a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128`；
- ConvNeXt-T 权重 SHA256：`983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`；
- targeted ruff 全绿，HERA/official/grouped OOF 专项测试 19 项通过；
- GPU smoke 使用 128 train + 128 validation、25 细类全覆盖、group overlap=0，
  五项损失有限且反向成功，耗时约 12.1 秒。

fold0 严格按冻结合同训练 4 epoch；40,559 条训练候选、24,742 条 held-out 候选、
173/82 个 source group，交叉为 0。训练和完整 held-out 推理共 327.8 秒。PAV 本身的
held-out 诊断为：

| 指标 | fold0 |
|---|---:|
| foreground AUC / AP | 0.972776 / 0.953092 |
| protect AUC / AP | 0.987514 / 0.970878 |
| foreground 上细类准确率 | 0.860491 |

固定融合结果（相对 fold0 corrected-OER baseline）：

| 变体 | ΔRecall@FDR=.12 | ΔRecall@FDR=.10 | Δ六指标最差项 | 门禁 |
|---|---:|---:|---:|---|
| guard-soft | +0.001088 | +0.001224 | +0.070510 | 未过主增益门禁 |
| guard-balanced | +0.000952 | +0.002041 | +0.070510 | 未过主增益门禁 |
| guard-strong | +0.000680 | +0.002313 | +0.079213 | 未过主增益门禁 |
| guard-soft + safe relabel | **+0.003946** | **+0.003401** | **+0.069724** | **通过** |

通过变体将 fold0 Recall@FDR=.12 从 0.930748 提高到 0.934694，TP/FP 从
6841/930 变为 6870/929；固定规则共改写 812/24,742 个候选，规则仍限定在同粗类、
PAV 概率≥0.85、margin≥0.40 且 protect≤0.40。

该结果说明两件事：

1. PAV 的主要独立价值首先体现在 proposal 域细类纠错和舰船 macro 短板，而不是
   大幅改变 pooled 前景排序；
2. fold0 只负责准入与冻结变体，不能当最终成绩。已按相同训练合同启动 fold1/2；
   只有预先选定的 `guard-soft + safe relabel` 在未参与选择的 fold1/2 仍保持收益，
   才进入正式 nested MAR。

可复现驱动：

- `scripts/server/run_hera_guard_task01.sh`；
- `scripts/server/run_hera_guard_task02.sh`；
- `scripts/merge_hera_pav_oof.py`。

## 8. PAV-V1 三折确认：表征有效，固定改类不具备稳定性

fold1/2 沿用 fold0 前已冻结的训练配置与五个解码变体。三折均为完整 held-out
推理，随后按 candidate ID 合并为 65,301 条唯一 OOF；合并文件与 manifest
逐行全覆盖、无重复、无 NaN/Inf。

| 评估域 | guard-strong ΔR@.12 | safe-relabel ΔR@.12 | safe-relabel ΔR@.10 |
|---|---:|---:|---:|
| fold0（选择域） | +0.000680 | +0.003946 | +0.003401 |
| fold1（确认域） | +0.001254 | -0.000836 | -0.000975 |
| fold2（确认域） | +0.001874 | -0.000312 | -0.000312 |
| merged 3-fold | **+0.001385** | +0.000525 | +0.001099 |

三折 `guard-strong` 将 Recall@FDR=.12 从 0.943104 提高到 0.944490，同时 FP
从 2,687 降至 2,673，六指标最差项 `+0.001993`。这个方向在三折中均为正，
但仍低于预注册 `+0.002` 主门禁，因此只能称为可靠的弱排序信号，不能宣告
PAV-V1 正式入选。

固定 safe relabel 三折共改 2,443 条：局部几何审计中 corrected 1,279、broken
132、neither-valid 1,032；但 FDR=.12 实际选中区的分布完全不同：

- fold0：selected corrected/broken = 16/4；
- fold1：selected corrected/broken = 4/6；
- fold2：selected corrected/broken = 6/15。

大量“局部可修正”候选位于低分尾部，而少量 broken 位于高分工作区，正好复现
SCOPE-Audit 所揭示的风险不对称。故 `guard-soft + safe relabel` **不进入正式链**。

### 8.1 发现并修正 V1 监督语义偏差

复核代码发现，V1 的 `target_foreground` 来自官方同细类 TP；这等价于“当前类别
已经正确匹配”，不是方案定义的“proposal 几何覆盖任意真实目标”。其后果是：

- 错细类但框覆盖真实目标的候选被当成背景；
- fine head 不直接学习这些 proposal 的真实细类；
- 背景、错类、重复框只能由一个 foreground/protect 组合间接区分；
- 改类阈值即使局部准确，也无法识别它是否位于 active-FP 风险区。

因此建立 PAV-V2 合同：

1. objectness：按每个 GT 所属粗类 IoU 阈值，忽略 detector fine class，选择最佳
   几何重叠；允许重复 proposal 都标为“覆盖对象”；
2. coarse/fine：监督最佳重叠 GT 的真实 3/25 类；
3. quality：监督该最佳重叠 IoU；
4. protect：保留工作点高价值 TP；
5. active-FP：新增显式风险头，区分重复、错类和背景风险；
6. official TP：单独保留为审计列，不再冒充 objectness。

V2 manifest 仍是相同 65,301 候选，objectness 正例从 20,391 变为 34,678，
official TP 保持 20,391，active-FP 保持 2,687。manifest SHA256：
`165cb51428cb2f721a9516ccd9f7eb459d09a1e96e80ec738c530642a35690ad`。

V2 不扩大网络或训练预算：仍为 ConvNeXt-T、4 epoch、24,000 samples/epoch；只
修复监督定义并增加一个轻量 active-FP 头。safe relabel 额外要求 objectness≥0.70、
active-FP≥0.50，避免再把大量 neither-valid 尾部当成可修正目标。V2 GPU smoke
已通过，峰值 allocated VRAM 约 580 MiB。

## 9. PAV-V2 fold0：监督语义修复正确，但排序收益没有扩大

PAV-V2 fold0 沿用 V1 的 formal fold、ConvNeXt-T 权重、训练预算和固定门禁，只改变
监督合同并增加 active-FP 风险头。40,559 条训练候选、24,742 条 held-out 候选，
4 epoch 完成用时 326.1 秒；峰值 allocated VRAM 约 1,869 MiB。关键表征指标为：

| 指标 | fold0 |
|---|---:|
| objectness AUC / AP | 0.977617 / 0.982328 |
| protect AUC / AP | 0.992129 / 0.978421 |
| active-FP AUC / AP | 0.825593 / 0.215514 |
| objectness 正例上的 fine accuracy | 0.797656 |

固定融合相对 corrected-OER fold0 baseline 的结果为：

| 变体 | ΔRecall@FDR=.12 | ΔRecall@FDR=.10 | Δ六指标最差项 |
|---|---:|---:|---:|
| guard-soft | +0.000272 | +0.000408 | +0.019013 |
| guard-balanced | -0.000544 | -0.000408 | +0.009097 |
| guard-strong | -0.002177 | -0.002177 | +0.004225 |
| guard-soft + V2 safe relabel | +0.000408 | +0.000544 | +0.018642 |

V2 safe relabel 只修改 45 条候选，局部结果为 corrected 29、broken 9、neither 7；相较
V1 的 812 条，风险门控确实大幅抑制了低价值尾部改写，但正式工作点 Recall 增益也只剩
0.041 个百分点，远低于预注册的 0.2 个百分点。因此 V2 **不扩展到 fold1/2**。

这个结果不能解释为“V1 标签更好”。正确结论是：V2 修复了方法定义与实现的偏差，
并证明 objectness/protect 表征可学；但 active-FP 的 AP 较低，且候选风险在工作点附近
不能由当前 crop 与 metadata 稳定分离。增加 epoch 或继续调固定权重不针对这一瓶颈。

回传包：

- `outputs/HERA-GUARD-TASK-03-V2/HERA-PAV-V2-OBJECTNESS-RISK-FOLD0-return-no-checkpoint.tar.gz`；
- SHA256 `d9f196ca3c9f1012bdec051da925f156c13f5554687a883fd3a0f9c35d41155a`。

## 10. MAR 低成本交叉拟合代理：不进入严格 nested stacking

PAV-V1 的 `guard-strong` 在三折均为正，但合并增益只有 +0.001385。为判断是否值得
再训练多套 inner-fold PAV，先做一个不用于正式结论的 MAR 代理：每个外层 held-out
fold 的 MAR 只使用另外两个 fold 的 PAV OOF 特征训练，输出仍覆盖全部 65,301 个
候选。输入六项单调证据为 foreground、quality、protect、当前 detector fine
概率、fine margin 与负熵。

该协议仍不是严格 stacking：训练 MAR 时使用的另外两个 fold 的 PAV logits 来自包含
当前 held-out 图像的 base-PAV 训练。因此产物明确标注
`outer_meta_crossfit_without_inner_pav_oof` 和 `formal_admission=false`；它只负责判断
是否值得付出 inner PAV OOF 的成本。

| 评估域 | ΔRecall@FDR=.12 | ΔRecall@FDR=.10 | Δ六指标最差项 |
|---|---:|---:|---:|
| fold0 | +0.000272 | +0.000136 | +0.000000 |
| fold1 | +0.000696 | +0.000836 | -0.014161 |
| fold2 | -0.000468 | +0.001093 | -0.021182 |
| merged 3-fold | **+0.000430** | **+0.000717** | **-0.011037** |

合并后 Recall@FDR=.12 从 0.943104 增至 0.943534，只增加 9 个净 TP；车辆/舰船等
最差指标反而下降 1.10 个百分点，且 fold2 的主 Recall 为负。代理门禁失败，决策为
`stop_learned_mar`。由于连乐观、存在 stacking 污染的代理都没有达到 +0.002 且破坏
六指标，没有科学理由投入严格 inner PAV OOF；否则只是把计算量增加约数倍来拟合一个
已显示不稳定的弱信号。

产物：

- `outputs/HERA-GUARD-MAR-CROSSFIT-PROXY-V1/mar_crossfit_proxy_summary.json`；
- `outputs/HERA-GUARD-MAR-CROSSFIT-PROXY-V1/mar_crossfit_proxy_scores.npz`。

## 11. 本轮最终决策与可保留资产

截至 2026-08-26，HERA-Guard 的评估修复、PAV-V1 三折、PAV-V2 监督修正和 MAR
代理均已闭环。正式决策如下：

1. corrected-OER 继续作为可信部署基线：Recall@FDR=.12 为 0.943104；
2. PAV-V1 `guard-strong` 保留为三折一致的弱正向消融（+0.001385），不进入正式链；
3. V1/V2 hard relabel 均不进入部署链；
4. PAV-V2 和 learned MAR 不再扩展训练；
5. HERA 的正式科学价值保留在：官方同源评估修复、proposal-domain 风险标签、SCOPE
   风险不对称证据，以及“固定改类/关系解析不稳定”的完整证伪链；
6. 后续精度资源应转向基础检测器、类别/粗类专用校准和真实错误域的数据改进，而不是
   继续堆叠 crop verifier；10K 工程应以冻结 corrected-OER 为基线独立推进。

最终科学状态：`hera_pav_mar_not_admitted`。这不是实验失败后丢弃结果，而是通过严格
OOF、六指标和预注册停止条件得到的可复现负向结论。
