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

## 5. 验证状态

- 全仓 pytest：654 passed，5 skipped；
- ruff：本轮文件全绿；
- 官方匹配、tie block、工作点标签、group OOF、PAV empty-positive loss、MAR 单调性均有单元测试；
- GPU fold0 尚待服务器认证后执行；当前科学状态为 `ready_for_pav_fast_screen`，不是正式入选。

## 6. 后续分支

1. fold0 PAV 通过：补 fold1/2，形成完整 PAV OOF；
2. 用外层训练域内的 inner-group OOF 拟合 MAR，不在外层 held-out 选融合权重/阈值；
3. 三折聚合评估官方 ranking macro-fine Recall/FDR、pooled 刚性门槛和错误分解；
4. 只在 PAV+MAR 稳定收益后做困难对象门控与 10K 时延；
5. 未通过：优先检查 crop 尺度、active-FP 采样和 protect loss，不直接增加 epoch。
