# HERA-Guard Final：当前实验、证据与正式提交策略说明（2026-08-31）

状态：`ext_v_three_gpu_coarse_running / no_new_submission_admitted_yet`

本文是一份可独立阅读的策略交接材料，面向需要在较短时间内判断“当前实验在做什么、为何
值得做、完成后怎样使用正式提交机会”的讨论。所有数字均来自仓库内已冻结报告或正在运行的
服务器审计；尚未完成的 EXT-V 结果明确标为进行中，不把训练损失当成比赛收益。

## 1. 比赛目标与当前锚点

平台展示三粗类 `ship / aircraft / vehicle` 的 Recall、FDR，以及平均推理时延；综合分是七项
相对排名的组合，并非六个质量数的简单平均。官方 V1.6 还对 ship 4 个细类、aircraft 20 个
细类做粗类内等权，因此 pooled 指标不能替代 25 类 macro。

当前唯一正式 incumbent 是 `trial-v2.0`：full YOLO26s（历史文档简称 Y5-S）、identity
单视图、safe-1024、统一产品阈值 0.15、safe fusion。

|指标|trial-v2.0|
|---|---:|
|综合分|86.2274|
|ship Recall / FDR|0.942287 / 0.126937|
|aircraft Recall / FDR|0.999246 / 0.024300|
|vehicle Recall / FDR|0.946309 / 0.237838|
|平均时延|2.704833 s|

结论很集中：aircraft 已接近饱和；最大短板是 vehicle FDR=23.78%，同时不能牺牲其 94.63%
Recall；ship 的 Recall 和 FDR 均有约 5--13pp 的可见缺口。下一次候选必须改变 ship/vehicle
的表示或低风险排序，而不是继续换 aircraft Recall 或无条件增加视图。

`trial-v3.0` 是重要反例：双视图使 ship Recall 升至 0.953614、aircraft Recall 达到 1，
vehicle FDR 降至 0.156250；但 ship FDR 升至 0.165644、vehicle Recall 降至 0.906040，时延
升至 4.888833 s，综合分反降到 85.0018。因此不能用 Recall/FDR 交换和翻倍时延制造“部分
指标更好”的假升级。

## 2. 固定的两级本地评估

### 2.1 方法选择账本

所有需要学习参数或改变候选排序的方法，只能按同一合同比较：

1. `Normal-CV3`：检查正常域 pooled Recall、25 类 macro 和三粗类候选地板；
2. `Hard10K-CV3`：检查复杂背景、低分真目标和高分结构化背景的排序；
3. `source-disjoint Sentinel`：冻结阈值做一次来源互斥的外推验证，不参与选参。

固定准入门：

```text
Normal pooled Recall 与 25类 macro 降幅均 <= 0.3pp
任一粗类 Recall 降幅 <= 0.5pp
总体 FDR 恶化 <= 1.0pp
Hard10K@FDR<=0.15 的 ship 或 vehicle Recall 增益 >= 0.5pp
Sentinel 与 Normal/Hard 同方向
aircraft 基本不退化
```

Hard/Sentinel 不用于扫描阈值、融合权重或 checkpoint。失败即停止，而不是围绕代理集继续
调参。这是防止本地“94分”虚高、官方却不能迁移的核心约束。

### 2.2 提交前部署账本

只有通过方法选择门的唯一候选才会训练 full 权重，并按 Docker 完全相同的切片、阈值、NMS、
融合、输入 schema 在 Linux/amd64 与 RTX3090 上验收：结果有限性、确定性复跑、10K 时延、
trial-mix V1.6 macro 审计。它判断候选是否可提交，不再反向选择训练参数。

## 3. 为什么当前选择外部粗类预训练

Hard10K 四源候选池对 2,158 个 GT 的 oracle Recall 为 97.31%，Sentinel 为 98.17%；但低
FDR 工作点只有约 85--86% Recall。约 80% 的可见 FN 属于“已有候选但排序在结构化背景之后”，
而不是绝对没有框。322 个 FP 中 248 个是 FP_BG（77%）。固定 proposal 上的 Q0、crop、D4、
OTO、FPN 质量头、GroupDRO、soft-FDR 等多数已经到千分点级边际，说明仅在相同表示上训练
小重排器很难跨档。

外部粗类预训练试图直接改变底层表示：用合规遥感数据学习大量小目标、道路、停车区、港口、
水域和 `other_remote_object`，随后丢弃外部四类 head，重建官方 25 类 head。外部细类从不
伪装成比赛型号标签，外部数据也不与 Normal/Hard/Sentinel 混用。

本轮完整 DOTA train+val 包含：

|资产|数量|
|---|---:|
|源图|1,869|
|1024 tiles|9,153（6,461 positive + 2,692 empty）|
|有效粗类标注|102,518|
|aircraft / ship / vehicle / other|10,264 / 36,344 / 49,319 / 6,591|

96 张视觉审核卡全部通过。外部路线最初预注册 `EXT-G`（通用/ship）和 `EXT-V`（vehicle/
结构化城市背景）。由于正式时间有限、HAD 已提供清晰负证据且官方最大短板是 vehicle FDR，
当前只推进信息密度更高的 EXT-V；早期 EXT-G/EXT-V 各 4 epoch 的未完成 run 不用于结论。

## 4. 官方标注修补为什么需要配对对照

三折 OOF Y5+D-FINE 产生 67 个 vehicle 疑似漏标候选，逐图审核得到：32 confirmed missing、
20 ambiguous ignore、15 rejected。为避免把疑似真实对象继续当成背景负样本，patch 和 control
都排除 ambiguous 图；唯一差别是 confirmed 框 add 或 omit。最终两边均为 4,470 图，实际新增
18 个框，另 14 个 confirmed 与排除图重合。

因此后续三个 fine 分支分别是：

|分支|初始化|官方微调数据|回答的问题|
|---|---|---|---|
|A：EXT-V→patch|DOTA EXT-V coarse|reviewed patch|完整候选是否有效|
|B：official→patch|官方 YOLO26s|同一 reviewed patch|收益是否只来自标注修补|
|C：official→omit|官方 YOLO26s|同排除集、不加 confirmed|原始 omission 配对基线|

它们是因果对照，不是三个最终部署模型。A-B 隔离外部初始化贡献，B-C 隔离 annotation patch
贡献；只有 A 同时优于原 incumbent 和配对对照，才能把两种变化一起带入 full 训练。

## 5. 当前三卡执行合同与实时进展

当前服务器为 3×RTX3090。coarse 阶段三卡 DDP 同步训练**同一个** EXT-V YOLO26s：80 epoch、
1024 输入、AdamW、seed=20260831、无 resume。资源剖析后将总 batch 从 12 调整为 30：

```text
原合同：batch 12 × gradient accumulation 5 = effective batch 60
当前合同：batch 30 × gradient accumulation 2 = effective batch 60
```

有效优化 batch、学习率、权重衰减、epoch、seed 和样本轮数保持不变；只提高每卡并行度。
旧 run 在仅完成 1 epoch 后归档，新 run 从初始权重 SHA 重新开始。

2026-08-31 20:06 CST 快照：

- 状态 `extv_coarse_ddp_3gpu`，完成 2/80，正在第 3 轮；
- 第 1 轮 117.817 s，第 2 轮累计 226.312 s；
- 三卡利用率瞬时 87%--97%，显存约 10.8--12.2 GiB，温度 63--69°C；
- box/cls/dfl 损失有限并下降，无 OOM、NaN、AMP 或 DDP 错误；
- coarse 预计约 2.5--2.7 小时；完成后 A/B/C 三路各占一张卡并行 fine 40 epoch，预计
  30--50 分钟；冻结三域评测约 10--20 分钟。

fine 使用 `8 epoch freeze前10层 + 32 epoch full`，总 batch=12/单卡。fine 的动态 Trainer
无法被 Ultralytics 临时 DDP 子进程导入，因此没有为形式上的多卡而重构训练框架；三路单卡
并行既更稳定，也能最早给出配对结论。

## 6. 已完成但正式停止的路线

### 6.1 D-FINE 直接部署

full D-FINE-M 本身训练健康，但作为第二检测器进入部署时，Hard/Sentinel vehicle Recall 分别
下降 9.239pp/8.025pp，时延约从 4.3s 增至 9.4s；正式拒绝。D-FINE 只保留为训练期教师。

### 6.2 In-model HAD 蒸馏

HAD 把 D-FINE support 蒸馏进单 Y5 的 vehicle bounded residual，结构和训练链均真实可用。
terminal-FPN 因 Normal ship -0.815pp、Hard ship -0.629pp 先行拒绝。branch-only 完整 CV3：

|域|关键变化|
|---|---|
|Normal|总体 Recall -0.104pp；ship -0.543pp；vehicle -1.220pp|
|Hard|vehicle +2.174pp；总体 +0.093pp|
|Sentinel|vehicle +1.235pp；总体 +0.152pp|

它在压力域有信号，却破坏 Normal 粗类地板，固定门
`all_coarse_recall_floor_minus_0p5pp=false`，因此全路线停止，不补 epoch、不扫权重。

### 6.3 其余已有负向证据

全量 rot90 TTA、D-FINE/Y5-L/M3 无条件并集、普通 crop verifier、FPN-Q1/Q2、SAHI/P2、旧困难
背景微调、全局阈值/NMS 网格、DINO/CleanDIFT 直接拼接均已有负向、不稳定或时延证据。它们
不应在今天重新占用正式机会。

## 7. EXT-V 完成后的唯一决策树

### 7.1 A 未通过三域门

- 不训练 full EXT-V，不制作实验镜像；
- `trial-v2.0` 继续作为 Safe incumbent；
- 若今天必须使用一次正式机会，只能在已有、部署等价且风险明确的候选中选择，不能根据
  EXT-V 的训练 loss 或单一 Normal 指标贸然提交。

### 7.2 patch 有效、外部初始化无独立收益

- B 优于 C，但 A 不优于 B：只将 reviewed partial-label-safe patch 作为候选；
- 用同一 full official 初始化训练唯一 full patch 权重；
- 完成三域、macro、3090 时延和 Docker 复测后再决定是否提交。

### 7.3 EXT-V 有独立且稳定收益

- A 优于 B/C，并通过 Normal/Hard/Sentinel：从已完成 coarse checkpoint 训练唯一 full
  official patch 权重；
- 最终 Docker 仍是 identity 单视图、单 YOLO26s，不携带 DOTA head、D-FINE 或第二模型；
- 预计时延应接近 trial-v2，而不是 trial-v3 的约 4.9s；
- 容器完全通过后，才把它作为下一次正式提交候选。

### 7.4 只在 Hard 正向、Normal 负向

按照 HAD 的先例拒绝。Hard 是风险压力测试，不是允许牺牲正常域的优化目标；不能为了达到
本地某个 94% 数值而放宽粗类地板。

## 8. 正式提交机会建议

1. `trial-v2.0` 已建立 86.2274 的可复现锚点，不需要重复消耗机会。
2. 下一次机会优先留给通过全部门禁且 full/Docker 完成的 EXT-V 或 patch 候选。
3. 不以 coarse loss、fold0 单点或 Hard 单域正向替代正式准入。
4. 若新候选只改善 FDR 但 vehicle Recall 明显下降，它本质上重演 trial-v3，不提交。
5. 后续机会至少保留一次给真正的全局 Pareto 最优版本，避免在同一方法上做隐藏集阈值扫描。

当前最值得等待的证据不是“EXT-V 能否收敛”，而是：A 相对 B/C 是否在三域同时提高
ship/vehicle 固定风险排序，并保持 25 类 macro、aircraft 和单视图时延。只有这个答案为真，
它才有潜力把官方 86.23 提升一个档次。

## 9. 关键文件索引

- 当前总执行报告：`reports/experiments/HERA_GUARD_FINAL_PREFLIGHT_EXECUTION_20260831.md`
- 官方 v1--v3 深度分析：
  `reports/experiments/OFFICIAL_TRIAL_V1_V3_DEEP_ANALYSIS_AND_GPT_HANDOFF_20260830.md`
- 理论上限与瓶颈：
  `reports/experiments/CURRENT_SYSTEM_THEORETICAL_CEILING_AND_BREAKTHROUGH_20260831.md`
- 正式五次机会合同：
  `reports/experiments/FORMAL_FIVE_SUBMISSION_AND_LOCAL_TOURNAMENT_20260831.md`
- 方案11证据合并：
  `reports/experiments/HERA_GUARD_FINAL_PLAN11_RECONCILIATION_20260831.md`
- 外部数据科学合同：`docs/data/EXTERNAL_REMOTE_SENSING_PRETRAINING_CONTRACT.md`
- 服务器执行合同：`docs/server/HERA_GUARD_FINAL_4GPU_EXECUTION.md`
- 三卡快速驱动：`scripts/server/run_hera_guard_final_extv_3gpu_fast.sh`
- 外部粗类训练：`scripts/train_external_y5_coarse.py`
- fresh 25类 head 迁移：`scripts/train_external_initialized_y5_fine.py`
- partial-label-safe 数据：`scripts/materialize_partial_label_safe_dataset.py`
- 冻结候选决策：`scripts/decide_hera_guard_final_candidate.py`

## 10. 讨论时需要回答的核心问题

1. 在“不提高部署时延、不牺牲任一粗类 Recall”的条件下，EXT-V 表示最可能首先改善
   vehicle FDR 还是 ship Recall？
2. A/B/C 的三域结果应满足怎样的最小效应量，才值得承担一次正式提交的域偏移风险？
3. 若 A 只比 B 好 0.3--0.5pp，但 B-C 的 patch 收益更大，最终 full 应选 A 还是 B？
4. 若 EXT-V 失败，剩余时间应优先做“同架构第二 seed 的一致性蒸馏”，还是接受 trial-v2
   作为稳健版本并保留提交机会？
5. 如何在官方排名函数不公开时，把质量收益、细类 macro 与时延风险合成保守的提交决策？
