# B 新阶段任务执行合同：来源审计与低分候选重排

> 执行范围说明：本文件保留立项时的“来源组”假设。最终 B1 实现将 60 个
> MAR20 机场代理组折叠为一个飞机粗来源族，只比较飞机/舰船/车辆三个来源族；
> 实际边界、负向结果与停止结论以 `B_STAGE_FINAL_REPORT_v1.md` 为准。

更新日期：2026-08-08
状态：`completed_historical_contract`
收尾说明：B0/B1 已执行；B2 坐标抽样已被主线 N0-4 正式盲审包取代，最终结论见 `B_STAGE_FINAL_REPORT_v1.md`。

本文是成员 B 在第二阶段的当前执行依据。它承接已完成的数据划分交付，
不重新生成 `dev_v1`、`dev_v2` 或 `cv3`，也不修改任何已冻结的 fold/group 归属。

## 1. 我已经确认的项目状态

项目目标是 25 个细类的遥感目标检测，正式门槛为 Overall Recall >= 0.85、
Overall FDR <= 0.20，10K 端到端推理 <= 20 s。官方排名口径另外按大类内细类
指标简单平均，因此不能只看 pooled Overall。

已完成且已合入 `master` 的 B 交付：

- `dev_v1` 单次开发划分；
- `cv3_airport_proxy_k60_v2` 正式三折划分；
- 舰船景 ID、发射车经纬度、飞机机场代理组的来源隔离；
- 跨划分/跨折 group 泄漏审计与划分使用边界说明。

当前 M1 正式 YOLO26-s CV3 OOF 已完成：4,481 张图恰好预测一次，20,933 个 GT，
55,548 个低阈值候选。探索工作点 `score=0.051` 的结果为：

| 范围 | Recall | FDR | 主要问题 |
|---|---:|---:|---|
| Overall pooled | 0.9172 | 0.1957 | 仅有很小的 FDR 安全余量 |
| 舰船 pooled | 0.8512 | 0.3828 | 背景误检；官方 macro FDR 约 0.52 |
| 飞机 pooled | 0.9338 | 0.1463 | 细类混淆 |
| 车辆 pooled | 0.6169 | 0.6161 | 候选形成不足且背景误检 |

错误分解为 `FP_BG=3303`、`FP_CLS=1115`、`FP_DUP=187`、`FP_LOC=66`；
候选下限 `score>=0.001` 的车辆候选 Recall 只有 0.7985。`score=0.051` 是同一
OOF 上选择的探索点，不能当作最终无偏成绩。

## 2. 队长新任务对应的研究问题

### 主问题

在固定候选预算和官方评分口径下，**来源组、目标尺寸/难度和背景语义是否解释了
M1 的分数漂移与漏检；只使用已有低阈值候选做来源/难度条件重排，能否在不增加
模型推理成本的情况下恢复更多真目标或删除更多背景 FP？**

### 可证伪假设

**H1：** M1 的候选分数在来源组和尺寸段之间存在系统性漂移。若对每个来源组/难度
段使用只由其它 fold 拟合的分数标准化或分层阈值，在相同 top-K/每图预算下，
候选 Recall 将提高且增益不只来自单一来源组。

**H2：** `FP_BG` 不是均匀背景，而是集中在少数来源组、预测类别和分数区间。若成立，
分层抽检能找到高可信 `clear_background` 子集；使用该子集做离线拒识/重排时，
可降低 FP_BG，同时保护 TP Recall。

反证条件：分层后的分数分布差异很小；收益只出现在一个窄来源组；或在至少 2/3
外层 fold 中无法保持方向一致，则停止该方向，不进入正式三折。

## 3. 第一轮最小实验（无训练）

### B0：候选与错误审计

输入 M1 正式 aggregate 和 formal GT，复用项目现有
`load_oof_aggregate`、`decompose_official_errors` 与官方评估器，生成每个候选/GT
的审计行。至少包含：

- `image_id`、`fold`、`group_id`、来源大类和来源组；
- 预测细类、分数、框面积、宽高、长边/短边、边界状态；
- 官方匹配结果与互斥错误类型；
- score bin（0.001--0.01、0.01--0.03、0.03--0.051、0.051--0.10、>=0.10）；
- GT 尺寸段和 near-miss 标记（中心覆盖、IoU 0.10--正式阈值）。

输出按总体、三大类、细类、fold、来源组和尺寸段汇总。该阶段只回答“问题集中在
哪里”，不改变预测，也不声称改进。

### B1：固定预算下的离线重排

保持候选集合、框和模型不变，只改变候选排序/保留规则，比较：

1. 全局原始 score（基线）；
2. 来源组分位数标准化：每个来源组的变换参数只用其它 fold 拟合；
3. 来源组 + 尺寸段的收缩标准化：样本不足的组回退到大类或全局统计；
4. 不做新网络的简单分层保留：保持每图/每 tile 固定 top-K，记录被替换的候选。

每个方法都在同一候选预算下重新运行官方评估，并同时报告 pooled 与官方 macro。
阈值/变换参数必须 cross-fit：评估 fold 的标签和错误不能参与其参数拟合。

### B2：FP_BG 分层抽检包

从 `FP_BG` 按大类、fold、来源组、预测细类和分数分位数分层抽样，建立盲审表。
人工标签至少为：

`clear_background`、`plausible_unlabeled_or_ambiguous_target`、
`poor_localization_of_known_target`、`duplicate_or_fragment_not_captured`、
`invalid_crop_or_render`。

只有 `clear_background` 才能作为拒识证据；其余样本保留为审计样本，不得自动当作
背景负样本训练。

## 4. 验收标准

B1 只作为探索性证据，进入正式三折前必须满足：

- 相同候选预算下，候选 Recall 或工作点 Recall 有净增益；
- pooled FDR 不恶化，且官方 macro FDR 不因某一小类崩坏；
- 至少 2/3 外层 fold 同方向；
- 收益不能主要由单一来源组或 TU-160 单组贡献；
- 所有删除的 FP 经抽检确认主要为真背景，不能把已知 GT 误删。

若 B1 不满足，保留 B0 审计结论，停止重排，不继续堆叠新规则。若 B1 通过，才由
A 负责把规则接入对象学生/风险校准的正式 outer-fold-pure replay。

## 5. 当前缺失输入与向队长索取的文件

本地 Git 仓库没有保存服务器大文件，当前 `outputs/` 只有 `.gitkeep`。开始 B0 前
必须取得以下文件，并保留原始 SHA，不要重新生成替代品：

1. `M1-CV3-OOF-aggregate/`：
   `oof_metadata.json`、`oof_images.csv`、`oof_proposals.csv`、
   `predictions_oof_low.json`；
2. `formal_crop_manifest_v2` 的 `formal_crop_manifest.csv`；
3. 如有，M1 描述性分析包中的
   `exploratory_workpoint_error_cases.csv` 和 `per_fold_metrics.json`，用于交叉核对；
4. 服务器运行环境/资产锁和回传包 SHA（只用于 provenance，不需要权重）。

文件必须满足项目合同中的版本与 SHA：

- split `cv3_airport_proxy_k60_v2`，manifest SHA
  `27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331`；
- formal crop SHA
  `a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128`；
- D00 数据锁 SHA
  `03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a`；
- M1 aggregate 预测 SHA 见服务器回传 `oof_metadata.json`，不得凭文件名猜测。

## 6. 时间安排（8 月 8 日起）

| 时间 | 交付 |
|---|---|
| 8/8 | 索取并验收 M1/formal crop 输入；冻结 B0 schema 和审计脚本入口 |
| 8/9 | 完成 B0 全量审计，输出来源组/尺寸/分数段表；给队长第一版结论 |
| 8/10--8/11 | 完成 B1 三种离线重排和固定预算对照 |
| 8/12 | 完成 B2 FP_BG 分层抽检包，交给队友盲审 |
| 8/13--8/15 | 汇总审计与重排证据，决定保留/停止；必要时做一次修正复验 |
| 8/16--8/20 | 将通过门禁的规则交给 A 做正式 outer-fold-pure replay，并补齐报告 |

## 7. 代码与报告边界

新增脚本必须复用项目官方 evaluator 和 OOF loader，不另写匹配规则；输出放在
`outputs/B-*`（不提交大文件），代码和小型 schema/报告走独立分支与 PR。所有结果
登记 `reports/experiments/leaderboard.csv` 或对应实验报告，并注明：
`exploratory`、`cross_fit`、`source_group_breakdown`、`not_deployment_final`。
