# 《改进方案7》执行收尾与正式决策（2026-08-26）

## 1. 结论先行

《改进方案7》提出的最后实验树已经完成到当前数据和资产允许的边界：评估合同已修复，
corrected-OER 已重建，PAV 完成 V1 三折和 V2 监督修正，MAR 完成低成本准入代理，
D3/D4 完成实验身份追溯，10K 工程已有 RTX 3090 冻结基准。最终没有证据支持把
PAV、hard relabel、MAR 或现有 D3/D4 加入部署链。

当前可信精度基线为 corrected-OER：

| 指标 | 数值 |
|---|---:|
| Recall@FDR=.12 | **0.943104** |
| TP / FP | 19,742 / 2,687 |
| Recall@FDR=.10 | 0.936655 |
| 六项 macro 最差值 | 0.614286（vehicle 1-FDR） |

最强但未过门禁的 PAV-V1 score-only 变体为三折 `guard-strong`：
`ΔRecall@FDR=.12=+0.001385`，三折方向均为正，六项最差值 `+0.001993`；它是
可信弱正向消融，不是正式升级。最终科学状态为 `hera_pav_mar_not_admitted`。

## 2. 阶段状态表

| 方案7阶段 | 状态 | 证据与决策 |
|---|---|---|
| 阶段 0：评估合同冻结 | **完成** | prediction-first 官方匹配、tie block、fixed-risk frontier、workpoint roles、formal group OOF 均实现并有测试；旧 0.9620 作废 |
| 阶段 1：PAV | **完成并停止** | V1 三折、V2 fold0；V1 score-only 未过 +0.002，relabel 跨折反转；V2 修正标签后仍未过门禁 |
| 阶段 2：MAR | **代理门禁失败，停止** | 乐观 cross-fit 代理仅 +0.000430，六项最差值 -0.011037；不投入 inner-PAV nested stacking |
| 阶段 3：D3/D4 | **现有结果无正式资格** | hard curriculum 由全量三折 GT/错误统计生成；D4 又混合过采样与 loss，不是单因素；不补 fold2 |
| 阶段 4：10K 工程 | **现有输入下完成** | RTX 3090 合成 10K：M1 p95 4.6166s；M1+M3 预算加总 p95 14.46s，均低于 20s；官方真实 10K 未提供，不能宣称官方实测 |

## 3. 阶段 0：评估合同修复

### 3.1 已修复内容

1. 预测按分数降序逐条匹配最佳未使用同细类 GT，替代 GT-first 近似；
2. 相同分数按完整 tie block 扫描，不能从同分候选中只挑 TP；
3. `protected_tp / active_fp / inactive_tail` 只按 FDR=.12 实际工作点定义；
4. OER 标签、SCOPE scorer、增量 scorer 统一使用官方 trace；
5. outer split 固定 formal CV3，inner split 按 source group，候选动作不可跨折；
6. sentinel 训练代码只允许 non-sentinel 拟合，改类不查询 GT。

### 3.2 关键影响

旧、新官方标签的 TP 总量虽然都为 20,391，但有 924 个旧 TP 变 FP、924 个旧 FP
变 TP，1,848 个候选身份改变。旧约 0.962 的 OER 数字依赖错误标签/评估合同，不能
继续使用。旧 sentinel 已被多轮开发查看，修复后的代码可保留，但其结果不能恢复为
真正独立 lockbox；正式结论以 outer group OOF 为主。

### 3.3 代码与产物

- `src/rsdet/evaluation/official_frontier.py`
- `src/rsdet/evaluation/grouped_oof.py`
- `src/rsdet/analysis/oer_labels.py`
- `src/rsdet/analysis/workpoint_labels.py`
- `outputs/HERA-GUARD-PRECHECK/object-graph-official-v1/`
- `outputs/HERA-GUARD-PRECHECK/corrected-oer-oof-v1/`

## 4. 阶段 1：PAV 实验链

### 4.1 V1

V1 使用 tight 1.10× 与 context 1.60× 两视图、共享 ConvNeXt-T、12 维 metadata，
输出 foreground/coarse/fine/quality/protect。fold0 safe relabel 曾得到 +0.003946，
但冻结后在 fold1/2 分别为 -0.000836/-0.000312，三折不稳定，正式拒绝。

score-only `guard-strong` 三折分别 +0.000680/+0.001254/+0.001874，合并
+0.001385。它证明 proposal crop 存在弱排序信息，但未达到 +0.002 准入线。

### 4.2 V2

V1 把“同细类官方 TP”误作 objectness。V2 改为忽略 detector fine class、按几何覆盖
任意 GT 定义 objectness，并监督最佳 GT 的 coarse/fine/IoU，新增 active-FP 风险头。
65,301 条候选中，objectness 正例 34,678、official TP 20,391、active FP 2,687。

V2 fold0 active-FP AP 仅 0.215514；最佳固定变体只提升 +0.000408。V2 在正确合同下
没有放大收益，因此不扩展 fold1/2。

### 4.3 入口

- `src/rsdet/hera_guard/labels.py`
- `src/rsdet/hera_guard/verifier.py`
- `src/rsdet/hera_guard/losses.py`
- `scripts/build_hera_pav_manifest.py`
- `scripts/train_hera_pav.py`
- `scripts/evaluate_hera_pav_fast_screen.py`
- `scripts/merge_hera_pav_oof.py`
- `scripts/server/run_hera_guard_task01.sh`
- `scripts/server/run_hera_guard_task02.sh`
- `scripts/server/run_hera_guard_task03_v2.sh`
- `reports/HERA_GUARD_PRECHECK_AND_FAST_SCREEN_20260826.md`

## 5. 阶段 2：MAR 准入判断

严格 MAR 需要为每个 outer fold 再构造 inner-PAV OOF，成本约增加数倍。先执行一个
明确标注非正式的乐观代理 `outer_meta_crossfit_without_inner_pav_oof`：每个 outer
held-out fold 的 MAR 只在另两个 fold 的 PAV 特征上拟合，不使用 held-out 标签选参数。

结果：fold0/fold1/fold2 的 ΔRecall@.12 分别为 +0.000272、+0.000696、-0.000468；
合并 +0.000430，六项最差值 -0.011037。连乐观代理都不满足增益和安全门禁，故停止
learned MAR，不用正式算力拟合弱且不稳定的信号。

入口与产物：

- `src/rsdet/hera_guard/mar_training.py`
- `scripts/train_hera_mar_crossfit.py`
- `outputs/HERA-GUARD-MAR-CROSSFIT-PROXY-V1/`

## 6. 阶段 3：D3/D4 身份审计

服务器最终驱动证明参数确已启用：D3 使用 `--hard-curriculum`，D4 使用
`--innovation worstgroup --hard-curriculum --wg-gain 1.5`。但共同 curriculum 来自
全量 formal GT 与三折错误统计，并被所有 outer fold 复用，held-out 信息参与训练域
选择；D4 还同时包含 D3 过采样和 loss 加权。因此：

- fold0/fold1 的 40 epoch checkpoint 只作泄漏诊断归档；
- 不能把它们称为 D3-clean 或 D4-loss-only；
- 不补 fold2，不进入正式比较；
- 只有未来从每个 outer-train 内部生成 worst group，并拆 sampler/loss/both，才能重启。

鉴于 PAV/MAR 主链已经不晋级，当前不为这个条件分支追加 GPU。

## 7. 阶段 4：10K 工程边界

现有 E 流水线已经覆盖切片、批推理、全局坐标恢复、跨 tile 聚合、COCO 导出与分段
计时。冻结 RTX 3090 合成 10K 基准为：

| 链路 | total-after-read p95 | 判定 |
|---|---:|---|
| M1 | 4.6166s | 低于官方 20s 硬门槛 |
| M3 | 12.4843s | 低于 20s |
| M1+M3 串行预算 | 14.46s | 低于 18s 内部安全线 |

PAV/MAR 未入选，因此无需实现“只对困难候选运行 PAV”的额外工程分支。正式系统应
冻结 corrected-OER 所需的候选与后处理，避免引入未获精度收益的计算。项目目前没有
官方真实 10000×10000 测试图；上述是合成输入的软件/硬件基准，不可写成官方实测。

证据：

- `reports/experiments/E_FORMAL_BENCHMARK/audit.json`
- `reports/experiments/E_FORMAL_BENCHMARK/M3/audit.json`
- `reports/experiments/E_COMBINED_FORMAL_20260819/E_COMBINED_FORMAL_REPORT.md`
- `src/rsdet/pipeline/large_image.py`
- `src/rsdet/postprocess/global_aggregation.py`

## 8. 最终保留、停止与待外部解锁

### 正式保留

- corrected official scorer 与 fixed-risk frontier；
- formal CV3/source-group OOF；
- corrected-OER 0.943104 基线；
- 现有 10K 切片、融合、COCO 导出和分段计时；
- PAV-V1 score-only 作为弱正向消融；
- PAV-V2 标签合同、SCOPE 风险不对称分析作为方法与负向实验资产。

### 正式停止

- PAV hard relabel；
- PAV-V2 扩三折；
- strict nested learned MAR；
- 当前泄漏版 D3/D4；
- 把旧 0.9620 或旧 sentinel 0.9847 当作正式成绩。

### 只能等待外部输入

- 官方真实 10K 图上的最后一次时延复测；
- 最终提交环境/硬件确定后的完整部署模拟。

这些等待项不影响当前精度结论，也不构成继续搜索模型超参数的理由。

## 9. 验证与版本

- 全仓测试：658 passed，5 skipped；
- HERA/official 专项：37 passed；
- ruff：本轮文件全绿；
- shell 驱动：`bash -n` 全过；
- PAV-V2 回传包 SHA256：
  `d9f196ca3c9f1012bdec051da925f156c13f5554687a883fd3a0f9c35d41155a`；
- 实现提交：`0686ea4`；
- 公开报告连接信息清理提交：`d915838`。

