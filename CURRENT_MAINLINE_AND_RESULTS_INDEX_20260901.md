# 当前主线与 2026-09-01 实验结果索引

本文档是当前状态的单一入口，只索引有效主线、现行评估协议、2026-09-01
收口实验及其复现代码。已否决候选不得被写成可部署能力。

## 1. 一页状态

| 项目 | 当前结论 |
|---|---|
| 当前可部署基座 | full YOLO26-s / Y5 旋转增强，单模型、identity 分数链 |
| 历史预测评最高 | 86.2274；Ship 0.942287/0.126937，Aircraft 0.999246/0.024300，Vehicle 0.946309/0.237838，2.704833s |
| 正式 Attempt 1 | 72.1331；Ship 0.874969/0.320177，Aircraft 0.967641/0.064691，Vehicle 0.852632/0.325000，2.473167s |
| 硬门 | 三粗类平均 Recall 0.898414 通过；FDR 0.236623 失败；时延通过 |
| 主瓶颈 | Ship/Vehicle 隐藏域宏 FDR，且降 FP 时容易丢失低分 TP |
| 2026-09-01 新模块 | 全部未通过 Normal/Hard/Sentinel 联合门；无 full 准入 |
| 打包/提交 | 本次未打包、未构建 Docker、未提交 |

`Recall/FDR` 均按对应粗类的官方口径报告。预测评与正式隐藏集分布显著不同，
86.2274 不得再写成正式 incumbent 得分。

## 2. 当前有效主线

### 2.1 模型、训练和评分血缘

- 基座：YOLO26-s、1024 输入、三粗类下 25 细类直接检测。
- 开发基线：`Y5-ROT90-CV3-OOF`，三折 fixed-last，旋转增强为已证明的稳定增益。
- 部署基线：4,481 张官方训练图的 full Y5-S，单视图、单模型、无级联专家。
- 打分：detector identity score；未过门的质量头、融合、蒸馏和背景微调都不开启。
- 正式协议：`platform_observed_20260831`，Ship/Aircraft/Vehicle 等权，pooled 计数只作诊断。

主线证据：

- `reports/experiments/FORMAL_ATTEMPT1_ABSOLUTE_SCORE_FREEZE_20260831.md`
- `reports/experiments/FORMAL_HIDDEN_DISTRIBUTION_INFERENCE_AND_PROXY_V1_20260901.md`
- `reports/experiments/IMPROVEMENT_PLAN13_AUDIT_AND_NEXT_ACTION_20260901.md`
- `configs/evaluation/formal_hidden_anchor_v1.json`
- `configs/evaluation/metric_protocol_registry.json`
- `src/rsdet/evaluation/platform_protocol.py`
- `src/rsdet/evaluation/absolute_score.py`

### 2.2 固定内部评估

1. **Normal CV3 / MacroMirror**：4,481 图唯一 OOF，保护已有 TP 并比较 paired delta。
2. **Hard10K**：压力分布，暴露结构化背景与低分真目标的排序冲突。
3. **source-disjoint Sentinel-A**：阈值从 Hard 冻结，Sentinel 上不再选阈值。
4. **Background-100MP**：只评估 FP/100MP，不替代带 GT 的 Recall/FDR。

模块只有在 Normal 保护 Recall，且 Hard/Sentinel 同方向降低 FDR 时才能进入
full。不允许用单一开发集最佳阈值或事后融合权重替代该门禁。

## 3. 2026-09-01 七步 MacroShift

总报告：`reports/experiments/MACROSHIFT_FROZEN_7STEP_IMPLEMENTATION_AND_LOCAL_RESULTS_20260901.md`

| 步骤 | 实现 | 结果 |
|---|---|---|
| 协议迁移 | `platform_protocol.py`、`audit_metric_protocol_migration.py` | 13/13 正式入口通过 |
| fine 阈值 | `postprocess/thresholds.py` 与 direct/safe/global/Docker 接入 | 能力完成；无阈值候选准入 |
| MacroRisk V2 | `evaluation/macro_risk_v2.py` | 代理分 +1.7624，但宏 Recall -8.596pp，拒绝 |
| Vehicle Reject+Rescue | `submission/vehicle_rescue.py` | Vehicle Recall +1.244pp，FDR +1.169pp，拒绝 |
| Ship quality | `train_official_quality_head.py` 及 CV3 分析 | Ship Recall -1.058pp/FDR +1.150pp，拒绝 |
| Background-100MP | build/review/freeze/evaluate 脚本 | 382 crop、100.139MP，冻结为压测资产 |
| Sentinel-B | `freeze_sentinel_b.py` | 代码完成；无真正未见分组，fail-closed |
| 唯一 full recipe | `compose_macroshift_final_recipe.py` | 通过模块数 0，正确阻止 full |

## 4. 80/85 分突破实验总账本

详细合同和每粗类变化见
`reports/experiments/HERA_GUARD_BREAKTHROUGH_80_85_EXECUTION_20260901.md`。

| 路线 | Normal | Hard | Sentinel | 决策 |
|---|---:|---:|---:|---|
| DOTA EXT-V | Recall -0.731pp | -0.973pp | -0.457pp | 拒绝 |
| 同架构 support | 无稳定正增益 | Recall 大幅下降 | Recall 大幅下降 | 拒绝 |
| Varifocal | Recall 0.8702→0.7199 | 0.6923→0.4586 | 0.7407→0.4715 | 拒绝 |
| full hard-negative focal | Recall 0.8702→0.7801 | 0.6923→0.4305 | 0.7407→0.4886 | 拒绝 |
| 成熟最终行 focal | -0.425pp / FDR +4.213pp | -1.019pp / +0.312pp | +0.152pp / +1.339pp | 拒绝 |
| 成熟最终行 BCE | -0.024pp / -0.042pp | -0.788pp / +0.090pp | +0.254pp / +0.681pp | 拒绝 |
| 1.98% 背景+BCE | -0.024pp / -0.068pp | -0.741pp / +0.173pp | +0.152pp / +0.802pp | 拒绝 |
| 有界最终行 | -0.024pp / -0.072pp | -0.741pp / +0.173pp | +0.152pp / +0.802pp | 拒绝 |
| 有界空间分支 | +0.139pp / -0.076pp | -0.371pp / +0.109pp | +0.559pp / +0.914pp | 跨域拒绝 |
| 成熟 teacher 蒸馏 | -0.965pp / +0.638pp | -1.715pp / +0.115pp | -0.457pp / +0.657pp | 拒绝 |

表内为 pooled 工作点的配对差值，准入还同时检查平台三粗类等权口径和每粗类
floor。“Normal 正、Hard 负”不算有效模块。

## 5. 代码索引

### 5.1 评估、阈值与准入

- `src/rsdet/evaluation/platform_protocol.py`
- `src/rsdet/evaluation/macro_risk_v2.py`
- `src/rsdet/evaluation/module_admission.py`
- `src/rsdet/evaluation/background_stress.py`
- `src/rsdet/evaluation/error_route.py`
- `src/rsdet/postprocess/thresholds.py`
- `scripts/analyze_macro_risk_v2.py`
- `scripts/analyze_formal_anchor_threshold_transfer.py`
- `scripts/audit_metric_protocol_migration.py`

### 5.2 训练与候选模块

- `src/rsdet/innovation/quality_aware_loss.py`
- `src/rsdet/data/background_regularization.py`
- `src/rsdet/submission/vehicle_rescue.py`
- `src/rsdet/submission/same_arch_support.py`
- `scripts/train_selective_classifier_finetune.py`
- `scripts/train_mature_background_distillation.py`
- `scripts/train_macroshift_full.py`
- `scripts/train_external_initialized_y5_fine.py`
- `scripts/train_official_quality_head.py`

### 5.3 数据、审核与服务器工具

- `scripts/build_background_100mp.py`
- `scripts/build_background_review_sheets.py`
- `scripts/compile_background_visual_review.py`
- `scripts/freeze_background_100mp.py`
- `scripts/freeze_sentinel_b.py`
- `scripts/build_ship_error_review.py`
- `scripts/render_ship_error_review.py`
- `scripts/decide_ship_training_direction.py`
- `docs/server/MACROSHIFT_FROZEN_7STEP_SERVER_RUNBOOK_20260901.md`
- `scripts/server/run_macroshift_ship_quality_single_gpu.sh`
- `scripts/server/run_selective_classifier_single_gpu_screen.sh`
- `scripts/audit_selective_classifier_checkpoint.py`
- `scripts/audit_spatial_classifier_residual_checkpoint.py`

### 5.4 未评估原型

`src/rsdet/models/ultralytics_adapter.py` 中的 `coarse_purity_sqrt` 从 YOLO26 原始 25 类
logits 重建粗类纯度，飞机旁路，Ship/Vehicle 使用固定几何平均分数。状态为
`implemented_not_evaluated_not_admitted`；只为保留工作轨迹而入库，不得在 Docker 中开启。

## 6. 结果文档链

建议阅读顺序：

1. `reports/experiments/S1280_FULL_CANDIDATE_AND_PLAN14_CLOSURE_20260902.md`
2. `reports/experiments/YOLO_CAPACITY_SCALE_2X2_PLAN_20260902.md`
3. `reports/experiments/IMPROVEMENT_PLAN14_MACROEXPERT_AND_GATE_AUDIT_20260901.md`
4. `reports/experiments/FORMAL_ATTEMPT1_ABSOLUTE_SCORE_FREEZE_20260831.md`
5. `reports/experiments/FORMAL_HIDDEN_DISTRIBUTION_INFERENCE_AND_PROXY_V1_20260901.md`
6. `reports/experiments/IMPROVEMENT_PLAN13_AUDIT_AND_NEXT_ACTION_20260901.md`
7. `reports/experiments/MACROSHIFT_FROZEN_7STEP_IMPLEMENTATION_AND_LOCAL_RESULTS_20260901.md`
8. `reports/experiments/MACROSHIFT_ATTEMPT2_CANDIDATE_TOURNAMENT_20260901.md`
9. `reports/experiments/HERA_GUARD_BREAKTHROUGH_80_85_EXECUTION_20260901.md`

机器可读的本地复算摘要：

- `reports/experiments/improvement_plan13_local_analysis_v1.json`
- `configs/experiments/formal_anchor_threshold_transfer_v1.json`
- `configs/experiments/macroshift_final_baseline_v1.json`
- `configs/experiments/macroshift_vehicle_rescue_v1.json`
- `configs/experiments/macroshift_ship_objectness_quality_v1.json`

`outputs/`、checkpoint、大预测和数据集按规则不进 Git。关键统计、SHA、服务器路径和决策已
抽取到上述文档，代码可从冻结清单重生资产。

## 7. 弃用与保留边界

- 历史 P/Y/R/HERA 报告保留以便复现，不代表当前准入。
- 任何标记 `rejected`、`diagnostic_only`、`not_admitted` 的路线不得进入 full/Docker。
- 根 `README.md` 只保留在内部研发主线；GitHub 公开分支继续删除该文件。
- 本次收口不包含新训练、Docker 打包、镜像推送或正式提交。
