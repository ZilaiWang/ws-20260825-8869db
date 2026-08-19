# E0: 协议/UID/旁路审计报告(DCR²-YOLO 前置, 2026-08-20)

> 对应 DCR²-YOLO 总纲 §1.1-1.4 与实验队列 E0。
> 规则: 未完成 E0 不启动后续实验。全部计数须守恒、口径唯一。

## 1.1 vehicle IoU 口径审计 —— ✅ 通过(无需修改)

**结论**: 官方 V1.5 要求 vehicle IoU=0.35(aircraft/ship=0.50), 当前配置已正确。

- `configs/project.yaml` → `official_evaluation.iou_thresholds`:
  `ship: 0.50 / aircraft: 0.50 / vehicle: 0.35` ✅
- `src/rsdet/evaluation/protocol.py` 从 project.yaml 读取, 无硬编码;
- 全仓库匹配路径审计(均走 `protocol.iou_thresholds`, 即 0.35 生效):
  - `scripts/evaluate_experiment.py`(主评估)✅
  - `scripts/analyze_m3_paired_oof.py`(配对分析)✅
  - `scripts/build_m3_teacher_evidence.py`(教师证据)✅
  - `scripts/evaluate_bg_gate.py`(N2 门控评估)✅
  - `scripts/analyze_single_cv3_oof.py` / `scripts/run_e_pipeline.py` ✅
- 未发现任何 `vehicle IoU=0.50` 的硬编码匹配;
- 历史 M1/Y5/Y3/Y4 评估数字无需重跑, 口径一致。

**结论**: GPT 指出的 "vehicle 0.50" 在主文档中确有一处笔误(评估口径描述),
实际评估代码从未用错。已在主文档概念中确认 0.35 为准。

## 1.2 N2-CFG aircraft bypass 矛盾 —— ✅ 已修复(shadow 统计层)

**审计发现**:
- 配置合同(`n2_cfg_background_gate_v1.yaml`):
  `active_coarse: [ship, vehicle, aircraft]`(三粗类都训练, aircraft 仅 shadow 评估)
  + `active_coarse_first_round: [ship, vehicle]`(aircraft 部署旁路);
- g2 门禁已正确旁路 aircraft(`COARSE_FP_BG_MIN_RATIO = {ship:0.15, vehicle:0.10}`,
  aircraft `min_ratio=null, pass=null` = shadow 报数)✅;
- **但 g7 zero_tp_loss 统计未旁路**: `evaluate_bg_gate.py` 的 applied_removed
  对全部 coarse(含 aircraft)计数, `run_sealed_admission.py` 的 `_applied_tp_removed`
  未排除 aircraft → 误删的 F-22 TP(score 0.0171)计入 g7 → g7 FAIL。

**修复**(commit 见下):
1. `scripts/evaluate_bg_gate.py`: applied_removed 每条记录加
   `"shadow_coarse": coarse == "aircraft"`(aircraft 删除标记为 shadow);
2. `scripts/run_sealed_admission.py`: `_applied_tp_removed(exclude_shadow_coarse=True)`
   默认排除 aircraft → g7 只统计 ship/vehicle 的真实删除。

**修复后预期**: g7 的误删 TP 从 3 → 2(vehicle + ship 各 1, aircraft 不再计入)。
g7 仍 FAIL(recall_budget=0 的 0 容忍), 但口径正确——aircraft 旁路合同在统计层生效。
(注: 真正过 g7 需团队决策是否给 recall_budget 极小预算, 属门禁设计而非代码 bug。)

## 1.3 M3 hard-positive 对象级 UID —— ✅ 已重建

**审计发现**: 旧 `hard_positives.csv` 1313 行但 `gt_uid` 仅 643 个唯一值,
268 组同 uid 对应多个不同 bbox——gt_uid 退化到 image 级。

**根因**: `build_m3_teacher_evidence.py` 用 `gt.get("uid", f"i{image_id}-gt")` 访问
GT dict, 但 `formal.boxes` 的 GT dict 只有 bbox/category_id, 对象级
`annotation_uid` 在 `formal.objects[(image_id, gt_index)].annotation_uid`。

**修复**: 改为 `formal.objects.get((image_id, gt_index)).annotation_uid`,
输出列同时保留 `gt_uid`(= annotation_uid)与新增 `annotation_uid`。

**验证(重跑 V2)**:
- 行数 1313, `gt_uid` 唯一值 **1313**(0 重复)✅;
- 计数守恒: vehicle 56 / ship 221 / aircraft 1036; tiny 64 / small 854 /
  medium 388 / large 7; fold 0-584-1-324-2-405 —— 与旧版完全一致, 仅 UID 精确化;
- 新产物: `outputs/M3-TEACHER-EVIDENCE-V2/` + 归档
  `reports/experiments/M3_TEACHER_EVIDENCE_V2_20260820/`。

**下游影响**: 双漏×M3 交叉分析、蒸馏样本计数、P2 准入等凡按 gt_uid 去重的
分析, 一律改用 V2。

## 1.4 统一 evaluator 三栏状态规范 —— ✅ 落地(文档规范)

今后所有评估表格固定三列状态(报告层规范, 写入本项目报告模板):
- `candidate_floor`: 只回答"有没有可匹配候选"(低阈值 0.001 口径);
- `deploy_working_point`: 实际部署 Recall/FDR(工作点阈值口径);
- `scientific_status`: formal outer-pure / exploratory OOF / oracle / diagnostic。

已在 GPT 讨论包主文档 + 本报告落实该规范。

## 1.5 其他审计项

- V1.5/V1.6 排名口径: `project.yaml` 已含 `ranking.fine_macro_average: true`
  (大类内细类简单平均), 与 GPT 方案一致 ✅;
- 130/320 oracle 上限: 已确认是 GT-oracle 口径(主文档 1.2 标注"已验证/理论"),
  后续可部署方案走风险头/选择器/蒸馏, 不直接写 0.9803/0.9952 为部署预期 ✅。

## 结论

E0 四项全部闭合: ①IoU 口径本就正确(文档笔误已澄清) ②N2 aircraft 旁路
在统计层修复(g7 不再误伤 aircraft) ③M3 对象级 UID 重建(1313 唯一) ④三栏
状态规范落地。**可以启动 E1**。
