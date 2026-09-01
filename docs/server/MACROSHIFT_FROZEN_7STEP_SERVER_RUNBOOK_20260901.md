# MacroShift 冻结七步服务器执行手册

本手册只描述服务器恢复后的执行顺序。不得重跑已拒绝的 MacroRisk V2 或 Vehicle replay，不得在
Sentinel-B 冻结之前查看其预测，不得自行扫描融合权重。

> 2026-09-01 执行状态：Ship quality 三折及 FDR-matched 外层回放已完成并拒绝。以下 B–C
> 命令保留用于复现，不得重复运行以寻找随机有利结果；当前不得进入 F–full。

## A. 前置门禁

```bash
cd /workspace/xh-202625
python scripts/audit_metric_protocol_migration.py \
  --registry configs/evaluation/metric_protocol_registry.json \
  --output /workspace/results/MACROSHIFT/protocol_audit.json
python -m pytest -q
python -m ruff check \
  src/rsdet/evaluation/platform_protocol.py \
  src/rsdet/evaluation/background_stress.py \
  src/rsdet/evaluation/error_route.py \
  src/rsdet/evaluation/macro_risk_v2.py \
  src/rsdet/evaluation/module_admission.py \
  src/rsdet/postprocess/thresholds.py \
  src/rsdet/submission/vehicle_rescue.py \
  scripts/analyze_macro_risk_v2.py \
  scripts/analyze_vehicle_reject_rescue.py \
  scripts/audit_metric_protocol_migration.py \
  scripts/build_background_100mp.py \
  scripts/build_background_review_sheets.py \
  scripts/build_ship_error_review.py \
  scripts/compile_background_visual_review.py \
  scripts/compose_macroshift_final_recipe.py \
  scripts/decide_ship_training_direction.py \
  scripts/decompose_coco_oof_errors.py \
  scripts/evaluate_background_100mp.py \
  scripts/freeze_background_100mp.py \
  scripts/freeze_sentinel_b.py \
  scripts/render_ship_error_review.py \
  scripts/train_macroshift_full.py
```

正式结果必须写明 `metric_protocol=platform_observed_20260831`。历史 pooled/fine25 指标可以输出，
但必须标记 diagnostic，不得用于 selection/admission。

## B. Ship official-match quality 三折

输入 NPZ 必须包含：`features`, `detector_score`, `best_same_fine_iou`, `coarse_id`,
`protected_tp`, `active_fp`, `active_mask`, `group_id`, `fold`。特征只能来自可部署证据。

每个 fold 运行：

```bash
python scripts/train_official_quality_head.py \
  --data /workspace/inputs/macroshift/official_quality_features.npz \
  --output-dir /workspace/results/MACROSHIFT/ship_quality/fold_0 \
  --held-out-fold 0 \
  --coarse-filter ship \
  --sampling group_balanced \
  --robustness group_dro \
  --rank-enabled \
  --epochs 20 --batch-size 512 --hidden-dim 192 \
  --residual-limit 1.75 --device cuda:0
```

fold 1/2 只替换 `held-out-fold`、输出目录和 GPU。三折可以并行，但每折只能读取另外两折训练行。
导出的 sparse `candidate_index` 只覆盖 Ship；非 Ship 必须使用原 detector score，不得写 0。

单卡服务器可用冻结执行器顺序运行三折：

```bash
HELD_OUT_FOLD=0 scripts/server/run_macroshift_ship_quality_single_gpu.sh
HELD_OUT_FOLD=1 scripts/server/run_macroshift_ship_quality_single_gpu.sh
HELD_OUT_FOLD=2 scripts/server/run_macroshift_ship_quality_single_gpu.sh
```

三折完成后，用 `scripts/export_sparse_quality_oof.py` 合并 Ship 稀疏分数；该脚本会强制要求 Ship
全覆盖、非 Ship 零覆盖并保持 identity。随后用 `scripts/analyze_ship_quality_cv3.py` 做两折选择、
一折应用。不得直接根据三个 held-out fold 的标签共同挑阈值。

## C. 外层评估和准入

阈值/残差裁剪只用两个训练 fold 冻结，held-out fold 不参与选择。三折合并后至少输出：

- 三大类逐类 macro Recall/FDR；
- gate Recall/FDR 和官方绝对分；
- 每折、每 group、每 fine 结果；
- Background-100MP 的 FP/100MP；
- 3090 平均时延及 specialist/quality 调用比例；
- 相对 frozen baseline 的差值。

模块准入 JSON 字段与 `src/rsdet/evaluation/module_admission.py::ModuleAdmission` 完全一致。最小门禁：

- Recall 增益 ≥ 0.005；
- FDR 差值 ≤ 0；
- absolute score 增益 > 0；
- 任一大类 Recall 最大下降 ≤ 0.005；
- 时延增加 ≤ 2 秒；
- Background FP/100MP 不增加；
- Sentinel-B 存在时必须通过。

## D. Background-100MP

冻结 manifest：

`outputs/MACROSHIFT-BACKGROUND-100MP-FROZEN/background_100mp_manifest.jsonl`

SHA256：`ed3cbbe6952ea5a7792821a316bd3b0ed93888f74a50eda2630f630c9c9020e7`

先将 `images/` 作为推理输入生成 COCO detections，再运行：

```bash
python scripts/evaluate_background_100mp.py \
  --manifest outputs/MACROSHIFT-BACKGROUND-100MP-FROZEN/background_100mp_manifest.jsonl \
  --predictions /workspace/results/MACROSHIFT/ship_quality/background_predictions.json \
  --output /workspace/results/MACROSHIFT/ship_quality/background_100mp.json
```

严禁重新筛掉模型容易误报的背景图；否则测试被污染。

## E. Sentinel-B

只有得到全新来源 registry 才运行：

```bash
python scripts/freeze_sentinel_b.py \
  --registry /workspace/inputs/sentinel_b/registry.csv \
  --forbidden-registry /workspace/inputs/macroshift/all_development_registry.csv \
  --output /workspace/results/MACROSHIFT/SENTINEL-B
```

必须先 freeze，再生成任何 prediction。失败时保持 `formal_admission=false`，不得改用旧组补齐。

## F. 组合和唯一 full

只有独立模块通过后：

```bash
python scripts/compose_macroshift_final_recipe.py \
  --baseline configs/experiments/macroshift_final_baseline_v1.json \
  --module /workspace/results/MACROSHIFT/ship_quality/module_admission.json \
  --output /workspace/results/MACROSHIFT/final_recipe.json
sha256sum /workspace/results/MACROSHIFT/final_recipe.json
```

记录上一步文件 SHA，然后唯一一次：

```bash
python scripts/train_macroshift_full.py \
  --recipe /workspace/results/MACROSHIFT/final_recipe.json \
  --recipe-sha256 <上一步SHA> \
  --manifest /workspace/inputs/cv3/formal_manifest.json \
  --data-root /workspace/data \
  --weights /workspace/assets/y5_base.pt \
  --expected-weight-sha256 <冻结权重SHA> \
  --output-dir /workspace/results/MACROSHIFT/FULL \
  --epochs 160 --batch 12 --workers 8
```

若 accepted_modules 为空、协议错误或 recipe SHA 不一致，launcher 必须失败。禁止绕过。

## G. 停止条件

- Ship quality 任一核心门禁失败：停止，不改跑 fine-tail。
- Background 回退：停止。
- Sentinel-B 不可构造：不宣称已通过；可保留为 blocked，但不能伪造。
- 没有模块独立通过：不训练新 full，继续使用已冻结 incumbent。
