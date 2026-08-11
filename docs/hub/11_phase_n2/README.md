# Phase N2：共享对象学生

更新日期：2026-08-10
状态：`v1_superseded_v2_code_ready_replay_optional`

> N2 v1 的训练与评估结果已作废，不得进入 leaderboard 或创新消融。
> 当前权威修复报告为
> [`N2_V2_CONTRACT_REPAIR_20260810.md`](../../../reports/experiments/N2_V2_CONTRACT_REPAIR_20260810.md)。

## 1. v1 作废原因

- oracle positive 错用检测器预测类作真值；
- 未经人工确认的 FP_BG 被自动标成 background；
- 用 `(image_id, score)` 回填学生结果，存在分数碰撞；
- held-out fold 同时选 best checkpoint 和报成绩。

## 2. v2 数据流

```text
N0-EVIDENCE-M1-v2（candidate-specific oracle）
  → proposal crop manifest v2
     ├─ deployable positive：官方匹配 GT 类
     ├─ oracle positive：oracle GT 类
     ├─ clear background：仅人工盲审确认 UID
     └─ 其他 hard negative：审计池，不参训练
  → 三折固定 epoch ConvNeXt-T
  → reclassify / background_reject / joint 分开消融
  → cross-fit 阈值 + pooled/V1.6 macro 评估
```

已重生的当前 manifest 有 20,628 条：19,199 deployable + 1,429 oracle，
背景为 0；3,242 个未确认 hard negative 已跳过。因此它现在只能支持
“重分类”重放，不能支持背景拒识结论。

## 3. 准入

N2 重启后至少同时满足：

- 三折外层纯净，固定 epoch，至少 2/3 folds 同向；
- pooled 硬门槛与完整 4/20/1 macro 同时报告；
- 重分类不破坏舰船/车辆，拒背景不破坏已正确 TP；
- 收益不只来自 TU-160 大组；
- 全候选收益成立后才做困难门控和时延压缩。

## 4. 实现

- `src/rsdet/analysis/proposal_crops.py`
- `src/rsdet/analysis/proposal_reclassification.py`
- `scripts/build_proposal_crop_manifest.py`
- `scripts/train_object_student.py`
- `scripts/reclassify_proposals.py`
- `scripts/evaluate_reclassified.py`
