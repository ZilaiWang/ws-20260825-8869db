# N2 对象学生 v2 合同修复与收尾

日期：2026-08-10
状态：`code_repaired_replay_required_before_use`

## 1. 为什么原 N2 结果作废

N2 v1 存在四个会直接污染科学结论的问题：

1. `oracle_positive` 使用检测器预测类作训练标签，而不是 oracle 匹配 GT 类；
2. 未经人工确认的 `FP_BG` 全部自动标成 background；
3. 结果以 `(image_id, score)` 回填候选，分数碰撞时可对错对象；
4. held-out fold 同时用于选 best checkpoint 和报最终指标，不是外层纯净评估。

因此 `N2_OBJECT_STUDENT_EXECUTION_20260809.md` 中的训练数字和模块增益不得进入
leaderboard、总结或后续消融。

## 2. v2 合同

- 候选唯一键：`(image_id, source_prediction_index)`；
- `deployable_positive` 标签：官方匹配 GT 细类；
- `oracle_positive` 标签：`oracle_gt_category`；
- hard negative：只有人工盲审为 `clear_background` 的 UID 可标 background=25；
- 未确认 hard negative：保留在审计池，不进入当前训练 manifest；
- checkpoint：固定 epoch 的 `final_checkpoint.pt`，held-out fold 只评估、不选模；
- 模式语义：`reclassify` 只改细类，`background_reject` 只删背景，
  `joint` 才同时做两者；
- 科学等级：重放完成前统一为 `exploratory_level_e`，`formal_admission=false`。

## 3. 已完成的 CPU 重放

`outputs/N2-PROPO-CROP-v2/proposal_crop_manifest.csv` 已按修复合同生成：

| 项 | 数量 |
|---|---:|
| N0 v2 候选 | 23,870 |
| 进入当前 manifest | 20,628 |
| deployable positive | 19,199 |
| oracle positive | 1,429 |
| 人工确认 background | 0 |
| 未确认 hard negative（跳过） | 3,242 |

按 held-out fold 记录数为 7,245 / 7,092 / 6,291。当前这份 manifest 可用于
“纯重分类”重放，不能回答背景拒识的收益。

## 4. 重启条件

N2 不是创新阶段开始的前置阻塞。若后续重启，按以下顺序：

1. 先完成 N0-4 v2 人工盲审，冻结 `clear_background` UID 及 SHA；
2. 重生 proposal manifest；
3. 三个外层 fold 分别从同一 ImageNet 权重训固定 epoch 学生；
4. 交付低阈值原始 logit，分别评估 reclassify / reject / joint；
5. 重新 cross-fit 阈值，同时报 pooled 门槛、V1.6 4/20/1 macro、三折方向和时延。

## 5. 实现索引

- `src/rsdet/analysis/proposal_crops.py`
- `src/rsdet/analysis/proposal_reclassification.py`
- `scripts/build_proposal_crop_manifest.py`
- `scripts/train_object_student.py`
- `scripts/reclassify_proposals.py`
- `scripts/evaluate_reclassified.py`
- `tests/test_proposal_crops.py`
- `tests/test_proposal_reclassification.py`
