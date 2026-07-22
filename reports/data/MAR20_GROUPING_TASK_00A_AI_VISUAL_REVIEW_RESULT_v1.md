# MAR20-GROUPING-TASK-00A AI 视觉审核结果 v1

## 1. 审核范围与身份

- 审核日期：2026-07-21。
- 审核者：Codex，属于 AI 视觉审核，不应表述为独立人工标注。
- 视图方法审核：120/120 个节点、15 张 contact sheet，逐项检查完成。
- 校准 pair 审核：389/389 张匿名卡、98 张 contact sheet，逐项检查完成。
- 初次判断期间未打开 `blind_card_mapping.csv`；先冻结两份决策文件 SHA256，之后才解封映射并复核盲重复冲突。

## 2. 视图方法门禁

正式编译结果：`status=fail`，`formal_view_admission=false`。

| 方法 | 飞机残留率 | 修补伪影率 | 预注册门槛 | 结论 |
|---|---:|---:|---:|---|
| blur | 89.17% | 75.83% | 残留 ≤5%，伪影 ≤10% | 不通过 |
| local_mean | 5.83% | 80.83% | 残留 ≤5%，伪影 ≤10% | 不通过 |
| telea | 13.33% | 75.83% | 残留 ≤5%，伪影 ≤10% | 不通过 |

补充结果：120 个节点均可审核；114 个存在 background tile，其中 1 个 tile 仍含疑似飞机局部，超过预注册上限 0。因此三个目标擦除方法和 background tile 均不能按原协议进入正式 descriptor 流程。

## 3. 校准 pair 门禁

初次盲评后有 7 个盲重复 pair 的标签层级不一致。解封映射后只复核这 7 组，统一相同 pair 的判断层级；复核后：

- 盲重复总数：29；
- 重复一致率：1.000；
- 重复冲突：0；
- 唯一 pair：360；
- 严格正 pair：19；
- 负 pair：285；
- 排除的弱证据/不确定 pair：56；
- 编译状态：`pass_with_insufficient_positive_evidence`；
- `formal_threshold_admission=false`，原因是严格正 pair 少于预注册下限 30。

严格正标签仅包括 `same_frame`、`geometric_overlap` 和 `same_local_site`；`likely_same_airport` 只表示弱机场级相似性，不作为正 pair。

## 4. 科学结论

1. 当前三种擦除方法都会留下明显目标信息或制造强烈方法特征，不能作为可靠的机场场景 descriptor 输入。
2. calibration 集能够验证重复判断的一致性，但严格正 pair 数量不足，不能据此冻结正式相似度阈值。
3. 下一轮不应放宽既有门槛来“通过”；应改进目标移除视图，或转向对目标区域更不敏感的全图/多区域场景描述方案，再生成新的独立校准集。
4. 本轮结果可作为 AI 辅助的初筛与方法淘汰证据。若要作为正式数据划分的唯一依据，建议对 19 个严格正 pair、56 个弱证据 pair 以及 1 个异常 background tile 做一次独立人工复核。

## 5. 产物

- 服务器输入：`view-review/manual_view_review_v2.csv`。
- 服务器输入：`calibration-review/manual_calibration_decisions.csv`。
- 可读工作簿：`MAR20_GROUPING_TASK_00A_AI_VISUAL_REVIEW.xlsx`。
- 审核元数据：`ai_review_metadata.json`。
- 视图编译结果：`compiled/view-review-decision.json`。
- 校准编译结果：`compiled/calibration-compiled/calibration_compile_summary.json`。
- 初次盲评 SHA：`compiled/BLINDED_DECISIONS_SHA256.txt`。
- 重复冲突复核后 SHA：`compiled/POST_ADJUDICATION_DECISIONS_SHA256.txt`。
