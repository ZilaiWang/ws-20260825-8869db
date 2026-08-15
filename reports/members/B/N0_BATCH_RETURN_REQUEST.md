# N0-4 v3 盲审：分批回传请求

致 B（蔡婕）：

主线 N2-CFG（粗类条件式前景门控）三折训练被阻塞在 `clear_background` 负样本上，
而负样本唯一合法来源是 N0-4 v3 盲审的 `manual_review_decisions.csv`（322 张卡）。
为压缩整体时间，请**按 fold 分批回传**，不必等 322 张全部完成。

## 回传批次与顺序

| 批次 | 范围 | 卡片数（约） | 说明 |
|---|---|---:|---|
| 第 1 批 | `fold = 0` 全部卡 + 该 fold 内盲重复卡 | ~107 | 优先，先点亮 Level-E 方向验证 |
| 第 2 批 | `fold = 1` 全部卡 + 盲重复卡 | ~107 | |
| 第 3 批 | `fold = 2` 全部卡 + 盲重复卡 | ~108 | |

> fold 字段在 `manual_review_decisions.csv` 里没有直接暴露（盲化表只有 card_id）。
> 请按 `card_id` 顺序（card-0000 … card-0321）回传即可：盲化包已按 fold 分层抽样，
> 前段即 fold0 密集区。或者更简单——**每完成约 1/3 的卡就回传一次**。

## 每批的最低要求

1. `label` 列全部填写，只能使用 5 个合法标签：
   `clear_background` / `plausible_unlabeled_or_ambiguous_target` /
   `poor_localization_of_known_target` / `duplicate_or_fragment_not_captured` /
   `invalid_crop_or_render`；
2. `labeler` 填审核人标识；`notes` 可选；
3. 盲重复卡（54 张）**独立标注**，不要回头对照原卡——一致性由编译脚本计算；
4. **不要查看** `sealed_card_mapping.csv`、`audit_samples.csv` 或任何候选身份表。

## 主线会怎么用

- 每收到一批，本地跑 `compile_fp_bg_review.py` 校验该批一致性；
- 累计达到 fold 级的 `clear_background` 白名单后，先跑该 fold 的 Level-E 快筛
  （S0/S1/S2 方向验证），不等整包；
- 整包 322 张收齐后，一致性 >=0.85 编译门槛（0.90 + κ>=0.75 为科学放行）
  才进入正式三折训练。

这样你可以边审边交，主线边收边验，两头都不空转。
