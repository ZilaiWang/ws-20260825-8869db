# N0-4 盲审扩审：第 2 批（566 卡）回传请求

致 B（蔡婕）：

第 1 批 322 卡已收到并完成编译，门禁失败原因与 13 冲突对重裁请求已发你
（REWORK 包）。**但即使 13 冲突对全部重裁通过，白名单也只有 30 个
clear_background proposal——远不够 N2-CFG 三折训练**（每折负样本仅 3-10 个/粗类）。

因此启动第 2 批扩审，目标是把白名单补到 ~105-115 个（覆盖 fold 平衡）。

## 一、审阅包

- 文件：`N0-BATCH2-REVIEW-package.tar.gz`（98MB，SHA256 `26479ac7…720cb`）
- 内容：`N0-BATCH2-REVIEW/` 下 566 张卡（`cards/card-0000.jpg …`）+ 142 张
  contact sheet + `manual_review_decisions.csv`（空 label 待填）+ `render_summary.json`
- 卡面格式同第 1 批：红框=待审预测，绿框=已知 GT；每卡从左到右为全图、
  4 倍上下文、1.35 倍紧裁剪

## 二、抽样构成（与第 1 批同口径）

| 粗类 | 正卡 | 说明 |
|---|---|---|
| aircraft | 234 | 覆盖 3 fold，优先补 f2（当前白名单最缺） |
| ship | 245 | 覆盖 3 fold，优先补 f1 |
| vehicle | 87 | 全部（未审池 vehicle 仅 144，已尽量多抽） |

- 正卡 472 + 盲重复 94（20%） = **566 卡**；
- 盲重复规则同第 1 批：独立标注，一致性由编译脚本计算。

## 三、回传要求（同 V3 指南）

1. 只填 `manual_review_decisions.csv` 的 `label` / `labeler` / `notes`，不改 `card_id`；
2. 5 个合法标签同前：`clear_background` / `plausible_unlabeled_or_ambiguous_target` /
   `poor_localization_of_known_target` / `duplicate_or_fragment_not_captured` /
   `invalid_crop_or_render`；
3. 审阅期间不打开 `sealed_card_mapping.csv` / `audit_samples.csv` / 任何身份表；
4. 可按 fold 分批回传（不必等 566 张全完成），但每批需 `label` 全填。

## 四、预期效果

- 按第 1 批 12.1% 命中率，566 卡预期新增 **~68 个 clear_background**；
- 白名单从 30 → ~98，每折每粗类负样本 10-30 个，**足以支撑三折训练与
  paired bootstrap 门禁（0.85 一致性 + 统计功效）**；
- 第 2 批完成后编译白名单 → 立即解锁 N2-CFG。

## 五、两批进度对照（供你安排）

| 批次 | 卡数 | 状态 |
|---|---|---|
| 第 1 批 | 322 | 已回传，13 冲突对待重裁 |
| 第 2 批 | 566 | **本次待审** |

谢谢，辛苦了。两批回传后我这边立即编译签发白名单。
