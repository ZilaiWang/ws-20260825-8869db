# N0-4 v3 盲审：13 个冲突对退回重裁请求

致 B（蔡婕）：

你回传的 322 卡 `manual_review_decisions.csv` 已收到并完成解封编译。**总体完成度 OK（322/322 全部填写，无空标签）**，
但盲重复一致性校验未过门禁（41/54 = 75.9% < 0.85），原因是 13 个 proposal 的原卡与盲重复卡被标成不同标签。
编译脚本因此**不签发 clear_background 白名单**，主线 N2-CFG 仍被阻塞。

请对这 13 对（26 张卡）做一次**复核重裁**：每对给出**最终唯一标签**（一个 proposal 只能有一个最终 label）。

## 一、冲突对清单（26 张卡，`cards/` 目录内）

> 同一 proposal 的两张卡内容相同（一张为原卡、一张为盲重复控制卡）。请两张都看，然后**只回传最终裁定标签**。

| # | proposal_uid | 卡 A（现标） | 卡 B（现标） | fold | 类别 | score |
|---|---|---|---|---|---|---|
| 1 | m1-f0-i1554-p007682 | card-0052（poor_localization） | card-0108（duplicate_or_fragment） | 0 | aircraft | 0.265 |
| 2 | m1-f0-i3226-p013239 | card-0204（clear_background） | card-0313（duplicate_or_fragment） | 0 | aircraft | 0.307 |
| 3 | m1-f0-i4474-p019825 | card-0065（clear_background） | card-0163（plausible_unlabeled） | 0 | vehicle | 0.246 |
| 4 | m1-f0-i485-p002717 | card-0061（clear_background） | card-0171（plausible_unlabeled） | 0 | ship | 0.342 |
| 5 | m1-f0-i799-p005001 | card-0100（plausible_unlabeled） | card-0223（clear_background） | 0 | ship | 0.080 |
| 6 | m1-f1-i1104-p004411 | card-0003（poor_localization） | card-0277（plausible_unlabeled） | 1 | ship | 0.720 |
| 7 | m1-f1-i1593-p006620 | card-0020（poor_localization） | card-0298（duplicate_or_fragment） | 1 | aircraft | 0.278 |
| 8 | m1-f1-i3333-p013709 | card-0140（plausible_unlabeled） | card-0312（duplicate_or_fragment） | 1 | aircraft | 0.252 |
| 9 | m1-f1-i711-p002441 | card-0139（plausible_unlabeled） | card-0217（poor_localization） | 1 | ship | 0.300 |
| 10 | m1-f2-i1754-p007172 | card-0240（plausible_unlabeled） | card-0257（duplicate_or_fragment） | 2 | aircraft | 0.318 |
| 11 | m1-f2-i3179-p012372 | card-0153（plausible_unlabeled） | card-0248（duplicate_or_fragment） | 2 | aircraft | 0.749 |
| 12 | m1-f2-i430-p001890 | card-0042（plausible_unlabeled） | card-0301（poor_localization） | 2 | ship | 0.718 |
| 13 | m1-f2-i4420-p016352 | card-0072（poor_localization） | card-0315（plausible_unlabeled） | 2 | vehicle | 0.065 |

> 注：#2/#3/#4/#5 共 4 对涉及 `clear_background` 裁定，直接影响白名单数量，请重点复核。

## 二、回传格式

在 `cards/` 同目录下提供 `rework_decisions.csv`（或直接改 `conflict_pairs.csv` 的 `label` 列后回传），每行一个 proposal：

```csv
proposal_uid,label,labeler,notes
m1-f0-i1554-p007682,<最终标签>,<你的标识>,
```

只填这 4 列；`proposal_uid` 保持不变。**一个 proposal 只允许一个最终 label。**

## 三、标签定义（同 V3 指南）

- `clear_background`：红框内没有可合理解释为目标的结构，也不是附近已知 GT 的定位误差、重复框或碎片。**只有这一类可进入背景白名单。**
- `plausible_unlabeled_or_ambiguous_target`：红框内存在未标注目标，或分辨率不足。不确定时优先用此标签。
- `poor_localization_of_known_target`：红框对应已知 GT 但偏差大，官方 IoU 未达门槛。
- `duplicate_or_fragment_not_captured`：红框是已知目标的重复输出、部件/碎片。
- `invalid_crop_or_render`：技术问题无法审阅。

## 四、审阅约束（同 V3 指南）

1. 不要打开 `sealed_card_mapping.csv` / `audit_samples.csv` 或任何候选身份表。
2. 按每张卡独立判断，不要回头对照其他卡。
3. 目标是识别**安全负样本**，不是强行证明 FP_BG 大部分为背景。

## 五、裁定后的预期

- 若 13 对重裁后重复一致性回到 ≥0.85（54 对中 ≥46 对一致），编译即通过并签发白名单；
- 若仍不足，我们会在 0.85 门槛下按"冲突即不录用"原则处理（冲突对的 proposal 不进白名单），
  同时评估白名单量是否够 N2-CFG 使用。

谢谢，辛苦复核。
