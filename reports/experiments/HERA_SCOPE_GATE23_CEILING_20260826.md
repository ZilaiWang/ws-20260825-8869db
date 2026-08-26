# SCOPE Gate 2/3 深化：oracle 改类收益的天花板 = crop 分类器精度

## 结论速览

**deploy 改类/DROP 的收益受限于 crop 分类器精度，关系建模（Gate 2 U4）与门控都无法突破。**

oracle 改类收益（+1.86pp，三折）97% 来自「减 FP」，本质是 OER 给「位置对但类别错」的
候选（fn_wrong）虚高分，让它们成为 frontier 高分区的 FP。修复这个虚高分的收益，
完全取决于能否区分「fn_wrong（该 drop）」和「tp（不该 drop）」，而这两者的唯一
deploy 可观测区分信号是 `detector_crop_agree`（crop 分类器与 detector 是否一致）——
它受 crop 分类器精度限制。

## 关键数据（三折）

### oracle DROP 收益按 fn_wrong 的 OER 分段

| fn_wrong OER 区间 | 候选数 | frontier 增量 |
|---|---:|---:|
| [0, 0.03) | 2871 | +0.0000 |
| [0.03, 0.1) | 4351 | +0.0000 |
| [0.1, 0.3) | 1757 | +0.0060 |
| [0.3, 0.5) | 601 | +0.0055 |
| [0.5, 1.0) | 818 | +0.0070 |

收益**全部来自 OER≥0.1 的高分 fn_wrong（3176 个，合计 +0.0185）**。OER<0.1 的 7222 个
fn_wrong 贡献为 0（在 frontier 转折点之后，drop 不影响 Recall）。

### 高分区域的特征重叠（致命）

| 候选类型 | OER | y5_score median |
|---|---:|---:|
| 高分 fn_wrong（该 drop，3176） | ≥0.1 | 0.117 |
| 高分 tp（不该 drop，22790） | ≥0.1 | 0.90 |

y5_score 有 8 倍差距但仍重叠，任何阈值都引入 broken。

### broken 的本质 = crop 分类器误差

- tp 里 `crop_top1_class != category_id` 仅 5.5%（1333/24280）——这 1333 个是「detector 对但
  crop 判错」的候选，被 `detector_crop_agree=0` 误判为 fn_wrong，drop 掉即 broken。
- fn_wrong 里 crop_top1_class 命中 GT 仅 80%——20% 的 fn_wrong 改类也改不对。

## 验证过的修复路径（全部无效或边际）

| 路径 | 结果 |
|---|---|
| fine_correct 标签替代 is_valid | **−0.0043**（贪心匹配非唯一，"位置对类别对但被抢占"非 TP）|
| 加 disagree_x_crop 交互特征 | +0.0000（OER 已用尽特征信息）|
| trust_label 路由器 + 各门控 | 恢复 7-12% |
| pairwise Δ_ij（Gate 2 U4） | 非零 0.4% 且全负，证伪 |

## 结论

1. **is_valid（贪心匹配 TP）是 OER 唯一正确标签**，fine_correct 因贪心非唯一性不可用。
2. **deploy 改类/DROP 天花板 = crop 分类器精度**：
   - tp 上 5.5% 误差 → broken（破坏力 >> 收益，frontier 非对称）
   - fn_wrong 上 80% 命中 → 最多改对 80%
3. **Gate 2 关系集合网络（U4 pairwise）证伪**，门控只能恢复 7-12%，不值得上 torch/GPU。
4. **真正突破方向 = 更强的类别证据**（方案6 Gate 5：条件式高分辨率复核，teacher 特征），
   而非候选间关系建模。

## 下一步建议（按方案6 顺序）

- 跳过 Gate 2 重网络，直接做 **Gate 5 条件式高分辨率复核**：对 top-M 高分模糊候选
  （detector_crop_agree=0 且 OER 高）用 teacher/frozen 特征复核，看能否把 broken rate 压到
  远低于 crop 分类器的 5.5%。
- 这是唯一能突破"crop 分类器精度天花板"的方向。
