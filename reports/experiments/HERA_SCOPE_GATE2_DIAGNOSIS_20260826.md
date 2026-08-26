# SCOPE 方案6 Gate 2 诊断：oracle 改类收益的真实机制

## 结论速览

**oracle 改类收益 97% 来自「减少 FP」而非「增加 TP」，且 pairwise 交互证伪。**

- base 0.9421 → oracle RELABEL 0.9608（+0.0186）
- 只改 delta_tp>0 的 135 个候选：+0.0052（TP 已到顶）
- 只改 delta_fp<0 的 3544 个候选：+0.0238（**97% 收益**）
- **oracle 改类候选全部 DROP = +0.0181（≈ RELABEL 的 97%）**

## 真实机制（推翻方案6 Gate 2 的 pairwise 假设）

改类候选 = 位置匹配 GT 但类别错（fn_wrong，10398 个）。这类候选在 base 里是 FP
（类别错匹配不上 GT）。改类后：

1. 变成正确类别，与同 image 的「正确类别候选」IoU>0.5 → 被 NMS 抑制，从 kept 移除；
2. 结果：FP 大幅减少（9829 → 6296），TP 几乎不变（7103 → 7151）。

**等价地，直接 DROP 掉这些冗余错类候选，就能拿到几乎全部收益**（无需 RELABEL）。

## pairwise 交互证伪

对 top 模糊候选做二阶交互项 Δ_ij = uij − ui − uj + u0：
- 1350 对里非零仅 0.4%，且**全是负（互斥），无正协同**。
- 方案6 Gate 2 的 U4「pairwise action utility」不成立——组合收益不存在。

## deploy 的结构性瓶颈（为什么无法复现 oracle 收益）

deploy 无法用 GT 判断「位置匹配但类别错」，只能用 trust_label 路由器近似（AUC 0.97）。
但路由器有 broken（把「位置对类别对」的 tp 误判为需改），破坏力 >> corrected 收益：

| 方案 | 三折 frontier | 恢复率 |
|---|---:|---:|
| base | 0.9421 | — |
| oracle DROP | 0.9603 | 100% |
| trust + y5 门控 + DROP | 0.9434 | 7.1% |
| OER 门控 + DROP | 0.9442 | 11.3% |
| trust + 同类抑制 + DROP | 0.9427 | 3.2% |

**根因（frontier 非对称）**：
- fn_wrong 的 OER median=0.05（低分 FP，在 frontier 转折点之后，减掉不影响 Recall）
- tp 的 OER median=0.98（高分 TP，在转折点之前，broken 掉直接损失 Recall）
- 损失 1 个高分 TP 的代价 >> 减少 1 个低分 FP 的收益

## 关键数据（三折）

- tp(位置对类别对)=24280，fn_wrong(该 drop)=10398，bg(背景)=30623
- tp 里 crop_top1_class != category_id 仅 5.5%，fn_wrong 里 91.6%（类别分歧是良好区分特征）
- fn_wrong 的 OER=0.05 / y5=0.006 / crop_top1=0.90；tp 的 OER=0.98 / y5=0.90

## 结论与方向

1. **Gate 2 关系集合网络（U4 pairwise）证伪**——无需 torch/GPU 训练。
2. **base 0.9421 已接近安全 deploy 上限**：oracle +0.0186 本质是「牺牲高分 TP 换低分 FP」
   的权衡，GT-blind 口径无法安全获取。
3. **deploy 的真正空间在集合级证据**（F5 集合上下文 +0.06pp），而非候选级动作。
4. 下一步建议：跳过 Gate 2 重网络，聚焦 F5 集合上下文的 deploy 化，或 Gate 3 校准
   （确认 conformal LCB 能否进一步压低 broken rate）。
