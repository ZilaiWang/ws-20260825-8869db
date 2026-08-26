# SCOPE 候选级动作天花板最终确认 + 整合实验（2026-08-26）

## 一、本轮核心突破：delta_fp 标签（减 FP 机制）

之前 trust_label（学"位置对但类别错"）的 broken 率 = tp 里 crop 判错 = 5.5%。
本轮发现 **delta_fp<0 标签（学"drop 后减 FP"）的 broken 率仅 1.2%**：

| 候选类型 | delta_fp<0 占比 |
|---|---:|
| tp（不该 drop）| **1.2%**（几乎全 0）|
| fn_wrong（该 drop）| **93.5%** |
| bg（背景）| 6.3% |

delta_fp 标签直接对应"减 FP"这个真正目标，天然避开 broken。

### 可学习性（严格 OOF）

| 特征 | AUC |
|---|---:|
| 基础特征（14+has_oto）| 0.9402 |
| +集合上下文（F5 15 特征）| **0.9505**（AP 0.777）|

## 二、但 frontier 非对称依然无法突破

| 方案 | frontier | 恢复率 |
|---|---:|---:|
| base | 0.9421 | — |
| oracle DROP（全 fn_wrong）| 0.9603 | 100% |
| 硬 drop（prob≥0.9 OER<0.3）| 0.9435 | 7.4% |
| 软融合（oer×(1−0.5×p)）| 0.9436 | 7.7% |

**根本矛盾**：185 个 broken（tp 误 drop）的损失 ≈ 1717 个 corrected（减 FP）的收益。

## 三、最终结论（deploy 候选级动作天花板彻底确认）

1. **oracle 改类收益 = 减 FP**（drop 冗余错类候选，+1.86pp），机制是"批量减 FP"的
   累积效应（10398 个一起 drop，FP 9829→6296，frontier 转折点大幅右移）。
2. **frontier 非对称**：损失 1 个高分 TP 的代价 >> 减少 1 个低分 FP 的收益。
3. **deploy 无法安全获取**：识别"该 drop 的 fn_wrong"本质依赖 GT（位置匹配且类别错），
   deploy 只能用 crop 分类器 + 特征近似，近似误差（broken）在 frontier 非对称下被放大。
4. **天花板 = 0.9436**（软融合最优），oracle 的 +1.86pp 空间无法在 deploy 口径获取。

## 四、方案6 全部 Gates 最终判断

- **Gate 0/1** ✅ 完成（底座 0.9421 + 动作价值可学习 AUC 0.98）
- **Gate 2** ❌ 证伪（pairwise 无协同）
- **Gate 3** ❌ 天花板（crop 精度 + frontier 非对称）
- **Gate 4** ⏳ 待 D3/D4 重跑（但域平衡在 SCOPE 框架下意义有限）
- **Gate 5** ⚠️ 需先训更强 teacher，边际有限
- **Gate 6** 最终集成

## 五、下一步建议

候选级动作（改类/DROP）已触达 deploy 天花板，**真正的增长空间在**：
1. **更强的类别证据**（Gate 5 teacher：更大分类器/336 分辨率）——唯一能降低 broken 率的方向，
   但需要 GPU 训练且边际有限。
2. **集合级证据深挖**（F5 集合上下文 oracle +0.26pp，deploy 只 +0.06pp，gap 0.2pp 待挖掘）。

当前 deploy 最佳 = 0.9436（软融合 delta_fp 信号），oracle 上界 = 0.9642。
