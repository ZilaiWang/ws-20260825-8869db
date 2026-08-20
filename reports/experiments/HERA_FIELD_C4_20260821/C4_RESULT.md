# C4: 结构证据(边界闭合/结构延续)—— 空间 grid FPN 无效(2026-08-21)

> 方案5 §六.2: closure/border_crossing 结构证据, 从空间 grid FPN 计算。

## 实现

- 重提取空间 grid FPN(out_size=5, channel-mean 得到 5x5 空间激活图);
- 计算 closure(中心/边缘)/ring_cont(core≈ring)/region_diff/core_energy/ring_homo;
- 225 维 grid → 15 维结构证据(3 尺度 × 5 统计量)。

## 判别力(TP vs FP_BG)

| 证据 | TP | FP_BG | \|Δ\| |
|---|---:|---:|---:|
| p5_closure | −0.267 | 0.294 | 0.561 |
| p5_ring_cont | 0.784 | 0.871 | 0.087 |
| p4_closure | 0.983 | 0.997 | 0.014 |

closure 是判别力最大的结构证据(0.561), 但远弱于 y5_score(差 300 倍)。

## OER frontier 验证

| OER | R@FDR=.12 | Δ |
|---|---:|---:|
| 14特征 | 0.9616 | — |
| 14 + 结构证据 | 0.9618 | +0.0002 |

## 结论: 结构证据无效(FPN 空间信息已被 Y5 建模)

1. 空间 grid 的 channel-mean 结构证据(closure/border_crossing)判别力极弱,
   OER frontier 仅 +0.0002;
2. **根因**: Y5 检测器的 backbone/head 已经用了 FPN 特征做检测/分类,
   空间 grid 的结构信息已被 y5_score 吸收——FPN 不是"检测头没用到的独立信息源";
3. 这印证了 C3 的结论: 从 Y5 FPN 提取的证据(无论全局平均还是空间 grid)
   都无法提供 y5_score 之外的独立信息。

## 唯一正向方向: 集合级证据

今天 8 个方向的完整验证, 唯一正向的是 **C7-lite 集合上下文(+0.21pp)**:
- 集合上下文(同图邻居统计 + 空间邻居)是"候选间关系", 这是 Y5/单框特征
  完全没有的维度;
- 下一步应深挖集合级证据(真 DeepSets/注意力聚合, 而非简单统计)。

## 产物

- scripts/c4_structure_evidence.py
- outputs/Y5-OER-RESTORE/grid-feats/(空间 grid FPN, 225维, 已持久化)
