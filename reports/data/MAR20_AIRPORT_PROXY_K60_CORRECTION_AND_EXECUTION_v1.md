# MAR20 机场代理 K=60：分组层级更正与最终执行说明

## 1. 更正结论

此前生成的 2,882 个 `group_id` 是严格局部同源/局部场景连通分量，不是机场级代理分组。绝大多数图一图一组并非算法异常，而是 pair 检索和人工复核只证明了少量 L0/L1 关系；它无法自动把同一机场中互不重叠的远距离视角连起来。

将这份结果直接交给 B 做 CV3 会保留大量潜在同机场跨折风险。因此已撤销它的正式交付地位，只把它保留为 must-link 约束。

## 2. 两层结果的正确关系

| 层级 | 解决的问题 | 预计组数 | 用法 |
|---|---|---:|---|
| local-scene strict core | 相同画面、重叠画面、确定局部同源不得拆开 | 约 2,920 | 聚类硬约束与泄漏审计 |
| airport proxy K=60 | 将完整 MAR20 组织成机场来源代理域 | 60 | B 的 CV3 `group_id` |

MAR20 公开称完整 3,842 张图来自 60 个军用机场，但没有逐图机场标签。因此 K=60 是有依据的来源数先验，输出仍只能称“机场代理视觉簇”，不能称机场真值，也不提供机场名称。

## 3. 最终算法

1. 使用完整 MAR20 3,842 张图，包含 3,073 张比赛图和 769 张只作结构桥接的原始 MAR20 图；
2. 复用 Round-B 已入选的 `masked_block10_vlad_k32_pca512` 与 `masked_block11_vlad_k32_pca512`；
3. 每路对 0/90/180/270 四个视图取均值并 L2 归一化；
4. 两路等权拼接，再次 L2 归一化；
5. 先将 strict local-scene 组件折叠为不可拆原子；
6. 对组件质心做 cosine average-link 层次聚类，并按 MAR20 最小编号稳定命名为 60 个代理组；
7. 将标签展开到图像，验证 strict component split 为 0；
8. 只把 3,073 张比赛图的 `competition_image_id → group_id` 交给 B。

选择该方案是为了在已有充分特征、几何和人工证据基础上快速收尾。它不再引入 HDBSCAN、LightGlue、更多盲审或模型敏感性循环。block10、block11 与融合分区之间的 ARI、节点到簇心相似度和 margin 会完整报告，但不据此反复修改方案。

## 4. 正式产物

服务器任务完成后，正式主文件为：

```text
outputs/MAR20-AIRPORT-PROXY-K60-v1/mar20_airport_proxy_assignments_target.csv
```

旧文件：

```text
outputs/MAR20-FINAL-GROUPING-v1/mar20_final_group_assignments.csv
```

已经废弃，不得再用于 CV3。

## 5. 边界

这 60 组是基于遮挡目标后的机场背景表征形成的代理域。它比 singleton-heavy 的局部场景分量更符合“机场隔离 CV”的目的，但仍可能把同一真实机场拆开，或把不同机场合在一起。正式报告应写“MAR20 airport-proxy grouped CV3”，不得写“真实 airport-disjoint CV3”。
