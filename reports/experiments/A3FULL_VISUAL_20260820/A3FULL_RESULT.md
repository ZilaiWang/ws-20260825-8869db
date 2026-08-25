# A3-full: 可观测性视觉特征(2026-08-20)

> 方案4 §五.3: 补 local contrast / blur / edge energy / tile edge distance 视觉质量特征。

## 特征提取(65,301 候选, 全量三折)

| 特征 | 均值 | 含义 |
|---|---:|---|
| contrast | 0.200 | 目标 vs 环带对比度 |
| blur | 510.7 | Laplacian 方差(高=清晰) |
| edge_energy | 64.4 | Sobel 梯度均值 |
| edge_dist | 0.091 | 到图边缘归一化距离 |

## 路由器价值验证(严格 fold cross-fit)

- 路由器"预测 crop 对错" AUC: 基础特征 **0.8649** → 加视觉特征 **0.8699**(Δ +0.0051);
- 视觉特征区分度(blur 最明显): crop 对 blur=419.8 vs crop 错 294.7(清晰目标更易判类);
- 但 short_edge / crop_margin 已部分覆盖"清晰度"信息, 视觉特征边际收益很小。

## 两个重要结论

1. **视觉特征边际价值小, 不投入**: +0.005 AUC, 已被现有几何/crop margin 特征覆盖;
2. **更正 A3 口径**: 之前 A3 用 5 折随机 CV 报路由器 AUC 0.9459 是高估
   (随机分折导致同 source group 泄漏); 严格 fold cross-fit 实际 AUC = **0.8649**。
   注意: 这不影响 A3 端到端"全改 0.9584"的结论——全改不依赖路由器筛选,
   且已证 broken=0。

## 决策

- A3-full 视觉特征: 停(边际);
- 路由器筛选: 停(全改已最优, 筛选反而减少 corrected);
- 下一关键: has_oto_support(A5)加入 OER node_validity, 看 frontier 突破 0.9584。
