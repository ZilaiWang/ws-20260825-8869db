# A2: OER-v0 对象证据解析器(2026-08-20)

> HERA-YOLO 方案4 §六: 把 SoftRisk/E5/NMS/全局对象层统一为对象证据解析器 OER。

## A1 对象图数据(前置)

- 节点 65,301 个(Y5 全量候选), 特征 = Y5 score + crop 25类 logits(top1/margin/entropy/agree) + 几何 + 局部密度;
- 边 234,127 条, 标签: same_object 34,549 / different_object 18,149 / background 181,429;
- 脚本: scripts/a1_build_object_graph.py。

## 阶段一: node_validity 重排序(HistGB vs SoftRisk vs Y5)

| 分数 | Recall@FDR=0.12 | FDR@Recall=0.94 |
|---|---|---|
| Y5 原始 score | 0.8578 | 0.4410 |
| SoftRisk v0(逻辑回归) | 0.8631 | 0.4048 |
| **OER node_validity(HistGB+丰富特征)** | **0.9036** | **0.2381** |

OER node_validity 相对 Y5 原始 +4.58pp Recall、−20.3pp FDR——丰富特征(crop 证据
+几何+密度)的树模型远强于简单逻辑回归。

## 阶段二: OER 排序 + 全类 NMS(核心成果)

| 组合 | Recall@FDR=0.12 | Recall@FDR=0.15 | FDR@Recall=0.94 |
|---|---|---|---|
| Y5 + NMS | 0.9100 | 0.9228 | 0.2183 |
| SoftRisk + NMS | 0.9132 | 0.9215 | 0.2632 |
| **OER + NMS** | **0.9415** | **0.9481** | **0.1136** |
| (对照) Safe 链 | 0.9357 | 0.9452 | 0.1310 |

## 结论(本批次最重要发现)

1. **OER node_validity 单模型超过 Safe 三模块串行链**:
   - OER+NMS Recall@FDR=0.12 = **0.9415** vs Safe 0.9357(**+0.58pp**);
   - FDR@Recall=0.94 = **0.1136** vs Safe 0.1310(**−1.74pp**);
2. **机制**: HistGB 用 crop 证据(top1/margin/entropy/agree)+ 几何 + 密度联合学习
   "官方 TP 概率", 本质上吸收了 R3 融合(crop 证据)与 SoftRisk(风险)的信息,
   且树模型捕捉了逻辑回归看不到的非线性交互;
3. **印证方案4 核心主张**: "候选/细类/有效性/唯一性必须对象级联合裁决, 而非串行补丁";
4. **edge 学习式去重(阶段二第一版)误删 TP**(串行贪心 + 阈值 0.5 太激进),
   当前全类 NMS(OER 分数排序)已足够好, edge 联合优化留作 OER-v1(DeepSets/GNN);
5. HERA-Core 靶点更新: 从 0.9357 推进到 **0.9415**(仍距 0.945 目标 0.35pp)。

## 产物

- scripts/a1_build_object_graph.py / a2_oer_v0.py / a2_oer_edge.py
- /tmp/a1-object-graph/{nodes,edges}.csv
- /tmp/oer_scores.csv / /tmp/a2_oer_v0.json / /tmp/a2_oer_edge.json
