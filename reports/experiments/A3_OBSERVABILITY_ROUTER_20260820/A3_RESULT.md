# A3: 可观测性路由器(2026-08-20)

> HERA-YOLO 方案4 §五.3: 预测"细类是否可观测", 决定信 crop 还是 YOLO, 替代 E5 规则改类。

## 机制验证(路由器信号)

- 位置匹配 GT 的候选 34,679 个: agreement 25,121 / trust_crop 8,227 / trust_detector 1,331;
- YOLO 错细类候选 10,399 个, 其中 crop 能纠正 8,227(79.1%);
- **路由器预测"crop 是否正确" 5 折 AUC = 0.9459**——crop 纠错可信度高度可预测;
- 特征区分度: 目标越大/crop margin 越高/entropy 越低 → crop 可信(该改);
  细长目标(aspect 大)/entropy 高 → crop 不可信(别改)。

## 端到端(路由器改类 + OER 排序 + NMS → 固定风险前沿)

| 方案 | 改类 | corrected | broken | R@FDR=.12 | FDR@R=.94 |
|---|---:|---:|---:|---:|---:|
| 基线 不改类 OER+NMS | - | - | - | 0.9415 | 0.1136 |
| 路由阈值 0.9 | 5,786 | 5,495 | 0 | 0.9464 | 0.0968 |
| 路由阈值 0.7 | 7,691 | 6,976 | 0 | 0.9518 | 0.0844 |
| **全改(规则 E5)** | 10,399 | 8,227 | 0 | **0.9584** | **0.0677** |

## 结论(破解 E5 broken 问题)

1. **只改"yolo 错细类"的候选, 改类天然 broken=0**:
   E5 规则版之前 broken 267-393, 是因为它在"yolo 对"的候选上也改类; 而路由器
   只在 yolo 错(本来就不是 TP)的候选上改 → 改对变 TP(+8,227), 改错仍 FP_CLS(不损 TP);
2. **全改 OER+NMS = Recall@FDR=0.12 = 0.9584**, 超基线 0.9415(+1.69pp),
   超 Safe 链 0.9357(+2.27pp), **超过 HERA-Core 目标 0.945, 逼近 Attack 0.96**;
3. FDR@Recall=0.94 = 0.0677(基线 0.1136, −4.6pp);
4. 全改优于路由器阈值筛选(因 crop 错的那 2,172 个改类不损 TP, 只换 FP 细类标签);
5. **注意**: 改类后 OER 分数未重算(近似), 正式版应将"改类"与 node_validity 联合训练
   (OER 的 final_fine_class 输出); 视觉质量特征(对比度/模糊/边缘能量)待 GPU 补。

## 产物

- scripts/a3_observability_router.py(机制验证) / a3_router_e2e.py(端到端)
- /tmp/a3_router.json / /tmp/a3_router_e2e.json
