# E11: DCR²-YOLO Balanced 主提交组合定义(2026-08-20)

## Balanced 组合(只组合已独立过 gate 的模块)

```
Balanced = COPH 候选(存在性正则, 训练期)
         + R3 融合(细类重排, P03-F 教师 crop)
         + 全类 post-rerank NMS(压 FP_DUP)
         + SoftRisk(位置标签 v0, 风险重排)
```

## 各模块 gate 状态

| 模块 | gate | 证据 |
|---|---|---|
| COPH 存在性正则 | ✅ L0/L1 通过 | fold0: 候选+63%, 完整链 R=0.9423(vs Y5 0.9280) |
| R3 融合重排 | ✅ E1 通过 | 三折一致选 R3, Macro +1.86pp(Y5 上) |
| 全类 post-rerank NMS | ✅ E2 通过 | 零 TP 损失, FP -6,597 |
| SoftRisk v0 | ✅ E3 通过 | t≤0.05 Recall/FDR 双改善, AUC 0.95 |
| E4 FGR 微调 | ⚠️ 边际 | M1 域 gap, 不加入 Balanced |
| E5 规则版 pair | ⚠️ 净~0 | 待学习式触发, 不加入 Balanced |
| E6 重点类 | ❌ 无增量 | R3 已内化 |
| E7 困难课程 | ⏳ 待验证 | 与 COPH 叠加训练(fold1/2 后) |
| E9 P2-lite | ❌ 停止 | Y2 快筛 + SAHI 双证据 |
| E10 SparseZoom | ❌ 停止 | 2× 放大仅救 7% |

## 门槛

- Recall ≥ 0.95 / FDR ≤ 0.12(目标起步, 相对 Y5 基线 0.9355/0.1173 再上一档);
- 重点类(TU-160/F-22/HM/LQS)不退化;
- 任一模块互相抵消(组合 < 单独)即失败, 需归因。

## 三折验证流程(L2)

1. COPH fold0/1/2 训练(40ep, 从 Y5 fold 权重初始化)✅ fold0 完成, fold1/2 训练中
2. 三折推理(conf=0.001) → 三折候选
3. 三折 crop 推理(P03-F 教师) → R3 融合
4. 全类 NMS → SoftRisk → 三折完整链评估
5. 与 Y5 三折完整链基线(0.9355/0.1173@t=0.1)公平对比
