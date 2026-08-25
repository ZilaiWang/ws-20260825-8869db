# A0: 完整 PR frontier 重算(固定风险前沿, 2026-08-20)

> 对照 HERA-YOLO 方案4 §十二: 主指标从固定阈值改为 Recall@FDR / FDR@Recall。

## 结果(Y5 三折链 vs COPH 三折链, 同口径 R3+NMS-all+SoftRisk)

| 前沿指标 | Y5 三折链 | COPH 三折链 | Δ(COPH−Y5) |
|---|---:|---:|---:|
| Recall@FDR=0.10 | **0.9286** | 0.9207 | **−0.79pp** |
| Recall@FDR=0.12 | **0.9357** | 0.9295 | **−0.62pp** |
| Recall@FDR=0.15 | **0.9452** | 0.9391 | −0.61pp |
| FDR@Recall=0.94 | **0.1310** | 0.1528 | +2.18pp |
| FDR@Recall=0.95 | **0.1754** | 0.2046 | +2.92pp |
| FDR@Recall=0.96 | **0.2709** | 0.2929 | +2.20pp |
| candidate-floor Recall | 0.9715 | 0.9751 | +0.36pp |

## 重点类(候选 floor 口径)

| 类 | Y5 | COPH | Δ |
|---|---:|---:|---:|
| HM | 0.882 | 0.882 | 0 |
| LQS | 0.767 | 0.767 | 0 |
| TU-160 | 0.895 | 0.914 | +1.9pp |
| F-22 | 0.996 | 0.988 | −0.8pp |
| FSC | 0.808 | 0.816 | +0.8pp |

## 结论(直接印证方案4 §2)

1. **COPH 在固定风险前沿上全面劣于 Y5**: 每个 Recall@FDR / FDR@Recall 点都更差;
   COPH 只把 candidate-floor 抬高 +0.36pp, 代价是每个前沿点 +0.6~0.8pp Recall 的损失
   (FP 增加 11,687 拖累排序);
2. **决策**: COPH 降级为 Candidate-Heavy 分支, **取消 Balanced 主提交地位**(与方案4 一致);
3. **当前真实最优前沿 = Safe(Y5 链)**: Recall@FDR=0.12 = **0.9357**;
4. **HERA-Core 靶点**: 在 FDR=0.12 前沿上超过 0.9357(且目标 0.945);
5. COPH 的价值只在个别类(TU-160 +1.9pp/FSC +0.8pp), 但整体前沿被 FP 拖累——
   这正是"候选/细类/风险/唯一性必须对象级联合裁决"的直接证据。

## 产物

- scripts/a0_pr_frontier.py(可复用, 任意预测文件 → 固定风险前沿)
- /tmp/a0_frontier.json
