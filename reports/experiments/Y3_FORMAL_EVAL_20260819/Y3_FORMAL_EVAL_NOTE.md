# Y3-HIER 正式三折评估(补跑, 2026-08-19 00:37)

> 背景: 队列在 Y3 评估段因 formal-crop-manifest 默认路径缺失崩溃(set -e 中止)。
> 修复 run_innovation.sh(显式传 --formal-crop-manifest)后补跑评估+诊断, 未重训。
> 数据: 服务器 /workspace/results/Y3-HIER-CV3-OOF/evaluate_Y3-HIER.json (622257 proposals)

## 结果(低阈值 0.001 原始输出, 未后处理)

| 指标 | Y3-HIER | M1 基线 | Δ |
|---|---|---|---|
| pooled recall | **0.9891** | 0.9316 | +0.0575 |
| pooled fdr | 0.9667 | 0.6489 | (+0.3178, 低阈值无意义) |
| macro_recall | **0.9870** | 0.8961 | +0.0909 |
| macro_fdr | 0.9680 | 0.5534 | (+0.4146, 低阈值无意义) |
| FN_MISS | **127** | 390 | -263 |

## 错误分解(Y3)

- FP: FP_BG 594471 / FP_DUP 6980 / FP_CLS 16 / FP_LOC 86
- FN: FN_MISS 127 / FN_CLS 16 / FN_LOC 86

## 判读

1. **Recall 大幅提升**: pooled +0.058, macro +0.091, FN_MISS 降 67%——层次损失
   显著减少漏检, 与早期单折信号一致(fold0 0.9883 / fold1 0.9929);
2. FDR 高是低阈值原始输出(62 万候选 vs M1 5.5 万)所致, 需 NMS+阈值后处理,
   最终 FDR 以服务器 evaluate 链路的后处理结果为准;
3. FP_CLS 仅 16(极低): 层次损失对细类判别几乎不产生 FP_CLS——与混淆基线
   (M1 FP_CLS 1114)相比大幅改善, 但需注意此时是未后处理候选, 口径不完全同。

## 产物

- evaluate_Y3-HIER.json (服务器拉回)
- diagnose_Y3-HIER.json (服务器, 错误诊断)
- Y3-HIER-CV3-OOF-return-no-checkpoints.tar.gz (44MB, 服务器)
