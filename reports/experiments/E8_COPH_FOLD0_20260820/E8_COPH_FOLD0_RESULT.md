# E8: COPH 存在性正则——fold0 完整验证(2026-08-20)

> CophPresenceLoss: 对每个正样本锚点要求 max_c p(c)≈1(类别无关存在性),
> 对抗"细类不确定 → 候选消失"。40ep 短训(从 Y5 fold0 last.pt 初始化, lr0=5e-4)。

## 原始候选量

| | Y5 fold0 | COPH fold0 | Δ |
|---|---|---|---|
| 候选(0.001) | 24,742 | 40,304 | **+63%** |
| fold0 Recall(原始) | - | +0.80pp(TP +59, FN -59) | 细类不确定不再压掉候选 |

## COPH fold0 + R3 融合(P03-F 教师 crop 推理, 40,304 候选)

- 指定变体 R3_fuse_a0.40(与 Y5 三折选中一致): R=0.9178 F=0.1968
- D0(COPH 原始): R=0.9137 F=0.1979 → **macroR +2.0pp, Recall +0.41pp**

## 完整链公平对比(fold0 同口径: R3 → 全类 NMS → SoftRisk)

| 链 | t=0.1 R/F | t=0.2 R/F |
|---|---|---|
| Y5 fold0 + 完整链 | 0.9280 / 0.1375 | 0.9140 / 0.0944 |
| **COPH fold0 + 完整链** | **0.9423 / 0.1768** | **0.9295 / 0.1261** |
| Δ | **+1.43pp / +3.9pp** | **+1.55pp / +3.2pp** |

**关键结论**:
1. COPH 的 Recall 收益在风险过滤(R3+NMS+SoftRisk)后**稳定保持 +1.4~1.6pp**;
2. **COPH t=0.2(R=0.9295, F=0.1261) vs Y5 t=0.1(R=0.9280, F=0.1375):
   Recall 更高且 FDR 更低**——COPH 分数更"诚实"(高分候选更可信), 允许选
   更高工作点, 是 FDR 门禁(≤0.20)下更优的版本;
3. 剩余 FDR 差距(+3~4pp)主要来自 FP_CLS(COPH 保留更多细类不确定框)——
   由 R3 融合部分缓解, 后续可叠加 E4 FGR 微调或学习式 E5。

## 错误分解(t=0.1, 未加链)

| | COPH | Y5 | Δ |
|---|---|---|---|
| TP | 6,806 | 6,646 | +160 |
| FP_BG | 454 | 344 | +110 |
| FP_CLS | 1,427 | 911 | +516 |
| FP_DUP | 455 | 96 | +359 |

FP 增量集中在 FP_CLS(+516, R3 融合职责)与 FP_DUP(+359, 全类 NMS 已压)。

## 产物

- 权重: 服务器 /workspace/results/E8-COPH-FOLD0-40EP/runs/foundation/weights/{best,last}.pt
- fold0 预测: /tmp/COPH-fold0-preds.json(40,304 条)
- fold0 crop logits: /tmp/E8-COPH-CROP-LOGITS/fold_0_logits.npz
- 脚本: scripts/e8_coph_softrisk_verify.py(公平对比); run_safe_chain.py 增 --nms-all/单折CV/图域限制
