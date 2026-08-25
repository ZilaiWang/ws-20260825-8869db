# F1: Foreground Rejector(Y5 域负样本)—— 判别力有但 frontier 持平(2026-08-21)

> 方案5 §七.2: crop 教师两阶段开放拒绝, 负样本来自 Y5 proposal 域(解决 bg_gate 负样本不足)。

## 实现

- 复用 train_bg_gate.py(ConvNeXt-T 粗类前景门控), 但负样本从 N2-CFG 的"白名单
  clear_background"改为 **Y5 proposal 域的 FP_BG**(近分布结构化背景);
- 结构化背景 = FP_BG 且 crop_top1_class ∈ {MS/QHS/FSC/SU-24}(crop 幻觉类);
- manifest: 正样本 20,391(TP)+ 结构化背景 15,000 + 普通背景 11,711;
- 三折训练(4ep, 冻结前三阶段), 输出 foreground logit(shared + coarse residual)。

## 判别力

| 指标 | 值 |
|---|---|
| fg_logit TP 均值 | 1.4234 |
| fg_logit FP_BG 均值 | −0.7449 |
| fg_logit vs y5_score 相关 | 0.4443 |

fg_logit 有判别力(TP vs FP_BG 差 2.17), 但与 y5_score 中等相关(0.44)。

## OER frontier 验证

| OER | R@FDR=.12 | R@FDR=.11 |
|---|---:|---:|
| 14特征(baseline) | 0.9616 | 0.9591 |
| 14 + fg_logit | 0.9614 | 0.9592 |
| Δ | **−0.0002** | +0.0001 |

## 结论: F1 foreground logit 作为独立特征无效(被 OER 覆盖)

1. fg_logit 的判别力(TP 1.42 vs FP_BG −0.74)已被 OER 现有特征(y5_score + has_oto +
   d4 + crop 证据)吸收——OER 的"是否真实目标"信息已饱和;
2. **"轻证据拼接"第五次失败**(C3 反事实 +0.0006 / C6 硬背景 +0.0000 / C7 listwise
   −0.0008 / V1 车辆 −0.79pp / F1 fg_logit −0.0002);
3. fg_logit 与 y5_score 相关 0.44, 信息大量重叠。

## 未探索: F1 三分类(结构化背景专门类)

方案 §七.2 主张三分类(真实目标/结构化背景/普通背景), 当前实现是二分类。
结构化背景(像 MS/QHS 的跑道线)是专门混淆类, 若单独建模(三分类 softmax 而非
二分类 BCE), 可能提供 y5_score 没有的判别信息——但需进一步验证。

## 产物

- scripts/f1_fg_rejector_manifest.py / f1_fg_rejector_infer.py / f1_verify_oer.py
- 权重: 服务器 /workspace/results/F1-FG-REJECTOR/bg_gate_fold{0,1,2}_final.pt
- outputs/Y5-OER-RESTORE/f1-fg-logits.json
