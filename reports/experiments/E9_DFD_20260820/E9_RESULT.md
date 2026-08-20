# E9/B4: DFD 密集前景监督(2026-08-20)

> 方案4 §四.3 训练期密集前景监督: 治疗候选缺失漏检(vehicle/ship hard-blind)。

## 动机(诊断驱动)

- 诊断结论: 当前 HERA-Core(OER+改类+has_oto) Recall@FDR=0.12 = 0.9607,
  剩余瓶颈是**候选缺失**——vehicle(FSC) 17.9% / ship 4.3% 的 GT 完全无候选,
  而细类已饱和(aircraft 98.2% 正确);
- COPH 存在性正则的局限: 只对 TAL 分配的正样本 anchor 做 ``max_c≈1`` 监督,
  但"漏检"目标 anchor 响应太低、TAL 分配不到正样本 → 无梯度。

## 机制

- **独立于 TAL**: 直接用 GT box 生成自适应高斯中心热力图
  (``sigma~box/6``, 最小 ``stride[0]`` 保证小目标覆盖 3x3 邻域);
- 复用现有 head 的 ``max_c logit`` 作前景响应(结构零改动);
- penalty-reduced focal loss 密集监督(软标签 + (1-heatmap)^4 负调制);
- 只在 one2many 分支, 训练期启用推理零成本;
- 实现: ``src/rsdet/innovation/dfd_presence.py`` + ``train_cv3_oof.py --innovation dfd``。

## fold0 40ep 诊断结果

| 模型 | 候选 | cand-floor | aircraft NO_CAND | ship NO_CAND | vehicle NO_CAND |
|---|---:|---:|---:|---:|---:|
| Y5 | 24,742 | 0.9693 | 79 | 29 | 17 |
| COPH | 40,304 | 0.9773 | 32 | 28 | 10 |
| **DFD** | **54,709** | **0.9804** | **23** | **17** | **3** |

- **candidate-floor +1.11pp(vs Y5)/+0.31pp(vs COPH)** —— 超方案4 门槛(+0.5pp);
- vehicle NO_CAND 17→3(−82%), ship 29→17(−41%), aircraft 79→23(−71%);
- **漏检目标性质**(解释为何 DFD 有效): ship 84% 是 MS(导弹, 小目标中位 57px),
  vehicle 尺寸正常(是低对比但特征可学), 密集高斯监督直接补强这些中心区域的前景响应。

## 待验证(完整链 frontier)

- 候选 +36%(54,709 vs COPH 40,304)可能拖累固定风险前沿(COPH 教训);
- 需跑完整链(R3 融合 + 改类 + OER + NMS + has_oto)看 Recall@FDR=0.12
  能否突破 0.9607;
- 若完整链 frontier 提升 → 推三折; 若候选膨胀拖累 → 调 dfd_gain 或加场景预算。

## 产物

- 权重: 服务器 /workspace/results/E9-DFD-FOLD0-40EP/
- fold0 预测: /tmp/DFD-fold0-preds.json(54,709)
- 诊断脚本: scripts/e9_dfd_diag.py

## 完整链验证(fold0 同口径, 固定风险前沿 Recall@FDR)

| 链 | n | R@FDR=.10 | R@FDR=.12 | R@FDR=.15 |
|---|---:|---:|---:|---:|
| Y5 fold0 + SoftRisk | 16,848 | 0.9136 | 0.9197 | 0.9321 |
| DFD fold0 + SoftRisk | 29,789 | 0.9141 | **0.9220** | 0.9299 |
| DFD fold0 + 改类(无SoftRisk) | 24,275 | 0.9054 | 0.9176 | 0.9294 |
| DFD fold0 改类增益 | — | — | **+0.96pp** | — |

## 最终结论

1. **机制验证通过**: DFD 独立于 TAL 的密集高斯监督显著治疗候选缺失——
   vehicle 17→3(−82%)、ship 29→17(−41%)、aircraft 79→23(−71%)、floor +1.11pp;
2. **但固定风险前沿增益边际**: SoftRisk 链 +0.23pp(0.9220 vs 0.9197),
   候选 +121%(24,742→54,709)带来的 FP 膨胀在固定风险前沿上吃掉了大部分 Recall 收益——
   这是 COPH 教训的轻量重演(但 DFD +0.23pp 正向, COPH 是 −0.62pp 负向);
3. **改类在 DFD 上仍有效(+0.96pp)**: 说明 DFD 救回的候选里"位置对细类错"的目标
   可被 crop 教师纠正, 但改类后的分数排序仍是瓶颈;
4. **核心矛盾**: DFD 救回的低对比真阳 score 偏低, 弱排序(SoftRisk 逻辑回归)无法把它们
   抬到 FP 之前 → 需要 OER 强排序(HistGB + crop 证据 + 几何)才能释放 DFD 的 candidate-floor 潜力。

## 下一步决策

| 方案 | 内容 | 成本 | 预期 |
|---|---|---|---|
| **A(推荐)** | DFD 三折训练 + OER 链(改类+has_oto) | ~4h GPU | OER 强排序可能释放 floor +1.11pp 的潜力 |
| B | 调 dfd_gain 降低(0.5 或 0.3)减少候选膨胀 | 单折重训 | 候选更少、FP 更低, 但可能损失 floor 收益 |
| C | DFD + 场景密度预算(方案4 B3) | 结构改动 | 控制候选预算, 治本 |

**建议先做 A**: DFD 的 candidate-floor 收益是真实的(+1.11pp), 问题在排序而非候选;
OER(当前三折 0.9607 的排序核心)正是对症的强排序器。若 OER 链上 DFD 仍无增益,
则 DFD 降级为可选增强, 主提交维持 Y5 + OER。
