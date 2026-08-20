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
