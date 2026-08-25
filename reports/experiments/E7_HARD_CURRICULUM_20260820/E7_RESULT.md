# E7: M3 hard-positive 困难样本课程验证(2026-08-20)

## 实现
- --hard-curriculum: M3 找回且 Y5 漏检的 320 目标(186 图)在训练中重复 1 次(权重 2x);
- 与 COPH(--innovation coph --coph-presence-gain 1.0)叠加训练 fold0 40ep;
- 训练日志: 服务器 /workspace/results/E7-HARD-CUR-40EP/train.log(TRAIN_OOF_PASS)。

## 结果(fold0 完整链: R3 → 全类 NMS → SoftRisk)

| 链 | t=0.1 R/F | t=0.2 R/F | 候选 |
|---|---|---|---|
| Y5 fold0 | 0.9280/0.1375 | 0.9140/0.0944 | 24,742 |
| COPH fold0 | 0.9423/0.1768 | 0.9295/0.1261 | 40,304 |
| **E7(COPH+hard) fold0** | **0.9434/0.1771** | **0.9309/0.1245** | 39,454 |
| E7 − COPH | **+0.11pp / +0.03pp** | **+0.14pp / −0.16pp** | −850 |

## 结论

1. **困难课程增益边际**(+0.1pp)——320 目标/186 图的重复采样在 COPH 候选
   扩增面前被稀释; 与 E4/E5/E6 一致: 规则级增量已耗尽;
2. E7 **不加入 Balanced**(边际, 增加训练复杂度), 保留为可选组件;
3. 机制确认: hard-curriculum 代码链路完整(图级重复 → train.txt 加权 → 训练),
   可复用; 更大增益需要更高重复倍率或课程调度(超出本次预算)。

## 产物

- 权重: 服务器 /workspace/results/E7-HARD-CUR-40EP/(覆盖在 E8-COPH-FOLD0-40EP 目录, COPH fold0 评估产物已单独保存)
- fold0 预测: /tmp/E7-fold0-preds.json(39,454)
- 完整链: /tmp/E7-fold0-safechain.json
