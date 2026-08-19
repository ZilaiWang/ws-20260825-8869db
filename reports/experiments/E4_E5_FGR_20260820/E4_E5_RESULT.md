# E4 + E5: Proposal-domain FGR + 兄弟机型 pair experts(2026-08-20)

## E4: Proposal-domain aircraft FGR(微调)

- 流程: 复用 R1-1 训练框架(N2-PROPO-CROP-v2 18K aircraft proposal-domain 行,
  5 epoch, 三折, 从 P03-F 初始化 CE 微调)→ 对 Y5 候选 infer(ce 微调 logits);
- 服务器训练 5 分钟/折, infer 20 秒/折;
- **E1+E4 screen**: R=0.9354(+0.06pp vs 纯 E1)/macroR=0.8343(+0.32pp)/
  aircraftR=0.9498(+0.02pp), FDR +0.46pp;
- **完整链 E1E4E2**: t=0.1 R=0.9294(比 E1E2 低 0.3pp)/F=0.1460(低 0.45pp);
- **结论: E4 微调收益边际**——训练数据是 M1 域(N2-PROPO-CROP-v2), 与 Y5 候选域
  存在 gap; 5 epoch 增益有限。E1(P03 零训练)已足够好, 保持为 Safe 链组件;
  E4 若要更大收益需用 Y5 域 proposal 数据重新训练(待后续)。

## E5: 兄弟机型 pair experts(规则版)

- 专家组: G1{TU-160,TU-22,E-3} G2{SU-35,SU-34,SU-24} G3{KC-135,E-8,E-3}
  G4{F-22,F-16,F-15};
- 触发: Y5 标签 ∈ 组 且 crop top-1 ∈ 同组 且 top1≠当前 且 margin/top-prob 达标
  → 修正为 crop top-1(分数不变, 保守);
- **改类质量分析(关键)**:

| 触发条件 | 改类数 | broken TP | 可救 FN_CLS | 净潜力 |
|---|---|---|---|---|
| top≥0.5/m≥0.3 | 4,500 | 369 | 2,860 | **+2,491** |
| top≥0.7/m≥0.4 | 4,143 | 325 | 2,752 | +2,427 |
| top≥0.8/m≥0.5 | 3,775 | 267 | 2,624 | +2,357 |
| E4 微调后 top≥0.7/m≥0.4 | 4,362 | 299 | 2,834 | +2,535 |

- **端到端**: E1E2+E5 t=0.1 R=0.9321(不变)/F=0.1444(-0.61pp)——**净收益 ~0**,
  因为 broken TP 抵消了救回的 FN_CLS;
- **根因**: crop 教师(P03/E4)在 7-9% 兄弟机型上比 Y5 更错——系统性分歧,
  单纯提高置信阈值无法消除(收紧到 0.8/0.5 仍 broken 267);
- **结论**: 规则版 E5 不能独立准入; 正确路径是**学习式触发**(训练"何时改"的
  决策器, 输入 crop logits + Y5 分数 + 位置特征)或与 RCR 组合(改类后由风险头
  把关)。潜力确认(+2,400~2,500 净), 留待 Balanced 阶段。

## 下一步

- E6 稀有类原型(HM/LQS/TU-160/F-22)轻量版
- 验证体系落地(sentinel/漏斗/转移账本)
- Balanced 组合: Safe 链 + 学习式 E5 + E3-v1
