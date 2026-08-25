# R1-9 舰船/车辆固定官方 IoU 后置 NMS 结果（2026-08-14）

状态：`complete_failed_gate`  
结论：停止，不替换 R1-6，不改变 N0-4 v3 的最终候选链。

## 1. 问题与固定协议

R1-6 在飞机对象头重分类和 C2 校准之后，以官方飞机匹配 IoU `0.50`
补做同细类 NMS，在 TP/FN 不变的情况下减少 847 FP。R1-9 检查这一规则
能否安全扩展到仍剩余较高 FDR 的舰船与车辆。

本实验直接读取冻结 R1-6 预测，不训练、不改类别/分数/框、不搜索阈值：

- 舰船细类 0--3：固定 IoU `0.50`；
- 车辆细类 24：固定 IoU `0.35`；
- 飞机细类 4--23：严格旁路；
- 只有 TP、FN、pooled/macro Recall 全部不变才允许准入。

配置：`configs/experiments/r1_major_post_nms_v1.yaml`。

## 2. 结果

| 条件 | 删除预测 | TP 变化 | FP 变化 | FN 变化 | Recall 变化 | FDR 变化 |
|---|---:|---:|---:|---:|---:|---:|
| 仅舰船 | 191 | -54 | -137 | +54 | -0.002580 | -0.005303 |
| 仅车辆 | 86 | -4 | -82 | +4 | -0.000191 | -0.003305 |
| 舰船+车辆 | 277 | -58 | -219 | +58 | -0.002771 | -0.008659 |

组合条件使 `FP_DUP 104 → 2`，但以 58 个 TP 为代价：

- 舰船 pooled Recall：`0.838553 → 0.818419`；
- 舰船 macro Recall：`0.717669 → 0.699519`；
- 车辆 Recall：`0.597015 → 0.587065`；
- 总体 macro Recall：`0.891217 → 0.887915`。

飞机所有指标完全不变，证明实现的类别旁路正确。自动决策门禁中
`exact_tp_fn_recall_parity=false`，因此 `scientific_admission=false`。

## 3. 解释

R1-6 的飞机收益不能被理解为“官方 IoU 等于安全 NMS 阈值”。飞机对象头在
检测器 NMS 之后改变了细类别，确实会产生新的同细类重复；再次 NMS 修复的是
这个后处理顺序问题。

舰船和车辆没有经过该细类重排，且同类真实目标可能密集、相互重叠。直接使用
官方匹配 IoU 做 NMS 会把相邻真实目标一起抑制。虽然 FDR 数值下降，但 Recall
与 TP 的损失违反任务目标，不能准入。

同时，组合后 `FP_BG` 数量的变化不是“确认了更多纯背景”，因为误删 TP 会改变
后续匹配和错误归因。N0-4 v3 仍严格使用冻结 R1-6 母池与已发布的 322 卡盲审包。

## 4. 决策

1. R1-6 继续作为低 FDR 主工作点；
2. 不做舰船/车辆 NMS 阈值网格，避免在同一 OOF 上为 Recall/FDR 反复调参；
3. 舰船/车辆 FDR 转由 N0-4 → N2 背景拒识和未来异构检测证据处理；
4. R1-9 作为负向消融保留，避免将飞机特定机制错误推广到密集目标。

权威本地产物：`outputs/R1-9-SHIP-VEHICLE-POST-NMS/decision.json`。

Gitee Release 小型审计包：
`https://gitee.com/<UPSTREAM_OWNER>/xh-202625/releases/download/v0.2-r1-evidence/R1-9-SHIP-VEHICLE-POST-NMS-return-no-predictions.tar.gz`

SHA256：
`b10c8bb09993d2367ef54997db2c83eaaf71ef5f2481be5e2a62c80ab76ae502`。
