# V1: Context-Gated Vehicle Seeding fold0 诊断(2026-08-21)

> 方案5 §九: 独立车辆中心种子图, 受控恢复 vehicle 无候选。

## 实现

- DFD 密集前景监督加 `only_classes` 过滤, 只对 vehicle(class 24)生成高斯中心热力图;
- 结构零改动(复用 max_c logit), 训练期启用推理零成本;
- fold0 40ep 训练(从 Y5 fold0 初始化)。

## fold0 诊断结果

| 模型 | 候选 | cand-floor | aircraft_NC | ship_NC | vehicle_NC |
|---|---:|---:|---:|---:|---:|
| Y5 | 24,742 | 0.9693 | 79/6312 | 29/905 | 17/133 |
| DFD(全类) | 54,709 | 0.9804 | 23/6312 | 17/905 | 3/133 |
| **V1(仅车辆)** | **44,374** | **0.9786** | 26/6312 | 25/905 | **9/133** |

## 方案 §九.5 门槛对照

| 门槛 | 目标 | V1 实际 | 判定 |
|---|---|---|---|
| vehicle 无候选率 | ≤10% | **6.8%(9/133)** | ✅ 达标 |
| 新增候选 | ≤1.15x | 1.79x | ❌ 超标(但 < DFD 2.21x) |
| 车辆 Recall | +3pp | 待完整链验证 | ⏳ |

## 关键观察

1. **vehicle NO_CAND 17→9(达标)**: vehicle-only 密集监督受控地救回 8 个车辆 GT;
2. **意外收获**: aircraft NO_CAND 79→26(vehicle-only 监督间接帮助了飞机边界目标);
3. **候选受控性优于 DFD**: 1.79x vs 2.21x, 但仍超 1.15x 门槛, 需调 dfd_gain 或加 density gating(V4);
4. candidate-floor +0.94pp(接近 DFD +1.11pp, 但候选更少)。

## 待验证(完整链 frontier)

- V1 候选 → crop 推理 → R3 融合 → OER → 固定风险 frontier;
- 关键判断: V1 的 candidate-floor 收益能否在 frontier 上保留(DFD 教训: floor 升但 frontier 不升);
- 若 frontier 有增益 → 调 dfd_gain 受控(1.79x→1.15x)并推三折;
- 若 frontier 无增益 → vehicle 方向需 density gating(V4)或独立 head。

## 产物

- 权重: 服务器 /workspace/results/V1-VEHICLE-SEED-FOLD0-40EP/(foundation-3)
- fold0 预测: outputs/Y5-OER-RESTORE/V1-fold0-preds.json(44,374)
