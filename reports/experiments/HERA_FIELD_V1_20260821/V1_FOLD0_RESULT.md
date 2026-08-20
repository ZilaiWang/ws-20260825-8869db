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

---

## 完整链 frontier 验证(改类 oracle + NMS, score 排序)

| 链 | R@FDR=.10 | R@FDR=.12 | R@FDR=.15 |
|---|---:|---:|---:|
| Y5 fold0 | 0.9322 | 0.9395 | 0.9478 |
| V1 fold0 | 0.9174 | 0.9316 | 0.9434 |
| **Δ** | **−1.48pp** | **−0.79pp** | −0.44pp |

## 决定性结论: V1 停止(按方案 §九.5 门槛)

1. **vehicle NO_CAND 17→9 达标**(≤10%), candidate-floor +0.94pp, 但这些收益
   在固定风险前沿上**不保留**(V1 −0.79pp@FDR=.12);
2. 候选 1.79x 超标(门槛 1.15x), 新增 ~19,600 候选里 FP 拖累排序;
3. **这是 COPH/DFD 教训的第四次复现**:
   - COPH 存在性正则 −0.62pp / DFD 全类密集 −0.04pp / V1 车辆密集 −0.79pp;
   - **三种"扩候选"方式(存在性正则/密集前景监督/vehicle-only 密集)都无法提升
     固定风险前沿**——救回的真阳 score 偏低, 弱排序抬不动, 反被新增 FP 拖累;
4. 按方案明确门槛"任何只能抬 candidate-floor 却无法改善前沿的版本都停止"。

## 方向修正

- "扩候选"路线(COPH/DFD/V1)彻底证伪;
- 转向**"减 FP"路线**: F1 foreground rejector(压 1899 FP_BG, 方向相反);
- 车辆短板(0.7264)的根因不是"无候选", 而是"低对比真阳排序弱"——需在 OER 层
  加强车辆证据, 而非再扩候选。
