# N2-CFG：粗类条件式前景门控 实现与计划

日期：2026-08-14
状态：`code_ready_local_pending_gpu_and_n0_csv`
权威依据：《改进方案 1》（2026-08-14 冻结），尤其第 2、3、5 节

## 1. 目标与假设

> 对 R1-6 保留下来的舰船/车辆候选，原始图像中的对象证据能提供与检测器
> 置信度互补的前景判断；在不改变类别和框的情况下，安全删除一部分真实
> 背景误报（FP_BG）。

只做 `background_reject`，不改类、不改框、不恢复候选。舰船/车辆正式门控，
飞机 shadow 旁路（输出逐条不变）。

## 2. 门控位置

```
M1 → Y1-C2 → 飞机 proposal-domain CE → D4 → 飞机同细类 NMS@0.50
  → N2-CFG（仅 ship/vehicle 生效）
  → 最终输出
```

部署阶段改放在 E 全局对象唯一化之后（每个 global object 只判断一次）。

## 3. 冻结设计

| 维度 | 冻结值 |
|---|---|
| 输入 | context_1.25 224×224 单视图 crop（proposal bbox 长边 ×1.25 正方形） |
| 附加输入 | R1-6 score `s`、当前粗类 `c` |
| 模型 | ConvNeXt-T shared trunk → shared head → 3 粗类 residual head |
| 前景概率 | `p_fg = sigmoid(z_shared + z_c)` |
| 损失 | 粗类宏均衡 BCE |
| 采样 | 50% fg / 50% audited clear bg；正样本粗类/细类近似等概率；负样本按 fold×score-bin×group 分层 |
| 正样本 | deployable_positive（R1-6 官方 TP）；oracle_positive 属 N2-P 范畴，首轮不进入 |
| 负样本 | 仅 N0 盲审 `clear_background` 白名单 |
| 训练 | 快筛 freeze_backbone；正式 freeze_first_three_stages；固定 5 epoch、final checkpoint、单 seed |
| 校准器 | `q = σ(α·logit(s) + β·logit(p_fg) + γ_c)`，α,β≥0，单一 `τ_drop` |

## 4. S0/S1/S2 快筛

| 编号 | 模型 | 目的 |
|---|---|---|
| S0 | 仅 R1-6 score 单调校准（β=0） | 排除收益只是重调阈值 |
| S1 | shared head + score（单一 γ） | 判断图像前景证据是否存在 |
| S2 | coarse head + score（γ_c） | 判断粗类条件化是否必要，主候选 |

## 5. 已实现代码（本地，无需 GPU）

| 文件 | 内容 |
|---|---|
| `src/rsdet/analysis/background_gate.py` | 粗类映射、context_1.25 扩展、校准器、删除规则、S0/S1/S2 |
| `src/rsdet/analysis/background_gate_manifest.py` | 训练 manifest 构建（TP 归因 + clear_bg + context 扩展） |
| `src/rsdet/models/background_gate_classifier.py` | ConvNeXt-T shared trunk + 3 粗类 residual head |
| `scripts/build_bg_gate_manifest.py` | manifest CLI |
| `scripts/train_bg_gate.py` | 三折 BCE 训练（含均衡采样） |
| `scripts/infer_bg_gate_logits.py` | 候选 context crop → 前景 logit |
| `scripts/evaluate_bg_gate.py` | S0/S1/S2 校准器拟合 + cross-fit 评估 |
| `configs/experiments/n2_cfg_background_gate_v1.yaml` | 冻结合同 |
| `tests/test_background_gate.py`、`test_background_gate_fit.py` | 18 项核心逻辑测试 |

**已验证（本机 CPU）**：
- R1-6 候选归因与报告逐项一致：TP 19470，FP_BG 1539 / FP_CLS 777 /
  FP_DUP 104 / FP_LOC 62，FN_MISS 624；
- manifest 构建：19470 deployable_positive，1539 unconfirmed_fp_bg 待白名单；
- context_1.25 正方形扩展与 formal `context_1p25` 语义一致；
- 完整测试套件 603 passed + 5 skipped，无回归。

## 6. 待服务器执行（GPU + N0 CSV）

1. **等 B 回传 `manual_review_decisions.csv`** → `compile_fp_bg_review.py` 产出
   `clear_background_whitelist.csv`（一致率 ≥0.85 编译门槛；0.90+κ≥0.75 科学门槛）；
2. `build_bg_gate_manifest.py` 用白名单重建 manifest（负样本到位）；
3. `train_bg_gate.py` 三折训练（freeze_backbone，5 epoch）；
4. `infer_bg_gate_logits.py` 三折推理前景 logit；
5. `evaluate_bg_gate.py` 跑 S0/S1/S2 cross-fit；
6. 按门禁判定。

## 7. 门禁（《改进方案 1》3.2 / 3.4）

- pooled `FP_BG` 减少 ≥10%（≥154），舰船 ≥15%、车辆 ≥10%；
- Overall Recall 下降 ≤0.2pp，任一粗类 ≤0.5pp；
- 车辆 / HM / LQS 零 TP 损失；舰船 TP 损失 ≤4 且不集中单类；
- 飞机 TP/FP/FN 逐条一致；
- ≥2/3 fold 同向；S2 在相同 Recall 约束下必须优于 S0。

任一停止条件触发即停，不继续搜网络/loss/阈值。

## 8. 回退

feature flag `enable_n2_cfg=false` 时输出与冻结 R1-6 prediction parity 完全
一致；每个被删候选记录 `proposal_uid / r1_score / fg_score / q / coarse /
tau_drop / checkpoint_sha / config_sha`。
