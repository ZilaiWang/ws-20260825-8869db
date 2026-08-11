# YOLO 创新快速验证流水线

## 1. 目的与边界

从 Y2 开始，新的 YOLO 结构不再默认直接运行三折 × 160 epoch。统一采用：

> 同折短训配对筛选 → 目标机制与官方指标联合门禁 → 仅胜者进入正式 CV3。

快筛只回答“是否值得投入完整算力”，不能形成正式模型入选结论，也不能替代
`Y1_Y2_Y3_FORMAL_EXECUTION_20260811.md` 中的三折协议。

## 2. 三层漏斗

| 层级 | 数据与训练 | 典型耗时 | 结论 |
|---|---|---:|---|
| S0 技术门禁 | 构图、权重迁移、真实前向、data lock | 1–5 分钟 | 代码能否进入实验 |
| S1 配对快筛 | fold0，M1S 与候选各 40 epoch，seed42 | 约 1–2 小时 | 停止、补 fold1 或进入正式队列 |
| S2 边界复核 | 仅边界候选追加 fold1 的同样配对快筛 | 约 1–2 小时 | 要求 2/2 折方向一致 |
| F 正式验收 | 仅入围者三折 × 160 epoch + 严格 OOF | 约 10–12 小时 | 可以形成正式准入结论 |

S1 的控制组 `M1S` 只需首次生成，后续在训练合同、fold、seed、环境与数据 SHA
不变时复用；候选不得与 160 epoch 的 M1 直接作早期训练公平性比较。

## 3. 当前 Y2 快筛

- 控制：`yolo26s.yaml`，40 epoch；
- 候选：`yolo26s-p2.yaml`，40 epoch；
- 两者都从同一 `yolo26s.pt` 独立初始化；
- fold0、seed42、1024、batch12、AdamW 和数据增强完全相同；
- `close_mosaic=5`，对应正式 160 epoch 合同的最后 12.5% 关闭比例；
- 推理均为 conf=0.001、IoU=0.70、max_det=500 的 held-out fold 推理。

快筛比较两类证据：

1. 各自同折探索工作点的官方 Recall/FDR、宏平均 Recall/FDR；
2. 车辆 Recall、candidate-floor Recall 和完全无候选车辆数。

## 4. 自动决策

基础安全条件：

- overall Recall 相对控制下降不超过 0.015；
- overall FDR 相对控制增加不超过 0.02；
- macro Recall 相对控制下降不超过 0.02。

目标信号满足其一：

- vehicle Recall 提升至少 0.02；
- candidate-floor vehicle Recall 提升至少 0.02；
- 无候选车辆减少至少 5 个。

输出动作：

- `promising_for_formal_cv3`：信号强，进入正式队列；
- `promising_for_second_screen_fold`：边界结果，追加 fold1；
- `stop_candidate`：停止该结构，不追加模块。

无论输出何种动作，`formal_admission` 始终为 `false`。只有正式 F 阶段可以改变正式
主线。

## 5. 服务器执行

环境变量与正式 Y2 相同，然后执行：

```bash
bash scripts/server/run_y2_fast_screen.sh 0
```

若 fold0 输出 `promising_for_second_screen_fold`，再执行：

```bash
bash scripts/server/run_y2_fast_screen.sh 1
```

主要产物：

- `Y2-FAST-SCREEN-FOLD0/M1S/`：40 epoch 配对控制；
- `Y2-FAST-SCREEN-FOLD0/Y2S/`：40 epoch P2 候选；
- `screening_result.json`：完整曲线、机制证据与自动决策；
- `Y2-FAST-SCREEN-FOLD0-return-no-checkpoints.tar.gz`：回传包。

## 6. 后续创新统一规则

每个创新只允许一次主变量变化。新的损失、采样、颈部或检测头先接入同一 40 epoch
fold0 管线；不得同时改模型、训练时长和增强。已有快筛控制在资产 SHA 与环境一致时
直接复用。只有目标机制明确、总体安全且门禁通过的候选，才消耗正式三折预算。
