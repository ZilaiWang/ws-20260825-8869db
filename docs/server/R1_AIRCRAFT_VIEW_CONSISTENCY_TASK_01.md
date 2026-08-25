# R1-5 服务器任务：飞机 D4 双视图一致性

## 目标

在 R1-1 CE proposal-domain 适配的固定合同上，运行唯一一个双 D4 视图 symmetric-KL
工作点。三折各 5 epoch，随后同时执行 identity/full-D4 OOF、outer cross-fit variant/阈值
选择和冻结 C2。禁止权重网格、best checkpoint、held-out 选模或修改 ship/vehicle。

## 前置与复用

- 复用 R1-1 的 P03/CE/KD bundle、M1 aggregate、P03 fixed-epoch30 checkpoints；
- R1-4 状态必须为 `complete`，确保没有 GPU/evaluator 竞争；
- 配置：`configs/experiments/r1_aircraft_view_consistency_v1.yaml`；
- 驱动：`scripts/server/run_r1_aircraft_view_consistency.sh`；
- 运行环境与 R1-4 相同：torch 2.5.1+cu121 / torchvision 0.20.1+cu121。

## 冻结变量

- 两个不同 D4 视图、双 CE、symmetric KL；
- loss weight `0.20`、temperature `1.0`；
- natural sampling、fixed-last、seed42、5 epoch；
- batch 48 对象，即每 step 96 个视图；
- identity 为主条件，D4 为安全性条件。

## 执行

使用与 R1-4 相同的环境变量，只替换驱动：

```bash
bash scripts/server/run_r1_aircraft_view_consistency.sh
```

任务会自动执行代码 SHA、pytest、ruff、数据审计、真实 smoke、三折训练、三折 D4 推理、
快速精确 cross-fit evaluator、结果审计和无 checkpoint 回传包。已有 run root 时拒绝覆盖。

## 验收

技术验收：

- 3 个 run summary 和 3 个 runtime；
- 每折 inference proposal coverage 完整；
- ship/vehicle 最大指标差为 0；
- `FINAL_GATE_PASS` 和可移植的 package SHA。

科学验收由 `decision.json` 自动给出，但 `formal_admission` 仍为 false：identity 至少提升
aircraft macro Recall 0.0015、净增 20 TP 且 FDR 安全；full D4 相对 CE + D4 的 aircraft
macro Recall/FDR 各不得退化超过 0.002。失败即停止对象头辅助损失路线。

