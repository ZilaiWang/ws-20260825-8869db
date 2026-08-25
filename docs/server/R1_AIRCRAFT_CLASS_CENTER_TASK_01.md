# R1-2 TASK-01：飞机类中心约束三折快筛

## 目标

在 R1-1 完整验收后，复用其 P03/CE/KD bundles，仅新增三折 class-center
训练与推理，并执行三条件 cross-fit 评估。不得改 fold、训练 epoch、学习率、
阈值网格或输入 manifest。

## 前置

- `R1-1-AIRCRAFT-PROPOSAL-REFINEMENT/FINAL_GATE_PASS` 存在；
- R1-1 三类 bundle 保留；
- P03 三折 checkpoint、N2-v2 manifest、R1 inference manifest、M1 aggregate、
  formal crop manifest、Y1 calibration 和 ConvNeXt 权重沿用 R1-1 路径；
- Python 环境沿用 `/workspace/venvs/p06-cu121`。

## 执行

环境变量与 R1-1 相同，额外指定：

```bash
export PROJECT_ROOT=/workspace/xh-202625-next
export R11_ROOT=/workspace/results/R1-1-AIRCRAFT-PROPOSAL-REFINEMENT
bash scripts/server/run_r1_aircraft_class_center.sh
```

驱动脚本自带不可覆盖目录、锁、代码 SHA、pytest、ruff、数据审计、GPU smoke、
三折训练、D4 推理、cross-fit 评估和无 checkpoint 回传包。

## 必须回传

- `R1-2-AIRCRAFT-CLASS-CENTER-return-no-checkpoints.tar.gz` 及 SHA256；
- `decision.json`、`condition_summary.json`、三个 condition result；
- 三折 `run_summary.json`、runtime JSON、checkpoint SHA 清单；
- 最终状态和失败/重试记录。

## 禁止

- 使用 PSP.Plane、FAIR1M 或 MAR20 bridge 图训练；
- 读取 held-out 指标选 checkpoint；
- 同时加入 D4 蒸馏、属性头、类别均衡或背景负样本；
- 删除 R1-1、P03、M1 等既有资产。
