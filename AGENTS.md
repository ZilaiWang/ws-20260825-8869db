# 仓库约定

## 目标和指标

本项目检测 `ship`、`aircraft`、`vehicle`。核心指标为 Overall Recall ≥ 0.85、FDR ≤ 0.20、10000×10000 图像端到端推理 ≤ 20 秒。车辆 IoU 为 0.35，其余为 0.50；重复框计为 FP。

## 修改前必读

1. `README.md`
2. 本文件
3. `docs/DATA_CONTRACT.md`
4. 对应模块和测试

修改公共接口、类别映射或评估假设前，先更新 `docs/DECISION_LOG.md`。

## 目录职责

| 目录 | 职责 |
|---|---|
| `src/rsdet/data/` | 数据清单和划分 |
| `src/rsdet/tiling/` | 切片与坐标恢复 |
| `src/rsdet/models/` | 模型适配器和注册表 |
| `src/rsdet/engine/` | 训练与推理流程 |
| `src/rsdet/postprocess/` | NMS、跨切片融合、校准 |
| `src/rsdet/evaluation/` | Recall/FDR 和端到端计时 |
| `scripts/` | argparse CLI 入口 |
| `reports/` | 可共享的脱敏结果 |

## 稳定契约

- 公共结构：`ImageRecord`、`AnnotationRecord`、`Prediction`、`TileRecord`。
- 内部框：原图像素坐标 `xyxy`；COCO 导出时转 `xywh`。
- `Prediction`：同一图像的 `boxes_xyxy`、`scores`、`labels` 长度一致。
- 模型实现继承 `BaseDetector`，`predict()` 返回 `Prediction` 列表。
- 25 个数据细类必须按 `configs/project.yaml` 归并为三大类后评估。
- 配置使用 YAML；个人路径只写 `configs/local.yaml`。

## 禁止事项

- 不重排 `category_id`，不混用归一化坐标和像素坐标。
- 不把普通 AP 写成官方 Recall/FDR。
- 不把 model forward 时间写成端到端时间。
- 不提交原始数据、测试集、权重、密钥、个人绝对路径和大型缓存。
- 不删除失败实验记录，不伪造未实现功能或实验结果。
- 不直接修改或 push `master`。

## 代码和测试

- Python ≥ 3.10；路径用 `pathlib.Path`；正式输出用 `logging`。
- 公开函数写类型标注和 docstring；未实现功能明确报错并返回非零状态。
- 基础测试必须在 CPU、无原始数据、无模型权重时通过。
- 合并前运行：`compileall`、`pytest`、`ruff` 和五个 CLI 的 `--help`。
- 正式实验字段见 `docs/EXPERIMENT_PROTOCOL.md`。

完成修改后汇报：改动文件、运行命令、测试结果、已知限制、下一步。
