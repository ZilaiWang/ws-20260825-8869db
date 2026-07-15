# 实验记录规范

跨成员的最小输入、预测和模型 adapter 约定见
[`INTEGRATION_CONTRACT.md`](INTEGRATION_CONTRACT.md)。不同模型不要求统一训练框架，
但必须交付相同格式的预测和实验元数据。

## 必填字段

| 字段 | 说明 |
|------|------|
| `experiment_id` | 唯一实验 ID |
| `date` | 运行日期 (YYYY-MM-DD) |
| `git_commit` | 代码 commit hash |
| `dataset_version` | 数据集版本 |
| `split_version` | 划分版本 |
| `config_path` | 配置文件路径 |
| `seed` | 随机种子 |
| `model_name` | 模型名称 |
| `pretrained_weight` | 预训练权重来源 |
| `tile_size` | 切片尺寸 |
| `tile_overlap` | 切片重叠量 |
| `score_thresholds` | 各类别分数阈值 |
| `ship_recall / ship_fdr` | 舰船指标 |
| `aircraft_recall / aircraft_fdr` | 飞机指标 |
| `vehicle_recall / vehicle_fdr` | 车辆指标 |
| `overall_recall / overall_fdr` | 总体指标 |
| `runtime_total / runtime_p95` | 推理时间 |
| `peak_vram` | 峰值显存 |
| `notes` | 备注 |

## 硬性规则

1. 无 git commit 的实验不进入正式结果表。
2. 无配置文件的实验不进入正式结果表。
3. 只报 mAP 的结果不能用于比赛方案决策。
4. model forward 时间不能代替完整推理时间。
5. 失败实验也要保留简短结论。

## 输出目录

```
outputs/YYYYMMDD-task-model-tag/
├── config.yaml
├── meta.json
├── metrics.json
├── runtime.json
├── predictions.json
├── train.log
└── error_cases/
```

## 当前基线编号

- `M1`：主线快速 one-stage 基线；
- `M2`：与 M1 同系列的更高容量或更高分辨率基线；
- `M3`：RT-DETR 类备选基线。

实验 ID 示例：`E-M1-model-1024-devv1-seed42`。
