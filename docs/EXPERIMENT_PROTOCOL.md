# 实验记录规范

## 必填字段

| 字段 | 说明 |
|------|------|
| `experiment_id` | 唯一实验 ID |
| `owner` | 负责人 |
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
outputs/YYYYMMDD-owner-task-model-tag/
├── config.yaml
├── meta.json
├── metrics.json
├── runtime.json
├── predictions.json
├── train.log
└── error_cases/
```
