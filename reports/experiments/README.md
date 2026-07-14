# 实验记录

存放实验汇总：

- 实验汇总 Markdown 表格
- 小型 metrics JSON（Recall/FDR/运行时）
- 消融表
- 失败实验简短结论

**不存放模型权重和大型日志**。

## 实验输出命名

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
