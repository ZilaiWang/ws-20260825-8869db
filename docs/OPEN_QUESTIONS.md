# 待确认事项

1. 总体指标中的类别错误具体如何计数？（class-aware vs class-agnostic matching）
2. 测试方是否对所有 JSON 中出现的预测框全部计数？
3. 是否存在统一的 score 阈值？
4. 是否限制每张图最大预测数量？
5. 是否允许使用外部预训练权重和外部数据？
6. 是否允许 TensorRT、FP16 和 ONNX 推理？
7. 测试时间是否包含 JSON 序列化？
8. 数据是否存在稀疏标注（未标注的目标）？
9. 测试集是否包含纯背景图（无任何目标）？
10. 正式提交 JSON 应使用 25 个细类 ID，还是三大类 ID？

> 当前评估默认采用 class-aware matching。此假设写入函数文档和实验报告。

## 已确认

- 当前训练数据：`0–3=ship`、`4–23=aircraft`、`24=vehicle`。
- 本地评估先按上述映射归并为三大类；正式规则变化时统一修改 `configs/project.yaml`。
