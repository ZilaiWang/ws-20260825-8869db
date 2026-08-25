# E 当前任务合同：10K 工程闭环

状态：`ready`

E 的唯一主问题是：冻结模型在完整 10K 输入上，切片、批推理、坐标恢复、
跨 tile 去重和序列化是否正确，并且每幅图读取完成后的完整处理是否不超过
20 秒。

## 交付边界

- 首轮以 M1 adapter/checkpoint 跑通；
- M3 进入候选时使用相同 tile 合同补测；
- 最终组合系统冻结后再做最终 p50/p95/max；
- 不训练模型、不调 25 类阈值、不承担 P05/P06。

## 强制证据

- 图像来源和内容 SHA；
- 模型、checkpoint、config SHA；
- tile size、overlap、tile count；
- raw/fused proposal 数；
- 分阶段 p50/p95/max；
- peak VRAM；
- CUDA synchronize 和计时方法；
- warmup 与 measured run 原始台账；
- 每幅图的 20 秒门禁，而非均值门禁。

合成或拼接 10K 图只构成工程 smoke，不得称作官方时延结果。

完整执行合同：

`docs/server/E_10K_PIPELINE_TASK.md`

