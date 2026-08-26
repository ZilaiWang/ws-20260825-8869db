# 已解锁、等待与停止项

更新日期：2026-07-25  
状态：`current_after_M1_formal_OOF`

完整状态、依赖和执行门禁统一见：

[`reports/experiments/DEFERRED_WORK_REGISTER.md`](../../../reports/experiments/DEFERRED_WORK_REGISTER.md)

## 现在已经解锁

1. N0：M1 cross-fit 阈值、分层稳健性、Pred-OOF 对象证据 manifest；
2. P05 前置：3,303 个 `FP_BG` 的分层人工语义审计；
3. P04：三个关键教师的正式 CV3 缓存复验；
4. P03：tight-224 canonical 正式三折复验；
5. X-CROP-03 / X-BG-01：共享对象学生的重分类、背景拒识与联合消融；
6. D：RT-DETR-L foundation 三折 OOF；
7. E：正式 M1 的 10K 工程闭环。

当前入口：
[`NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md`](../../../reports/experiments/NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md)。

## 仍然必须等待

- M1/M3 配对：等两者同协议 OOF；
- P06-REAL：等待新的边界/尺寸分层证据推翻“定位容量很小”的当前结论；
- P06-DIFF：等确定性 P06-REAL 强基线后仍有空间；
- 最终模型：等 OOF、模块消融和 10K 时延汇合。

## 已停止

- SD1.5 保护目标背景融合；
- 当前 HPR 直接作为正式二阶段模块；
- CleanDIFT 作为唯一教师；
- 在没有真实 OOF 和确定性强基线前训练 bbox diffusion；
- 同时更改模型规模和输入分辨率后宣称单一因素增益。
