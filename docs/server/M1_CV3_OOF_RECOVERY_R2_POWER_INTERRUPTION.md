# M1-CV3-OOF-R2：fold2 外部断电恢复补充协议

日期：2026-07-25  
故障类型：用户主动关闭旧实例，训练进程被外部终止  
原任务：`M1-CV3-OOF`

## 1. 现场结论

原运行没有模型、代码、数据或 CUDA 故障：

- fold0、fold1 均完成 160 epoch、低阈值 OOF 推理和
  `fold_metadata.json`；
- fold2 完成 134 epoch，在第 135 epoch 的 77/260 batch 被外部关机
  中断；
- fold2 的 `last.pt` 可完整反序列化，checkpoint 内部 `epoch=133`，
  optimizer 和 EMA 均存在；
- `train.log` 没有 Traceback、OOM、非有限 loss 或 adapter 错误；
- 正式权重、D00 数据锁和 formal crop manifest 的 SHA-256 均与冻结值
  一致。
- 克隆实例使用另一张同型号 RTX 4080 SUPER；与原环境锁相比，重算结果只有
  `nvidia_smi.uuid` 和由此派生的 lock fingerprint 不同。驱动
  `595.71.05`、显存、Python、全部冻结包、CUDA 12.1、cuDNN、模型权重和
  数据均一致。原环境锁保留，并为恢复段单独创建不可变环境锁。

原 fold2 的 checkpoint 可以作为本次外部关机恢复点，但必须先完整冻结中断
现场；它不得被当作最终正式 checkpoint。

## 2. 本次恢复的唯一例外

`CV3_OOF_COMMON_CONTRACT.md` 和 `M1_CV3_OOF_TASK.md` 原本规定任一 fold
中断后整批重新开始。负责人在审阅 checkpoint 完整性后批准一次外部关机续跑
例外：

1. fold0、fold1 是彼此独立、已完成且已经冻结的训练和 OOF 结果，原字节
   保留；
2. 中断 fold2 的 checkpoint、日志、结果表、配置和数据门禁完整复制到独立
   归档目录，不删除、不覆盖；
3. fold2 从 epoch 134 保存的 `last.pt` 恢复，重新开始未完成的 epoch 135，
   继续到固定终点 160；第 135 epoch 已运行但未保存的 77 个 batch 不计入
   恢复状态；
4. 数据、split view、seed、模型、输入尺寸、epoch、优化器、代码锁、权重
   SHA 和推理阈值全部不变；
5. 三折聚合前重新验证 fold0/1 关键产物的逐文件 SHA，确保恢复过程没有
   改写已完成折；
6. 最终科学状态记为
   `formal_with_power_interruption_resume_amendment`，
   不伪称为“同一不可变批次从未中断”。

这一例外不引入跨折 checkpoint、验证集调参或重复择优。恢复点来自 fold2
自身固定 epoch 的 `last.pt`，optimizer、EMA 和 AMP scaler 均随 checkpoint
恢复，因此 OOF 的统计定义保持成立。

## 3. 冻结常量

```text
model: YOLO26-s
seed: 42
input: 1024
epochs: 160
resume: true（仅 fold2、仅本次外部关机恢复）
checkpoint: last
low score threshold: 0.001
max detections: 500

yolo26s.pt:
646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b

D00:
03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a

formal crop manifest:
a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128
```

## 4. 路径约定

```text
正式结果：
/workspace/results/M1-CV3-OOF

中断现场快照：
/workspace/results/M1-CV3-OOF-INTERRUPTED-20260725-FOLD2-E134

聚合：
/workspace/results/M1-CV3-OOF-aggregate

运行状态与恢复证据：
/workspace/results/M1-CV3-OOF-ops

恢复段环境锁：
/workspace/results/M1-CV3-OOF-ops/RECOVERY_R2_MODEL_ASSET_ENV_LOCK.json
```

中断目录至少保留：

- 原 fold2 的 134 行 `results.csv`；
- 80 MB 未 strip 的 `last.pt`；
- 中断在 epoch 135、batch 77/260 的 `train.log`；
- dry-run、配置、数据锁验证和 prepared-data 元数据；
- 全量归档文件 SHA-256。

## 5. 启动与完成门禁

恢复脚本：

```text
scripts/server/run_m1_cv3_recovery_r2.sh
```

启动前必须满足：

- GPU 无 M1/M3 训练进程；
- fold0、fold1 各 160 行 `results.csv`，且 metadata 状态为
  `fold_delivery_complete`；
- fold2 恰为 134 个完整 epoch，checkpoint `epoch=133`；
- aggregate 和正式回传包尚不存在；
- 三个冻结输入 SHA 正确；
- 主仓与模型仓代码锁、测试及环境验证仍能通过。

完成时必须满足：

- fold2 恰有 160 个完整 epoch，续跑段原始摘要明确记录
  `resume=true`；
- 另保留一份兼容既有聚合器的全程 lineage 摘要；其中必须显式包含
  `recovery_amendment.resume_used=true`，不得隐藏恢复事实；
- 三折 metadata 均为 `fold_delivery_complete`；
- fold0/1 恢复前后关键文件 SHA 完全一致；
- 三折 OOF 恰覆盖 4,481 张唯一图，允许零预测图但不能漏行；
- aggregate 四件套完整；
- 回传包排除 checkpoint 和 prepared data，并生成 SHA-256；
- 恢复状态为 `complete`，最终恢复协议记录 `resume_used=true`。

任何门禁失败都保留现场并停止。不得临时改变 batch、epoch、模型、seed、
阈值或输入尺寸。
