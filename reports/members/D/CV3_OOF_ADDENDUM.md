# D 交付补充：M3 正式 CV3 OOF

D 的模型固定为 RT-DETR-L/1024。本补充把原任务合同中的 D4 正式化：

- 三折均从同一个官方 `rtdetr-l.pt` 独立开始；
- 每折 foundation 固定跑满 120 epoch、seed 42，`val=false`、
  `patience=0`，选择 `foundation/weights/last.pt`；
- held-out fold 不参与逐轮验证、early stop、选模或训练期调参；训练后用
  固定 `last.pt` 生成一次正式外部 OOF；
- 若 Ultralytics 在最终 epoch 产生一次框架内部终局验证，只允许留档，
  不得据此改变 checkpoint、参数或正式结论；
- 每折训练前全量复验 D00 的正式检测数据锁，并把 verification 报告绑定到
  fold metadata；
- held-out fold 以 0.001 候选阈值输出，不超过 300 queries；
- 先完成 M3 自身 OOF，再与 M1 做相同图像/GT 的配对错误分析；
- 不因 GPU 排期晚于 P05/P06，就把 P05/P06 视为 M3 的科学前置依赖。

可执行任务单：

- `docs/server/CV3_OOF_COMMON_CONTRACT.md`
- `docs/server/CV3_DETECTION_DATA_LOCK_TASK_00.md`
- `docs/server/M3_CV3_OOF_TASK.md`

M3 是否进入最终系统由 OOF 互补性、官方 Recall/FDR 和 E 的 10K 时延共同
决定，不能只看 Ultralytics mAP。
