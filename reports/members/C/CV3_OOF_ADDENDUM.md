# C 交付补充：M1 正式 CV3 OOF

本补充不改变 C 已完成的 M1 foundation 结论，只定义正式三折交付。

C 负责：

- 使用主仓库生成的三个 `split_view.json`；
- YOLO26-s/1024 每折都从同一个官方 `yolo26s.pt` 独立开始；
- 每折 foundation 固定跑满 160 epoch，`val=false`、`patience=0`，选择
  `foundation/weights/last.pt`；held-out fold 不参与逐轮验证或选模；
- 若 Ultralytics 在最终 epoch 产生一次框架内部终局验证，只允许留档，
  不得据此改变 checkpoint、参数或正式结论；
- 不恢复 rare-rebalance 或 HPR；
- 对 held-out fold 输出阈值 0.001 的完整预测；
- 每折训练前全量复验 D00 的正式检测数据锁，并把 verification 报告绑定到
  fold metadata；
- 保留 resolved config、环境、checkpoint、预测和运行时间 SHA。

C 不负责：

- 修改正式 CV3；
- 选择正式阈值；
- P03/P04 教师实验；
- P05/P06 二阶段模块；
- 10K 融合参数选择。

可执行任务单：

- `docs/server/CV3_OOF_COMMON_CONTRACT.md`
- `docs/server/CV3_DETECTION_DATA_LOCK_TASK_00.md`
- `docs/server/M1_CV3_OOF_TASK.md`

验收终点是 `M1-CV3-OOF-aggregate/oof_metadata.json` 状态为
`complete_downstream_ready` 且 `downstream_admission=true`，4481 张图恰好
覆盖一次。旧 dev_v1 checkpoint 只保留历史，
不能替代三折重训。
