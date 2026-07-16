# 实验记录

正式实验统一追加到 [`leaderboard.csv`](leaderboard.csv)，字段和阈值规则见
[`docs/EXPERIMENT_PROTOCOL.md`](../../docs/EXPERIMENT_PROTOCOL.md)。失败实验也保留一行，
`notes` 写清停止原因。

本目录只提交小型汇总、消融表和结论，不提交模型权重、大型日志或完整预测。
详细产物保存在本地 `outputs/实验ID/`，并由 leaderboard 的 `artifact_ref` 指向。
