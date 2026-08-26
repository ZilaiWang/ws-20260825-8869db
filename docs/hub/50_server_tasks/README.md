# 计算任务与大文件交付

更新日期：2026-08-10

服务器不是团队公共资产库，其路径不具有跨成员复现效力。交付规则为：

1. 代码、配置、小型报告和 SHA 登记进 Gitee；
2. checkpoint、完整预测和大型日志走 Gitee Release/仓库附件；
3. 使用者下载后先校验 SHA256，再按登记用途使用；
4. 服务器仅是计算执行地，不是唯一事实源。

当前权威登记表：

[`reports/experiments/ARTIFACT_RELEASE_REGISTER.csv`](../../../reports/experiments/ARTIFACT_RELEASE_REGISTER.csv)

历史服务器路径快照仅用于追溯：

[`reports/experiments/SERVER_ARTIFACT_REGISTER.csv`](../../../reports/experiments/SERVER_ARTIFACT_REGISTER.csv)

## 当前已知资产状态

| 系列 | 当前状态 | 处理原则 |
|---|---|---|
| F00/D00/A00 | 已由正式 M1 完整复验 | 后续 M3/P03/P04/E 继续绑定同一锁，不得重新定义数据或资产 |
| M1 | 正确 YOLO26-s 三折 OOF 已完成 | 正式血缘是三折 `last.pt`；现有 best.pt Release 只能做工程 smoke，三个 last.pt 待按 SHA 发布 |
| P03 | 探索训练完成；CV3 正式复验为高优先级 | P04 后运行 tight-224 canonical；随后接 Pred-OOF crop |
| P04 | 教师 cache、probe checkpoint 留在服务器；CV3 正式复验为高优先级 | 优先复用 cache，先核对 UID/SHA，禁止无证据重提取全量特征 |
| P05 | M1 OOF 已显示 3,303 个 FP_BG | 先在本地完成对象 manifest 和人工语义审计，再提交 cross-fit 训练 |
| P06 | 合成 checkpoint 留存；M1 定位错误容量很小 | 暂缓 P06-REAL，继续停止 P06-DIFF |
| P07 | 扩散 pilot 已停止 | 保留盲评证据，模型资产可在确认归档后清理 |
| MAR20 | K=60 最终映射已进入 Git | 中间大 cache 可在最终回传 SHA 登记后清理 |

## 当前正式 CV3 任务入口

统一调度先读
[`CV3_FORMAL_STAGE_MASTER.md`](../../server/CV3_FORMAL_STAGE_MASTER.md)，
科学设计与依赖见
[`CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md`](../../../reports/experiments/CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md)。

M1 已完成后的推荐顺序为：

1. 本地 CPU：N0 cross-fit、对象证据 manifest、`FP_BG` 分层人工审计；
2. 服务器短任务：`P04_FORMAL_CV3_V2_REPLAY.md`；
3. 服务器 GPU：`P03_FORMAL_CV3_V2_REPLAY.md`；
4. 对象层：X-CROP-03 与 X-BG-01 的共享学生消融；
5. D 支线：`M3_CV3_OOF_TASK.md`，用于补候选和车辆证据；
6. E 支线：立即用正式 M1 开始 `E_10K_PIPELINE_TASK.md`；
7. M3 到齐后才跑 `M1_M3_CV3_OOF_ANALYSIS_TASK_01.md`。

科学顺序以
[`NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md`](../../../reports/experiments/NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md)
为准；旧总纲保留冻结参数和 lineage 合同。

M1 首次启动若出现
`train() got an unexpected keyword argument 'resume'`，先执行
[`M1_CV3_OOF_RECOVERY_R1.md`](../../server/M1_CV3_OOF_RECOVERY_R1.md)，
保留并归档失败现场后再从三折计划起点重跑，禁止在失败目录续训。

## 后续每次计算交付必须登记

- `task_id` 和科学状态；
- 使用的 Git commit 与 split；
- 初始权重、最终 checkpoint 和预测的 Gitee Release/附件引用；
- 所有大文件 SHA256；
- 本地验收报告路径；
- 服务器临时路径可选记录，但不可作交付唯一引用。

不能仅写“跑完了”或只保存聊天摘要。
