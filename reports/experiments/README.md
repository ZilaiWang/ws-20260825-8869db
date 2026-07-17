# 实验记录

正式实验统一追加到 [`leaderboard.csv`](leaderboard.csv)，字段和阈值规则见
[`docs/EXPERIMENT_PROTOCOL.md`](../../docs/EXPERIMENT_PROTOCOL.md)。失败实验也保留一行，
`notes` 写清停止原因。

本目录只提交小型汇总、消融表和结论，不提交模型权重、大型日志或完整预测。
详细产物保存在本地 `outputs/实验ID/`，并由 leaderboard 的 `artifact_ref` 指向。

P03 对象 crop 分类系列按以下顺序维护：

1. [`P03-00-crop-classification-master-plan.md`](P03-00-crop-classification-master-plan.md)：总纲、阶段状态和统一决策规则；
2. [`P03-01-linear-probe-results.md`](P03-01-linear-probe-results.md)：预注册细节和冻结特征结果；
3. [`P03-02-fine-tune-results.md`](P03-02-fine-tune-results.md)：全量微调验收、误差分层和分辨率决策；
4. [`P03-03-balance-and-jitter-results.md`](P03-03-balance-and-jitter-results.md)：类别均衡消融、人工 proposal 扰动鲁棒性和 sampler 决策；
5. [`P03-04-seed-stability-results.md`](P03-04-seed-stability-results.md)：多随机种子验收、fold/seed 波动分解和 P03 封板决策。

P03 普通 crop 分类系列已经封板。服务器任务号与报告号各自连续维护，例如 P03 阶段 5 对应服务器任务 `P03_TASK_04`；后续 DINOv2/扩散教师实验另按 P0-4/X-CROP 编号维护，不继续扩展 P03 网格。

P04 教师特征实验由 [`P04-00-teacher-feature-probe-master-plan.md`](P04-00-teacher-feature-probe-master-plan.md) 统一管理。该总纲冻结 DINOv2-S/B、CleanDIFT、条件式 SatDiFuser 的角色，规定 canonical224 信息控制、D4 离线缓存、native/PCA384 双轨、formal split 门禁、互补性比较与停止条件。B 的正式同源分组到达前可以完成环境、权重、缓存、无标签稳定性和预注册 probe 通路诊断，但探索 split 不得用于淘汰教师、搜索 layer/timestep 或宣告最终优劣。

P04 当前服务器执行顺序固定为：

1. `P04_TASK_00_ASSETS_ENV_SMOKE.md`：官方资产锁、独立环境和真实模型 smoke；
2. `P04_TASK_01_CONVNEXT_CACHE_EQUIVALENCE.md`：ConvNeXt D4 缓存及 P03 等价门禁；
3. `P04_TASK_02_DINOV2_CACHE.md`：DINOv2-S/B 全量缓存、native/PCA384 探针通路；
4. `P04_TASK_03_DIFFUSION_CALIBRATION.md`：raw DIFT ensemble 与 CleanDIFT 无标签稳定性校准；
5. `P04_TASK_04_CLEANDIFT_FULL_CACHE.md`：CleanDIFT 全量缓存和探索性 native/PCA384 probe。

五份任务单位于 `docs/server/`，必须依次通过门禁；预先写好后续任务不表示允许跳过前置证据。
