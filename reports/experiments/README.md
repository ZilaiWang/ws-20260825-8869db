# 实验记录

正式实验统一追加到 [`leaderboard.csv`](leaderboard.csv)，字段和阈值规则见
[`docs/EXPERIMENT_PROTOCOL.md`](../../docs/EXPERIMENT_PROTOCOL.md)。失败实验也保留一行，
`notes` 写清停止原因。

本目录只提交小型汇总、消融表和结论，不提交模型权重、大型日志或完整预测。
详细产物保存在本地 `outputs/实验ID/`，并由 leaderboard 的 `artifact_ref` 指向。

跨实验状态统一查看：

- [`PRE_INNOVATION_CLOSURE_20260810.md`](PRE_INNOVATION_CLOSURE_20260810.md)：创新阶段立项的当前权威入口，统一官方 V1.6 口径、可信数字、作废项和实验准入；
- [`NEXT_STAGE_TEAM_INNOVATION_EXECUTION_MASTER_v1.md`](NEXT_STAGE_TEAM_INNOVATION_EXECUTION_MASTER_v1.md)：下一阶段 A—E 共同创新的待启用总纲；在 D/M3、E/10K 和 A 的 N0 前置收尾完成前状态为 draft，不覆盖当前执行合同；
- [`NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md`](NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md)：拿到正确 YOLO26-s 正式 OOF 后的当前执行总纲；覆盖证据补齐、P03/P04 正式复验、真实背景拒识、对象学生、M3 与 10K 的新顺序；
- [`CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md`](CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md)：正式 CV3 后 P03/P04、M1/M3 OOF 与 10K 的统一执行总纲；
- [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md)：待解锁、暂缓和停止项目；
- [`ARTIFACT_RELEASE_REGISTER.csv`](ARTIFACT_RELEASE_REGISTER.csv)：可跨成员使用的 Gitee
  Release/附件、SHA256、权重血缘和用途（当前权威登记）；
- [`YOLO_INNOVATION_DIRECTIONS_20260811.md`](YOLO_INNOVATION_DIRECTIONS_20260811.md)：
  依据当前错误分解和本地论文清单形成的 YOLO 改进优先级、单变量实验顺序与停止条件；
- [`Y1_CROSSFIT_CALIBRATION_RESULT_20260811.md`](Y1_CROSSFIT_CALIBRATION_RESULT_20260811.md)：
  M1 严格外层 cross-fit C0-C3 校准结果，C2 类别先验分支准入；
- [`Y2_Y3_FORMAL_IMPLEMENTATION_20260811.md`](Y2_Y3_FORMAL_IMPLEMENTATION_20260811.md)：
  真正 s 级 P2 正式三折合同、历史 P2 命名问题、以及受 Y2 决策门控的单对 IBS 质量实验；
- [`Y2_FAST_SCREEN_RESULT_20260811.md`](Y2_FAST_SCREEN_RESULT_20260811.md)：
  Y2 完整 P2 的 fold0 配对快筛、官方匹配结果和停止结论；Y2 正式三折、Y3 与 P2-Lite 均不启动；
- [`R1_PROPOSAL_RERANKING_RESULT_20260811.md`](R1_PROPOSAL_RERANKING_RESULT_20260811.md)：
  P03-F 对 M1 全部 OOF proposal 的 outer cross-fit 重排；确认 aircraft 细类纠错信号，拒绝当前全类别统一策略并放行 aircraft-only proposal-domain refinement；
- [`R1_AIRCRAFT_REFINEMENT_PLAN_20260812.md`](R1_AIRCRAFT_REFINEMENT_PLAN_20260812.md)：
  aircraft-only 提议域固定短微调、正确教师选择性锚定蒸馏与 D4 概率集成的 2×2 因子实验；ship/vehicle 结构性旁路；
- [`R1_AIRCRAFT_REFINEMENT_RESULT_20260814.md`](R1_AIRCRAFT_REFINEMENT_RESULT_20260814.md)：
  六条件正式结果；CE proposal-domain adaptation + D4 为最强条件，停止现有同视图 KD，并放行类中心与困难对象 D4 成本压缩；
- [`R1_AIRCRAFT_CLASS_CENTER_PLAN_20260814.md`](R1_AIRCRAFT_CLASS_CENTER_PLAN_20260814.md)：
  以 CE identity 为公平参考的训练期类中心角度约束单因素实验；部署模型结构不变；
- [`R1_AIRCRAFT_CLASS_CENTER_RESULT_20260814.md`](R1_AIRCRAFT_CLASS_CENTER_RESULT_20260814.md)：
  identity 主条件与 full D4 横向比较均未优于 CE 工作点；停止训练域动态类中心；
- [`R1_AIRCRAFT_STRUCTURED_ATTRIBUTE_PLAN_20260814.md`](R1_AIRCRAFT_STRUCTURED_ATTRIBUTE_PLAN_20260814.md)：
  以五个俯视可见物理属性辅助 proposal-domain 微调；属性头仅用于训练，单因素比较
  CE identity，并额外检查能否与 full D4 叠加；
- [`R1_AIRCRAFT_STRUCTURED_ATTRIBUTE_RESULT_20260814.md`](R1_AIRCRAFT_STRUCTURED_ATTRIBUTE_RESULT_20260814.md)：
  属性头仅造成极小预测置换且损害飞机 macro Recall；停止类别派生属性辅助监督，转向
  同对象离散旋转视图一致性；
- [`R1_POST_RERANK_NMS_RESULT_20260814.md`](R1_POST_RERANK_NMS_RESULT_20260814.md)：
  在 CE+D4 飞机对象头和冻结 C2 之后，仅对 20 个飞机细类用官方 IoU=0.50
  补做确定性同类 NMS；保持 TP/FN 和所有 Recall 完全不变，减少 847 FP，
  pooled/macro FDR 分别下降 0.03295/0.04518；保留为完整流水线候选；
- [`R1_AIRCRAFT_VIEW_CONSISTENCY_RESULT_20260814.md`](R1_AIRCRAFT_VIEW_CONSISTENCY_RESULT_20260814.md)：
  双 D4 视图对称 KL 短微调的三折结果；identity 净增 78 TP，full D4 净增
  50 TP，但后者增加 32 FP；
- [`R1_VIEW_CONSISTENCY_COMPOSITE_RESULT_20260814.md`](R1_VIEW_CONSISTENCY_COMPOSITE_RESULT_20260814.md)：
  consistency+D4+NMS 形成高召回 Pareto 备选，固定 50/50 概率融合失败；
  主工作点继续保留低 FDR 的 R1-6；
- [`R1_SHIP_VEHICLE_POST_NMS_RESULT_20260814.md`](R1_SHIP_VEHICLE_POST_NMS_RESULT_20260814.md)：
  将固定官方 IoU 后置 NMS 扩展到舰船/车辆会误删 58 个 TP；证明 R1-6 是
  飞机重分类后重复框的特定修复，不可泛化为全类别 NMS；
- [`R1_ADAPTIVE_D4_RESULT_20260814.md`](R1_ADAPTIVE_D4_RESULT_20260814.md)：
  单视图置信度门控只用 38.3% 的 D4 视图计算，但无法保留 full D4 的正式收益；停止继续搜门控阈值；
- [`N0_FP_BG_VISUAL_REVIEW_PACKAGE_20260814.md`](N0_FP_BG_VISUAL_REVIEW_PACKAGE_20260814.md)：
  324 条 FP_BG 的盲化三视图审阅包；在人工白名单形成前背景拒识继续禁用；
- [`N0_FP_BG_FINAL_CHAIN_REVIEW_V3_20260814.md`](N0_FP_BG_FINAL_CHAIN_REVIEW_V3_20260814.md)：
  按 R1-6 最终候选链重建的 322 张盲审卡、修正后的原卡/重复卡一致性门禁
  与背景白名单编译器；
- [`SERVER_ARTIFACT_REGISTER.csv`](SERVER_ARTIFACT_REGISTER.csv)：历史服务器路径快照，
  不再作跨成员交付依据；
- [`M1_CV3_OOF_FORMAL_RESULT_AND_RECOVERY_AUDIT_v2.md`](M1_CV3_OOF_FORMAL_RESULT_AND_RECOVERY_AUDIT_v2.md)：正确 YOLO26-s 正式三折 OOF、关机续跑、官方指标、错误分解与后续准入的当前唯一入口；
- [`M1_CV3_OOF_TRAINING_RETURN_AUDIT_v1.md`](M1_CV3_OOF_TRAINING_RETURN_AUDIT_v1.md)：误用 YOLOv8-s 的首批历史诊断审计，已由 v2 取代；
- [`docs/hub/30_p_series/README.md`](../../docs/hub/30_p_series/README.md)：P0-1 至 P07 总结。

正式服务器共同前置为 `F00 → D00` 数据链与独立的 `A00` 模型资产/环境锁；
M1/M3 必须同时通过 D00、A00，不能仅凭 split manifest 开训。任务顺序见
[`CV3_FORMAL_STAGE_MASTER.md`](../../docs/server/CV3_FORMAL_STAGE_MASTER.md)。

P03 对象 crop 分类系列按以下顺序维护：

1. [`P03-00-crop-classification-master-plan.md`](P03-00-crop-classification-master-plan.md)：总纲、阶段状态和统一决策规则；
2. [`P03-01-linear-probe-results.md`](P03-01-linear-probe-results.md)：预注册细节和冻结特征结果；
3. [`P03-02-fine-tune-results.md`](P03-02-fine-tune-results.md)：全量微调验收、误差分层和分辨率决策；
4. [`P03-03-balance-and-jitter-results.md`](P03-03-balance-and-jitter-results.md)：类别均衡消融、人工 proposal 扰动鲁棒性和 sampler 决策；
5. [`P03-04-seed-stability-results.md`](P03-04-seed-stability-results.md)：多随机种子验收、fold/seed 波动分解和 P03 封板决策。

P03 普通 crop 分类系列已经封板。服务器任务号与报告号各自连续维护，例如 P03 阶段 5 对应服务器任务 `P03_TASK_04`；后续 DINOv2/扩散教师实验另按 P0-4/X-CROP 编号维护，不继续扩展 P03 网格。

P04 教师特征实验由 [`P04-00-teacher-feature-probe-master-plan.md`](P04-00-teacher-feature-probe-master-plan.md) 统一管理。该总纲冻结 DINOv2-S/B、CleanDIFT、条件式 SatDiFuser 的角色，规定 canonical224 信息控制、D4 离线缓存、native/PCA384 双轨、formal split 门禁、互补性比较与停止条件。探索阶段已完成；正式 CV3 v2 现已冻结，当前只按 [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md) 复用 cache 重跑三个关键教师，探索 split 不得用于最终淘汰教师或宣告正式优劣。

以下五份是 **P04 探索阶段的历史执行链**，已完成且不再按原 split 重跑：

1. `P04_TASK_00_ASSETS_ENV_SMOKE.md`：官方资产锁、独立环境和真实模型 smoke；
2. `P04_TASK_01_CONVNEXT_CACHE_EQUIVALENCE.md`：ConvNeXt D4 缓存及 P03 等价门禁；
3. `P04_TASK_02_DINOV2_CACHE.md`：DINOv2-S/B 全量缓存、native/PCA384 探针通路；
4. `P04_TASK_03_DIFFUSION_CALIBRATION.md`：raw DIFT ensemble 与 CleanDIFT 无标签稳定性校准；
5. `P04_TASK_04_CLEANDIFT_FULL_CACHE.md`：CleanDIFT 全量缓存和探索性 native/PCA384 probe。

五份历史任务单位于 `docs/server/`，用于解释服务器 cache 的来源，不是当前
正式入口。当前正式 CV3 v2 只执行：

1. [`P03-P04-FORMAL-CV3-V2-REPLAY-PLAN.md`](P03-P04-FORMAL-CV3-V2-REPLAY-PLAN.md)；
2. `docs/server/P04_FORMAL_CV3_V2_REPLAY.md`。

正式矩阵只含 ConvNeXt、DINOv2-B、CleanDIFT map0 的 native/PCA384 三折，
任一 cache 门禁失败即停止整套 18-run 配对矩阵。
