# 实验记录

- [HERA_GUARD_PLAN19_BATIS_EXECUTION_20260905.md](HERA_GUARD_PLAN19_BATIS_EXECUTION_20260905.md)：方案19 BATIS 全链闭环；确认 Safe Fusion 阈值反转和网格相位敏感性真实存在，但 H1/H2、K=8 稀疏重居中、overlap320 与 Ship 跨细类去重均未跨原生/Natural/Trial 测试稳定增益；按停止条件不启动 K=16/随机相位微调，生产继续保持 P40 Safe Fusion + Aircraft-D4；
- [HERA_GUARD_PLAN18_APRR_AND_TILING_AUDIT_20260905.md](HERA_GUARD_PLAN18_APRR_AND_TILING_AUDIT_20260905.md)：方案18离线APRR资格赛与大图差距审计；Ship支持虚警爆炸、Vehicle过度掉Recall，按预注册门禁停止；原生连续图whole-vs-tiled未见切片退化，但10×10伪大图人工缝将Ship/Vehicle推向相反方向，从此降级为工程压力测试；
- [HERA_GUARD_ATTEMPT4_DEEP_DIAGNOSIS_20260905.md](HERA_GUARD_ATTEMPT4_DEEP_DIAGNOSIS_20260905.md)：方案17后续只读深挖；确认Vehicle有效统计仅21个站点、task-vector无逐折同FP Pareto、TP/背景FP分数重叠，短OOF不能选full绝对阈值；D4同进程AB/BA复测后预计仅净增约0.11--0.13分；
- [HERA_GUARD_ATTEMPT4_PLAN17_EXECUTION_20260904.md](HERA_GUARD_ATTEMPT4_PLAN17_EXECUTION_20260904.md)：方案17两条冻结链；Vehicle class-row task-vector拒绝，P40+Aircraft-D4-only边际保留，完整实现、门禁、结果和深层诊断入口；
- [P40_SHIP_VEHICLE_RECALL_20260904.md](P40_SHIP_VEHICLE_RECALL_20260904.md)：保留飞机正向候选；P40漏检容量诊断、细类阈值Hard +0.016停止、弱类整图RFS Hard -1.309停止、targeted EQLv2三折Normal正向但Hard -0.960停止；RFS/EQL/EFL方法族关闭；

正式实验统一追加到 [`leaderboard.csv`](leaderboard.csv)，字段和阈值规则见
[`docs/EXPERIMENT_PROTOCOL.md`](../../docs/EXPERIMENT_PROTOCOL.md)。失败实验也保留一行，
`notes` 写清停止原因。

本目录只提交小型汇总、消融表和结论，不提交模型权重、大型日志或完整预测。
详细产物保存在本地 `outputs/实验ID/`，并由 leaderboard 的 `artifact_ref` 指向。

跨实验状态统一查看：

- [FIXED_PROXY_LITE_AND_P40_VIEWS_20260903.md](FIXED_PROXY_LITE_AND_P40_VIEWS_20260903.md)：当前默认旧Hard筛选→正向才Sentinel确认；三个补框配方停止；P40+CE及view-consistency飞机精识别两套测试正向，单/八视图成本、细类诊断与逐框回放已记录；不改变历史失败、不打包；
- [`PAIRED_TREND_REFERENCE_AND_BN_RESULT_20260903.md`](PAIRED_TREND_REFERENCE_AND_BN_RESULT_20260903.md)：17879固定160+40/A/B/anchor全部完成；新A开发方向+3.244而确认−10.194，官方趋势可靠性未通过；19864的train-only BN开发−2.962，停止且不看确认、不打包；完整结果、环境修复、错误结构及产物索引；
- [`P40_NEXT_IMPROVEMENT_AUDIT_20260903.md`](P40_NEXT_IMPROVEMENT_AUDIT_20260903.md)：正式76.601之后的方案13–15逐项核对、A漏检诊断和BN事前设计；实跑闭环以上一项为准，固定BN配方已拒绝；
- [`PAIRED_TREND_GPU_AND_DIRECTION_AUDIT_20260903.md`](PAIRED_TREND_GPU_AND_DIRECTION_AUDIT_20260903.md)：17879基线启动记录、预声明S1024→P40趋势合同；历史质量贡献Hard +6.402、Sentinel +2.379与正式+4.622同向，但新A确认反向，完成结果见最上方实跑报告；
- [`PAIRED_TREND_IMPLEMENTATION_20260903.md`](PAIRED_TREND_IMPLEMENTATION_20260903.md)：固定流程的数据/代码落地记录，3,136/673/672图、25类覆盖、来源隔离、A/B入口及白天86项相关CPU测试；后续GPU状态见上一项，不能以已见full诊断替代；
- [`EVALUATION_WORKFLOW_SIMPLIFICATION_20260903.md`](EVALUATION_WORKFLOW_SIMPLIFICATION_20260903.md)：最初测试链审计与“一套配对验证+一套部署回归”的依据，当前落地状态见上一项；
- [`FORMAL_ATTEMPT2_P40_PLATFORM_RESULT_ANALYSIS_20260903.md`](FORMAL_ATTEMPT2_P40_PLATFORM_RESULT_ANALYSIS_20260903.md)：正式v2.0/ID3953为76.6010分、读取时第16名、三硬门全过；完整本队API响应、七子分、TP/FP/FN、相对v1净增4.468分与召回代价、本地同源约85分不可作预估的差距分析；
- [`IMPROVEMENT_PLAN15_SCALEROUTE_EXECUTION_20260903.md`](IMPROVEMENT_PLAN15_SCALEROUTE_EXECUTION_20260903.md)：方案15、P40 full40/40及含泄漏诊断（Hard84.965/Sentinel85.278，非官方预估）；3913框入口一致，已由用户提交并获得正式76.6010分；背景误检2→10风险仍保留；
- [`P40_SUBMISSION_FREEZE_20260903.md`](../submission/P40_SUBMISSION_FREEZE_20260903.md)：本次唯一P40镜像/权重SHA、导出与封装修复、验收边界、带镜像身份防错的用户提交流程；
- [`S1280_CV3_FINAL_ANALYSIS_20260903.md`](S1280_CV3_FINAL_ANALYSIS_20260903.md)：S1280 160e 全量权重、三折 outer-CV、共同阈值因果诊断、细类错误结构、purity 否决和下一步最小高价值路径的最终入口；
- [`S1280_FULL_CANDIDATE_AND_PLAN14_CLOSURE_20260902.md`](S1280_FULL_CANDIDATE_AND_PLAN14_CLOSURE_20260902.md)：S1280 单折配对尺寸/错误分析、三卡 160e 全量候选、方案14逐项闭环与提交前硬验收；最终 CV3 结论以上一项为准；
- [`IMPROVEMENT_PLAN14_MACROEXPERT_AND_GATE_AUDIT_20260901.md`](IMPROVEMENT_PLAN14_MACROEXPERT_AND_GATE_AUDIT_20260901.md)：方案14六类 MacroExpert-M、互斥路由、fold0 实施证据，以及 40ep/160ep 与小样本 Hard/Sentinel 门禁分辨率自查；
- [`YOLO_CAPACITY_SCALE_2X2_PLAN_20260902.md`](YOLO_CAPACITY_SCALE_2X2_PLAN_20260902.md)：MacroExpert/DEIM-HCL 复核后的标准 25 类 `S/M × 1024/1280` 同长度四格合同、官方三粗类宏平均门禁与 GPU 驱动；
- [`PEER_DEIM_HCL_REPLAY_20260901.md`](PEER_DEIM_HCL_REPLAY_20260901.md)：同赛道公开 DEIM 解耦查询 + HCL 方案的严格单因素复现、固定 Hard/Sentinel-B 压力测试与主线集成门禁；
- [`PRE_INNOVATION_CLOSURE_20260810.md`](PRE_INNOVATION_CLOSURE_20260810.md)：创新阶段立项的当前权威入口，统一官方 V1.6 口径、可信数字、作废项和实验准入；
- [`../HERA_GUARD_PRECHECK_AND_FAST_SCREEN_20260826.md`](../HERA_GUARD_PRECHECK_AND_FAST_SCREEN_20260826.md)：修正后的官方同源评估、PAV-V1 三折、PAV-V2 监督合同修复、MAR 代理与 D3/D4 泄漏身份审计；当前结论为 PAV/MAR 不进入正式部署链；
- [`IMPROVEMENT_PLAN7_EXECUTION_CLOSURE_20260826.md`](IMPROVEMENT_PLAN7_EXECUTION_CLOSURE_20260826.md)：《改进方案7》阶段 0–4 的统一完成状态、正式保留/停止项、10K 工程边界和全链文件索引；
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
- [`../members/B/B_STAGE_FINAL_REPORT_v1.md`](../members/B/B_STAGE_FINAL_REPORT_v1.md)：
  三粗来源族与尺寸分位重排的 cross-fit 固定逐图预算负向消融；不代表逐机场
  代理组校准结论，规则已停止；
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
