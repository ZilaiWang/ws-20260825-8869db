# P40 加 Aircraft D4 技术报告证据索引

日期：2026-09-05

用途：供技术报告作者或审阅仓库的模型快速定位当前稳定主线、实验依据、实现代码、冻结资产和结果边界。本文是导航页，不替代各项原始报告；没有列出的探索性分支不属于 P40+D4 主线。

## 1 当前主线一句话定义

当前稳定主线是：**YOLO26-s 的全量 Progressive-40 模型 P40，采用 1024 像素切片、256 像素重叠、1280 网络输入、Safe Fusion 和融合后统一阈值 0.536；仅对 Aircraft 类别 4--23 追加 ConvNeXt-Tiny 八视图一致性分类器 D4，Ship 类别 0--3 与 FSC 类别 24 保持 P40 原输出。**

两项冻结权重：

| 资产 | 用途 | SHA256 | 仓库中的身份依据 |
|---|---|---|---|
| P40 净化部署权重 | 25 类检测、切片与候选生成 | `b0df7981f6ad58fe8eca65fb0deef54feed55caf300c2c219ac1ccb3500c8012` | [P40 提交冻结](../reports/submission/P40_SUBMISSION_FREEZE_20260903.md) |
| Aircraft-D4 full checkpoint | 仅重判 Aircraft 4--23 | `5f1b175f8b2c310a3c0583e652f5cf7cfd444d0b0d74d794a9ac59e4537832d5` | [P40+D4 冻结配置](../configs/experiments/p40_aircraft_d4_full_runtime_candidate_v1.json) |

权重文件本身不进入公开 Git 仓库；路径和 SHA 是身份合同。技术报告写作应区分“Git 中的可复现代码与配置”和“按 SHA 另行归档的大文件资产”。

## 2 最短阅读路径

按下列顺序阅读即可建立完整认识：

1. [P40 正式提交冻结与工程验收](../reports/submission/P40_SUBMISSION_FREEZE_20260903.md)：P40 配方、部署阈值、权重净化、逐框一致性和正式 v2.0 身份。
2. [方案15 Progressive-40 执行报告](../reports/experiments/IMPROVEMENT_PLAN15_SCALEROUTE_EXECUTION_20260903.md)：S1024 到 S1280 的三折消融、full 训练血缘、背景压力与时延。
3. [正式 v2.0 平台结果](../reports/experiments/FORMAL_ATTEMPT2_P40_PLATFORM_RESULT_ANALYSIS_20260903.md)：P40 官方 76.6010 分、三类 Recall/FDR、TP/FP/FN 和七项分数复算。
4. [Aircraft 一致性训练结果](../reports/experiments/R1_AIRCRAFT_VIEW_CONSISTENCY_RESULT_20260814.md)：D4 的来源、三折对照和训练期双视图一致性机制。
5. [正式 v3.0 平台复盘](../reports/experiments/FORMAL_ATTEMPT3_V3_PLATFORM_RESULT_AND_PROXY_POSTMORTEM_20260904.md)：Aircraft-D4 在隐藏集净增 21 TP、减少 21 FP；同时说明当次 Vehicle 专家和第二检测器为何失败。
6. [Attempt 4 决策报告](../reports/experiments/HERA_GUARD_FINAL_TWO_ATTEMPTS_DECISION_20260905.md)：P40+D4 安全线、工程优化结论与当前共享 OTM 攻击候选状态。

## 3 对应官方研究报告章节的索引

### 3.1 研究背景与问题分析

- [项目实验协议](EXPERIMENT_PROTOCOL.md)：数据隔离、实验标识、门禁与可复现规则。
- [数据划分总索引](../reports/data/DATA_SPLITS_MASTER_INDEX_v1.md)：检测与分类视图、CV3 划分和资产口径。
- [正式 v2.0 结果分析第 3--7 节](../reports/experiments/FORMAL_ATTEMPT2_P40_PLATFORM_RESULT_ANALYSIS_20260903.md)：官方类间不均衡、Ship/Vehicle 召回与 Vehicle 隐藏背景风险。

报告可据此描述：25 个细类被聚合为 Ship、Aircraft、Vehicle 三个粗类；样本规模和类别支持差异大；大图切片、小目标、跨来源泛化、细类宏平均 Recall/FDR 与工程时延共同决定得分。

### 3.2 数据集与划分

- [项目数据配置](../configs/project.yaml)：类别表、路径和基础任务配置。
- [数据划分总索引](../reports/data/DATA_SPLITS_MASTER_INDEX_v1.md)：正式 4,481 图、三折视图和来源分组说明。
- [机场代理分组链索引](../reports/data/MAR20_AIRPORT_PROXY_GROUPING_MASTER_CHAIN_INDEX_v1.md)：Aircraft 来源相关分组和防泄漏链。
- [P40 full 输入合同](../outputs/SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-3GPU-R2/frozen_input_contract.json)：4,481 图、无外部训练数据、固定 last 与阈值来源。

### 3.3 总体技术路线

- [P40 full 训练驱动](../scripts/server/run_scaleroute_plan15_progressive40_full.sh)：冻结资产校验、40 epoch S1280 适配、背景测试、时延和汇总。
- [P40 CV3 驱动](../scripts/server/run_scaleroute_plan15_progressive40_cv3.sh)：三折适配、低阈值推理与 OOF 汇总。
- [Progressive resolution 训练实现](../scripts/train_progressive_resolution_adaptation.py)：分辨率适配训练入口。
- [三卡恢复实现](../scripts/resume_progressive_resolution_ddp.py)：full 训练第 3--40 epoch 的 DDP 恢复与状态审计。
- [P40+D4 部署配置](../configs/experiments/p40_aircraft_d4_full_runtime_candidate_v1.json)：当前安全候选的完整运行合同。
- [正式 Docker 推理入口](../src/rsdet/submission/competition.py)：配置、资产 SHA、切片推理、阈值与 D4 接线。
- [大图切片流水线](../src/rsdet/pipeline/large_image.py)：1024/256 切片与融合实现。
- [YOLO 推理适配器](../src/rsdet/models/ultralytics_adapter.py)：1280 输入、低候选阈值和模型输出适配。

### 3.4 Aircraft D4 模型设计与训练

- [R1-5 三折冻结配置](../configs/experiments/r1_aircraft_view_consistency_v1.yaml)：Aircraft 双视图一致性主实验。
- [R1-5 full 冻结配置](../configs/experiments/r1_aircraft_view_consistency_full_v1.yaml)：17,948 个飞机训练框、5 epoch、优化器和一致性损失。
- [Aircraft full 训练入口](../scripts/train_aircraft_view_consistency_full.py)：从固定 P03 checkpoint 训练 full 分类器，禁止验证选 checkpoint。
- [Aircraft 训练与评估实现](../scripts/r1_aircraft_refinement.py)：D4 视图采样、一致性损失和三折执行。
- [Aircraft 训练行审计](../src/rsdet/analysis/aircraft_refinement.py)：类别范围、训练清单和数据完整性。
- [ConvNeXt 分类器定义](../src/rsdet/models/crop_classifier.py)：Aircraft 细分类模型结构。
- [D4 部署运行时](../src/rsdet/submission/aircraft_d4.py)：八视图批处理、0.9 重标阈值和飞机类内 NMS。

### 3.5 评价指标、对比实验和消融实验

- [平台确认评分实现](../src/rsdet/evaluation/platform_protocol.py)：Recall、FDR、时延及七项等权总分。
- [官方匹配与细类宏平均](../src/rsdet/evaluation/official_metric.py)：类别一致匹配和粗类内细类宏平均。
- [分数与硬门测试](../tests/test_platform_protocol.py)：评分公式和门槛的回归测试。
- [P40 无泄漏三折与 20e/40e 消融](../reports/experiments/IMPROVEMENT_PLAN15_SCALEROUTE_EXECUTION_20260903.md)：基础 S1024、P20、P40 的 Normal/Hard/Sentinel 对照。
- [D4 三折消融](../reports/experiments/R1_AIRCRAFT_VIEW_CONSISTENCY_RESULT_20260814.md)：CE、identity、full-D4 的 TP/FP/FN 与宏指标。
- [P40+D4 精确运行时测试驱动](../scripts/server/run_attempt4_p40_aircraft_d4_only_3090_v1.sh)：Ship/Vehicle 旁路、D4 增量和 3090 时延复核。
- [D4 batch 工程消融](../scripts/server/run_attempt4_aircraft_d4_batch128_3090_v1.sh)：64 对 128 的逐框一致性与配对时延。
- [固定代理运行入口](../scripts/run_competition_runtime_coco.py)：让真实 competition runtime 直接跑带标注代理集。

写入报告的正式平台锚点应使用 v2.0 与 v3.0 两份平台报告；Normal、Hard、Sentinel 和 full-seen 结果必须按各自报告标注为 OOF、来源隔离迁移、压力测试或同源诊断，不能统称验证集成绩。

### 3.6 综合性能与创新点

建议从已被证据支持的两项创新组织正文：

1. **渐进式高分辨率适配 P40**：成熟 S1024 检测器以低学习率、无 mosaic、RandomRotate90 的 40 epoch S1280 适配，在来源隔离三折上相对 S1024 提升；最终 P40 v2.0 官方得分 76.6010，三项硬门通过。
2. **Aircraft D4 视图一致性细分类**：只修改 Aircraft 类别；训练期约束两个随机 D4 视图的一致性，推理期八视图集成。在 v3.0 官方隐藏集上相对 P40 净增 21 TP、减少 21 FP，Aircraft Recall `94.5967%→95.0685%`、FDR `3.7265%→3.2303%`。

不能把 v3.0 的 Vehicle hierarchy 写为创新收益；它使 Vehicle FDR 恶化并导致总分下降。P40+D4 的正式组合尚待下一次平台提交，提交前只可表述为“由官方分别确认的 P40 基线与 Aircraft 类别增量组成，Ship/Vehicle 保持 P40”。

### 3.7 工程实现与可复现性

- [Dockerfile](../submission/docker/Dockerfile.overlay)：复用已验证基础镜像的单模型覆盖构建。
- [Docker 主入口](../submission/docker/app/main.py)：官方 `--input /input --output /output` 合同。
- [P40 正式部署配置](../submission/docker/configs/progressive40_full_s1280_frozen0536_v1.json)：已提交 v2.0 配置。
- [P40+D4 实验部署配置](../configs/experiments/p40_aircraft_d4_full_runtime_candidate_v1.json)：待物化安全候选。
- [P40 权重净化工具](../scripts/sanitize_yolo_checkpoint.py)：去除训练期依赖且逐张量保持一致。
- [P40 逐框一致性工具](../scripts/validate_progressive_submission_parity.py)：原始推理与交付入口对照。
- [提交合同测试](../tests/test_submission_contract.py)：配置、类别、路径、阈值和输出边界。
- [D4 运行时测试](../tests/test_hier_d4_runtime_candidate.py)：Aircraft 所有权、重标和 NMS 回归测试。

## 4 关键结果账本

| 阶段 | 模型或模块 | 结果 | 证据等级 | 结论 |
|---|---|---|---|---|
| P40 CV3 | S1024→P40 | Hard `+6.212`，Sentinel-B `+2.214` 分 | 来源隔离方向证据 | P40 准入 full |
| P40 full | P40 v2.0 | 官方 `76.6010`，Recall/FDR/时延三门通过 | 官方隐藏集 | 当前正式最佳锚点 |
| D4 CV3 | consistency+D4 | 相对 CE+D4 净增 50 TP、32 FP；飞机宏 Recall `+0.0732pp`、FDR `-0.0197pp` | 三折开发证据 | 可训练 full |
| D4 official | v3.0 Aircraft 分量 | `+21 TP/-21 FP`，R `+0.4718pp`，FDR `-0.4962pp` | 官方隐藏集 | D4 模块成功 |
| v3.0 整体 | P40+Vehicle hierarchy+D4 | 官方 `75.9405`，比 v2.0 低 `0.6605` | 官方隐藏集 | Vehicle 分支与双检测器拒绝；D4 保留 |
| D4 batch | 64→128 | 输出完全一致；Hard 快约1.7%，Sentinel 慢约2.4% | 3090 配对工程审计 | 继续 batch64 |
| Shared OTM QHS/MS | P40+D4+OTM 2/3 | 短 OOF 合并约 `+0.214`，三折同向；full-seen机制约 `+0.873` | 后验方向证据 | 第四次攻击候选，仍需整链复核 |

## 5 当前候选边界

### 5.1 稳定线

`P40 + Aircraft-D4 only`。类别所有权为：Ship 0--3 由 P40；Aircraft 4--23 由 P40 候选加 D4 重判；FSC 24 由 P40。D4 batch 固定 64。该线是技术报告应描述的主线。

### 5.2 第四次攻击候选

`P40 + Aircraft-D4 + shared OTM(QHS/MS 2/3, threshold 0.560)`。它只替换 QHS/MS 两个 Ship 细类，Aircraft-D4 和 FSC 保持不变。配置与实现：

- [攻击候选配置](../configs/experiments/hera_sprint20_p40_d4_otm_ship23_t0560_candidate_v2.json)
- [共享检测头实现](../src/sprint20/heads.py)
- [共享运行时装配](../src/sprint20/runtime.py)
- [共享 OTM OOF 分析](../scripts/analyze_sprint20_oof_routing.py)
- [共享 OTM 整链 3090 驱动](../scripts/server/run_attempt4_shared_otm_runtime_3090_v1.sh)

该候选在完成 Hard/Sentinel 的类别旁路、质量方向和时延复核前不应写成最终方法；即使通过，也应在报告中与稳定主线分开说明。

## 6 技术报告附录建议清单

官方模板附录B要求列出训练代码、推理代码、评估代码、说明文档、权重和测试结果。可从本索引选择以下最小集合：

- 训练：`train_progressive_resolution_adaptation.py`、`resume_progressive_resolution_ddp.py`、`train_aircraft_view_consistency_full.py`；
- 推理：`competition.py`、`large_image.py`、`ultralytics_adapter.py`、`aircraft_d4.py`；
- 评估：`platform_protocol.py`、`official_metric.py`、`run_competition_runtime_coco.py`；
- 配置：P40 正式配置、P40+D4 安全候选配置、P40 CV3/full 驱动、R1-5 三折/full 配置；
- 结果：方案15报告、正式 v2.0 报告、R1-5结果、正式 v3.0报告、Attempt 4决策报告；
- 权重：在附件清单写资产名、大小、SHA 和获取位置，不放入代码报告 ZIP；
- 复现：使用本索引作为仓库阅读入口，不另建会改变公开仓库首页的 README。

## 7 写作时必须保留的限制

- 官方最终成绩只引用平台记录；本地代理分不能改名为官方验证成绩。
- P40 full 的 Normal/Hard/Sentinel 含训练同源成分，只作机制和工程诊断。
- P40 CV3 与 P40 full 的训练成熟度不同；三折方向不能直接当 full 隐藏分数。
- Aircraft-D4 的官方正增量来自 v3.0 中 Aircraft 分项；P40+D4-only 组合尚无平台独立成绩。
- v3.0 总分下降不是 D4 失败，而是 Vehicle FDR 和第二检测器时延覆盖了 D4 收益。
- 权重、数据和服务器路径不是 Git 资产；SHA、冻结配置和训练报告共同构成可追溯身份链。
