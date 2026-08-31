# HERA-Guard Final 前置实现、真实 smoke 与 3 GPU 执行报告（2026-08-31）

状态：`three_gpu_screen_running / dfine_direct_deployment_rejected`

## 1. 结论先行

《改进方案 11》中仍有信息增量、且能保持单 Y5-S 部署的三条路线已经全部从“想法”推进到
可执行代码：

1. vehicle OOF 漏标审计与 partial-label-safe 配对数据集；
2. DOTA/DIOR 四粗类外部预训练、fresh 25 类 head 的两阶段迁移；
3. D-FINE→Y5 的 tile-space in-model agreement distillation（HAD）。

当前单卡服务器已完成 D-FINE-M full 40 epoch；外部迁移与 HAD 均通过真实数据 CPU/GPU
smoke，固定 Normal/Hard/Sentinel 候选替换评测已实现。3×3090 主机已经完成 DOTA 全量准备、
HAD 四组训练并启动外部预训练和 HAD 冻结评测。尚未声称这些模块“提高成绩”：只有配对筛选
和冻结外层评测通过后，才能进入 HERA-Guard Final。

## 2. 完整 D-FINE 教师

正式 full 训练输入为 4,481 图、20,933 框，固定 D-FINE-M、40 epoch、epoch40 last：

|产物|值|
|---|---|
|状态|`complete`|
|训练日志行数|40|
|last checkpoint|314,410,486 bytes|
|last SHA256|`a6d21b7a6fb40d762d3ffee1b8fc22117619c4f37375fb7b499acead6dcfb49d`|
|epoch39 内部 COCO AP/AP50/AP75|0.8586 / 0.9863 / 0.9706|

内部 AP 使用同一 full 数据，仅证明训练健康，不能用于选择方法。其后冻结评测矩阵为：

- Y5-S identity；
- 已有 Y5-L（其历史 checkpoint 含 NumPy 2.2 RNG pickle，当前冻结环境无法安全加载；该路线
  已有负向证据且不参与 D-FINE 决策，因此停止而不改写 checkpoint）；
- Y5-S + D-FINE vehicle 0.059；
- Y5-S + D-FINE aircraft 0.05 + vehicle 0.059；
- Hard10K 与 source-disjoint Sentinel 均不重调阈值。

第一次固定评测因服务器同时存在新 `competition.py` 和旧 `PipelineConfig` 而退出。训练权重
未受影响。驱动已增加真实模块路径与构造签名门禁，并将工作目录移出仓库根；正式 V4 已
确认两个模块均来自同一 `/workspace/xh-202625/src`。失败 V1--V3 只作为环境事故，不参与结论。

V4 固定结果：

|条件/路线|vehicle Recall Δ|vehicle FDR Δ|aircraft FDR Δ|时延|
|---|---:|---:|---:|---:|
|Hard vehicle-only|-9.239pp|-4.695pp|0|9.38s|
|Sentinel vehicle-only|-8.025pp|-4.241pp|0|9.54s|
|Hard aircraft+vehicle|-9.239pp|-4.695pp|-6.524pp|9.80s|
|Sentinel aircraft+vehicle|-8.025pp|-4.241pp|-6.475pp|9.39s|

基线时延约 4.3s。两条路线都在以严重 Recall 换 FDR，且双模型时延翻倍；
`formal_submission_admission=false`。因此 full D-FINE 不进入 Docker，也不再做阈值扫描；它只
保留为 HAD 训练期教师。Hard vehicle-only 还使 ship macro Recall 下降 2.273pp，说明在最终
safe fusion/唯一匹配链中，即使只改 vehicle 分数也不能假定其他粗类严格不变。

## 3. vehicle 漏标审计与安全训练

三折 OOF Y5+D-FINE 共生成 67 个 vehicle 疑似漏标候选。代理逐张检查 full/zoom 卡片后冻结：

|决定|数量|训练语义|
|---|---:|---|
|confirmed missing|32|只允许以 class 24 加入配对 patch 实验|
|ambiguous ignore|20|不伪造标签；安全实现为排除含该区域的图像|
|rejected|15|继续作为普通背景负监督|

审计 SHA：

- review CSV `f18a053a...3fd25`；
- decisions CSV `570f699b...70ce3`；
- confirmed JSON `11966b4d...3fbeb`；
- ignored JSON `69fe26e4...751d`；
- rejected JSON `b7a4004f...e2d15`。

`scripts/materialize_partial_label_safe_dataset.py` 强制 patch/control 共用同一 ambiguous 排除集，
因此二者唯一差别是已确认框 add/omit。全量物化审计为 4,470 图；18 个已确认框实际加入；
其余 14 个 confirmed 与被排除图重合而不加入。该设计牺牲少量图像，换取在不修改
Ultralytics loss 的情况下保证“疑似真实目标不再被当作负样本”。

## 4. 外部数据资产与转换

### 4.1 DOTA-v1.0 真实数据验证

已从官方 Google Drive 获得 train part1 图像包和全量 train labelTxt，锁定原压缩包 SHA；
真实导入 469 图：

|字段|数值|
|---|---:|
|保留标注|26,777|
|aircraft|1,526|
|ship|15,759|
|vehicle|9,417|
|other_remote_object|75|
|丢弃 difficult|1,637|
|丢弃巨型场景结构标注|4,654|
|非法标注/框|0 / 0|

16 图类别覆盖 smoke 经 1024、overlap256、visibility0.7 切片后得到 184 tiles、3,532 标注，
标注无重复、无可见率损失，包含 22 个确定性空 tile。EXT-G/EXT-V 分别产生 232/298 个采样
行。24 张卡片、2 张 contact sheets 已逐图审核：紧凑目标 HBB 正常，无 harbor/bridge/
sports-field 巨型前景框，无 tile 边界异常，准入流水线和短训练 smoke。

完整 train+val 仍需在正式主机使用官方资产下载，建议 100GiB 可用盘；下载、解压、转换、
切片、YOLO 导出、role sampler 和视觉审核由
`scripts/server/run_hera_guard_final_prepare_dota.sh` 串联。

3×3090 主机实际复用了 AutoDL 只读官方缓存；五个压缩包逐一通过既有官方 SHA256。全量
train+val 为 1,869 张源图、102,530 个可用粗类标注。16 worker 确定性切片得到 9,153 个
1024 tile（6,461 positive + 2,692 deterministic empty），保留 102,518 个标注，仅 12 个因
可见率门禁丢弃，195 个截断框按合同保留。类别为 aircraft 10,264、ship 36,344、vehicle
49,319、other 6,591；EXT-G/EXT-V 分别为 13,276/13,831 个采样行。

并行切片保持源图排序后的 ID 汇总，因此 worker 数不影响 COCO、tile 文件名或像素。串行/并行
单测逐项比较 JSON 与像素，全量输出 SHA 写入 audit。代理已检查 96 张卡、8 张 contact sheet：
类别颜色与语义一致，HBB 对齐，无系统性错类、坐标漂移、巨型场景框或明显空框；DOTA
difficult 与主动未映射类别造成的可见未标目标符合冻结导入合同。视觉决定状态为 `pass`，
准备状态为 `ready_for_external_pretraining`。

### 4.2 DIOR 与其余来源

DIOR 的官方 Google Drive downloader、VOC→四粗类转换、坐标与 difficult 单元测试已完成。
首轮策略不是无归因混合：先比较 DOTA EXT-G/EXT-V，再把 DIOR 作为第二来源验证 ship/
结构化背景迁移。SODA-A、xView/AI-TOD、FAIR1M 和 RarePlanes 的访问/许可/科学优先级已写入
`configs/external/source_admission_matrix.json`。当前硬盘不足以安全保留多套完整大数据，
因此未制造不完整资产；正式服务器按磁盘门禁下载。

## 5. 外部预训练→官方细类迁移

实现：

- `scripts/train_external_y5_coarse.py`；
- `src/rsdet/external/transfer.py`；
- `scripts/train_external_initialized_y5_fine.py`。

早期 smoke 暴露并修复一个关键错误：对 Detect 递归通用 `reset_parameters` 会破坏原生
`bias_init`，分类损失曾异常到约 `7.39e4`。当前实现从 fresh 原生 DetectionModel 移植完整
Detect head，并在 optimizer/EMA 创建前完成，分类损失恢复约 5.5。

最终 staged-v2 真实数据 smoke：32 张、25 类覆盖、320 输入、1 epoch frozen + 1 epoch full，
正常完成；checkpoint SHA `006e71de...`，迁移审计 SHA `71482197...`。正式合同为 8 epoch
冻结前10层 + 32 epoch全模型，两个阶段重建 optimizer。

一次加载历史 checkpoint 时发现 NumPy 2.2 保存的 RNG pickle 不能被 NumPy 1.26 环境读取。
这不是模型结果，但证明正式链必须在一个冻结 venv 中从外部预训练连续运行到官方微调。

### 5.1 并行缓存隔离

EXT-G/EXT-V 同时训练会让 Ultralytics 尝试写同一 `labels/train.cache`。新增
`scripts/materialize_external_role_view.py`：图像字节只读共享，标签硬链接/必要时复制，
每个 role 拥有独立 cache。脚本可重复运行并逐文件校验，单测已覆盖 resume 与路径改写。

## 6. In-model HAD

冻结 teacher cache 包含 65,301 个 Y5 proposal，其中 5,131 个 vehicle、2,961 个有 D-FINE
支持。Y5-S Detect 输入层已审计为 P3/P4/P5：layer 16/19/22，stride 8/16/32，channel
128/256/512。

实现结构：

- 在原始 tile proposal 上做 P3/P4/P5 core/context ROIAlign；
- 只修改 class24 vehicle 分数，ship/aircraft 逐字节旁路；
- branch-only 只训低维投影与 bounded residual；
- terminal-FPN 只额外解冻 Detect 的三个输入层，并用冻结基线特征 anchor；
- BCE 连续异构支持目标 + OOF 官方匹配 BCE + protected-TP/active-FP 同图风险排序 +
  feature anchor；
- residual 零初始化，初始最大分数差门禁 `<=1e-7`；
- D-FINE 只在训练期生成 teacher cache，部署不运行 D-FINE。

复核 v1 后发现“只模仿 D-FINE support”会同时学习其支持的背景框，因此在 4 GPU 开训前升级
为 v2：每图采样优先保留 cache 中的 OOF `protected_tp/active_fp`，在 support 蒸馏之外加入
类平衡官方匹配 BCE 与 TP>FP rank。16 图真实 smoke 同时激活 support、match、match-rank
三类非零损失，branch-only 与 terminal-FPN 均完成。

部署位置在 `TileAgreementRuntime`：大图 safe fusion 之前、原始 tile FPN 上重排。v2 真实
vehicle batch smoke 共 497 框、308 个 vehicle，308 个 vehicle 全部经过分支；所有非 vehicle
分数精确不变、无 NaN/Inf，1 epoch 最大变化 0.000227，符合“训练不足但链路有效”的预期。

正式 3×3090 执行已完成 fold0/fold2 × branch-only/terminal-FPN 四组 8 epoch 训练。四组均为
`complete_diagnostic`，初始最大分数差均不超过 `9.313e-10`，满足零残差等价门；共输出 8 个
带 SHA256 的 detector/adapter checkpoint。GPU2 随后只对 fold0 两个模式执行冻结
Normal/Hard/Sentinel 候选替换评测；未扫描权重或阈值。

首次 AMP 候选推理暴露了 ROIAlign 的 dtype 约束：hook 捕获 FP16 FPN，而 proposal/adapter
为 FP32。修复为逐层把 ROI 坐标转为 feature dtype、池化后恢复 projection dtype，并增加
float64 feature + float32 geometry 的回归测试；已生成的冻结 baseline 被完整复用。

fold0 冻结结果：terminal-FPN 在 Normal ship Recall 下降 0.815pp、Hard ship 下降 0.629pp，
正式拒绝并停止。branch-only 的 Normal Recall/macro 完全不变；Hard vehicle Recall +1.087pp，
但 vehicle FDR +2.171pp；总体 Hard Recall +0.093pp、FDR +0.073pp；Sentinel 持平。它通过最低
扩展门但效应很小，因此只补 fold1 branch-only，并运行三折完整 adapter 外层复验；不调训练
或融合参数。三折若不能保持正向，HAD 路线整体停止。

三折完整复验现已完成，branch-only 也正式拒绝：Normal 总 Recall -0.104pp，ship
-0.543pp、vehicle -1.220pp；Hard vehicle 虽 +2.174pp，但不能补偿 Normal 粗类地板破坏；
Sentinel vehicle +1.235pp、总体 +0.152pp。固定门禁
`all_coarse_recall_floor_minus_0p5pp=false`，因此 HAD 全路线停止，不补 epoch、不扫描参数。

terminal-FPN 训练同时输出 `adapted_detector.pt` 与 `adapter_last.pt`；正式评测必须成对使用。
branch-only 使用原 base detector + adapter。任何只加载 adapter 而漏掉 adapted detector 的
terminal-FPN 结果无效。

## 7. 固定的两套评测与停止门

后续所有新方法只使用：

1. Normal-CV3：候选地板、pooled、25类 macro、各粗类；
2. Hard10K + source-disjoint Sentinel：FDR=0.15 外层工作点。

`scripts/replace_prediction_fold.py` 只替换已训练候选 fold，另外两折完全复用基线，避免每次
重复三折推理。`scripts/decide_hera_guard_final_candidate.py` 固定门禁：

- Normal Recall 与 macro 下降均 <=0.3pp；
- 任一粗类 Recall 下降 <=0.5pp；
- FDR 恶化 <=1pp；
- Hard ship 或 vehicle >=+0.5pp；
- Sentinel 同方向。

失败即停止该路线，不补 epoch、不扫融合权重。通过后才扩其余 folds。

## 8. 3/4×3090 执行矩阵

驱动：`scripts/server/run_hera_guard_final_4gpu_screen.sh`；完整操作见
`docs/server/HERA_GUARD_FINAL_4GPU_EXECUTION.md`。

Stage1 并行 DOTA EXT-G、DOTA EXT-V、HAD fold0、HAD fold2；Stage2 并行两种 external init
到 reviewed fold0，以及 official-init patch/control。每个 run 已有 checkpoint 但缺少完成审计
时固定退出并保留现场，绝不自动 resume。

2026-08-31 实际开放主机为 3×RTX 3090。动态调度只改资源排队：GPU0/1 保持
EXT-G/EXT-V，GPU2 串行 HAD fold0/fold2；Stage2 中 GPU2 串行 patch/control 对照。
数据、seed、epoch、batch、模型和准入门均未改变。`run_hera_guard_final_had_early.sh`
允许在 DOTA 资产准备期提前使用 GPU2，后续主驱动按 `training_result.json` 幂等跳过。

在 HAD 拒绝后，外部路线收敛为与当前官方短板最一致的 `EXT-V`：DOTA 粗类预训练中提高
vehicle/other 采样，部署仍是单视图 YOLO26s。两路早期 run 各仅完成 4 epoch 后停止并保留为
未完成诊断，没有据此形成结论。3 卡 DDP 真实 smoke 已通过；fine 阶段的动态 Trainer 无法由
Ultralytics DDP 子进程导入，因此不冒险改训练框架，而是在 coarse 完成后三卡并行执行三个
严格配对分支：`EXT-V→patch`、`official→patch`、`official→omit`。三者分别回答外部预训练、
人工复核标注修补、原始 omission 对照的独立贡献，不是三个部署模型。

资源剖析显示总 batch=12 时每卡仅 4 图，三卡平均利用率约 50--60%，而 112 CPU 核、755 GiB
内存和 `/dev/shm` 数据均无压力。正式 coarse 因而改用总 batch=30（每卡 10 图）：Ultralytics
梯度累积从 5 步变 2 步，`12×5=30×2=60`，有效优化 batch、学习率、权重衰减、epoch、seed
和样本轮数保持一致。旧 run 只完成 1 epoch 后归档，从初始 SHA 权重重新训练，未 resume。
新 run 显存约 10--15 GiB/卡、利用率约 66--79%，首轮预计由 180 秒降至约 120 秒。

实际启动时还发现两个 Ultralytics 进程会并发下载同一个 AMP 自检权重 `yolo26n.pt`，其中一份
出现临时损坏。此时尚未进入 epoch；现场被停止，使用服务器已有且可成功加载的官方自检权重
（SHA256 `9b09cc8b...4fef`）预置后从零重启。两路均显示 `AMP checks passed`，没有复用任何
不完整 checkpoint。该修复只消除启动竞态，不改变训练合同。

第一轮结束只允许选择：

- EXT-G/EXT-V 中 0 或 1 个；
- HAD branch-only/terminal-FPN 中 0 或 1 个；
- annotation patch 只有在配对 control 上通过才保留。

之后才组成唯一候选扩 CV3，最后训练一个 full 权重。最终 Docker 仍是单视图、单 Y5-S；
D-FINE 与外部 coarse head 不进入部署。

## 9. 验证与当前剩余项

- 全仓测试：871 passed，5 skipped；
- 新增外部/HAD/partial-label/候选替换测试全部通过；
- 本轮全部新增/修改服务器 shell 驱动通过 `bash -n`；
- 全仓 ruff 仍有 131 个早期遗留脚本问题，未把无关机械修复混进本实验；本次新增/修改文件
  需单独 ruff 全绿后提交。

剩余项只有：

1. 完成正在运行的 EXT-G/EXT-V 80 epoch、两路 fine transfer 与 patch/control；
2. 完成正在运行的 HAD branch-only 三折完整复验；
3. 只扩通过门禁的候选，生成唯一 full 权重、3090 时延与 Docker。

这意味着“能在单卡、有限本地磁盘上诚实完成的准备和验证”已经完成；下一步需要的是训练
证据，不再是继续增加离线模块。
