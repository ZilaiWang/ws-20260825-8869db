# 改进方案15：ScaleRoute 机制核验与执行记录

日期：2026-09-03

状态：`full_seen_diagnostic_and_submission_image_ready_user_push_pending`

## 1. 结论

《改进方案15》的核心判断合理：S1280 的主要作用不是“模型整体更强”，而是提高
Vehicle/小目标的空间采样质量；因此不能用 S1280 全面替代 S1024，而应先做训练×推理
分辨率正交实验，再决定是否按类别路由。该因果链已经由三折 2×2 交叉矩阵确认。

ScaleRoute R2 是已经通过三套代理的双模型候选，但本轮最终得到的最强单模型候选是
**Progressive-40**：成熟 S1024 fold checkpoint 经过 40 epoch、无 mosaic、低学习率的
S1280 适配。它相对 S1024 在 Normal、Hard、Sentinel 的三个粗类 Recall 全部提高，同时
三个条件的 Gate FDR 全部降低；因此不再只是“Vehicle 增益换取 Aircraft 损失”。

**15:28 full 验收更新：** 三卡续跑40/40及自动检查已完成，全部产物已经下载并验证31项
SHA，模型权重没有非有限数。单卡3090的6张伪大图平均4.530秒。但冻结Background-100MP
上误检从原S1024部署工作点的2个增加到10个，因此当前是“技术验收完成、背景风险待复核”，
而非“全部科学门禁通过”。不能把训练集上的97.34% Recall视为官方预期成绩。详见第8节。

作为结构对照，ScaleRoute R2：

- S1024 负责标签 0–23（Ship + Aircraft）；
- S1280 训练、1280 推理的模型只负责标签 24（Vehicle）；
- 两条分支分别完成大图流水线和安全融合后再按类别互斥合并；
- Normal、Hard、Sentinel 三个固定代理上，三类 Recall 都不会因路由下降，整体 Recall
  均提高；
- 代价是双模型顺序推理约 7.85–8.04 秒/图，仍低于 20 秒硬限制，但不是最终正式提交
  结论。

此前额外测试的 X10-SA 路由（S1024 只负责 Aircraft，S1280-train/1024-infer 负责
Ship + Vehicle）在 Normal 与 Sentinel 更强，但 Hard Recall 下降，因此只保留为机制性
探索结果，不替代 R2。它不是方案15原文中改变 source tile 几何的正式 R3；正式 R3
另行按 `主分支 tile1280→network1024 + Vehicle tile1024→network1280` 验收。

## 2. 训练×推理分辨率正交矩阵

四个条件全部使用相同三折、YOLO26-s、Y5、40 epoch、固定 last checkpoint、低阈值
候选集和 outer-fold 阈值选择。矩阵值为平台三粗类宏平均：

| 训练 / 推理 | Gate Recall | Gate FDR |
|---|---:|---:|
| 1024 / 1024（X00） | 43.440% | 17.119% |
| 1024 / 1280（X01） | 42.054% | 17.430% |
| 1280 / 1024（X10） | 51.693% | 21.619% |
| 1280 / 1280（X11） | 47.544% | 17.869% |

直接结论：

1. 仅把 S1024 权重放大到 1280 推理使 Recall `-1.386pp`，不是收益来源；
2. 保持 1024 推理、改为 1280 训练使 Recall `+8.253pp`，真实收益来自训练分辨率；
3. X10 的 Recall 最高，但 Ship FDR 明显偏高；
4. X11 是较稳的 Vehicle 专家，不能全面替代 X00。

`max_det=500` 有 22 个模型×折×图实例触顶，但正式工作阈值下每图最高保留数仅约 11；
触顶发生在 0.001 候选底部，当前不值得重跑 `max_det=1500`。

原始证据：

- `outputs/SCALEROUTE-PLAN15-CV3-V1/resolution_cross_matrix.json`；
- `outputs/SCALEROUTE-PLAN15-CV3-V1/audits/`；
- `scripts/analyze_resolution_cross_matrix.py`；
- `scripts/audit_max_det_saturation.py`。

## 3. ScaleRoute R2

### 3.1 Normal/source-grouped CV3

| 条件 | Gate Recall | Gate FDR | 相对 X00 |
|---|---:|---:|---|
| X00 | 43.440% | 17.119% | — |
| R2 | 48.995% | 17.607% | Recall `+5.556pp`，FDR `+0.488pp` |

Ship、Aircraft 完全沿用 X00，所以指标不变；Vehicle Recall `+16.667pp`、FDR
`+1.463pp`。该结构满足“增强弱类但不伤已有强类”的可解释性要求。

### 3.2 冻结阈值迁移

阈值只来自 Normal CV3，Hard/Sentinel 不参与阈值选择：

| 固定代理 | X00 R/FDR | R2 R/FDR | Recall 差值 | FDR 差值 |
|---|---:|---:|---:|---:|
| Hard | 41.627% / 23.018% | 43.076% / 20.692% | +1.449pp | -2.325pp |
| Sentinel-B | 43.755% / 14.482% | 48.694% / 15.092% | +4.938pp | +0.610pp |

以两分支实测顺序时延计入平台公式，Hard 约 `+0.518` 分，Sentinel 约 `+0.616` 分。
这说明 R2 是真实但幅度有限的稳健增量，而不是达到官方 90+ 的单独突破。

证据：

- `outputs/SCALEROUTE-PLAN15-CV3-V1/route/s1024_sa_s1280_vehicle.json`；
- `outputs/SCALEROUTE-PLAN15-R2-FIXED-BENCHMARKS-V1/`。

## 4. X10-SA 额外探索及其边界

X10-SA 使用 X10 的 FDR10 工作点接管 Ship + Vehicle，S1024 仅保留 Aircraft：

| 固定代理 | Recall 差值 | FDR 差值 |
|---|---:|---:|
| Normal | +4.390pp | -5.775pp |
| Hard | **-1.134pp** | -10.282pp |
| Sentinel-B | +1.276pp | -2.272pp |

X10-SA 显著降低 FDR，但 Hard 的 Ship/Vehicle Recall 同时下降，违反全方向准入要求。它证明
X10 可用于低虚警专家研究，却不能在当前证据下成为正式主线。

证据：

- `outputs/SCALEROUTE-PLAN15-R3-EXPLORATORY-V1/`；
- `outputs/SCALEROUTE-PLAN15-R3-FIXED-BENCHMARKS-V1/`。

## 5. 部署实现与验证

新增 `src/rsdet/submission/class_resolution_router.py`，将两条流水线的输出在后融合阶段
按互斥标签集合组合。`CompetitionDetector` 已支持以下独立配置：

- `resolution_expert_model`；
- `resolution_expert_pipeline`；
- `resolution_route`。

加载阶段会验证：25 类必须被两分支无重叠覆盖、权重 SHA、分支阈值范围、候选底阈值
不能高于路由阈值，并禁止与旧 agreement 双模型同时启用。路由保留原始像素坐标，按
分数、标签和坐标确定性排序。相关 26 项单元测试和 Ruff 当前全部通过。

这只是可复用实现，不等于已打包 Docker；本轮明确不进行镜像物化和官方提交。

## 6. Progressive resolution adaptation 最终结果

### 6.1 Progressive-20

三折分别从本折成熟 S1024 40e checkpoint 初始化，固定：

- 20 epoch、1280、batch8、seed42；
- AdamW，`lr0=2e-4`，`lrf=0.10`；
- RandomRotate90；
- mosaic 关闭；
- 固定 last，不使用验证集选 checkpoint；
- 每折只在本折 held-out 数据推理，最后做 outer CV3。

Progressive-20 已证明方向有效：

| 条件 | S1024 R/FDR | Progressive-20 R/FDR | Recall 差值 | FDR 差值 |
|---|---:|---:|---:|---:|
| Normal CV3 | 43.440% / 17.119% | 51.361% / 16.016% | +7.921pp | -1.103pp |
| Hard 冻结迁移 | 41.627% / 23.018% | 45.373% / 21.699% | +3.746pp | -1.319pp |
| Sentinel-B 冻结迁移 | 43.755% / 14.482% | 52.026% / 16.486% | +8.271pp | +2.004pp |

计入单模型实测时延后，Hard 平台代理分 `52.009→55.048`（`+3.039`），Sentinel-B
为 `58.200→58.949`（`+0.749`）。

### 6.2 Progressive-40 时长消融

Progressive-40 不是从 20e 权重 resume，而是从相同成熟 S1024 fold checkpoint 独立
训练 40 epoch；除适配时长外，其余数据、seed、batch、优化器、学习率、增强与 last
checkpoint 合同不变。三折均为 40/40，完整 SHA 验收通过。

| 条件 | Progressive-20 R/FDR | Progressive-40 R/FDR | Recall 差值 | FDR 差值 |
|---|---:|---:|---:|---:|
| Normal CV3 | 51.361% / 16.016% | **51.741% / 15.560%** | +0.380pp | -0.457pp |
| Hard 冻结迁移 | 45.373% / 21.699% | **48.520% / 16.218%** | +3.147pp | -5.481pp |
| Sentinel-B 冻结迁移 | 52.026% / 16.486% | **51.921% / 13.548%** | -0.105pp | -2.939pp |

计入实测时延，Progressive-40 相对 Progressive-20：

- Hard：`55.048→58.221`，`+3.173` 分；
- Sentinel-B：`58.949→60.414`，`+1.466` 分。

相对原 S1024，Progressive-40：

- Normal：Recall `+8.301pp`、FDR `-1.560pp`；
- Hard：Recall `+6.893pp`、FDR `-6.800pp`、平台代理分 `+6.212`；
- Sentinel-B：Recall `+8.166pp`、FDR `-0.935pp`、平台代理分 `+2.214`。

三个条件上的 Ship、Aircraft、Vehicle Recall 均高于原 S1024。Progressive-40 因而
替代 Progressive-20，成为方案15唯一准入的单模型 full 配方来源。

### 6.3 Progressive-40 full 正式链

2026-09-03 14:37 在单张 RTX 3090 上启动唯一 full 配方，服务器任务目录为
`/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-V1`。正式合同为：

- 初始化权重：S1024 全量 160e 成熟权重，SHA
  `f7e30fac3391d7048314d1c41df0e878a4d5f0423e5c55b68ee7df5917418229`；
- 正式数据：4481 图全量清单，SHA
  `30d7bac4bbdc5069becf5b54d9cc3cf89f348459584f3713799dc072572dcb19`；
- 适配：40 epoch、1280、batch8、AdamW、`lr0=2e-4`、`lrf=0.10`、无 mosaic、
  RandomRotate90、seed42、固定 last；
- 部署阈值：P40 CV3 在平台 FDR 0.15 工作点的预先冻结 pooled 阈值 `0.536`；
- 后处理：训练结束后自动运行 Background-100MP 和三张 Hard pseudo-10K 的 3090
  纯时延检查；全量模型在 pseudo-10K 上不作精度结论，避免训练集泄漏；
- 当前未打包 Docker、未进行正式提交。

启动后已扫描全部 4481 张训练图，模型为 YOLO26s 25 类，首个 epoch 损失有限。

2026-09-03 14:53，经用户要求加速，迁移到三张 RTX 3090，任务改为
`/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-3GPU-R2`。
原单卡主机已无法连接，三卡克隆中完整保存了第 2 epoch checkpoint；未修改或覆盖原目录。

- 续跑输入 SHA：`cbcc62b1dfa74fb63a8a4f44420572b33ec5c8b39b356e7ef5ca2222432c3964`；
- 真正 resume：从第 3 epoch 到第 40 epoch，不重新训练前两轮，也不是权重微调重启；
- 三卡 DDP 共同训练一个模型，每卡 batch8、总 batch24、8 workers/进程；
- 这是用户授权的硬件迁移修订：原总 batch8、梯度累积8（名义有效64），现总 batch24、
  梯度累积3（名义有效72）。未放大学习率；加载 checkpoint 后 weight decay 恢复为
  原值0.0005。不能描述为与原单卡或 CV3 逐步等价；
- 每张卡均核验优化器全部状态张量、scaler、EMA与其更新数恢复；起始 epoch、scheduler、
  无 mosaic、RandomRotate90、25类/4481图也通过断言；随机数按上游的 rank seed 重新
  初始化，并非逐 bit RNG 重放；
- 实际解析后的4481张图均唯一、可读，所用标签与冻结标签完全一致。按清单顺序的实际标签
  digest为 `c8e7555a3d6257c9925bb9fe6dc0289e5882f6aff1c02d72ff8450d0b34b03ac`；
- 三卡40MB NCCL all-reduce约6.2ms，恢复后第3轮50.48秒、第4轮48.16秒；原单卡
  120–125秒/epoch，稳定阶段约2.5倍训练吞吐。14:56已完成4/40轮、进入第5轮；
  GPU大多采样85%–100%（同步/保存时短暂回落），每卡约11.1GiB、295–306W；
- 训练后仍执行同一冻结阈值0.536的背景审计和单卡3090时延检查。全量模型在官方训练源
  构造的伪大图上只作时延检查，不生成新的无泄漏精度结论。

新增实现与验收入口：

- `scripts/resume_progressive_resolution_ddp.py`：预检、不可覆盖的新目录、保留epoch历史、续跑；
- `src/rsdet/innovation/progressive_resume.py`：每个rank的恢复状态断言与审计；
- `scripts/check_three_gpu_collective.py`：不改模型的短时通信预检；
- `scripts/server/run_scaleroute_plan15_progressive40_full.sh`：训练、背景审计、时延、汇总串行链；
- `scripts/finalize_scaleroute_progressive_full.py`：40轮连续性、有限loss、权重SHA及逐图时延汇总；
- `tests/test_progressive_ddp_resume.py`、`tests/test_finalize_scaleroute_progressive_full.py`。

### 6.4 Background-100MP

冻结的 382 图/100.139MP 无目标集已经完成：

- S1024 full @1024；
- S1280 full @1280；
- S1280 full @1024；
- R2 与 R3 在 OOF 全量重拟合阈值下的 FP/100MP 与三粗类组成。

manifest SHA 固定为
`ed3cbbe6952ea5a7792821a316bd3b0ed93888f74a50eda2630f630c9c9020e7`。
R2 在全量重拟合阈值下仅产生 `2 FP/100MP`，与 S1024 相同，且 Vehicle FP 为 0。
该压力集没有参与阈值选择。Progressive-40 的 full 背景审计必须等唯一 full 权重产生后
再执行，不能用三个 fold 权重替代正式部署权重。

### 6.5 Ship-only MacroRisk 与正式 R3 tile 几何

两项低成本验证均已结束：

1. Ship-only MacroRisk 令 Gate Recall `+1.196pp`，但 Gate FDR `+0.637pp`，平台分
   `56.453→56.402`，没有准入；
2. 正式 R3 在 Hard 为 `39.660%/23.939%`，比 R2 更差；Sentinel-B 为
   `47.026%/14.703%`，以 Recall 损失换取很小的 FDR 改善，亦不准入。

## 7. 冻结决策顺序

1. 唯一下一步训练候选冻结为 `S1024 full → Progressive-40 S1280 adaptation`；
2. full 配方保持学习率、无 mosaic、RandomRotate90 与固定 last；三卡迁移的 batch 修订
   已按6.3节单独记录，不隐去与单卡/CV3的差别；
3. full 完成后必须重新执行 Background-100MP、3090 时延、Docker 逐框一致性；
4. 不叠加 Ship-only MacroRisk、R3 tile 几何、purity、MacroExpert、DEIM 或 HAD；
5. 当前没有打包或提交，正式分仍未因本轮实验改变。

## 8. Progressive-40 full 最终验收（2026-09-03 15:28）

### 8.1 完成与可追溯性

- 训练于15:24结束，自动背景/时延检查于15:25:40结束；screen与训练/评测进程均退出，
  三张GPU均空闲；
- 第1–2轮来自原单卡checkpoint，第3–40轮为三卡DDP；CSV恰好40行、epoch连续1–40，
  loss均有限，三个rank的优化器/EMA/scaler恢复审计均通过；
- 三卡恢复阶段训练耗时1836.75秒（30.61分钟）；第4–39轮平均47.85秒，末轮包含框架
  自带训练视图验证，不能将其验证值作为独立精度；
- 最终固定last：`outputs/SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-3GPU-R2/adaptation/runs/resolution_adaptation/weights/last.pt`；
- 权重大小20,509,822 bytes、25类、非有限权重数0，SHA
  `904c4935a85484a83d98930b0862bd1b5a1b0e9e7c6ed4eea7525391d383123f`；
- 全量回收目录：`outputs/SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-3GPU-R2/`，总计约128MB，
  含原恢复输入、last/best、日志、预测和审计；远端与本地均通过全部31项结果SHA。

最终last已由Ultralytics剥离优化器，其checkpoint字段epoch=-1是发布权重的正常格式，
不能把该文件声称为可精确resume的中间checkpoint。第2轮原始恢复输入仍单独保留。

### 8.2 背景风险：没有通过“未退化”结论

冻结阈值0.536不变，382张裁片共100.139MP的预定义无目标集得到：

| 类别 | 原S1024 @0.646误检 | Progressive-40 full @0.536误检 |
|---|---:|---:|
| Ship | 2 | 7 |
| Aircraft | 0 | 1 |
| Vehicle | 0 | 2 |
| 合计 | 2 | 10 |
| FP/100MP | 1.997 | 9.986 |

这是两个模型**各自冻结部署工作点**的比较，阈值不同，因此补充仅作诊断的2×2固定点计数：

| 模型 | @0.536 | @0.646 |
|---|---:|---:|
| S1024 full | 5 | 2 |
| Progressive-40 full | 10 | 7 |

上述是对两个已经存在阈值的离线解释性复算，没有扫描阈值或改部署配置。候选在相同阈值
下也有更多背景框，不能把差异全归因于阈值降低；7个候选误检分数高于0.646，也不是全在
阈值边缘。具体哪些属于真实背景误识别、裁片边界效应或未标注目标，尚需图像复核，当前
不能擅自删掉或改标这些误检。也没有证据将此退化单独归因于DDP、full重拟合或分辨率。

FP/100MP是无目标压力集的误检密度，**不等于**官方FDR，不可把10个框解读为10% FDR，
也不可据此单独断言官方门槛通过或失败。它是提交前需要处理的风险提示。

### 8.3 单卡3090时延

同一full权重、1280推理、1024切片、256重叠、batch4、候选底阈值0.001，在6张固定伪大图
上逐图时延为5.185、4.430、4.344、4.283、4.582、4.353秒：

- 平均4.530秒；最大5.185秒；
- 是单卡模型流水线测量，不是三卡并行推理，也不是Docker端到端启动时延；
- 低于既有20秒限制，但仍需Docker接口与逐框一致性验证；
- full已见过这些伪大图的源训练数据，因此本次预测只作时延记录，不作新的泛化精度/得分。

### 8.4 当前决策

1. 全量训练无需重跑；权重和所有必要证据已安全落到本地。
2. 保留之前无泄漏P40 CV3的正向结论，但它不是本次迁移后full权重的直接精度验收。
3. 先复核新增背景误检，再决定是否执行Docker一致性和提交；目前未打包、未正式提交，
   也未开启新一轮训练、阈值调优或模块组合。

结果索引：

- `outputs/SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-3GPU-R2/validation_summary.json`；
- `outputs/SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-3GPU-R2/adaptation/training_result.json`；
- `outputs/SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-3GPU-R2/adaptation/resume_audits/`；
- `outputs/SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-3GPU-R2/background_100mp/frozen_threshold_result.json`；
- `outputs/SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-3GPU-R2/timing_only_hard/run_summary.json`；
- `outputs/SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-3GPU-R2/SHA256SUMS.txt`。

## 9. 用户要求的full同源诊断及提交准备（2026-09-03）

用户明确允许全量训练后回看旧测试，并随后要求准备一次提交。没有重训、调阈值或把同源
测试变为新的无泄漏准入依据；第8节的历史检查和背景风险不改写。

训练核查：YOLO26-s、25类一致、原始last SHA为904c4935…，源S1024权重和data YAML
SHA均匹配；40行epoch连续且数值有限，三rank优化器/EMA/scaler审计通过。用于本次
诊断的13个代码/配置SHA在执行时与本机一致，Normal旧fold0路径已仅在诊断配置重定位，
1507/1613/1361图的ID与原GT/frontier完全一致。

固定最终阈值0.536；Normal新推理4481图，Hard复用同一last现成预测，Sentinel新推理6图。
两套伪大图各600个源图均在full训练内；预测、GT、日志与33项SHA已下载本地并校验。

| 测试 | Ship R/FDR | Aircraft R/FDR | Vehicle R/FDR | Gate R/FDR | 本地分数 |
|---|---|---|---|---|---:|
| Normal | 89.101% / 4.214% | 99.015% / 1.417% | 85.572% / 6.267% | 91.229% / 3.966% | — |
| Hard | 86.500% / 2.888% | 99.034% / 2.893% | 80.978% / 3.871% | 88.838% / 3.217% | 84.965 |
| Sentinel | 86.148% / 9.139% | 98.963% / 2.678% | 88.889% / 2.703% | 91.333% / 4.840% | 85.278 |

这些是`platform_observed_20260831`口径的**含泄漏诊断**。Normal没有大图时延，所以不
编造其综合分数。Hard/Sentinel时延使用对应单卡流水线测量，不能外推为官方得分。
主要剩余短板是Ship/Vehicle召回；Hard的QHS为189TP/41FN，MS为584TP/111FN，
Vehicle为149TP/35FN。Sentinel的HM只有6GT，2FP就形成25%细类FDR，须注意小样本波动。

提交准备额外发现并处理两处工程差异：旧Docker单视图额外NMS、最终阈值阶段未显式
对齐。新配置共享离线适配器，底阈值0.001融合后再过滤0.536。原始last附带训练增强对象，
断网镜像无法加载，遂导出仅保留推理信息的权重b0df7981…，708个模型张量逐项完全一致。

最终交付入口在Linux3090上与Hard+Sentinel的3913个框逐框一致（坐标/分数差均为0），
12图平均4.439秒；本机linux/amd64镜像通过断网加载、177文件SHA与CPU前向。
GPU验收是交付代码直接运行，不冒充GPU容器实测。新镜像已准备，但尚未推送/正式提交。

完整身份与用户流程见：[P40提交冻结与操作](../submission/P40_SUBMISSION_FREEZE_20260903.md)。

## 10. 正式v2回传：76.6010分，三门全部通过（2026-09-03）

用户已完成推送与提交。只读结果API确认：ID3953、tag v2.0、SUCCESS，成绩时间
16:20:12，读取时第16名，三门分别为Recall86.2480%、FDR10.4918%、时延3.551833s，
平台三个Pass字段均为“是”。用户推送日志digest与第9节final镜像一致。

相对正式v1.0的72.1331分，净提升4.4680分：FDR收益+7.0894、Recall代价−2.4673、
时延代价−0.1541。舰船/飞机/车辆Recall为80.9894%/94.5967%/83.1579%，均低于v1；
FDR为6.7489%/3.7265%/21.0000%，均改善。不能将此次结果解释为三类召回同时提升。

全量同源Hard/Sentinel约85分仍不是隐藏集预估；本次正式低约8.4–8.7分，其中车辆FDR
比同源测试高17–18pp，是下一步负样本与工作点校准的核心缺口。评分公式精确复算一致，
没有因线上分数不高而修改评分器、测试集或部署阈值。

完整响应、TP/FP/FN、七项子分、同源与无泄漏测试的区分、下一步分析顺序见
[正式提交2结果分析](FORMAL_ATTEMPT2_P40_PLATFORM_RESULT_ANALYSIS_20260903.md)。
本次仅归档和分析，不启动新的训练或官方提交；第8、9节历史状态与风险保留。
