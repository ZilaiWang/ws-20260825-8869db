# HERA-Guard V4 / Q-Route 实验执行总账（2026-08-30）

## 1. 目标与冻结基线

本轮目标是将 `trial-v2.0` 作为唯一官方 incumbent，在两个冻结代理评测上提高舰船、车辆的官方匹配质量，同时保护已接近饱和的飞机指标。官方 incumbent 为：

- 综合分：86.2274；
- ship Recall/FDR：0.942287 / 0.126937；
- aircraft Recall/FDR：0.999246 / 0.024300；
- vehicle Recall/FDR：0.946309 / 0.237838；
- 平均推理时间：2.704833 s。

冻结的准入门为：Normal-CV3 pooled Recall 降幅不超过 0.3pp、任一粗类 Recall 降幅不超过 0.5pp、Hard10K@FDR15 Recall 至少提高 0.5pp、source-disjoint sentinel 同方向、飞机不退化。任何 pooled oracle 只作诊断，不形成准入结论。

## 2. 代码合同与修复

方案10交付的是实现骨架，审计后完成以下正式化修复：

1. `metric_aligned` 分别计算 `best_same_fine`、`best_same_coarse`、`best_any`，避免更近的错类 GT 隐藏有效同细类支持；
2. `crop_top1_probability` 使用显式字段，并兼容历史概率别名 `crop_top1`；两个别名冲突或概率越界时 fail closed；
3. safe tile fusion 禁止 `merge_iou=0` 或 `merge_ios=0`，避免 `OR` 条件把所有跨切片同类候选合并；
4. OMQ 标签严格按“粗类工作点阈值 → 同类 NMS → 官方 prediction-first 细类匹配”生成；修复 trace 全局候选 ID 被误当图内下标的错误；
5. D0/D1/D2/D3 分离为 ERM、仅 group-balanced sampler、仅 GroupDRO、两者组合，禁止默认同时启用；
6. active-pair Rank 默认只允许同粗类、同 source/airport group，不能因 batch 稀疏静默跨组；
7. OMQ 最终 residual 增加直接质量监督：active 区间学习 canonical TP/FP，inactive 区间学习 intrinsic same-fine matchability。

GPU 端专项门禁：28 项初始测试通过；补充最终质量梯度测试后 6/6 通过。相关实现：

- `src/rsdet/innovation/official_quality.py`
- `src/rsdet/innovation/group_dro.py`
- `src/rsdet/innovation/yolo_feature_quality.py`
- `src/rsdet/submission/selective_tta.py`
- `scripts/build_omq_cache_from_y5.py`
- `scripts/train_official_quality_head.py`
- `scripts/export_omq_oof_predictions.py`
- `scripts/server/run_hera_guard_v4_omq_q0_q2.sh`

完成 D-FINE/DEIM 配对快筛实现后再次执行全仓门禁：`pytest 817 passed, 5 skipped`；
本轮全部 modified/untracked Python 文件 `ruff` 通过，两个服务器 runner 均通过
`bash -n`。跳过项为仓库既有可选环境测试，不属于本轮失败。

## 3. Phase 0 因果账本

### 3.1 Identity 与 identity+rot90 双视图

正式 CV3 两折定阈值、一折评估的 per-coarse FDR15 结果：

| 缓存输出 | pooled Recall | pooled FDR | ship R/FDR | aircraft R/FDR | vehicle R/FDR |
|---|---:|---:|---:|---:|---:|
| identity | 0.948563 | 0.110773 | 0.903564 / 0.145689 | 0.997059 / 0.059204 | 0.913043 / 0.207547 |
| identity+rot90 双视图 | 0.957368 | 0.096633 | 0.918239 / 0.145366 | 0.997059 / 0.040566 | 0.940217 / 0.143564 |

血缘复核发现，历史目录名中的 `ROT90CW` 指的是配置 `rot90_views=[0,1]` 的双视图
融合输出，而不是纯 90° 单视图。两个输入 SHA 分别为
`70879b13…` 与 `c12c9e2a…`，逐一匹配对应正式推理目录。此前把第二行简称为
`rot90` 的表述已在本报告纠正；数值本身不变。

结论：双视图候选确有补充价值；它在冻结 Hard10K 开发账本上同时提高 pooled Recall
0.880pp、降低 pooled FDR 1.414pp。官方 v3 的 vehicle Recall 损失来自“双视图融合 +
更高车辆阈值”的交互，而不是第二视图完全无效。因此后续研究按粗类保留双视图输出，
飞机旁路第二视图，不把该表误写成纯单视图替换。

### 3.2 clean coarse verifier 历史 readout

独立 coarse verifier direct readout 在 FDR15 只有 pooled Recall 0.5463；pixel-OER identity/dual 在 FDR15 分别为 0.8624/0.8577，明显弱于 Y5 incumbent。它只证明 proposal crop 含前景信息，不具备独立主线能力。P0-2 记为完成的负向消融，不阻塞 OMQ。

### 3.3 score-sqrt

九个 score-sqrt checkpoint 已完整训练，但此前 GPU readout 因运行预算中止，不能记为科学负结果。P0-3 保持 `training_complete / evaluation_pending`，只补科学账，不阻塞 OMQ。

原始产物索引：`outputs/HERA-GUARD-V4-PHASE0-20260830/`。

## 4. OMQ 数据合同

Q0 缓存完成：

- 65,301 个候选，fold0/1/2 为 24,742 / 21,152 / 19,407；
- 255 个 source/airport group；
- v2 工作点 ship/aircraft/vehicle = 0.150 / 0.301 / 0.366；
- NMS 前 active 21,067，NMS 后 20,721；
- protected TP 18,763，active FP 1,958；
- 固定工作点 pooled Recall/FDR = 0.896336 / 0.094494；
- formal crop SHA256 = `a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128`。

Q0 为 65 维 deployable metadata；Q1/Q2 使用同一 65 维 metadata 加 3,200 维 Y5 FPN core/ring/region/scene evidence。GT 只生成标签，不进入特征。

## 5. 实验状态与结果

| 阶段 | 特征 | Rank | Sampler | Robustness | 状态 | 结论 |
|---|---|---|---|---|---|---|
| Q0-initial | metadata | off | uniform | ERM | complete | detector/quality 前沿逐点相同；定位为 residual 梯度断路，结果作实现诊断，不作方法结论 |
| Q0 | metadata | off | uniform | ERM | complete | FDR15 Recall 0.916830→0.922801（+0.597pp），真实正向 |
| Q1 | metadata+FPN | off | uniform | ERM | complete | FDR15 Recall 0.912387，低于 detector 与 Q0，否决 |
| Q2 | metadata+FPN | on | uniform | ERM | complete | FDR15 Recall 0.916066 / FDR 0.157247，否决 |
| D1 | metadata+FPN | on | group-balanced | ERM | stopped | Q2 未通过前置门，不运行 |
| D2 | metadata+FPN | on | uniform | GroupDRO | stopped | Q2 未通过前置门，不运行 |
| D3 | metadata+FPN | on | group-balanced | GroupDRO | stopped | D1/D2 均未准入，不运行 |

Q0 在 FDR10/12/15 的 Recall 分别相对 detector 提高约 0.994/0.865/0.597pp；
FDR15 的 ship、aircraft、vehicle Recall 均未下降。说明低维、可部署元数据确实包含
官方 TP/active-FP 排序信号。直接拼接 3200 维 FPN 后，Q1 在 FDR15 比 detector
低 0.444pp；再加 active-pair Rank 的 Q2 仍未恢复，且实际 FDR 超出 0.15。
因此本轮不继续用 D1/D2 扫鲁棒损失，保留 Q0，停止高维 FPN 路线。

### 5.1 按粗类路由双视图：Hard10K 冻结读数

有限策略集合只比较 identity、全双视图、三种粗类替换和一个同细类支持并集；没有扫描
融合权重。FDR15 关键结果如下：

| 策略 | pooled R/FDR | ship R/FDR | aircraft R/FDR | vehicle R/FDR |
|---|---:|---:|---:|---:|
| identity | 0.944393 / 0.110044 | 0.894130 / 0.145291 | 0.997059 / 0.058333 | 0.913043 / 0.207547 |
| aircraft=identity；ship/vehicle=双视图 | **0.952734 / 0.102967** | **0.907757 / 0.143422** | **0.997059 / 0.058333** | **0.940217 / 0.139303** |
| 全双视图 | 0.952734 / 0.094672 | 0.907757 / 0.143422 | 0.997059 / 0.039660 | 0.940217 / 0.139303 |
| identity + 有 identity 支持的双视图船/车 | 0.949954 / 0.102845 | 0.901468 / 0.143426 | 0.997059 / 0.058333 | 0.940217 / 0.139303 |

冻结类别路由相对 identity：pooled Recall +0.834pp、FDR -0.708pp；ship Recall
+1.363pp，vehicle Recall +2.717pp，飞机完全保持。它通过 Hard10K 的 +0.5pp 门，
并且只要求飞机不运行第二视图，理论计算量小于全双视图。Normal-CV3 与 source-disjoint
sentinel 正在用同一组三折 Y5 权重重新生成 identity/dual 证据；正式准入前不会用语义
不一致的 `Y5-ROT90` 训练权重代替推理视图。

一次错误的 Normal 复核任务在发现上述输入语义不一致后立即停止，状态明确记录为
`invalid_input_semantics_rot_file_was_different_model_not_dual_view`，不进入科学结论。

### 5.2 类别路由的 Normal 与来源隔离复验

使用相同的三折权重重新生成 Normal identity/dual，并把 Hard 开发前沿阈值冻结后应用到
source-disjoint sentinel。结果表明，直接把船/车整个替换成双视图输出不能准入：

| 数据集 | 策略 | pooled R/FDR | ship R | aircraft R | vehicle R |
|---|---|---:|---:|---:|---:|
| Normal | identity | 0.898916 / 0.152731 | 0.724087 | 0.938260 | 0.318408 |
| Normal | 船/车双视图路由 | 0.900588 / 0.152567 | 0.731171 | 0.938260 | 0.358209 |
| Sentinel | identity，Hard 阈值冻结 | 0.809040 / 0.190137 | 0.744909 | 0.913043 | 0.617284 |
| Sentinel | 船/车双视图路由，Hard 阈值冻结 | 0.820213 / 0.205217 | 0.772776 | 0.913043 | 0.592593 |

Normal 同方向，但 sentinel 的 vehicle Recall 下降 2.469pp，且 pooled FDR 恶化
1.508pp。因此“按粗类整路替换”记为完整负向结论，不进入 Docker。

### 5.3 第二视图新增候选的残差账本

为了判断第二视图的收益能否通过安全并集获得，对 identity 不动、只分析船/车第二视图
新增框：

- identity 同细类 IoU `<0.25` 的完全 novel 候选共 1,548 个，残差 TP 为 0，全部为 FP；
- `0.25≤IoU<0.50` 的弱支持候选共 1,077 个，仅 12 个残差 TP、1,065 个 FP；
- 原始分数和严格 nested logistic 在边际 FDR≤0.20 时均不能安全加入任何候选。

这说明第二视图的有效价值主要是对 identity 候选重新定位、去重或重排，而不是扩张候选集。
因此 novel budget 路线停止，转入 identity-only dual-consistency reranking。

### 5.4 Dual-consistency identity reranker

模型不增加、不删除 identity 候选，只用原分数、box 几何以及同细类双视图支持重排
ship/vehicle；aircraft 保持原分数。严格 nested Hard10K 结果：

| 目标风险 | Recall | FDR | ship R/FDR | aircraft R/FDR | vehicle R/FDR |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.930491 | 0.078476 | 0.880503 / 0.098710 | 0.997059 / 0.059200 | 0.820652 / 0.090360 |
| 0.12 | 0.940222 | 0.089726 | 0.895178 / 0.117770 | 0.997059 / 0.059200 | 0.858696 / 0.122220 |
| **0.15** | **0.951807** | **0.110052** | **0.910901 / 0.153021** | **0.997059 / 0.059200** | **0.913043 / 0.164179** |
| 0.20 | 0.957368 | 0.166599 | 0.919287 / 0.188714 | 0.998039 / 0.139476 | 0.929348 / 0.204651 |

相对同协议 raw identity 的 FDR15 `0.944393 / 0.110044`，nested 版本 Recall
提高 0.741pp，ship 提高 1.677pp，vehicle Recall 不降且 FDR 从 0.20755 降至
0.16418。该结果通过 Hard 门，但仍需最终全开发拟合模型的冻结外推。

### 5.5 全开发拟合后的冻结外推

正式部署形态只在 Hard 开发三折生成 OOF 分数并选择阈值，随后在全部 Hard 行拟合最终
模型；Normal 与 Sentinel 标签只用于一次性评估。为隔离阈值域偏移，每个目标集同时运行
同一 Hard 开发协议生成的 raw-score 基线。

12 维线性质量模型结果：

| 数据集 | Hard 冻结 raw R/FDR | reranker R/FDR | ΔRecall | ship ΔR | aircraft ΔR | vehicle ΔR |
|---|---:|---:|---:|---:|---:|---:|
| Hard OOF | 0.949027 / 0.107236 | 0.952734 / 0.109185 | +0.371pp | +0.419pp | 0 | +2.174pp |
| Normal | 0.891941 / 0.112046 | 0.894377 / 0.113835 | +0.244pp | +1.715pp | 0 | +1.244pp |
| Sentinel | 0.813103 / 0.198699 | 0.825292 / 0.195545 | +1.219pp | +2.144pp | 0 | +2.469pp |

因此它不是目标集拟合造成的伪增益：三套数据所有粗类 Recall 均不下降，Sentinel pooled
FDR 还下降 0.315pp。但正式全开发形态的 Hard 增益为 0.371pp，低于预注册 +0.5pp，
暂记为 `robust_positive_below_full_admission_margin`，不单独替换 v2 Docker。

增加 10 个预注册非线性交互的 22 维版本，在 Normal/Sentinel 上保持同方向并降低部分
FDR，但 Hard Recall 为 0.952271，未超过 12 维版本。随后按方案10的 bounded residual
只比较 `alpha=0.25/0.50/0.75`：

| alpha | Hard ΔR | Normal ΔR/ΔFDR | Sentinel ΔR/ΔFDR |
|---:|---:|---:|---:|
| 0.25 | +0.093pp | +0.091 / -0.023pp | +0.305 / -0.421pp |
| 0.50 | +0.232pp | +0.220 / +0.148pp | +0.965 / -0.147pp |
| 0.75 | +0.232pp | +0.220 / +0.098pp | +1.067 / -0.286pp |

残差强度与 Sentinel 增益单调，但 Hard 增益均低于完全质量重排。停止继续扫描 alpha；
保留 12 维完全重排作为后续新 detector 的部署候选，并优先寻找更强候选形成能力。

本节小型模型、审计 JSON 与预测血缘位于：
`outputs/HERA-GUARD-V4-DUAL-CONSISTENCY-20260830/`。

## 6. Phase 2 检测器升级

### 6.1 已有异构/容量资产复核

- Y5-L 在同一 Pseudo10K 代理 FDR15 的 Recall 为 0.955514，低于 Y5-S 的
  0.961075；vehicle 也从 0.940217 降至 0.913043，停止扩展 Y5-L；
- M3 RT-DETR-L 在同代理 FDR15 仅为 `0.784986 / 0.150877`，vehicle Recall
  0.326087；虽然 full checkpoint 已完成，但不具备独立或无条件并集准入价值；
- 上述结论使第二阶段只保留 D-FINE/DEIM 与外部 coarse/objectness 预训练，不再重复
  同族放大。

### 6.2 D-FINE-M fold0 快筛

已冻结 `D-FINE` 官方仓库 commit `956d1709314c2c6a4df6f34de232054578a7449f`，
使用 COCO 预训练 D-FINE-M，避免 Objects365 权重许可不确定性。fold0 数据合同：

- train/val 图像：2,974 / 1,507；
- train/val 框：13,583 / 7,350；
- split-view SHA：`a647ce03…`；
- COCO train/val SHA：`41e93416…` / `2641d3bb…`；
- 跨 split 图像：0；
- 1024 输入、40 epoch、固定最后 epoch、score floor 0.001。

状态：`running_r2`。首次启动完成 epoch 0–1 后因 8 个 DataLoader worker 的
Linux 文件描述符传递异常退出；训练数据、权重和优化参数均未改变，只把 train/val
worker 从 8 降至 2，并增加失败状态 trap。为防止旧 `last.pth` 和 `log.txt` 污染
血缘，正式重跑使用全新 `DFINE-M-FOLD0-40EP-V1-R2` 目录从零开始。该事件属于运行
恢复，不构成科学结果。单折结果只作快筛，不用于选择正式提交阈值；准入后才扩三折并执行
Normal/Hard/Sentinel 三段审计。

### 6.3 DEIM-D-FINE-M fold0 配对快筛

方案10要求验证的是带 Dense O2O 与 Matchability-Aware Loss 的 DEIM-D-FINE，不能
用原版 D-FINE 代替。为得到可归因结果，服务器2并行运行官方 DEIM 仓库 commit
`09d35d53d39ee3145a1e61e3a989b28b9468d1dd`，与 6.2 共用完全相同的 fold0、
COCO 标注、1024 输入和 40 epoch 快筛合同。主要差异仅为官方 DEIM 训练机制：

- COCO 预训练 DEIM-D-FINE-M SHA256：
  `2b6cd0582a4aa711f583982057b7fb0f3daebdd98e4dc168824714014c3219bc`；
- Dense O2O 的 mosaic/mixup 调度与 MAL；
- 单卡 batch 4，按官方 batch32 配置线性缩放主学习率到 `5e-5`、backbone 到
  `5e-6`；前 500 iteration warmup，24 epoch 后 cosine，最后 8 epoch no-aug；
- 25 类头由正式训练集重新学习，aircraft/ship/vehicle 仍按官方细类和 IoU 规则评估。

状态：`running_r2`。首批 iteration 有限、无 NaN/OOM，峰值训练显存约 9.1 GiB。
预检发现官方 runner 默认在 `stop_epoch=32` 时加载 held-out 验证集上的
`best_stg1.pth`，与本快筛“固定第 40 轮、不按 outer fold 选模”的合同冲突。早期运行
因此停止，不进入结果；R2 保留增强 policy 在 epoch 32 关闭，但把 collate/EMA stage
的 `stop_epoch` 设为 40，使 epoch 0–39 持续保存固定 `last.pth`，并使用全新结果目录。
D-FINE 与 DEIM 先分别输出 held-out fold0 的 0.001 score-floor 候选，再用同一个
official-matching 单折诊断前沿比较。只有 DEIM 或 D-FINE 在 ship/vehicle 候选能力和
FDR15 上明显优于 Y5 fold0，才扩正式三折；二者同时负向则停止 DETR 异构路线。

### 6.4 fold0 冻结 Y5-S 比较点

在新 detector 结果产生前，先把正式 Y5-S fold0 的 1,507 张低阈值预测下载到本地，
用与 D-FINE/DEIM 完全相同的 COCO GT、prediction-first matcher 和 0.005 网格计算：

| 工作点 | pooled R/FDR | ship R/FDR | aircraft R/FDR | vehicle R/FDR |
|---|---:|---:|---:|---:|
| score floor 0.001 | 0.969388 / 0.712028 | 0.955801 / 0.886557 | 0.973384 / 0.582495 | 0.872180 / 0.951687 |
| oracle pooled FDR≤0.15 | 0.895782 / 0.149354 | 0.850829 / 0.289012 | 0.907953 / 0.121147 | 0.624060 / 0.389706 |

该单折前沿使用 held-out 标签选阈值，只用于 detector 快筛，不能作为正式提交阈值或
隐藏集估计。它揭示 fold0 的核心瓶颈同时包含候选形成和排序：vehicle 在 0.001 floor
仍漏 17/133，而达到 pooled FDR15 时只保留 83/133。新 detector 必须至少提高 vehicle
floor Recall，并在 FDR15 保留更多 ship/vehicle TP；只增加低质量候选不准入。

本地冻结输入与输出索引：
`outputs/HERA-GUARD-V4-DETECTOR-SCREEN-20260830/y5_fold0/`。

## 7. 下一步停止规则

1. Q0 若不改变 TP–FP 排序，则 metadata-only 停止；
2. Q1 相对 detector 在 Normal-CV3 FDR15 必须提高 Recall，且任一粗类 Recall 不得下降超过 0.5pp；否则不进入部署；
3. Q2 只有在 Q1 基础上继续改善 active TP/FP 才进入 D1/D2；
4. 所有正向 Normal-CV3 结果必须复验 Hard10K 与 source-disjoint sentinel；
5. selective TTA 只允许 aircraft bypass、ship/vehicle 小比例 tile 路由，以及 same-fine support 或高 OMQ 质量的严格 novel budget；
6. 外部数据只做 coarse/objectness 预训练，不把外部细类映射为官方 25 类；
7. 未经过两套冻结评测和正式部署计时的模型，不得替换 v2 Docker incumbent。
