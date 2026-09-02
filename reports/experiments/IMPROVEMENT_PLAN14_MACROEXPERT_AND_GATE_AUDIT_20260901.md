# 改进方案14：MacroExpert-M 实施与评测门禁自查

## 1. 结论先行

方案14值得做，当前已进入 fold0/40 epoch 快筛。它不是再做一次共享 25 类模型微调，而是把正式平台已经很强的 Aircraft 完全旁路，只让新专家负责正式薄弱的 Ship 四类与 Vehicle，并显式学习 `AIRCRAFT_REJECT`。该设计和正式结果的误差结构一致：Aircraft 已强，Ship/Vehicle 的 Recall 与 FDR 才是总分瓶颈。

同时，过去一天“全部失败”不能直接解释为所有思想均无效。现行首筛存在两个系统性偏严点：

1. 用 40 epoch 候选直接对 160 epoch incumbent；既有配对审计显示，40 epoch 普通模型相对 160 epoch 在 Normal/Hard/Sentinel 可分别低约 2.612/3.550/5.840pp。40 epoch 只能用于同长度、同初始化、同 fold 的方向性比较。
2. Hard/Sentinel 每折只有 2 张 10K 合成图，Vehicle 只有约 50–70 个 GT，却执行“任一粗类 Recall 最多下降 0.5pp”。当 Vehicle GT=50 时，少 1 个 TP 就下降 2pp；0.5pp 门禁实际上连 1 个随机目标波动也不允许。

旧门禁结果保持原样，不能事后改判。自本实验开始同时报告“探索门禁”和“最终准入门禁”。

## 2. 冻结模型设计

### 2.1 主模型

- Aircraft 类别 4–23：沿用已部署的 Y5-S；
- 主模型的 Ship 0–3 与 Vehicle 24 输出在路由图像上全部丢弃；
- 这样新专家不可能损伤正式 Aircraft 的分类与置信度。

### 2.2 MacroExpert-M

- 架构：官方 `YOLO26-m`；
- 输入：1280；
- 六类标签：`HM, LQS, QHS, MS, vehicle, AIRCRAFT_REJECT`；
- 映射：0–3 原样，24→4，4–23→5；
- 推理：类别5先丢弃，0–3映射到正式0–3，4映射到正式24；
- 路由后主/专家的正式类别集合严格不相交，不做跨模型融合权重扫描。

### 2.3 fold0 数据视图审计

正式 CV3 manifest SHA256：`27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331`。

- 原始 train/val：2974 / 1507 图；
- 专家视图实际 train/val：2197 个采样实例 / 1507 图；
- 专家训练唯一源图：1556；
- aircraft-only 按 seed42 确定性保留 30%，丢弃 1418 张；
- repeat：HM×12、LQS×8、QHS×2、MS×1、Vehicle×8；
- 重映射后物化框数 0..5：138 / 182 / 1810 / 2123 / 2285 / 9821；
- cross-split image=0，cross-split group=0；
- 无缺图、缺标签、非 25 类源标签。

另按方案14抽取 100 张唯一源图形成 5 张带框 contact sheet，并由 Codex 逐卡视觉
审核：100/100 重映射与可见对象一致，未发现飞机成为未标背景、类别跨粗类错映、
越界框或退化框。结构化决定见
`improvement_plan14_macroexpert_review100_v1.json`。

注意：repeat 是图像级采样，混合图里的其他框也随图重复；因此必须报告上述物化框数，不能把它描述成精确的框级采样。

## 3. 已实现代码

- `scripts/build_macroexpert_fold_view.py`：确定性六类视图、图像级 repeat、aircraft-only 下采样与泄漏审计；
- `scripts/train_yolo_fixed_dataset.py`：固定 last、YOLO26-m、1280、AdamW、RandomRotate90 的单候选训练；
- `src/rsdet/models/ultralytics_adapter.py`：新增 `drop_labels`，保证拒识类先丢弃再执行正式标签映射；
- `scripts/compose_macroexpert_predictions.py`：按图像 fold 做互斥路由；
- `scripts/run_multifamily_cv3_pseudo_eval.py`：支持 MacroExpert 标签空间、1280 和可配置 tile；
- `scripts/audit_proxy_gate_resolution.py`：把 0.5pp 门禁换算成实际可损失目标数和二项分辨率；
- 对应专项测试覆盖数据映射、泄漏、拒识丢弃、标签映射和互斥路由。

## 4. 服务器执行证据

- 隔离代码目录：`/root/autodl-tmp/xh-202625-macroexpert`，未修改服务器旧 dirty worktree；
- 官方 YOLO26-m SHA256：`401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`；
- 1 epoch / 3% 数据 smoke：complete；
- 模型：21,782,140 参数，预训练迁移 756/768 项；
- 1280/batch4 训练峰值约 9.61GiB，AMP/RandomRotate90/有限损失均正常；
- smoke last SHA256：`57f9fad532a953bbc98cb82df25fd556b6fd095c37a8967218d37e6c63d1c092`；
- fold0 40 epoch 当前在 screen `macroexpert-40` 运行，产物目录 `/root/autodl-tmp/results/MACROEXPERT-M-V1/fold0-40ep`。

## 5. 两级评测合同（冻结）

### 5.1 探索门禁：用于 40 epoch/fold0 排序

1. 仅与相同 epoch、相同 fold、相同初始化策略的控制比较；
2. Normal CV3 是主排序集，使用官方 matching 与平台评分公式；
3. 目标是 Ship+Vehicle 的平台分或宏 Recall/FDR 有正向变化；
4. Aircraft 由主模型身份旁路，要求输出逐框一致；
5. Hard/Sentinel 只作灾难性退化否决，不再要求每个小样本粗类都在 0.5pp 内；
6. 每粗类容忍度按离散计数报告：`max(0.5pp, 2 / GT_count)`，这只是探索容忍度，不是最终准入放宽。

### 5.2 最终准入门禁

候选先扩到三折且达到与控制相同的充分训练长度，再执行：

- Normal 来源/组隔离 CV3；
- Hard10K；
- 全新 Sentinel；
- Background-100MP；
- 3090 固定时延与 Docker 逐框一致性；
- 独立通过的模块才能组合；
- 正式平台仅用有限提交做最终验证，不能围绕一次隐藏结果反复拟合。

旧的 0.5pp 严格门禁继续并列输出。探索通过不等于可提交，严格门禁失败仍不得进入 full/Docker。

## 6. 后续顺序

1. 完成 MacroExpert-M official-only fold0/40ep；
2. 跑 Normal 路由 replay，并与 Y5 原基线比较；
3. 跑 Hard 与 Sentinel 的固定阈值和独立 FDR frontier；
4. 若有方向性，再跑同长度 M25-1280 控制，区分“模型容量/分辨率”与“专家标签空间/采样”的收益；
5. official-only 有效后才加入 frozen Background100，避免一次改变两个因素；
6. 只有探索门禁通过才进入 folds1/2；三折确认后才考虑唯一 full 配方。

未物化的 `Formal-Anchor Proxy` 当前不能作为真实 P10 门禁。现有文件只是正式结果与分布推断记录，不能虚构成已经存在的独立数据集；应等其按来源/组完成真实抽样并冻结 SHA 后再加入。

## 7. MacroExpert-M fold0/40ep 最终结果（2026-09-02）

服务器任务状态为 `complete`，训练与三套冻结评测均正常结束：

- `results.csv` 恰好 40 个 epoch，有限损失；
- fold0 `last.pt` SHA256：`6606dc24f41da603f6e63dc4134ad59e9e20def82952834dc4e8e8c07021d837`；
- 评测的六个 frontier SHA 已写入 `evaluation/RESULT_SHA256.txt`；
- 训练标签映射、100 张视觉审核和门禁分辨率审计均保留；
- 完整必要回传已落到 `outputs/MACROEXPERT-M-V1/`；仅排除无决策价值的
  1-epoch smoke 权重，保留最终 best/last、全部逐框预测、评测、审核与日志。

以下比较均为冻结阈值网格、相同数据和 official matching。2026-09-02
回传后的复核发现，frontier 同时保存了三种不同聚合：pooled、25 个细类整体
macro，以及官方实际使用的“先在每个粗类内对细类 macro，再对 Ship/Aircraft/Vehicle
三大类求平均”。旧表误把 25 细类整体 macro 标成了平台门禁口径，现已更正为
`ranking_per_coarse` 的三大类平均；这项更正不会改变负向结论，只会使负向证据更强。
单位为百分比，`Δ` 为候选减基线。

| 数据集 / pooled-FDR 目标 | 基线 Recall | 候选 Recall | ΔRecall | 基线 FDR | 候选 FDR | ΔFDR |
|---|---:|---:|---:|---:|---:|---:|
| Normal / 0.15 | 73.636 | 72.746 | -0.889 | 29.862 | 31.849 | +1.987 |
| Hard / 0.15 | 52.011 | 52.289 | +0.278 | 19.917 | 20.457 | +0.539 |
| Sentinel / 0.15 | 67.267 | 63.730 | -3.537 | 16.933 | 17.806 | +0.874 |

Normal 0.15 的粗类分解揭示了主要问题：

- Ship：Recall `72.04% → 71.73%`（-0.31pp），FDR `34.67% → 40.52%`（+5.85pp）；
- Vehicle：Recall `57.21% → 54.98%`（-2.24pp），FDR `41.77% → 42.15%`（+0.38pp）；
- Aircraft：由主模型旁路，变化仅来自跨折阈值选择的细小扰动，未形成可利用收益。

Sentinel 0.15 上 Vehicle Recall `51.85% → 43.21%`（-8.64pp），属于明确的跨域退化。Hard 0.15 虽然 Vehicle Recall 增加约 3.26pp，但同时 Ship Recall 下降约 2.87pp、官方三大类平均 FDR 增加约 0.54pp；这不是三个代理集一致的方向。

MacroRisk V2 也不支持准入：候选的交叉拟合宏 Recall 为 `62.892%`，低于基线 `63.902%`；绝对分 `63.771`，低于基线 `64.438`。两者 Recall 门禁通过概率均为 0，候选没有改善稳健性。

## 8. 门禁判定与后续控制

结论：`complete_negative_no_admission`。

1. 旧严格门禁：失败。Normal、Hard、Sentinel 均未形成 Recall/FDR 同向改善，且 Sentinel Vehicle 有明显退化。
2. 同长度探索门禁：同样失败。即使允许按离散目标数放宽小样本波动，Normal 主排序集仍无正收益，三套代理的粗类方向也不一致。
3. 不启动 folds1/2、不训练 full、不进入 Docker。
4. 不启动 M25-YOLO26M-1280 同长度控制。该控制原本只用于拆分“明确正向候选”的容量/分辨率来源；当前 MacroExpert 本身没有正向信号，继续运行不能改变准入结论，只会回答一个已失去决策价值的问题。
5. 本实验否定的是当前 `六类专家 + repeat + 30% aircraft reject + 互斥替换` 的整体配方，不应泛化为“所有专家模型均无效”。尤其是 Ship FDR 上升表明当前专家校准和重复采样比模型容量更可能是瓶颈。

## 9. 资产索引

- 本地结果根目录：`outputs/MACROEXPERT-M-V1/`；
- Normal：`evaluation/normal/{baseline,candidate}_frontier.json` 与两份 `*_macro_risk_v2.json`；
- Hard：`evaluation/hard/{baseline,candidate}_frontier.json`；
- Sentinel：`evaluation/sentinel/{baseline,candidate}_frontier.json`；
- 数据审计：`fold0-view/audit.json`；
- 视觉审核：`review100-v1/review_manifest.json` 与 5 张 contact sheet；
- 训练合同与曲线：`fold0-40ep/train_contract.json`、`fold0-40ep/runs/foundation/results.csv`；
- 结果校验：`evaluation/RESULT_SHA256.txt`。

服务器 A（AutoDL 内蒙B区 394机，实例 `678c4cb81e-4d0c1e52`）已在
2026-09-02 01:02（Asia/Shanghai）完成结果回传与报告记录后关机；Safari
控制台复核状态为“已关机”。

## 10. 同赛道 DEIM-HCL-M fold0/40ep 对照（2026-09-02）

该路线复现公开同赛道仓库中的 DEIM-HCL 思路，定位为
`research_only_unlicensed_reference`：仅用于验证结构方向，不具有部署、分发或正式提交授权。
本次是同一 fold、初始化、训练长度与 seed 下的 DEIM-M / DEIM-HCL-M 配对快筛，不能替代三折结论。

重要口径限定：该同赛道复现实验沿用了
`analyze_single_split_official_frontier.py`。其中 `macro_recall/macro_fdr` 是 25 个细类
整体平均，`per_coarse` 是粗类 pooled counts；它没有输出正式
`platform_observed_20260831` 所要求的三个 `ranking_per_coarse`。因此下述“门禁”是历史
单折诊断门禁，不是平台正式准入门禁，不能与 MacroRisk V2 的官方三大类平均直接比较。

训练正常完成 40 epoch，最终 checkpoint 为 epoch 39，`last.pth` SHA256 为
`c3544555bf0c3791fb83307b5fbd978f825291085301bdf4f2249b010485e908`。
训练后首次推理暴露两个实现问题：HCL decoder 未在载入 state dict 前物化，以及非切片路径未清理裁剪后零面积框。
恢复仅修复推理代码并复用同一 checkpoint，没有重训、改模型、改数据、改阈值或改超参数；最终在
452,082 个低阈值输出中审计并丢弃 18 个零面积框。恢复证据见
`posttrain_recovery_manifest.json`。

### 10.1 Normal 配对结果

Normal fold0 在 pooled FDR=0.15 的同长度比较为：

| 指标 | DEIM-M 基线 | DEIM-HCL-M | 变化 |
|---|---:|---:|---:|
| pooled Recall | 75.143% | 76.639% | +1.497pp |
| macro Recall | 65.195% | 67.153% | +1.958pp |
| Ship Recall | — | — | +0.884pp |
| Aircraft Recall | — | — | +1.378pp |
| Vehicle Recall | — | — | +11.278pp |

Normal 的 pooled Recall、macro Recall 与三个粗类 Recall 均同向改善，配对快筛门禁全部通过。
这证明 HCL 在同分布、同长度条件下具有明确的正向学习信号，尤其改善 Vehicle；它不是“训练完全无效”的路线。

对已回传的候选逐框预测按正式协议复算后，在同一阈值 0.591 下三大类 macro 为：

| 粗类 | Recall | FDR | 诊断 pooled Recall | 诊断 pooled FDR |
|---|---:|---:|---:|---:|
| Ship | 32.484% | 7.205% | 67.403% | 14.085% |
| Aircraft | 75.489% | 15.944% | 78.755% | 15.084% |
| Vehicle | 39.098% | 5.455% | 39.098% | 5.455% |
| 三大类平均 | 49.024% | 9.534% | — | — |

Ship pooled 67.40% 与 Ship 内四个细类 macro 32.48% 的巨大差距，说明模型主要命中高频
Ship 细类，稀有 Ship 细类仍很弱。由于 DEIM-M 基线逐框预测未保存在当前两台可用服务器，
不能诚实补算正式口径的配对增量；历史的 +1.958pp 只能继续作为 25 细类整体 macro 的
方向性证据，不能升级为官方口径收益。

### 10.2 Hard 与 Sentinel-B 冻结复核

Hard 使用各模型自己的 FDR=0.15 frontier；Sentinel-B 使用相应 Hard 阈值冻结外推，未在 Sentinel 上重新选阈值。

| 数据集 | Δ pooled Recall | Δ macro Recall | Δ FDR | 关键粗类变化 |
|---|---:|---:|---:|---|
| Hard / FDR15 | +1.036pp | +1.559pp | -0.252pp | Ship +1.356pp；Aircraft +1.511pp；Vehicle -4.000pp |
| Sentinel-B / frozen threshold | +0.285pp | +1.734pp | +2.433pp | Ship -0.938pp；Aircraft 0；Vehicle +10.000pp |

Hard 的 Vehicle 只有 50 个 GT，-4pp 等于恰好少检 2 个目标；按探索容忍度
`max(0.5pp, 2 / GT_count)` 可视为边界内波动。Sentinel-B 的 Ship 有 320 个 GT，-0.938pp
等于少检 3 个目标，超过两目标容忍度 0.625pp；同时 Sentinel-B FDR 上升 2.433pp。
因此：

1. 旧严格门禁失败：Hard Vehicle 与 Sentinel Ship 均违反每粗类最多下降 0.5pp；
2. 同长度探索门禁仍失败：Hard Vehicle 可由离散计数解释，但 Sentinel Ship 超过两目标容忍度；
3. pooled/macro 指标在两套固定代理上仍为正，说明 HCL 是有价值的研究证据，但粗类稳定性不足以授权扩折或 full；
4. 冻结后续动作是 `stop_peer_hcl_route_without_scale_or_parameter_scan`：不扫尺度、不扫参数、不进入 folds1/2、full 或 Docker。

最终状态：`complete_positive_direction_but_no_admission`。这不是对 HCL 思路的彻底否定，而是对当前单折、当前公开实现和当前稳定性证据的准入拒绝。若未来重新研究，应先解决
Sentinel Ship 校准与跨域粗类稳定性，再以新的预注册合同重开；不能在本次结果上事后放宽门禁。

候选的正式口径回算也支持“不扩展”决定：Hard 阈值 0.696 的三大类 macro Recall/FDR
为 `27.804% / 4.721%`，其中 Vehicle Recall=0；相同冻结阈值在 Sentinel-B 为
`36.284% / 7.601%`。这些数字不是候选阈值优化后的正式得分，而是揭示当前模型在
稀有细类和跨域 Vehicle 上的绝对能力远弱于现有 YOLO 主线。

### 10.3 资产与完整性

- Normal 完整必要结果：`outputs/PEER-DEIM-HCL-M-FOLD0-40EP-V1/`，包含实际评测的
  `training/last.pth` 与完整 `predictions.json`；
- Hard/Sentinel-B 完整必要结果：
  `outputs/PEER-DEIM-HCL-M-FOLD0-FIXED-BENCHMARKS-V1/`，包含四份完整逐框预测与两份
  SHA 锁定的 GT；
- Normal 决策：`paired_decision.json`；
- 固定代理决策：`fixed_benchmark_decision.json`；
- 技术恢复审计：`posttrain_recovery_manifest.json`、`inference_summary.json`；
- 服务器完整 SHA 清单：两个结果根目录中的 `SHA256SUMS.txt`；
- 推理修复与恢复驱动：`scripts/infer_deim_coco.py`、`scripts/server/run_peer_deim_hcl_m_posttrain_recovery.sh`；
- 服务器端两个终态 SHA 清单均逐项校验通过；Normal 12 个正式文件和固定代理 27 个
  正式文件均在本地复核无缺失、无哈希不一致。GT 是随后从相同 SHA 的冻结代理目录补回。

MacroExpert 回传则按“服务器全量文件列表 → 本地逐文件 SHA”复核：排除 smoke 权重后
共 3,777 个文件，missing=0、bad=0。`evaluation/RESULT_SHA256.txt` 使用服务器绝对路径，
不能在本机直接运行 `sha256sum -c`，但清单内文件已经通过上述逐文件远端/本地哈希比较；
这不是完整性失败。

服务器 B（AutoDL 内蒙B区 317机，实例 `daeb408f8d-ebb5e3df`）已在
2026-09-02 03:10（Asia/Shanghai）完成两条结果 SHA 验收、33 个本地小型文件逐项校验和报告记录后关机；Safari 控制台复核状态为“已关机”。

## 11. 逐细类错误面复核（2026-09-02）

在 Normal 4,481 图、20,933 GT 上，使用两边各自 outer-fold 选择的 FDR15
阈值重新执行同一套官方匹配，并将每个 FP/FN 守恒分解为
`DUP/CLS/LOC/BG/MISS`。两边分解计数均与官方 FP/FN 完全守恒。

| 项目 | 基线 | MacroExpert | 候选−基线 |
|---|---:|---:|---:|
| TP | 19,237 | 19,180 | -57 |
| FP | 3,433 | 3,482 | +49 |
| FN | 1,696 | 1,753 | +57 |
| FP_DUP | 392 | 465 | +73 |
| FP_CLS | 918 | 939 | +21 |
| FP_LOC | 55 | 52 | -3 |
| FP_BG | 2,068 | 2,026 | -42 |
| FN_MISS | 723 | 762 | +39 |

关键细类结论：

- `MS`：TP -28、FP +84，其中 FP_DUP +63、FP_BG +24、FN_MISS +21；
- `QHS`：只增加 1 TP，却增加 27 FP；
- `HM/LQS`：TP 均不变，FP 分别 +4/+3；
- `Vehicle`：TP -9、FN_MISS +10、FP_DUP +11；虽然 FP_BG -14，Recall
  损失仍使官方车辆子分下降。

因此失败机制已经定位：当前图像级 repeat 没有提高稀有 Ship 的 TP 下限，反而把
高频 MS 的重复框与背景响应放大；Vehicle 也不是单纯阈值偏高，而是有效 TP 与完全漏检
同时恶化。后续不得在此六类专家上继续扫描 repeat 倍数或阈值。

结构化原件：
`outputs/MACROEXPERT-M-V1/evaluation/normal/paired_fine_error_surface.{json,csv}`。

## 12. DEIM-HCL 的正式口径补算

历史单折脚本曾按 pooled FDR 选择工作点。修复为
`platform_observed_20260831` 后，Hard 的三粗类宏平均结果为：

| 模型 | 阈值 | Gate Recall | Gate FDR | 六质量子分均值 |
|---|---:|---:|---:|---:|
| DEIM-M | 0.626 | 32.370% | 8.593% | — |
| DEIM-HCL-M | 0.621 | 35.874% | 14.807% | — |

两者各自的单折质量 oracle（使用 held-out 标签，仅诊断）为：

- DEIM-M：阈值 0.706，六质量子分均值 56.136；
- DEIM-HCL-M：阈值 0.651，六质量子分均值 55.771。

HCL 在 Hard 上以 +6.213pp Gate FDR 换取 +3.504pp Gate Recall，按官方分段函数后的
六质量子分仍略低。将各自 Hard FDR15 阈值冻结到 Sentinel-B 后，HCL 的 Gate Recall
增加 4.580pp，但 Gate FDR 仍增加 1.479pp。三个粗类 Recall 在这次正式口径补算中均未
下降，拒绝原因已经收敛为“Recall/FDR 交换不合算”，而不是旧脚本所报告的 Vehicle
离散退化。结论仍为不扩折、不进 full，但原因记录必须采用本节。

正式口径补算原件位于
`outputs/PEER-DEIM-HCL-M-FOLD0-FIXED-BENCHMARKS-V1/` 下的
`*_platform_v2.json`。

## 13. 下一轮：独立的容量/尺度 2×2

MacroExpert 失败后，不再把 `M25-1280` 当作专家配方的事后解释控制。它被重新注册为
一个独立问题：保持 25 类标签空间和 Y5 训练方法不变时，模型容量与输入尺度本身是否
改善官方三粗类宏平均。冻结四格为 `S/M × 1024/1280`，全部 fold0、40 epoch、
seed42、总 batch8、RandomRotate90、last checkpoint。

四格只允许同长度比较；任何 pooled-only 改善都不能准入。完整合同、驱动与停止条件见
[`YOLO_CAPACITY_SCALE_2X2_PLAN_20260902.md`](YOLO_CAPACITY_SCALE_2X2_PLAN_20260902.md)。
四格现已全部完成。S→M 在 1024/1280 两个尺度均为负；S1024→S1280 是唯一强正向
尺度信号，但因候选自身 FDR15 oracle 比基线增加 0.47pp，未通过原预注册严格门。
原门禁结论保持不变。随后另立的 S1280 全量操作候选、配对错误分解与风险边界见
[`S1280_FULL_CANDIDATE_AND_PLAN14_CLOSURE_20260902.md`](S1280_FULL_CANDIDATE_AND_PLAN14_CLOSURE_20260902.md)。
