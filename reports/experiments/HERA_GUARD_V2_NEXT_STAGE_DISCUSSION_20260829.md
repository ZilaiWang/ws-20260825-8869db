# HERA-Guard V2 官方难度代理实验总览与下一阶段讨论材料

日期：2026-08-29
状态：`active / results-partially-complete`
用途：供项目负责人、团队成员和外部推理模型共同讨论下一阶段；不是最终比赛报告，也不把代理集结果表述为隐藏测试集成绩。

## 0. 结论摘要

当前工程链已经完成 Docker、输入输出合同、类别映射、10K 切片与融合的闭环，预测评平均推理时间为 2.33 秒，工程不是主要风险。预测评使用临时 M1 fold0 权重得到总体 Recall 0.8022 / FDR 0.2078，说明正式版本不能继续使用该临时模型。

本轮在更接近预测评类别组成的 `trial-mix` 10K 代理集上重新审视了全部候选形成与风险解析方案。最重要的事实是：

1. 四个 held-out 候选源 `Y5-ROT + Y5-800 + M3 RT-DETR-L + COPH` 在同细类 NMS=0.60 后，候选 Recall 已达到 **0.9731**；
2. 候选层已经超过 0.94，但把候选排序为低虚警输出后，当前最好只有 **Recall 0.8601 / FDR 0.1478**，在 FDR≈0.20 时为 **0.8791 / 0.1982**；
3. 飞机在代理集上已接近饱和，主要缺口集中在舰船和车辆的目标/背景区分；
4. 工作点错误中 77.0% 的 FP 为 `FP_BG`，91.1% 的 FN 为阈值后缺少可用输出的 `FN_MISS`；bbox 定位和细类错误数量不足以解释 0.8601→0.94 的缺口；
5. 即便违规使用整个开发代理集标签，为三粗类分别选最佳阈值，同集上界也只有 0.8638@FDR≈0.15、0.8823@FDR≈0.20。继续扫描阈值不可能解决问题；
6. 当前最值得等待的三项结构实验是：四源自然 proposal verifier 重训、背景完整检测训练、三粗类检测器再由 P03 恢复细类；独立 source-disjoint sentinel 已冻结，用于拒绝开发代理集过拟合。

因此，本阶段不是“找一个更好的阈值”，而是检验以下结构假设：

> 先提高检测器对真实背景的目标性，再把细类识别从候选形成中部分解耦，并用 proposal-domain 的像素验证器只解析少量困难对象。

94% 目前是候选上限层已经满足、风险可用层尚未满足的目标，不能宣称已经实现。

---

## 1. 赛题、预测评与本地评价之间的关系

### 1.1 官方刚性约束

当前初赛核心约束为：

- 25 个细类输出，官方按舰船、飞机、车辆三个大类汇总 Recall/FDR；
- 总体 Recall 至少 85%；
- 总体 FDR 不高于 20%；
- RTX 3090 或同等算力处理 10K 大图不超过 20 秒；
- 正式阶段最多提交 5 次，必须减少依赖平台试错。

比赛最终又分别比较三大类 Recall/FDR 和推理时间，因此不能只优化一个由飞机数量主导的 pooled 指标。

### 1.2 预测评结果

平台预测评使用临时 M1 fold0 工程镜像，整数计数可由六项指标精确还原：

| 大类 | TP | FP | FN | Recall | FDR |
|---|---:|---:|---:|---:|---:|
| 舰船 | 784 | 169 | 143 | 0.845739 | 0.177335 |
| 飞机 | 535 | 142 | 128 | 0.806938 | 0.209749 |
| 车辆 | 76 | 55 | 73 | 0.510067 | 0.419847 |
| 总体 | 1395 | 366 | 344 | **0.802185** | **0.207836** |

平均推理时间为 **2.3318 秒**。这说明：

- Docker、GPU、输入输出和时延链路成功；
- 舰船基本保持本地水平；
- 飞机是绝对 TP 缺口的主要来源；
- 车辆 Recall/FDR 都很差；
- 当前镜像不是正式首发候选。

### 1.3 为什么原始 OOF 高估了预测评

官方训练/OOF 目标组成约为舰船 12.8%、飞机 85.3%、车辆 1.9%；预测评约为舰船 53.3%、飞机 38.1%、车辆 8.6%。原始 OOF pooled Recall 受到大量、较容易飞机样本主导。

仅保持本地三大类 Recall 不变、替换为预测评类别比例，总体 Recall 就会由约 0.917 降至约 0.863；其余约 6 个百分点才更接近真实域外退化、训练数据不足和大图部署差异。

### 1.4 评价账本必须分层

| 层级 | 作用 | 能否宣称隐藏集泛化 |
|---|---|---|
| 正式 CV3 OOF | 检验来源/机场代理组互斥下的基础训练能力 | 可以作为本地无偏估计，但类别组成偏斜 |
| `trial-mix` 10K 开发代理 | 配对比较切片、候选池、排序与大图错误 | 不可以；已用于多轮开发 |
| source-disjoint sentinel | 对冻结方案做方向复验 | 比开发代理更强，但仍不是官方隐藏集 |
| 预测评平台 | 工程与域外诊断 | 官方明确不计正式成绩 |
| 正式测评 | 最终依据 | 可以 |

后续所有正结果必须标明属于哪一本账，不能把 OOF、开发代理、sentinel 和平台数字串成一条“持续上涨”的曲线。

---

## 2. 官方难度代理集合同

### 2.1 开发 `trial-mix`

- 6 张 10000×10000 mosaic；
- 每折 2 张，只使用该折 held-out 来源图；
- 共 600 张唯一来源图，不复用；
- 2,158 个 GT：舰船 954、飞机 1,020、车辆 184；
- 类别占比 44.2% / 47.3% / 8.5%；
- 每张大图保留来源图、fold、seed 与组合审计。

该比例比原始 OOF 更接近预测评，尤其显著提高了舰船和车辆权重；实际舰船比例仍低于平台约 53%，所以不能认为它比隐藏集更难或完全等价。

### 2.2 独立 sentinel

- 6 张 10000×10000 mosaic；
- 600 张新来源图；
- 显式排除开发 `trial-mix` 的全部 600 张来源图；
- 1,969 个 GT：舰船 933、飞机 874、车辆 162；
- 类别占比 47.4% / 44.4% / 8.2%；
- GT SHA256：`eb1f8850624d77252b568b3515b1953d57adbd6234eed1dec6b467ff5f48c211`。

sentinel 只允许：

1. 使用开发集已经冻结的候选配置；
2. 使用开发集冻结的阈值；
3. 报告迁移方向和实际 FDR 漂移；
4. 不在 sentinel 上重新选 NMS、融合权重、阈值或训练轮数。

### 2.3 为什么仍需要平台提交

两套代理集都由官方训练来源图重新组合，能够模拟类别比例、10K 切片和背景数量，却不能创造真实未见传感器、GSD、压缩方式、机场/港口背景和标注习惯。它们用于减少无效提交，不能替代最终平台验证。

---

## 3. 从单模型到四源候选的实验演化

### 3.1 候选源

| 缩写 | 模型/视图 | 主要角色 |
|---|---|---|
| Y5-ROT | YOLO 系主检测器，identity + 90°视图 | 基础候选、旋转一致性 |
| Y5-800 | 更小 tile / 更高目标有效像素尺度 | 小目标、车辆候选 |
| M3 | RT-DETR-L | 异构候选，降低同族错误相关性 |
| COPH | 存在性正则化 Y5 分支 | 新背景/长尾互补候选 |

所有代理推理均严格使用对应 held-out fold checkpoint，不允许某张来源图由见过它的模型生成预测。

### 3.2 多尺度与异构候选结果

| 候选池 | candidate Recall | ship / aircraft / vehicle | Recall@FDR≈0.15（原始分数） |
|---|---:|---:|---:|
| Y5-ROT + Y5-800 | 0.9551 | 0.9434 / 0.9843 / 0.8533 | 0.7238 |
| Y5-800 + M3 | 0.9671 | 0.9518 / 0.9971 / 0.8804 | 0.7655 |
| Y5-ROT + Y5-800 + M3，NMS=.70 | 0.9713 | 0.9570 / 0.9961 / 0.9076 | 0.7655 |
| 三源，NMS=.60 | 0.9690 | 0.9518 / 0.9961 / 0.9076 | 0.7776 |
| **三源 + COPH，NMS=.60** | **0.9731** | **0.9570 / 0.9971 / 0.9239** | **0.8035** |

结论：

- 候选 Recall 已过 0.94；
- M3 比另一个同族 Y5 视图更互补；
- COPH 单独不强，但加入候选池使固定风险 Recall 增加约 2.60pp；
- NMS=.60 是当前开发代理上的候选安全/重复框折中点；
- 若最终时延只允许两个模型，优先比较 Y5-800 + M3，而不是两个 Y5 视图。

### 3.3 候选层残余错误

三源阶段未得到正确细类候选的 60 个 GT 中：

- 50 个定位不足；
- 9 个细类失败；
- 1 个粗类失败。

这说明候选层最后约 3% 缺口偏向小目标/边界/尺度定位；但从 97.31% 候选到 86.01% 可用结果的 11.3pp 损失主要发生在排序与背景拒识阶段。

---

## 4. proposal 像素验证与风险解析

### 4.1 P03 对象分类能力

P03 探索期对 GT tight crop 做 ConvNeXt-Tiny 微调：

| 分辨率 | 三折 macro Recall | macro F1 | accuracy | aircraft20 Recall |
|---|---:|---:|---:|---:|
| tight-224 | 0.9703 ± 0.0078 | 0.9709 | 0.9797 | 0.9843 |
| tight-336 | 0.9753 ± 0.0012 | 0.9761 | 0.9807 | 0.9867 |

这证明正确对象 crop 上的细类识别上限很高，但不能推出真实 proposal 的目标/背景可分性同样高。

在后来冻结的正式 CV3-V2 口径下，20,933 个对象 pooled OOF 为 macro Recall
**0.9287**、macro F1 0.9367、accuracy 0.9593；aircraft20 macro Recall 0.9524，
ship4 仅 0.7950。探索期约0.97应理解为早期上限诊断，后续正式决策优先引用0.9287。

### 4.2 P04 教师结论

探索期统一 train-RMS 读出为：

| 教师特征 | exploratory linear probe macro Recall |
|---|---:|
| DINOv2-B CLS+patch mean | 0.9098 |
| ConvNeXt R0 | 0.8797 |
| DINOv2-S | 0.8629 |
| CleanDIFT map0 | 0.8293 |

正式 CV3-V2 frozen-feature probe 的 native macro Recall 则为 DINOv2-B 0.8294、
ConvNeXt 0.7815、CleanDIFT 0.7036；绝对值下降但教师排序不变。DINOv2-B 在干净
crop 表征探测中更强，但后续 proposal-domain DINO 线性开放集头没有转化为固定风险
收益；CleanDIFT 更低且计算更重，因此当前不恢复扩散教师。

### 4.3 25 类 + background 的 open-set verifier

使用三源 NMS=.60 的真实 held-out proposal 构造训练集：

- IoU 达到对应粗类官方门槛：foreground；
- 与所有 GT 最大 IoU≤0.05：clear background；
- 中间不确定 IoU 区间不用于训练；
- 每个 held-out fold 模型只使用另两折 proposal；
- 从同折合法 P03 checkpoint 初始化；
- tight=1.0 训练与推理保持一致。

三源结果：

| 解析方式 | Recall/FDR @.15 | Recall/FDR @.20 |
|---|---:|---:|
| 原始三源分数 | 0.7776 / 0.1499 | — |
| tight identity 风险头 | **0.8378 / 0.1492** | 0.8633 / 0.1997 |
| 同粗类双假设 | 0.8285 / 0.1498 | **0.8665 / 0.1940** |

四源沿用相同思想：

| 解析方式 | candidate Recall | Recall/FDR @.15 | Recall/FDR @.20 |
|---|---:|---:|---:|
| direct open-set 融合 | 0.9731 | 0.8499 / 0.1470 | 0.8740 / 0.1978 |
| **identity 像素风险头** | 0.9731 | **0.8601 / 0.1478** | **0.8791 / 0.1982** |
| 同粗类双假设 | 0.9754 | 0.8582 / 0.1508 | 0.8791 / 0.1996 |

当前默认仍保持 detector 细类，避免无保护 relabel。

### 4.4 预算化像素复核

全量处理 46,566 个 proposal 的 tight crop 推理约 365 秒，即每张代理 10K 图约 60.9 秒，无法直接部署。metadata router 每图只复核最有价值的候选：

| 每图 crop 预算 | 估计 crop 时间 | Recall/FDR @.15 | Recall/FDR @.20 |
|---:|---:|---:|---:|
| 0 | 0.0s | 0.7850 / 0.1487 | 0.8272 / 0.2003 |
| 256 | 2.0s | 0.8211 / 0.1468 | 0.8582 / 0.1958 |
| 512 | 4.0s | 0.8438 / 0.1467 | 0.8689 / 0.1980 |
| **1024** | **8.0s** | **0.8531 / 0.1493** | **0.8716 / 0.1968** |
| 2048 | 16.1s | 0.8545 / 0.1471 | 0.8744 / 0.1980 |
| 全量 | 60.9s | 0.8601 / 0.1478 | 0.8791 / 0.1982 |

1024 是当前工程 Pareto 点；这些时间尚未包含全部 detector、切片、融合与 JSON 开销。最终部署必须重新在 RTX 3090 真实 10K 链路测速。

---

## 5. 当前最好工作点的错误分解

四源+tight identity、目标 FDR≈0.15：

- TP=1,856；
- FP=322；
- FN=302；
- Recall=0.8601；
- FDR=0.1478。

| FP 原因 | 数量 | FP 占比 |
|---|---:|---:|
| FP_BG | **248** | **77.0%** |
| FP_DUP | 47 | 14.6% |
| FP_LOC | 18 | 5.6% |
| FP_CLS | 9 | 2.8% |

| FN 原因 | 数量 | FN 占比 |
|---|---:|---:|
| FN_MISS | **275** | **91.1%** |
| FN_LOC | 18 | 6.0% |
| FN_CLS | 9 | 3.0% |

分项 Recall 约为：

- ship：0.7914；
- aircraft：0.9863；
- vehicle：0.5163。

FP_BG/FN_MISS 主要集中于舰船细类 2、3 和车辆类 24；FN_MISS 又以 small/tiny 为主。由此得到三条边界：

1. bbox 扩散/框修正最多只直接覆盖少量 `FN_LOC`，暂不作为主线；
2. 仅做细类重分类最多覆盖少量 `FN_CLS`，也不能承担 8pp 增益；
3. 主线必须提高舰船/车辆目标性，同时保护低分真目标不被阈值删除。

### 5.1 94% 目标的数量级

开发代理共有 2,158 个 GT：

- 94% Recall 约需 2,029 TP；
- 当前 1,856 TP，还差约 173 TP；
- FDR≤15% 时，2,029 TP 最多容纳约 358 FP；
- 当前已有 322 FP，因此恢复 173 TP 时只能净增加约 36 FP。

四源候选 TP 上限约为 2,100，说明理论上仍有空间，但要求风险解析器保留约 96.6% 的候选真阳性，同时从四万级候选中只留下数百个 FP。这是严格的 proposal ranking 问题。

### 5.2 阈值同集上界

允许使用整个开发代理集标签，为三粗类联合选择最佳阈值：

| 目标 FDR | TP | FP | Recall | 实际 FDR |
|---:|---:|---:|---:|---:|
| 0.15 | 1,864 | 327 | **0.8638** | 0.1492 |
| 0.20 | 1,904 | 474 | **0.8823** | 0.1993 |

即使给予阈值不合法的同集优势，也几乎不能超过现有统一分数。因此后续关闭进一步全局/粗类阈值扫描。

---

## 6. 已完成但未准入的负向或局部互补实验

### 6.1 来源支持先验

四源候选中单/双/三/四源支持分别为 28,826 / 8,930 / 5,367 / 3,443。虽然不是常量，但加入 HGB 后只得到 0.8156/0.1423；低容量贝叶斯 odds 校准为 0.8100/0.1045，均过度淘汰单源真目标。停止扫描支持 IoU 和先验强度。

### 6.2 粗类均衡背景采样

整体 0.8587/0.1461，略低于自然采样 0.8601/0.1478；vehicle Recall 从 0.5163 增至 0.5489，但 ship/aircraft 下降。不能全局替换，只允许一次预注册的“自然 ship/aircraft + 均衡 vehicle”统一重校准。

### 6.3 收紧 NMS

NMS=.50 后 candidate Recall 从 0.9731 降至 0.9634；0.15 点整体仅增至 0.8638，vehicle 提升而 ship 下降。粗类组合 ship=.60、aircraft/vehicle=.50 在原始分数上又与基线等价。因此 NMS 不是独立主模块。

### 6.4 DINOv2 proposal 证据

- 26 类同细类融合：0.5885/0.1494；
- DINO foreground 与 detector 几何平均：0.8401/0.1464；
- 加入统一风险头：0.8411/0.1467；
- 均低于 0.8601。

DINO foreground AUC 约 0.97，说明有表征信号，但当前线性头/统一风险模型不能把它转为排序收益。停止融合权重扫描，不恢复 CleanDIFT。

### 6.5 三粗类阈值 cross-fit

原始四源分数 cross-fit 只由约 0.8035 提升至约 0.817，且实际 FDR 漂移。校准缺少像素可分性时不能解决车辆/舰船内部背景排序。

### 6.6 为什么这些负向结果仍有价值

它们共同排除了：

- 单纯调阈值；
- 继续收紧 NMS；
- 把多模型 agreement 当作硬先验；
- 冻结 DINO 线性头直接替换 ConvNeXt；
- 只靠细类 relabel；
- 在当前阶段优先做 bbox 扩散。

下一步的搜索空间因此明显收窄，不再重复低价值后处理。

---

## 7. 当前正在运行的结构实验

本节状态以 2026-08-29 本文提交时为准，所有数值结果为空的任务都不得提前宣称正向。

### 7.1 四源自然 tight verifier 重训

问题：旧 tight verifier 主要由三源 proposal 训练，再应用到新增 COPH 后的四源候选，存在 proposal-domain 迁移。

合同：

- 直接用四源 NMS=.60 proposal 重建 foreground/background manifest；
- 仍为 fold-heldout、tight=1.0、P03 初始化；
- 3 epoch、自然背景采样；
- 不改变融合权重；
- 重新报告 direct、identity、dual frontiers。

当前状态：推理中。

### 7.2 自然舰船/飞机 + 均衡车辆专家

问题：均衡背景模型只在车辆上表现出互补，不能直接拼接两个 HGB 阈值。

合同：

- 对完全对齐的同一 proposal 集，只替换车辆 crop evidence；
- ship/aircraft 使用四源自然 verifier；
- vehicle 使用既有 coarse-balanced verifier；
- 路由后重新训练统一 fold-heldout pixel risk head；
- 不扫描组合。

当前状态：已排队，等待自然 verifier 完成后自动运行。

### 7.3 coarse-specific 二分类 verifier

问题：26 类 + background 让目标性与细类竞争，尤其可能伤害舰船/车辆。

合同：

- 分别训练 ship/aircraft/vehicle 三个二分类 ConvNeXt-Tiny；
- 每个 held-out fold 只用另两折 proposal；
- foreground 按细类均衡、background 自然抽样；
- P03 backbone 初始化、tight=1.0；
- 先固定几何平均，再作为统一风险头单一新增字段复验。

当前状态：训练已完成，评估等待自然四源推理释放资源。

### 7.4 score-aware hard-negative 单因素

现有二分类器每个模型仅 3×40 batch，训练补集内大量背景从未进入梯度；均匀负采样又浪费容量在不可能跨阈值的极低分背景。

预注册变体：

- 架构、P03 初始化、分粗类、3 epoch、tight=1.0 均不变；
- 负样本按 `sqrt(max(detector_score, 1e-4))` 加权抽样；
- 每 epoch batch 从 40 增至 80，增加风险尾部覆盖；
- 正样本仍不按分数加权，保护低分 TP；
- 只有普通 coarse-binary 显示方向性但不足时才启动；
- 不扫描采样指数、融合权重或训练轮数。

当前状态：代码与测试完成，未启动。

### 7.5 背景完整检测训练

问题：官方 4,481 张训练图没有纯背景，而代理工作点 248 个 FP_BG 是绝对主因。

合同：

- 从 held-out OOF 高分错误区域构造 hard-negative tiles；
- 每折 hard tile 只来自该模型未训练过的来源折；
- 纯空 tile 与含其他真实目标的 tile 都保留，后者保留全部真实 YOLO 标签，避免把真目标教成背景；
- 每折加入 640 张 tile；fold0/1/2 纯空 tile 为 31/43/44；
- 从 Y5-ROT 固定 checkpoint 继续 20 epoch；
- 不用 held-out 标签选 epoch，只用固定 last；
- candidate Recall 下降不得超过 0.3pp，且固定风险必须提高。

当前状态：三折训练中，fold0 已持续正常推进。

### 7.6 三粗类检测器 + P03 细类恢复

问题：25 类长尾检测头可能把“是否是目标”和“是哪一细类”过早耦合。

合同：

1. 训练标签从 25 类映射为 ship/aircraft/vehicle；
2. 每折从同折 Y5-ROT checkpoint 迁移，固定 30 epoch；
3. 三粗类检测结果先完成大图候选形成；
4. 用对应 held-out P03 tight-224 checkpoint，只在预测粗类内部恢复细类；
5. 固定 detector 0.60 / crop 0.40；
6. 只有单源候选或固定风险正向才允许进入四源池。

当前状态：三折训练中；数据物化、类别映射与 held-out 纯度 smoke 均已通过。

### 7.7 Y3 历史候选增强复验

Y3 在原图 OOF 低阈值候选 Recall 约 0.989，但候选爆炸、排序和尾类失败，不能作为独立模型。唯一复验角色是给四源补候选：若 merged candidate Recall 净增不足 0.3pp立即停止；只有候选明确正向才允许沿用相同 tight verifier。

当前状态：已排队。

### 7.8 full-data 训练

- M3 RT-DETR-L full：训练中；
- Y5-L full：训练中；
- COPH full：已完成；
- full 模型不能在由其训练来源图拼成的代理集上给出无偏科学结果，主要用于最终 Docker 候选与隐藏平台验证。

---

## 8. 当前方法应如何定型

建议继续将最终方法定义为：

> **HERA-Guard V2：多尺度异构候选形成、全局对象唯一化与预算化 proposal-domain 风险解析。**

逻辑结构：

```text
10K 原图
  ├─ 多尺度切片
  ├─ 主检测器 / 异构候选器
  ├─ 全局坐标恢复
  ├─ 同细类安全聚合与唯一对象候选
  ├─ metadata 风险路由
  ├─ 只复核困难对象的 tight crop verifier
  ├─ 可选：粗类目标性 + 粗类内细类恢复
  └─ 统一风险分数与最终唯一结果
```

创新点不应描述为“用了多个模型”或“加入分类器”，而应强调：

1. **候选形成与风险解析解耦**：低阈值异构模型负责覆盖，不要求原始置信度可直接比较；
2. **全局对象而非 tile 预测**：跨 tile 证据先恢复到统一坐标并唯一化；
3. **proposal-domain 对齐监督**：验证器训练的正负例来自 held-out 检测器真实错误分布，而非普通随机背景 crop；
4. **非对称风险预算**：像素模型只处理接近工作点、最可能改变 TP/FP 的候选；
5. **严格停止条件**：来源支持、DINO、NMS、阈值等负向实验被系统关闭，避免把无效复杂度包装为贡献。

### 8.1 可能的最终部署档位

| 档位 | 模型 | 风险解析 | 预期用途 |
|---|---|---|---|
| Safe | 单 full Y5 | 简单融合/冻结阈值 | 工程回退，速度最稳 |
| Balanced | Y5-800 + M3 | metadata + 512/1024 crop | 当前最可能的正式首发 |
| Accuracy | 多源候选 | 1024 crop + 背景完整/粗类证据 | 只有真实 3090 <20s 才准入 |

最终不能直接部署三个 fold checkpoint。代理集使用 fold 模型是为了无偏；正式镜像必须使用全量训练对应资产，并记录初始化、训练数据、checkpoint 和 Docker digest。

---

## 9. 下一阶段决策树

```text
四源自然 verifier 完成
├─ 明显高于 0.8601/.1478
│  ├─ 执行唯一 natural/vehicle-balanced hybrid
│  └─ 冻结开发方案，进入 sentinel
└─ 不提高
   ├─ 读取 coarse-binary 结果
   └─ 不再延长 26-way verifier

coarse-binary 完成
├─ ship/vehicle 有方向性正增益但整体不足
│  └─ 执行一次 score_sqrt hard-negative
└─ 无方向性
   └─ 关闭 crop verifier 继续加容量

背景完整检测器完成
├─ fixed-risk 提升且 candidate loss ≤0.3pp
│  ├─ 作为 Y5-ROT 替换候选，不新增第五个同族模型
│  └─ 进入 sentinel
└─ 失败
   └─ 不再扩大伪背景训练

三粗类检测器完成
├─ 单源候选或 fixed-risk 正向
│  ├─ P03 恢复细类
│  └─ 与异构池做一次替换实验
└─ 不正向
   └─ 保持25类端到端检测

Y3 revisit
├─ candidate Recall 净增 ≥0.3pp
│  └─ 只沿用同一 verifier 解析
└─ <0.3pp
   └─ 永久关闭

任一开发正结果
├─ sentinel 冻结阈值同方向
│  └─ 进入 full-data + Docker 3090 测速
└─ sentinel 反向/严重漂移
   └─ 视为开发代理过拟合，不提交平台
```

---

## 10. 希望下一轮讨论重点回答的问题

以下问题需要基于本文完整证据回答，而不是泛泛推荐新模块：

1. 在候选 Recall 0.9731、但同集三粗类阈值上界仅 0.8638@FDR=.15 的条件下，最可能产生 5pp 以上提升的风险学习目标是什么？
2. 当前 26-way+background CE、coarse binary CE 和 HGB 风险头是否存在明显的目标函数错配？是否应改为 pairwise/listwise 排序、partial AUC、precision-constrained recall 或直接可微 FDR 约束？
3. 只有 6 张开发 mosaic、3 个来源折，但有 4.6 万 proposal。怎样设计分组训练/校准，避免把 proposal 数量误当成独立场景数量？
4. 对高分 FP_BG 与低分 TP 的区分，tight crop 是否缺少必要上下文？如果引入双视图，怎样做到单因素、低时延且不重复已经失败的 context=1.25 单视图？
5. 背景完整检测器和三粗类检测器若都只有局部收益，应优先替换候选源、融合 logits，还是只作为训练期教师蒸馏到 full Y5？
6. 在正式最多 5 次提交的条件下，应如何安排 Safe/Balanced/Accuracy 三个镜像的提交次序与停止规则？
7. 当前代理集仍来自训练分布。除官方平台外，是否存在不违反赛题约束、且一天内可构造的高价值外部背景压力测试？
8. 目标 0.94 是否应继续定义为 pooled Recall@FDR=.15，还是更合理地改为三大类底线 + pooled 排名的多目标准入？

不建议下一轮重新讨论：完整 DiffusionDet、全图扩散增强、更多同族 TTA、无门禁的注意力模块、继续扫描阈值或 NMS。这些方向与当前主要证据不匹配。

---

## 11. 代码、结果与文档索引

### 11.1 总报告与官方诊断

- `reports/experiments/HERA_GUARD_V2_94_RECALL_EXECUTION_20260829.md`：逐项实验流水账、结果与资源状态；
- `reports/submission/MODEL_AND_WEIGHT_FREEZE_AUDIT_20260829.md`：当前 Docker 模型/权重边界；
- `reports/experiments/M1_CV3_OOF_FORMAL_RESULT_AND_RECOVERY_AUDIT_v2.md`：M1 正式 OOF；
- `reports/experiments/P03-02-fine-tune-results.md`：对象 crop 分类上限；
- `reports/experiments/N1_FORMAL_P03_P04_EXECUTION_20260809.md`：正式 P03/P04 CV3-V2 结果。

### 11.2 代理集、候选与前沿

- `scripts/build_cv3_oof_pseudo_10k.py`：fold-heldout、来源互斥 10K 构造；
- `scripts/run_multifamily_cv3_pseudo_eval.py`：多模型 held-out 代理推理；
- `scripts/merge_pseudo_candidate_sources.py`：候选合并；
- `scripts/analyze_cv3_pseudo_candidate_ceiling.py`：candidate Recall；
- `scripts/analyze_cv3_oof_pseudo_frontier.py`：fold-heldout 风险前沿；
- `scripts/analyze_cv3_oof_pseudo_coarse_thresholds.py`：三粗类阈值；
- `scripts/evaluate_pseudo_with_frozen_thresholds.py`：sentinel 冻结阈值评估；
- `scripts/analyze_pseudo_workpoint_errors.py`：FP/FN 守恒错误分解。

### 11.3 proposal verifier 与风险头

- `scripts/build_cv3_pseudo_foreground_manifest.py`：真实 OOF proposal 正负例；
- `scripts/train_pseudo_open_set_verifier.py`：25类+background ConvNeXt；
- `scripts/rerank_cv3_pseudo_with_open_set_verifier.py`：open-set 推理；
- `scripts/train_pseudo_pixel_oer.py`：像素+几何风险头；
- `scripts/train_pseudo_coarse_binary_verifier.py`：粗类二分类与 score-aware 采样；
- `scripts/rerank_cv3_pseudo_with_coarse_binary_verifier.py`：粗类前景概率；
- `scripts/merge_coarse_crop_verifier_experts.py`：自然/均衡证据路由；
- `scripts/simulate_budgeted_pixel_verifier.py`：困难候选预算路由。

### 11.4 新检测结构

- `scripts/build_pseudo_hard_negative_tiles.py`：来源合法 hard-negative tile；
- `scripts/train_y5_background_complete.py`：背景完整短程续训；
- `scripts/train_y5_coarse_detector.py`：三粗类 detector；
- `scripts/classify_coarse_proposals_with_p03.py`：粗类内 P03 细类恢复；
- `scripts/train_full_m3.py`：M3 full-data；
- `scripts/train_full_coph.py`：COPH full-data。

### 11.5 服务器入口

- `scripts/server/run_four_source_tight_verifier_train.sh`；
- `scripts/server/run_existing_open_set_verifier_eval.sh`；
- `scripts/server/run_coarse_binary_verifier_train.sh`；
- `scripts/server/run_coarse_binary_hardscore_train.sh`；
- `scripts/server/run_coarse_crop_expert_hybrid.sh`；
- `scripts/server/run_y5_background_complete_cv3.sh`；
- `scripts/server/run_y5_background_complete_pseudo_audit.sh`；
- `scripts/server/run_y5_coarse_detector_cv3.sh`；
- `scripts/server/run_y5_coarse_detector_pseudo_audit.sh`；
- `scripts/server/run_four_source_sentinel_audit.sh`；
- `scripts/server/run_y3_trial_mix_revisit.sh`。

### 11.6 测试边界

本轮新增测试覆盖：

- held-out fold 与类别映射；
- 代理集来源排除和确定性；
- 候选合并/NMS；
- P03 粗类恢复；
- open-set、DINO、coarse binary 和 pixel OER 特征合同；
- frozen threshold sentinel；
- hard-negative tile 保留 GT；
- full M3/COPH 训练参数；
- safe fusion 与 Docker submission contract。

本文提交前仍需执行全仓 `pytest`、`ruff`、shell `bash -n`、敏感信息扫描和 GitHub 脱敏同步；服务器 checkpoint、数据集、账号、SSH 地址及密码不得进入仓库。

---

## 12. 当前最谨慎的判断

当前证据足以支持：

- 多尺度异构候选显著提高覆盖；
- proposal-domain tight 像素验证显著优于原始 detector 分数；
- 主要错误是舰船/车辆背景排序；
- 阈值、NMS、DINO 线性头和简单来源支持不足以继续提升；
- 工程时延有加入预算化困难对象复核的空间。

当前证据不足以支持：

- 固定风险 Recall 已达到 0.94；
- 开发代理结果可直接外推正式隐藏集；
- 四模型全量部署一定满足 20 秒；
- 扩散特征对最终检测有独立收益；
- 任一正在运行的新结构已经准入。

下一次正式提交应等到：至少一个结构实验在开发代理正向、source-disjoint sentinel 同方向、全量权重完成、3090 Docker 端到端测速通过，并形成可回退的单模型 Safe 镜像。
