# M1 正式 OOF 后下一阶段总计划 v1

更新日期：2026-07-25  
适用起点：`M1-CV3-OOF` 已完成  
状态：`current_execution_master_after_M1_formal_OOF`

## 1. 现在处于什么阶段

项目已经越过“建立可信划分、跑出第一个正式基线”的阶段。当前拥有：

- 正式来源组隔离三折 `cv3_airport_proxy_k60_v2`；
- 4,481 张图、20,933 个 GT 的 D00 数据锁；
- 正确 YOLO26-s 的三折低阈值 OOF；
- 4,481 张图恰好一次 held-out 预测和 55,548 个候选；
- 官方口径阈值曲线、逐折结果和计数守恒错误分解；
- P03/P04 的探索结果及正式 CV3 重放代码；
- M3、10K 工程和二阶段对象路线的既有合同。

因此，下一阶段不应继续泛泛搜索模型，也不能只按旧编号机械地“做下一个
P 实验”。现在应把正式 OOF 转化为三类证据：

1. **无偏基线证据**：cross-fit 阈值、逐折/逐来源组稳定性；
2. **对象级证据**：每个候选对应的 TP、细类错、重复、定位错、未归因 FP
   及其可追溯 crop；
3. **系统级证据**：对象学生、背景拒识、M3 互补和 10K 全局融合分别能解决
   哪些真实错误。

## 2. M1 结果真正说明了什么

### 2.1 总体门槛：有可行区间，但安全余量不足

同 OOF 探索阈值 `0.051`：

| 指标 | 结果 | 判断 |
|---|---:|---|
| Overall Recall | 0.9172 | 高于 0.85 |
| Overall FDR | 0.1957 | 仅比 0.20 好约 0.0043 |
| TP / FP / FN | 19,199 / 4,671 / 1,734 | 具备可用基线 |

在 TP 不变的理想化条件下，当前只多 128.75 个 FP 就会越过 FDR=0.20。
fold 0 和 fold 2 的 FDR 已分别为 0.2136、0.2160。因此：

- 可以说 M1 已找到通过总体门槛的可行工作区间；
- 不能说阈值 `0.051` 已正式冻结；
- 当前系统离内部目标 FDR≤0.17 还需净减少约 739 个 FP，且不能明显损失 TP；
- 第一优先级是建立 cross-fit 无偏基线并提高 FDR 安全余量。

### 2.2 总体成绩被飞机样本主导

飞机 GT 占全部 GT 的 85.3%。总体过线不能代替三大类判断：

| 大类 | 工作点 Recall | 工作点 FDR | 候选下限 Recall |
|---|---:|---:|---:|
| 舰船 | 0.8512 | 0.3828 | 0.8993 |
| 飞机 | 0.9338 | 0.1463 | 0.9394 |
| 车辆 | 0.6169 | 0.6161 | 0.7985 |

在当前原始分数上，允许每个大类各自选择一个阈值时，FDR≤0.20 下的最高
Recall 约为：

| 大类 | 最好阈值 | Recall | FDR |
|---|---:|---:|---:|
| 舰船 | 0.301 | 0.7983 | 0.1996 |
| 飞机 | 0.021 | 0.9357 | 0.1808 |
| 车辆 | 0.581 | 0.3806 | 0.1947 |

这证明单纯做大类阈值搜索不能解决舰船和车辆。后续必须按大类分治：

- 飞机：重点解决细类；
- 舰船：重点解决未归因 FP，同时保护已经刚过线的 Recall；
- 车辆：先提高候选召回，再解决极高 FDR。

### 2.3 官方排名口径：大类 = 细类指标简单平均（评分方案 V1.6）

评分方案 V1.6（2026-08-04）明确了官方排名口径：三大类各自的 Recall/FDR
= **大类内细类指标的简单平均**（船 4 型各 1/4、飞机 20 型各 1/20、车辆 1
型即 FSC 本身），每个队伍 7 项排名（三大类 × Recall/FDR + 时效性）二次
排序直接决定初赛方案/创新/落地三项各 10 分的打分区间；刚性门槛仍按三类
合并 pooled 计算。同一 M1 工作点 `0.051` 的两套口径：

| 大类 | 官方 macro Recall/FDR | pooled Recall/FDR |
|---|---:|---:|
| 舰船 | 0.7235 / 0.5201 | 0.8512 / 0.3828 |
| 飞机 | 0.9076 / 0.1571 | 0.9338 / 0.1463 |
| 车辆 | 0.6169 / 0.6161 | 0.6169 / 0.6161 |
| Overall | 0.8665 / 0.2335（25 细类平均） | 0.9172 / 0.1957 |

由此得到本计划与所有实验的指标纪律：

- **舰船 macro FDR 0.52 是当前最大官方排名风险**，由小细类权重放大：
  LQS（30 框）Recall 0.50 / FDR 0.667、HM（17 框）Recall 0.706 / FDR
  0.625，在船类中各占 25% 权重；飞机侧 TU-160 Recall 0.332（FN_CLS=
  241）、F-22 Recall 0.789 拖低 macro 均值；
- 小细类权重 = 类别数倒数，与样本量无关：修好 1 个 TU-160 框的宏观收益
  与 1 个 FA-18 框相同，因此 LQS/HM/TU-160/F-22 是各模块的优先靶点；
- 所有正式实验必须同时报告 pooled（门槛校验）与官方 macro（排名优化）
  两套数字，`scripts/evaluate.py` 已默认输出 `official_ranking` 块；
- 内部目标（FDR≤0.17 等）一律以官方 macro 口径计。

### 2.4 错误容量决定模块优先级

探索工作点的错误分解：

| 错误 | 数量 | 直接含义 |
|---|---:|---|
| FP_BG | 3,303 | 全部 FP 的 70.7%，P05 获得准入 |
| FP_CLS / FN_CLS | 1,115 | 全部 FN 的 64.3%，对象细分类获得准入 |
| FP_DUP | 187 | 当前普通图 OOF 中不是首要模型错误 |
| FP_LOC / FN_LOC | 66 | 只占 FN 3.8%，P06 暂缓 |
| FN_MISS | 553 | M3、车辆候选和 detector/tile 路线的入口 |

若能无副作用地修复所有 `FN_CLS`，总体 Recall 的理论容量约为 5.33 个
百分点；`FN_LOC` 的理论容量只有约 0.32 个百分点。这个数量级差异足以决定：

> 近期 GPU 和实现时间应投入对象细分类与背景拒识，不投入 bbox diffusion。

### 2.5 三个大类的具体瓶颈

#### 飞机

- `FN_CLS=1002`，`FN_MISS=173`，`FN_LOC=6`；
- 细类混淆远大于定位问题；
- TU-160 单类贡献 241 个 `FN_CLS`，与正式 CV3 中 fold 0 仅 9-shot 的
  来源组压力一致；
- SU-35、F-22、SU-34、KC-135、F-15 等也是重点混淆类。

结论：P03/P04 与 Pred-OOF crop 是飞机主线；DINOv2 只在正式证据支持后作为
教师，不能直接无条件进入推理。

#### 舰船

- `FP_BG=1125`，其中 MS 预测贡献 806；
- `FN_MISS=233`，其中 MS 为 173；
- 工作点 Recall 只有 0.8512，几乎没有允许错误拒绝的空间；
- 候选下限 Recall 可到 0.8993，说明低分候选中还有目标，但同时带来大量 FP。

结论：舰船需要“低阈值保召回 + 强背景拒识”，而不是继续抬阈值。

#### 车辆

- 工作点只有 248/402 个 TP；
- 候选下限也只有 321/402，Recall 上限 0.7985；
- `FN_MISS=147`，没有细类问题可供二阶段分类修复；
- 当前 FDR≤0.20 的分数工作点 Recall 只有 0.3806。

结论：车辆需要 M3 互补、单因素高分辨率/P2 小目标路径或大图切片尺度证据。
对象背景头能改善 FDR，但无法凭空找回没有候选的 81 个目标。

## 3. 对原计划的保留、调整和停止

### 3.1 保留

- M1 作为快速 25 类安全底座；
- 全局坐标对象唯一化与 10K 融合；
- P03 tight-224 ConvNeXt-T 轻量对象学生；
- P04 DINOv2-B 教师对照；
- M3 作为异构候选召回/互补实验；
- P05 真实 OOF hard-negative；
- 所有二阶段实验的 cross-fit 和安全回退。

### 3.2 调整

原计划把 P03、P04、P05 视为相邻但相对独立的实验。现在应把它们收敛为
一个共享对象层：

```text
M1 OOF proposal
  → 可追溯 Pred-OOF crop
  → 轻量 ConvNeXt 对象学生
       ├─ foreground/background
       └─ 25 类细分类
  → 分数校准与困难门控
```

实验仍应分别消融“只重分类”“只拒背景”“联合”，但工程上共享数据合同、
crop 渲染、骨干和推理入口，避免做两套重复系统。

### 3.3 停止或暂缓

- P06-REAL：真实定位错误容量太小，暂缓；
- P06-DIFF：继续停止；
- P07/SD1.5 背景融合：已有充分否定证据；
- CleanDIFT 扩展：只保留正式 P04 对照，若仍弱于 DINO 即封板；
- 无错误依据的 M2、大输入或多模块网格：停止；
- 25 个独立阈值：样本不足且过拟合风险高，不进入当前正式方案。

## 4. 下一阶段唯一正确的执行顺序

## Phase N0：补齐 M1 决策证据，CPU 优先

这是现在的第一任务，不需要 GPU。

### N0-1：cross-fit 阈值基线

对每个 held-out fold：

1. 只用另外两个 fold 的 OOF 选择全局阈值；
2. 把阈值原样应用到当前 fold；
3. 合并三份 held-out 结果；
4. 报告 pooled 和逐折 Recall/FDR；
5. 记录三个训练侧阈值及其离散程度。

必须同时给出：

- 官方目标 FDR≤0.20；
- 内部目标 FDR≤0.17；
- 全局单阈值为正式主行；
- 三大类曲线只作诊断，不立即把三阈值写成最终系统；
- 25 类阈值禁止进入本轮。

完成定义：得到不在同一对象上“选阈值又回评”的正式 M1 baseline。

### N0-2：补齐定位/分类解耦和稳健性

原总纲要求但当前还没有完整交付：

- `R_loc@oracle-class`；
- `Acc_fine@localized`；
- 按 fold、source group、尺寸、边界、head/middle/tail 的结果；
- source-group cluster bootstrap 区间；
- TU-160 9-shot 压力折；
- 逐类分数与 FP/FN 集中度。

这一步用于验证“分类主导、定位次要”的结论是否在更严格定义下仍成立。

### N0-3：Pred-OOF 对象证据层

建立统一 manifest，而不是让 P03/P05 各自重新配对：

```text
proposal_uid
image_id / source_image_id / source_relative_path
fold / group_id
checkpoint_sha256
predicted category/coarse/score/bbox
official TP/FP status
FP_DUP / FP_CLS / FP_LOC / FP_BG
matched annotation_uid/category/iou
box size / edge risk / score bin
crop geometry / crop checksum
manual review status
```

需要生成三种视图：

1. `localization_oracle_positive`：不看预测细类，按 IoU 选择可定位 GT 的最佳
   proposal，用于测真实细类上限；
2. `deployable_positive`：正式候选链中实际保留的匹配对象；
3. `hard_negative_candidate`：`FP_BG` 和高风险未归因 proposal。

这一层是 X-CROP-03、X-BG-01、困难门控和最终对象学生的共同唯一输入。

### N0-4：FP_BG 人工语义审计

`FP_BG` 不是已经确认的纯背景。先对 3,303 个工作点候选做分层抽检：

- 三大类；
- 三个 fold；
- score 低/中/高分位；
- 主要预测细类，特别是 MS、FSC、QHS；
- 每批保留盲重复卡用于一致性。

人工标签至少区分：

```text
clear_background
plausible_unlabeled_or_ambiguous_target
poor_localization_of_known_target
duplicate_or_fragment_not_captured
invalid_crop_or_render
```

只有 `clear_background` 和满足严格几何条件的高可信未归因样本才能直接作为
背景训练样本。其他类型保留为审计或其他错误路线，不得强行标背景。

## Phase N1：P03/P04 正式 CV3 复验，GPU 快速完成

服务器恢复在线后，优先按已有冻结任务单执行：

1. P04-F：复用 cache，18 个 frozen-feature probe；
2. P03-F：tight-224、natural、seed 42，三折固定 30 epoch。

顺序上 P04 先行是因为 cache probe 成本低、能先验证旧 cache 是否仍可用；
P03 紧随其后。两项都不重开模型、分辨率、sampler、seed 或扩散层网格。

决策：

- P03-F 确认 GT crop 正式上限及 TU-160 压力；
- P04-F 决定 DINOv2-B 是否保留为教师；
- CleanDIFT 若仍明显弱于 DINO 且无稳定困难子集价值，扩散特征线正式结束；
- 两项都不是端到端收益，必须继续进入 Pred-OOF 验证。

## Phase N2：X-CROP-03 与 X-BG-01，共享对象学生

### N2-1：普通 Pred-OOF crop 强基线

先使用 P03 的 ConvNeXt-T，不加入蒸馏：

| 视图 | 回答的问题 |
|---|---|
| GT crop | 理想上限 |
| localization-oracle proposal crop | 定位存在时细类是否可修复 |
| deployable proposal crop | 实际候选上的端到端输入质量 |
| hard negative crop | 是否能稳定拒背景 |

每个 held-out fold 的对象模型只能用另外两个 fold 的 GT/OOF 派生样本训练。
不能用 held-out fold hard negatives 训练后再回评该 fold。

### N2-2：分开消融，再联合

固定顺序：

1. M1 + cross-fit threshold；
2. + 25 类重分类，不训练背景；
3. + 背景拒识，不改细类；
4. + 联合对象学生；
5. + 困难对象门控；
6. 只有 P04 正式支持时，加入 DINOv2 蒸馏。

每一行重新执行 cross-fit 阈值，不允许沿用对某一模块最有利的同 OOF 阈值。

### N2-3：准入门槛

对象模块进入正式主线至少满足：

- pooled 官方指标改善；
- 至少 2/3 folds 同方向；
- Overall Recall 不低于 0.88，FDR 明显优于 0.20 并朝 0.17 收敛；
- 舰船 Recall 不因拒识稳定下降；
- 对飞机 `FN_CLS` 有明确净恢复；
- 对舰船/车辆 FP 有明确净减少；
- 收益不只来自 TU-160 单一大组或一两个极少样本；
- 全候选离线收益成立后，困难门控能保留主要收益并降低时延。

## Phase N3：M3 正式 OOF，解决“没有候选”的问题

M3 不阻塞 N0/N1/N2，但仍值得正式完成，因为：

- M1 的 `FN_MISS=553` 无法由分类器修复；
- 车辆候选下限 Recall 只有 0.7985；
- 只有异构 OOF 才能判断是否存在稳定 `M3_only` TP。

M3 固定 RT-DETR-L/1024、120 epoch、三折，不改已有合同。完成后只按
paired/oracle-union 决策：

- 若 M3-only 总体增益稳定且对车辆/困难组显著，研究困难对象或困难区域门控；
- 若只提高 mAP、没有独立 TP，停止 M3；
- 若精度好但 10K 成本过高，转为训练教师/难例发现器，不进入全量推理；
- 不默认双模型全量集成。

## Phase N4：E 的 10K 全局对象工程，与模型实验并行

E 现在已有可用 M1 checkpoint lineage，不再等待最终模型才开始。首轮只形成
工程证据：

```text
10K 读入
→ 1280 tile / 256 overlap / stride 1024
→ M1 批推理
→ 全局坐标恢复
→ 类别无关对象聚合
→ COCO JSON
→ 分段 p50/p95/max
```

当前 OOF 的 `FP_DUP=187` 发生在普通图像内，不能代表 10K 跨 tile 重复。
因此全局融合优先级不能因当前 FP_DUP 较少而下降。

E 必须先证明：

- 坐标恢复正确；
- 同一跨 tile 对象唯一输出；
- 边界对象没有系统丢失；
- 候选数和最坏图时延可控；
- 4080 SUPER 结果只标工程基线；
- 最终模型冻结后在 3090 或认可等效设备复测 20 秒门槛。

## Phase N5：按证据开启车辆专项

车辆专项晚于 M3 paired 与 E tile 尺度结果，避免同时改模型和输入：

1. 判断 M3 是否补回车辆；
2. 判断 10K tile resize 是否进一步缩小车辆；
3. 若仍是短板，只选一个因素：
   - P2/stride-4 小目标头；或
   - 更高输入分辨率；或
   - 车辆条件专家。

每次只改变一个因素，并同时报告候选 Recall、FDR、时延和显存。对象背景头
可以减少车辆 FP，但不能替代候选生成改进。

## Phase N6：最终组合与冻结

只有单模块分别通过后才做：

| 行 | 组合 |
|---|---|
| B0 | M1 + 全局融合 |
| B1 | B0 + 对象细分类 |
| B2 | B0 + 背景拒识 |
| B3 | B0 + 联合对象学生 |
| B4 | B3 + 困难门控 |
| B5 | B4 + 条件 M3/车辆分支（若准入） |

每一行必须重新 cross-fit 阈值，并在相同 10K 工程合同下测增量时延。最终
系统不以模块数量取胜，只保留有独立净收益的最小组合。

## 5. 当前实验优先级

### CPU 队列

1. N0-1 cross-fit 全局阈值；
2. N0-2 定位/分类解耦和 group bootstrap；
3. N0-3 Pred-OOF 对象 manifest；
4. N0-4 FP_BG 盲审包；
5. M1/M3 paired 分析代码保持等待 M3。

### 单 GPU 队列

1. P04-F；
2. P03-F；
3. N2 普通 Pred-OOF 对象学生；
4. M3 三折长任务，或由 D 的独立 GPU 并行；
5. 通过准入后的 DINO 蒸馏；
6. 有证据后才做车辆单因素实验。

P06、扩散生成和完整 DiffusionDet 不进入当前队列。

### 工程并行队列

1. E 接 M1 checkpoint 跑通 10K；
2. 全局坐标聚合/NMS/WBF 强基线；
3. 对象学生接口和困难门控；
4. 最终模型后复测。

## 6. 五人任务重新对齐

| 成员 | 立即任务 | 不应承担 |
|---|---|---|
| A | N0 全部、P03/P04 正式复验、N2 共享对象学生、最终准入与文档 | 不再扩展 P06/SD1.5 |
| B | 冻结 split；协助 N0-4 FP_BG/漏标审计和来源组统计 | 不再修改正式 fold 归属 |
| C | 归档 M1 checkpoint/配置；支持 E 接入；仅在车辆门禁通过后做单因素 detector 实验 | 不重开 HPR/rare-rebalance 网格 |
| D | 完成 M3 正式三折 OOF和 paired 输入 | 不做 P05/P06 或大图工程 |
| E | 立即以 M1 跑 10K 工程闭环、融合与测速 | 不调训练参数或二阶段阈值 |

## 7. 未来 72 小时和一周里程碑

### 72 小时

- 完成 N0-1/N0-2 的 CPU 报告；
- 完成 Pred-OOF manifest schema、构建和自动审计；
- 生成 FP_BG 人工审查包；
- 服务器可用时完成 P04-F 和 P03-F；
- E 开始 M1 10K 工程 smoke；
- M3 恢复排队，但不阻塞上述工作。

### 一周

- 完成 FP_BG 人工审计；
- 得到普通 Pred-OOF ConvNeXt 的重分类/背景拒识强基线；
- 决定 DINOv2 是否进入蒸馏；
- 得到 M3 至少完整 OOF，随后做 paired/oracle；
- 得到第一版 10K 全局融合工程指标；
- 冻结是否需要车辆专项。

## 8. 最终方法故事的当前版本

当前证据最支持的方案不再是“扩散模型检测”，而是：

> 以快速 25 类检测器产生高召回候选，在全局坐标中形成唯一对象；依据正式
> OOF 暴露的错误类型，只对高风险对象执行轻量的联合细分类与背景拒识，并
> 对车辆等候选不足类别保留条件式异构检测或小目标路径。DINOv2 只作为训练
> 教师，正式推理保持轻量。

这条故事同时对应：

- 小样本：来源组隔离、TU-160 9-shot、细类共享表征；
- 评分：细类错同时造成 FP/FN，背景 FP 和重复框直接受罚；
- 大图：跨 tile 唯一化和完整对象重裁；
- 速度：重模型和对象头均条件启用；
- 创新：错误路由的全局对象层，而不是简单堆模块。

## 9. 当前禁止的错误做法

- 把同 OOF 阈值 `0.051` 当成最终无偏成绩；
- 因 Overall 过线而忽略舰船/车辆；
- 将全部 `FP_BG` 自动当背景训练；
- 在 held-out fold 的 OOF FP 上训练后回评同一 fold；
- 直接运行 DINOv2/CleanDIFT 于所有正式候选；
- 因 P03 GT crop 约 0.97 就宣称二阶段已经有效；
- 用 P06 修正 66 个定位错误，却推迟 1,115 个细类错误和 3,303 个未归因 FP；
- 未完成 10K 全局融合就用普通图 FP_DUP 判断重复问题不重要；
- 同时改变模型、输入分辨率和采样后归因于单一创新。

## 10. 证据与执行入口

- M1 正式结果：
  [`M1_CV3_OOF_FORMAL_RESULT_AND_RECOVERY_AUDIT_v2.md`](M1_CV3_OOF_FORMAL_RESULT_AND_RECOVERY_AUDIT_v2.md)
- 正式实验旧总纲：
  [`CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md`](CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md)
- P03/P04 正式重放：
  [`P03-P04-FORMAL-CV3-V2-REPLAY-PLAN.md`](P03-P04-FORMAL-CV3-V2-REPLAY-PLAN.md)
- M1/M3 paired：
  [`M1_M3_CV3_OOF_POSTPROCESS_ANALYSIS_PLAN_v1.md`](M1_M3_CV3_OOF_POSTPROCESS_ANALYSIS_PLAN_v1.md)
- 延期待办：
  [`DEFERRED_WORK_REGISTER.md`](DEFERRED_WORK_REGISTER.md)
- 服务器资产：
  [`SERVER_ARTIFACT_REGISTER.csv`](SERVER_ARTIFACT_REGISTER.csv)
- P03 服务器任务：
  `docs/server/P03_FORMAL_CV3_V2_REPLAY.md`
- P04 服务器任务：
  `docs/server/P04_FORMAL_CV3_V2_REPLAY.md`
- M3 服务器任务：
  `docs/server/M3_CV3_OOF_TASK.md`
- 10K 任务：
  `docs/server/E_10K_PIPELINE_TASK.md`

