# DCR²-YOLO → HERA-YOLO 完整实验总结(2026-08-20)

> 本文档完整记录 2026-08-20 全天实验: DCR²-YOLO 收尾(E0-E12) 与 HERA-YOLO 方法级重构(方案4)。
> 主线结论: 主提交 = **Y5 + OER(14特征) + 改类 + NMS = Recall@FDR=0.12 = 0.9620**。

---

## 目录

1. [背景与目标](#一背景与目标)
2. [DCR²-YOLO 收尾(E0-E12)](#二dcr-yolo-收尾e0-e12)
3. [HERA-YOLO 方法级重构](#三hera-yolo-方法级重构)
4. [批次 A: replay/缓存级实验](#四批次-a-replay缓存级实验)
5. [批次 B: DFD 密集前景监督](#五批次-b-dfd-密集前景监督)
6. [批次 C: OER-v1 演进](#六批次-c-oer-v1-演进)
7. [全面深层剖析](#七全面深层剖析)
8. [完整里程碑数字链](#八完整里程碑数字链)
9. [证伪方向汇总](#九证伪方向汇总)
10. [完整产物清单](#十完整产物清单)

---

## 一、背景与目标

**比赛**: 小样本遥感卫星图像陆上目标检测(不均衡小样本学习)。
**官方口径**: 25 细类(20 飞机 + 4 舰船 + 1 车辆), IoU 阈值 ship/aircraft=0.50 / vehicle=0.35, 三大类 macro Recall 排名。
**官方门槛**: Recall≥0.85 / FDR≤0.20 / 时延≤20s(10K×10K)。

**方案演进**:
- 方案1/2/3 → DCR²-YOLO(候选-识别-风险三解耦);
- 方案4 → **HERA-YOLO**(方法级重构): 把候选/细类/风险/去重从"串行补丁"升级为"对象级联合裁决"。

**HERA-YOLO 三阶段**:
1. **看见对象**: 类别无关候选证据;
2. **看清对象**: 可观测性感知的层次细类识别;
3. **确定对象**: 对象图上的证据聚合、去重、最终提交。

**核心主张**(方案4): 遥感细粒度检测的预测不是一个框 + 一个 25 类分数, 而是由"对象存在性、官方定位质量、细类可观测性、对象唯一性"共同构成的分层证据; 训练期用异构教师各教一个因子, 推理期收缩为轻量 YOLO + 对象解析器。

---

## 二、DCR²-YOLO 收尾(E0-E12)

> 方案4 之前按方案2/3 完成的全实验队列, 共 12 个实验。此处只记录结论, 详细见各实验报告。

| # | 实验 | 结论 | 状态 |
|---|---|---|---|
| E0 | 协议审计 | IoU=0.35 本就正确; N2 旁路统计层修复; M3 对象级 UID | ✅ |
| E1 | Y5+R1 重排 | R3 融合三折一致选中, Macro +1.86pp | ✅ |
| E2 | post-rerank NMS | 零 TP 损失, FP −6,597 | ✅ |
| E3 | Soft-Risk | v0 保 Recall / v1 压 FDR, 互补 | ✅ |
| E4 | FGR 微调 | 边际(M1 域 gap) | ⚠️ |
| E5 | pair experts | 规则版净潜力 +2500 但端到端~0 | ⚠️ |
| E6 | 重点类原型 | 无增量(R3 已内化) | ❌ |
| E7 | 困难课程 | +0.1pp 边际 | ⚠️ |
| E8 | COPH 存在性正则 | 候选 +44~63%, 但固定风险前沿不升 | ⚠️ |
| E9 | P2-lite | 停止(Y2 快筛 + SAHI 双证据) | 🛑 |
| E10 | SparseZoom | 停止(2× 放大仅救 6/82) | 🛑 |
| E11 | Balanced 组合 | COPH+R3+NMS+SoftRisk | ⚠️ |
| E12 | 验证体系 | ledger/sentinel/漏斗 L0-L3 | ✅ |

**DCR² 冻结版本**:
- Safe = Y5 + R3 + aircraft NMS + SoftRisk;
- Balanced = COPH + R3 + 全类 NMS + SoftRisk;
- Attack = Balanced + 激进 SoftRisk。

**关键教训**(为方案4 铺垫): COPH 只证"能抬候选"未证"提前沿"; E5 规则改类 broken 267-393; 小目标(P2/SAHI)救不回硬盲区 → 催生 HERA-YOLO 的对象级重构。

---

## 三、HERA-YOLO 方法级重构

### 3.1 核心架构

```
输入图像 / 大图切片
   │
   ▼
Y5 Backbone + Neck
   │
   ├── 候选存在性 / 定位质量 / 粗类证据
   ├── YOLO 细类分布
   └── ROI 与场景特征
   │
   ▼
可观测性感知的层次识别器
   │
   ▼
对象证据图(候选框/OTM/OTO/crop/场景/几何/风险)
   │
   ▼
对象级联合解析 OER
   ├── 是否真实对象(node_validity)
   ├── 哪些候选同一对象(edge_same)
   ├── 最终细类 / canonical box / 提交分数
```

### 3.2 主指标: 固定风险前沿(方案4 §12)

不再报固定阈值, 改报:
- Recall@FDR=0.10/0.12/0.15
- FDR@Recall=0.94/0.95/0.96
- Macro Recall / 重点类 Recall / candidate-floor Recall / FP 分解 / 时延

---

## 四、批次 A: replay/缓存级实验

> 方案4 第一优先级, 全部基于已有候选/crop logits/OOF 标签, 不训练 detector。

### A0: 完整 PR frontier 重算

**目的**: 判断 COPH 是否真正提升同 FDR 前沿。

| 前沿指标 | Y5 三折链 | COPH 三折链 | Δ |
|---|---:|---:|---:|
| Recall@FDR=0.12 | **0.9357** | 0.9295 | −0.62pp |
| Recall@FDR=0.15 | **0.9452** | 0.9391 | −0.61pp |
| candidate-floor | 0.9715 | 0.9751 | +0.36pp |

**结论**: COPH 在固定风险前沿全面劣于 Y5(只抬 candidate-floor +0.36pp, 前沿 −0.62pp)。取消 COPH 主提交地位, 改称 Candidate-Heavy 分支。**Safe(Y5) = 0.9357 确立为 HERA-Core 靶点**。

### A1: OOF 对象图数据

- 节点 = 65,301 个候选(Y5 全量), 特征 = Y5 score + crop 25类 logits(top1/margin/entropy/agree) + 几何 + 局部密度;
- 边 = 234,127 条, 标签: same_object 34,550 / different_object 18,133 / background 181,407。

### A2: OER-v0 对象证据解析器(表格模型)

**目的**: 联合学习 node_validity + edge_same + fine + unique, 替代串行 SoftRisk+E5+NMS。

| 分数 | Recall@FDR=0.12 | FDR@Recall=0.94 |
|---|---:|---:|
| Y5 原始 score | 0.8578 | 0.4410 |
| SoftRisk v0(逻辑回归) | 0.8631 | 0.4048 |
| **OER node_validity(HistGB)** | **0.9036** | **0.2381** |

**关键成果**: OER 排序 + 全类 NMS = **Recall@FDR=0.12 = 0.9415**, 单模型超 Safe 三模块串行链(0.9357) **+0.58pp**。印证方案4"对象级联合裁决优于串行补丁"。

### A3: 可观测性路由器(替代 E5 规则改类)

**机制**: 只改"yolo 错细类"的候选(本来就不是 TP), 改类天然 broken=0——破解 E5 的 broken 问题。

| 方案 | corrected | broken | R@FDR=.12 |
|---|---:|---:|---:|
| 基线 OER+NMS(不改类) | — | — | 0.9415 |
| 全改(规则 E5) | 8,227 | **0** | **0.9584** |

- 路由器信号 AUC 0.9459(严格 fold cross-fit 更正为 0.8649, 不影响全改结论);
- sentinel(555 冻结图)泛化: 0.9698 → 0.9840(+1.42pp, 非 overfit)。

### A4: 基础模型 proposal probe —— 网络阻塞

服务器外网受限(HF/GitHub 不可达), DINOv2/RemoteCLIP 无法下载, 降级 ConvNeXt-T 维持。

### A5: OTM/OTO 诊断 → has_oto 特征

**目的**: 判断 one-to-one 是否提供 precision/uniqueness 证据。

| 类型 | OTO 支持率(score>0.5) |
|---|---:|
| TP | 0.892 |
| FP | **0.094** |

**关键**: OTO 支持必须带 score>0.5 门槛才有判别力(score>0 时 99.8% 候选都有支持, 无判别)。

| 组合 | R@FDR=.12 |
|---|---:|
| OER + 改类(基础) | 0.9583 |
| **OER + has_oto + 改类** | **0.9607** |

**+0.24pp, 突破 Attack 目标 0.96**。sentinel 泛化 0.9847。

### A6: 低保真相关性 —— 暂缓

历史只存 last.pt 无中间 epoch checkpoint, 需重训; 且服务对象(B 系列)前景不明。

---

## 五、批次 B: DFD 密集前景监督

### 5.1 诊断驱动

细类已饱和(aircraft 98.2% 正确), 剩余瓶颈是**候选缺失**:
- vehicle(FSC) 17.9% / ship 4.3% / aircraft 0.6% 的 GT 完全无候选。

### 5.2 实现(E9)

**机制**: COPH 只对 TAL 正样本 anchor 做存在性监督, 漏检目标(TAL 分配不到)无梯度; DFD 独立于 TAL, 用 GT box 生成自适应高斯中心热力图密集监督所有 anchor。结构零改动(复用 max_c logit)。

### 5.3 fold0 诊断

| 模型 | 候选 | cand-floor | vehicle NO_CAND | ship NO_CAND |
|---|---:|---:|---:|---:|
| Y5 | 24,742 | 0.9693 | 17 | 29 |
| **DFD** | 54,709 | **0.9804(+1.11pp)** | **3** | **17** |

### 5.4 三折完整链验证(决定性)

| 链(全量 OOF) | R@FDR=.12 |
|---|---:|
| Y5 三折 OER + has_oto | **0.9607** |
| DFD 三折 OER + has_oto | 0.9603(**−0.04pp**) |

**结论**: candidate-floor +1.11pp 不转化为前沿增益(候选 +121% 带来 FP 膨胀)。**DFD 不纳入主提交**。

---

## 六、批次 C: OER-v1 演进

### 6.1 edge 聚类去重(学习式替代 NMS)—— 负结果

- edge_same 分类器 AUC 0.9908(same/different 高度可分);
- 但 union-find 连通分量聚类去重 = 0.9363 vs NMS 0.9603(**−2.4pp**);
- 根因: 传递性链式过度合并 + FP_DUP 仅 0.4% 非瓶颈。

### 6.2 剩余错误诊断(定位方向)

全量候选错误分布: FP_BG 68.6% / FP_CLS 17.2% / FP_DUP 0.4%。工作点剩余: FP_BG 69% / FP_CLS 31% / FP_DUP 0。

### 6.3 D4 旋转一致性 —— 突破

**假说**: 真阳旋转后重现, 背景误检旋转后消失。

判别力: TP 支持率 0.998, FP_BG 有 15% 完全不重现(d4=0, TP 仅 0.2%)。

| OER 特征 | R@FDR=.12 | R@FDR=.11 |
|---|---:|---:|
| 基础(12特征) | 0.9579 | 0.9552 |
| +d4 | 0.9605 | 0.9581 |
| +has_oto | 0.9608 | 0.9587 |
| **+d4+has_oto** | **0.9620** | **0.9601** |

**主提交升级 0.9607 → 0.9620(+0.42pp)**。sentinel 泛化 +0.11pp(非 overfit)。R@FDR=.11 也破 0.96。

---

## 七、全面深层剖析

### 7.1 剩余误差三维分解

工作点 TP=20138, FP=2740(FDR=0.1198):

| 类型 | 数量 | 占比 |
|---|---:|---:|
| FP_BG(背景误检) | 1899 | 69% |
| FP_CLS(细类错误) | 841 | 31% |
| FP_DUP(重复框) | 0 | 0% |

candidate-floor 0.9774 vs frontier 0.9620 = **322 GT 排序损失**。

### 7.2 三条深层根因链

**根因 A: MS/QHS 系统性视觉模糊**(唯一贯穿性病根)

MS(导弹)/QHS(小舰船)贯穿所有错误象限:

| 错误象限 | MS | QHS |
|---|---:|---:|
| 排序损失 | 83(第1) | 50(第2) |
| 低估真阳(y5<0.1) | 366(第1) | 222(第2) |
| 自信误检(y5>0.5) | 396(第1) | 204(第2) |
| crop 幻觉 | 23.4%(第1) | 15.4%(第3) |
| 细类混淆 | MS↔QHS 288 | — |

检测器对 MS/QHS 双向犯错(既低估真阳又自信误检), 根因是"单框外观与背景结构(跑道划线/建筑/车辆)同形", 可区分信息在上下文而非单框。

**根因 B: crop 教师域偏差**

72% 背景候选被 crop 教师(ConvNeXt-T)幻觉成 MS/FSC/QHS/SU-24 4 个简单类。M1 域训练 → Y5 proposal 域部署, 陌生背景 patch 退化为"输出训练里最常见的小目标类"。直接封死"用 crop 证据压 FP_BG"。

**根因 C: 证据链缺失上下文**

86% 的 FP_BG 是 d4=3(4 方向全重现)——它们是结构化背景(机场跑道标记/规则纹理), 非随机噪声, D4 对它们无判别力。所有现有证据(Y5/crop/OTO/D4)都是"单框/单目标"证据。

### 7.3 其他深层发现

- FP_BG 画像: y5_score 中位仅 0.071、local_density 25(真目标密集区)、近半幻觉成 MS;
- FP_CLS 混淆结构化: 轰炸机(TU-22↔TU-160 347)/舰船(MS↔QHS 288)/苏霍伊(SU-35↔SU-34/24)/预警加油(E-8↔KC-135)/F系列(F-15/16/22) 五个家族;
- 类不平衡: HM 17 vs FA-18 2147(126倍), 尾类 FSC 402/TU-160 361/E-8 432;
- 数据域: 255 个 source_group(255 个不同机场), 域多样性极大。

---

## 八、完整里程碑数字链

### 8.1 Recall@FDR=0.12 主链

```
Safe(Y5+R3+NMS+SoftRisk)        0.9357
  → OER node_validity + NMS     0.9415  (A2, +0.58pp)
  → + 改类(可观测性路由)        0.9584  (A3, +1.69pp)
  → + has_oto(OTO 强支持)       0.9607  (A5, +0.24pp)
  → + d4(旋转一致性)            0.9620  (E10, +0.42pp vs 基础)
```

### 8.2 最终主提交指标

| 指标 | 值 |
|---|---|
| Recall@FDR=0.12 | **0.9620** |
| Recall@FDR=0.11 | **0.9601** |
| Recall@FDR=0.10 | 0.9565 |
| candidate-floor(改类后) | 0.9774 |
| sentinel(555 冻结图) | 0.9847 |

### 8.3 与门槛关系

- 官方门槛(0.85/0.20): 大幅超额(FDR=0.20 时 Recall≈0.97);
- 团队硬门禁(0.93/0.11): 超额(Recall@FDR=0.11 = 0.9601);
- Attack 目标(0.96/0.12): 已超 0.9620。

---

## 九、证伪方向汇总(重要)

| 方向 | 结果 | 证据 |
|---|---|---|
| 扩候选(COPH 存在性正则) | 前沿 −0.62pp | A0 |
| 扩候选(DFD 密集前景监督) | floor +1.11pp 但前沿 −0.04pp | E9 |
| 去重(edge 聚类) | −2.4pp(FP_DUP 仅 0.4%) | E10 |
| 小目标(P2/SAHI/2×放大) | 救 6/82 | E10 |
| 基础模型教师(DINOv2) | 网络受限 | A4 |
| 视觉质量特征(对比度/模糊) | +0.005 AUC 边际 | A3-full |
| 困难课程 | +0.1pp 边际 | E7 |
| 规则式 pair expert 改类 | broken 267-393 | E5 |

**核心教训**: 瓶颈从来不是候选缺失或去重, 而是"对象级联合裁决的排序质量 + 背景/细类的判别证据"。

---

## 十、完整产物清单

### 10.1 新增脚本(scripts/)

| 脚本 | 作用 |
|---|---|
| a0_pr_frontier.py | 固定风险前沿计算 |
| a1_build_object_graph.py | OOF 对象图构建 |
| a2_oer_v0.py / a2_oer_edge.py | OER-v0 表格模型 + edge 去重 |
| a3_observability_router.py / a3_router_e2e.py / a3_sentinel_check.py | 可观测性路由器 |
| a5_oto_oer.py | has_oto + OER 完整链 |
| e9_dfd_diag.py / gen_dfd_manifest.py | DFD 诊断 + manifest |
| oer_v1_edge.py / oer_v1_d4.py | OER-v1 edge/d4 验证 |
| src/rsdet/innovation/dfd_presence.py | DFD 密集前景监督 loss |
| train_cv3_oof.py(改) | 加 --innovation dfd |

### 10.2 报告(reports/experiments/)

- HERA_FULL_ANALYSIS_20260820.md(深层根因剖析)
- HERA_A_BATCH_20260820.md(批次 A 总结)
- A0_PR_FRONTIER / A2_OER_V0 / A3_OBSERVABILITY_ROUTER / A5_OTO_DIAG / E9_DFD / E10_OERV1(各实验)
- DCR2_YOLO_IMPLEMENTATION_20260820.md(DCR² 总报告)

### 10.3 权重(服务器 /workspace/results/)

- Y5-ROT90-CV3-OOF/fold{0,1,2}(Y5 基线, 主提交检测器)
- E8-COPH-FOLD{0,1,2}-40EP(COPH)
- E9-DFD-FOLD{0,1,2}-40EP(DFD)
- P03-FORMAL-CV3-V2(ConvNeXt-T crop 教师)

### 10.4 本地持久化资产(outputs/)

- E9-DFD-FOLD0/(DFD 三折候选 150,204 + crop3 logits + 对象图)
- Y5-OER-RESTORE/(Y5 对象图 + d4 + OTO + E1 logits)

### 10.5 Commit(本轮 35+ 个)

核心里程碑 commit:
- `5eb5149` A0 frontier → `2cb9c98` A1+A2 → `08b62eb` A3 → `d0fc32b` A5
- `cf46d93` DFD 实现 → `470d261` DFD 三折
- `a582e62` edge 负结果 → `505917e` D4 突破
- `9bf8636` / `efd1473` 全面剖析

---

## 附: 关键数字速查表

| 数字 | 值 | 来源 |
|---|---|---|
| 主提交 frontier | 0.9620 | E10 |
| 工作点 TP/FP | 20138/2740 | 剖析 |
| FP 分解 | BG 1899 / CLS 841 / DUP 0 | 剖析 |
| 排序损失 | 322 GT | 剖析 |
| candidate-floor | 0.9774 | 剖析 |
| sentinel | 0.9847 | A5/E10 |
| OER 特征数 | 14(12基础+d4+has_oto) | E10 |
| 最大混淆 | TU-22→TU-160 347 | 剖析 |
| 最小类 | HM 17 GT | 剖析 |
| source_group | 255 | 剖析 |
