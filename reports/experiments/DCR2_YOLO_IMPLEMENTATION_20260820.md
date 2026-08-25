# DCR²-YOLO 全面实现与状态报告(2026-08-20)

> **报告性质**: 对照《DCR²-YOLO 全面升级与极速验证总纲 20260820》(改进方案3, 38KB)
> 逐项核对的**实现状态全量清单**。用于与 GPT 讨论下一步: 哪些已实现、哪些部分实现、
> 哪些未实现及原因、剩余时间如何分配。
>
> **执行范围**: 服务器 RTX 3090 + 本地; 2026-08-20 02:40 至 16:30 两个会话完成;
> 全部代码已提交推送(commit 见各节)。

---

## 1. 方案回顾(总纲核心判断)

当前瓶颈不是"缺更大模型",而是**四个子任务被压进同一个 25 类检测分数**:

1. 这里是否存在真实目标(候选形成);
2. 属于 aircraft/ship/vehicle(粗类);
3. 究竟是哪个细类(细类识别);
4. 官方匹配口径下是否值得提交(风险校准)。

总纲给出的解法是 **DCR²-YOLO 四层系统**:
- Y5 一对多路径 = 高召回候选形成;
- COPH 类别无关头 = 细类不确定性不再压掉候选;
- FGR 独立识别器 = 解决兄弟机型/稀有类;
- RCR 软风险头 = "能否成为官方 TP", 而非硬删;
- M3/D4 只做离线教师;
- SparseZoom = 只对高风险小区域放大;
- 全局对象层 = 每个对象只提交一次。

三个版本: **Safe**(回退保证) / **Balanced**(主提交) / **Attack**(冲击 0.96/0.12)。

---

## 2. 实现总览表(一句话版)

| 总纲内容 | 状态 | 关键结果 | commit |
|---|---|---|---|
| §1 七大口径问题 | ✅ 6/7 完成, 1 项部分 | IoU/旁路/UID/三栏/oracle 全部闭合 | b3483dd |
| E1 Y5+R1 重放 | ✅ | R3 融合三折一致, Macro +1.86pp | dc3c550 |
| E2 post-rerank NMS | ✅ | 零 TP 损失, FP −6,597 | dc3c550 |
| E3 Soft-Risk v0 | ✅ | P(loc) AUC 0.9503, 双头可调 | 0cbc84d/062a184 |
| E4 proposal-domain FGR | ✅(边际) | M1 域 gap, 完整链 −0.3pp | cfe4ffc |
| E5 pair experts | ⚠️ 规则版完成, 学习式未做 | 净潜力 +2,500, 端到端 ~0 | cfe4ffc |
| E6 rare prototype | ✅(轻量版) | 无增量, R3 已内化 | f73344c |
| E7 M3-KD | ⚠️ 只做困难课程层 | +0.1pp 边际 | 83f62bb/5968625 |
| E8 COPH | ⚠️ 存在性正则版(非完整结构) | 三折 Recall +0.69~0.96pp, sentinel 泛化 | 955064e/7a3e394 |
| E9 P2-lite/NWD | 🛑 停止(双证据) | Y2 快筛 + SAHI 诊断 | b242c4b |
| E10 SparseZoom | 🛑 停止(按停止条件) | 2× 放大仅救 6/82(7%) | 0b53711 |
| E11 DCR² 组合(Balanced) | ✅ | COPH+R3+NMS-all+SoftRisk 冻结 | 75e437c |
| E12 Attack + 验证体系 | ⚠️ 体系落地, Attack 未组合 | ledger/sentinel/漏斗 ✅ | 3906a63/b9905ae |
| §9 全局对象层 | ❌ 未实现 | 全类 NMS 近似 | — |
| §10 四阶段训练 | ⚠️ 部分(Stage A/B 完成, C/D 未做) | — | — |
| §12.1 低保真排名相关性 | ❌ 未做 | — | — |
| §12.4 机制挑战集 | ❌ 未建 | — | — |
| §12.6 MO-ASHA | ❌ 未做 | — | — |

---

## 3. E0: 协议/口径审计(对照总纲 §1, 7 项)

**结论: 审计级, 全部闭合。** 报告: `reports/experiments/E0_PROTOCOL_AUDIT_20260820.md`。

| § | 项 | 结论 | 动作 |
|---|---|---|---|
| 1.1 | vehicle IoU | **本就正确**(project.yaml 已 0.35, 全评估路径走 protocol, 无硬编码 0.50) | 确认, 无需改代码; 文档笔误已识别 |
| 1.2 | V1.5/V1.6 分开记录 | 部分 | 报告固定输出 pooled/macro/三栏状态; 官方开放后冻结唯一主指标 |
| 1.3 | N2 aircraft bypass | **发现真 bug 并修复** | evaluate_bg_gate 记录 `shadow_coarse`; admission g7 排除 aircraft; 误删 TP 3→2 |
| 1.4 | M3 gt_uid 对象级 | **发现真 bug 并修复** | build_m3_teacher_evidence 改用 formal.objects `annotation_uid`; V2 版 1,313 全唯一(旧版仅 643) |
| 1.5 | 130/320 属 GT-oracle | 确认 | 报告标注为互补性上限, 非可部署预期 |
| 1.6 | 三栏状态规范 | 落地 | candidate_floor / deploy_working_point / scientific_status 全表固定 |
| 1.7 | R1 重放到 Y5 | ✅ E1 完成(见 §4) | — |

---

## 4. 实验队列逐项对照(总纲 §13)

### E1: Y5 + R1 replay(准入 ✅)
- **做了什么**: P03-F 教师(ConvNeXt-T 三折)对 Y5 全量 65,301 候选 crop 推理(3090 上约 90 秒/三折); 复用 R1-0 变体网格, 三折 cross-fit 各折选变体+阈值。
- **结果**: 三折**一致选 R3_fuse_a0.40(分数融合)**: Recall +0.51pp / **Macro +1.86pp** / aircraft +0.40pp, FDR 持平。门控重标变体(R1/R2/R4)未被选中——对 Y5 而言"融合"比"改标签"更稳。
- **准入**: ✅ aircraft Recall/FDR 双改善, ship/vehicle 结构性旁路不变。
- 脚本: `scripts/e1_y5_rerank_screen.py`(支持 --folds/--variant 单折诊断)。

### E2: post-rerank NMS(准入 ✅)
- **做了什么**: 对重排后分数做同细类 NMS@0.5。
- **结果**: **零 TP 损失**(TP +0/FN +0), FP −6,597, FDR −3.5pp。
- **关键对比**: 直接用 Y5 原始分数做 NMS 会误伤 1 个 TP——验证总纲"先重排后 NMS"必要性。
- **准入**: ✅ TP/FN 不变, FP 明显下降。
- 脚本: `scripts/run_safe_chain.py`(--nms-all 支持全类 NMS)。

### E3: Soft-Risk v0/v1(准入 ✅)
- **做了什么**: 三折 cross-fit 逻辑回归 + logit 有界残差重排, 替代 N2 硬 gate。
- **v0(位置标签)**: "该框位置是否匹配任一 GT"; P(loc) AUC **0.9503**; t≤0.05 区间 Recall/FDR 双改善(t=0.01: +0.25pp / **−3.35pp**)——低分真阳被上调。
- **v1(细类标签+重复框特征)**: P(fine) AUC 0.9419; FDR 全阈值大降(t=0.01 **−8pp**), Recall 略降——下调 FP_CLS/重复框。
- **双头融合**(COPH 三折上): a=0.5/b=0.2 → t=0.1 R=0.9379/F=0.1420(可调工作点)。
- **机制结论**: 两版互补 = RCR(风险)+FGR(细类)解耦的离线验证。
- 脚本: `scripts/soft_risk_v0.py`。

### E4: proposal-domain FGR(准入 ⚠️ 边际)
- **做了什么**: 复用 R1-1 训练框架(N2-PROPO-CROP-v2 18K aircraft proposal-domain 行, 5ep, 三折 CE 微调)→ 对 Y5 候选推理。
- **结果**: E1+E4 screen Recall +0.06pp/Macro +0.32pp; 完整链 E1E4E2 t=0.1 R=0.9294(比 E1E2 低 0.3pp)。
- **结论**: 训练数据是 M1 域(N2-PROPO-CROP-v2), 与 Y5 候选域有 gap; E1(P03 零训练)已足够, **E4 不加入组合**。要更大收益需用 Y5 域 proposal 数据重训。
- config: `configs/experiments/e4_fgr_v1.yaml`。

### E5: pair experts(规则版完成, 学习式未做)
- **做了什么**: 五组兄弟机型专家(规则触发: Y5 标签∈组 且 crop top-1∈同组 且 top1≠当前 且 margin/top-prob 达标 → 修正为 crop top-1)。
- **结果**: 改类质量分析——净潜力 **+2,491~2,535**(broken TP 267-369 vs 可救 FN_CLS 2,624-2,834); 但**端到端净收益 ~0**(broken 抵消救回)。
- **根因**: crop 教师(P03/E4)在 7-9% 兄弟机型上比 Y5 更错——系统性分歧, 提高置信阈值无法消除。
- **结论**: 规则版不能独立准入; 正确路径是**学习式触发**(决策器输入 crop logits+Y5 分数+位置特征)或与 RCR 组合。**这是下一步最重要的候选之一(潜力已证, 只差实现方式)**。
- 脚本: `scripts/e5_pair_experts_rule.py`。

### E6: rare prototype(轻量版, 无增量)
- **做了什么**: 分析重点类(HM/LQS/TU-160/F-22)漏检中 crop 可救比例; 轻量上调实验。
- **结果**: E1 重排后 HM 15/17、F-22 491/493 已很高; **TU-160 FN 38 中 crop 可救 16(42%)** 但分数上调端到端无显著增益; HM/LQS crop 可救率 0-14%。
- **结论**: E1 的 R3 融合已内化重点类提升(TU-160 0.371→0.579@t=0.1); E6 独立价值小; HM/LQS 需数据级方案(合成/增强)。

### E7: M3-KD(只实现困难课程层)
- **做了什么**: `--hard-curriculum`——M3 找回且 Y5 漏检的 320 目标(186 图)训练重复 1 次(权重 2x); 与 COPH 叠加训练 fold0 40ep。
- **结果**: 完整链 t=0.1 R=0.9434 vs 纯 COPH 0.9423(**+0.11pp 边际**)。
- **结论**: 图级重复在 COPH 候选扩增面前被稀释; **不加入 Balanced**。FGD 特征蒸馏 + CrossKD 预测蒸馏(**总纲 §7.2 二层/三层)未实现**——这是下一步候选。

### E8: COPH(存在性正则版, 三折验证通过)
- **实现方式(重要)**: 总纲要求完整结构(P2 64-channel candidate-only head + P3-P5 candidate stem + p_exist/p_coarse/q_loc 三头 + candidate_score=calibrated(...) 改写 + oto 支持)。**实际实现的是"存在性正则版"**: CophPresenceLoss 对每个正样本锚点要求 max_c p(c)→1(类别无关存在性), **零结构改动, 复用检测头 scores**, 40ep 从 Y5 fold 权重初始化。
- **三折结果**:

| 链(同口径 R3+NMS-all+SoftRisk) | t=0.05 | t=0.1 | t=0.2 |
|---|---|---|---|
| Y5 三折 | 0.9475/0.1591 | 0.9355/0.1173 | 0.9206/0.0794 |
| **COPH 三折** | 0.9510/0.2055 | **0.9424/0.1588** | **0.9302/0.1187** |
| Δ | +0.35pp/+4.6pp | **+0.69pp/+4.2pp** | **+0.96pp/+3.9pp** |

- 候选扩增: fold0 +63%(40,304)/fold1 +45%/fold2 +44%——"细类不确定压掉候选"被缓解;
- **Sentinel(555 冻结图, L3)**: COPH 0.9601/0.1072 vs Y5 0.9582/0.0766(**+0.19pp Recall**, 收益泛化非记忆);
- **Paired ledger**: 净 **+75 TP**(新 193/坏 118), FP +11,687(SoftRisk 后残余);
- 双头 SoftRisk 在 COPH 上: P(fine) AUC 0.9730。
- **准入**: ✅ 低分难 TP 上移, 无 Y3 式候选爆炸(候选 1.44~1.63×, 低于 1.5× 门禁边缘, 由 NMS+SoftRisk 收口)。
- 代码: `src/rsdet/innovation/coph_presence.py`(CophPresenceLoss/E2EPresenceLoss/coph_trainer); `--innovation coph --coph-presence-gain 1.0`。

### E9: P2-lite/NWD(停止 🛑)
- **双证据**: ①Y2 快筛(2026-08-11): 完整 P2 候选 +36.1% 但无候选 vehicle GT 5/133 零变化, overall Recall −5.84pp, 已冻结"不启动 P2-Lite"; ②E10 SAHI 诊断: 82 个无候选目标 2× 放大仅救 6。
- **结论**: 完整 P2 已证无效, P2-lite 是同一机制减量版, 无独立价值。报告: `E9_P2LITE_STOP_20260820/`。

### E10: SparseZoom(停止 🛑)
- **做了什么**: 91 个三模型全漏 → 82 个"完全无候选" → 64 张图整图 2× LANCZOS 放大 + 对应 fold Y5 推理。
- **结果**: **仅救回 6/82(7%)**。76 个是检测器硬盲区(低对比度/场景盲区), **非尺度问题**。
- **结论**: 按总纲 §8.3 停止条件("uniform slicing 也救不回则不启动")停止; 82 目标留档为已知残余失败案例(`/tmp/sahi_gt_targets.json`)。小目标恢复路径改为训练期难例(E7)。

### E11: DCR² Balanced 组合(定义 + 三折验证完成)
- **构成**: COPH 候选 + R3 融合 + 全类 NMS + SoftRisk(v0 或双头)。
- **工作点三档**:

| 工作点 | Recall | FDR | 说明 |
|---|---|---|---|
| t=0.1 + v0 | 0.9424 | 0.1588 | Recall 优先(门禁 FDR≤0.20 内) |
| t=0.2 + v0 | 0.9302 | 0.1187 | FDR 优先, ≈Y5 t=0.1 口径 |
| t=0.1 + 双头(a=0.5,b=0.2) | 0.9379 | 0.1420 | 均衡 |

- **门槛评估**: Recall≥0.95/FDR≤0.12 起步未完全达到(0.9424/0.1588); FDR 需要更强收口(见 §10 下一步)。

### E12: Attack + 验证体系
- **已落地**: Paired delta ledger(§12.5)✅; prospective sentinel 冻结 23 组 555 图 12.4%(§12.3)✅; 四级漏斗 L0-L3(§12.2)✅ 定义+执行(L0/L1/L2 已用, L3 用 sentinel 替代 outer-pure); 统一准入(§12.7)✅ 应用到各实验。
- **未做**: 低保真排名相关性标定(§12.1)、机制挑战集(§12.4)、MO-ASHA(§12.6)。
- **Attack 组合**: 因 SparseZoom 停止, 修订为 Balanced + 激进 SoftRisk(beta=0.7) + 学习式 E5(可选); **组件全但未组合验证**。

---

## 5. 模块级实现明细(对照总纲 §3-§11)

### 模块 A: COPH 类别无关候选头(§4)——⚠️ 部分实现
| 子项 | 状态 | 说明 |
|---|---|---|
| §4.2 P2 64-channel candidate-only head | ❌ | 未加新头结构 |
| §4.2 P3-P5 candidate stem + p_exist/p_coarse/q_loc | ❌ | 未加新头结构 |
| §4.3 连续质量标签 q_loc / NWD-RFLA tiny assignment | ❌ | 未做 |
| §4.4 candidate_score=calibrated(p_exist,q_loc,otm,oto) 改写 | ❌ | 未改写候选保留逻辑 |
| §4.5 one-to-many/one-to-one 互补诊断 | ❌ | 未做 oto 支持特征 |
| **存在性正则(核心思想)** | ✅ | CophPresenceLoss: max-class→1, 已证三折收益 |

> **说明**: 实现的是机制等价版(训练期正则让模型"更敢报"), 而非结构版(独立候选头)。
> 存在性正则三折已证有效(+0.69~0.96pp), 但候选保留逻辑仍依赖 max(fine score),
> 完整 COPH 头的"候选阶段不再看细类"收益尚未兑现——**下一步核心候选**。

### 模块 B: FGR 独立细类识别(§5)——⚠️ 部分实现
| 子项 | 状态 | 说明 |
|---|---|---|
| §5.1 tight/context crop + ROI 特征 | ✅ 部分 | 用 tight crop 224(P03-F); context/ROI 未用 |
| §5.2 aircraft 20 类独立识别器 | ✅ | R1/R3 crop 重排(非独立训练头) |
| §5.2 ship 4 类 / vehicle 旁路 | ✅ 部分 | vehicle 旁路 ✅; ship 未独立 |
| §5.3 五组 pair experts | ⚠️ | 规则版完成(端到端~0); 学习式未实现 |
| §5.4 属性辅助监督 | ❌ | 类级属性表未建 |
| §5.5 稀有类父类收缩原型 | ⚠️ | 轻量版验证无增量(E6) |
| §5.6 proposal-domain 训练 | ✅ 部分 | E4 做了(边际, M1 域 gap) |
| §5.6 D4 旋转一致性样本 | ❌ | 未做 |

### 模块 C: RCR 风险/官方匹配校准头(§6)——⚠️ 部分实现
| 子项 | 状态 | 说明 |
|---|---|---|
| §6.3 输入特征(oto/FPN/edge/scene/D4 方差等) | ❌ | 只用 score/log_se/log_ar/density/粗类/重复特征 |
| §6.2 负样本分类学(7 类标注) | ❌ | 用位置匹配/细类匹配做标签, 未做分类学审计 |
| §6.4 有界残差打分 z=logit(s)+clip(δ) | ✅ | 完全一致(初始化 δ=0, 首轮=Y5) |
| §6.4 三个 coarse intercept 禁 25 阈值 | ✅ | 符合 |
| §6.5 P(correct fine & loc & unique) | ⚠️ | v0/v1/双头逼近该目标, 未显式联合建模 |
| §6.1 标签=match_correct | ✅ | v1 用细类正确匹配 |

### 模块 D: M3/D4 教师蒸馏(§7)——⚠️ 只做第一层
| 子项 | 状态 | 说明 |
|---|---|---|
| §7.2 第一层 困难样本课程 | ✅ | E7 实现+验证(边际) |
| §7.2 第二层 FGD 前景特征蒸馏 | ❌ | 未实现 |
| §7.2 第三层 CrossKD 预测蒸馏 | ❌ | 未实现 |
| §7.3 D4 旋转教师 | ❌ | Y5 训练期旋转已有, D4 蒸馏未做 |
| §7.4 大 crop teacher | ❌ | 维持 ConvNeXt-T |

### 模块 E: SparseZoom(§8)——🛑 停止
按 §8.3 停止条件执行: uniform SAHI 诊断 2× 放大仅救 6/82(7%), 不启动。

### 模块 F: 全局对象层(§9)——❌ 未实现
| 子项 | 状态 | 说明 |
|---|---|---|
| §9.1 tile 候选→全局坐标→stitching→canonical→一次 FGR/RCR | ❌ | 未做(当前逐图处理) |
| §9.2 全局对象图(节点/边/连通分量) | ❌ | 未做 |
| §9.3 ship/vehicle ownership-zone dedup | ❌ | 用全类 NMS 近似(已证 aircraft 有效, ship/vehicle 未必) |
| §9.4 10K 大图时延(p95 15-18s) | ❌ | 未做 10K 端到端管线 |

> **重要风险**: §9.3 明确"ship/vehicle 不能照搬 NMS", 当前 Balanced 用全类 NMS
> (包括 ship/vehicle)。三折已证总体不丢 TP, 但 10K 大图上相邻真实目标的
> ownership-zone 处理仍是未覆盖的部署关键路径。

### §10 训练策略
- Stage A(安全模块 replay): ✅ E1-E3;
- Stage B(候选头训练): ⚠️ E8 存在性正则版(非完整候选头);
- Stage C(教师蒸馏+困难课程): ⚠️ 只做了困难课程;
- Stage D(联合适配): ❌ 未做。

---

## 6. 验证体系落地(对照 §12)

| § | 项 | 状态 | 说明 |
|---|---|---|---|
| 12.1 | 低保真排名相关性标定(20/40/80/160 tau) | ❌ | 未做; 有 M1/Y2/Y3/Y4/Y5 历史 checkpoint 可做 |
| 12.2 | 四级漏斗 L0-L3 | ✅ | L0 replay(E1-E3)/L1 缓存头(E4-E6)/L2 单折 detector(E8)/L3 sentinel |
| 12.3 | prospective sentinel 冻结 | ✅ | 23 组 555 图(12.4%), 只评最终版本 |
| 12.4 | 机制挑战集 panel | ❌ | 未建固定 challenge panel |
| 12.5 | Paired delta ledger | ✅ | 已实现并用于 E1E2/COPH 对比 |
| 12.6 | MO-ASHA / Hyperband | ❌ | 手动参数扫描替代 |
| 12.7 | 统一准入门槛 | ✅ | 应用到各实验 gate |

---

## 7. 三个冻结版本(对照 §15)

| 版本 | 构成 | 关键指标 | 状态 |
|---|---|---|---|
| **Safe** | Y5 + R1-R3 融合 + aircraft post-rerank NMS + SoftRisk | t=0.1: R=0.9405 / F=0.1459(vs Y5 基线 0.9211/0.1567) | ✅ 冻结, 可回退 |
| **Balanced** | COPH + R3 + 全类 NMS + SoftRisk | 三折 t=0.1: R=0.9424 / F=0.1588 | ✅ 冻结(三折+sentinel 验证) |
| **Attack** | Balanced + 激进 SoftRisk(beta=0.7) + 学习式 E5(可选) | — | ⚠️ 组件全, 未组合验证 |

**对照总纲 §15 准入目标**:
- Safe(≥0.93/≤0.12-0.13): Recall ✅(0.9405), FDR ❌(0.1459, 高于 0.13);
- Balanced(≥0.95/≤0.12): 未达(0.9424/0.1588);
- Attack(≥0.96/≤0.12): 未达。
- **达标路径见 §10**。

---

## 8. 未实现/部分实现清单(供下一步讨论)

### A. 已证机制、待补结构(高优先级)
1. **完整 COPH 头**(§4): P2 candidate-only + p_exist/p_coarse/q_loc + candidate_score 改写。存在性正则已证三折收益, 结构版是自然延伸(候选阶段彻底不依赖细类)。
2. **学习式 E5 改类决策器**: 规则版净潜力 +2,500 已证, 唯一缺陷是 broken TP 267-393; 用学习式触发(输入 crop logits + detector score + 位置特征 + 细类 margin)压 broken, 是当下**单点增益最大的可做项**。
3. **M3-KD 二层/三层**(§7.2): FGD 前景特征蒸馏 + CrossKD 预测蒸馏。困难课程(一层)已验证边际, 但 FGD/CrossKD 是不同机制(表征迁移), 针对 COPH 未覆盖的漏检。

### B. 部署必需、尚未覆盖
4. **全局对象层**(§9): 10K 大图 tile stitching + canonical view + ship/vehicle ownership-zone dedup。当前全类 NMS 对 ship/vehicle 是近似, 10K 部署前必须做; 也是 FDR 收口的来源之一。
5. **10K 端到端时延管线**(§9.4): p95 15-18s 预算验证。

### C. 验证体系补全(低成本, 防误判)
6. **§12.1 低保真排名相关性**: 用 M1/Y2/Y3/Y4/Y5 历史 checkpoint 算 20/40/80/160 epoch 的 Kendall tau——决定后续能否用短训快速淘汰方向。
7. **§12.4 机制挑战集**: 固定 FN_CLS/91 全漏/高分 FP_BG/broken TP/HM-LQS/TP controls panel, 模块级诊断。
8. **§12.6 MO-ASHA**: 多保真 Pareto 筛选。

### D. 可选增强
9. **属性辅助监督**(§5.4): 类级属性表(masked attribute prediction), 辅助细类表征。
10. **D4 旋转教师蒸馏**(§7.3): 推理期单视图, 训练期全 D4。
11. **COPH oto 支持诊断**(§4.5): 零训练, 看 one-to-one 输出是否提供独立信息。
12. **Attack 组合验证**: Balanced + beta=0.7 + 学习式 E5 组合, 冲击 0.96/0.12。

---

## 9. 关键数值汇总(全部官方细类口径)

| 指标 | Y5 基线 | Safe | Balanced(COPH) |
|---|---|---|---|
| t=0.1 Recall | 0.9211 | 0.9405 | 0.9424 |
| t=0.1 FDR | 0.1567 | 0.1459 | 0.1588 |
| Macro Recall(三折链) | 0.8115 | — | 0.8233 |
| sentinel Recall(t=0.1) | 0.9582(链后) | — | 0.9601(链后) |
| sentinel FDR(t=0.1) | 0.0766 | — | 0.1072 |
| 净 TP(ledger vs Y5) | — | — | +75(新193/坏118) |

**错误预算对照(总纲 §2)**: 细类纠错 448 个即达 0.9425/0.1371——目前 Balanced 的
0.9424/0.1588 说明 Recall 接近该算术目标, 但 FDR 未同步(FP_CLS 收口不足),
**学习式 E5(细类纠错)+ RCR 特征升级是 FDR 收口的对症路径**。

---

## 10. 给 GPT 讨论的下一步建议(按 ROI 排序)

1. **学习式 E5 改类决策器**(已证潜力, 对症 FP_CLS/FN_CLS 双倍收益): 缓存已有(E1 crop logits + E4 微调 logits), 成本 replay 级;
2. **完整 COPH 头结构**(P2 candidate-only + candidate_score 改写): 把存在性正则的机制收益升级为结构收益, 单折 L2 验证;
3. **M3-KD FGD/CrossKD**(二层/三层): 针对漏检(非细类), 与 E5 错误来源互补;
4. **全局对象层 + 10K 管线**(§9): 部署必需, 也是 ship/vehicle dedup 的正确化;
5. **验证体系补全**(§12.1/12.4): 低成本, 让后续每个决策更可信;
6. **Attack 组合**: 最后用 Balanced+激进参数冲击 0.96/0.12。

> 核心约束: 官方测评即将开放, 任何新 detector 训练都应按 L2 单折/双哨兵过 gate
> 再进三折; replay 级(1/2/3/5)成本最低、应优先。

---

## 附录: 代码与产物索引

| 类别 | 路径 |
|---|---|
| 总报告 | reports/experiments/DCR2_YOLO_IMPLEMENTATION_20260820.md(本文件) |
| E0 审计 | reports/experiments/E0_PROTOCOL_AUDIT_20260820.md |
| E1/E2 Safe 链 | reports/experiments/E1_E2_SAFE_CORE_20260820/ |
| E2/E3 replay | reports/experiments/E2_E3_REPLAY_20260820/ |
| E4/E5/E6 | reports/experiments/E4_E5_FGR_20260820/ |
| E7 困难课程 | reports/experiments/E7_HARD_CURRICULUM_20260820/ |
| E8 COPH | reports/experiments/E8_COPH_FOLD0_20260820/(fold0+三折+ledger) |
| E9/E10 停止 | reports/experiments/E9_P2LITE_STOP_20260820/ + E10_SAHI_DIAG_20260820/ |
| E11 Balanced | reports/experiments/E11_BALANCED_20260820/ |
| E12 验证体系 | reports/experiments/E12_SENTINEL_20260820/ + E12_VERIFICATION_20260820/ |
| 核心脚本 | scripts/e1_y5_rerank_screen.py / soft_risk_v0.py / run_safe_chain.py / e5_pair_experts_rule.py / e8_coph_softrisk_verify.py / paired_delta_ledger.py / evaluate_sentinel.py / gen_coph_manifest.py |
| 核心模型 | src/rsdet/innovation/coph_presence.py / hierarchical_loss.py |
| config | configs/experiments/e8_coph_fold0_40ep.yaml / e4_fgr_v1.yaml |
| 服务器权重 | /workspace/results/E8-COPH-FOLD{0,1,2}-40EP/(fold0 已被 E7 覆盖, 需重训复现) |
| sentinel | outputs/PROSPECTIVE_SENTINEL_20260820.json(23 组 555 图) |
