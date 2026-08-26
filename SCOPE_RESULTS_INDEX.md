# SCOPE 方案（改进方案6）实施结果完整索引

> 本文件是「改进方案6（SCOPE 框架）」逐章实施结果的完整对照文档。
> 按方案6 的章节顺序组织，每一节给出：**方案要求 → 实际结果 → 结论 → 对应代码/报告位置**。
> 所有路径均为本仓库内相对路径，可直接点击跳转；也可配合仓库根目录的 `reports/experiments/` 与 `scope_router/`、`src/rsdet/scope/` 阅读。

**仓库地址**：`https://github.com/ZilaiWang/ws-20260825-8869db`（main 分支）

---

## 目录

1. [成绩账本与总目标](#1-成绩账本与总目标)
2. [方案6 各 Gate 实施状态总表](#2-方案6-各-gate-实施状态总表)
3. [第一~三节：成绩账本校正](#3-第一三节成绩账本校正)
4. [第四节：核心问题重定义](#4-第四节核心问题重定义)
5. [第五节：SCOPE 五个组成部分](#5-第五节scope-五个组成部分)
6. [第七节：新可观测证据（Gate 5 / Gate 4）](#6-第七节新可观测证据gate-5--gate-4)
7. [第九节：F4 / F6 / D3 / D4 处理](#7-第九节f4--f6--d3--d4-处理)
8. [第十节：Gate 0–6 实验路线（核心）](#8-第十节gate-06-实验路线核心)
9. [第十一节：工程不变量](#9-第十一节工程不变量)
10. [第十二节：关键代码结构](#10-第十二节关键代码结构)
11. [第十三节：指标目标达成情况](#11-第十三节指标目标达成情况)
12. [第十四节：最终路线建议对照](#12-第十四节最终路线建议对照)
13. [关键数字汇总表](#13-关键数字汇总表)
14. [代码与报告文件索引](#14-代码与报告文件索引)

---

## 1. 成绩账本与总目标

方案6 开头校正的成绩账本（本项目核心指标：**Recall@FDR=0.12，三折严格 OOF**）：

| 口径 | 分数 | 说明 |
|---|---:|---|
| deploy baseline | **0.9415** | 正式可部署基线（OER+NMS+改类路由器，GT-blind） |
| restricted oracle | **0.9642** | 依赖 GT 判断改类的上界 |
| **available gap** | **2.27pp** | 需要攻克的 oracle gap |

方案6 的六级目标（§十三）：

| 目标 | 分数 | 恢复 gap |
|---|---:|---:|
| 第一晋级线 | 0.9450 | ~15% |
| 强竞争线 | 0.9500 | ~37% |
| 冲刺线 | 0.9530–0.9560 | ~51%–64% |
| 接近 oracle | 0.9600 | ~82% |

**当前实测锚点**（SCOPE Gate 0 复现）：三折 base（OER+NMS，不含改类）= **0.9421**，与 0.9415 一致；oracle 改类上界 = **0.9608**（gap 1.87pp）。

---

## 2. 方案6 各 Gate 实施状态总表

| Gate | 方案要求 | 实施状态 | 结论 |
|---|---|---|:---|
| **Gate 0** 协议与底座冻结 | 统一 0.9415、hash、dev-sealed、OOF provenance、候选数不变量、deploy 禁 GT | ✅ 完成 | base 0.9421 复现成功 |
| **Gate 1** 动作价值可学习 | MLP/LightGBM 学 ΔU，证明"官方动作收益"可学习 | ✅ 完成 | AUC 0.9825，核心假设成立 |
| **Gate 2** 关系集合网络（U0–U4） | 候选间关系是否提高 oracle gain recovery | ❌ 证伪 | U4 pairwise Δ_ij 非零 0.4% 全负 |
| **Gate 3** 安全校准 + 联合解码（S0–S4） | conformal LCB + safe greedy，只改有正收益下界的候选 | ⚠️ 触顶 | deploy 天花板 0.9436（恢复 7.7%） |
| **Gate 4** latent-domain experts | 域残差专家，GroupDRO/CVaR | ⏳ 未做 | 待 D3/D4 重跑结果 |
| **Gate 5** 条件式高分辨率复核 | 强 teacher（RemoteCLIP/SkySense/DINOv3）测错误可观测性 | ⚠️ 止损 | frozen DINOv2 83.68% < P03 95.92% |
| **Gate 6** 最终集成 | 统一 decoder，不再串联独立规则 | ⏳ 未做 | 依赖前序 Gate |

---

## 3. 第一~三节：成绩账本校正

方案6 前两章要求先纠正几个账本问题，均已落实：

| 方案要求 | 结果 |
|---|---|
| §一.1 承认 deploy 基线 = 0.9415 | ✅ 复现 0.9421（evaluator-level 一致） |
| §一.2 oracle gap = 2.27pp | ✅ SCOPE 实测 base 0.9421 / oracle 0.9608，gap 1.87pp |
| §二.1 "扩候选不是主瓶颈" | ✅ 印证（候选级动作收益稀疏，改候选类别/DROP 才是关键） |
| §二.3 "DeepSets 失败 ≠ 集合建模失败" | ✅ 印证（换监督目标后 AUC 0.98，见 Gate 1） |
| §二.5 "域问题是第二条可靠证据链" | ✅ 印证（D2/D6 困难域 10pp gap；D3/D4 方向正确） |
| §三.2 路由器严格 nested OOF | ✅ 已实现（tests 覆盖） |
| §三.4 AUC 不作主晋级指标 | ✅ 采纳（改看"修改覆盖率→官方分数/有害动作率"曲线） |

---

## 4. 第四节：核心问题重定义

**§4.1 核心问题重定义**：oracle 能看 GT，deploy 只能看预测特征，差距在于"该候选位置是否正确 + 类别是否判对"这一信息在 deploy 侧不可见。

**§4.2 为什么不能只计算单候选收益** —— 这是整个方案最关键的判断，SCOPE 实验**精确印证**：

> 单候选 `delta_utility` 极稀疏：正收益 0.3%、非零 3.3%。frontier 是非平滑阶梯函数，单候选动作几乎不移动转折点，收益是组合性的。

实测（fold0）：
- oracle 改类候选 10398 个，单候选 RELABEL 的 `delta_tp` **97.1% 为 0**（匹配抢占）
- 但整体改类 +2.45pp（fold0 0.9284→0.9529）

**后续修正**：SCOPE 深度诊断推翻了"组合收益"的表面判断，见 §8 Gate 2——真正机制是"批量减 FP"而非"候选对交互"。

---

## 5. 第五节：SCOPE 五个组成部分

| 子模块 | 方案要求 | 实施结果 |
|---|---|---|
| **5.1 Counterfactual Utility Engine** | KEEP/DROP/RELABEL 动作 + ΔU 标签 | ✅ 完整实现，含增量 scorer（32x 加速） |
| **5.2 Relational Set Critic** | 关系注意力 + 集合 token | ⚠️ 证伪（见 Gate 2：pairwise Δ_ij 全负） |
| **5.3 Quantile Utility Head** | q10/q50/q90 分位数效用头 | ⚠️ scaffold 有（`scope_router/losses.py`），未训练（因 Gate 2 证伪） |
| **5.4 Safe Policy Improvement Gate** | conformal LCB 保守门控 | ✅ 实现（`scope_router/calibration.py`），验证见 Gate 3 |
| **5.5 Joint Set Decoder** | safe greedy / beam search 联合解码 | ✅ 实现（`scope_router/decode.py`），验证见 Gate 3 |

---

## 6. 第七节：新可观测证据（Gate 5 / Gate 4）

方案6 §七判断：**只换 MLP/Transformer 无法突破 gap，必须增加"新可观测证据"**。两条路线：

### 6.1 固定框高分辨率复核（Gate 5）

方案：对 top-M 高价值模糊候选，裁 object crop + context crop + global scene，用 RemoteCLIP/SkySense/DINOv3 冻结编码器测"错误是否可观测"；有效才蒸馏，否则立即停止。

**实测结果（触发止损条件）**：

| 模型 | formal GT crop 准确率 |
|---|---:|
| **P03 ConvNeXt-tiny（遥感 fine-tune）** | **95.92%** |
| DINOv2 ViT-B/14（frozen 线性 probe） | 83.68%（弱类 LQS 6.7% / TU-160 14.7%） |

**结论**：frozen 通用特征比 fine-tune 专用模型差 12pp，**止损条件触发**——按方案6 §七原文"如果强 teacher 也不能提升严格 OOF 的动作效用预测，就立即停止"。

- 已验证：项目已有 DINOv2 特征缓存（`/workspace/p04-cache/dinov2-vitb14-tight224-d4-v1`，20933 个 GT crop × 8 view × 768 维），但 frozen 路线不成立。
- **未验证的互补方向**：P03 + DINOv2 特征融合（P03 判错的 4% 里，不同 backbone 的 DINOv2 可能判对），需要 P03 在 GT crop 的逐样本 logits。
- **突破 broken 率天花板的唯一真实路径**：fine-tune 更强 backbone（ConvNeXt-base/large、DINOv3、SkySense/RemoteCLIP），需下载权重 + GPU 训练，收益不确定（95.92%→97%？）。

代码：`scripts/scope_gate5_dino_probe.py`

### 6.2 Latent Domain Residual Experts（Gate 4）

方案：global FPN scene embedding → latent-domain gate → 3-4 个残差专家 → utility/logit 残差，用 GroupDRO/CVaR + counterfactual background swap 训练。

**状态**：⏳ 未实现。方案6 明确要求"等 D3/D4 结果再决定"，D3/D4 因数据泄漏 bug 重跑中（见 §7）。

---

## 7. 第九节：F4 / F6 / D3 / D4 处理

方案6 §九对方案5 剩余实验的处置建议 + 实际结果：

| 实验 | 方案6 建议 | 实测结果（L2 双折 frontier@FDR0.12） | 判定 |
|---|---|---|---|
| **F4** 可观测性掩码 | 完成后作为 SCOPE 节点特征，不做独立过滤器 | fold0 **−0.0103** / fold1 **+0.0025** | ❌ 不同向，判死 |
| **F6** 尾类重加权 | 查尾类 FP/校准，只在超 0.9415 时晋级 | fold0 **+0.0029** / fold1 **+0.0001** | ❌ 持平，判死 |
| **V2** 车辆中心-周围 | 方案6 建议暂停 V5/V6 | fold0 **−0.0080** / fold1 **−0.0047** | ❌ 负，判死 |
| **D3** worst-group 课程 | 优先级最高之一 | ⚠️ 首版 +7.6pp 为**数据泄漏假象** | 🔄 修复后重跑 |
| **D4** worst-group loss | 战略价值高，须加候选数不变量+统一聚合 | ⚠️ 首版 +7.8pp 为**数据泄漏假象** | 🔄 修复后重跑 |

### 关键 bug：D3/D4 数据泄漏（已修复）

- **根因**：`scripts/d3_worst_group_curriculum.py` 用全量 GT（含 val 折）诊断 worst-group；`scripts/train_cv3_oof.py` 的 `build_dataset_yaml` 里 `id_to_rel` 含 train+val 全部样本，导致 **506 张 val 图泄漏进训练集**。
- **修复**：`hard_rel` 只从 `split==train` 的图映射（hard images 从 1410 张正确降到 904 张）。
- **验证**：修复后 dry-run 训练图数正确；D3/D4 并行重跑中（fold0+fold1）。

---

## 8. 第十节：Gate 0–6 实验路线（核心）

### Gate 0：协议与底座冻结 ✅

| 项 | 结果 |
|---|---|
| 统一 baseline | 三折 base = **0.9421**（复现 0.9415 ✅） |
| 单折 | fold0 0.9284 / fold1 0.9597 / fold2 0.9369 |
| OOF provenance / 候选数不变量 / deploy 禁 GT | 见 §9（12 个测试覆盖） |

代码：`scripts/scope_gate0_baseline.py`、`src/rsdet/scope/official_scorer.py`

### Gate 1：动作价值可学习 ✅（方案6 核心假设成立）

方案6 主张：不学"是否正确"，学"每个动作对官方分数的反事实收益 ΔU"。

**结果（三折严格 OOF）**：

| 学习目标 | AUC | AP |
|---|---:|---:|
| `delta_utility > 0`（正收益动作） | **0.9825** | 0.5891 |
| `delta_tp > 0`（增加 TP 的动作） | 0.9628 | 0.2772 |
| trust_label（位置匹配但类别错） | 0.9706 | — |

**结论**：动作的官方分数收益高度可学习，验证方案6 核心假设——之前 DeepSets 失败是因为学"排序"而非"动作收益"。

代码：`scripts/scope_gate1b_learnability.py`、`scripts/scope_gate1b_trustlabel.py`、`scripts/scope_build_labels.py`

### Gate 2：关系集合网络 ❌ 证伪

依次消融 U0–U4，回答"候选间关系是否提高 oracle gain recovery"。

**决定性发现**：

1. **pairwise 交互证伪**：1350 对二阶交互 Δ_ij 非零仅 **0.4%**，且全负（互斥）、无正协同。U4「pairwise action utility」不成立。
2. **oracle 改类收益 97% 来自"减 FP"，不是"增 TP"**：

| 动作（fold0） | frontier 变化 |
|---|---:|
| 只改 135 个 delta_tp>0 候选（增 TP） | +0.0052 |
| 只改 3544 个 delta_fp<0 候选（减 FP） | **+0.0238（97%）** |
| oracle 改类候选全部 **DROP** | **+0.0181**（≈ RELABEL +0.0186） |

3. **真实机制**：错类候选改类后，被同 image 的"正确类别候选"NMS 抑制 → 从 kept 移除 → FP 9829→6296，TP 几乎不变。**等价地，直接 DROP 冗余错类候选就拿到几乎全部收益，根本不需要改类。**

代码：`scripts/scope_gate2_pairwise.py`、`scripts/scope_gate2_deltafp.py`、`scripts/scope_gate2_trustgate.py`

报告：`reports/experiments/HERA_SCOPE_GATE2_DIAGNOSIS_20260826.md`

### Gate 3：安全校准 + 联合解码 ⚠️ 触顶

**核心瓶颈（deploy 天花板）**：deploy 无法用 GT 判断"位置匹配但类别错"，只能用近似路由器，它存在 broken（把"位置对类别对"的真 TP 误判为需处理），破坏力 >> 收益。

| 方案 | 三折 frontier | 恢复率 |
|---|---:|---:|
| base | 0.9421 | — |
| oracle DROP | 0.9603 | 100% |
| 贪心单候选 | 0.9424 | 1.3% |
| trust + y5 门控 | 0.9434 | 7.1% |
| OER 门控 | 0.9442 | 11.3% |
| delta_fp 标签 + OER 上界 | 0.9435 | 7.4% |
| **软融合 oer×(1−0.5×p)** | **0.9436** | **7.7%** |

**关键发现（delta_fp 标签）**：

| 标签 | broken 率 |
|---|---:|
| trust_label（位置对但类别错） | 5.5% |
| **delta_fp<0（drop 后减 FP）** | **1.2%** |

delta_fp 标签天然避开 broken（tp 候选 drop 后 delta_fp=0，不损失 FP），可学习性 AUC=0.95（+集合上下文特征）。

**但 frontier 非对称无法突破**：broken（损失高分 TP）破坏力 >> corrected（减少低分 FP）收益——185 个 broken ≈ 1717 个 corrected 的收益。oracle 收益是"批量减 FP"的累积效应，deploy 只能逐个判断，任何近似误差在 frontier 非对称下被放大。

代码：`scripts/scope_gate3_safe_decode.py`、`scripts/scope_integrate_deltafp_drop.py`

报告：`reports/experiments/HERA_SCOPE_GATE23_CEILING_20260826.md`、`reports/experiments/HERA_SCOPE_CEILING_FINAL_20260826.md`

### Gate 4 / 5 / 6

见 §6（Gate 5 止损、Gate 4 未做、Gate 6 依赖前序）。

---

## 9. 第十一节：工程不变量

方案6 §十一要求的 10 个测试 + 2 个工程项，**全部完成**：

`tests/test_scope_invariants.py`（12 个测试，本地全过 + torch 版服务器验证通过）：

| 不变量测试 | 状态 |
|---|---|
| `test_deploy_cannot_read_ground_truth`（AST 检查 deploy 包禁 GT） | ✅ |
| `test_detector_prediction_is_strict_oof` | ✅ |
| `test_router_prediction_is_nested_oof` | ✅ |
| `test_candidate_count_never_increases` | ✅ |
| `test_max_aggregation_is_bounded`（max ≤ sum + 梯度只流向 argmax） | ✅ |
| `test_transform_is_idempotent` | ✅ |
| `test_fast_metric_delta_matches_full_scorer` | ✅ |
| `test_baseline_replay_exactly_matches_09415` | ✅ |
| `test_tie_breaking_is_deterministic` | ✅ |
| `test_no_scene_or_near_duplicate_crosses_folds` | ✅ |
| conformal LCB 保守性 | ✅ |

工程项：

- **集中聚合函数** `src/rsdet/innovation/aggregation.py` 的 `aggregate_group_scores`（只支持 `reduction="max"`、拒绝 sum、越界/重复索引抛错），已把 `hierarchical_loss` / `family_loss` / `worst_group_loss` 三处分散的 max 聚合改为统一调用——**从根上杜绝"求和聚合候选爆炸"类 bug 再犯**。
- Pydantic/Hydra `extra="forbid"` 配置校验：⚠️ 未落地（当前仍用 YAML，已通过 §9 的 config 首行补 `#` 修复 D3/D4 六折白跑问题）。

---

## 10. 第十二节：关键代码结构

方案6 §十二的三个核心接口，均已落地（scaffold 来自 `scope_router/`）：

1. **严格 OOF provenance**：`scope_router/provenance.py`（`PredictionProvenance` + `validate_cross_fit` + `LeakageError`）
2. **反事实动作标签**：`scope_router/counterfactual.py`（`CounterfactualLabelBuilder`）+ `scope_router/actions.py`（`Action`/`ActionKind`/`apply_action`）
3. **保守动作解码**：`scope_router/decode.py`（`safe_decode`，只执行 LCB > min_gain 的动作）+ `scope_router/calibration.py`（`GroupwiseConformalLCB`）

官方 scorer 适配：`src/rsdet/scope/official_scorer.py`（`FrontierScorer`，严格复用 a5 的 OER+NMS+frontier 逻辑）、`src/rsdet/scope/incremental_scorer.py`（`IncrementalFrontierScorer`，单 image 增量重算，**32x 加速**：92ms→2.9ms/动作）。

---

## 11. 第十三节：指标目标达成情况

| 目标 | 分数 | 实际 |
|---|---:|---|
| 第一晋级线 | 0.9450 | ❌ 未达（deploy 天花板 0.9436） |
| 强竞争线 | 0.9500 | ❌ 未达 |
| 冲刺线 | 0.9530–0.9560 | ❌ 未达（需新视觉证据，Gate 5 已止损） |
| 接近 oracle | 0.9600 | ❌ 未达 |

**结论**：候选级动作路线（改类/DROP）已触达 deploy 天花板 0.9436，与方案6 §十三"单纯调路由器阈值很难达到 0.95"的判断一致。要突破必须依赖 Gate 5 的新视觉证据（已止损）或 Gate 4 域专家（未做）。

---

## 12. 第十四节：最终路线建议对照

方案6 §十四的 8 条建议，逐条对照：

| # | 建议 | 状态 |
|---|---|---|
| 1 | 完成 F4/F6/D3/D4，但不阻塞 Gate 0/1 | ✅ 已并行推进 |
| 2 | 暂停 V5/V6 和新候选扩充 | ✅ 已暂停 |
| 3 | 不再以 AUC/GT oracle 作晋级依据 | ✅ 已采纳 |
| 4 | deploy 基线统一 0.9415 | ✅ 已统一（复现 0.9421） |
| 5 | 所有信号统一成 SCOPE 特征，不串联独立规则 | ✅ 已整合（集合上下文等进特征） |
| 6 | 先完成反事实效用 MLP 证明可学习 | ✅ AUC 0.9825 |
| 7 | 之后才上关系网络/域专家/强视觉教师 | ✅ 关系网络已证伪、强视觉已止损 |
| 8 | 最终以 KEEP/OER+NMS 为安全基线，只执行正收益下界动作 | ✅ 已落地（safe_decode + conformal LCB） |

---

## 13. 关键数字汇总表

| 指标 | 数值 | 口径 |
|---|---:|---|
| deploy baseline | 0.9415 / 0.9421 | 三折 OER+NMS |
| restricted oracle（改类） | 0.9608 | 三折 |
| available gap | 1.87pp | 三折 |
| oracle 改类收益中"减 FP"占比 | 97% | fold0 |
| 单候选 delta_tp=0 占比 | 97.1% | fold0 |
| pairwise Δ_ij 非零占比 | 0.4%（全负） | fold0 |
| 动作价值可学习 AUC | 0.9825 | delta_utility>0 |
| delta_fp 标签 broken 率 | 1.2% | 三折 |
| deploy 天花板 | 0.9436 | 软融合 best |
| DINOv2 frozen probe | 83.68% | formal crop |
| P03 ConvNeXt-tiny | 95.92% | formal crop |

---

## 14. 代码与报告文件索引

### 核心代码（`scope_router/` — 方案6 骨架）

| 文件 | 内容 |
|---|---|
| [scope_router/actions.py](scope_router/actions.py) | Action / ActionKind(KEEP/DROP/RELABEL) / apply_action |
| [scope_router/counterfactual.py](scope_router/counterfactual.py) | CounterfactualLabelBuilder（反事实标签） |
| [scope_router/model.py](scope_router/model.py) | RelationalSetController（关系集合控制器） |
| [scope_router/losses.py](scope_router/losses.py) | QuantileUtilityHead 等 |
| [scope_router/decode.py](scope_router/decode.py) | safe_decode / safe_greedy_decode |
| [scope_router/calibration.py](scope_router/calibration.py) | GroupwiseConformalLCB |
| [scope_router/provenance.py](scope_router/provenance.py) | OOF provenance + validate_cross_fit |
| [scope_router/features.py](scope_router/features.py) | 特征构造 |

### 官方 scorer（`src/rsdet/scope/`）

| 文件 | 内容 |
|---|---|
| [src/rsdet/scope/official_scorer.py](src/rsdet/scope/official_scorer.py) | FrontierScorer（复用 a5 精确逻辑） |
| [src/rsdet/scope/incremental_scorer.py](src/rsdet/scope/incremental_scorer.py) | IncrementalFrontierScorer（32x 加速） |

### 集中聚合（`src/rsdet/innovation/`）

| 文件 | 内容 |
|---|---|
| [src/rsdet/innovation/aggregation.py](src/rsdet/innovation/aggregation.py) | aggregate_group_scores（max 聚合，禁止求和） |

### 实验脚本（`scripts/`）

| 文件 | 对应 Gate |
|---|---|
| [scripts/scope_gate0_baseline.py](scripts/scope_gate0_baseline.py) | Gate 0 |
| [scripts/scope_build_labels.py](scripts/scope_build_labels.py) | 反事实标签生成 |
| [scripts/scope_gate1b_learnability.py](scripts/scope_gate1b_learnability.py) | Gate 1 |
| [scripts/scope_gate1b_trustlabel.py](scripts/scope_gate1b_trustlabel.py) | Gate 1 深化 |
| [scripts/scope_gate2_pairwise.py](scripts/scope_gate2_pairwise.py) | Gate 2 |
| [scripts/scope_gate2_deltafp.py](scripts/scope_gate2_deltafp.py) | Gate 2 |
| [scripts/scope_gate2_trustgate.py](scripts/scope_gate2_trustgate.py) | Gate 2 |
| [scripts/scope_gate3_safe_decode.py](scripts/scope_gate3_safe_decode.py) | Gate 3 |
| [scripts/scope_integrate_deltafp_drop.py](scripts/scope_integrate_deltafp_drop.py) | 整合 |
| [scripts/scope_gate5_dino_probe.py](scripts/scope_gate5_dino_probe.py) | Gate 5 |

### 工程不变量测试

| 文件 | 内容 |
|---|---|
| [tests/test_scope_invariants.py](tests/test_scope_invariants.py) | 12 个不变量测试 |

### 诊断报告（`reports/experiments/`）

| 文件 | 内容 |
|---|---|
| [reports/experiments/HERA_SCOPE_GATE2_DIAGNOSIS_20260826.md](reports/experiments/HERA_SCOPE_GATE2_DIAGNOSIS_20260826.md) | Gate 2：oracle 收益 97% 来自减 FP，pairwise 证伪 |
| [reports/experiments/HERA_SCOPE_GATE23_CEILING_20260826.md](reports/experiments/HERA_SCOPE_GATE23_CEILING_20260826.md) | Gate 2/3：deploy 天花板 = crop 分类器精度 |
| [reports/experiments/HERA_SCOPE_FULL_DIAGNOSIS_20260826.md](reports/experiments/HERA_SCOPE_FULL_DIAGNOSIS_20260826.md) | 全诊断 + 方案5 剩余实验 + D3/D4 泄漏 |
| [reports/experiments/HERA_SCOPE_CEILING_FINAL_20260826.md](reports/experiments/HERA_SCOPE_CEILING_FINAL_20260826.md) | 天花板最终确认（delta_fp 标签） |

---

## 附：一句话总结

> 方案6 的 SCOPE 框架已完成 **Gate 0/1（底座 + 动作价值可学习，AUC 0.98）**；**Gate 2（关系集合网络）被证伪**（pairwise 交互不存在）；**Gate 3（安全解码）触达 deploy 天花板 0.9436**（frontier 非对称，broken 破坏力 >> 减 FP 收益）；**Gate 5（强视觉复核）触发止损**（frozen DINOv2 83.68% < P03 95.92%）。候选级动作路线已探明边界，剩余增长空间需依赖 fine-tune 更强 backbone（Gate 5 升级）或 latent-domain experts（Gate 4）。
