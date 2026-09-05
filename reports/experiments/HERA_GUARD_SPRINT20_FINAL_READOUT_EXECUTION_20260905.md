# HERA-Guard Sprint20 最终读出冲刺执行报告

日期：2026-09-05

源码基准：`ab51106949ad8369a5cccd862fcc19e1739cdeb2`

评估口径：`platform_observed_20260831`

服务器环境：Ultralytics `8.4.103`、PyTorch `2.5.1+cu121`、RTX 3090

状态：**实验与证据审计闭环；没有新的攻击候选获得部署准入**

## 1. 结论先行

方案20提出的两个问题都已用真实权重和真实数据检查完毕：

1. 成熟 P40 权重内的 one-to-many（OTM）头确实包含新的 Ship 检出能力；只让 OTM 接管 `QHS/MS`（类别 2/3）后，在短 OOF 三折和 255 个来源组 bootstrap 中均呈小幅正向。这一范围是在查看多个候选范围、三折和 full-seen 结果后选出，必须标为**后验选择的开发信号**，不是三次独立确认。
2. 但是节省计算所必需的共享前向没有通过全量逐框一致性：shared-OTO 完全一致，shared-OTM 在 4,481 张图中的 61 张存在框坐标差异。按预注册规则不得放宽容差，因此 OTM 路由不进入 Docker。
3. D4 概率上界早退在所有检查图上保持逐框一致；但在与当前 view-consistency D4 相匹配的 100 张硬例上总耗时增加 3.93%，因此 `bounded_d4=false`，继续使用原始 D4。
4. 当前工程安全线仍是 **冻结 P40 路径 + 原版 Aircraft-D4 only**。按本地冻结参考，Ship 和 FSC 保持 v2.0 的 P40 输出，Aircraft 使用 v3.0 已经由官方隐藏集确认有效的 D4；平台没有返回镜像 digest/head 字段，所以这里是强本地身份链，不写成平台端独立逐框证明。方案20没有发现一个证据充分、可以替代安全线的攻击版本。

这不是“OTM 没有效果”。准确结论是：**OTM 的 Ship 信号存在，但证据仍受短训练血缘、后验范围选择和非嵌套交叉选择限制；部署实现也没有同时满足精确等价与时延要求。**

## 2. 冻结边界

本轮没有修改以下正式资产：

- P40 权重：`b0df7981f6ad58fe8eca65fb0deef54feed55caf300c2c219ac1ccb3500c8012`
- Aircraft-D4 权重：`5f1b175f8b2c310a3c0583e652f5cf7cfd444d0b0d74d794a9ac59e4537832d5`
- 切片：`tile=1024`、`overlap=256`
- 网络输入：`imgsz=1280`
- 候选地板：`0.001`
- Safe Fusion 和正式 P40 最终阈值：`0.536`
- 官方 25 类类别顺序与粗类映射

特别确认：类别 24 `FSC` 是“发射车”，不是 generic vehicle。本轮没有加入外部普通车辆正例，也没有临时补标。

## 3. 资产和接线审计

真实 checkpoint 审计结果：

| 项目 | 结果 |
|---|---:|
| `nc` | 25 |
| `end2end` | `true` |
| `reg_max` | 1 |
| OTM `cv2/cv3` | 存在，286,476 / 189,131 参数 |
| OTO `one2one_cv2/cv3` | 存在，286,476 / 189,131 参数 |
| 仓库关键源码 blob | 与基准提交一致 |
| P40 权重 SHA | 通过 |

因此原生 OTM 对照读取的是同一次 P40 训练形成的完整检测头，没有随机初始化，也没有把已融合删除的头当成有效资产。

原生 OTM 的选择发生在模型第一次推理之前。代码拒绝“先默认 OTO 推理、再在同一实例切 OTM”的不可靠路径，因为 Ultralytics 的融合过程可能删除 OTM 分支。

### 3.1 正式 v2.0 参考资产连接

方案21要求不能从当前默认参数反推历史正式镜像。本轮因此对 v2.0 的实际交付链进行了单独核验：

| 证据 | 冻结值或状态 |
|---|---|
| 正式 tag / submission | `v2.0` / `3953`，官方分数 `76.6010` |
| materialized manifest SHA | `1627addf575858ba39c879fc7f565d10d2cde98f548fdeecb7edb1cd6793312d` |
| 配置 SHA | `50156c2d3143f930cb6f07f0a72e76b69ad64897f363e2fdea05718c80a52e22` |
| 部署权重 SHA | `b0df7981f6ad58fe8eca65fb0deef54feed55caf300c2c219ac1ccb3500c8012` |
| 历史 competition 源码 SHA | `68a8a59d1de6cad54946dfe01ed04ff5fc6d36c67138d0ac617c78b0282acb2a` |
| 历史 adapter SHA | `e65f4afb237fa0206fdadf1b11f66f660966073051d8fc6519d0b42547f43097` |
| 本地镜像 ID / registry push digest | 同为 `sha256:db2a0e…ae1303` |
| 平台实际拉取 digest | `UNKNOWN`：平台结果没有返回该字段 |
| 交付入口 vs 冻结 ledger | 12 图、3,913 框逐框一致 |

这是一条很强的本地交付身份链，但不是平台端 digest 证明。`BUILD_MANIFEST.json` 还明确记录
`source_tree_dirty=true`，所以历史 materialized 源码与清单才是权威实物，单独的 Git commit
不足以重建该镜像。精确配置与固定版本源码强烈支持历史运行路径，但平台记录没有显式写出
`OTO/OTM`；本文不再仅凭当前框架默认值给正式镜像的 head 贴标签。

## 4. 三层证据及其边界

### 4.1 真实连续图：接线和切片 sanity

数据是 26 张真实连续图、216 个 GT；只覆盖 Ship 与 Aircraft，因此不能用于 FSC 结论，也不是隐藏分数代理。

固定阈值 `0.536`：

| 读出 | Ship TP/FP/FN | Aircraft TP/FP/FN |
|---|---:|---:|
| 原生 OTO | 54 / 4 / 3 | 159 / 0 / 0 |
| 原生 OTM | 54 / 6 / 3 | 159 / 0 / 0 |

结论：接线正常，但该小样本上 OTM 没有 Ship 召回空间，只增加 2 FP，属于轻微负向 sanity。

### 4.2 成熟 full-seen：机制诊断，不作准入

数据为 4,481 图、20,933 GT，使用正式成熟 P40 full 权重。由于训练见过这些图，只能检查机制和细类行为，不能选阈值或预测官网成绩。

固定阈值 `0.536`：

| 读出 | 粗类平均 Recall | 粗类平均 FDR | 非官方缓存时延伪分 |
|---|---:|---:|---:|
| 原生 OTO | 91.2154% | 2.7993% | 87.5577 |
| 原生 OTM | 94.5573% | 5.1882% | 89.3292 |

完整 OTM 相对 OTO 增加召回，但同时明显增加 FDR。类别互斥接管进一步显示：

| 路由（其余类保持 OTO） | 相对 OTO 的同时间诊断分差 | 解释 |
|---|---:|---|
| OTM 接管全部 Ship 0–3 | -0.1756 | 稀有 HM/LQS 的 FP 代价抵消收益 |
| OTM 只接管 QHS/MS 2–3 | **+0.8811** | +114 TP、+23 FP，方向最干净 |
| OTM 只接管 FSC 24 | +1.6423 | 后续 OOF 反向，属于记忆/不泛化 |
| OTM 接管 Ship+FSC | +1.4668 | 混入不稳的 FSC，不可用 |

同一全局 FP 预算下，OTM 在阈值约 `0.586` 时得到 93.8157% 粗类平均 Recall，OTO 为 91.2154%；这是机制上界，不是部署阈值。

### 4.3 短外层 OOF：方向证据，不是成熟分数估计

三折覆盖全部 4,481 图且按来源隔离；训练血缘为 `S1024/40e → P40/40e`，短于正式 `S1024/160e → P40/40e`。因此结果标记为 `outer_oof_short`，没有伪装成成熟 OOF。

固定 `0.536` 时，OTM 相对 OTO：

- Ship macro Recall：37.0751% → 44.8115%；
- Ship macro FDR：14.2667% → 9.8449%；
- QHS：`+50 TP/+15 FP`；
- MS：`+94 TP/+13 FP`；
- FSC 虽增加 TP，但 FDR 从 21.08% 升到 28.15%，不能接管。

早期按“另外两折选阈值、当前折评估”的 cross-fit，在目标粗类 macro FDR `0.10` 下：

| 方法 | 三折完整政策分差 | 各折分差 | 结论 |
|---|---:|---|---|
| 全部 OTM | +0.8365 | -2.9933 / +2.5309 / +1.8621 | 不稳定 |
| OTM 接管全部 Ship | +1.7373 | +0.4161 / +0.6166 / +1.9716 | 强方向，但稀有类 full-seen 风险大 |
| OTM 只接管 QHS/MS | **+0.2361** | **+0.2985 / +0.2155 / +0.2625** | 幅度小、三折一致 |
| OTM 只接管 FSC | -1.4787 | -3.5072 / +1.4532 / -0.7300 | 明确拒绝 |

上述表是开发过程中的 `v5` 记录，不是最终固定政策的严格对照。基线固定阈值 `0.536`
本身未必满足任意指定的 `0.10` 风险预算，所以“双方各自在目标预算下重选阈值”还改变了
基线政策。最终又执行了 `v7`：基线固定为实际 P40 `0.536`，每个评估折只在另外两折上
为 QHS/MS 选择达到**同一 Ship macro FDR**的 OTM 阈值。结果如下：

| v7 精确政策 | 合并分差 | 各折分差 | 选中阈值 |
|---|---:|---|---|
| 固定 P40 `0.536` + OTM QHS/MS 同风险替换 | **+0.2401** | **+0.2847 / +0.1666 / +0.2601** | `0.566 / 0.566 / 0.546` |

对应来源组 bootstrap：2,861 个有效重采样，均值 `+0.2442`，P10/P50/P90 为
`+0.1029/+0.2486/+0.3796`，正增益比例 `98.32%`。反之，若把双方都重新约束到任意固定
FDR `0.10/0.12/0.15/0.20`，分差分别为
`-1.6044/-1.6660/-1.6691/-1.1726`。这些不是部署同政策比较，只保留为风险曲线诊断。

来源组配对 bootstrap（255 组、3,000 次）：

| 路由 | 有效重采样 | 均值 | P10 / P50 / P90 | 正增益概率 |
|---|---:|---:|---:|---:|
| OTM 全 Ship | 2,861 | +1.5303 | +0.3201 / +1.1604 / +3.3609 | 97.24% |
| OTM QHS/MS | 2,861 | **+0.2544** | **+0.1083 / +0.2629 / +0.3842** | **98.32%** |
| OTM FSC | 2,861 | -1.4955 | -3.4388 / -1.4798 / +0.3483 | 15.06% |

139 次重采样缺少完整稀有类别，按合同排除并报告；没有把缺失类别填成 100% Recall。
QHS/MS 是本轮唯一在这些已查看数据上跨折、跨来源组和成熟 full-seen 同方向的读出信号。

### 4.4 方案21证据等级复核

三个分折权重确实存在，直接 held-out 预测也按 255 个来源组隔离；但它们的训练血缘为
`S1024/40e → P40/40e`，正式全量模型则为 `S1024/160e → P40/40e`。训练阶段轮数、数据范围、
父 checkpoint、有效 batch 和恢复过程均不等价，因此这些结果保持 `outer_oof_short`，只作方向开发证据。

更重要的是，普通 cross-fit 只隔离了“当前折的直接预测模型”。例如在 fold0 上评估时，
阈值由 fold1/fold2 的预测选择，而 fold1/fold2 模型的训练数据包含 fold0 来源。完整依赖链
仍间接接触当前评估来源，不满足严格的两层嵌套独立性。通用元数据审计对三个
`strict_independence` 声明全部返回 `CONTRADICTED`，同时把短 OOF 与成熟 full 的训练签名判为
`DIFFERENT`；这是预期的诚实审计结果，不是程序故障。

QHS/MS 还是在查看“全部 OTM、全部 Ship、QHS/MS、FSC”、三个折和 full-seen 后才选定。
因此三折同向不能表述为三次独立成功；255 来源组 bootstrap 只描述已查看开发数据上的条件波动，
不能消除范围选择偏差，也不能生成新的隐藏域。完整选择台账和依赖图见第 8 节索引。

## 5. Shared 前向验收

共享实现只做一次 backbone/neck 前向，分别保留 OTO 与 OTM 的原生解码和 NMS；两个分支先各自走完整 Safe Fusion，最后才按类别互斥接管。

| 数据 | shared-OTO vs native-OTO | shared-OTM vs native-OTM |
|---|---:|---:|
| 26 张连续图 | 26/26 完全一致 | 26/26 完全一致 |
| 4,481 图成熟全量 | **4,481/4,481 完全一致** | **未通过：61 图有差异** |

OTM 差异表现为检测数量、类别和 score 基本保持，但个别最终框坐标不同；例如 image 91 的一个 MS 框在纵向和宽度上产生像素级偏移。它可能与完全同分候选在独立融合中的代表框选择有关，但本轮没有用推测替代证据。

预注册要求是逐框多重集精确一致，不允许在失败后添加 IoU 容差。因此：

```text
shared_head_deployment_admission = false
```

若改用两个原生模型实例，可保留科学上的 OTM 输出，但会重新引入第二次完整前向。v3 已证明第二检测器的时延代价约 3.41 秒，远大于当前 QHS/MS 短 OOF 约 +0.24 分的后验开发增益，因此不值得作为最后攻击提交。

## 6. D4 有界早退验收

早退规则只在数学上能证明任何其他飞机类都不可能达到原始 D4 的 `0.9` 重标阈值时提前 KEEP；其余对象重新执行完整八视图，并始终保留飞机同细类 NMS。

| 数据 | 飞机框 | 提前 KEEP / 完整 D4 | 视图减少 | 输出 | 局部总耗时变化 |
|---|---:|---:|---:|---:|---:|
| 26 张连续图 | 159 | 159 / 0 | 87.5% | 完全一致 | 约 -37.9% |
| 历史 CE 选择硬例100 | 747 | 357 / 390 | 35.3% | 完全一致 | 约 -3.6% |
| 当前 consistency 选择硬例100 | 767 | 373 / 394 | 36.1% | 完全一致 | **约 +3.9%** |

最后一行与当前正式 D4 的概率来源相匹配，优先级最高。视图减少并没有转化为端到端收益，原因是提前分支、重新批处理和许多小批次的固定开销。结论：

```text
bounded_d4 = false
```

原始 Aircraft-D4 不变；本结论仅针对可选的早退优化。

## 7. 最终候选排序

### A. 安全候选：可继续正式交付准备

`frozen P40 path + original Aircraft-D4 only`

- Ship、FSC 与本地冻结 v2.0 P40 参考逐框不变；平台侧身份限制见 3.1 节；
- Aircraft-D4 在官方 v3.0 上已确认 `+21 TP/-21 FP`；
- 不含 hierarchy、Vehicle rescue、APRR、BATIS、shared OTM 或 bounded D4；
- 按已测局部开销推算，总分约比 v2.0 高 `0.11–0.13`，这是条件估计而非承诺。

### B. 攻击候选：保留代码，不准入

`P40 shared OTO + OTM(QHS/MS) + original Aircraft-D4`

- 在后验选择的短 OOF 开发数据上方向为正，不能称为独立确认；
- 配置明确标记 `experimental_not_admitted`；
- 因 shared-OTM 精确一致性失败，不得打包或提交；
- 不用 full-seen 的 +0.881 分替代独立证据，也不把短 OOF 的绝对分当官网预测。

### C. 已拒绝

- OTM 接管 FSC；
- OTM 接管全部类别；
- 两个原生头双前向部署；
- bounded D4；
- 根据本轮结果继续扫阈值、融合权重或增加 rescue。

## 8. 代码与结果索引

核心实现：

- `src/sprint20/heads.py`：原生 OTM 切换、共享头捕获、原生解码/NMS、缓存一致性；
- `src/sprint20/runtime.py`：在现有 submission runtime 中最小接入；
- `src/sprint20/policy.py`：融合后互斥类别接管；
- `src/sprint20/bounded_d4.py`：有证明的 D4 提前 KEEP；
- `src/sprint20/evaluation.py`：正式 matcher/计分复用、来源组计数和配对 bootstrap；
- `src/sprint20/cli.py`：资产审计、head probe、parity、replay、D4 AB/BA；
- `src/sprint20/run_submission.py`：实验入口，不替换原提交入口；
- `scripts/build_sprint20_p40_oof_probe.py`：构造来源隔离的 P40 短 OOF 探针；
- `scripts/aggregate_sprint20_head_caches.py`：三折缓存聚合与唯一覆盖核验；
- `scripts/analyze_sprint20_head_probe.py`：固定阈值、细类计数、同 FP 诊断；
- `scripts/analyze_sprint20_oof_routing.py`：cross-fit 路由与来源组 bootstrap；
- `scripts/select_sprint20_d4_hard_cases.py`：冻结 D4 硬例集合；
- `scripts/build_sprint20_evidence_bundle.py`：由冻结 SHA、CV3 来源组和历史资产生成只读证据包；
- `configs/experiments/hera_sprint20_p40_d4_otm_ship23_candidate_v1.json`：明确未准入的研究候选；
- `configs/experiments/hera_sprint20_policy_template.json`：防止未冻结政策误用的模板。

方案21证据包：

```text
reports/audits/HERA_SPRINT20_EVIDENCE_20260905/
├── actual_reference_assets.json
├── training_and_selection_lineage.json
├── selection_history.md
├── test_case_coverage.json
└── evidence_limitations.md
```

通用只读审计器位于 `tools/evaluation_evidence_audit/`。对真实 lineage manifest 的运行结果为
3 个严格独立性声明错误、5 个预期限制警告；三个 fold 的间接接触路径均被正确检出。
审计器的退出码 `2` 表示声明与证据冲突，不是模型或审计器崩溃。
下载原包 SHA 为 `1eadea68c6445f00b164b23a421c8359fff45be5189afac416ab54430fe583b3`；
包内 `SHA256SUMS.txt` 全部通过，工具源码、合成示例、原始测试记录和两份边界说明均原样保留。

服务器执行手册见 `docs/server/HERA_GUARD_SPRINT20_FINAL_READOUT.md`。

小型证据保存在本地（被 `.gitignore` 排除，不上传权重或完整预测）：

```text
outputs/HERA-SPRINT20-20260905/
├── asset_audit.json
├── native26/
├── full_seen/
├── p40_short_oof/
├── d4_hard_cases100/
├── d4_consistency_hard100/
└── SHA256SUMS.txt
```

## 9. 验证结果

- 新增代码 scoped Ruff：通过；
- Sprint20 与证据构建本地测试：**53 passed、3 skipped**（3 项为本机无 PyTorch 的可选集成测试）；
- 外部通用证据审计器：**46 passed、0 skipped**；
- 最终服务器完整仓库测试：**1231 passed、3 skipped**；
- 服务器真实 P40 checkpoint、原生 OTO/OTM、shared CUDA 路径、真实连续图和完整 4,481 图均已执行；
- 本机全仓测试因本地轻量环境没有 PyTorch，在收集 8 个 GPU/torch 模块时中止；这不是 Sprint20 测试失败，完整依赖环境的服务器全仓结果已覆盖。

## 10. 决策字段

```text
sprint20_execution = complete
p40_otm_mechanism_signal = positive_for_ship_common_2_3
p40_otm_evidence_role = posthoc_development_outer_oof_short
strict_nested_independence = false
historical_v2_platform_digest_attestation = unknown
fsc_otm_admission = false
shared_oto_parity = true
shared_otm_parity = false
shared_head_deployment_admission = false
bounded_d4_exactness = true
bounded_d4_speed_admission = false
safe_candidate = frozen_p40_path_plus_original_aircraft_d4_only
attack_candidate_admission = false
official_submission_or_packaging_performed = false
```
