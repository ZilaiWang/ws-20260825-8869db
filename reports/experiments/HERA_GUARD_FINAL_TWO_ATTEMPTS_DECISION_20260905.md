# HERA Guard 最后两次提交候选决策

日期：2026-09-05

状态：第四次 `v4.0` 已完成官方测评并以 `77.1910` 成为当前最佳；第五次不得退回已构建但理论上更弱的 P40+D4-only，详见 `FORMAL_ATTEMPT4_SHARED_OTM_PLATFORM_RESULT_20260905.md`。

## 1 决策

原计划采用“第四次进攻、第五次保底”的非对称策略：

- 稳定线：冻结 P40 路径 + 原始 Aircraft-D4，Ship 和 FSC 仍由 P40 输出，Vehicle hierarchy 和 rescue 均撤销。
- 第四次候选：在稳定线上只增加共享前向 OTM 对 QHS/MS 两个 Ship 细类的互斥接管，阈值固定 0.560。
- 第五次候选：原条件是第四次未高于 P40 v2.0 时提交稳定线 P40+D4。第四次实际已高于 v2.0 `0.5900` 分，因此该条件不再成立，P40+D4-only 退出第五次冲分候选。

官方 v4.0 结果为 Ship `0.823344/0.060705`、Aircraft `0.950685/0.032303`、Vehicle `0.831579/0.210000`、时延 `3.978833s`。Shared OTM 与 Aircraft-D4 均获官方同向确认，Vehicle 与 v2.0 完全不变。第五次若无新的独立正证据，保留 v4.0 为最终最佳。

第四次只增加一个未知的官方迁移因素。Aircraft-D4 已由 v3.0 官方结果证明；OTM 只拥有 Ship 2/3；Vehicle 和 FSC 不改，因此结果能够清楚归因。

## 2 稳定线的证据

P40 v2.0 官方分数为 76.6010。Aircraft-D4 在 v3.0 中使 Aircraft：

- TP `4711→4732`，FP `168→147`，FN `231→210`；
- Recall `94.5967%→95.0685%`，增加 0.4718 个百分点；
- FDR `3.7265%→3.2303%`，降低 0.4962 个百分点。

这两项质量变化折合总分约 `+0.3215`。D4-only 的净收益取决于它相对 P40 的官方新增时延；既有 3090 工程测量支持其新增时延小于“2.2505 秒质量原始收益”的盈亏平衡点，但这仍是提交前条件估计，不是平台承诺。

## 3 D4 工程优化结论

本轮在同一 RTX 3090、同一权重和同一代理图上，将飞机分类批大小从 64 改为 128：

| 固定代理 | 输出 | batch64 均值 | batch128 均值 | 结果 |
|---|---|---:|---:|---|
| Hard | 2,039 框逐框完全一致 | 7.0721s | 6.9504s | 快 1.72% |
| Sentinel | 1,870 框逐框完全一致 | 6.7625s | 6.9258s | 慢 2.42% |

性能方向不稳定，提升量小于运行波动，故不修改冻结参数，继续使用 batch64。早先的 tensorized 变体和 bounded-D4 也没有获得稳定端到端提速；不在剩余机会前继续做 GPU 微优化搜索。

## 4 Shared OTM 的新增证据

历史 Sprint20 报告曾因 shared-OTM 与 native-OTM 在 4,481 图中的 61 图不满足精确一致而拒绝部署。该结论保持不变：shared 不能被称作 native 的精确加速实现。

本轮改变的是问题定义：将 shared 路径自身视为一个待验证候选，而不是要求它复现 native；直接用部署共享前向缓存重新完成来源隔离短 OOF。P40 固定阈值仍为 0.536，只允许 OTM 接管 QHS/MS 2/3，并在另外两折按“不得高于固定 P40 Ship macro FDR”选择阈值。

结果：

| fold | 选中 OTM 阈值 | 相对固定 P40 分差 |
|---:|---:|---:|
| 0 | 0.565 | +0.2847 |
| 1 | 0.565 | +0.1716 |
| 2 | 0.539 | +0.1808 |
| 合并 | 全 OOF 拟合约 0.560 | **+0.2137** |

255 个来源组、3,000 次 paired bootstrap 的有效重采样中，均值约 `+0.2223`、P10约 `+0.0565`、正增益比例约 95.35%。这是后验选定范围上的开发证据，不能解释为隐藏集提升概率。

完整 Ship 0--3 接管虽然合并分差约 `+1.1075`，但 fold1 为 `-3.1719`，来源组 bootstrap P10约 `-1.4721`、正增益比例约 68.82%，故拒绝。QHS/MS 是唯一保留的窄所有权范围。

## 5 成熟 full 权重机制复核

在 P40 full 已见训练图上的 shared 路径诊断中，固定 P40 0.536 与 OTM 0.560：

- 诊断分差 `+0.8735`；
- Ship macro Recall `89.0478%→91.2988%`；
- Ship macro FDR `2.2065%→2.1507%`；
- QHS 净增 41 TP、2 FP；MS 净增 52 TP、减少 2 FP。

这证明成熟 full checkpoint 的共享 OTM 分支仍有同方向能力，但由于图像已见，不能用于正式准入或官方分数预测。

## 6 最终整链复核结果

3090 整链以 P40+D4 为基线、P40+D4+shared OTM(QHS/MS) 为候选，在 Hard 与 Sentinel 各执行真实 competition runtime。冻结要求：

1. 标签 0/1、4--24 的预测多重集必须完全一致；
2. 只允许标签 2/3 改变；
3. 两套固定代理报告相对质量方向；
4. 报告共享前向相对 P40+D4 的新增时延，不用代理绝对分预测官方成绩；
5. 不扫描阈值、融合权重或类别范围。

结果：

| 固定代理 | Ship Recall | Ship FDR | 受保护标签 | 原共享候选增量时延 | 原实现分差 |
|---|---:|---:|---|---:|---:|
| Hard | `86.5002%→88.6350%` | `2.8880%→2.8480%` | 0/1、4--24 完全一致 | `+2.8075s` | `+0.4236` |
| Sentinel | `86.1479%→88.2833%` | `9.1387%→8.9714%` | 0/1、4--24 完全一致 | `+2.7002s` | `+0.4756` |

QHS/MS 在两套代理上都提高约 2.14 个百分点 Ship macro Recall，FDR 没有恶化。Aircraft 与 Vehicle 逐框不变，说明类别所有权实现正确。

随后实现只消除重复切片、复制和哈希的等价路径，仍对 OTO/OTM 分别做完整 Safe Fusion。预测 JSON 在两套代理上完全一致：

| 固定代理 | 旧共享实现 | 优化共享实现 | 提速 | 相对 P40+D4 增量时延 | 最终代理分差 |
|---|---:|---:|---:|---:|---:|
| Hard | 9.2785s | 7.4581s | 1.244× | `+0.5712s` | **`+0.7431`** |
| Sentinel | 9.1841s | 7.2505s | 1.267× | `+0.9455s` | **`+0.7262`** |

工程优化不改变任何预测、阈值、融合或类别所有权，因此准入第四次配置。执行入口分别为 `scripts/server/run_attempt4_shared_otm_runtime_3090_v1.sh`、`scripts/server/run_attempt4_shared_otm_optimized_3090_v1.sh` 与 `scripts/server/derive_attempt4_shared_otm_optimized_quality_v1.sh`。

## 7 其他方案的结论归并

| 方向 | 证据 | 决策 |
|---|---|---|
| P40 progressive S1280 | CV3正向且v2.0官方76.6010 | 保留基座 |
| Aircraft-D4 | v3.0官方 `+21 TP/-21 FP` | 保留 |
| Vehicle hierarchy/rescue | v3.0 Vehicle FDR +5.6055pp，双检测器慢3.4128s | 撤销 |
| Vehicle V96 / task vector | 外折不稳或负向 | 拒绝 |
| APRR / BATIS | 正式口径无稳健增益 | 拒绝 |
| APEX classifier / MacroExpert / M25 / DEIM | 来源稳定性或同长度对照失败 | 拒绝 |
| EXT-V / HAD / RFS / EQL / EFL | 三折或准入失败 | 拒绝 |
| 双检测器 / D-FINE / dual-view | 质量不稳且时延成本过高 | 拒绝 |
| shared OTM 全 Ship / FSC | 跨折或来源组不稳 | 拒绝 |
| shared OTM QHS/MS | 三折同向、成熟机制同向，尚无官方证据 | 第四次唯一攻击候选 |

## 8 风险与提交条件

OTM 的短 OOF 来自 `S1024/40e→P40/40e`，而正式 full 是 `S1024/160e→P40/40e`；QHS/MS 又是在比较多个所有权范围后选择。即使整链复核通过，也只能说它是当前最有依据的攻击候选，不能保证正式增分。

第四次准入条件已经满足：类别旁路完全正确、Hard/Sentinel 同向、工程优化逐框等价且新增时延低于 1 秒/图的两套代理测量。它仍可能因隐藏分布迁移失败，因此不能取代第五次保底。

最终身份：

- 第四次：`submission/docker/configs/p40_aircraft_d4_shared_otm_ship23_t0560_v1.json`；
- 第五次：`submission/docker/configs/p40_aircraft_d4_only_v1.json`；
- 统一物化/构建入口：`scripts/build_final_two_attempt_submissions.py`。

第五次固定为 P40+D4 batch64；不得从 v3.0 历史镜像改 tag，因为其中含已经失败的 Vehicle hierarchy。

本地构建身份：

- 第四次镜像：`xh-detector:hera-attempt4-20260905`，候选 ID `p40_d4_shared_otm_ship23_t0560_optimized_20260905`；
- 第五次镜像：`xh-detector:hera-attempt5-20260905`，候选 ID `p40_aircraft_d4_only_batch64_20260905`；
- 两个镜像均为 `linux/amd64`，包内 P40/D4 资产 SHA 与冻结配置一致；第四次额外验证 `sprint20.runtime` 可导入；
- 本机构建环境无 NVIDIA GPU，因此 GPU 端真实整链依据是前述 RTX 3090 源码同路径审计，镜像本机检查只覆盖封装、导入和资产身份。
