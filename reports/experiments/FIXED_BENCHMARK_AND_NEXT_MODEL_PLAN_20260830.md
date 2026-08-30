# 固定双基准与下一阶段模型路线（2026-08-30）

## 1. 为什么只保留两套测评

此前多个同源 OOF 结果达到 94% 以上，但官方预测评只有约 80.7% 飞机、84.6% 船舶、
51.0% 车辆 Recall，说明“正常域同源泛化”和“超大图复杂背景泛化”不是同一难度。
后续不再为每个新方法临时选择测试集，而固定为以下两套。

### A. Normal-CV3

- 4481 张官方训练图、20933 个目标；
- 机场代理组和来源组隔离的三折 OOF；
- 阈值只能由另两折拟合；
- 用途：防止创新破坏正常目标识别、细类分类和常规定位能力。

### B. Hard10K-CV3

- 冻结的 6 张 pseudo-10K、2158 个 GT、600 个唯一来源；
- 三折 OOF、官方一对一细类匹配；
- 开发集之外再使用冻结的 source-disjoint sentinel 做方向复验；
- 用途：测量复杂背景、尺度变化、候选排序和虚警控制，作为接近官方隐藏集的压力测试。

两套基准都固定输出 FDR=0.10/0.12/0.15/0.20 下的 pooled Recall/FDR、官方
25 细类宏平均、三粗类宏平均、逐细类 TP/FP/FN、候选上限、折间阈值和折间波动。
匹配唯一实现为 `src/rsdet/evaluation/official_metric.py`。

## 2. 当前可信起点

| 协议 | 当前可信结果 | 解释 |
|---|---:|---|
| Normal-CV3 | Recall 约 0.9475 / FDR 约 0.1511 | 正常域能力较强 |
| Hard10K-CV3 四源候选上限 | Recall 约 0.9731 | 候选仍有提升空间但不是主瓶颈 |
| Hard10K-CV3 四源固定风险 | Recall 约 0.8601 / FDR 约 0.148 | 背景排序是主瓶颈 |
| source-disjoint sentinel | Recall 约 0.8507 / FDR 约 0.1506 | 开发代理没有明显虚高，但任务确实更难 |

困难域粗类短板集中在 ship 和 vehicle；aircraft 已明显更高。因此，再做全局统一阈值、
简单 NMS、SAHI、纯背景续训、通用 DINO/CleanDIFT 拼接，信息价值已经很低。

## 3. 固定准入规则

候选先完成短筛，再完成双基准正式验证。正式准入需要同时满足：

1. Normal-CV3 Recall 相对冻结基线下降不超过 0.3pp；
2. 任一粗类 Recall 下降不超过 0.5pp；
3. Hard10K-CV3 在 FDR≤0.15 时 Recall 至少提升 0.5pp；
4. source-disjoint sentinel 方向一致，不能只在开发代理集增益；
5. 输出官方 V1.6 的 7 项排名指标，不用 pooled Overall 代替官方排名；
6. 声称“内部 94 对应官方至少 92”前，Normal 与 Hard/Sentinel 的差距必须压到 2pp 内。

## 4. 第一优先级：层级细类校准

旧 `tune_fine_thresholds.py` 在同一数据上选阈值并评估，只能作为乐观诊断。新脚本
`scripts/analyze_cv3_oof_hierarchical_thresholds.py` 对每个留出折只使用另两折：先拟合
全局阈值与三粗类锚点，再为 25 个细类拟合阈值，并按该细类训练 GT 数量在 logit
空间向粗类锚点收缩。尾类证据不足时自动退回粗类阈值。

这项实验不改变候选框和权重，专门检验官方细类宏平均是否能在不破坏 pooled 门槛的
条件下提升。只有 Normal 与 Hard 两套协议都同方向才进入 Docker。

## 5. 第二优先级：困难背景回放微调

此前 background-complete 训练把困难背景直接加入训练集并短训 20 epoch，候选 Recall
从约 0.973 降到 0.868，属于明显灾难性遗忘。新实验不重复该合同：

- 从已完成的 Y5 折模型开始；
- 每个 epoch 保留全部/高比例原始正样本回放；
- 困难背景只占固定 10% 或 15%，且来源严格来自训练折；
- 低学习率、5--8 epoch、冻结 backbone 前段，只调整检测头与高层特征；
- 同时监控 Normal 留出折候选 Recall 与 Hard10K 背景 FDR，任一正常域候选下降超过
  0.3pp 立即停止。

先做 fold0 单因素快筛；通过后才扩展三折。

## 6. 第三优先级：Y5 + M3 异构候选

服务器已有完整的 M3 RT-DETR-L 三折权重与全量权重。Hard10K 上 M3 单模型候选上限
约 0.9235，Y5+M3 与旋转/尺度视图的联合候选上限接近 0.97。下一步不是重新训练
M3，而是在固定双基准下重新建立可迁移的分数校准和去重：

- 每个模型先做各自折外单调校准，禁止直接比较原始置信度；
- 坐标聚类后保留模型身份、视图一致性和 coarse/fine 冲突特征；
- 只训练轻量 pairwise/listwise 排序器，标签严格来自训练折；
- 若不能同时超过单 Y5 与单 M3，则停止，不把双模型成本带入 Docker。

## 7. 暂停方向

- SAHI/P2 与简单多尺度补漏：已有负向结果；
- background-only / background-complete 再训练：已有候选崩塌证据；
- 单一全局/粗类阈值网格：上限不足；
- DINO/CleanDIFT 直接拼接：正式增益不足；
- 新建第三套代理集：会继续扩大测评口径漂移。

配置锁：`configs/experiments/fixed_benchmark_v1.yaml`。

## 8. trial-v2.0 官方回传对双基准的校准

2026-08-30 的稳定单视图 Y5-S 全量模型得到综合分 86.2274、暂列第 33：

| 大类 | Recall | FDR |
|---|---:|---:|
| ship | 0.942287 | 0.126937 |
| aircraft | 0.999246 | 0.024300 |
| vehicle | 0.946309 | 0.237838 |

平均推理时间为 2.704833 秒。与 trial-v1.0 的 67.0171 相比，代码、权重与提交链修复
已经产生跨档提升。该结果对内部测评的含义是：

1. Normal-CV3 对“正常目标 Recall 已接近 94%”的判断成立；
2. Hard10K 的绝对 Recall 明显偏悲观，不能直接当作官方分数预测器；
3. Hard10K 对风险排序仍然有效：飞机最强，车辆 FDR 是当前唯一超过 20% 的官方项，
   船舶 Recall/FDR 次之；
4. 因此双基准定位为“Normal 防退化 + Hard 压力筛选”，不再要求 Hard 的绝对值与
   官方一一相等；新方法必须在 Hard 上方向正向，再由 Normal 保证不破坏主能力。

按 V1.6 真正排名口径（大类内细类 Recall/FDR 简单平均）重算同一全量模型的
`trial-mix` 部署审计，与官方 trial-v2 对照如下：

| 粗类 | 内部宏平均 R/FDR | 官方 R/FDR | 官方-内部 |
|---|---:|---:|---:|
| ship | 0.948936 / 0.142899 | 0.942287 / 0.126937 | -0.66pp / -1.60pp |
| aircraft | 0.997373 / 0.078783 | 0.999246 / 0.024300 | +0.19pp / -5.45pp |
| vehicle | 0.945652 / 0.236842 | 0.946309 / 0.237838 | +0.07pp / +0.10pp |

这说明“全量权重 + 精确部署配置 + trial-mix + V1.6 macro”可作为提交前的唯一
部署预测器：三类 Recall 与 vehicle FDR 已经高度对齐官方；ship/aircraft FDR
偏悲观，因此只能作保守风险上界。它不取代 CV3 的方法选择，避免用同源回看
选训练方法。后续测评口径固定为两层：

1. 方法选择：Normal-CV3 + Hard10K-CV3 双账本；
2. 提交前预测：全量部署审计，只运行冻结候选，不在该账本上反复扫参。

现有下一提交候选仍是同一全量权重的 identity+90° 双视图及冻结粗类阈值。内部回看
显示它把 vehicle FDR 从高风险区压到约 0.135，同时保留约 0.940 的 vehicle Recall，
且 RTX 3090 真实 10K smoke 约 14 秒。该候选在允许再次提交时优先于未经双基准准入的
新训练模型。

## 9. 层级细类校准实测：否决

两个固定协议均完成严格 cross-fit：

| 协议 | global @ FDR≈0.15 | hierarchical @ FDR≈0.15 | 结论 |
|---|---:|---:|---|
| Normal-CV3 | 0.947260 / 0.150064 | 0.929728 / 0.130190 | Recall −1.75pp |
| Hard10K-CV3 | 0.546339 / 0.150576 | 0.540315 / 0.150765 | Recall 与 macro 均下降 |

Normal 的官方 25 类 macro Recall 从 0.910099 降到 0.876519；最大粗类 macro Recall
下降约 23.63pp。该方法确实降低部分 FDR，但牺牲过多 Recall，两个固定基准方向一致为
负，因此直接停止，不继续扫描 prior strength、最小样本数或细类阈值。

原始结果：

- `outputs/FIXED-BENCHMARK-V1/hierarchical_normal_cv3.json`；
- `outputs/FIXED-BENCHMARK-V1/hierarchical_hard10k_cv3.json`。

## 10. 困难背景回放实现

新增 `scripts/train_y5_hard_replay.py`，与旧 background-complete 的差异冻结如下：

- 每折全部 2868--3120 张官方正样本都保留；
- 640 张困难 tile 中按四个来源图均衡选择 320 张，约占训练图 9%--10%；
- 从正式 Y5 fold checkpoint 开始，仅训 6 epoch；
- `lr0=5e-5`，冻结前 10 层，mosaic 降到 0.2；
- 不用验证集选 epoch，固定取 last；
- 先只做 fold0 替换的配对快筛，通过后才扩展三折。

该合同的目的不是用训练集扫描比例，而是验证“保留正样本知识的低强度检测头适配”能否
修复旧 background-complete 的灾难性遗忘。若候选上限仍下降超过 0.3pp，路线立即停止。

fold0 配对快筛已完成，结果为明确负向：

| 指标 | candidate - baseline |
|---|---:|
| candidate-floor Recall | -0.071826 |
| Recall @ FDR≈0.15 | -0.007878 |
| FDR @ 相同风险带 | +0.003618 |
| ship / aircraft / vehicle Recall drop | 0.001048 / 0.012745 / 0.016304 |

候选形成上限下降 7.18pp，且固定风险下 Recall/FDR 同时变差；这不是只有某个
粗类的局部退化。`screen_passed=false`，因此不扩展三折，不扫描回放比例、学习率
或训练轮数。原始证据位于
`outputs/Y5-HARD-REPLAY-FOLD0-SCREEN-V1/`。

执行时发现旧 cv3 环境的 NumPy 1.26.4 无法反序列化该权重内由 NumPy 2.2.6
写入的 RNG 对象；失败发生在训练前。保留失败现场后，改用能通过原权重加载门禁的
既有 p06-cu121 环境从零执行同一冻结合同，6 epoch 完整成功；这是技术恢复，未改实验参数。

## 11. trial-v2 暴露风险后的最小部署修订

trial-v2 在统一 0.15 阈值下已得到 ship 0.942287/0.126937 和 aircraft
0.999246/0.024300，两项都没有支持继续提高 ship 阈值的官方证据；vehicle
0.946309/0.237838 则是唯一超过 20% FDR 的粗类。因此在旧 B 候选之外冻结
一个更保守的官方风险校准候选：

```text
submission/docker/configs/y5_full_s_safe_1024_rot90cwtta_trial_v2_calibrated_v1.json
identity + 90°
ship=0.150 / aircraft=0.301 / vehicle=0.366
```

该修订不改权重、切片、NMS 或融合逻辑，不扫描新阈值；只保留已证实的双视图
补候选，并针对官方暴露的 vehicle 虚警与 aircraft 大量余量应用预注册粗类
阈值。其内部同源回看为 Recall/FDR 0.965709/0.150774；ship 保留 0.935010
Recall，aircraft 为 0.999020/0.048553，vehicle 为 0.940217/0.135000。这些数值
仅用于检查修订方向，不当作隐藏集估计。

官方手工转录数据固化于
`outputs/OFFICIAL-TRIAL-CALIBRATION-20260830/trial_v1_v2_metrics.json`。
