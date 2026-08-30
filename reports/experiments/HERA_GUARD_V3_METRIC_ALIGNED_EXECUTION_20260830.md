# HERA-Guard V3：指标对齐对象验证执行报告（2026-08-30）

## 1. 本轮问题与实验边界

本轮严格落实项目根目录 `改进方案9.md`，但不把其中每个建议都无条件投入长训练。
进入本轮前已有三条决定性证据：

1. 当前四源 + tight crop + pixel OER 在 trial-mix 伪 10K 的正式外层 CV3 中，
   FDR≈0.15 时 Recall 约为 0.862；距离 0.94 不是阈值微调能够弥补的差距。
2. 旧 HERA PAV V1/V2 与 MAR 已在正式 OOF 中做过：PAV 最好仅约 +0.14pp，
   MAR 代理仅约 +0.04pp 且损伤六指标最小值。因此本轮不重跑旧 PAV/MAR。
3. natural retrain、background-complete、3 粗类检测器、Y3、粗类 NMS 等路线已有负向
   消融结论。新实验必须先证明提供了不同信息，不能换名重复。

因此执行顺序冻结为：

`E0 官方一目标一胜者标签审计 → E1 粗类污染清理 → E2 指标残差风险头 → E3 七通道新像素证据 → E4 VOI 稀疏路由`。

E5 对象簇解析、E6 稀疏重裁只在 E3/E4 提供正证据后进入；E7 已有负向同类实验，
不重复；E8 D-FINE 属候选生成器替换，必须等验证器链仍无法越过候选/排序瓶颈后再启动。

## 2. 冻结数据与基线

- 开发难集：`CV3-OOF-PSEUDO10K-TRIAL-MIX-V1`，2158 GT，6 张 10K 图，三折外层 CV3。
- 候选证据：`COARSE-BINARY-TIGHT-EVAL-TRIAL-V1/coarse_predictions.json`，46566 条。
- 分数锚点：同目录 `pixel_oer/identity_predictions.json`。
- 官方匹配：prediction-first、类别正确、逐图一对一、一目标一胜者；粗类 IoU 阈值沿用
  `configs/project.yaml`。
- 0.001 阈值网格基线（FDR 0.15 目标）：TP=1863、FP=322、Recall=0.863299、
  FDR=0.147368。
- 该伪集是部署难度代理，不是独立隐藏测试；所有结论仍须来源互斥 sentinel 验证。

## 3. E0：官方一目标一胜者角色审计

实现：

- `src/rsdet/hera_guard/metric_aligned.py`
- `scripts/audit_metric_aligned_pseudo_labels.py`
- `tests/test_metric_aligned_risk.py`

46566 条候选被互斥拆为：

| 角色 | 数量 | 含义 |
|---|---:|---|
| canonical_positive | 2100 | 当前全候选排序下由官方匹配唯一接收的候选 |
| duplicate_negative | 533 | 同细类、同对象，但输掉一对一匹配 |
| cross_fine_negative | 2477 | 几何覆盖对象且粗类相同，但细类错误 |
| cross_coarse_negative | 381 | 几何覆盖对象但粗类错误 |
| localization_negative | 10072 | 与对象相交但未达到粗类 IoU 门槛 |
| background_negative | 31003 | 最大 IoU≤0.05 的清晰背景候选 |

关键解释：候选全集能匹配 2100/2158 个 GT（上限 Recall 0.9731），而 FDR≈0.15
工作点仅保留 1863 TP。主要矛盾是高召回条件下对 4.4 万负候选的排序和拒识，
而不是候选完全不存在；但剩余 58 个 GT 仍属于候选生成/定位上限，验证器不能解决。

## 4. E1：粗类前景污染修复

实现：

- `scripts/build_clean_coarse_manifest.py`
- 服务器产物：`HERA-GUARD-V3-E1/clean_foreground_manifest.csv`
- 服务器摘要：`HERA-GUARD-V3-E1/clean_manifest_summary.json`

在旧 coarse-binary 训练清单 36494 行中，发现 381 条“预测粗类与 support GT 粗类不同，
却被标成 foreground”的样本，数量与 E0 的 cross-coarse 角色完全一致。新合同将其改为
对应预测粗类的 hard negative，不删除像素样本，不查看 held-out 阈值。

三折 × 三粗类干净重训已排队；它复用同一 ImageNet 权重、同一 224 tight crop、同一
3 epoch/40 batch 合同，唯一变量是这 381 个标签。当前 `score_sqrt` 单因素实验先占用
server1，E1 会在它结束后自动开始，不与其争抢 GPU。

## 5. E2：受限指标残差风险头

实现：

- `src/rsdet/hera_guard/metric_risk.py`
- `scripts/train_metric_aligned_risk.py`
- `scripts/compare_metric_risk_stages.py`
- `scripts/server/run_hera_guard_v3_metric_aligned.sh`
- `tests/test_metric_risk_model.py`
- `tests/test_compare_metric_risk_stages.py`

模型不是替换 OER，而是在 incumbent logit 上增加最大 ±2.5 的有界残差。每个 held-out
fold 的归一化、训练阈值、模型参数均只来自另两个 fold。累计阶段固定为 BCE、RankNet、
soft-FDR、one-winner，不扫描融合权重。

### 5.1 默认 0.005 网格

| 阶段 | TP | FP | Recall | FDR | 相对基线 |
|---|---:|---:|---:|---:|---:|
| 基线 | 1861 | 320 | 0.862373 | 0.146722 | — |
| BCE | 1865 | 320 | 0.864226 | 0.146453 | +4 TP / +0 FP |
| RankNet | 1861 | 316 | 0.862373 | 0.145154 | +0 TP / -4 FP |
| soft-FDR | 1859 | 325 | 0.861446 | 0.148810 | -2 TP / +5 FP |
| one-winner | 1861 | 321 | 0.862373 | 0.147113 | +0 TP / +1 FP |

### 5.2 0.001 网格复核

- 基线：TP=1863、FP=322、Recall=0.863299、FDR=0.147368。
- BCE：TP=1866、FP=326、Recall=0.864690、FDR=0.148723。
- BCE 只增加 3 TP，同时增加 4 FP；船召回不变，飞机 -0.098pp，车辆 +2.174pp。

结论：表格风险头有真实但很小的车辆方向信号，未达到预注册 +0.5pp 且任一粗类
下降不超过 0.5pp 的准入门槛。RankNet/soft-FDR/one-winner 没有独立收益并开始损伤船舶。
停止继续扫描损失权重，转入真正的新像素证据。

## 6. 来源互斥四源 sentinel

此前 sentinel 的前三源已经完成，COPH 因配置残留 `/workspace/.../p03-links` 绝对路径而
失败。本轮只补齐同 SHA 权重链接并从 COPH 阶段续跑，没有重跑前三源、没有修改模型参数。

- 600 个 sentinel 来源与开发来源互斥；1969 GT。
- 四源输入 69779，class-aware NMS 后 47576。
- 候选 oracle：1933 个 GT 有正确细类候选（0.98172）；6 个细类失败，30 个定位失败。
- sentinel 自身外层 CV3 @FDR≈0.15：TP=1685、FP=293、Recall=0.855764、FDR=0.148129。
- 冻结开发阈值直接迁移：TP=1675、FP=297、Recall=0.850686、FDR=0.150609。

开发集 0.863 与来源互斥 0.851–0.856 的差距约 0.8–1.3pp，远小于此前担心的完全失真，
但也说明开发阈值不能被当作隐藏集精确最优阈值。两个集合共同指向相同瓶颈：飞机接近
饱和，船和车辆的低 FDR 排序不足；候选上限很高，排序/拒识仍是第一优先级。

## 7. E3：七通道新像素证据

实现：

- `src/rsdet/hera_guard/dual_view.py`
- `scripts/train_dual_view_metric_verifier.py`
- `scripts/server/run_dual_view_metric_v1.sh`
- `tests/test_hera_dual_view_voi.py`

输入固定为：

1. 3 通道 tight RGB；
2. 3 通道 1.75× context，但 proposal 核心像素由邻域 ring median 替换；
3. 1 通道 proposal mask。

越界区域先对局部 patch 做 reflection padding，避免对整张 10K 图复制/填黑。ConvNeXt-T
首层由 3 通道扩为 7 通道；tight/context 各继承 0.5 倍 ImageNet stem 权重，mask 权重
初始化为 0，因此 tight=context 时初始响应与原预训练 stem 一致。

模型同时学习 foreground、25 细类、IoU quality 和 canonical residual 四个头，最终分数仍是
OER anchor logit + ±2.5 有界像素残差。三折训练严格排除 held-out fold；固定 3 epoch、
100 batch/epoch、D4 遥感增强，不做参数扫描。server2 已启动正式运行。

## 8. E4：VOI 稀疏验证

实现：

- `src/rsdet/hera_guard/voi.py`
- `scripts/apply_voi_dual_scores.py`
- `tests/test_apply_voi_dual_scores.py`

VOI 仅使用部署可得信息：距冻结决策阈值的距离、分数熵、检测/前景分歧、小目标程度和
图像边界风险。预算固定为每张 10K 图 32/64/128/256 个候选，这是延迟—精度曲线而非
融合权重搜索。未入选对象完全保留 incumbent 分数；入选对象才使用 E3 分数。

E4 已在 server2 排队，E3 完成后自动执行四个预算和官方外层 CV3；即使 E3 全量应用
略有损失，VOI 仍可判断局部使用能否保留车辆收益而避免船舶受损。

## 9. E5/E6/E7/E8 的当前决定

- E5：已有 coarse-NMS/对象唯一化负向消融。只有 E3 提供可可靠区分同簇候选的新像素
  证据后才重开，避免重复做“换 IoU 的 NMS”。
- E6：`src/rsdet/hera_guard/voi.py::recenter_windows` 已实现中心 + 四方向稀疏平移窗口；
  只在 E4 证明高 VOI 子集有收益后接入模型推理，否则不制造额外时延。
- E7：已有 background-complete V1 正式负向结果，本轮 E1 已直接修复明确的 381 条标签
  污染，不再重新生成一套大背景数据。
- E8：D-FINE 是检测器替换，不与当前验证器实验同时改变。四源候选 oracle 已达 98.17%，
  当前没有证据支持优先投入完整三折长训练；若 E3/E4 仍失败，再以单折快筛进入。

## 10. 服务器状态与产物索引

### server1 (`cv3-seetacloud`)

- 运行：`COARSE-BINARY-HARDSCORE-TIGHT-TRIAL-V1`。
- 等待队列：`HERA-GUARD-V3-E1-CLEAN-COARSE-V1`。
- 干净 manifest：`/workspace/results/HERA-GUARD-V3-E1/`。

### server2 (`cv3-seetacloud-2`)

- 完成：`HERA-GUARD-V3-E0`。
- 完成：`HERA-GUARD-V3-METRIC-RISK-V1`。
- 完成：`FOUR-SOURCE-SENTINEL-V1`。
- 运行：`HERA-GUARD-V3-E3-DUAL-VIEW-V1`。
- 等待：`HERA-GUARD-V3-E4-VOI-V1`。

所有任务均使用独立目录、`status.txt`、输入/输出 SHA、外层折合同；未启动官方提交，
未改变当前 Docker 正式方案。

## 11. 当前最重要的判断

方案9的核心诊断得到支持，但“更复杂的损失”本身不是答案：在没有新像素信息时，
指标对齐损失最多只挪回 3–4 个 TP。真正值得继续的是 E3/E4；它们分别回答：

1. 遮挡上下文与显式 proposal mask 是否提供了旧 tight crop 没有的背景/边界证据；
2. 这种重证据是否只需作用于每图几十到几百个临界对象，就能在可控时延内改善
   船/车辆 Recall–FDR。

下一次决策只依据 E1、E3、E4 的官方外层 CV3 和来源互斥 sentinel，不依据训练 loss、
单折最好值或 pooled oracle。

## 12. E3 正式 4,481 图 CV3 复验：否决

为避免只在六张 trial-mix 开发图上判断七通道模型，补建了严格的正式 CV3 输入：

- 4,481 张官方图像；
- 20,933 个 GT；
- 65,301 条候选；
- 每个 held-out fold 的候选只使用对应 OOF 检测结果；
- tight/context 像素只从该对象原始官方图像读取；
- 三折训练、归一化与阈值选择均不读取 held-out fold 标签。

为使实验可执行，新增了确定性 view precache；随后进一步修复了性能问题：缓存项保存
源图的 `uint8 numpy` 数组，而不是在每个候选上重复执行整图 `PIL→numpy` 转换。该修复只
改变渲染开销，不改变任何像素、标签、参数或阈值合同。

正式 0.001 网格、FDR≈0.15 的结果如下：

| 方法 | TP | FP | Recall | FDR |
|---|---:|---:|---:|---:|
| corrected OER 基线 | 19,835 | 3,530 | 0.947547 | 0.151081 |
| 七通道 dual-view | 19,566 | 3,590 | 0.934696 | 0.155035 |

dual-view 相对基线少 269 TP、多 60 FP，Recall 下降 1.285pp；粗类 Recall 变化为：
舰船 -6.600pp、飞机 -0.235pp、车辆 -12.438pp。训练 loss 正常收敛，因此这是科学负向
结果而不是运行故障。正式准入门禁失败，禁止全量拟合或进入 Docker。

本地产物：`outputs/HERA-GUARD-V3-FORMAL-DUAL-VIEW-V1/`；回传包 SHA256：
`e2d6bde64f23fc20a5deff41b55d31d8edb8ada0d64d376ecbef9f494d7f5936`。

## 13. 全量 Y5-S 部署回看：通过，但不是独立泛化分数

全量 Y5-S 已按冻结合同使用全部 4,481 张官方训练图、YOLO26-s 初始化、1024 输入、
RandomRotate90、160 fixed epochs 完成拟合；固定使用净化后的 `last.pt`，没有验证集选模。
权重 SHA256：
`f7e30fac3391d7048314d1c41df0e878a4d5f0423e5c55b68ee7df5917418229`。

在 trial-mix 上按同一 safe-1024 推理合同进行**部署回看**：

| 风险点 | Recall | FDR | TP / FP / FN |
|---|---:|---:|---:|
| FDR≈0.10 | 0.948100 | 0.099075 | 2046 / 225 / 112 |
| FDR≈0.12 | 0.955514 | 0.113500 | 2062 / 264 / 96 |
| FDR≈0.15 | 0.961075 | 0.145095 | 2074 / 352 / 84 |

FDR≈0.15 的三折选择阈值为 0.153/0.148/0.148，说明统一部署阈值 0.15 是合理、稳定的
工程工作点。候选地板 Recall 为 0.980074，主要剩余漏检来自舰船。

必须明确：trial-mix 的 600 张来源图来自官方训练集，而全量权重已经见过全部官方训练
来源。因此上述 96.11% 只能用于检查“全量训练是否工作、分数尺度是否稳定、Docker 阈值
应落在哪里”，不能作为独立泛化成绩，也不能覆盖此前 fold-heldout 和 source-disjoint
实验。最终方法选择仍由 OOF 证据决定，全量训练只承担 CV 完成后的最终拟合。

当前同合同 Y5-L 全量权重正在做部署回看；只有它在 Recall/FDR 与运行时间上同时优于
Y5-S，才允许替换部署主权重，否则冻结 Y5-S + 全局阈值 0.15。

## 14. 收尾比较：Y5-L、两视图与粗类风险对齐

### 14.1 Y5-L 容量对照：否决

Y5-L 已完成全部 4,481 图、160 fixed epochs；固定 `last.pt` SHA256 为
`5124b4070b8b847e8385aaafea69ccbaa227ce744525eeb440cb2beb88e2d348`。同一部署回看合同：

| 模型 | FDR≈0.10 Recall / FDR | FDR≈0.15 Recall / FDR |
|---|---:|---:|
| Y5-S | 0.948100 / 0.099075 | **0.961075 / 0.145095** |
| Y5-L | 0.936515 / 0.098573 | 0.955514 / 0.148989 |

L 在两个工作点均低于 S，并生成更多低分 FP；代理 10K 单图平均时延也由 S 的约
4.0–6.3 秒增加到 L 的约 5.6–8.4 秒。容量扩大没有收益，Y5-L 不进入镜像。

### 14.2 Y5-S identity+90° 两视图：部署候选

冻结同一 Y5-S 权重，只增加 `[0°, 90°]` 两个推理视图，并按细类内 NMS 合并；不训练
新参数。与单视图比较：

| 方法 | 候选地板 | FDR≈0.10 Recall / FDR | FDR≈0.15 Recall / FDR |
|---|---:|---:|---:|
| 单视图 | 0.980074 | 0.948100 / 0.099075 | 0.961075 / 0.145095 |
| 两视图 | **0.981928** | **0.959222 / 0.100391** | **0.966636 / 0.144381** |

两视图在两个风险点均正向，FDR≈0.15 增加 12 TP、FP 不变；代价是推理约翻倍，但仍在
项目冻结的 20 秒上限内。

官方页面分别显示三大类 Recall/FDR，因此新增了只用于 `fusion=safe` 的
`score_threshold_by_coarse` 部署合同；完整校验要求恰好覆盖 ship/aircraft/vehicle，
每个阈值在 `[0,1]`，并在融合前按候选细类所属粗类过滤。相关单测覆盖配置拒绝、粗类
映射和过滤时序。

两视图三折阈值的中位数固定为 ship=0.371、aircraft=0.301、vehicle=0.366；不再扫描。
固定阈值回看得到：总体 Recall=0.958295、FDR=0.099303；舰船 0.918239/0.145366，
飞机 0.999020/0.048553，车辆 0.940217/0.135000。该工作点比统一阈值少部分 TP，但避免
统一阈值下舰船和车辆 FDR 超过 20%，更贴近官方分粗类展示与综合计分风险。

fold-heldout 两视图复验已经完成。与 identity OOF 的粗类/global 原始分数基线约
`Recall=0.7099, FDR=0.1503` 相比，两视图在同一困难 trial-mix OOF 上得到：

| OOF 两视图工作点 | Recall | FDR | 说明 |
|---|---:|---:|---|
| global，FDR≈0.15 | 0.744208 | 0.148462 | 相对 identity 方向约 +3.43pp |
| coarse cross-fit，FDR≈0.15 | 0.733086 | 0.153105 | 舰/机/车阈值折间波动大 |

两视图候选地板由 identity 的约 0.9263 提高到 0.941149，说明它确实补回了一批旋转敏感的
候选；但分数校准仍然存在明显域差异：OOF global 阈值约为 0.594，而全量同源回看阈值
约为 0.214；coarse cross-fit 中位数为 ship=0.656、aircraft=0.476、vehicle=0.741，
也显著高于全量同源回看的 0.371/0.301/0.366。尤其车辆在 coarse cross-fit 下仅
Recall=0.2065、FDR=0.2400，说明把高 OOF 阈值直接写入部署会严重损失车辆召回。

因此科学门禁的结论是：**两视图方向准入，低阈值校准不准入为“已验证泛化阈值”**。
正式部署保留两个同权重候选：

1. 主回退镜像：Y5-S 单视图、统一阈值 0.15；
2. 高收益候选镜像：同一 Y5-S 权重、identity+90° 两视图、冻结粗类阈值
   0.371/0.301/0.366。

第二个候选同时包含视图增广和冻结的分粗类工作点，因此官方 A/B 只能回答“整套部署策略
是否更好”，不能把差异完全归因于视图或阈值中的某一个。它只能通过一次官方提交与主
回退镜像做域校准；不能用同源 95.83% 召回
宣称隐藏集泛化。这里的 coarse 指标仍是三大类 pooled 风险代理，并非 V1.6 的粗类内
细类宏平均排名本身。两个候选共用唯一全量权重，避免把模型差异、阈值差异和视图差异
混在同一次比较中。

### 14.3 E3/E4 开发代理收尾：均否决

E3 在 trial-mix 的 FDR≈0.15 结果为 Recall=0.804449、FDR=0.162162，相对基线少 127 TP、
多 14 FP；与正式 CV3 的负向方向一致。E4 即使每图仅替换 32 个最高 VOI 候选，仍少
12 TP；64/128/256 预算分别少 13/15/27 TP。E3/E4 均停止，不进入全量训练和 Docker。
