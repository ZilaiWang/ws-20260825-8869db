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
