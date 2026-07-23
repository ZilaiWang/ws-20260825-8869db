# MAR20 来源分组 TASK-01 回传审查与 01A 修订结论

## 1. 审查结论

TASK-01 的检索、局部证据计算和盲审材料在科学方向上有效，但本次回传不能直接作为正式完成结果。问题不是模型效果不足，而是原始执行链没有完整通过：上游决策文件的预注册 SHA 写错，且 29 对候选的退化 affine RANSAC 每对产生 1 个非有限矩阵和 2 个非有限误差字段，共 87 个 `NaN/Inf` 字段，使几何 verifier 按合同停止；之后存在的分析和盲审文件没有对应的一条连续成功日志。因此当前状态应记为：

```text
retrieval evidence: technically available
geometry evidence: computed but requires audited sanitization
blind review pack: internally valid but provisional
formal_grouping_admission: false
```

不能跳过这次修订后直接人工审核，也不能把 Q1 候选自动合并成 group。

## 2. 已确认有效的信息

### 2.1 正式检索

- 两条冻结路由均已按 K=50 完成检索，K=100 仅用于召回审计；
- 正式 target candidate edge 为 110,604 条，全体 K=50 edge 为 173,028 条；
- calibration 的联合召回为 R@20=0.9758、R@50=0.9866、R@100=0.9892；
- held-out audit 为 R@20=0.9839、R@50=1.0、R@100=1.0；
- 路由和 K 没有利用 held-out 重新选择，仍满足 protocol。

这些结果说明两条 masked DINOv2-B VLAD 路由足以把绝大多数人工正对带入后续局部验证，当前不需要增加 K 或改回全图特征。

### 2.2 局部证据队列

- 队列共 6,000 对，包含 600 个冻结控制对和 5,400 个新候选；
- target-target 3,955 对、target-bridge 1,796 对、bridge-bridge 249 对；
- patch overlap cache 覆盖 3,824/3,842 个节点；
- 6,000 对均已完成 SIFT/patch 证据计算；
- 退化只出现在 29 对的 affine 矩阵及其两个误差字段中，没有证据表明 DINO/VLAD、图像、SIFT cache 或队列损坏。

因此无需重跑 DINO/VLAD 特征、检索或 6,000 对几何。正确处理是把失败的 affine 模型记为“该模型证据缺失”，而不是把误差写成 0，也不是丢弃整对样本。

### 2.3 临时 calibration 读出

现有临时读出可用于修订后的一致性核验，但在 01A 通过前不作为正式产物：

| 项目 | calibration | held-out audit |
|---|---:|---:|
| pair 数 | 402 | 133 |
| 正对数 | 186 | 62 |
| precision | 0.9024 | 0.9118 |
| recall | 0.9946 | 1.0000 |
| AP | 0.9961 | 0.9961 |
| ROC AUC | 0.9967 | 0.9968 |

队列读出为 Q1=4,564、Q2=518、Q3=602、Q4=316。Q1 只表示优先人工检查，不表示可以自动连边。高指标说明局部证据排序有很强区分度，但 calibration 数量有限，且人工标签代表“同来源/同场景证据”而不是机场真值，所以仍保持保守人工确认。

### 2.4 临时盲审包

回传的盲审包内部结构完整：

- 300 个新候选；
- 48 个隐藏控制对；
- 28 个盲重复（服务器文字回报中的 27 是口径误写，JSON 和实际卡片均为 28）；
- 共 376 张匿名卡、94 张 contact sheet；
- manual template、private mapping 和图片 SHA 一致；
- 每张卡包含 A/B 原图、背景掩蔽视图和边缘视图，能够支持人工判断。

它目前只作为 01A 输出的预期参照。01A 必须从清洗后的正式证据重新生成同构盲审包。

## 3. 技术问题及修订策略

### 3.1 上游 SHA 写错

TASK-01 文档将 `TASK-00B2/task_decision.json` 的 SHA 写成错误值，正确值为：

```text
deea8b6831e7b9e03302abea3fd73eadb83816d02d3236b79e99736ca9f70334
```

原任务文档已经修正。该错误只影响门禁，不改变上游文件内容和科学结果。

### 3.2 OpenCV 退化拟合

29 对中 affine 矩阵拟合失败后，两个误差字段分别留下 29 个非有限值。采用以下统一语义：

```text
finite transform + finite errors + positive inliers -> model evidence valid
degenerate/nonfinite fit                         -> model evidence missing
```

清洗程序只清空非有限字段并追加各模型 `fit_valid` 标志，原始文件不覆盖。未来重新计算时，底层 geometry 函数也会在矩阵、投影或误差非有限时直接返回空模型证据，防止同类问题再次进入 CSV。

### 3.3 可审计性

01A 使用原始 `pair_evidence.csv` 的冻结 SHA：

```text
97920b081d1137e72817191ccca1dea90955045e460ace9af03cc7253efa2051
```

清洗后重新运行既有分析与盲审生成器，并要求 assignment SHA、指标、Q1～Q4 数量和盲审规模与临时读出完全一致。任何变化都停止，不允许临时放宽门禁或继续手工补文件。

## 4. 01A 之后的直接路径

1. 服务器执行 `MAR20_GROUPING_TASK_01A_SANITIZE_AND_RECOMPILE.md`；
2. 本地核验清洗报告、分析决策、376 张卡和 94 张 sheet；
3. 由独立审阅者完成匿名 pair review，先不读取 private mapping；
4. 编译重复一致率与隐藏控制 precision/recall；
5. 只将高置信、局部结构一致、人工通过的严格核心边加入图；
6. 对核心图做冲突审计、连通分量拆分和 group_id 输出；
7. 将每张 MAR20 图的 `group_id`、证据等级、是否 target/bridge 和不确定标记交付给 B，用于 CV3 和 train/val 分组。

最终 group 仍是“来源/场景代理分组”，不声称恢复了真实机场标签。低证据节点允许保持 singleton 或 unresolved；宁可少合并，也不能用不可靠边把不同机场串成大组件。
