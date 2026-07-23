# MAR20 机场代理分组：00B1 人工复核结果

## 1. 结论

00B1 的两项人工门禁均已完成并通过本地编译验证：

- patch-mask：120/120 节点有效，三档膨胀均完整覆盖标注飞机，主设置 `dilation=0.15` 未发现不可接受的背景损失；
- 富集候选：264 张匿名卡片全部独立审核，包含 240 个唯一 pair 和 24 个盲重复；重复标签一致率为 1.0；
- 合并 Round-A 后得到 600 个标定 pair，其中严格正例 248 个，calibration 186 个、held-out 62 个；
- 推荐证据门槛已经满足，可以进入 Round-B 路由比较；
- 这一步只批准检索路由评估，不批准正式机场分组，`formal_grouping_admission` 仍为 `false`。

## 2. 盲评纪律与输入完整性

服务器回传包错误地包含了 `blind_card_mapping.csv` 和 `enriched_candidate_pairs.csv`。为避免人工判断被候选来源或重复关系污染，审核时执行了以下隔离：

1. 核验原始回传包 SHA256 为 `beb761cdcae9c446acc6d3016f5a713f09cf0c4e30a2188021ec5b07f883fd10`；
2. 重新解压时显式排除 mapping 和候选明细，仅查看匿名 contact sheets；
3. 完成全部判断后固定人工决策表 SHA256；
4. 仅在决策冻结后解封 mapping 并运行正式编译器。

人工文件冻结值：

- `manual_patch_mask_review.csv`: `adba9dca47494520da3b62ea1687a3db4bb9637d4ee19e0a0e331d51d87bac6a`
- `manual_enriched_decisions.csv`: `c664d1dedc8be26911ccf072bff5d918742e4a204c38b141dd0b06dc67a01eff`

审核者没有在人工决策冻结前查看 mapping。另以 contact-sheet 像素做无身份的重复检索，恰好发现 24 组重复，24/24 的标签和置信度一致；正式编译解封后同样得到 `repeat_agreement=1.0`。

## 3. Patch-mask 结果

人工逐图检查 30 张原分辨率 contact sheet、共 120 个节点。结果：

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| 有效节点率 | 1.000 | ≥0.95 |
| dilation 0.10 覆盖率 | 1.000 | ≥0.95 |
| dilation 0.15 覆盖率 | 1.000 | ≥0.95 |
| dilation 0.20 覆盖率 | 1.000 | ≥0.95 |
| dilation 0.15 过量背景损失率 | 0.000 | ≤0.10 |

00B 原始 summary 的 `automatic_geometry_gate=fail` 是因为低有效背景比例，不代表掩膜几何错误。00B1 已对 19 个低背景节点完成专门复核并生成不可覆盖原始失败记录的 admitted summary。因此人工编译必须引用：

`00B1/low-valid-review/patch_mask_audit_summary_admitted.json`

不能继续引用 00B 原始失败 summary。用 admitted summary 编译后，`formal_patch_mask_admission=true`。

## 4. 富集候选结果

人工标签分布（含 24 张重复卡片）：

| 标签 | 卡片数 | 用途 |
|---|---:|---|
| same_frame | 187 | 严格正例 |
| geometric_overlap | 57 | 严格正例 |
| same_local_site | 9 | 严格正例 |
| likely_same_airport | 2 | 不确定证据，不并入严格正例 |
| uncertain | 7 | 排除 |
| not_same_local_site | 2 | 负例 |
| different_airport | 0 | 负例 |

去除盲重复后，00B1 新增 229 个严格正例；加上 Round-A 的 19 个严格正例，共 248 个。富集检索的主要价值不是直接构造分组，而是成功解决了 Round-A 正例不足的问题。

正式编译结果：

- `status=pass_recommended_evidence_target`；
- pair 总数 600；
- 严格正例 248；
- calibration 正例 186；
- held-out 正例 62；
- 严格正例连通分量 194；
- 跨 split 连通分量 0；
- 盲重复 24，一致率 1.0；
- merge conflict、repeat conflict、failure 均为 0。

## 5. 对当前方法的判断

本批富集集有意集中在高相似候选，因此 229/240 的唯一 pair 被判为严格正例并不代表全库误合并率很低，也不能直接推出机场级身份已经恢复。它说明 masked GeM/VLAD 与几何支持能够高效挖出同帧、重叠或同一局部场地的样本。

下一步必须用预先隔离的 calibration/held-out 证据比较九条检索路由，重点同时观察：

- held-out Recall@20/50/100；
- Wilson 区间下界，而非只看点估计；
- 负例进入候选集合的比例；
- 每个 query 的候选负载；
- 单路和并集是否在召回提升的同时显著放大错误连接风险。

只有 Round-B 达到门槛，才能进入 `TASK-01 retrieval + geometry`；即使达到，也仍只是机场代理分组流程的下一阶段，不是最终 group_id。

## 6. 已发现的流程问题

1. 00B1 回传包不应在人工审核前包含 `blind_card_mapping.csv`。本地通过安全解压规避了污染，服务器后续应只在人工表回传后解封 mapping。
2. 00B1 回报没有提供“完全相同 VLAD 命令复跑后 241/241 shard 全部 SKIP”的明确证据。Round-B 前应补跑一次相同命令，只允许全部 SKIP，不得重算或改变指纹。
3. 原 00B 文档的 Phase B 编译命令引用了原始失败 summary；经过 00B1 修订后必须改为 admitted summary。新的 00B2 任务单已经固定这一点。

## 7. 下一步

执行 `MAR20-GROUPING-TASK-00B2`：上传两份冻结人工表，补 VLAD resume 审计，编译 patch 与 v1.2 标定集，运行九路 Round-B，生成唯一 `task_decision.json`。本阶段不允许生成或交付最终 group_id。
