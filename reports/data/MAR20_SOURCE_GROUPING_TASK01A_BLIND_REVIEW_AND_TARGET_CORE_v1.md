# MAR20 严格局部同源分量报告（机场级代理聚类的中间产物）

> **更正（2026-07-22）**：本报告最初把 2,882 个 strict/guard 连通分量误称为最终 CV3 分组。它们只解决“相同画面或局部场景不得跨折”，没有把同一机场内互不重叠的视角聚合为约 60 个机场代理域。因此第 1、6、7 节的“最终交付”结论已废止；本报告中的连通分量仅作为机场级聚类的 must-link 约束和审计证据。

## 1. 阶段结论

MAR20 分组流程已经得到一版高精度的局部同源连通分量，但它不能单独用于正式 CV3。

本次产物不是机场真值，而是基于背景检索、局部几何、匿名人工复核得到的“局部场景代理组”。strict-core 与 guard 两层证据用于机场级聚类前的约束和审计，不再直接交给 B 作为最终 `group_id`。

```text
blind review: complete
strict-core negative conflicts: 0
CV-guard negative conflicts: 0
status: strict_local_scene_components_ready
formal_grouping_admission: false
```

## 2. TASK-01A 技术验收

原始 `pair_evidence.csv` 有 6,000 个唯一 pair。29 个退化 affine fit 每个包含 1 个非有限矩阵、`median_error` 和 `p95_error` 两个非有限误差，共 87 个非有限字段。服务器最初按旧任务单中的 58 计数停止；后续清洗文件虽已补跑生成，但原 driver 日志没有形成连续成功链。

本地进行了独立复算：29 个 pair、87 个字段全部被清为空证据，`affine_fit_valid=0`，清洗后非有限值为 0；候选 assignment SHA、校准/held-out 指标和 Q1～Q4 数量均未变化。因此该问题是合同计数遗漏了矩阵字段，不是特征、检索、排序或图像数据损坏。相关脚本和 01A 文档已把预期值修正为 87。

## 3. 盲审协议与结果

### 3.1 隔离执行

共审阅 376 张匿名卡：300 个新候选、48 个控制项、28 张重复卡。审阅期间只读取原图、目标遮挡后的背景图和边缘图，没有读取 `blind_mapping_private.csv`。

解盲前冻结文件：

- `manual_review_decisions_ai.csv`：376 行；
- SHA256：`1b92730e68ca48cd7a0616cc446101d22a4e4eebb733543ef78653720380fde2`。

解盲后只复核了两组重复卡中的方向冲突，以及一个隐藏负控制误判；最终文件：

- `manual_review_decisions_ai_resolved.csv`：376 行；
- SHA256：`ed30b2dcf0e7aa563abc67fc32dd9a6225431095e70f9f6884e2f0075342ab93`。

### 3.2 质量门禁

| 门禁 | 结果 |
|---|---:|
| 重复 pair 组 | 28 |
| 解盲复核后 strict/non-strict 一致 | 28/28（100%） |
| 正控制进入 strict-core | 25/26（96.15%） |
| 负控制进入 strict-core | 0/24（0%） |
| 负控制被明确判负 | 12/24 |
| 负控制保留为弱/不确定、不合并 | 12/24 |

这说明该人工协议适合构造高精度 core：可能少合并一小部分真同源 pair，但没有把隐藏负控制错误并入 strict-core。

## 4. 图构造

### 4.1 Strict core

仅接受以下标签且置信度不低于 0.85：

- `same_frame`；
- `geometric_overlap`；
- `same_local_site`。

去重后得到 252 条严格边。合并后人工负边冲突为 0。`group_id` 由该图的连通分量确定。

### 4.2 CV guard

为了不让弱疑似同机场 pair 跨 fold，同时避免把 94 张风险图永久排除在 OOF 之外，内部额外构造 CV guard：

- 保留全部 strict-core 边；
- 加入置信度不低于 0.60 的 `likely_same_airport` 边；
- 已知为负的隐藏控制 pair 不进入 guard；
- 弱边仅用于 fold 约束，不宣称同一局部地点或同一机场；
- guard 图与全部人工负边的冲突仍为 0。

这是比“风险图强制只进训练集”更完整的做法：全部 3,073 张比赛 MAR20 图仍可各自成为一次 OOF 验证样本。

## 5. 分组统计

| 项目 | Strict core | CV guard |
|---|---:|---:|
| MAR20 target 图 | 3,073 | 3,073 |
| target 分组数 | 2,920 | 2,882 |
| 非 singleton 组 | 133 | 165 |
| 非 singleton 中的 target 图 | 286 | 356 |
| 2 张组 | 116 | 142 |
| 3 张组 | 14 | 20 |
| 4 张组 | 3 | 3 |
| 跨 MAR20 官方 train/test 的组 | 27 | 32 |
| 人工负边冲突 | 0 | 0 |

跨官方边界的 32 个 guard 组进一步证明：MAR20 原始 train/test 不能直接视作机场互斥划分。另一方面，大多数图仍是 singleton，说明本结果是高精度、低覆盖的泄漏防护，不是完整机场身份恢复。

## 6. 原交付方案已废止

以下原 `group_id` 方案不得再交给 B 作为正式 CV3 分组。它只保留为机场代理 K=60 聚类的 strict local-scene 约束。

旧文件：

```text
outputs/MAR20-FINAL-GROUPING-v1/mar20_final_group_assignments.csv
```

即使文件名中仍有 `final`，也不得使用。新的正式文件由 TASK-02 生成，名称为 `mar20_airport_proxy_assignments_target.csv`。

## 7. 后续决策（已更正）

本阶段不再增加 LightGlue、不再审更多 Q1 pair，也不尝试恢复机场名称。下一步改为：

1. 以本报告的 strict local-scene 分量为 must-link，复用两路入选 masked-VLAD 特征生成 60 个机场代理域；
2. B 使用新的机场代理主文件生成 CV3，并回传 fold 规模、25 类分布、最大组和跨 fold 泄漏审计；
3. 用新 CV3 重跑 P03/P04 的关键工作点，判断先前探索性结论在正式来源隔离划分上是否保持。

如果 CV3 求解因 4 张以内的小组约束出现明显类别失衡，再调整 fold 求解器，不回头修改 MAR20 证据图。
