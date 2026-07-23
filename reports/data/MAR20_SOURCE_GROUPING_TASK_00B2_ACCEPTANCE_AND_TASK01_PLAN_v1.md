# MAR20 来源分组 TASK-00B2 验收与 TASK-01 执行约束 v1

## 1. 本轮结论

`MAR20-GROUPING-TASK-00B2` 通过技术验收和预注册的描述子选择门禁，可以进入全量候选检索与局部几何验证。

本轮只证明：在已人工确认的同画面、几何重叠和同一局部地点正例中，选定的两条背景 VLAD 路由能够以较高召回率把对应图像送入后续候选集。它没有证明这些候选属于同一机场，也没有生成可供 CV3 使用的最终 `group_id`。因此 `formal_grouping_admission=false` 是正确且必须保留的状态。

下一阶段固定使用：

- `masked_block10_vlad_k32_pca512`；
- `masked_block11_vlad_k32_pca512`。

两条路由均来自 DINOv2-B/14 的背景 patch、VLAD K=32 和 PCA-512，但使用相邻的不同 transformer block。它们是互补的高召回描述子，不应被解释为两种完全独立的视觉模态。

## 2. 输入和技术验收

| 检查项 | 结果 | 判断 |
|---|---:|---|
| 回传包 SHA256 | `9d249cfe8f6e83ed5644f63ddb862cb730fcac88d138c94353a98b08dfae6a5a` | 与服务器一致 |
| 00B/00B1 代码与人工输入 SHA | 全部通过 | 输入链可追溯 |
| patch-mask 人工审查 | 120/120 有效 | 通过 |
| 0.10/0.15/0.20 膨胀覆盖率 | 均为 1.0 | 通过 |
| 主膨胀率过度背景损失 | 0 | 通过 |
| 校准 pair | 600 | 完整 |
| strict 正例 | 248 | 达到建议证据量 |
| calibration/held-out strict 正例 | 186/62 对 | 组件隔离 |
| 重复盲评 | 24 对，一致率 1.0 | 通过 |
| calibration 与 held-out strict 组件交叉 | 0 | 防近重复泄漏通过 |
| Round-B | 9 条路由、13 个候选组合 | 全部完成 |

248 个 strict 正例由人工标签中的 `same_frame`、`geometric_overlap` 和 `same_local_site` 构成；`likely_same_airport` 不作为局部重叠检索的正例。这一口径正确。

## 3. 描述子结果解释

### 3.1 正例召回

两条入选路由的并集结果如下。`K` 是每条路由的 K，因此并集实际候选数最多约为 `2K`，去重后更少。

| 数据部分 | R@20 | R@50 | R@100 | R@100 Wilson 下界 |
|---|---:|---:|---:|---:|
| calibration，372 个有向正例 | 0.9839 | 0.9892 | 0.9892 | 0.9727 |
| held-out，124 个有向正例 | 0.9839 | 1.0000 | 1.0000 | 0.9700 |
| 合计，496 个有向正例 | 0.9839 | 0.9919 | 0.9919 | 0.9795 |

所有像素等价正例在三个 K 下均为 1.0。held-out 没有参与两条路由的选择，只用于选择完成后的审计，因此没有直接的选路泄漏。

### 3.2 正式 K 冻结

按照预注册规则，若 K=50 到 K=100 新增的已知正例不超过 2%，采用较小的 K=50，并在反向审计时重新使用 K=100。

本轮 calibration 和 held-out 从 K=50 增至 K=100 均没有新增正例，因此冻结为：

- 正式全量候选检索：每条入选路由 `K=50`；
- 反向泄漏审计和召回压力测试：每条路由 `K=100`；
- K=100 的候选不得因为“排名更深”而降低几何与人工证据标准。

这样遵守了预注册的最小平台 K 原则，也避免在正式几何阶段无意义地扩大候选量。

### 3.3 负例命中不能解释为假阳性率

selected union 在 held-out 已知负例中的 top-K 命中率为：

- K=20：4.23%；
- K=50：9.86%；
- K=100：14.79%。

该指标表示“人工标注的某个已知负对是否进入对应查询的 top-K”，不是候选集合的 precision 或模型假阳性率。它仍说明：扩大 K 会明显增加困难相似背景，所以 DINO/VLAD 结果绝不能直接 union 成组，局部几何验证和人工复核是必要阶段。

### 3.4 未入选路由的处理

`all_masked_gem` 在 held-out 上表现看起来较好，但它没有在 calibration-only 选择中胜出。现在不能根据已经打开的 held-out 结果反向替换正式路由，否则会产生后验选择偏差。

后续允许：

- 在已由正式 VLAD 路由召回的 pair 上追加 GeM 分数，作为独立证据字段；
- 在最终跨折反向审计中把 GeM 作为诊断性 rescue route；

后续不允许：

- 因 held-out 表现较好而把 GeM 静默加入 TASK-01 的正式候选生成并据此扩大 must-link；
- 把六条 VLAD 路由全部并集。该并集候选量约翻倍，且已知负例命中明显升高。

## 4. 本轮结果的科学边界

### 4.1 held-out 是“标签隔离”，不是完全独立的候选发现评测

校准与 held-out 按 strict 组件隔离，能够避免同一近重复组件同时出现在两边；这一设计可靠。

但 229 个新增正例来自九条候选路由富集后的人审。也就是说，正例发现过程仍偏向这些描述子容易召回的关系。held-out 能检验“在已发现 strict 关系上的泛化”，不能证明尚未发现的同机场或同地点关系也具有相同召回率。

因此 TASK-01 之后仍需要：

- K=100 反向审计；
- 经典图像证据和局部几何证据；
- 跨 official side、高影响桥边和尾类的定向审计；
- core/guard/embargo 敏感性分析。

### 4.2 strict 局部关系不等于机场身份

当前正例主要证明同画面或存在固定地物共同区域。对“同一机场但画面完全不重叠”的关系，视觉算法通常只能给出风险证据，不能给出可靠真值。

后续必须区分：

- `same_frame` / `geometric_overlap` / `same_local_site`：人工确认后可进入 strict core；
- `likely_same_airport`：只能进入 guard 证书候选；
- 高风险但证据不足：进入 fold-specific embargo；
- `uncertain`：不得自动合并。

## 5. TASK-01 冻结执行路径

### 5.1 阶段一：正式全量候选检索

在完整 3,842 图节点库上运行，不按飞机细类、编号区间或 official train/test 过滤：

1. 两条入选 VLAD 路由各取 top-50；
2. 按无向 `pair_uid` 去重，保留双向 rank、最佳旋转、两路相似度和路由支持数；
3. 追加 pixel SHA、规范化像素 SHA、pHash/dHash 证据；
4. 分别标记 `target-target`、`target-bridge`、`bridge-bridge`；
5. 物理分离 target-only 正式候选和 full-bridge 诊断候选；
6. 额外生成 K=100 的只读反向审计索引，不直接进入普通人工队列。

### 5.2 阶段二：局部证据级联

所有 pair 先经过廉价证据层，再按优先级进入较重验证：

1. DINO patch overlap：飞机 patch 排除后，先 19×19 粗匹配；
2. 对高价值、冲突或边界候选再运行 37×37 细匹配；
3. SIFT：删除飞机 mask 内关键点，ratio test、双向 mutual check；
4. 分别估计 similarity、affine，homography 仅作诊断；
5. 记录内点数、内点率、对称误差、两端覆盖、4×4 网格、凸包面积、方向熵、主线集中度和固定 seed RANSAC 稳定率；
6. LightGlue 只用于 DINO 与 SIFT 冲突、组件桥边、target-bridge-target 关键路径等高影响候选；首轮不超过 500 对；
7. RoMa 仅用于不超过 50 个仍会改变分组结论的争议 pair。

不得按“内点多”单一条件判同源；规则跑道线和停机坪接缝容易产生几何爆发。

### 5.3 候选队列与自动化边界

| 队列 | 典型证据 | 操作 |
|---|---|---|
| Q0 | pixel/规范化像素等价 | 唯一允许自动 strict 的关系，仍抽查 |
| Q1 | 两路描述子支持，patch/SIFT 几何稳定且覆盖分散 | 优先盲审；不自动 union |
| Q2 | 部分几何支持、可能同地点或同机场 | 人工判断 likely/uncertain |
| Q3 | 高相似但匹配集中于飞机、跑道直线或通用结构 | hard-negative 复核 |
| Q4 | 低证据 tail 候选 | 保留用于 K=100 反向审计，不进入普通审查包 |

执行方案 v1.1 已规定：除像素等价 H0 外，几何强边也必须经过人工确认。TASK-01 不得擅自恢复旧版“高几何分自动 H1 union”的逻辑。

### 5.4 阈值拟合与评估

阈值只能在 calibration 部分拟合；held-out 只在阈值冻结后开启。局部几何正例仅使用：

- `same_frame`；
- `geometric_overlap`；
- `same_local_site`。

`likely_same_airport` 不得充当局部几何正例；`uncertain` 不参与阈值拟合。

报告至少包含：

- calibration/held-out 的正例召回、困难负例命中和 Wilson 区间；
- 每个队列的 pair 数、target/bridge 构成、official 跨侧构成；
- K=20/50/100 的边际召回和候选规模；
- patch、SIFT、LightGlue 各自新增证据量和冲突量；
- 低有效背景 patch 图像的单独表现；
- 几何爆发过滤前后差异；
- 固定 seed 重跑一致性、NaN/Inf、resume 审计。

## 6. TASK-01 停止门禁

TASK-01 的正常完成状态应是 `waiting_for_blind_pair_review`，而不是 `formal_grouping_admission=true`。

出现以下情况必须停止并回传，不得自行放宽：

- K=50 在冻结 held-out strict 正例上的召回低于 0.95；
- 候选表存在 self-pair、重复 pair 合并错误或双向 rank 不可复现；
- target-only 与 bridge 诊断产物混写；
- calibration 阈值在查看 held-out 后被改写；
- patch/SIFT 几何字段存在非有限值或坐标越界；
- 人工卡片泄露文件编号、official side、细类或机器建议标签；
- 任一非 H0 pair 被脚本自动 union；
- 局部匹配缓存无法 resume 或资产指纹不一致。

## 7. 下一轮预期产物

服务器应回传小型、可审查的结果包，至少包含：

- `retrieval_decision.json` 与 K=50/K=100 饱和报告；
- `candidate_edges_target.csv`；
- `candidate_edges_full_bridge.csv` 或其统计摘要；
- `pair_evidence.csv`；
- geometry calibration/held-out 报告；
- Q0～Q4 队列统计；
- 匿名 pair review CSV、contact sheets 和盲化映射（映射与卡片分离）；
- cache/index/资产 SHA、resume、环境与日志；
- `task_decision.json`，预期为 `waiting_for_blind_pair_review`。

大型 DINO/VLAD、局部 token 与匹配缓存继续留在同一服务器复用，不在回传包中重复下载。

## 8. 对总项目的影响

00B2 解决了 MAR20 分组流程中最关键的“候选发现是否有足够召回”问题，说明这条路线值得继续，不需要退回文件编号分段或官方 train/test 伪机场标签。

但当前仍不能把任何结果交给 B 作为正式 `group_id`。最早可向 B 提供的稳定结果是 TASK-02 后的 `target_core`；正式 CV3 首选则需要 `target_guard` 或在时间不足时采用 `target_core + unresolved-risk embargo`。在此之前，B 可以继续完成非 MAR20 的舰船、车辆来源分组与 CV3 求解接口，但不应冻结飞机 fold。
