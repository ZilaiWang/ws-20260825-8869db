# MAR20 Round-A 验收与 TASK-00B 修复执行方案 v1

## 1. 文档定位

| 项目 | 结论 |
|---|---|
| Round-A 技术状态 | 完成，无代码故障 |
| Round-A 科学状态 | `complete_round_a_no_admission` |
| 能否直接生成 group ID | 不能 |
| 能否据此否定 DINOv2 | 不能 |
| 下一步 | TASK-00B：特征级掩码、正例富集、域内 VLAD 和四旋转重新准入 |

TASK-00B 是独立的 v1.2 修复链。Round-A v1.1 的原始失败记录保持不变，不通过改阈值、改人工标签或覆盖缓存来“补通过”。

## 2. Round-A 已经得到的事实

### 2.1 输入级擦除不适合正式地点表征

120 个节点的审核结果：

| 方法 | 飞机残留率 | 修补伪影率 | 准入 |
|---|---:|---:|---|
| blur | 89.17% | 75.83% | 否 |
| local mean | 5.83% | 80.83% | 否 |
| Telea | 13.33% | 75.83% | 否 |
| background tile | 114 张可用 tile 中 1 张可见飞机 | — | 否 |

因此不再继续增加像素修补算法。旧 blur/local-mean/Telea 缓存只允许用于扩大人工候选队列，不得进入正式描述子选择或自动成边。

### 2.2 人工复核稳定，但标定集不富集正例

360 个唯一 pair 中：

- 严格正例 19：`same_frame=2`、`geometric_overlap=9`、`same_local_site=8`；
- `likely_same_airport=56`，保持不确定，不转成严格正例；
- 明确负例 285；
- 盲重复一致率 1.0。

严格正例被分为 calibration 16 对、held-out 3 对，后者只有 6 个检索方向，低于预注册门禁 10。即使观察到 `Recall@100=1.0`，也只代表 6/6，不具备描述子选择效力。

### 2.3 实现与原始掩码合同存在差距

原始协议要求：飞机掩码覆盖率超过 20% 的 DINO patch token 不参与 GeM/VLAD。Round-A 实现实际使用修补后图像，对所有 patch token 做 mean/GeM。

这是 TASK-00B 的直接修复对象：不再依赖修补图像是否逼真，而是从聚合计算中显式删除前景 token。

## 3. v1.2 科学合同

### 3.1 飞机掩码

对每个 HBB 使用各向同距外扩：

\[
d=\operatorname{clip}(r\max(w,h),8,40)\text{ pixels},\quad r\in\{0.10,0.15,0.20\}.
\]

主工作点为 `r=0.15`。图像和掩码必须经过同一 EXIF 处理、`518×518` resize 和四旋转；mask resize 固定为 nearest-neighbor。

### 3.2 patch 有效性

DINOv2-B/14 产生 `37×37=1369` 个 patch token。若某 patch 与外扩飞机掩码的面积交比超过 20%，则该 token 不参与后续 mean、GeM、VLAD 或视觉词典拟合。

每行缓存必须记录：

- `patch_mask_sha256`；
- `valid_patch_count / patch_count`；
- `valid_patch_fraction`；
- `foreground_fraction`；
- 图像输入 SHA、旋转角和描述子名称。

主提取中任一图像的有效 patch 比例小于 25% 时科学门禁失败，不允许自动改掩码阈值重跑。

### 3.3 人工 patch audit

120 张抽样必须覆盖 20 类飞机和飞机数量最多的高密度图。联系表同时展示 10%/15%/20% 外扩。准入条件：

- 图片可判读率至少 95%；
- 三种外扩的飞机覆盖通过率均至少 95%；
- 主工作点 15% 的“过量删除背景”率不高于 10%。

## 4. 两遍特征提取与 VLAD

### 4.1 第一遍：masked GeM + 等额 patch sample

对 3,842 张 MAR20 图像提取 block 9/10/11，四旋转仅保存聚合描述子。每张图像只在 0° 从有效背景 token 中确定性抽取 16 个 patch，确保每图对视觉词典贡献相同。

保存：

- masked mean；
- masked signed-GeM `p=2/3/4`；
- 分片 patch sample，不保存全量 token。

### 4.2 视觉词典

每层先将等额 patch sample 用无监督 PCA 降到 128 维，再分别拟合 MiniBatchKMeans `K=16/32`。固定 `seed=202625`、`n_init=3`，词典路径和 SHA 写入 manifest。

### 4.3 第二遍：VLAD

仅用有效背景 token 计算 residual VLAD，cluster 内和全局均做 L2 归一化。再仅用 0° 全库无标签描述子拟合 global PCA-whitening，输出 512 维检索向量。

完整 MAR20 的额外 769 张图仅用于无监督表征和桥接诊断，不进入任何比赛检测/分类模型训练。

## 5. 严格正例富集

### 5.1 候选召回

候选 pair 为以下路由的并集：

- block 9/10/11 masked-GeM p=3；
- block 9/10/11 masked-VLAD K=16/32 + PCA512；
- pHash64 近拷贝；
- 旧 Telea cache 仅作候选路由；
- 完整 RGB 像素 SHA。

检索对全库进行，不按飞机类别、编号段或原官方 train/test 侧过滤。该些字段只用于审计。

### 5.2 几何富集

对候选的前 1,600 对在飞机掩码外提取 SIFT，用 Lowe ratio 和 RANSAC homography 记录：

- good match 数；
- inlier 数和比例；
- 两图 inlier 空间覆盖；
- 中位重投影误差。

几何只排人工队列，本任务不自动产生 union edge。

### 5.3 盲评批次

第一批固定 240 个唯一 pair，至少 75% 为 target-target，附加 10% 左右对换位盲重复。新候选不包含 Round-A 已审核 pair。

目标：

- 最低准入：严格正例总数至少 30，held-out 至少 5 对/10 方向；
- 推荐目标：严格正例总数至少 60，held-out 至少 15 对；
- 严格正例连通分量不得跨 calibration/held-out；
- `likely_same_airport` 不得为凑数转成 strict positive。

如第一批新增 strict positive 少于 20，允许且只允许再构建一个去重的富集批次；第二批后仍不足则停止无限补标，转保守 core/embargo 方案。

## 6. Round-B 比较与准入

### 6.1 比较路由

| 类型 | 路由 |
|---|---|
| GeM | block 9/10/11 masked signed-GeM p=3 |
| VLAD | block 9/10/11 × K=16/32 × PCA512 |
| 预注册并集 | calibration 最优两路、全 GeM、全 VLAD、最优两路+最优 VLAD |

单路和并集的排序只读 calibration。held-out 仅在选定后用于独立验收，不反向选路由。并集 `K=100` 表示每条路由 top-100 的去重并集，必须同时报告平均候选量。

### 6.2 准入条件

同时满足：

1. patch audit 正式通过；
2. 标定集最低证据门禁通过；
3. held-out strict positive 方向至少 10；
4. 已知 `same_frame` 的多路并集 `Recall@100=100%`；
5. held-out strict positive 多路并集 `Recall@100≥95%`；
6. 报告 Wilson 区间；
7. 任何描述子分数都没有自动 union 权限。

通过后只获得 `TASK-01 retrieval + geometry` 准入，`formal_grouping_admission` 仍为 false。

## 7. 终止与回退

| 终止状态 | 下一步 |
|---|---|
| `ready_for_task01_retrieval_and_geometry` | 运行正式全量召回与几何验证 |
| `needs_second_enrichment_batch` | 仅再审核一批去重高概率候选 |
| `complete_00b_patch_mask_no_admission` | 最多一次显式掩码协议修订，否则回退 |
| `complete_00b_retrieval_no_admission` | 直接回退保守 core + fold-specific embargo |

保守回退不企图恢复 60 个真实机场：精确重复、近重复和确认局部重叠构成 `target_core`；其余大部分为单例组；高风险视觉邻居用 fold-specific embargo 防止直接进入训练侧。

## 8. 实现与产物

| 产物 | 作用 |
|---|---|
| `patch_mask_audit.csv` + overlays | 特征级掩码人工验收 |
| masked-patch feature cache | 四旋转 mean/GeM |
| patch sample shards | 每图等额背景 token |
| `codebook_manifest.json` | PCA128 + VLAD16/32 词典锁 |
| VLAD cache + PCA512 cache | Round-B 检索描述子 |
| `enriched_candidate_pairs.csv` | 多路召回 + SIFT 富集队列 |
| enriched blind review | 240 pair + 10% 盲重复 |
| `calibration_pairs_v1p2.csv` | 新旧证据合并且组件隔离 |
| `round_b_descriptor_bakeoff.csv` | calibration 选择 + held-out 验收 |
| `task_decision.json` | 唯一终止状态与下一步 |

服务器任务单为 `docs/server/MAR20_GROUPING_TASK_00B_MASKED_PATCH_VLAD_AND_ENRICHMENT.md`。
