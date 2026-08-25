# MAR20 来源分组 TASK-01 实现报告 v1

## 1. 实现目标

本批把 00B2 选出的两条背景 VLAD 路由转为一次可执行的全量流程：

1. 完整 3,842 图、四旋转、两路 VLAD 的正式 K=50 检索；
2. 独立生成 K=100 反向审计索引；
3. 选择最多 6,000 个高价值 pair，并强制包含全部 600 个校准/held-out pair；
4. 用背景排除后的 SIFT/RANSAC 和 DINO patch overlap 生成局部证据；
5. 只用 calibration 拟合人工复核排序，held-out 只审计；
6. 生成 300 个新 pair、48 个隐藏控制和 8% 盲重复的人工审查包。

正常终态是 `waiting_for_blind_pair_review`。除像素完全等价外，任何几何结果都不会自动 union。

## 2. 为尽快得到可用答案所做的收敛

本轮不立即运行 LightGlue/RoMa，也不实现自动机场聚类。原因是：当前最急需的是稳定 strict core，而不是把所有不重叠机场关系一次猜完。

首轮使用：

- DINOv2-B block 10/11 VLAD：高召回候选发现；
- DINOv2-B block 11 局部 patch：粗粒度共同区域证据；
- SIFT + similarity/affine/homography RANSAC：可解释几何证据；
- 人工盲审：strict 关系最终决定。

LightGlue 只在首轮盲审后用于会改变组件结构的冲突 pair。这样不会拖慢 core 交付，也不会降低证据标准。

## 3. 关键工程设计

### 3.1 检索

- 每条路由按四旋转任意组合的最大余弦分数检索；
- 保存两个方向的 rank、分数和最佳旋转组合；
- K=50 和 K=100 文件物理分开；
- target-target、target-bridge、bridge-bridge 明确标记；
- pixel SHA 与 pHash 只作为附加证据；
- 任何类别、编号或 official side 都不参与候选过滤。

### 3.2 patch overlap

直接缓存 37×37×768 token 会造成不必要的磁盘与计算负担。本实现复用 00B1 已冻结的 block11/K32 `local PCA-128`：

1. 提取 block11 的 37×37 token；
2. 使用已冻结且无标签拟合的 PCA 投影至 128D；
3. 在飞机无效 patch 排除后做 mask-aware 19×19 自适应池化；
4. 保存 L2 归一化 token 和有效位；
5. pair 级计算 mutual nearest neighbour、相似度分布和空间网格覆盖。

它是局部证据，不是独立分类器，也不自动决定 pair 标签。

### 3.3 SIFT/RANSAC

每张图的 SIFT 特征单独缓存，可断点续跑。飞机膨胀 mask 内不提取关键点。pair 级执行：

- 双向 ratio-test mutual matches；
- similarity、affine 和诊断性 homography；
- 对角线比例 RANSAC 阈值；
- 内点数/率、误差、4×4 覆盖、凸包覆盖；
- 方向熵与主方向集中度；
- 20 次固定 seed RANSAC 稳定率。

这比仅按 homography 内点数排序更能抑制跑道直线造成的几何爆发。

### 3.4 人工队列

人工排序模型是固定的 calibration-only L2 logistic regression。它仅用于把 pair 分成 Q0～Q4 的复核优先级：

- Q0：像素等价；
- Q1：排序分高且至少一类局部几何稳定；
- Q2：部分支持；
- Q3：检索强但局部证据弱，优先作为 hard negative；
- Q4：低证据 tail，本轮不展示。

Q1 仍然不是自动边。held-out 只生成审计指标，不参与特征、阈值或队列选择。

## 4. 代码与产物

新增核心模块：

- `src/rsdet/grouping/retrieval.py`；
- `src/rsdet/grouping/geometry.py`。

新增执行脚本：

- `retrieve_mar20_task01_candidates.py`；
- `build_mar20_geometry_queue.py`；
- `extract_mar20_patch_overlap_cache.py`；
- `verify_mar20_task01_geometry.py`；
- `analyze_mar20_task01_geometry.py`；
- `build_mar20_task01_blind_review.py`。

配置与测试：

- `configs/grouping/mar20_task01_retrieval_geometry_v1.json`；
- `tests/test_mar20_grouping_task01.py`。

## 5. 首轮完成后的最短路径

1. 完成人工盲审；
2. 编译重复一致率与隐藏控制质量；
3. 人工确认 Q0/Q1/Q2 中的 strict edge；
4. 立即生成 `target_core` 和每图稳定 `group_id`，先交给 B 开始 CV3；
5. 只对组件桥边和 unresolved 高影响 pair 补 LightGlue；
6. guard 来不及成熟时采用 `target_core + unresolved-risk embargo`，不延误正式实验。

这条路径优先交付可信、可用、可解释的 core，不再把“完全恢复机场身份”设为开始 CV3 的前置条件。
