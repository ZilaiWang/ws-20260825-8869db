# MAR20 机场代理 K=60 最终验收报告

## 1. 验收结论

```text
status: airport_proxy_k60_ready_for_cv3
formal_grouping_admission: true
strict_component_split_count: 0
decision: accepted_for_B_cv3
```

本次结果已经完成机场级代理分组目标；该输入随后已用于生成并验收正式
`cv3_airport_proxy_k60_v2`。旧的 2,882 组局部同源文件已经废弃，
不能再作为 CV 分组键。

本结果是“机场代理视觉域”，不是带真实机场名称的逐图机场真值。正式材料应使用 `MAR20 airport-proxy grouped CV3` 或“MAR20 机场代理来源隔离 CV3”，不得写成真实 airport-disjoint ground truth。

## 2. 输入与可复现性

| 项目 | 结果 |
|---|---|
| 完整 MAR20 | 3,842 张 |
| 比赛 target | 3,073 张 |
| bridge | 769 张 |
| 机场数先验 | 60 |
| 算法 | strict component collapse + rotation-mean route fusion + cosine average-link |
| 描述子 | masked block10/block11 VLAD K32 PCA512 |
| strict 组件 | 3,590 |
| strict 组件被拆分 | 0 |
| 服务器确定性复跑 | 两份 CSV 完全一致 |

回传包：

```text
outputs/MAR20-AIRPORT-PROXY-K60-v1-return.tar.gz
SHA256 0e0b720e00874b532f912495e542fb28592a9bd8333781f122c63de730082f4e
```

正式 target 文件：

```text
outputs/MAR20-AIRPORT-PROXY-K60-v1/final/mar20_airport_proxy_assignments_target.csv
SHA256 afde2a3d9b9941ad5fc603d979adcdf68a0c9819541eeb96a06993654529cf87
```

归档备注：本次回传包中的 `logs/` 目录为空，因此本地不能再次读取服务器的 pytest、ruff 和确定性复跑日志正文；服务器完成回报已经声明这些门禁通过。输入指纹、两份正式 CSV、决策文件及本地逐行复核均一致，此归档缺项不阻塞 B 使用，但服务器日志暂时不要主动删除。

## 3. 完整性审计

| 检查 | 结果 |
|---|---:|
| all CSV 行数 | 3,842 |
| target CSV 行数 | 3,073 |
| 唯一 target `competition_image_id` | 3,073 |
| 新旧 target ID 集合差异 | 0 |
| 完整 MAR20 代理组 | 60 |
| 含 target 的代理组 | 60 |
| 连续组名 | `mar20-airport-proxy-001` ～ `060` |
| 空 `group_id` | 0 |
| strict component split | 0 |
| CSV 内组大小字段错误 | 0 |

所有 60 个组都至少含一张 bridge 图。每组 bridge 数量为 1～35，中位数 12，说明 769 张外部 MAR20 图确实参与了完整来源结构组织，而不是只在文件中挂名。

## 4. 组规模

| 范围 | min | median | max |
|---|---:|---:|---:|
| 完整 3,842 图 | 9 | 59.5 | 148 |
| 3,073 target | 6 | 48 | 114 |

不存在 singleton，也没有由错误链式合并形成的数百张超大组。该规模能够作为 CV3 的组级原子，同时仍给三折类别和对象数量平衡留下空间。

MAR20 官方 train/test 两侧在 55/60 个代理组内同时出现。这不是泄漏，而说明最终分组没有机械复制原始 train/test 编号边界；原始边界继续只用于审计。

## 5. 描述子一致性与置信度

| 比较 | ARI |
|---|---:|
| block10 vs block11 | 0.7014 |
| block10 vs fused | 0.7544 |
| block11 vs fused | 0.8106 |

两路独立层级与融合结果具有较高但非完全一致的结构，符合“同一机场代理结构由多路背景表征共同支持”的预期。融合结果更接近 block11，但没有退化为只复制单一路由。

| 指标 | p05 | median |
|---|---:|---:|
| 节点—本组簇心 cosine | 0.1764 | 0.3531 |
| 相对次近簇心 margin | 0.0249 | 0.2427 |

133/3,842 张图的簇心 margin 小于 0，占 3.46%。这是 average-link 层次聚类的正常边界现象：聚类优化的是成对层次结构，不保证每个节点最终都离本组算术簇心最近。相关图已经通过 `centroid_margin` 保留为审计对象；不重新逐图改簇，因为最近簇心重分配会破坏层次结构、strict 组件约束和预注册确定性。

## 6. 给 B 的使用合同

B 只使用：

```text
competition_image_id → group_id
```

具体要求：

1. MAR20 的 3,073 张图全部按新 `group_id` 作为不可拆分原子；
2. 与舰船、车辆和非 MAR20 图的来源组一起进入统一 CV3 求解器；
3. 同一 `group_id` 不得跨 fold；
4. 每张训练图恰好成为一次验证样本；
5. 优先满足 25 类在各折的结构可行覆盖，再平衡各类对象数和总对象数；
6. `membership_cosine` 和 `centroid_margin` 只做审计，不能据此把低置信图改成 singleton；
7. 不再使用旧 `MAR20-FINAL-GROUPING-v1/mar20_final_group_assignments.csv`。

## 7. 对后续 P 系列实验的影响

该结果解除的是正式来源划分的核心阻塞。全数据
`cv3_airport_proxy_k60_v2` 已于 2026-07-23 完成并验收，因此下列工作
当前已经放行：

1. 在正式 CV3 上重跑 P03 的 224 工作点，校准原探索划分下约 0.97 crop 分类上限；
2. 在同一划分上重跑 P04 的 ConvNeXt、DINO-B CLS+patch 和 CleanDIFT 关键教师，形成正式教师选择；
3. 生成正式 OOF；P06-TASK-02 仍只等待该四文件 OOF 输入，不再等待 CV3；
4. 后续增强、蒸馏和模型创新统一使用同一 CV3，不再回到 MAR20 原 train/test 边界。

当前入口见
[`DATA_SPLITS_MASTER_INDEX_v1.md`](DATA_SPLITS_MASTER_INDEX_v1.md)。
