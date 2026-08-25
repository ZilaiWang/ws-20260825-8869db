# E 主线 3（全局对象聚合）工作评审报告

日期：2026-08-10
评审人：A
评审对象：E 的 `feat/e-10k-pipeline` 分支（已合并至 master `6402054`）
依据：项目可回溯纪律 + 8/4 规划主线 3 + 比赛评分 V1.6

## 1. 结论摘要

| 维度 | 评级 | 说明 |
|---|---|---|
| 代码质量 | ✅ **合格** | 核心模块结构清晰、契约完整、性能优化到位 |
| 测试 | ✅ **合格** | 94 测试全过，覆盖聚合语义/契约/困难对象/边界 |
| 真实数据评估 | ⚠️ **部分合格** | 数字自洽，但**评估脚本缺失，不可复现** |
| 报告与存档一致性 | ⚠️ **有出入** | "FDR 3.83%" 无存档支撑 |
| 与团队协作 | ✅ **合格** | 只改自己文件，未碰队友代码 |
| 主线 3 完整性 | ⚠️ **未完成** | 核心场景（跨 tile 切分）未验证，自己已声明 |

## 2. 代码质量（合格）

`src/rsdet/postprocess/global_aggregation.py`（约 700 行）：

- **机制转译清晰**：Group Evidence Matters (arXiv:2509.10779) → Spatial Gate（网格哈希 union-find）+ Semantic Gate（IoU 连通分量）+ score 加权细类投票 + canonical 框 + 同类 NMS。最小机制落地，替换路径明确。
- **性能工程到位**：
  - `spatial_cluster` 网格哈希把配对从 O(n²) 降到近线性；
  - `class_aware_nms` numpy 向量化，10K proposals 从 11.68s → 0.009s（~1300×）；
  - 真实 55,548 proposals 聚合 0.81s。
- **契约完整**：`GlobalObject`（object_id/parent_image_id/bbox/category/evidence/source_tile_ids/category_votes）、`fuse_global_predictions` 与 `fuse_tile_predictions` 接口对齐、`global_object_manifest` 输出对象级证据。
- **WP6 条件计算**：`HardObjectCriteria`/`gate_hard_objects`/`re_detect_hard_objects` 困难对象完整重裁二次检测 + 证据融合。

## 3. 测试（合格）

94 个测试全过（5.62s），覆盖：
- `test_e_global_fusion`：全局融合（坐标恢复/聚合/NMS）
- `test_e_oof_aggregation`：聚合语义（空间聚类/投票/子簇）
- `test_e_output_contract`：GlobalObject/pipeline 契约
- `test_e_hard_object_refinement`：WP6 条件计算
- `test_e_pipeline`/`test_e_boundary_accuracy`/`test_e_image_formats`/`test_e_fusion`

## 4. 真实数据评估（部分合格 —— 有缺陷）

### 4.1 数字自洽（已验证 ✅）

`tests/wp3_real_eval.json`：
- n_proposals 55,548 / n_objects 31,092 / n_gt 20,933
- recall 0.9694 = tp/gt 自洽 ✅
- fdr 0.3474 = fp/(tp+fp) 自洽 ✅
- dedup_rate 0.4403 = 1 - 31092/55548 自洽 ✅

### 4.2 缺陷 1：评估脚本缺失（⚠️ 不可复现）

**全仓库找不到生成 `wp3_real_eval.json` 的脚本**——只提交了结果存档，没有评估脚本。违反项目可回溯纪律。且输入的 `oof_proposals.csv`、`formal_crop_manifest.csv` 的 SHA 未在存档中记录，无法确认评估用的是哪个数据版本。

### 4.3 缺陷 2：报告与存档不一致（⚠️）

- 报告说 "FDR（score≥0.25）= 3.83%"——存档里只有全量 fdr 0.3474，**没有 0.25 阈值过滤的评估**；
- 报告说 "聚合前 Recall 98.4% → 聚合后 96.9%"——存档只有聚合后，**没有聚合前基线**；
- 这两个关键数字没有脚本/存档支撑，无法验证。

### 4.4 缺陷 3：macro_recall 口径未说明（⚠️）

存档 `macro_recall 0.9550`，但没说明是"按大类 macro"还是"按细类 macro"、是否与官方 V1.6 口径（大类内细类简单平均）一致。

## 5. 主线 3 完整性（未完成 —— 本人已声明）

E 自己明确说了：**"本次 OOF 是整图级预测，同一目标整图上只有 1 个框，因此未触发'同一目标被切成多块 + 多细类'的跨细类冲突归并场景"**。

这是**主线 3 的核心场景**（10K 大图 → tile 切分 → 同一目标跨 tile 多个框/多个细类 → 聚合归并）。当前只验证了"整图级多框"（去重/投票），**没验证"跨 tile 切片"**。

所以：**E 的工作是"完成度约 70%"**——聚合算法本身可靠、性能达标，但核心场景（跨 tile）和真实 10K 测试未完成。

## 6. 给 E 的整改清单

1. **补交 WP3 评估脚本**（`scripts/eval_wp3_global_aggregation.py` 或类似），输入输出可复现，记录输入数据 SHA；
2. **补 FDR(score≥0.25) 和聚合前基线的评估**，与报告数字一致；
3. **明确 macro_recall 口径**（建议对齐官方 V1.6：大类内细类简单平均）；
4. 用合成 10K 大图跑通 `fusion="global"` 路径（跨 tile），记录跨细类冲突归并的实际触发情况——这是主线 3 的核心证据；
5. 评估存档加入 `git`（当前 `wp3_real_eval.json` 已提交但脚本缺失）。

## 7. 合并决策

**结论：可以保留合并（代码/测试合格），但 E 的工作标记为"partial"**：
- 聚合算法（WP1/2/5/6）✅ 可用，不阻塞主线 2；
- WP3 真实数据评估的**报告数字暂不采信**（不可复现），待 E 补脚本后复核；
- 主线 3 核心场景（跨 tile）待 10K 图到位后验证。

**下一步**：把本评审给 E，让他补评估脚本 + 用合成 10K 图跑通跨 tile 路径。同时我继续 P2 三折验证（fold1/fold2 训练中）。
