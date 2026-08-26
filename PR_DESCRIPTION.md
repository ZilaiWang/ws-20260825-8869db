feat(e): 主线3 全局对象聚合与条件计算（WP1/2/5/6）+ 真实数据 GT 评估

---

## 关联 Issue 和目标

- Issue：主线3（全局对象重构与条件计算）——同一目标在 10K 大图被预测为多个框/多个细类，聚合为一个对象并选择最可靠的类别
- 目标：为下游（主线2"每个目标只处理一次"的输入）提供对象级输出，满足 10K 端到端 ≤20s（3090）预算

## 主要修改

- **修改 2 个**：
  - `src/rsdet/postprocess/tile_fusion.py` — 适配 master 规范版 API（fine/coarse NMS 参数化）
  - `src/rsdet/pipeline/large_image.py` — 新增 `fusion="global"` 路径 + `collect_objects` 契约输出
- **新建 1 个核心模块 + 5 个测试文件**：
  - `src/rsdet/postprocess/global_aggregation.py` — 全局聚合核心：
    - `spatial_cluster`（网格哈希加速 union-find）
    - `_iou_subcluster`（语义门 IoU 连通分量）
    - `class_vote`（score 加权细类投票）
    - `class_aware_nms`（同类内 numpy 向量化 NMS）
    - `aggregate` / `fuse_global_predictions` / `global_object_manifest`
    - `GlobalObject` 契约（object_id / bbox / category / score / evidence / source_tile_ids / category_votes）
    - WP6 条件计算：`HardObjectCriteria` / `gate_hard_objects` / `re_detect_hard_objects`（困难对象完整重裁二次检测 + 证据融合）
  - `tests/test_e_global_fusion.py` — 全局融合测试
  - `tests/test_e_oof_aggregation.py` — 聚合语义测试
  - `tests/test_e_output_contract.py` — GlobalObject / pipeline 契约测试
  - `tests/test_e_hard_object_refinement.py` — WP6 条件计算测试
  - `tests/wp3_real_eval.json` — WP3 真实数据 GT 评估结果存档

## 验证

- 测试结果：**104 tests，全部通过**（本地 Windows）
  ```
  104 passed in 19.72s
  ```
- 性能：`class_aware_nms` 10K proposals 从 11.68s → 0.009s（numpy 向量化，~1300×）；整条聚合在真实 55,548 proposals 上 0.81s
- 数据验证（WP3）：真实 M1 OOF 预测（4,481 图 / 55,548 框）+ 真实 GT（20,933 对象）：
  - 去重 44%（每图 12.4 框 → 6.9 框）
  - Recall 96.9%、细类 Macro Recall 95.5%
  - score≥0.25 时 FDR 3.83%（目标 ≤17%）
  - 聚合几乎不伤召回（98.4% → 96.9%）
- 未触发场景：OOF 为整图级预测，无 tile 切分，故跨细类冲突归并（evidence 来自多细类）=0，该场景需在 10K 切片路径上验证

## 风险和回滚

- 本 PR 新增 `global_aggregation.py` 独立于现有模块；`large_image.py` 的 `fusion` 参数默认保持 `"tile"`，不影响既有 tile 路径
- 跨细类冲突归并的最终验证需在 10K 切片路径上完成
- 回滚方式：删除 `global_aggregation.py` + 恢复 `large_image.py` 默认路径即可

## 检查清单

- [x] 测试通过（104 tests / 104 passed）
- [x] 无个人路径、数据、权重和密钥
- [x] 只新增/修改 E 相关文件，未改动队友代码
- [x] `git diff` 已检查
