# 工作总结(1)-E

日期：2026-08-16
成员：E（潘扬东杰）
分工：主线 3——大图推理、融合和测速

> 本文件是 E 的阶段性工作总结，面向团队其他成员。
> 正式的评审整改响应报告见 [`E_MAINLINE3_RESPONSE_20260815.md`](E_MAINLINE3_RESPONSE_20260815.md)。

---

## 1. 我的任务（分工文件 E）

按 [`XH-202625_20260715.md`](../../../D:/QQ%20download/files/XH-202625_20260715.md) 第 165-192 行，E 负责"大图推理、融合和测速"：

- 构造带可追溯坐标的 **10000×10000 合成大图**（覆盖目标位于切片内部、边界、多切片重叠区）；
- 跑通 **重叠切片 → 批量推理 → 全局坐标恢复 → 跨切片去重/融合 → COCO detection JSON 导出**；
- 比较一组**准确率档**和一组**速度档**的切片参数，记录切片数量与资源占用；
- 建立 **model_only** 与 **pipeline** 两类计时，正式判断以图像读入内存后的完整 pipeline 为准；
- M1 可用前用 mock 完成坐标/融合测试，M1 可用后接入真实模型做端到端验证；
- 对越界框、空预测、重复目标、不同图像通道、显存不足做基本防护；
- 只负责切片/重叠率/batch size/融合方式/推理精度/测速等**工程参数**，不承担模型训练。

验收硬指标（官方 V1.6）：Recall ≥ 0.85 / FDR ≤ 0.20 / 单幅 10K 图推理 ≤ 20s（RTX 3090）。

---

## 2. 我完成了什么

### 2.1 合成 10K 大图 + 真值

- `src/rsdet/tiling/synthetic.py` 的 `generate_synthetic_scene()` 生成 10000×10000 带 GT 的合成图（船/飞机/车辆，含边界与重叠区目标）。

### 2.2 大图流程闭环（切片→推理→坐标恢复→融合→导出）

| 环节 | 实现 | 状态 |
|---|---|---|
| 切片 | `src/rsdet/tiling/slicer.py` `generate_tiles` | ✅ |
| 批量推理 | `src/rsdet/engine/predictor.py` `predict_batches` | ✅ |
| 坐标恢复 | `src/rsdet/tiling/coordinates.py` `tile_to_full` | ✅ |
| 跨 tile 去重/融合 | `src/rsdet/postprocess/global_aggregation.py`（全局聚合，E 主线 3 核心） | ✅ |
| 越界防护 | `clip_bbox`（越界框裁回图内） | ✅ |
| COCO detection JSON 导出 | `src/rsdet/predictions.py` `predictions_to_coco_records` / `write_coco_predictions` | ✅ |
| 统一格式校验 | `src/rsdet/predictions.py` `validate_coco_prediction_records`（类别 0-24、score∈[0,1]、bbox 不越界） | ✅ |
| 端到端编排 | `src/rsdet/pipeline/large_image.py` `run_pipeline` | ✅ |

### 2.3 全局聚合（主线 3 核心创新）

同一目标在 10K 大图上被预测成多个框/多细类时，聚合为一个对象并投票选最可靠类别：

- `spatial_cluster`：跨 tile 空间聚类（同一目标归为一簇）；
- `class_vote`：簇内按分数加权的细类投票；
- `fuse_global_predictions`：聚合入口，输出全局 `Prediction`。

**验证**：合成 10K 图，22 个跨 tile 目标 → **22/22（100%）归并为单对象**，0 误合并；evidence=出现 tile 数。脚本 `scripts/validate_cross_tile_merge.py`。

### 2.4 M1 接入后的端到端实测（工程 smoke）

真实 M1（YOLO26s best.pt）+ RTX 3090 + 合成 10K 图，`perf_counter + torch.cuda.synchronize` 计时，1 warmup + 5 measured，每 run 硬门禁 `total_after_read ≤ 20s`：

| 项 | 值 |
|---|---|
| 冻结几何 | tile 1280 / overlap 256 → 恰好 100 tiles / batch 8 |
| **total_after_read** | **p50 = 1.35s，p95 = 1.41s，max = 1.42s，6/6 ≤ 20s ✅** |
| model-only / tiling / fusion | ≈1.17s / ≈0.13s / ≈0.00s |
| 峰值显存 | 0.26 GiB |
| COCO 导出 | 17 个聚合对象 → 17 条 COCO detection → 统一校验 **passed** |

存档：`outputs/e_wp3/e2e_result_fold0_3090.json`、`e2e_result_coco_smoke_3090.json`。

### 2.5 切片参数对比（准确率档 / 速度档）

| 档 | tile | overlap | stride | batch | 切片数 | p50 | p95 | max | 显存 | ≤20s |
|---|---|---|---|---|---|---|---|---|---|---|
| 速度 | 1280 | 128 | 1152 | 16 | 81 | 1.03s | 1.14s | 1.15s | 0.26GiB | ✅ |
| 冻结 | 1280 | 256 | 1024 | 8 | 100 | 1.28s | 1.40s | 1.41s | 0.26GiB | ✅ |
| 准确率 | 1024 | 384 | 640 | 8 | 256 | 2.34s | 2.48s | 2.50s | 0.26GiB | ✅ |

- 三档全部 ≤20s（最差 max=2.50s，余量 8×）；
- 显存三档恒 0.26GiB（由模型自身占用主导，切片参数不抬升）。

存档：`outputs/e_wp3/param_{speed,freeze,accuracy}.json`。

### 2.6 防护项

越界框（clip_bbox）、空预测、重复目标（跨 tile 归并）、多通道图像、显存峰值监控均已处理。

---

## 3. 涉及的脚本

| 脚本 / 模块 | 作用 |
|---|---|
| `scripts/eval_wp4_end2end_10k_3090.py` | **E 端到端实测入口**：合成/真实图 + M1/mock + 计时 + COCO 导出 + 统一校验（`--coco`）。服务器同步版 `/root/autodl-tmp/e2e/run_e2e_10k.py` |
| `src/rsdet/pipeline/large_image.py` | `run_pipeline` 编排：切片→推理→融合→计时；`PipelineTiming` 分段计时 |
| `src/rsdet/pipeline/m1_wrapper.py` | M1（YOLO26s）适配器，CUDA 推理 |
| `src/rsdet/pipeline/mock_model.py` | 占位检测器（M1 可用前验证链路） |
| `src/rsdet/tiling/slicer.py` | 滑窗切片坐标生成 |
| `src/rsdet/tiling/synthetic.py` | 合成 10K 大图 + GT 生成 |
| `src/rsdet/tiling/coordinates.py` | `tile_to_full` 坐标恢复、`clip_bbox` 越界防护 |
| `src/rsdet/engine/predictor.py` | 批量推理入口 |
| `src/rsdet/postprocess/global_aggregation.py` | **主线 3 核心**：空间聚类 + 细类投票 + 全局聚合 |
| `src/rsdet/postprocess/tile_fusion.py` | 基线 tile 融合（细类分组 NMS） |
| `src/rsdet/predictions.py` | 统一预测格式、COCO 导出、统一校验 |
| `scripts/validate_cross_tile_merge.py` | 跨 tile 冲突归并验证（22/22） |
| `scripts/eval_wp3_global_aggregation.py` | A 提供的官方细类口径评估 |
| `tests/test_predictions.py` 等 | 单测（统一格式/COCO 导出/切片/融合） |

---

## 4. 关键结论与边界

- **20s 硬门禁余量充足**：三档切片参数最差 max=2.50s，余量 8×；工程 smoke 全通过。
- **不能宣称"官方时延通过"**：输入为合成图、权重为工程 best.pt。官方结论需 `real_official` 10K 图 + 正式 `last.pt` + 独占 GPU（当前 `real_official` 注册表为空）。
- 合成图让 M1 检出的是假阳性（纯色块场景），精度结论以真实 OOF 数据为准。
- 真实 10K 图到手后可直接用 `--image-path` 跑真实图端到端（脚本已就绪）。

---

## 5. 建议他人关注

- 主线 3 全局聚合的 `global_aggregation.py` 是 E 的核心贡献，A 的评估脚本已接入；
- COCO 导出与统一校验已打通，A 的统一评测可直接消费 E 的预测结果；
- 需要团队协助：官方真实 10K 图（C 服务器 `real_10000x10000.png`）到手后补一次真实图端到端。

---

*提交：`feat/e-10k-pipeline` 分支，commit `169c705`。*
