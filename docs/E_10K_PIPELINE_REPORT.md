# E：10K 大图推理流水线报告

更新日期：2026-07-29

## 架构

```
大图 (numpy, 已在内存)
  → generate_tiles(tile_size, overlap)
  → 裁图 → InferenceSample[]
  → predict_batches(detector, samples, batch_size)
  → fuse_tile_predictions(preds, offsets, W, H)
  → Prediction (全局 xyxy)
  → to_coco_json() → 提交
```

各模块职责：
- `tiling/slicer.py` — 滑窗切片坐标生成（A 提供的公共模块）
- `tiling/coordinates.py` — tile_to_full / clip_bbox（A 提供的公共模块）
- `pipeline/large_image.py` — 编排：裁图、推理、计时
- `postprocess/tile_fusion.py` — 坐标恢复 + 细类分组 NMS
- `tiling/synthetic.py` — 合成 10K 大图 + GT 生成（E 自建测试工具）
- `pipeline/mock_model.py` — 占位检测器（E 自建测试工具）

## 推荐配置

| 参数 | 值 | 理由 |
|------|-----|------|
| tile_size | 2048 | 36 tiles，开销最小 |
| overlap | 128 | 边界目标不丢 |
| batch_size | 16 | RTX 3090 24GB 充裕 |
| score_threshold | 由 A 阈值扫描决定 | E 不过滤 |

## 性能（mock, RTX 4090D, 10000×10000）

| Config | P50 | P95 | Tiles | 开销 (切片+融合) |
|--------|-----|-----|-------|------------------|
| 1024/128/16 | 0.085s | 0.221s | 144 | 0.084s |
| 1024/256/16 | 0.111s | 0.155s | 169 | 0.110s |
| 2048/256/8  | 0.068s | 0.093s | 36  | 0.067s |
| 2048/128/16 | 0.069s | 0.103s | 36  | 0.068s |

M1 预估（RTX 3090）：
- 单 batch YOLO26-s 1024 推理：~0.12s
- 2048/128/16：36 tiles / 16 batch = 3 batches × 0.12s = 0.36s
- **总计预估：0.069s (overhead) + 0.36s (M1) ≈ 0.43s** ✅（远低于 17s 门槛）
- 最坏情况 1024/128：0.085s + 1.08s ≈ 1.17s，仍充裕

## 精度验证（mock, 无噪声）

- 内部目标 Recall ≥ 85%，Mean best-match IoU > 0.80
- 边界目标 Recall ≥ 75%
- 重叠区无重复框（det ≤ GT×1.3）
- 坐标恢复精确（mock 无噪声时与 GT 一致）
- 所有框 ∈ [0, 10000]
- 跨 tile 截断车辆目标：IoU≥0.5 匹配率 ≥ 60%

流水线本身不引入坐标误差。

## 图像格式防护

| 格式 | 处理方式 | 验证 |
|------|----------|------|
| 灰度 (1ch) | 自动复制为 3ch | ✅ |
| RGB (3ch) | 原样传递 | ✅ |
| RGBA (4ch) | 取前 3ch | ✅ |
| float32 | 原样传递 | ✅ |
| 小图 (< tile_size) | 单 tile 覆盖全图 | ✅ |

## 测试覆盖

**50 tests，全部通过。**

| 测试文件 | 测试数 | 覆盖内容 |
|----------|--------|----------|
| test_e_fusion.py | 20 | IoU 计算、NMS、坐标恢复、边界、分数过滤、细类隔离 |
| test_synthetic.py | 10 | 合成图生成、GT COCO 格式、可复现性、tile 追踪 |
| test_e_pipeline.py | 8 | 端到端 mock、模型可替换、空预测、计时字段 |
| test_e_boundary_accuracy.py | 6 | 内部目标精度、重叠区去重、边界目标不丢、截断恢复 |
| test_e_image_formats.py | 6 | 灰度/RGB/RGBA/float32/小图 |

## 已知问题与风险

1. **边界截断目标可能被真实模型漏检** → 非 pipeline 问题，是模型能力问题。解决方案：增大 overlap（256）或减小 tile_size（1024）可降低风险
2. **官方 10K 测试图未获得**，当前仅合成图验证。获取后可追加真实图测试
3. **显存峰值待 M1 接入后回填**，当前 mock 不占 GPU VRAM
4. **Benchmark 在 4090D 上跑**，RTX 3090 预计慢 ~30%，pipeline 开销仍 < 0.15s，远低于门槛
5. **灰度/RGBA 图已防护**，官方图大概率是 RGB PNG

## M1 接入方式

等 C 交付权重后，只需：

```python
from rsdet.pipeline.m1_wrapper import M1Wrapper

detector = M1Wrapper(weights_path="path/to/m1_best.pt")
detector.load()
detector.to("cuda")
detector.eval()

# 其余 pipeline 代码不变
from rsdet.pipeline.large_image import run_pipeline, PipelineConfig
pred, timing = run_pipeline(image, detector, config=PipelineConfig())
```

## M3（RT-DETR）接入方式

D 交付后同理：

```python
from rsdet.pipeline.m1_wrapper import M1Wrapper  # 可复用同一 wrapper 或写 M3Wrapper

detector = M1Wrapper(weights_path="path/to/m3_best.pt", imgsz=1024)
detector.load()
detector.to("cuda")
```

## 交付清单

| 文件 | 类型 | 状态 |
|------|------|------|
| `src/rsdet/postprocess/tile_fusion.py` | 核心实现 | ✅ |
| `src/rsdet/tiling/synthetic.py` | 测试工具 | ✅ |
| `src/rsdet/pipeline/__init__.py` | 包声明 | ✅ |
| `src/rsdet/pipeline/mock_model.py` | 测试工具 | ✅ |
| `src/rsdet/pipeline/large_image.py` | 核心实现 | ✅ |
| `src/rsdet/pipeline/m1_wrapper.py` | M1 接入封装 | ✅ |
| `scripts/run_e_pipeline.py` | CLI 入口 | ✅ |
| `scripts/e_benchmark_10k.py` | 测速入口 | ✅ |
| `tests/test_e_fusion.py` | 测试 | ✅ |
| `tests/test_synthetic.py` | 测试 | ✅ |
| `tests/test_e_pipeline.py` | 测试 | ✅ |
| `tests/test_e_boundary_accuracy.py` | 测试 | ✅ |
| `tests/test_e_image_formats.py` | 测试 | ✅ |
| `docs/E_10K_PIPELINE_REPORT.md` | 文档 | ✅ |
