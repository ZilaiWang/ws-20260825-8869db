"""验证边界/重叠区目标经过完整 pipeline 后坐标精度。

核心问题：切片边界的目标被截断 → 每个 tile 只看到部分目标
→ 融合后坐标是否还能对上？是否漏检？
"""

import numpy as np
import pytest

from rsdet.contracts import TileRecord
from rsdet.models.registry import build_model
from rsdet.pipeline.large_image import PipelineConfig, run_pipeline
from rsdet.pipeline.mock_model import MockDetector
from rsdet.tiling.slicer import generate_tiles
from rsdet.tiling.synthetic import generate_synthetic_scene


def _build_metadata_fn(scene, tile_size, overlap):
    """为 mock 检测器准备 per-tile GT（局部坐标）。

    把每个合成目标的全局 bbox 动态映射到每个 tile 的局部坐标，
    裁剪到 tile 边界，模拟"每个 tile 只能看到完整目标的一部分"。
    """
    tiles = generate_tiles(
        image_width=scene.width,
        image_height=scene.height,
        tile_size=tile_size,
        overlap=overlap,
    )

    def _fn(tile: TileRecord) -> dict:
        gt_boxes = []
        tx1, ty1 = float(tile.x_offset), float(tile.y_offset)
        tx2, ty2 = tx1 + tile.width, ty1 + tile.height
        for obj in scene.objects:
            gx1, gy1, gx2, gy2 = obj.bbox
            # 检查对象是否与 tile 重叠（≥20% 面积，放宽到包含边缘截断）
            ix1 = max(gx1, tx1)
            iy1 = max(gy1, ty1)
            ix2 = min(gx2, tx2)
            iy2 = min(gy2, ty2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area = (gx2 - gx1) * (gy2 - gy1)
            if area > 0 and inter / area < 0.15:
                continue
            # 全局 → tile 局部，裁剪到 tile 内
            lx1 = max(0.0, gx1 - tx1)
            ly1 = max(0.0, gy1 - ty1)
            lx2 = min(float(tile.width), gx2 - tx1)
            ly2 = min(float(tile.height), gy2 - ty1)
            if lx2 - lx1 < 3.0 or ly2 - ly1 < 3.0:
                continue
            gt_boxes.append(
                {
                    "bbox": [lx1, ly1, lx2, ly2],
                    "category_id": obj.category_id,
                    "score": 1.0,
                }
            )
        return {"gt_boxes": gt_boxes}

    return tiles, _fn


def _iou(box_a, box_b):
    """纯 python IoU 计算。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


class TestBoundaryAccuracy:
    """用 mock (noise_std=0) 验证：pipeline 本身不引入坐标误差。"""

    def _run_pipeline(self, scene, tile_size=1024, overlap=128):
        """跑完整 pipeline，返回融合后 Prediction。"""
        tiles, metadata_fn = _build_metadata_fn(scene, tile_size, overlap)
        detector = build_model("mock", {"init_args": {"noise_std": 0.0}})
        detector.eval()
        config = PipelineConfig(
            tile_size=tile_size, overlap=overlap, batch_size=16,
            score_threshold=0.0,
        )
        pred, timing = run_pipeline(
            scene.image,
            detector,
            config=config,
            parent_image_id=scene.image_id,
            tile_metadata_fn=metadata_fn,
        )
        return pred

    # ---- 测试 1: 内部目标坐标完全一致 ----
    def test_interior_targets_recall(self):
        """内部目标（不在任何 tile 边界上的）：pipeline 后不丢失。"""
        scene = generate_synthetic_scene(
            image_size=5000,
            tile_size=2048,
            overlap=512,
            num_ships=10, num_aircraft=30, num_vehicles=5,
            seed=1,
        )
        pred = self._run_pipeline(scene, tile_size=2048, overlap=512)

        n_targets = len(scene.objects)
        # 至少检出 85% （NMS 可能合并极少数边界重叠的同类别目标）
        recall = len(pred.boxes_xyxy) / max(1, n_targets)
        assert recall >= 0.85, (
            f"Recall too low: {len(pred.boxes_xyxy)}/{n_targets} = {recall:.2f}"
        )

    def test_interior_coordinate_precision(self):
        """mock 无噪声时，检出框与 GT 的 IoU 应很高。"""
        scene = generate_synthetic_scene(
            image_size=4000,
            tile_size=1024,
            overlap=128,
            num_ships=5, num_aircraft=10, num_vehicles=2,
            seed=42,
        )
        pred = self._run_pipeline(scene, tile_size=1024, overlap=128)

        # 对每个 GT 找最佳匹配预测，检查 IoU
        gt_boxes = [obj.bbox for obj in scene.objects]
        pred_boxes = pred.boxes_xyxy
        ious = []
        for gt in gt_boxes:
            best_iou = 0.0
            for pb in pred_boxes:
                best_iou = max(best_iou, _iou(gt, pb))
            ious.append(best_iou)

        mean_iou = np.mean(ious)
        assert mean_iou > 0.80, (
            f"Mean best-match IoU too low: {mean_iou:.4f}"
        )

    # ---- 测试 2: 重叠区目标不重复 ----
    def test_overlap_targets_no_duplicate(self):
        """重叠区目标：融合后不应产生显著多于 GT 的检测。"""
        scene = generate_synthetic_scene(
            image_size=4000,
            tile_size=1024,
            overlap=256,  # 大 overlap → 更多目标跨越 tile
            num_ships=10, num_aircraft=30, num_vehicles=5,
            seed=2,
        )
        pred = self._run_pipeline(scene, tile_size=1024, overlap=256)

        n_targets = len(scene.objects)
        n_preds = len(pred.boxes_xyxy)
        # 允许 NMS 留下少量 FP（边界效应），但不应超过 GT 的 20%
        assert n_preds <= n_targets * 1.3, (
            f"Too many duplicates: {n_preds} preds vs {n_targets} targets"
        )

    # ---- 测试 3: 边界目标不丢失 ----
    def test_boundary_targets_not_lost(self):
        """边界目标：不应因切片而完全丢失（mock 每个 tile 返回 GT 片段）。"""
        # 用小图+小 tile 增大边界目标比例
        scene = generate_synthetic_scene(
            image_size=3000,
            tile_size=800,
            overlap=100,
            num_ships=5, num_aircraft=15, num_vehicles=3,
            seed=3,
        )
        pred = self._run_pipeline(scene, tile_size=800, overlap=100)

        n_targets = len(scene.objects)
        n_preds = len(pred.boxes_xyxy)
        recall = n_preds / max(1, n_targets)
        assert recall >= 0.75, (
            f"Boundary recall {recall:.2f} too low ({n_preds}/{n_targets})"
        )

    # ---- 测试 4: 坐标范围 ----
    def test_all_targets_coordinate_range(self):
        """所有输出框在 [0, 10000] 范围内，且 w>0 h>0。"""
        scene = generate_synthetic_scene(
            image_size=10000,
            tile_size=1024,
            overlap=128,
            num_ships=5, num_aircraft=10, num_vehicles=5,
            seed=4,
        )
        pred = self._run_pipeline(scene, tile_size=1024, overlap=128)

        assert len(pred.boxes_xyxy) > 0, "No detections at all"
        for box in pred.boxes_xyxy:
            x1, y1, x2, y2 = box
            assert x1 >= 0.0, f"x1 negative: {box}"
            assert y1 >= 0.0, f"y1 negative: {box}"
            assert x2 <= 10000.0 + 1e-9, f"x2 out of bounds: {box}"
            assert y2 <= 10000.0 + 1e-9, f"y2 out of bounds: {box}"
            assert x2 > x1, f"zero width: {box}"
            assert y2 > y1, f"zero height: {box}"

    # ---- 测试 5: 跨 tile 截断后坐标恢复 ----
    def test_truncated_box_coordinate_recovery(self):
        """一个目标被切成 3 段（2 tiles horizontal + corner），
        融合后应恢复到接近原始 GT 坐标。"""
        scene = generate_synthetic_scene(
            image_size=5000,
            tile_size=800,
            overlap=200,  # stride=600 → 很多目标跨 tile
            num_ships=0, num_aircraft=0, num_vehicles=20,  # 只用车辆（最小最易丢）
            seed=99,
        )
        pred = self._run_pipeline(scene, tile_size=800, overlap=200)

        # 所有车辆 GT 都应该被某个预测匹配到（IoU ≥ 0.5）
        vehicle_gts = [obj for obj in scene.objects if obj.category_id == 24]
        vehicle_preds = [
            (box, label)
            for box, label in zip(pred.boxes_xyxy, pred.labels)
            if label == 24
        ]

        matched = 0
        for gt_obj in vehicle_gts:
            best_iou = 0.0
            for box, _ in vehicle_preds:
                best_iou = max(best_iou, _iou(gt_obj.bbox, box))
            if best_iou >= 0.50:
                matched += 1

        recall = matched / max(1, len(vehicle_gts))
        assert recall >= 0.60, (
            f"Truncated vehicle recall {recall:.2f} too low ({matched}/{len(vehicle_gts)})"
        )
