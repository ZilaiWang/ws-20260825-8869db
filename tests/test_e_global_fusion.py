"""主线 3 全局融合（fuse_global_predictions）的单元 + 端到端测试。

覆盖：跨 tile 去重、跨型号冲突归并、坐标恢复、边界裁剪、输入校验，
以及通过 run_pipeline 端到端跑通（mock 检测器 + 合成切片图）。
"""

from __future__ import annotations

import pytest

import rsdet.pipeline.mock_model  # noqa: F401  触发 mock 模型注册
from rsdet.contracts import Prediction, TileRecord
from rsdet.models.registry import build_model
from rsdet.pipeline.large_image import PipelineConfig, run_pipeline
from rsdet.postprocess.global_aggregation import fuse_global_predictions
from rsdet.tiling.synthetic import generate_synthetic_scene


def _make_tile_prediction(
    tile_id: int = 0,
    boxes: list | None = None,
    scores: list | None = None,
    labels: list | None = None,
) -> Prediction:
    return Prediction(
        image_id=tile_id,
        boxes_xyxy=boxes or [],
        scores=scores or [],
        labels=labels or [],
    )


def _make_tile_record(
    tile_id: int = 0,
    x_offset: int = 0,
    y_offset: int = 0,
    width: int = 1024,
    height: int = 1024,
    parent_image_id: int = 0,
) -> TileRecord:
    return TileRecord(
        tile_id=tile_id,
        parent_image_id=parent_image_id,
        x_offset=x_offset,
        y_offset=y_offset,
        width=width,
        height=height,
    )


def _tile_metadata_for_mock(scene, dual_class: tuple[int, int, float] | None = None):
    """为每个 tile 生成 gt_boxes（mock 使用）。可选给指定目标注入双类别冲突。

    dual_class: (object_index, alt_category_id, alt_score)，
        该目标在所有出现过的 tile 上同时输出两个类别的同一框（真类 score 1.0）。
    """

    def _fn(tile):
        boxes = []
        for idx, obj in enumerate(scene.objects):
            if tile.tile_id not in obj.tile_ids:
                continue
            gx1, gy1, gx2, gy2 = obj.bbox
            lx1 = max(0.0, gx1 - tile.x_offset)
            ly1 = max(0.0, gy1 - tile.y_offset)
            lx2 = min(float(tile.width), gx2 - tile.x_offset)
            ly2 = min(float(tile.height), gy2 - tile.y_offset)
            if lx2 <= lx1 or ly2 <= ly1:
                continue
            bbox = [lx1, ly1, lx2, ly2]
            boxes.append({"bbox": bbox, "category_id": obj.category_id, "score": 1.0})
            if dual_class is not None and idx == dual_class[0]:
                # 同一框再报一个冲突类别（低分）→ 制造跨类别冲突
                boxes.append(
                    {
                        "bbox": list(bbox),
                        "category_id": dual_class[1],
                        "score": dual_class[2],
                    }
                )
        return {"gt_boxes": boxes}

    return _fn


# ---------------------------------------------------------------------------
# 单元：fuse_global_predictions
# ---------------------------------------------------------------------------


class TestFuseCoordinateRestore:
    def test_single_tile_restores_global_coords(self):
        """tile 局部框 + offset = 全局框。"""
        pred = _make_tile_prediction(
            tile_id=0, boxes=[[100, 200, 300, 400]], scores=[0.9], labels=[9]
        )
        rec = _make_tile_record(tile_id=0, x_offset=500, y_offset=300)
        fused = fuse_global_predictions(
            [pred],
            [rec],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert len(fused.boxes_xyxy) == 1
        assert fused.boxes_xyxy[0] == [600.0, 500.0, 800.0, 700.0]
        assert fused.scores[0] == 0.9
        assert fused.labels[0] == 9
        assert fused.image_id == 0

    def test_cross_tile_same_object_dedup(self):
        """同一目标跨 2 tile、同型号 → 聚成 1 对象。"""
        pred0 = _make_tile_prediction(
            tile_id=0, boxes=[[500, 500, 600, 600]], scores=[0.9], labels=[9]
        )
        pred1 = _make_tile_prediction(
            tile_id=1, boxes=[[400, 500, 500, 600]], scores=[0.85], labels=[9]
        )
        rec0 = _make_tile_record(tile_id=0, x_offset=0, y_offset=0)
        rec1 = _make_tile_record(tile_id=1, x_offset=100, y_offset=0)
        fused = fuse_global_predictions(
            [pred0, pred1],
            [rec0, rec1],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert len(fused.boxes_xyxy) == 1
        assert fused.scores[0] == 0.9  # canonical 取最高分框

    def test_cross_tile_conflict_resolved_by_vote(self):
        """同一目标跨 2 tile、报成不同型号 → 投票归并成 1 型号。"""
        box0 = [500, 500, 600, 600]
        pred0 = _make_tile_prediction(
            tile_id=0,
            boxes=[box0],
            scores=[0.8],
            labels=[22],  # SU-34
        )
        pred1 = _make_tile_prediction(
            tile_id=1,
            boxes=[box0],
            scores=[0.6],
            labels=[9],  # TU-160
        )
        rec0 = _make_tile_record(tile_id=0)
        rec1 = _make_tile_record(tile_id=1)
        fused = fuse_global_predictions(
            [pred0, pred1],
            [rec0, rec1],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert len(fused.boxes_xyxy) == 1
        assert fused.labels[0] == 22  # score 加权后 SU-34 胜

    def test_adjacent_distinct_objects_not_fused(self):
        """相邻但不同的同型号目标 → 不错误融合，输出 2 对象。"""
        pred0 = _make_tile_prediction(
            tile_id=0,
            boxes=[[0, 0, 20, 20], [30, 0, 50, 20]],
            scores=[0.8, 0.7],
            labels=[9, 9],
        )
        rec0 = _make_tile_record(tile_id=0)
        fused = fuse_global_predictions(
            [pred0],
            [rec0],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert len(fused.boxes_xyxy) == 2

    def test_out_of_bounds_clipped(self):
        """越界框裁剪到图像边界内。"""
        pred = _make_tile_prediction(
            tile_id=0, boxes=[[9900, 9900, 10100, 10100]], scores=[0.9], labels=[9]
        )
        rec = _make_tile_record(tile_id=0)
        fused = fuse_global_predictions(
            [pred],
            [rec],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert len(fused.boxes_xyxy) == 1
        box = fused.boxes_xyxy[0]
        assert 0 <= box[0] <= 10000 and 0 <= box[2] <= 10000
        assert 0 <= box[1] <= 10000 and 0 <= box[3] <= 10000

    def test_fully_out_of_bounds_dropped(self):
        """完全在图像外 → 裁剪后退化，被丢弃。"""
        pred = _make_tile_prediction(
            tile_id=0, boxes=[[20000, 20000, 21000, 21000]], scores=[0.9], labels=[9]
        )
        rec = _make_tile_record(tile_id=0)
        fused = fuse_global_predictions(
            [pred],
            [rec],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert len(fused.boxes_xyxy) == 0

    def test_empty_predictions(self):
        """无预测 → 空 Prediction。"""
        fused = fuse_global_predictions(
            [_make_tile_prediction(tile_id=0)],
            [_make_tile_record(tile_id=0)],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert fused.image_id == 0
        assert len(fused.boxes_xyxy) == 0


class TestFuseThresholds:
    def test_score_threshold_filters_low_score_objects(self):
        """聚合后按分值过滤（低分对象被丢弃）。"""
        pred = _make_tile_prediction(
            tile_id=0,
            boxes=[[10, 10, 110, 110], [500, 500, 600, 600]],
            scores=[0.9, 0.05],
            labels=[9, 9],
        )
        rec = _make_tile_record(tile_id=0)
        fused = fuse_global_predictions(
            [pred],
            [rec],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
            score_threshold=0.3,
        )
        assert len(fused.boxes_xyxy) == 1
        assert fused.scores[0] == 0.9

    def test_max_detections_limits_output(self):
        """max_detections 限制输出对象数。"""
        pred = _make_tile_prediction(
            tile_id=0,
            boxes=[[10, 10, 110, 110], [500, 500, 600, 600], [1000, 1000, 1100, 1100]],
            scores=[0.9, 0.7, 0.5],
            labels=[9, 9, 9],
        )
        rec = _make_tile_record(tile_id=0)
        fused = fuse_global_predictions(
            [pred],
            [rec],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
            max_detections=1,
        )
        assert len(fused.boxes_xyxy) == 1
        assert fused.scores[0] == 0.9


class TestFuseValidation:
    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError):
            fuse_global_predictions(
                [_make_tile_prediction(), _make_tile_prediction()],
                [_make_tile_record()],
                parent_image_id=0,
                image_width=10000,
                image_height=10000,
            )

    def test_raises_on_duplicate_tile_id(self):
        with pytest.raises(ValueError):
            fuse_global_predictions(
                [_make_tile_prediction(tile_id=0), _make_tile_prediction(tile_id=0)],
                [_make_tile_record(tile_id=0), _make_tile_record(tile_id=0)],
                parent_image_id=0,
                image_width=10000,
                image_height=10000,
            )

    def test_raises_on_prediction_image_mismatch(self):
        with pytest.raises(ValueError):
            fuse_global_predictions(
                [_make_tile_prediction(tile_id=99)],
                [_make_tile_record(tile_id=0)],
                parent_image_id=0,
                image_width=10000,
                image_height=10000,
            )

    def test_raises_on_parent_mismatch(self):
        with pytest.raises(ValueError):
            fuse_global_predictions(
                [_make_tile_prediction(tile_id=0)],
                [_make_tile_record(parent_image_id=7)],
                parent_image_id=0,
                image_width=10000,
                image_height=10000,
            )

    def test_raises_on_invalid_score(self):
        pred = _make_tile_prediction(
            tile_id=0, boxes=[[10, 10, 100, 100]], scores=[1.5], labels=[9]
        )
        with pytest.raises(ValueError):
            fuse_global_predictions(
                [pred],
                [_make_tile_record(tile_id=0)],
                parent_image_id=0,
                image_width=10000,
                image_height=10000,
            )

    def test_raises_on_invalid_label(self):
        pred = _make_tile_prediction(
            tile_id=0, boxes=[[10, 10, 100, 100]], scores=[0.9], labels=[99]
        )
        with pytest.raises(ValueError):
            fuse_global_predictions(
                [pred],
                [_make_tile_record(tile_id=0)],
                parent_image_id=0,
                image_width=10000,
                image_height=10000,
            )


# ---------------------------------------------------------------------------
# 端到端：run_pipeline(fusion="global")
# ---------------------------------------------------------------------------


class TestGlobalFusionEndToEnd:
    def test_recovers_every_object_exactly_once(self):
        """全局聚合：每个目标恰好一个对象（无重复、无遗漏）。"""
        scene = generate_synthetic_scene(
            image_size=4096,
            tile_size=1024,
            overlap=128,
            num_ships=8,
            num_aircraft=15,
            num_vehicles=5,
            seed=42,
        )
        detector = build_model("mock", {"init_args": {}})
        detector.eval()

        config = PipelineConfig(tile_size=1024, overlap=128, batch_size=8, fusion="global")
        prediction, _ = run_pipeline(
            scene.image,
            detector,
            config=config,
            tile_metadata_fn=_tile_metadata_for_mock(scene),
        )
        assert len(prediction.boxes_xyxy) == len(scene.objects)

    def test_no_worse_than_baseline_tile_fusion(self):
        """全局聚合输出框数不劣于基线 tile_fusion。"""
        scene = generate_synthetic_scene(
            image_size=4096,
            tile_size=1024,
            overlap=128,
            num_ships=8,
            num_aircraft=15,
            num_vehicles=5,
            seed=42,
        )
        detector = build_model("mock", {"init_args": {}})
        detector.eval()

        base_pred, _ = run_pipeline(
            scene.image,
            detector,
            config=PipelineConfig(tile_size=1024, overlap=128, batch_size=8, fusion="tile"),
            tile_metadata_fn=_tile_metadata_for_mock(scene),
        )
        global_pred, _ = run_pipeline(
            scene.image,
            detector,
            config=PipelineConfig(tile_size=1024, overlap=128, batch_size=8, fusion="global"),
            tile_metadata_fn=_tile_metadata_for_mock(scene),
        )
        assert len(global_pred.boxes_xyxy) <= len(base_pred.boxes_xyxy)

    def test_resolves_cross_tile_conflict(self):
        """某个目标被同时报成两个型号 → 聚合后只有 1 个对象、选回真类。"""
        scene = generate_synthetic_scene(
            image_size=4096,
            tile_size=1024,
            overlap=128,
            num_ships=8,
            num_aircraft=15,
            num_vehicles=5,
            seed=42,
        )
        detector = build_model("mock", {"init_args": {}})
        detector.eval()

        conflict_idx = 5  # 选第 5 个目标制造冲突
        # 冲突类必须选场景里其他目标没用过的类别，否则 alt_class 出现在输出是正常的
        used = {o.category_id for o in scene.objects}
        alt_class = next(c for c in range(25) if c not in used)
        config = PipelineConfig(tile_size=1024, overlap=128, batch_size=8, fusion="global")
        prediction, _ = run_pipeline(
            scene.image,
            detector,
            config=config,
            tile_metadata_fn=_tile_metadata_for_mock(
                scene, dual_class=(conflict_idx, alt_class, 0.5)
            ),
        )
        assert len(prediction.boxes_xyxy) == len(scene.objects)
        true_class = scene.objects[conflict_idx].category_id
        assert true_class in prediction.labels  # 真类被保留
        assert alt_class not in prediction.labels  # 冲突的低分类被归并掉

    def test_empty_scene_returns_empty(self):
        """空场景 → 空 Prediction。"""
        scene = generate_synthetic_scene(
            image_size=1024,
            tile_size=1024,
            overlap=128,
            num_ships=0,
            num_aircraft=0,
            num_vehicles=0,
            seed=42,
        )
        detector = build_model("mock", {"init_args": {}})
        detector.eval()
        prediction, _ = run_pipeline(
            scene.image,
            detector,
            config=PipelineConfig(tile_size=1024, overlap=128, batch_size=1, fusion="global"),
            tile_metadata_fn=_tile_metadata_for_mock(scene),
        )
        assert len(prediction.boxes_xyxy) == 0

    def test_fusion_timing_small(self):
        """小图全局聚合耗时远低于预算。"""
        scene = generate_synthetic_scene(
            image_size=2048,
            tile_size=1024,
            overlap=128,
            num_ships=3,
            num_aircraft=5,
            num_vehicles=2,
            seed=42,
        )
        detector = build_model("mock", {"init_args": {}})
        detector.eval()
        _, timing = run_pipeline(
            scene.image,
            detector,
            config=PipelineConfig(tile_size=1024, overlap=128, batch_size=4, fusion="global"),
            tile_metadata_fn=_tile_metadata_for_mock(scene),
        )
        assert timing.fusion_s < 1.0  # 预算 20s 的 5%
