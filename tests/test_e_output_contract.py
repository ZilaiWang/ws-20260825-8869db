"""WP5：主线 2 输出契约（GlobalObject / global_object_manifest）测试。

契约核心：聚合后每对象一个 ``GlobalObject``，保证"每个目标只处理一次"，
下游（容易对象校准输出 / 困难对象完整重裁）直接消费，无需再处理
跨 tile 重复或跨细类冲突。对象携带：全局框、细类、score、evidence、
来源 tile 列表、各细类投票明细。
"""

from __future__ import annotations

import pytest

import rsdet.pipeline.mock_model  # noqa: F401  触发 mock 模型注册
from rsdet.contracts import Prediction, TileRecord
from rsdet.models.registry import build_model
from rsdet.pipeline.large_image import PipelineConfig, run_pipeline
from rsdet.postprocess.global_aggregation import (
    GlobalObject,
    aggregate,
    global_object_manifest,
)
from rsdet.tiling.synthetic import generate_synthetic_scene


def _box(x, y, w, h, cat, score, source=None):
    p = {"x": x, "y": y, "width": w, "height": h, "category_id": cat, "score": score}
    if source is not None:
        p["source_tile_id"] = source
    return p


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


def _tile_metadata_for_mock(scene):
    def _fn(tile):
        boxes = []
        for obj in scene.objects:
            if tile.tile_id not in obj.tile_ids:
                continue
            gx1, gy1, gx2, gy2 = obj.bbox
            lx1 = max(0.0, gx1 - tile.x_offset)
            ly1 = max(0.0, gy1 - tile.y_offset)
            lx2 = min(float(tile.width), gx2 - tile.x_offset)
            ly2 = min(float(tile.height), gy2 - tile.y_offset)
            if lx2 <= lx1 or ly2 <= ly1:
                continue
            boxes.append(
                {"bbox": [lx1, ly1, lx2, ly2], "category_id": obj.category_id, "score": 1.0}
            )
        return {"gt_boxes": boxes}

    return _fn


# ---------------------------------------------------------------------------
# global_object_manifest：字段契约
# ---------------------------------------------------------------------------


class TestManifestFields:
    def test_single_tile_object_fields(self):
        """单 tile 对象：全局框 / 细类 / score / evidence / 来源 / 投票全齐。"""
        pred = _make_tile_prediction(
            tile_id=0, boxes=[[100, 200, 300, 400]], scores=[0.9], labels=[9]
        )
        rec = _make_tile_record(tile_id=0, x_offset=500, y_offset=300)
        objs = global_object_manifest(
            [pred],
            [rec],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert len(objs) == 1
        o = objs[0]
        assert isinstance(o, GlobalObject)
        assert o.object_id == 0
        assert o.parent_image_id == 0
        assert o.bbox_xyxy == [600.0, 500.0, 800.0, 700.0]
        assert o.category_id == 9
        assert o.score == 0.9
        assert o.evidence == 1
        assert o.source_tile_ids == [0]
        assert o.category_votes == {9: 0.9}

    def test_cross_tile_object_dedup_tracks_sources(self):
        """跨 2 tile 同一目标：evidence=2、来源=[0,1]、投票为两 tile 之和。"""
        pred0 = _make_tile_prediction(
            tile_id=0, boxes=[[500, 500, 600, 600]], scores=[0.9], labels=[9]
        )
        pred1 = _make_tile_prediction(
            tile_id=1, boxes=[[400, 500, 500, 600]], scores=[0.85], labels=[9]
        )
        rec0 = _make_tile_record(tile_id=0, x_offset=0, y_offset=0)
        rec1 = _make_tile_record(tile_id=1, x_offset=100, y_offset=0)
        objs = global_object_manifest(
            [pred0, pred1],
            [rec0, rec1],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert len(objs) == 1
        o = objs[0]
        assert o.evidence == 2
        assert o.source_tile_ids == [0, 1]
        assert o.category_votes == {9: 1.75}  # 0.9 + 0.85

    def test_conflict_keeps_vote_breakdown(self):
        """跨型号冲突：投票归并出一个型号，明细保留两个候选型号的票数。"""
        box = [500, 500, 600, 600]
        pred0 = _make_tile_prediction(tile_id=0, boxes=[box], scores=[0.8], labels=[22])
        pred1 = _make_tile_prediction(tile_id=1, boxes=[box], scores=[0.6], labels=[9])
        rec0 = _make_tile_record(tile_id=0)
        rec1 = _make_tile_record(tile_id=1)
        objs = global_object_manifest(
            [pred0, pred1],
            [rec0, rec1],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert len(objs) == 1
        o = objs[0]
        assert o.category_id == 22  # 0.8 > 0.6，加权投票 SU-34 胜
        assert o.category_votes == {22: 0.8, 9: 0.6}
        assert o.evidence == 2

    def test_adjacent_distinct_objects_separate_ids(self):
        """相邻不同目标：两个对象、object_id 唯一。"""
        pred0 = _make_tile_prediction(
            tile_id=0,
            boxes=[[0, 0, 20, 20], [30, 0, 50, 20]],
            scores=[0.8, 0.7],
            labels=[9, 9],
        )
        rec0 = _make_tile_record(tile_id=0)
        objs = global_object_manifest(
            [pred0],
            [rec0],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert len(objs) == 2
        assert objs[0].object_id != objs[1].object_id

    def test_empty_returns_empty_list(self):
        objs = global_object_manifest(
            [_make_tile_prediction(tile_id=0)],
            [_make_tile_record(tile_id=0)],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert objs == []

    def test_thresholds_and_limits_apply(self):
        """score_threshold / max_detections 作用于对象清单。"""
        pred = _make_tile_prediction(
            tile_id=0,
            boxes=[[10, 10, 110, 110], [500, 500, 600, 600], [1000, 1000, 1100, 1100]],
            scores=[0.9, 0.05, 0.7],
            labels=[9, 9, 9],
        )
        rec = _make_tile_record(tile_id=0)
        objs = global_object_manifest(
            [pred],
            [rec],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
            score_threshold=0.3,
            max_detections=1,
        )
        assert len(objs) == 1
        assert objs[0].score == 0.9


# ---------------------------------------------------------------------------
# aggregate：带来源时补 source_tile_ids + category_votes
# ---------------------------------------------------------------------------


class TestAggregateSourceTracking:
    def test_source_info_attached_when_proposals_carry_it(self):
        """proposal 带 source_tile_id → 输出对象带来源列表 + 投票明细。"""
        props = [
            _box(0, 0, 100, 100, 9, 0.8, source=0),
            _box(5, 5, 100, 100, 9, 0.7, source=1),
            _box(10, 10, 100, 100, 22, 0.5, source=2),
        ]
        out = aggregate(props)
        assert len(out) == 1
        assert out[0]["source_tile_ids"] == [0, 1, 2]
        assert out[0]["category_votes"] == {9: 1.5, 22: 0.5}

    def test_source_info_absent_without_source(self):
        """proposal 不带来源 → 对象不含来源字段（向后兼容）。"""
        props = [_box(0, 0, 100, 100, 9, 0.8), _box(5, 5, 100, 100, 9, 0.7)]
        out = aggregate(props)
        assert len(out) == 1
        assert "source_tile_ids" not in out[0]
        assert "category_votes" not in out[0]
        assert out[0]["evidence"] == 2


# ---------------------------------------------------------------------------
# run_pipeline(collect_objects=True)：pipeline 层直接产出契约
# ---------------------------------------------------------------------------


class TestManifestThroughPipeline:
    def test_collect_objects_returns_three_tuple(self):
        """collect_objects=True → (prediction, timing, objects) 三元组。"""
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
        prediction, timing, objs = run_pipeline(
            scene.image,
            detector,
            config=config,
            tile_metadata_fn=_tile_metadata_for_mock(scene),
            collect_objects=True,
        )
        assert len(prediction.boxes_xyxy) == len(scene.objects)
        assert len(objs) == len(scene.objects)
        # 与 Prediction 投影一致：框/类/分对齐
        assert [o.category_id for o in objs] == list(prediction.labels)
        # 每个对象都有来源与证据
        for o in objs:
            assert o.evidence >= 1
            assert o.source_tile_ids

    def test_collect_objects_rejects_tile_fusion(self):
        """collect_objects 仅限 global 路径。"""
        scene = generate_synthetic_scene(
            image_size=2048,
            tile_size=1024,
            overlap=128,
            num_ships=2,
            num_aircraft=2,
            num_vehicles=1,
            seed=42,
        )
        detector = build_model("mock", {"init_args": {}})
        detector.eval()
        with pytest.raises(ValueError, match="collect_objects"):
            run_pipeline(
                scene.image,
                detector,
                config=PipelineConfig(fusion="tile"),
                tile_metadata_fn=_tile_metadata_for_mock(scene),
                collect_objects=True,
            )
