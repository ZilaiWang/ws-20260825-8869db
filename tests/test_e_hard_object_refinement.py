"""WP6：条件计算 / 困难对象二次检测测试。

覆盖：困难对象判定（低置信 / 证据不足）、从原图完整重裁、复用检测器
二次检测、证据融合回写（重投票选类 / 采纳更完整框），以及时延预算。
"""

from __future__ import annotations

import numpy as np

import rsdet.pipeline.mock_model  # noqa: F401  触发 mock 模型注册
from rsdet.engine.predictor import predict_batches
from rsdet.models.registry import build_model
from rsdet.postprocess.global_aggregation import (
    GlobalObject,
    HardObjectCriteria,
    gate_hard_objects,
    re_detect_hard_objects,
)
from rsdet.tiling.synthetic import generate_synthetic_scene


def _scene_object_to_hard(obj, score: float = 0.2) -> GlobalObject:
    """把场景目标压成"低置信 + 单证据"的困难对象。"""
    x1, y1, x2, y2 = obj.bbox
    return GlobalObject(
        object_id=0,
        parent_image_id=0,
        bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
        category_id=obj.category_id,
        score=score,
        evidence=1,
        source_tile_ids=list(obj.tile_ids),
        category_votes={obj.category_id: score},
    )


def _crop_metadata_for_scene(scene):
    """给二次检测的 crop 生成 crop 局部坐标的 gt_boxes。"""

    def _fn(region, crop_id):
        x1, y1, x2, y2 = region
        boxes = []
        for obj in scene.objects:
            gx1, gy1, gx2, gy2 = obj.bbox
            lx1 = max(0.0, gx1 - x1)
            ly1 = max(0.0, gy1 - y1)
            lx2 = min(x2 - x1, gx2 - x1)
            ly2 = min(y2 - y1, gy2 - y1)
            if lx2 <= lx1 or ly2 <= ly1:
                continue
            boxes.append(
                {"bbox": [lx1, ly1, lx2, ly2], "category_id": obj.category_id, "score": 1.0}
            )
        return {"gt_boxes": boxes}

    return _fn


# ---------------------------------------------------------------------------
# 困难判定 gate
# ---------------------------------------------------------------------------


class TestHardObjectGate:
    def test_low_score_flagged(self):
        objs = [_scene_object_to_hard(o, score=0.2) for o in _scene(4).objects]
        idx = gate_hard_objects(objs, HardObjectCriteria(hard_score_threshold=0.3))
        assert idx == [0, 1, 2, 3]

    def test_high_score_not_flagged_when_evidence_ok(self):
        """高分且证据充足 → 不判困难（容易对象直接通过）。"""
        objs = [_scene_object_to_hard(o, score=0.9) for o in _scene(4).objects]
        idx = gate_hard_objects(
            objs, HardObjectCriteria(hard_score_threshold=0.3, hard_evidence_threshold=0)
        )
        assert idx == []

    def test_low_evidence_flagged_even_at_high_score(self):
        objs = [_scene_object_to_hard(o, score=0.9) for o in _scene(4).objects]
        idx = gate_hard_objects(
            objs, HardObjectCriteria(hard_score_threshold=0.3, hard_evidence_threshold=2)
        )
        assert idx == [0, 1, 2, 3]  # evidence=1 仍判困难

    def test_max_crops_caps_count(self):
        objs = [_scene_object_to_hard(o, score=0.2) for o in _scene(8).objects]
        idx = gate_hard_objects(objs, HardObjectCriteria(max_crops=3))
        assert len(idx) == 3

    def test_weakest_first(self):
        objs = [
            _scene_object_to_hard(o, score=s) for o, s in zip(_scene(3).objects, [0.8, 0.1, 0.4])
        ]
        idx = gate_hard_objects(objs, HardObjectCriteria(hard_score_threshold=0.9))
        assert idx == [1, 2, 0]  # 按 score 升序


# ---------------------------------------------------------------------------
# 二次检测 + 证据融合
# ---------------------------------------------------------------------------


class TestReDetectFusion:
    def test_hard_object_score_evidence_boosted(self):
        """低置信对象经重检后 score 提到真值、evidence 增加。"""
        scene = _scene(6)
        objs = [_scene_object_to_hard(o, score=0.2) for o in scene.objects]
        for i, o in enumerate(objs):
            o.object_id = i
        detector = build_model("mock", {"init_args": {}})
        detector.eval()
        updated, timing = re_detect_hard_objects(
            scene.image,
            objs,
            detect_batch=lambda samples: predict_batches(detector, samples, batch_size=4),
            crop_metadata_fn=_crop_metadata_for_scene(scene),
        )
        assert len(updated) == len(objs)
        for u, o in zip(updated, objs):
            assert u.score > o.score  # 0.2 → ~1.0
            assert u.evidence >= o.evidence
            assert u.category_id == o.category_id
        assert timing.n_hard == len(objs)
        assert timing.n_crops == len(objs)
        assert timing.refine_s < 1.0  # 20s 预算的 5%

    def test_bbox_adopts_full_redetection_box(self):
        """重检得到更高分框 → 采纳其坐标（完整重裁的框更全）。"""
        scene = _scene(1)
        objs = [_scene_object_to_hard(o, score=0.1) for o in scene.objects]
        detector = build_model("mock", {"init_args": {}})
        detector.eval()
        updated, _ = re_detect_hard_objects(
            scene.image,
            objs,
            detect_batch=lambda samples: predict_batches(detector, samples, batch_size=1),
            crop_metadata_fn=_crop_metadata_for_scene(scene),
        )
        gt = scene.objects[0].bbox
        got = updated[0].bbox_xyxy
        # 与真值框 IoU 很高（mock 完美重检，重裁框即真值框）
        inter_w = max(0.0, min(got[2], gt[2]) - max(got[0], gt[0]))
        inter_h = max(0.0, min(got[3], gt[3]) - max(got[1], gt[1]))
        iou = (inter_w * inter_h) / max(1e-9, (gt[2] - gt[0]) * (gt[3] - gt[1]))
        assert iou > 0.9

    def test_votes_merge_with_redetection(self):
        """重检测证据并入 category_votes，重投票选类。"""
        scene = _scene(2)
        objs = [_scene_object_to_hard(o, score=0.2) for o in scene.objects]
        detector = build_model("mock", {"init_args": {}})
        detector.eval()
        updated, _ = re_detect_hard_objects(
            scene.image,
            objs,
            detect_batch=lambda samples: predict_batches(detector, samples, batch_size=2),
            crop_metadata_fn=_crop_metadata_for_scene(scene),
        )
        for u in updated:
            assert u.category_votes[u.category_id] >= 0.2 + 1.0  # 原始票 + 重检票

    def test_no_hard_objects_unchanged(self):
        """无困难对象：原样返回，不付重检成本。"""
        scene = _scene(3)
        objs = [_scene_object_to_hard(o, score=0.9) for o in scene.objects]
        for o in objs:
            o.evidence = 3  # 高分 + 多证据 → 不困难
        detector = build_model("mock", {"init_args": {}})
        detector.eval()
        updated, timing = re_detect_hard_objects(
            scene.image,
            objs,
            detect_batch=lambda samples: predict_batches(detector, samples, batch_size=2),
            crop_metadata_fn=_crop_metadata_for_scene(scene),
            criteria=HardObjectCriteria(hard_score_threshold=0.3, hard_evidence_threshold=2),
        )
        assert updated == objs
        assert timing.n_hard == 0
        assert timing.n_crops == 0

    def test_empty_objects(self):
        updated, timing = re_detect_hard_objects(
            np.zeros((100, 100, 3), dtype=np.uint8),
            [],
            detect_batch=lambda s: [],
            crop_metadata_fn=lambda r, c: {},
        )
        assert updated == []
        assert timing.refine_s == 0.0

    def test_end_to_end_pipeline_refinement(self):
        """全链路：低分 mock 跑 pipeline → 对象低置信 → 完美 mock 二次检测提升。"""
        scene = generate_synthetic_scene(
            image_size=2048,
            tile_size=1024,
            overlap=128,
            num_ships=2,
            num_aircraft=3,
            num_vehicles=1,
            seed=42,
        )
        # 第一遍：低分检测器（score_offset 0.85 → 全 0.15，全部困难）
        weak = build_model("mock", {"init_args": {"score_offset": 0.85}})
        weak.eval()
        from rsdet.pipeline.large_image import PipelineConfig, run_pipeline

        def _tile_meta(tile):
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

        _, _, objs = run_pipeline(
            scene.image,
            weak,
            config=PipelineConfig(tile_size=1024, overlap=128, batch_size=4, fusion="global"),
            tile_metadata_fn=_tile_meta,
            collect_objects=True,
        )
        assert all(o.score < 0.3 for o in objs)  # 全部低置信

        # 第二遍：完美 mock 对困难对象二次检测
        perfect = build_model("mock", {"init_args": {}})
        perfect.eval()
        updated, timing = re_detect_hard_objects(
            scene.image,
            objs,
            detect_batch=lambda samples: predict_batches(perfect, samples, batch_size=4),
            crop_metadata_fn=_crop_metadata_for_scene(scene),
        )
        assert all(u.score > 0.9 for u in updated)
        assert len(updated) == len(objs)
        assert timing.refine_s < 1.0


def _scene(n: int):
    return generate_synthetic_scene(
        image_size=1024,
        tile_size=1024,
        overlap=0,
        num_ships=n,
        num_aircraft=0,
        num_vehicles=0,
        seed=7,
    )
