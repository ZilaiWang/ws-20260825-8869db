"""主线 3 全局对象聚合算法的最小验证（Group Evidence Matters 机制转译）。

实现位于 src/rsdet/postprocess/global_aggregation.py，此处为单元测试。
不依赖 OOF 数据文件；proposal 统一用 dict：
{'x','y','width','height','category_id','score'}（xywh，与 OOF CSV 同构）。

聚合流程：空间聚类（Spatial Gate）→ 簇内 IoU 子簇（区分相邻的不同目标）→
细类投票 → 选 canonical 框 → 同类 NMS。对应 Group Evidence Matters 的
Spatial Gate + Semantic Gate 两层门（此处以 IoU 连通代替 embedding 聚类）。
"""

from __future__ import annotations

from typing import Any

from rsdet.postprocess.global_aggregation import (
    _iou_subcluster,
    aggregate,
    class_aware_nms,
    class_vote,
    spatial_cluster,
)

proposal = dict[str, Any]


def _box(x, y, w, h, cat, score):
    return {"x": x, "y": y, "width": w, "height": h, "category_id": cat, "score": score}


class TestSpatialCluster:
    def test_far_boxes_not_clustered(self):
        """相距很远的框不该聚成一簇。"""
        props = [_box(0, 0, 10, 10, 0, 0.8), _box(1000, 1000, 10, 10, 0, 0.7)]
        cl = spatial_cluster(props, eps=50.0)
        assert len(cl) == 2

    def test_near_boxes_clustered(self):
        """中心距离小于 eps 的框聚成一簇。"""
        props = [_box(0, 0, 10, 10, 0, 0.8), _box(20, 0, 10, 10, 0, 0.7)]
        cl = spatial_cluster(props, eps=50.0)
        assert len(cl) == 1
        assert len(cl[0]) == 2

    def test_chain_clustering(self):
        """A-B、B-C 相接时 A-C 虽远也归并（传递性）。"""
        props = [
            _box(0, 0, 10, 10, 0, 0.8),
            _box(40, 0, 10, 10, 0, 0.7),
            _box(80, 0, 10, 10, 0, 0.6),
        ]
        cl = spatial_cluster(props, eps=50.0)
        assert len(cl) == 1
        assert len(cl[0]) == 3


class TestClassVote:
    def test_majority_class_wins(self):
        """多数细类获胜。"""
        props = [
            _box(0, 0, 10, 10, 9, 0.8),  # TU-160
            _box(20, 0, 10, 10, 9, 0.7),
            _box(40, 0, 10, 10, 22, 0.5),  # SU-34 少数
        ]
        assert class_vote([0, 1, 2], props) == 9

    def test_score_weighted_vote(self):
        """高分证据能压过低分多数。"""
        props = [
            _box(0, 0, 10, 10, 22, 0.9),  # SU-34 高分
            _box(20, 0, 10, 10, 9, 0.4),  # TU-160 两个低分
            _box(40, 0, 10, 10, 9, 0.4),
        ]
        assert class_vote([0, 1, 2], props) == 22

    def test_tie_deterministic(self):
        """平手时取较小 category_id，结果确定。"""
        props = [_box(0, 0, 10, 10, 22, 0.5), _box(20, 0, 10, 10, 9, 0.5)]
        assert class_vote([0, 1], props) == 9  # 分数同 0.5，取 cat=9


class TestIouSubcluster:
    def test_overlapping_boxes_merged(self):
        """同一目标被切出的两个框（IoU 高）归入同一子簇。"""
        props = [
            _box(0, 0, 100, 100, 9, 0.8),
            _box(0, 0, 50, 100, 9, 0.7),  # 被 tile 切掉一半的同一目标
        ]
        subs = _iou_subcluster([0, 1], props, merge_iou=0.3)
        assert len(subs) == 1
        assert sorted(subs[0]) == [0, 1]

    def test_adjacent_objects_split(self):
        """相邻但无重叠的不同目标分到不同子簇。"""
        props = [
            _box(0, 0, 20, 20, 9, 0.8),
            _box(30, 0, 20, 20, 9, 0.7),  # 与前者无重叠
        ]
        subs = _iou_subcluster([0, 1], props, merge_iou=0.3)
        assert len(subs) == 2


class TestClassAwareNms:
    def test_same_class_suppressed(self):
        """同类高 IoU 的重复框被压掉。"""
        boxes = [
            _box(0, 0, 100, 100, 9, 0.9),
            _box(10, 10, 100, 100, 9, 0.5),
        ]
        kept = class_aware_nms(boxes)
        assert len(kept) == 1
        assert kept[0]["score"] == 0.9

    def test_cross_class_not_suppressed(self):
        """跨细类的高 IoU 框互不压制（交给投票而非 NMS）。"""
        boxes = [
            _box(0, 0, 100, 100, 9, 0.8),
            _box(0, 0, 100, 100, 22, 0.7),
        ]
        kept = class_aware_nms(boxes)
        assert len(kept) == 2


class TestAggregate:
    def test_empty(self):
        assert aggregate([]) == []

    def test_single_proposal_passthrough(self):
        out = aggregate([_box(10, 10, 50, 50, 9, 0.8)])
        assert len(out) == 1
        assert out[0]["category_id"] == 9
        assert out[0]["evidence"] == 1

    def test_same_object_duplicates_merged(self):
        """同一目标 3 个候选 → 聚成一簇、只输出 1 个框、证据量 3。"""
        props = [
            _box(0, 0, 100, 100, 9, 0.8),
            _box(5, 5, 100, 100, 9, 0.7),
            _box(10, 10, 100, 100, 9, 0.6),
        ]
        out = aggregate(props)
        assert len(out) == 1
        assert out[0]["category_id"] == 9
        assert out[0]["evidence"] == 3

    def test_cross_class_conflict_resolved(self):
        """同一目标被预测成两个型号 → 投票选出一个，只输出 1 框。"""
        props = [
            _box(0, 0, 100, 100, 22, 0.8),  # SU-34 高分
            _box(0, 0, 100, 100, 9, 0.6),  # TU-160 低分
        ]
        out = aggregate(props)
        assert len(out) == 1
        assert out[0]["category_id"] == 22
        assert out[0]["evidence"] == 2

    def test_adjacent_distinct_objects_not_fused(self):
        """相邻但不同的同类型目标：中心近、IoU 低 → 不被错误融合，输出 2 框。"""
        props = [
            _box(0, 0, 20, 20, 9, 0.8),  # 目标 A
            _box(30, 0, 20, 20, 9, 0.7),  # 目标 B，中心距 20 < eps，但 IoU 低
        ]
        out = aggregate(props)
        assert len(out) == 2

    def test_tile_clipped_same_object_merged(self):
        """同一目标被 tile 切成整框+半框 → 合并成一个对象。"""
        props = [
            _box(0, 0, 100, 100, 9, 0.8),
            _box(0, 0, 50, 100, 9, 0.7),  # 另一半 tile 里的截断框
        ]
        out = aggregate(props)
        assert len(out) == 1
        assert out[0]["evidence"] == 2
        assert out[0]["width"] == 100  # canonical 取整框

    def test_canonical_box_is_highest_score(self):
        """canonical 框坐标取簇内最高分框。"""
        props = [
            _box(0, 0, 100, 100, 9, 0.5),
            _box(0, 0, 120, 120, 9, 0.9),  # 最高分、更大的框
        ]
        out = aggregate(props)
        assert len(out) == 1
        assert out[0]["width"] == 120
        assert out[0]["height"] == 120
        assert out[0]["score"] == 0.9
