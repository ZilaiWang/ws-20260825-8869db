"""N2-2 对象学生重分类/融合模块测试。"""

import pytest

from rsdet.analysis.proposal_reclassification import (
    MODE_BACKGROUND,
    MODE_JOINT,
    MODE_RECLASSIFY,
    build_reclassified_predictions,
)


class TestBuildReclassifiedPredictions:
    def _oof(self):
        return {
            1: [
                {
                    "proposal_uid": "p1",
                    "image_id": 1,
                    "category_id": 24,
                    "score": 0.9,
                    "bbox_xyxy": [0, 0, 10, 10],
                },
                {
                    "proposal_uid": "p2",
                    "image_id": 1,
                    "category_id": 24,
                    "score": 0.8,
                    "bbox_xyxy": [0, 0, 10, 10],
                },
            ],
            2: [
                {
                    "proposal_uid": "p3",
                    "image_id": 2,
                    "category_id": 5,
                    "score": 0.7,
                    "bbox_xyxy": [0, 0, 10, 10],
                }
            ],
        }

    def _reclassified(self):
        return [
            {
                "proposal_uid": "p1",
                "category_id": 4,
                "student_score": 0.95,
                "dropped": False,
            },
            {
                "proposal_uid": "p2",
                "category_id": 25,
                "student_score": 0.60,
                "dropped": True,
            },
        ]

    def test_reclassify_replaces_categories(self):
        # reclassify 模式下 background 保留原类别且不 drop（dropped=False）。
        reclassified = [
            {
                "proposal_uid": "p1",
                "category_id": 4,
                "student_score": 0.95,
                "dropped": False,
            },
            {
                "proposal_uid": "p2",
                "category_id": 24,  # 判为 background 时保留原类别
                "student_score": 0.60,
                "dropped": False,
            },
        ]
        result = build_reclassified_predictions(
            oof_predictions=self._oof(),
            reclassified=reclassified,
            mode=MODE_RECLASSIFY,
        )
        p1 = [r for r in result[1] if r["proposal_uid"] == "p1"][0]
        assert p1["category_id"] == 4
        assert len(result[1]) == 2  # 未丢弃
        assert len(result[2]) == 1

    def test_background_reject_drops(self):
        result = build_reclassified_predictions(
            oof_predictions=self._oof(),
            reclassified=self._reclassified(),
            mode=MODE_BACKGROUND,
        )
        uids = [r["proposal_uid"] for r in result[1]]
        assert "p2" not in uids  # dropped
        assert "p1" in uids

    def test_joint_replaces_and_drops(self):
        result = build_reclassified_predictions(
            oof_predictions=self._oof(),
            reclassified=self._reclassified(),
            mode=MODE_JOINT,
        )
        assert len(result[1]) == 1
        p1 = result[1][0]
        assert p1["category_id"] == 4
        assert p1["student_score"] == 0.95

    def test_unmatched_kept(self):
        result = build_reclassified_predictions(
            oof_predictions={3: [{"proposal_uid": "p9", "category_id": 0}]},
            reclassified=[],
            mode=MODE_JOINT,
        )
        assert result[3][0]["category_id"] == 0
