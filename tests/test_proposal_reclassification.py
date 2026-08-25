"""N2-2 对象学生重分类/融合模块测试。"""

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
                    "image_id": 1,
                    "category_id": 24,
                    "score": 0.9,
                    "bbox_xyxy": [0, 0, 10, 10],
                },
                {
                    "image_id": 1,
                    "category_id": 24,
                    "score": 0.8,
                    "bbox_xyxy": [0, 0, 10, 10],
                },
            ],
            2: [
                {
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
                "image_id": 1,
                "score": 0.9,
                "original_category_id": 24,
                "source_prediction_index": 0,
                "category_id": 4,
                "student_score": 0.95,
                "dropped": False,
            },
            {
                "image_id": 1,
                "score": 0.8,
                "original_category_id": 24,
                "source_prediction_index": 1,
                "category_id": 25,
                "student_score": 0.60,
                "dropped": True,
            },
        ]

    def test_reclassify_replaces_categories(self):
        # reclassify 模式下 background 保留原类别且不 drop（dropped=False）。
        reclassified = [
            {
                "image_id": 1,
                "score": 0.9,
                "original_category_id": 24,
                "source_prediction_index": 0,
                "category_id": 4,
                "student_score": 0.95,
                "dropped": False,
            },
            {
                "image_id": 1,
                "score": 0.8,
                "original_category_id": 24,
                "source_prediction_index": 1,
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
        p1 = result[1][0]
        assert p1["category_id"] == 4
        assert len(result[1]) == 2  # 未丢弃
        assert len(result[2]) == 1

    def test_background_reject_drops(self):
        result = build_reclassified_predictions(
            oof_predictions=self._oof(),
            reclassified=self._reclassified(),
            mode=MODE_BACKGROUND,
        )
        categories = [r["category_id"] for r in result[1]]
        assert categories == [24]  # p2 丢弃，p1 保留检测器原类别

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
            oof_predictions={3: [{"image_id": 3, "category_id": 0, "score": 0.5}]},
            reclassified=[],
            mode=MODE_JOINT,
        )
        assert result[3][0]["category_id"] == 0

    def test_duplicate_scores_are_matched_by_source_index(self):
        oof = {
            1: [
                {"image_id": 1, "category_id": 4, "score": 0.5},
                {"image_id": 1, "category_id": 5, "score": 0.5},
            ]
        }
        reclassified = [
            {
                "image_id": 1,
                "source_prediction_index": 1,
                "original_category_id": 5,
                "category_id": 6,
                "student_score": 0.9,
                "dropped": False,
            }
        ]
        result = build_reclassified_predictions(
            oof_predictions=oof,
            reclassified=reclassified,
            mode=MODE_RECLASSIFY,
        )
        assert [item["category_id"] for item in result[1]] == [4, 6]
