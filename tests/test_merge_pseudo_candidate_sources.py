import pytest

from rsdet.analysis.multi_detector_oer import class_aware_nms_records
from scripts.merge_pseudo_candidate_sources import (
    fast_class_aware_nms,
    normalize_sources,
    to_coco,
)


def test_source_variants_map_to_detector_families() -> None:
    base = {
        "image_id": 7,
        "category_id": 24,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "score": 0.5,
        "source_fold": 2,
    }
    rows = normalize_sources(
        [("Y5_ROT", [base]), ("Y5_800", [base]), ("M3_ID", [base])]
    )
    assert [row["source_model"] for row in rows] == ["Y5", "Y5", "M3"]
    assert [row["source_variant"] for row in rows] == ["Y5_ROT", "Y5_800", "M3_ID"]
    assert rows[0]["bbox_xyxy"] == [1.0, 2.0, 4.0, 6.0]
    output = to_coco(rows)
    assert output[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert output[2]["source_model"] == "M3"


def test_non_positive_boxes_are_removed() -> None:
    rows = normalize_sources(
        [
            (
                "Y5",
                [
                    {
                        "image_id": 1,
                        "category_id": 0,
                        "bbox": [0.0, 0.0, 0.0, 4.0],
                        "score": 0.2,
                        "source_fold": 0,
                    }
                ],
            )
        ]
    )
    assert rows == []


def test_vectorized_nms_matches_reference() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    rows = normalize_sources(
        [
            (
                "Y5",
                [
                    {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9, "source_fold": 0},
                    {"image_id": 1, "category_id": 0, "bbox": [1, 1, 10, 10], "score": 0.8, "source_fold": 0},
                    {"image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10], "score": 0.7, "source_fold": 0},
                    {"image_id": 2, "category_id": 0, "bbox": [1, 1, 10, 10], "score": 0.6, "source_fold": 1},
                ],
            )
        ]
    )
    expected = class_aware_nms_records(rows, iou_threshold=0.5)
    actual = fast_class_aware_nms(rows, iou_threshold=0.5)
    assert [row["stable_order"] for row in actual] == [
        row["stable_order"] for row in expected
    ]
