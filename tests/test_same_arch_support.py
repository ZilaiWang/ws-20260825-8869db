from rsdet.submission.same_arch_support import rescore_same_fine_support


def test_same_arch_support_preserves_candidates_and_bypasses_aircraft() -> None:
    primary = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.8},
        {"image_id": 1, "category_id": 4, "bbox": [20, 20, 5, 5], "score": 0.7},
        {"image_id": 1, "category_id": 24, "bbox": [40, 40, 10, 10], "score": 0.6},
    ]
    specialist = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.5},
        {"image_id": 1, "category_id": 24, "bbox": [100, 100, 10, 10], "score": 0.9},
    ]
    output, audit = rescore_same_fine_support(
        primary, specialist, label_iou_thresholds={0: 0.5, 24: 0.35}
    )
    assert len(output) == len(primary)
    assert output[0]["score"] == 0.4
    assert output[1] == primary[1]
    assert output[2]["score"] == 0.0
    assert audit["selected_count"] == 2
    assert audit["supported_count"] == 1


def test_same_arch_support_rejects_invalid_threshold() -> None:
    try:
        rescore_same_fine_support([], [], label_iou_thresholds={0: 1.1})
    except ValueError as error:
        assert "IoU thresholds" in str(error)
    else:
        raise AssertionError("invalid threshold was accepted")


def test_same_arch_support_bypasses_images_outside_specialist_scope() -> None:
    primary = [
        {"image_id": 2, "category_id": 24, "bbox": [0, 0, 10, 10], "score": 0.8}
    ]
    specialist = [
        {"image_id": 1, "category_id": 24, "bbox": [0, 0, 10, 10], "score": 0.9}
    ]
    output, audit = rescore_same_fine_support(
        primary, specialist, label_iou_thresholds={24: 0.35}
    )
    assert output == primary
    assert audit["selected_count"] == 0
    assert audit["scoped_image_count"] == 1
