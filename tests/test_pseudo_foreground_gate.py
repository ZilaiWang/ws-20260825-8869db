import pytest

from scripts.build_cv3_pseudo_foreground_manifest import classify_proposal


def test_foreground_threshold_follows_supported_gt_category() -> None:
    prediction = (0.0, 0.0, 10.0, 10.0)
    # IoU=0.36: foreground for a supported vehicle, ambiguous for aircraft.
    vehicle = [((0.0, 0.0, 6.0, 6.0), 24)]
    aircraft = [((0.0, 0.0, 6.0, 6.0), 4)]

    assert classify_proposal(prediction, vehicle, negative_iou=0.05)[0] == "foreground"
    assert classify_proposal(prediction, aircraft, negative_iou=0.05)[0] == "ambiguous"


def test_clear_background_and_ambiguous_band_are_separate() -> None:
    ground_truth = [((0.0, 0.0, 10.0, 10.0), 0)]
    background = classify_proposal((20.0, 20.0, 30.0, 30.0), ground_truth, negative_iou=0.05)
    ambiguous = classify_proposal((5.0, 5.0, 15.0, 15.0), ground_truth, negative_iou=0.05)

    assert background[0] == "background"
    assert background[1] == pytest.approx(0.0)
    assert ambiguous[0] == "ambiguous"


def test_empty_gt_image_is_clear_background() -> None:
    result = classify_proposal((0.0, 0.0, 10.0, 10.0), [], negative_iou=0.05)
    assert result == ("background", 0.0, None)
