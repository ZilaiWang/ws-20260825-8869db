from dataclasses import dataclass

import numpy as np
import pytest

from sprint20.policy import (
    _apply_aircraft_labels,
    certify_keep,
    full_view_labels,
    prediction_fingerprint,
    route_after_fusion,
)


@dataclass
class Prediction:
    image_id: int
    boxes_xyxy: list
    scores: list
    labels: list


def row(p0, p1):
    p = np.zeros(20)
    p[0] = p0
    p[1] = p1
    return p


def test_bound_strict_boundary_and_margin():
    p = np.array([row(0.801, 0.199), row(0.80, 0.20), row(0.79, 0.21)])
    assert certify_keep(p, [0, 0, 0], margin=0).tolist() == [True, False, False]
    assert certify_keep(p, [0, 0, 0], margin=0.0002).tolist() == [False, False, False]


def test_alternative_only_not_own_confidence():
    p = np.array([row(0.1, 0.9)])
    assert certify_keep(p, [1]).item()
    assert not certify_keep(p, [0]).item()


def test_late_views_can_overturn_identity():
    first = row(0.7, 0.3)
    views = np.tile(row(0, 1), (8, 1))
    views[0] = first
    assert not certify_keep(first[None], [0]).item()
    assert full_view_labels(views.mean(0)[None], [0]).item() == 1


def test_certification_against_adversarial_completions():
    rng = np.random.default_rng(123)
    for _ in range(250):
        p = rng.dirichlet(np.ones(20))
        label = int(rng.integers(20))
        if certify_keep(p[None], [label], margin=0).item():
            for alternative in range(20):
                if alternative == label:
                    continue
                final = p.copy()
                final[alternative] += 7
                final /= 8
                assert full_view_labels(final[None], [label]).item() == label


@pytest.mark.parametrize(
    "bad",
    [np.full((1, 20), np.nan), np.full((1, 20), -0.05), np.ones((1, 20)), np.ones((1, 19)) / 19],
)
def test_invalid_probabilities_fail_closed(bad):
    with pytest.raises(ValueError):
        certify_keep(bad, [0])


@pytest.mark.parametrize("labels", [[20], [-1], [0.5], [True], []])
def test_invalid_labels_fail_closed(labels):
    with pytest.raises(ValueError):
        certify_keep(row(0.9, 0.1)[None], labels)


@pytest.mark.parametrize("views", [0, -1, True, 1.5])
def test_invalid_view_count(views):
    with pytest.raises(ValueError):
        certify_keep(row(0.9, 0.1)[None], [0], views=views)


def test_multiset_keeps_duplicate_multiplicity():
    p = Prediction(1, [[0, 0, 1, 1]] * 2, [0.9, 0.9], [24, 24])
    assert sum(prediction_fingerprint(p).values()) == 2
    assert len(prediction_fingerprint(p)) == 1


def test_postfusion_ownership_keeps_all_aircraft_unchanged():
    p = Prediction(1, [[0, 0, 1, 1], [1, 1, 2, 2], [3, 3, 4, 4]], [0.9, 0.8, 0.7], [4, 0, 24])
    a = Prediction(1, [[9, 9, 10, 10], [8, 8, 9, 9], [6, 6, 7, 7]], [0.99, 0.65, 0.4], [4, 0, 24])
    out = route_after_fusion(
        p,
        a,
        alternative_labels=[0, 1, 2, 3, 24],
        primary_threshold=0.536,
        alternative_threshold=0.35,
    )
    assert prediction_fingerprint(p, {4}) == prediction_fingerprint(out, {4})
    assert out.boxes_xyxy == [[0, 0, 1, 1], [8, 8, 9, 9], [6, 6, 7, 7]]
    assert out.scores == [0.9, 0.65, 0.4]


def test_cannot_route_aircraft_to_alternative():
    p = Prediction(1, [], [], [])
    for labels in ([4], [24, 24], [True]):
        with pytest.raises(ValueError):
            route_after_fusion(
                p, p, alternative_labels=labels, primary_threshold=0.5, alternative_threshold=0.5
            )


def test_image_mismatch_fails():
    with pytest.raises(ValueError):
        route_after_fusion(
            Prediction(1, [], [], []),
            Prediction(2, [], [], []),
            alternative_labels=[24],
            primary_threshold=0.5,
            alternative_threshold=0.5,
        )


def test_empty_ownership_is_exact_primary_filter():
    p = Prediction(1, [[0, 0, 1, 1], [2, 2, 3, 3]], [0.536, 0.535], [24, 4])
    a = Prediction(1, [[6, 6, 7, 7]], [0.99], [24])
    out = route_after_fusion(
        p, a, alternative_labels=[], primary_threshold=0.536, alternative_threshold=0.1
    )
    assert out.labels == [24] and out.scores == [0.536]


def test_primary_fine_threshold_is_postfusion_and_class_bounded():
    primary = Prediction(
        1,
        [[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]],
        [0.545, 0.545, 0.70],
        [24, 4, 2],
    )
    alternative = Prediction(1, [[6, 6, 7, 7]], [0.60], [2])
    out = route_after_fusion(
        primary,
        alternative,
        alternative_labels=[2, 3],
        primary_threshold=0.536,
        alternative_threshold=0.56,
        primary_threshold_by_fine={24: 0.55},
    )
    assert out.labels == [2, 4]
    assert out.scores == [0.60, 0.545]
    with pytest.raises(ValueError, match="alternative-owned"):
        route_after_fusion(
            primary,
            alternative,
            alternative_labels=[2, 3],
            primary_threshold=0.536,
            alternative_threshold=0.56,
            primary_threshold_by_fine={2: 0.55},
        )


def test_same_class_nms_still_called_for_early_kept_objects():
    calls = []

    def toy_nms(boxes, scores, iou):
        calls.append((boxes, scores, iou))
        return [0] if boxes else []

    p = Prediction(1, [[0, 0, 1, 1], [0, 0, 1, 1], [3, 3, 4, 4]], [0.9, 0.8, 0.7], [4, 4, 24])
    out = _apply_aircraft_labels(p, [0, 0], toy_nms, nms_iou=0.5)
    assert len(calls) == 20
    assert out.labels == [4, 24] and out.scores == [0.9, 0.7]
    assert prediction_fingerprint(out, {24}) == prediction_fingerprint(p, {24})


def test_relabel_then_nms_handles_new_same_class_duplicate():
    def toy_nms(boxes, scores, iou):
        return [0] if boxes else []

    p = Prediction(1, [[0, 0, 1, 1], [0, 0, 1, 1]], [0.9, 0.8], [4, 5])
    out = _apply_aircraft_labels(p, [1, 1], toy_nms)
    assert out.labels == [5] and out.scores == [0.9]
