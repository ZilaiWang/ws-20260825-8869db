import numpy as np
import pytest

from sprint20.evaluation import (
    cache_predictions,
    gt_from_coco,
    paired_bootstrap,
    rates_from_counts,
    route_dicts,
    validate_frozen_policy,
)


def test_coco_xywh_conversion_and_negative_images():
    c = {
        "images": [{"id": 1}, {"id": 2}],
        "annotations": [{"image_id": 1, "category_id": 24, "bbox": [2, 3, 4, 5]}],
    }
    g = gt_from_coco(c)
    assert g[1][0]["bbox_xyxy"] == [2, 3, 6, 8] and g[2] == []


def test_crowd_annotations_not_silently_dropped():
    c = {
        "images": [{"id": 1}],
        "annotations": [{"image_id": 1, "category_id": 24, "bbox": [2, 3, 4, 5], "iscrowd": 1}],
    }
    with pytest.raises(ValueError):
        gt_from_coco(c)


def test_image_set_route_must_match():
    with pytest.raises(ValueError):
        route_dicts({1: []}, {2: []}, [24])


def test_empty_rare_gt_not_invented_perfect():
    c = np.ones((25, 3), dtype=int)
    c[0, [0, 2]] = 0
    assert rates_from_counts(c) is None


def test_macro_not_pooled_counts():
    c = np.zeros((25, 3))
    c[:, 0] = 10
    c[0] = [0, 10, 10]
    c[24] = [5, 5, 5]
    r = rates_from_counts(c)
    assert r["ship"]["recall"] == 0.75
    assert r["ship"]["fdr"] == 0.25
    assert r["vehicle"]["recall"] == 0.5
    assert r["aircraft"]["recall"] == 1.0


def toy_scorer(rows, t):
    # Synthetic test objective, not a replacement of the official score.
    return sum(v["recall"] - v["fdr"] for v in rows.values()) - t * 0.01


def test_bootstrap_identity_and_reproducibility():
    c = np.ones((4, 25, 3), dtype=int) * 10
    a = paired_bootstrap(c, c, 4, 4, repetitions=100, scorer=toy_scorer)
    b = paired_bootstrap(c, c, 4, 4, repetitions=100, scorer=toy_scorer)
    assert a == b and a["p10"] == 0 and a["valid_fraction"] == 1


def test_bootstrap_missing_taxonomy_is_explicit():
    c = np.ones((3, 25, 3), dtype=int)
    c[:, :, 0] = 0
    c[:, :, 2] = 0
    c[0, :, 0] = 10
    r = paired_bootstrap(c, c, 4, 4, repetitions=500, scorer=toy_scorer)
    assert r["missing_taxonomy_repetitions"] > 0
    assert r["conditional_on_complete_taxonomy"] is True
    assert r["interpretation"] == "inconclusive_sparse_taxonomy"


def test_more_latency_penalized_in_every_resample():
    c = np.ones((4, 25, 3), dtype=int)
    r = paired_bootstrap(c, c, 4, 5, repetitions=100, scorer=toy_scorer)
    assert r["p90"] < 0


def test_cache_threshold_exact_inclusive():
    c = {
        "images": [
            {
                "image_id": 1,
                "prediction": {
                    "boxes_xyxy": [[0, 0, 1, 1]] * 2,
                    "scores": [0.536, 0.535],
                    "labels": [24, 24],
                },
            }
        ]
    }
    assert len(cache_predictions(c, 0.536)[1]) == 1


def test_duplicate_cache_id_rejected():
    row = {"image_id": 1, "prediction": {"boxes_xyxy": [], "scores": [], "labels": []}}
    with pytest.raises(ValueError):
        cache_predictions({"images": [row, row]}, 0.5)


def good_policy():
    return {
        "frozen_before_evaluation": True,
        "primary_threshold": 0.536,
        "alternative_threshold": 0.4,
        "baseline_latency_seconds": 4.0,
        "candidate_latency_seconds": 5.0,
        "selection_groups": ["train"],
        "evaluation_groups": ["val"],
    }


def test_policy_rejects_selection_evaluation_overlap():
    p = good_policy()
    p["evaluation_groups"] = ["train"]
    with pytest.raises(ValueError, match="overlap"):
        validate_frozen_policy(p)


def test_policy_requires_valid_threshold_not_template_null():
    p = good_policy()
    p["alternative_threshold"] = None
    with pytest.raises(ValueError):
        validate_frozen_policy(p)


def test_policy_requires_explicit_freeze():
    p = good_policy()
    p["frozen_before_evaluation"] = False
    with pytest.raises(ValueError):
        validate_frozen_policy(p)


def test_policy_declared_nonempty_is_not_implied():
    p = good_policy()
    p["selection_groups"] = []
    assert not validate_frozen_policy(p)["declared_groups_nonempty"]


def test_policy_valid_disjoint_declaration():
    assert validate_frozen_policy(good_policy())["declared_groups_nonempty"]
