import math

import pytest

pytest.importorskip("torch")

from rsdet.innovation.weak_rfs import weak_image_weights


def test_only_weak_labels_are_reweighted_and_all_images_remain():
    labels = [{4}] * 95 + [{0}, {0, 0}, {1}, {24}, set()]
    weights, audit = weak_image_weights(labels)
    assert len(weights) == 100
    assert weights[:95] == [1.]*95 and weights[-1] == 1
    assert weights[95] == pytest.approx(math.sqrt(5))
    assert weights[95] == weights[96]
    assert weights[97:99] == [3., 3.]
    assert audit["image_counts_by_fine"][0] == 2
    assert audit["samples_per_epoch"] == 100
    assert audit["expected_image_draws_by_fine"][24] > 1


def test_deterministic_with_class_order_independent():
    assert weak_image_weights([{0, 1}, {4}, set()]) == weak_image_weights([{1, 0}, {4}, set()])


@pytest.mark.parametrize("labels", [[], [{25}], [{-1}], [{.5}], [{True}]])
def test_bad_labels_fail(labels):
    with pytest.raises(ValueError):
        weak_image_weights(labels)


def test_sampler_epoch_budget_repeatability_and_new_epoch_draw():
    torch = pytest.importorskip("torch")
    labels = [{4}]*95 + [{24}]*5
    weights, _ = weak_image_weights(labels)
    def make():
        return torch.utils.data.WeightedRandomSampler(weights, len(weights), True,
            generator=torch.Generator().manual_seed(42))
    a, b = make(), make()
    first, again, next_epoch = list(a), list(b), list(a)
    assert first == again and first != next_epoch
    assert len(first) == len(labels) and set(first) <= set(range(len(labels)))
