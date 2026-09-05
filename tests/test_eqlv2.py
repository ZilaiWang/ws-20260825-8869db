import pytest

from rsdet.innovation.eqlv2 import eqlv2_classification_loss

torch = pytest.importorskip("torch")


def test_nonfocus_classes_remain_exact_bce():
    logits = torch.tensor([[[0.2, -0.3, 0.4]]])
    target = torch.tensor([[[0.0, 0.7, 0.0]]])
    module = eqlv2_classification_loss(focus_class_indices=(0, 2))
    actual = module(logits, target)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    assert actual[..., 1] == pytest.approx(expected[..., 1])


def test_focus_weights_follow_accumulated_gradient_ratio():
    module = eqlv2_classification_loss(
        focus_class_indices=(0,), gamma=12.0, mu=0.8, alpha=4.0
    )
    logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    first = module(logits, target)
    audit = module.audit()
    assert audit["updates"] == 1
    assert audit["positive_gradient"][0] > 0
    assert audit["negative_gradient"][0] > 0
    assert audit["positive_weight"][1] == pytest.approx(1.0)
    assert audit["negative_weight"][1] == pytest.approx(1.0)
    second = module(logits, target)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert not torch.equal(first[..., 0], second[..., 0])
    assert torch.equal(first[..., 1], second[..., 1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"focus_class_indices": ()},
        {"focus_class_indices": (-1,)},
        {"focus_class_indices": (0,), "gamma": 0},
        {"focus_class_indices": (0,), "mu": 1.1},
        {"focus_class_indices": (0,), "alpha": -1},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        eqlv2_classification_loss(**kwargs)


def test_class_dimension_is_checked():
    module = eqlv2_classification_loss(focus_class_indices=(3,))
    with pytest.raises(ValueError, match="class dimension"):
        module(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))
