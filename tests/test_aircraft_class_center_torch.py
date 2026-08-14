"""Torch-only contract tests for the R1-2 training auxiliary.

The module is skipped in lightweight CPU environments that intentionally do
not install torch; the server task makes it a mandatory preflight test.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from scripts.r1_aircraft_refinement import (  # noqa: E402
    _build_attribute_heads,
    _ClassCenterMemory,
    _convnext_features_and_logits,
    _symmetric_view_consistency,
)


class _TinyConvNeXtLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 4, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Identity(), nn.Flatten(1), nn.Linear(4, 25))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = self.avgpool(self.features(images))
        return self.classifier(values)


def test_explicit_feature_path_preserves_logits() -> None:
    model = _TinyConvNeXtLike().eval()
    images = torch.randn(3, 3, 8, 8)
    features, logits = _convnext_features_and_logits(model, images)
    assert features.shape == (3, 4)
    torch.testing.assert_close(logits, model(images), rtol=0.0, atol=0.0)


def test_class_center_loss_is_finite_and_centers_remain_normalized() -> None:
    weight = torch.eye(4)
    memory = _ClassCenterMemory(
        weight,
        momentum=0.5,
        margin=0.1,
        negative_weight=1.0,
    )
    features = torch.tensor(
        [[1.0, 0.1, 0.0, 0.0], [0.1, 1.0, 0.0, 0.0]], requires_grad=True
    )
    labels = torch.tensor([0, 1])
    loss, diagnostics = memory.loss(features, labels)
    assert torch.isfinite(loss)
    assert diagnostics["positive_cosine"] > diagnostics["hardest_negative_cosine"]
    loss.backward()
    assert torch.isfinite(features.grad).all()
    memory.update(features, labels)
    torch.testing.assert_close(
        memory.centers.norm(dim=1), torch.ones(4), rtol=1e-5, atol=1e-5
    )


def test_training_only_attribute_heads_have_expected_shapes() -> None:
    taxonomy = {
        "dimensions": {
            "engine_count": {"values": ["one", "two", "four"]},
            "propulsion": {"values": ["jet", "turboprop"]},
        }
    }
    heads = _build_attribute_heads(torch, taxonomy, feature_dim=4)
    features = torch.randn(5, 4, requires_grad=True)
    outputs = {name: head(features) for name, head in heads.items()}
    assert outputs["engine_count"].shape == (5, 3)
    assert outputs["propulsion"].shape == (5, 2)
    sum(value.sum() for value in outputs.values()).backward()
    assert torch.isfinite(features.grad).all()


def test_symmetric_view_consistency_is_zero_only_for_matching_distributions() -> None:
    logits_a = torch.tensor([[4.0, 1.0, -2.0]], requires_grad=True)
    logits_b = torch.tensor([[1.0, 4.0, -2.0]], requires_grad=True)
    identical = _symmetric_view_consistency(
        logits_a,
        logits_a,
        temperature=1.0,
    )
    different = _symmetric_view_consistency(
        logits_a,
        logits_b,
        temperature=1.0,
    )
    assert identical.item() == pytest.approx(0.0, abs=1e-7)
    assert different.item() > 0.0
    different.backward()
    assert torch.isfinite(logits_a.grad).all()
    assert torch.isfinite(logits_b.grad).all()
