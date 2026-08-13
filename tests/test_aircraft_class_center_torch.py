"""Torch-only contract tests for the R1-2 training auxiliary.

The module is skipped in lightweight CPU environments that intentionally do
not install torch; the server task makes it a mandatory preflight test.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from scripts.r1_aircraft_refinement import (  # noqa: E402
    _ClassCenterMemory,
    _convnext_features_and_logits,
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
