from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from rsdet.hera_guard.losses import pav_multitask_loss  # noqa: E402
from rsdet.hera_guard.mar_training import (  # noqa: E402
    build_mar_features,
    fit_monotone_mar,
)
from rsdet.hera_guard.resolver import (  # noqa: E402
    MonotoneAsymmetricResolver,
    resolve_fine_category,
)
from rsdet.hera_guard.runtime_policy import route_ambiguous_candidates  # noqa: E402
from rsdet.hera_guard.verifier import ProposalAlignedVerifier  # noqa: E402


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GELU())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Identity(), nn.Flatten(1), nn.Linear(8, 2))


def test_pav_two_view_forward_contract() -> None:
    model = ProposalAlignedVerifier(_TinyBackbone(), metadata_dim=4, hidden_dim=16, dropout=0.0)
    output = model(
        torch.randn(3, 3, 16, 16),
        torch.randn(3, 3, 16, 16),
        torch.randn(3, 4),
    )
    assert output.foreground_logit.shape == (3,)
    assert output.coarse_logits.shape == (3, 3)
    assert output.fine_logits.shape == (3, 25)
    assert output.quality_logit.shape == (3,)
    assert output.protect_logit.shape == (3,)
    assert output.active_fp_logit.shape == (3,)


def test_pav_loss_is_finite_for_background_only_batch() -> None:
    output = SimpleNamespace(
        foreground_logit=torch.randn(4, requires_grad=True),
        coarse_logits=torch.randn(4, 3, requires_grad=True),
        fine_logits=torch.randn(4, 25, requires_grad=True),
        quality_logit=torch.randn(4, requires_grad=True),
        protect_logit=torch.randn(4, requires_grad=True),
        active_fp_logit=torch.randn(4, requires_grad=True),
    )
    losses = pav_multitask_loss(
        output,
        {
            "foreground": torch.zeros(4),
            "coarse": torch.zeros(4, dtype=torch.long),
            "fine": torch.zeros(4, dtype=torch.long),
            "quality": torch.zeros(4),
            "protect": torch.zeros(4),
            "active_fp": torch.zeros(4),
        },
        fine_class_counts=torch.ones(25),
    )
    assert torch.isfinite(losses["total"])
    assert losses["n_foreground"] == 0
    losses["total"].backward()


def test_resolver_is_monotone_and_bounded() -> None:
    model = MonotoneAsymmetricResolver(2, rho_max=1.0)
    base = torch.tensor([0.4, 0.4])
    evidence = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    resolved = model(base, evidence)
    assert 0 < resolved[0] <= resolved[1] < 1
    assert model.constrained_parameters()["weights"].min() > 0


def test_category_resolution_is_within_coarse_and_protected() -> None:
    coarse = [0, 0, 1]
    changed = resolve_fine_category(
        detector_category_id=0,
        fine_probabilities=[0.1, 0.8, 0.9],
        coarse_of_fine=coarse,
        minimum_probability=0.7,
        minimum_margin=0.5,
        protect_probability=0.1,
        maximum_protect_for_change=0.5,
    )
    assert changed.changed and changed.category_id == 1
    vetoed = resolve_fine_category(
        detector_category_id=0,
        fine_probabilities=[0.1, 0.8, 0.9],
        coarse_of_fine=coarse,
        minimum_probability=0.7,
        minimum_margin=0.5,
        protect_probability=0.9,
        maximum_protect_for_change=0.5,
    )
    assert not vetoed.changed and vetoed.reason == "protected_tp_veto"


def test_runtime_routing_honors_per_image_cap() -> None:
    records = [
        {
            "candidate_id": index,
            "image_id": 1,
            "crop_entropy": 3.0,
            "detector_crop_agree": 0,
        }
        for index in range(5)
    ]
    routed = route_ambiguous_candidates(records, max_per_image=2)
    assert [row.candidate_id for row in routed] == [0, 1]


def test_mar_feature_and_fit_contract() -> None:
    rows = [
        {"detector_category_id": index % 2, "detector_coarse": "aircraft"} for index in range(6)
    ]
    predictions = [{"score": 0.2 + index * 0.1} for index in range(6)]
    logits = {
        "candidate_id": np.arange(6),
        "foreground_logit": np.linspace(-1, 1, 6),
        "quality_logit": np.linspace(-0.5, 0.5, 6),
        "protect_logit": np.linspace(-2, 2, 6),
        "fine_logits": np.tile(np.linspace(-1, 1, 25), (6, 1)),
    }
    features = build_mar_features(rows=rows, predictions=predictions, logits=logits)
    fit = fit_monotone_mar(
        train_base_score=np.asarray([row["score"] for row in predictions[:4]]),
        train_features=features[:4],
        train_target=np.asarray([1, 0, 1, 0]),
        train_role=["protected_tp", "active_fp", "protected_tp", "active_fp"],
        train_coarse=[0, 0, 0, 0],
        validation_base_score=np.asarray([0.6, 0.7]),
        validation_features=features[4:],
        epochs=3,
    )
    assert fit.validation_scores.shape == (2,)
    assert np.isfinite(fit.validation_scores).all()
    assert np.all(fit.constrained_weights > 0)
