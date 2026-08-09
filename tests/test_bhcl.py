"""Formula-level tests for balanced hierarchical contrastive learning."""

from __future__ import annotations

import math
import subprocess
import sys

import pytest

# Probe in a child process because a broken Windows torch install can terminate
# the importing process while loading its DLLs instead of raising ImportError.
torch_probe = subprocess.run(
    [sys.executable, "-c", "import torch"],
    check=False,
    capture_output=True,
    text=True,
)
if torch_probe.returncode != 0:  # pragma: no cover - environment dependent
    reason = (torch_probe.stderr or torch_probe.stdout).strip().splitlines()
    pytest.skip(
        f"PyTorch is unavailable: {reason[-1] if reason else 'import failed'}",
        allow_module_level=True,
    )

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rsdet.models.bhcl import BalancedHierarchicalContrastiveLoss  # noqa: E402
from rsdet.models.hierarchy import XH_HIERARCHY, HierarchySpec  # noqa: E402

TWO_CLASS_HIERARCHY = HierarchySpec(
    root_name="root",
    level_names=("fine",),
    level_categories=(("a", "b"),),
    fine_to_level=((0, 1),),
)


def _dynamic_level_reference(features, labels, prototypes, temperature):
    """Former per-category Eq. (8)--(9) implementation used as an oracle."""

    pair_logits = features @ features.transpose(0, 1) / temperature
    prototype_logits = features @ prototypes.transpose(0, 1) / temperature
    class_counts = torch.bincount(labels, minlength=prototypes.shape[0])
    losses = []
    for anchor in range(features.shape[0]):
        denominator_terms = []
        for category in range(prototypes.shape[0]):
            members = torch.nonzero(labels == category, as_tuple=False).flatten()
            members = members[members != anchor]
            logits = torch.cat(
                (
                    pair_logits[anchor].index_select(0, members),
                    prototype_logits[anchor, category].reshape(1),
                )
            )
            denominator_terms.append(
                torch.logsumexp(logits, dim=0)
                - (class_counts[category].to(pair_logits.dtype) + 1.0).log()
            )
        log_denominator = torch.logsumexp(
            torch.stack(denominator_terms), dim=0
        )
        own_category = int(labels[anchor].item())
        positives = torch.nonzero(
            labels == own_category, as_tuple=False
        ).flatten()
        positives = positives[positives != anchor]
        positive_sum = pair_logits[anchor].index_select(0, positives).sum()
        positive_sum = positive_sum + prototype_logits[anchor, own_category]
        mean_positive = positive_sum / class_counts[own_category].to(
            pair_logits.dtype
        )
        losses.append(log_denominator - mean_positive)
    return torch.stack(losses).mean()


def _dynamic_bhcl_reference(features, labels, prototypes, hierarchy, temperature):
    normalized = F.normalize(features, dim=1)
    prototype_snapshot = F.normalize(prototypes.detach().clone(), dim=1).to(
        normalized.dtype
    )
    mapping = hierarchy.to(features.device)[:, labels]
    total = normalized.sum() * 0.0
    for level, (offset, size) in enumerate(
        zip(hierarchy.level_offsets, hierarchy.num_categories_per_level)
    ):
        level_loss = _dynamic_level_reference(
            normalized,
            mapping[level],
            prototype_snapshot[offset : offset + size],
            temperature,
        )
        total = total + hierarchy.level_weights[level] * level_loss
    return total


@torch.no_grad()
def _dynamic_prototype_update(
    features, labels, prototypes, counts, hierarchy, epsilon
):
    normalized = F.normalize(features.detach(), dim=1)
    mapping = hierarchy.to(features.device)[:, labels]
    for level, (offset, size) in enumerate(
        zip(hierarchy.level_offsets, hierarchy.num_categories_per_level)
    ):
        update_factor = epsilon ** (hierarchy.num_levels - level - 1)
        for category in range(size):
            category_mask = mapping[level] == category
            if not bool(category_mask.any()):
                continue
            mean = normalized[category_mask].mean(dim=0).to(prototypes.dtype)
            index = offset + category
            prototypes[index] = F.normalize(
                (1.0 - update_factor) * prototypes[index]
                + update_factor * mean,
                dim=0,
            )
            counts[index] += category_mask.sum().to(counts.dtype)


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device_name", _DEVICES)
def test_vectorized_bhcl_matches_dynamic_reference_loss_gradient_and_bank(
    device_name: str,
) -> None:
    """Guard the synchronization optimization against any equation drift."""

    device = torch.device(device_name)
    generator = torch.Generator().manual_seed(1729)
    base_features = torch.randn(14, 11, generator=generator).to(device)
    labels = torch.tensor(
        [0, 0, 1, 2, 4, 4, 7, 13, 13, 13, 20, 22, 24, 24],
        dtype=torch.long,
        device=device,
    )
    criterion = BalancedHierarchicalContrastiveLoss(
        11,
        XH_HIERARCHY,
        temperature=0.17,
        epsilon=0.2,
    ).to(device)
    initial_prototypes = torch.randn(
        criterion.prototypes.shape, generator=generator
    ).to(device)
    initial_counts = torch.arange(
        criterion.prototype_counts.numel(), dtype=torch.long, device=device
    )
    with torch.no_grad():
        criterion.prototypes.copy_(initial_prototypes)
        criterion.prototype_counts.copy_(initial_counts)

    optimized_features = base_features.clone().requires_grad_()
    reference_features = base_features.clone().requires_grad_()
    reference_prototypes = initial_prototypes.clone()
    reference_counts = initial_counts.clone()

    optimized_loss = criterion(optimized_features, labels)
    reference_loss = _dynamic_bhcl_reference(
        reference_features,
        labels,
        reference_prototypes,
        XH_HIERARCHY,
        criterion.temperature,
    )
    _dynamic_prototype_update(
        reference_features,
        labels,
        reference_prototypes,
        reference_counts,
        XH_HIERARCHY,
        criterion.epsilon,
    )
    optimized_loss.backward()
    reference_loss.backward()

    assert torch.allclose(optimized_loss, reference_loss, rtol=3e-5, atol=3e-5)
    assert torch.allclose(
        optimized_features.grad,
        reference_features.grad,
        rtol=5e-5,
        atol=5e-5,
    )
    assert torch.allclose(
        criterion.prototypes,
        reference_prototypes,
        rtol=3e-5,
        atol=3e-5,
    )
    assert torch.equal(criterion.prototype_counts, reference_counts)


def test_equation_8_excludes_anchor_but_keeps_i_prime_class_divisor() -> None:
    criterion = BalancedHierarchicalContrastiveLoss(
        2, TWO_CLASS_HIERARCHY, temperature=1.0
    )
    with torch.no_grad():
        criterion.prototypes.copy_(torch.eye(2))
    features = torch.eye(2, requires_grad=True)
    labels = torch.tensor([0, 1])

    loss = criterion(features, labels, update_prototypes=False)

    # For either anchor: own class contributes exp(1)/2 because the anchor is
    # excluded while |I'_c| remains 2.  The other class contributes 1.
    expected = math.log(math.e / 2.0 + 1.0) - 1.0
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_equation_9_uses_other_instances_and_own_prototype_as_positives() -> None:
    criterion = BalancedHierarchicalContrastiveLoss(
        2, TWO_CLASS_HIERARCHY, temperature=1.0
    )
    with torch.no_grad():
        criterion.prototypes.copy_(torch.eye(2))
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1])

    loss = criterion(features, labels, update_prototypes=False)

    class_zero_anchor = math.log(2.0 * math.e / 3.0 + 1.0) - 1.0
    class_one_anchor = math.log(1.0 + math.e / 2.0) - 1.0
    expected = (2.0 * class_zero_anchor + class_one_anchor) / 3.0
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_equation_10_uses_level_dependent_ema_and_skips_absent_nodes() -> None:
    criterion = BalancedHierarchicalContrastiveLoss(
        2, XH_HIERARCHY, temperature=1.0, epsilon=0.1
    )
    coarse_ship = XH_HIERARCHY.flat_node_index(0, 0)
    fine_zero = XH_HIERARCHY.flat_node_index(1, 0)
    absent_fine = XH_HIERARCHY.flat_node_index(1, 1)
    with torch.no_grad():
        criterion.prototypes[coarse_ship] = torch.tensor([1.0, 0.0])
        criterion.prototypes[fine_zero] = torch.tensor([1.0, 0.0])
        criterion.prototypes[absent_fine] = torch.tensor([-1.0, 0.0])

    features = torch.tensor([[0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0])
    loss = criterion(features, labels)

    expected_coarse = torch.nn.functional.normalize(
        torch.tensor([0.9, 0.1]), dim=0
    )
    assert torch.allclose(criterion.prototypes[coarse_ship], expected_coarse)
    assert torch.allclose(
        criterion.prototypes[fine_zero], torch.tensor([0.0, 1.0])
    )
    assert torch.equal(
        criterion.prototypes[absent_fine], torch.tensor([-1.0, 0.0])
    )
    assert criterion.prototype_counts[coarse_ship].item() == 1
    assert criterion.prototype_counts[fine_zero].item() == 1
    assert criterion.prototype_counts[absent_fine].item() == 0

    # Updating the live bank during forward must not invalidate autograd's
    # detached prototype snapshot.
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_empty_foreground_returns_differentiable_zero_without_update() -> None:
    criterion = BalancedHierarchicalContrastiveLoss(4, XH_HIERARCHY)
    features = torch.empty((0, 4), requires_grad=True)
    labels = torch.empty((0,), dtype=torch.long)

    loss = criterion(features, labels)

    assert loss.ndim == 0
    assert loss.item() == 0.0
    assert criterion.prototype_counts.sum().item() == 0
    loss.backward()
    assert features.grad is not None


def test_each_loss_instance_owns_an_independent_prototype_bank() -> None:
    first = BalancedHierarchicalContrastiveLoss(2, XH_HIERARCHY)
    second = BalancedHierarchicalContrastiveLoss(2, XH_HIERARCHY)

    first(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))

    assert first.prototype_counts.sum().item() == 2
    assert second.prototype_counts.sum().item() == 0
    assert not torch.equal(first.prototypes, second.prototypes)
