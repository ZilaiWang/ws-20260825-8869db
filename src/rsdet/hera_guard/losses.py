"""Numerically safe multi-task losses for Proposal-Aligned Verifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PAVLossWeights:
    foreground: float = 1.0
    coarse: float = 0.25
    fine: float = 0.75
    quality: float = 0.25
    protect: float = 0.50
    active_fp: float = 0.50


def _connected_zero(tensor: Any) -> Any:
    return tensor.sum() * 0.0


def balanced_softmax_cross_entropy(
    logits: Any,
    targets: Any,
    class_counts: Any,
) -> Any:
    """Balanced Softmax CE; empty target batches return a connected zero."""

    import torch
    from torch.nn import functional as functional

    if targets.numel() == 0:
        return _connected_zero(logits)
    counts = torch.as_tensor(class_counts, dtype=logits.dtype, device=logits.device)
    if counts.ndim != 1 or counts.numel() != logits.shape[1] or torch.any(counts <= 0):
        raise ValueError("class_counts must be positive and aligned with logits")
    adjusted_logits = logits + counts.log().unsqueeze(0)
    return functional.cross_entropy(adjusted_logits, targets.long())


def pav_multitask_loss(
    output: Any,
    targets: Mapping[str, Any],
    *,
    fine_class_counts: Any,
    weights: PAVLossWeights = PAVLossWeights(),
    protected_tp_weight: float = 4.0,
    active_fp_weight: float = 3.0,
) -> dict[str, Any]:
    """Compute PAV losses without NaNs on background-only mini-batches.

    Required target keys are ``foreground``, ``coarse``, ``fine``, ``quality``
    ``protect`` and ``active_fp``.  Coarse/fine identity is supervised only for
    geometry-level foreground rows.  Protection and active-risk positives use
    asymmetric weights because both are rare at the frozen workpoint.
    """

    import torch
    from torch.nn import functional as functional

    foreground = targets["foreground"].float()
    coarse = targets["coarse"].long()
    fine = targets["fine"].long()
    quality = targets["quality"].float()
    protect = targets["protect"].float()
    active_fp = targets["active_fp"].float()
    batch_size = output.foreground_logit.shape[0]
    if any(value.shape[0] != batch_size for value in targets.values()):
        raise ValueError("all PAV targets must align with the batch")
    if torch.any((quality < 0) | (quality > 1)):
        raise ValueError("quality targets must be in [0, 1]")

    foreground_loss = functional.binary_cross_entropy_with_logits(
        output.foreground_logit, foreground
    )
    positive = foreground > 0.5
    if positive.any():
        coarse_loss = functional.cross_entropy(output.coarse_logits[positive], coarse[positive])
        fine_loss = balanced_softmax_cross_entropy(
            output.fine_logits[positive], fine[positive], fine_class_counts
        )
    else:
        coarse_loss = _connected_zero(output.coarse_logits)
        fine_loss = _connected_zero(output.fine_logits)
    quality_loss = functional.binary_cross_entropy_with_logits(output.quality_logit, quality)
    protect_weights = torch.where(
        protect > 0.5,
        torch.full_like(protect, float(protected_tp_weight)),
        torch.ones_like(protect),
    )
    protect_loss = functional.binary_cross_entropy_with_logits(
        output.protect_logit,
        protect,
        weight=protect_weights,
    )
    active_weights = torch.where(
        active_fp > 0.5,
        torch.full_like(active_fp, float(active_fp_weight)),
        torch.ones_like(active_fp),
    )
    active_fp_loss = functional.binary_cross_entropy_with_logits(
        output.active_fp_logit,
        active_fp,
        weight=active_weights,
    )
    total = (
        weights.foreground * foreground_loss
        + weights.coarse * coarse_loss
        + weights.fine * fine_loss
        + weights.quality * quality_loss
        + weights.protect * protect_loss
        + weights.active_fp * active_fp_loss
    )
    return {
        "total": total,
        "foreground": foreground_loss,
        "coarse": coarse_loss,
        "fine": fine_loss,
        "quality": quality_loss,
        "protect": protect_loss,
        "active_fp": active_fp_loss,
        "n_foreground": int(positive.sum().item()),
    }


__all__ = [
    "PAVLossWeights",
    "balanced_softmax_cross_entropy",
    "pav_multitask_loss",
]
