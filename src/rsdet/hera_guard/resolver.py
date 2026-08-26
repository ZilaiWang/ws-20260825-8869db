"""Metric-Aligned Asymmetric Resolver (MAR).

All evidence columns are oriented so larger means "safer / more likely a TP".
Softplus-constrained weights make the resolver monotone in every evidence
column.  A bounded residual keeps the detector score as the dominant anchor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

try:
    from torch import nn as _nn
except ImportError:  # pragma: no cover

    class _ModuleBase:  # type: ignore[no-redef]
        pass
else:
    _ModuleBase = _nn.Module


class MonotoneAsymmetricResolver(_ModuleBase):
    """Bounded monotone residual reranker; it never performs a hard drop."""

    def __init__(self, evidence_dim: int, *, rho_max: float = 1.5) -> None:
        import torch
        from torch import nn

        super().__init__()
        if evidence_dim <= 0 or rho_max <= 0:
            raise ValueError("invalid resolver dimensions")
        self.evidence_dim = int(evidence_dim)
        self.rho_max = float(rho_max)
        self.raw_weights = nn.Parameter(torch.zeros(evidence_dim))
        self.bias = nn.Parameter(torch.zeros(()))
        self.rho_raw = nn.Parameter(torch.zeros(()))

    def forward(self, detector_score: Any, oriented_evidence: Any) -> Any:
        import torch
        from torch.nn import functional as functional

        if oriented_evidence.ndim != 2 or oriented_evidence.shape[1] != self.evidence_dim:
            raise ValueError("oriented_evidence shape mismatch")
        if detector_score.ndim != 1 or detector_score.shape[0] != oriented_evidence.shape[0]:
            raise ValueError("detector_score shape mismatch")
        eps = torch.finfo(detector_score.dtype).eps
        base = torch.logit(detector_score.clamp(eps, 1.0 - eps))
        weights = functional.softplus(self.raw_weights)
        evidence_delta = self.bias + (oriented_evidence * weights).sum(dim=1)
        rho = torch.sigmoid(self.rho_raw) * self.rho_max
        return torch.sigmoid(base + rho * torch.tanh(evidence_delta))

    def constrained_parameters(self) -> dict[str, Any]:
        from torch import sigmoid
        from torch.nn import functional as functional

        return {
            "weights": functional.softplus(self.raw_weights),
            "rho": sigmoid(self.rho_raw) * self.rho_max,
            "bias": self.bias,
        }


@dataclass(frozen=True)
class CategoryResolution:
    category_id: int
    changed: bool
    reason: str


def resolve_fine_category(
    *,
    detector_category_id: int,
    fine_probabilities: Sequence[float],
    coarse_of_fine: Sequence[int],
    minimum_probability: float,
    minimum_margin: float,
    protect_probability: float,
    maximum_protect_for_change: float,
) -> CategoryResolution:
    """Conservative within-coarse relabel rule with TP-protection veto."""

    probabilities = [float(value) for value in fine_probabilities]
    if len(probabilities) != len(coarse_of_fine) or not probabilities:
        raise ValueError("fine probabilities/coarse map mismatch")
    if any(not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("fine probabilities must be finite and non-negative")
    if detector_category_id not in range(len(probabilities)):
        raise ValueError("detector category is out of range")
    detector_coarse = int(coarse_of_fine[detector_category_id])
    allowed = [
        index for index, coarse in enumerate(coarse_of_fine) if int(coarse) == detector_coarse
    ]
    ranked = sorted(allowed, key=lambda index: (-probabilities[index], index))
    top1 = ranked[0]
    top2_probability = probabilities[ranked[1]] if len(ranked) > 1 else 0.0
    margin = probabilities[top1] - top2_probability
    if top1 == detector_category_id:
        return CategoryResolution(detector_category_id, False, "already_agrees")
    if protect_probability > maximum_protect_for_change:
        return CategoryResolution(detector_category_id, False, "protected_tp_veto")
    if probabilities[top1] < minimum_probability or margin < minimum_margin:
        return CategoryResolution(detector_category_id, False, "insufficient_fine_evidence")
    return CategoryResolution(top1, True, "within_coarse_high_confidence")


__all__ = [
    "CategoryResolution",
    "MonotoneAsymmetricResolver",
    "resolve_fine_category",
]
