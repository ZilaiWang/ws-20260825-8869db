"""Targeted EQLv2-style classification gradient balancing for YOLO26.

The detector, task-aligned assignment, box/DFL losses, label space and
inference path remain unchanged.  Only classification-loss elements for the
pre-registered weak Ship/Vehicle classes are reweighted from their accumulated
positive/negative logit-gradient ratio.  Other fine classes remain exact BCE.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def eqlv2_classification_loss(
    *,
    focus_class_indices: Sequence[int],
    gamma: float = 12.0,
    mu: float = 0.8,
    alpha: float = 4.0,
) -> Any:
    """Return elementwise targeted EQLv2 loss with paper-default dynamics.

    Ultralytics supplies soft task-aligned targets.  A target greater than zero
    is treated as positive for choosing the EQLv2 branch, while its original
    soft value is retained in BCE.  Gradient statistics are accumulated from
    the weighted BCE derivative and are training-only state.
    """
    focus = tuple(sorted({int(index) for index in focus_class_indices}))
    if not focus or focus[0] < 0:
        raise ValueError("focus_class_indices must contain non-negative indices")
    if gamma <= 0.0 or not 0.0 <= mu <= 1.0 or alpha < 0.0:
        raise ValueError("gamma must be positive, mu in [0, 1], alpha non-negative")

    import torch
    from torch import nn
    from torch.nn import functional as F

    class _TargetedEQLv2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.focus = focus
            self.gamma = float(gamma)
            self.mu = float(mu)
            self.alpha = float(alpha)
            self._positive_gradient: torch.Tensor | None = None
            self._negative_gradient: torch.Tensor | None = None
            self._updates = 0

        def _state(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if focus[-1] >= logits.shape[-1]:
                raise ValueError("focus class index exceeds logits class dimension")
            if self._positive_gradient is None:
                self._positive_gradient = torch.zeros(
                    logits.shape[-1], dtype=torch.float64, device=logits.device
                )
                self._negative_gradient = torch.zeros_like(self._positive_gradient)
            elif self._positive_gradient.shape[0] != logits.shape[-1]:
                raise ValueError("classification dimension changed during training")
            return self._positive_gradient, self._negative_gradient  # type: ignore[return-value]

        def current_weights(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            positive, negative = self._state(logits)
            ratio = positive / negative.clamp_min(1e-12)
            negative_weight = torch.sigmoid(self.gamma * (ratio - self.mu))
            positive_weight = 1.0 + self.alpha * (1.0 - negative_weight)
            class_mask = torch.zeros_like(negative_weight, dtype=torch.bool)
            class_mask[list(focus)] = True
            ones = torch.ones_like(negative_weight)
            return (
                torch.where(class_mask, positive_weight, ones).to(logits.dtype),
                torch.where(class_mask, negative_weight, ones).to(logits.dtype),
            )

        def forward(self, logits: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
            target_scores = target_scores.to(dtype=logits.dtype)
            positive_mask = target_scores.gt(0)
            positive_weight, negative_weight = self.current_weights(logits)
            weights = torch.where(positive_mask, positive_weight, negative_weight)
            loss = F.binary_cross_entropy_with_logits(
                logits.float(), target_scores.float(), reduction="none"
            ).to(dtype=logits.dtype)
            with torch.no_grad():
                gradient = (logits.detach().sigmoid() - target_scores).abs() * weights
                reduce_dims = tuple(range(gradient.ndim - 1))
                positive, negative = self._state(logits)
                positive.add_((gradient * positive_mask).sum(dim=reduce_dims).double())
                negative.add_((gradient * ~positive_mask).sum(dim=reduce_dims).double())
                self._updates += 1
            return loss * weights

        def audit(self) -> dict[str, Any]:
            if self._positive_gradient is None or self._negative_gradient is None:
                return {"updates": 0, "focus_class_indices": list(focus)}
            positive = self._positive_gradient.detach().cpu()
            negative = self._negative_gradient.detach().cpu()
            ratio = positive / negative.clamp_min(1e-12)
            negative_weight = (self.gamma * (ratio - self.mu)).sigmoid()
            positive_weight = 1.0 + self.alpha * (1.0 - negative_weight)
            class_mask = torch.zeros_like(negative_weight, dtype=torch.bool)
            class_mask[list(focus)] = True
            negative_weight = torch.where(class_mask, negative_weight, torch.ones_like(negative_weight))
            positive_weight = torch.where(class_mask, positive_weight, torch.ones_like(positive_weight))
            return {
                "updates": self._updates,
                "focus_class_indices": list(focus),
                "gamma": self.gamma,
                "mu": self.mu,
                "alpha": self.alpha,
                "positive_gradient": positive.tolist(),
                "negative_gradient": negative.tolist(),
                "positive_negative_ratio": ratio.tolist(),
                "positive_weight": positive_weight.tolist(),
                "negative_weight": negative_weight.tolist(),
            }

    return _TargetedEQLv2()


def eqlv2_trainer(
    *,
    focus_class_indices: Sequence[int],
    gamma: float = 12.0,
    mu: float = 0.8,
    alpha: float = 4.0,
) -> type:
    """Return a trainer replacing only the elementwise classification BCE."""
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class _EQLv2Trainer(DetectionTrainer):
        def _setup_train(self) -> None:
            super()._setup_train()
            model = unwrap_model(self.model)
            criterion = getattr(model, "criterion", None)
            if criterion is None and hasattr(model, "init_criterion"):
                criterion = model.init_criterion()
                model.criterion = criterion
            if hasattr(criterion, "one2many") and hasattr(criterion, "one2one"):
                criterion.one2many.bce = eqlv2_classification_loss(
                    focus_class_indices=focus_class_indices, gamma=gamma, mu=mu, alpha=alpha
                )
                criterion.one2one.bce = eqlv2_classification_loss(
                    focus_class_indices=focus_class_indices, gamma=gamma, mu=mu, alpha=alpha
                )
            elif hasattr(criterion, "bce"):
                criterion.bce = eqlv2_classification_loss(
                    focus_class_indices=focus_class_indices, gamma=gamma, mu=mu, alpha=alpha
                )
            else:
                raise TypeError(f"unsupported detection criterion for EQLv2: {type(criterion).__name__}")

    _EQLv2Trainer.__name__ = "TargetedEQLv2Trainer"
    return _EQLv2Trainer


def eqlv2_criterion_audit(model: Any) -> dict[str, Any]:
    """Extract both YOLO26 branches' accumulated training-only EQLv2 state."""
    from ultralytics.utils.torch_utils import unwrap_model

    criterion = getattr(unwrap_model(model), "criterion", None)
    if criterion is None:
        raise RuntimeError("training criterion is unavailable")
    if hasattr(criterion, "one2many") and hasattr(criterion, "one2one"):
        modules = {"one2many": criterion.one2many.bce, "one2one": criterion.one2one.bce}
    elif hasattr(criterion, "bce"):
        modules = {"detection": criterion.bce}
    else:
        raise TypeError(f"unsupported detection criterion for EQLv2 audit: {type(criterion).__name__}")
    if any(not hasattr(module, "audit") for module in modules.values()):
        raise TypeError("criterion does not contain the targeted EQLv2 loss")
    return {name: module.audit() for name, module in modules.items()}
