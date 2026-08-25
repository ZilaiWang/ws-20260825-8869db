from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def pinball_loss(pred: Tensor, target: Tensor, quantile: float) -> Tensor:
    error = target - pred
    return torch.maximum((quantile - 1.0) * error, quantile * error)


def pairwise_rank_loss(pred: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Listwise-within-set pair loss, restricted to pairs with distinct utility."""

    # pred/target/valid: [B,N,A]
    p_i = pred.unsqueeze(-1)
    p_j = pred.unsqueeze(-2)
    t_i = target.unsqueeze(-1)
    t_j = target.unsqueeze(-2)
    v = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    sign = torch.sign(t_i - t_j)
    informative = v & (sign != 0)
    if not torch.any(informative):
        return pred.sum() * 0.0
    margin = sign * (p_i - p_j)
    return F.softplus(-margin[informative]).mean()


class UtilityLoss(nn.Module):
    def __init__(
        self,
        *,
        rank_weight: float = 0.2,
        action_weight: float = 0.2,
        unsafe_weight: float = 0.5,
        max_sample_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.rank_weight = rank_weight
        self.action_weight = action_weight
        self.unsafe_weight = unsafe_weight
        self.max_sample_weight = max_sample_weight

    def forward(
        self,
        *,
        utility_quantiles: Tensor,
        action_logits: Tensor,
        target_delta: Tensor,
        action_valid: Tensor,
        sample_weight: Tensor | None = None,
    ) -> dict[str, Tensor]:
        # Shapes: [B,N,A,3], [B,N,A], [B,N,A], [B,N,A]
        q10, q50, q90 = utility_quantiles.unbind(dim=-1)
        target = target_delta
        valid = action_valid.bool()

        base = (
            pinball_loss(q10, target, 0.10)
            + F.smooth_l1_loss(q50, target, reduction="none")
            + pinball_loss(q90, target, 0.90)
        )
        if sample_weight is None:
            # High-impact decisions matter more, while capping rare outliers.
            sample_weight = (target.abs().sqrt() + 0.05).clamp_max(
                self.max_sample_weight
            )
        weighted = base * sample_weight
        quantile = weighted[valid].mean() if torch.any(valid) else weighted.sum() * 0.0

        masked_target = target.masked_fill(~valid, torch.finfo(target.dtype).min)
        best_action = masked_target.argmax(dim=-1)
        valid_candidate = valid.any(dim=-1)
        action_ce = F.cross_entropy(
            action_logits[valid_candidate], best_action[valid_candidate]
        ) if torch.any(valid_candidate) else action_logits.sum() * 0.0

        rank = pairwise_rank_loss(q50, target, valid)

        # Penalize confident positive lower bounds for actually harmful actions.
        unsafe = F.relu(q10) * (target < 0).to(q10.dtype) * valid.to(q10.dtype)
        unsafe_loss = unsafe.sum() / valid.sum().clamp_min(1)

        total = (
            quantile
            + self.rank_weight * rank
            + self.action_weight * action_ce
            + self.unsafe_weight * unsafe_loss
        )
        return {
            "loss": total,
            "quantile": quantile.detach(),
            "rank": rank.detach(),
            "action_ce": action_ce.detach(),
            "unsafe": unsafe_loss.detach(),
        }
