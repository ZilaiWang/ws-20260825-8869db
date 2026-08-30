"""Small, explicit GroupDRO controller for source/airport robust quality learning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GroupDROResult:
    loss: torch.Tensor
    group_losses: torch.Tensor
    group_counts: torch.Tensor
    group_weights: torch.Tensor


class GroupDROController:
    """Exponentiated-gradient GroupDRO over pre-defined source groups.

    Apply this to the lightweight quality head first, not to the whole detector.
    The outer-fold held-out groups must never be used when constructing or
    updating the training groups.
    """

    def __init__(
        self,
        num_groups: int,
        *,
        step_size: float = 0.05,
        min_group_count: int = 1,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_groups <= 0:
            raise ValueError("num_groups must be positive")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be finite and > 0")
        if min_group_count <= 0:
            raise ValueError("min_group_count must be positive")
        self.num_groups = int(num_groups)
        self.step_size = float(step_size)
        self.min_group_count = int(min_group_count)
        self.weights = torch.full(
            (num_groups,),
            1.0 / num_groups,
            dtype=torch.float32,
            device=device,
        )

    def to(self, device: torch.device | str) -> "GroupDROController":
        self.weights = self.weights.to(device)
        return self

    def state_dict(self) -> dict[str, torch.Tensor | int | float]:
        return {
            "num_groups": self.num_groups,
            "step_size": self.step_size,
            "min_group_count": self.min_group_count,
            "weights": self.weights.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["num_groups"]) != self.num_groups:
            raise ValueError("GroupDRO num_groups mismatch")
        self.weights = torch.as_tensor(state["weights"], dtype=torch.float32).to(
            self.weights.device
        )
        self.weights = self.weights / self.weights.sum().clamp_min(1e-12)

    def __call__(
        self,
        per_sample_loss: torch.Tensor,
        group_ids: torch.Tensor,
        *,
        update: bool,
    ) -> GroupDROResult:
        if per_sample_loss.ndim != 1 or group_ids.shape != per_sample_loss.shape:
            raise ValueError("per_sample_loss and group_ids must be aligned vectors")
        if group_ids.numel() and (
            int(group_ids.min()) < 0 or int(group_ids.max()) >= self.num_groups
        ):
            raise ValueError("group_ids are outside the configured range")

        device = per_sample_loss.device
        weights = self.weights.to(device)
        counts = torch.bincount(group_ids.long(), minlength=self.num_groups).to(
            device=device, dtype=per_sample_loss.dtype
        )
        sums = torch.zeros(self.num_groups, device=device, dtype=per_sample_loss.dtype)
        sums.scatter_add_(0, group_ids.long(), per_sample_loss)
        group_losses = sums / counts.clamp_min(1.0)
        observed = counts >= float(self.min_group_count)

        if update and bool(observed.any()):
            updated = weights.clone()
            updated[observed] = updated[observed] * torch.exp(
                self.step_size * group_losses[observed].detach()
            )
            updated = updated / updated.sum().clamp_min(1e-12)
            self.weights = updated.detach().to(self.weights.device)
            weights = updated

        effective = weights * observed.to(weights.dtype)
        effective = effective / effective.sum().clamp_min(1e-12)
        loss = (effective * group_losses).sum()
        return GroupDROResult(
            loss=loss,
            group_losses=group_losses,
            group_counts=counts,
            group_weights=effective,
        )


__all__ = ["GroupDROController", "GroupDROResult"]
