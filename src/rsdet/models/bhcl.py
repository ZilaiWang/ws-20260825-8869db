"""Balanced hierarchical contrastive loss from Eqs. (7)--(10)."""

from __future__ import annotations

import math
from typing import Any

from rsdet.models.hierarchy import XH_HIERARCHY, HierarchySpec

try:
    import torch
    import torch.nn.functional as F
    from torch import nn

    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("BalancedHierarchicalContrastiveLoss requires PyTorch")


if _TORCH_AVAILABLE:

    class BalancedHierarchicalContrastiveLoss(nn.Module):
        """Paper-faithful balanced hierarchical contrastive loss.

        One instance owns one flattened prototype bank.  Instantiate one loss
        module per decoder layer to reproduce the paper's independent banks.
        Input features are L2-normalized internally.  The loss is evaluated
        against a detached snapshot of the bank and, when requested, the live
        bank is updated afterwards using Eq. (10).
        """

        def __init__(
            self,
            projection_dim: int,
            hierarchy: HierarchySpec = XH_HIERARCHY,
            temperature: float = 0.1,
            epsilon: float = 0.1,
        ) -> None:
            super().__init__()
            if (
                not isinstance(projection_dim, int)
                or isinstance(projection_dim, bool)
                or projection_dim <= 0
            ):
                raise ValueError("projection_dim must be a positive integer")
            if not isinstance(hierarchy, HierarchySpec):
                raise TypeError("hierarchy must be a HierarchySpec")
            if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
                raise ValueError("temperature must be finite and > 0")
            if not math.isfinite(float(epsilon)) or not 0.0 < float(epsilon) <= 1.0:
                raise ValueError("epsilon must be finite and in (0, 1]")

            self.projection_dim = projection_dim
            self.hierarchy = hierarchy
            self.temperature = float(temperature)
            self.epsilon = float(epsilon)

            self.register_buffer(
                "prototypes",
                torch.zeros(hierarchy.num_nodes, projection_dim),
            )
            self.register_buffer(
                "prototype_counts",
                torch.zeros(hierarchy.num_nodes, dtype=torch.long),
            )
            self.register_buffer("fine_to_level", hierarchy.to("cpu"))
            self.register_buffer(
                "level_weights",
                torch.tensor(hierarchy.level_weights, dtype=torch.float32),
            )
            self.register_buffer(
                "level_offsets",
                torch.tensor(hierarchy.level_offsets, dtype=torch.long),
            )
            self.register_buffer(
                "level_sizes",
                torch.tensor(hierarchy.num_categories_per_level, dtype=torch.long),
            )

        def _validate_inputs(self, projected_features, fine_labels):
            if not isinstance(projected_features, torch.Tensor):
                raise TypeError("projected_features must be a torch.Tensor")
            if projected_features.ndim != 2:
                raise ValueError("projected_features must have shape [N, projection_dim]")
            if projected_features.shape[1] != self.projection_dim:
                raise ValueError(
                    "projected_features has dimension "
                    f"{projected_features.shape[1]}, expected {self.projection_dim}"
                )
            if not projected_features.is_floating_point():
                raise TypeError("projected_features must have a floating-point dtype")
            if not isinstance(fine_labels, torch.Tensor):
                raise TypeError("fine_labels must be a torch.Tensor")
            if fine_labels.ndim != 1:
                raise ValueError("fine_labels must have shape [N]")
            if fine_labels.shape[0] != projected_features.shape[0]:
                raise ValueError("projected_features and fine_labels must have equal length")
            if fine_labels.device != projected_features.device:
                raise ValueError("projected_features and fine_labels must be on the same device")
            if fine_labels.dtype not in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }:
                raise TypeError("fine_labels must have an integer dtype")
            labels = fine_labels.to(dtype=torch.long)
            if labels.numel():
                in_range = torch.all((labels >= 0) & (labels < self.hierarchy.num_fine_classes))
                message = f"fine_labels must be in [0, {self.hierarchy.num_fine_classes})"
                if labels.device.type == "cuda" and hasattr(torch, "_assert_async"):
                    # A Python bool(tensor) forces a host/device synchronization.
                    # PyTorch's CUDA async assertion preserves the validation
                    # contract without stalling every decoder layer.
                    torch._assert_async(in_range, message)
                elif not bool(in_range):
                    raise ValueError(message)
            return labels

        def _balanced_level_loss(
            self,
            features,
            level_labels,
            level_prototypes,
        ):
            """Compute Eqs. (8)--(9) for one non-root hierarchy level."""

            sample_count = features.shape[0]
            category_count = level_prototypes.shape[0]
            pair_logits = features @ features.transpose(0, 1) / self.temperature
            prototype_logits = features @ level_prototypes.transpose(0, 1)
            prototype_logits = prototype_logits / self.temperature
            class_counts = torch.bincount(level_labels, minlength=category_count)

            # Segment the N x N instance logits by their column category.  A
            # scatter log-sum-exp is algebraically identical to the former
            # per-category index_select loop, avoids all CUDA ``nonzero`` host
            # synchronizations, and needs O(N^2 + N*C) rather than O(N^2*C)
            # memory.  The anchor diagonal is excluded exactly as in Eq. (8).
            instance_logits = pair_logits.masked_fill(
                torch.eye(
                    sample_count,
                    dtype=torch.bool,
                    device=features.device,
                ),
                -torch.inf,
            )
            category_index = level_labels[None, :].expand(sample_count, -1)
            with torch.no_grad():
                instance_max = pair_logits.new_full((sample_count, category_count), -torch.inf)
                instance_max.scatter_reduce_(
                    1,
                    category_index,
                    instance_logits.detach(),
                    reduce="amax",
                    include_self=True,
                )
                class_max = torch.maximum(instance_max, prototype_logits.detach())
            centered_instance = instance_logits - class_max.gather(1, category_index)
            instance_exp_sum = pair_logits.new_zeros((sample_count, category_count)).scatter_add(
                1,
                category_index,
                centered_instance.exp(),
            )
            prototype_exp = (prototype_logits - class_max).exp()
            class_log_sum = class_max + (instance_exp_sum + prototype_exp).log()
            divisor = class_counts.to(pair_logits.dtype) + 1.0
            denominator_terms = class_log_sum - divisor.log()[None, :]
            log_denominator = torch.logsumexp(denominator_terms, dim=1)

            same_category = level_labels[:, None] == level_labels[None, :]
            same_category.fill_diagonal_(False)
            positive_instance_sum = pair_logits.masked_fill(~same_category, 0.0).sum(dim=1)
            own_prototype_logits = prototype_logits.gather(1, level_labels[:, None]).squeeze(1)

            # P'_l(i) contains all same-category instances except i, plus the
            # own ancestor prototype.  Its cardinality is therefore |I_c|.
            positive_count = class_counts[level_labels].to(pair_logits.dtype)
            mean_positive_logit = (positive_instance_sum + own_prototype_logits) / positive_count
            return (log_denominator - mean_positive_logit).mean()

        def forward(
            self,
            projected_features,
            fine_labels,
            update_prototypes: bool = True,
        ):
            """Return scalar BHCL for all matched foreground queries.

            Empty foreground sets produce a differentiable zero and leave the
            prototype bank unchanged.
            """

            labels = self._validate_inputs(projected_features, fine_labels)
            if labels.numel() == 0:
                return projected_features.sum() * 0.0
            if not isinstance(update_prototypes, bool):
                raise TypeError("update_prototypes must be a bool")

            features = F.normalize(projected_features, dim=1)
            # Clone because the live buffers can be changed below before
            # backward; autograd needs this constant snapshot for dL/df.
            prototype_snapshot = F.normalize(self.prototypes.detach().clone(), dim=1).to(
                dtype=features.dtype
            )
            mapping = self.fine_to_level[:, labels]

            total = features.sum() * 0.0
            for level, (offset, size) in enumerate(
                zip(
                    self.hierarchy.level_offsets,
                    self.hierarchy.num_categories_per_level,
                )
            ):
                level_loss = self._balanced_level_loss(
                    features,
                    mapping[level],
                    prototype_snapshot[offset : offset + size],
                )
                total = total + self.level_weights[level].to(features.dtype) * level_loss

            if update_prototypes:
                self._update_normalized_prototype_bank(features, labels)
            return total

        @torch.no_grad()
        def _update_normalized_prototype_bank(self, features, labels) -> None:
            """Vectorized Eq. (10) for already validated, normalized inputs."""

            mapping = self.fine_to_level[:, labels]
            level_count = self.hierarchy.num_levels
            for level, (offset, size) in enumerate(
                zip(
                    self.hierarchy.level_offsets,
                    self.hierarchy.num_categories_per_level,
                )
            ):
                level_labels = mapping[level]
                counts = torch.bincount(level_labels, minlength=size)
                sums = features.new_zeros((size, self.projection_dim))
                sums.index_add_(0, level_labels, features)
                means = sums / counts.clamp(min=1).to(features.dtype)[:, None]

                # Python level is zero-based; paper l = level + 1.
                update_factor = self.epsilon ** (level_count - level - 1)
                live = self.prototypes[offset : offset + size]
                means = means.to(dtype=live.dtype)
                updated = F.normalize(
                    (1.0 - update_factor) * live + update_factor * means,
                    dim=1,
                )
                present = counts > 0
                live.copy_(torch.where(present[:, None], updated, live))
                self.prototype_counts[offset : offset + size].add_(
                    counts.to(dtype=self.prototype_counts.dtype)
                )

        @torch.no_grad()
        def update_prototype_bank(self, projected_features, fine_labels) -> None:
            """Apply Eq. (10) to every hierarchy node present in this batch."""

            labels = self._validate_inputs(projected_features, fine_labels)
            if labels.numel() == 0:
                return
            features = F.normalize(projected_features.detach(), dim=1)
            self._update_normalized_prototype_bank(features, labels)

else:

    class BalancedHierarchicalContrastiveLoss:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


__all__ = ["BalancedHierarchicalContrastiveLoss"]
