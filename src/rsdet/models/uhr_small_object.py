"""Paper-derived sparse small-object utilities for the BHC-DETR adapter.

This module implements the parts of Li et al., *UHR-DETR: Efficient End-to-End
Small Object Detection for Ultra-High-Resolution Remote Sensing Imagery*
(arXiv:2604.21435v1) that are fully specified in the paper:

* the discrete Gain Map expectation on the squared support ``b**2``;
* IoF-sum Gain Map targets and Distribution Focal Loss;
* four-neighbour Local Peak Margin loss;
* the linear Iterative Soft-Subtraction Greedy Algorithm (ISSGA).

The competition training set contains pre-cropped images rather than annotated
10K canvases.  ``select_sparse_local_tokens`` is therefore an explicit
project-compatible adaptation: ISSGA selects bounded C4 tokens inside each
training/inference tile instead of pretending that crop-only data can supervise
the paper's full-image raw-patch router.  The distinction is recorded in the
training configuration and documentation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _group_count(channels: int, requested: int) -> int:
    """Return the largest valid GroupNorm group count no larger than requested."""

    _positive_int(channels, "channels")
    _positive_int(requested, "requested groups")
    for groups in range(min(channels, requested), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class GainMapHead(nn.Module):
    """Paper Fig. 2 gain head: 3x3 Conv -> GN -> ReLU -> 1x1 Conv."""

    def __init__(
        self,
        input_dim: int,
        *,
        bin_limit: int = 6,
        group_norm_groups: int = 32,
    ) -> None:
        super().__init__()
        _positive_int(input_dim, "input_dim")
        _positive_int(bin_limit, "bin_limit")
        groups = _group_count(input_dim, group_norm_groups)
        self.bin_limit = bin_limit
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, bin_limit + 1, kernel_size=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError("gain head features must be [B,C,H,W]")
        return self.layers(features)


def gain_map_expectation(logits: Tensor) -> Tensor:
    """Map gain logits to ``sum_b b^2 p_b`` from paper Eq. (3)."""

    if logits.ndim != 4 or logits.shape[1] < 2:
        raise ValueError("gain logits must be [B,M+1,H,W] with M >= 1")
    support = torch.arange(
        logits.shape[1],
        dtype=logits.dtype,
        device=logits.device,
    ).square()
    probability = logits.softmax(dim=1)
    return (probability * support[None, :, None, None]).sum(dim=1)


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        ),
        dim=-1,
    )


def build_gain_map_targets(
    targets: Sequence[Mapping[str, Tensor]],
    *,
    spatial_size: tuple[int, int],
    patch_fraction: tuple[float, float],
    bin_limit: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build IoF-sum targets from paper Eq. (4) in normalized image space.

    ``patch_fraction`` is ``(patch_height / image_height,
    patch_width / image_width)``.  Border patches are clipped to the valid
    image canvas.  The target is capped at ``bin_limit**2`` exactly as in the
    paper.
    """

    height, width = spatial_size
    _positive_int(height, "gain target height")
    _positive_int(width, "gain target width")
    _positive_int(bin_limit, "bin_limit")
    patch_height, patch_width = (float(value) for value in patch_fraction)
    if not all(
        math.isfinite(value) and 0.0 < value <= 1.0 for value in (patch_height, patch_width)
    ):
        raise ValueError("patch fractions must be finite and in (0, 1]")

    if device is None:
        device = next(
            (
                target["boxes"].device
                for target in targets
                if isinstance(target.get("boxes"), Tensor)
            ),
            torch.device("cpu"),
        )
    y_centers = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    x_centers = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    center_y, center_x = torch.meshgrid(y_centers, x_centers, indexing="ij")
    patch_x1 = (center_x - patch_width * 0.5).clamp(0.0, 1.0)
    patch_y1 = (center_y - patch_height * 0.5).clamp(0.0, 1.0)
    patch_x2 = (center_x + patch_width * 0.5).clamp(0.0, 1.0)
    patch_y2 = (center_y + patch_height * 0.5).clamp(0.0, 1.0)

    output: list[Tensor] = []
    maximum = float(bin_limit**2)
    for target in targets:
        boxes = target.get("boxes")
        if not isinstance(boxes, Tensor) or boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError("every target must contain boxes shaped [N,4]")
        boxes = boxes.to(device=device, dtype=dtype)
        if boxes.numel() == 0:
            output.append(torch.zeros((height, width), device=device, dtype=dtype))
            continue
        xyxy = _cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
        box_area = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=1e-8) * (xyxy[:, 3] - xyxy[:, 1]).clamp(
            min=1e-8
        )
        intersection_x1 = torch.maximum(patch_x1[..., None], xyxy[:, 0])
        intersection_y1 = torch.maximum(patch_y1[..., None], xyxy[:, 1])
        intersection_x2 = torch.minimum(patch_x2[..., None], xyxy[:, 2])
        intersection_y2 = torch.minimum(patch_y2[..., None], xyxy[:, 3])
        intersection = (intersection_x2 - intersection_x1).clamp(min=0.0) * (
            intersection_y2 - intersection_y1
        ).clamp(min=0.0)
        gain = (intersection / box_area).sum(dim=-1).clamp(max=maximum)
        output.append(gain)
    if not output:
        return torch.empty((0, height, width), device=device, dtype=dtype)
    return torch.stack(output, dim=0)


def distribution_focal_gain_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Distribution Focal Loss on the paper's non-uniform squared support.

    The paper does not publish its interpolation implementation.  This uses the
    auditable GFL convention: linearly interpolate between the two neighbouring
    values in ``{0^2, 1^2, ..., M^2}``.
    """

    if logits.ndim != 4 or targets.shape != (logits.shape[0], *logits.shape[-2:]):
        raise ValueError("gain logits/targets have incompatible shapes")
    support = torch.arange(
        logits.shape[1],
        dtype=logits.dtype,
        device=logits.device,
    ).square()
    bounded = targets.to(logits).clamp(0.0, float((logits.shape[1] - 1) ** 2))
    upper = torch.bucketize(bounded.detach(), support).clamp(1, len(support) - 1)
    lower = upper - 1
    lower_value = support[lower]
    upper_value = support[upper]
    upper_weight = (bounded - lower_value) / (upper_value - lower_value).clamp(min=1e-8)
    lower_weight = 1.0 - upper_weight
    log_probability = logits.log_softmax(dim=1).permute(0, 2, 3, 1)
    lower_loss = -log_probability.gather(-1, lower[..., None]).squeeze(-1)
    upper_loss = -log_probability.gather(-1, upper[..., None]).squeeze(-1)
    loss = lower_weight * lower_loss + upper_weight * upper_loss
    if valid_mask is not None:
        if valid_mask.shape != targets.shape or valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be bool with the gain target shape")
        weights = valid_mask.to(loss.dtype)
        return (loss * weights).sum() / weights.sum().clamp(min=1.0)
    return loss.mean()


def ground_truth_peak_mask(targets: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    """Extract deterministic four-neighbour peaks for LPM supervision.

    The paper leaves equal-valued plateau handling unspecified.  Exact value
    comparisons plus a row-major rank tie-break produce a peak for every
    separated plateau while avoiding contradictory margin constraints between
    adjacent cells of the same plateau.  Invalid neighbours are treated as
    negative infinity.
    """

    if targets.ndim != 3:
        raise ValueError("gain targets must be [B,H,W]")
    if valid_mask is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    else:
        if valid_mask.shape != targets.shape or valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be bool with the gain target shape")
        valid = valid_mask
    height, width = targets.shape[-2:]
    rank = (
        torch.arange(
            height * width,
            dtype=torch.long,
            device=targets.device,
        )
        .reshape(1, height, width)
        .expand_as(targets)
    )
    values = targets.masked_fill(~valid, -torch.inf)
    negative = torch.full((), -torch.inf, dtype=targets.dtype, device=targets.device)
    last_rank = torch.full(
        (),
        height * width,
        dtype=rank.dtype,
        device=targets.device,
    )
    neighbours = (
        (
            torch.cat((negative.expand_as(values[:, :1]), values[:, :-1]), dim=1),
            torch.cat((last_rank.expand_as(rank[:, :1]), rank[:, :-1]), dim=1),
        ),
        (
            torch.cat((values[:, 1:], negative.expand_as(values[:, :1])), dim=1),
            torch.cat((rank[:, 1:], last_rank.expand_as(rank[:, :1])), dim=1),
        ),
        (
            torch.cat((negative.expand_as(values[:, :, :1]), values[:, :, :-1]), dim=2),
            torch.cat((last_rank.expand_as(rank[:, :, :1]), rank[:, :, :-1]), dim=2),
        ),
        (
            torch.cat((values[:, :, 1:], negative.expand_as(values[:, :, :1])), dim=2),
            torch.cat((rank[:, :, 1:], last_rank.expand_as(rank[:, :, :1])), dim=2),
        ),
    )
    peaks = (targets > 0.0) & valid
    for neighbour_value, neighbour_rank in neighbours:
        peaks &= (values > neighbour_value) | (
            (values == neighbour_value) & (rank < neighbour_rank)
        )
    return peaks


def local_peak_margin_loss(
    predicted_gain: Tensor,
    target_gain: Tensor,
    *,
    margin: float = 0.05,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Four-neighbour Local Peak Margin loss from paper Eq. (5)."""

    if predicted_gain.shape != target_gain.shape or predicted_gain.ndim != 3:
        raise ValueError("predicted and target gain maps must share [B,H,W]")
    if not math.isfinite(float(margin)) or margin < 0.0:
        raise ValueError("LPM margin must be finite and non-negative")
    valid = torch.ones_like(target_gain, dtype=torch.bool) if valid_mask is None else valid_mask
    if valid.shape != target_gain.shape or valid.dtype is not torch.bool:
        raise ValueError("valid_mask must be bool with the gain target shape")
    peaks = ground_truth_peak_mask(target_gain, valid)
    losses: list[Tensor] = []
    pair_masks: list[Tensor] = []
    # (peak slice, neighbour slice) for north, south, west and east.
    pairs = (
        ((slice(None), slice(1, None), slice(None)), (slice(None), slice(None, -1), slice(None))),
        ((slice(None), slice(None, -1), slice(None)), (slice(None), slice(1, None), slice(None))),
        ((slice(None), slice(None), slice(1, None)), (slice(None), slice(None), slice(None, -1))),
        ((slice(None), slice(None), slice(None, -1)), (slice(None), slice(None), slice(1, None))),
    )
    for peak_slice, neighbour_slice in pairs:
        mask = peaks[peak_slice] & valid[neighbour_slice]
        value = F.relu(predicted_gain[neighbour_slice] + float(margin) - predicted_gain[peak_slice])
        losses.append(value)
        pair_masks.append(mask)
    total = predicted_gain.sum() * 0.0
    for loss, mask in zip(losses, pair_masks):
        weights = mask.to(loss.dtype)
        total = total + (loss * weights).sum()
    # Paper Eq. (5) normalizes the four-neighbour penalty by |Omega|, the
    # number of ground-truth peaks, rather than by the number of neighbour
    # pairs.  Boundary-invalid neighbours contribute zero to the numerator.
    peak_count = peaks.to(predicted_gain.dtype).sum()
    return total / peak_count.clamp(min=1.0)


@torch.no_grad()
def iterative_soft_subtraction(
    gain_map: Tensor,
    *,
    patch_shape: tuple[int, int],
    budget: int,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Linear ISSGA from paper Algorithm 1 and Eqs. (7)-(8).

    Returns integer ``(y, x)`` coordinates shaped ``[B,K,2]`` and the gain at
    each selection shaped ``[B,K]``.  The input is never modified.
    """

    if gain_map.ndim == 2:
        gain_map = gain_map.unsqueeze(0)
    if gain_map.ndim != 3:
        raise ValueError("gain_map must be [H,W] or [B,H,W]")
    patch_height, patch_width = patch_shape
    _positive_int(patch_height, "patch height on gain grid")
    _positive_int(patch_width, "patch width on gain grid")
    _positive_int(budget, "patch budget")
    work = gain_map.detach().float().clamp(min=0.0).clone()
    batch, height, width = work.shape
    if valid_mask is None:
        valid = torch.ones_like(work, dtype=torch.bool)
    else:
        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(0)
        if valid_mask.shape != work.shape or valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be bool with the gain_map shape")
        valid = valid_mask
    has_valid = valid.flatten(1).any(dim=1)
    # A zero-score sentinel keeps downstream attention finite for malformed
    # all-padding samples.  Normal samples still route strictly inside the
    # supplied valid mask.  Tensor indexing avoids a device synchronization.
    valid = valid.clone()
    valid[:, 0, 0] |= ~has_valid
    work[:, 0, 0] = torch.where(
        has_valid,
        work[:, 0, 0],
        torch.zeros_like(work[:, 0, 0]),
    )
    available = valid.clone()
    work = work.masked_fill(~available, -torch.inf)
    y_grid = torch.arange(height, device=work.device, dtype=work.dtype)
    x_grid = torch.arange(width, device=work.device, dtype=work.dtype)
    coordinates: list[Tensor] = []
    scores: list[Tensor] = []
    for _ in range(budget):
        flat_index = work.flatten(1).argmax(dim=1)
        selected_y = torch.div(flat_index, width, rounding_mode="floor")
        selected_x = flat_index.remainder(width)
        selected_score = work.flatten(1).gather(1, flat_index[:, None]).squeeze(1)
        exhausted = ~torch.isfinite(selected_score)
        previous = (
            coordinates[-1]
            if coordinates
            else torch.zeros((batch, 2), dtype=torch.long, device=work.device)
        )
        selected_y = torch.where(exhausted, previous[:, 0], selected_y)
        selected_x = torch.where(exhausted, previous[:, 1], selected_x)
        selected_score = torch.where(
            exhausted,
            torch.zeros_like(selected_score),
            selected_score,
        )
        coordinates.append(torch.stack((selected_y, selected_x), dim=-1))
        scores.append(selected_score)
        distance_y = (y_grid[None, :, None] - selected_y[:, None, None]).abs()
        distance_x = (x_grid[None, None, :] - selected_x[:, None, None]).abs()
        kernel_y = (1.0 - distance_y / float(patch_height)).clamp(min=0.0)
        kernel_x = (1.0 - distance_x / float(patch_width)).clamp(min=0.0)
        kernel = kernel_y * kernel_x
        work = (work - selected_score[:, None, None] * kernel).clamp(min=0.0)
        available[
            torch.arange(batch, device=work.device),
            selected_y,
            selected_x,
        ] = False
        work = work.masked_fill(~available, -torch.inf)
    return torch.stack(coordinates, dim=1), torch.stack(scores, dim=1)


def select_sparse_local_tokens(
    local_features: Tensor,
    local_position: Tensor,
    local_padding_mask: Tensor,
    gain_map: Tensor,
    *,
    gain_valid_mask: Tensor | None = None,
    patch_fraction: tuple[float, float],
    patch_budget: int,
    max_tokens: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Select bounded high-resolution tokens within ISSGA-routed windows.

    This is the documented project adaptation for crop-only training data.  It
    keeps the paper's coverage-maximizing spatial routing while substituting a
    bounded dense-MHA memory for unavailable CUDA Multi-Scale Deformable
    Attention operators.
    """

    if local_features.ndim != 4 or local_position.shape != local_features.shape:
        raise ValueError("local features/position must share [B,C,H,W]")
    batch, channels, height, width = local_features.shape
    if local_padding_mask.shape != (batch, height, width):
        raise ValueError("local_padding_mask has an incompatible shape")
    if gain_map.ndim != 3 or gain_map.shape[0] != batch:
        raise ValueError("gain_map must be [B,Hg,Wg]")
    _positive_int(patch_budget, "patch_budget")
    _positive_int(max_tokens, "max_tokens")
    patch_height_fraction, patch_width_fraction = patch_fraction
    if not all(
        math.isfinite(float(value)) and 0.0 < float(value) <= 1.0 for value in patch_fraction
    ):
        raise ValueError("patch fractions must be in (0,1]")

    gain_height, gain_width = gain_map.shape[-2:]
    gain_patch_height = max(1, int(round(gain_height * patch_height_fraction)))
    gain_patch_width = max(1, int(round(gain_width * patch_width_fraction)))
    coordinates, scores = iterative_soft_subtraction(
        gain_map,
        patch_shape=(gain_patch_height, gain_patch_width),
        budget=patch_budget,
        valid_mask=gain_valid_mask,
    )
    token_features = local_features.flatten(2).transpose(1, 2)
    token_position = local_position.flatten(2).transpose(1, 2)
    dense_scores = F.interpolate(
        gain_map[:, None].to(local_features),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    half_height = height * patch_height_fraction * 0.5
    half_width = width * patch_width_fraction * 0.5
    row_grid = torch.arange(height, device=local_features.device)[None, None, :, None]
    column_grid = torch.arange(width, device=local_features.device)[None, None, None, :]
    route_coordinates = coordinates.float()
    center_y = (route_coordinates[..., 0] + 0.5) * height / gain_height
    center_x = (route_coordinates[..., 1] + 0.5) * width / gain_width
    y1 = (center_y - half_height).floor().clamp(0, height)[..., None, None]
    y2 = (center_y + half_height).ceil().clamp(0, height)[..., None, None]
    x1 = (center_x - half_width).floor().clamp(0, width)[..., None, None]
    x2 = (center_x + half_width).ceil().clamp(0, width)[..., None, None]
    window_masks = (
        (row_grid >= y1)
        & (row_grid < y2)
        & (column_grid >= x1)
        & (column_grid < x2)
        & (~local_padding_mask[:, None])
    ).flatten(2)

    token_count = height * width
    valid_tokens = (~local_padding_mask).flatten(1)
    candidate_mask = window_masks.any(dim=1)
    has_routed_candidate = candidate_mask.any(dim=1)
    # Match the previous fallback exactly: if all routed windows are empty but
    # the sample has a valid local token, retain only its first row-major token.
    first_valid = valid_tokens.to(torch.long).argmax(dim=1)
    first_valid_mask = F.one_hot(first_valid, token_count).to(torch.bool)
    first_valid_mask &= valid_tokens
    candidate_mask = torch.where(
        has_routed_candidate[:, None],
        candidate_mask,
        first_valid_mask,
    )

    flat_scores = dense_scores.flatten(1)
    flat_indices = torch.arange(token_count, device=local_features.device)[None]

    def fixed_mask_selection(mask: Tensor, capacity: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Select a fixed-width equivalent of the former dynamic index list.

        When a mask fits in ``capacity``, indices remain in row-major order.
        Otherwise the highest Gain scores are returned in ``topk`` order.  The
        validity tensor carries the dynamic length entirely on device.
        """

        take = min(capacity, token_count)
        counts = mask.sum(dim=1)
        if take == 0:
            empty_indices = torch.empty((batch, 0), dtype=torch.long, device=local_features.device)
            empty_valid = torch.empty((batch, 0), dtype=torch.bool, device=local_features.device)
            return empty_indices, empty_valid, counts, empty_indices

        # Unique integer keys stably pack valid row-major indices ahead of all
        # masked locations without ``nonzero`` or a data-dependent allocation.
        row_keys = torch.where(mask, flat_indices, flat_indices + token_count)
        row_indices = row_keys.topk(take, dim=1, largest=False, sorted=True).indices
        score_indices = (
            flat_scores.masked_fill(~mask, -torch.inf).topk(take, dim=1, sorted=True).indices
        )
        use_scores = counts > capacity
        selected = torch.where(use_scores[:, None], score_indices, row_indices)
        selected_valid = (
            torch.arange(take, device=local_features.device)[None]
            < counts.clamp(max=capacity)[:, None]
        )
        if take < capacity:
            pad_width = capacity - take
            selected = F.pad(selected, (0, pad_width), value=0)
            selected_valid = F.pad(selected_valid, (0, pad_width), value=False)
            score_indices = F.pad(score_indices, (0, pad_width), value=0)
        return selected, selected_valid, counts, score_indices

    def compact_selection(indices: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """Stable-partition valid fixed-width selections without host sync."""

        width = indices.shape[1]
        if width == 0:
            return indices, valid
        positions = torch.arange(width, device=indices.device)[None]
        order = torch.where(valid, positions, positions + width).argsort(dim=1)
        packed_indices = indices.gather(1, order)
        counts = valid.sum(dim=1)
        packed_valid = positions < counts[:, None]
        return packed_indices, packed_valid

    # Reserve the same per-route quota as before.  Only the fixed route loop
    # remains; batch samples and window membership stay batched on the device.
    chosen_mask = torch.zeros_like(candidate_mask)
    route_indices: list[Tensor] = []
    route_validity: list[Tensor] = []
    base_quota, remainder = divmod(max_tokens, patch_budget)
    for window_index in range(patch_budget):
        quota = base_quota + int(window_index < remainder)
        if quota == 0:
            continue
        available_mask = window_masks[:, window_index] & ~chosen_mask
        indices, valid, counts, _ = fixed_mask_selection(available_mask, quota)
        route_indices.append(indices)
        route_validity.append(valid)

        # If the route overflows its quota, mark only its score-selected top-k;
        # otherwise every available row-major token was retained.  ``scatter``
        # receives unique top-k indices whenever its result is used.
        score_selected = counts > quota
        score_mask = torch.zeros_like(chosen_mask)
        score_mask.scatter_(1, indices, True)
        chosen_mask |= torch.where(score_selected[:, None], score_mask, available_mask)

    if route_indices:
        selected_indices, selected_valid = compact_selection(
            torch.cat(route_indices, dim=1),
            torch.cat(route_validity, dim=1),
        )
    else:  # ``max_tokens`` is positive, kept for type/shape robustness.
        selected_indices = torch.empty((batch, 0), dtype=torch.long, device=local_features.device)
        selected_valid = torch.empty((batch, 0), dtype=torch.bool, device=local_features.device)

    selected_count = selected_valid.sum(dim=1)
    remaining_capacity = max_tokens - selected_count
    remaining_mask = candidate_mask & ~chosen_mask
    # Produce the maximum fixed-width candidates once.  Taking the first
    # ``remaining_capacity`` entries is equivalent to the old dynamic top-k.
    row_fill, row_fill_valid, remaining_count, score_fill = fixed_mask_selection(
        remaining_mask, max_tokens
    )
    use_score_fill = remaining_count > remaining_capacity
    fill_indices = torch.where(use_score_fill[:, None], score_fill, row_fill)
    fill_valid = row_fill_valid & (
        torch.arange(max_tokens, device=local_features.device)[None] < remaining_capacity[:, None]
    )

    selected_indices, selected_valid = compact_selection(
        torch.cat((selected_indices, fill_indices), dim=1),
        torch.cat((selected_valid, fill_valid), dim=1),
    )
    selected_indices = selected_indices[:, :max_tokens]
    selected_valid = selected_valid[:, :max_tokens]
    gather_index = selected_indices[..., None].expand(-1, -1, channels)
    gathered_features = token_features.gather(1, gather_index)
    gathered_position = token_position.gather(1, gather_index)
    selected_features = torch.where(
        selected_valid[..., None],
        gathered_features,
        torch.zeros_like(gathered_features),
    )
    selected_position = torch.where(
        selected_valid[..., None],
        gathered_position,
        torch.zeros_like(gathered_position),
    )
    selected_padding = ~selected_valid
    # Safe zero sentinel for an all-padding local feature map.
    has_valid_token = valid_tokens.any(dim=1)
    selected_padding[:, 0] &= has_valid_token
    return (
        selected_features,
        selected_position,
        selected_padding,
        torch.cat(
            (coordinates.to(scores.dtype), scores[..., None]),
            dim=-1,
        ),
    )


__all__ = [
    "GainMapHead",
    "build_gain_map_targets",
    "distribution_focal_gain_loss",
    "gain_map_expectation",
    "ground_truth_peak_mask",
    "iterative_soft_subtraction",
    "local_peak_margin_loss",
    "select_sparse_local_tokens",
]
