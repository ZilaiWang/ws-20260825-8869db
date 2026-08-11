"""Formula and shape tests for the UHR-DETR small-object utilities."""

from __future__ import annotations

import subprocess
import sys

import pytest

# Importing torch can fail with a Windows DLL error before pytest can turn the
# exception into a module-level skip.  Probe it in a child process first, as in
# the BHC-DETR dataset tests.
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

from rsdet.models.uhr_small_object import (  # noqa: E402
    GainMapHead,
    build_gain_map_targets,
    distribution_focal_gain_loss,
    gain_map_expectation,
    ground_truth_peak_mask,
    iterative_soft_subtraction,
    local_peak_margin_loss,
    select_sparse_local_tokens,
)


def _dynamic_sparse_token_reference(
    local_features,
    local_position,
    local_padding_mask,
    gain_map,
    *,
    gain_valid_mask,
    patch_fraction,
    patch_budget,
    max_tokens,
):
    """Pre-optimization dynamic/nonzero selector used as a regression oracle."""

    batch, channels, height, width = local_features.shape
    gain_height, gain_width = gain_map.shape[-2:]
    patch_height_fraction, patch_width_fraction = patch_fraction
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
    selected_features = local_features.new_zeros((batch, max_tokens, channels))
    selected_position = local_features.new_zeros((batch, max_tokens, channels))
    selected_padding = torch.ones(
        (batch, max_tokens), dtype=torch.bool, device=local_features.device
    )
    half_height = height * patch_height_fraction * 0.5
    half_width = width * patch_width_fraction * 0.5
    row_grid = torch.arange(height, device=local_features.device)[None, :, None]
    column_grid = torch.arange(width, device=local_features.device)[None, None, :]
    for batch_index in range(batch):
        candidate_mask = torch.zeros(
            (height, width), dtype=torch.bool, device=local_features.device
        )
        window_candidates = []
        route_coordinates = coordinates[batch_index].float()
        center_y = (route_coordinates[:, 0] + 0.5) * height / gain_height
        center_x = (route_coordinates[:, 1] + 0.5) * width / gain_width
        y1 = (center_y - half_height).floor().clamp(0, height)[:, None, None]
        y2 = (center_y + half_height).ceil().clamp(0, height)[:, None, None]
        x1 = (center_x - half_width).floor().clamp(0, width)[:, None, None]
        x2 = (center_x + half_width).ceil().clamp(0, width)[:, None, None]
        window_masks = (row_grid >= y1) & (row_grid < y2) & (column_grid >= x1) & (column_grid < x2)
        window_masks &= ~local_padding_mask[batch_index][None]
        for window_mask in window_masks:
            candidate_mask |= window_mask
            window_candidates.append(torch.nonzero(window_mask.flatten(), as_tuple=False).flatten())
        candidate_mask &= ~local_padding_mask[batch_index]
        candidates = torch.nonzero(candidate_mask.flatten(), as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.nonzero(
                (~local_padding_mask[batch_index]).flatten(), as_tuple=False
            ).flatten()[:1]
        if candidates.numel() == 0:
            selected_padding[batch_index, 0] = False
            continue

        flat_dense_scores = dense_scores[batch_index].flatten()
        chosen_mask = torch.zeros(height * width, dtype=torch.bool, device=local_features.device)
        selected_chunks = []
        base_quota, remainder = divmod(max_tokens, patch_budget)
        for window_index, window in enumerate(window_candidates):
            quota = base_quota + int(window_index < remainder)
            if quota == 0 or window.numel() == 0:
                continue
            available = window[~chosen_mask.index_select(0, window)]
            if available.numel() > quota:
                local_scores = flat_dense_scores.index_select(0, available)
                available = available.index_select(0, local_scores.topk(quota, sorted=True).indices)
            chosen_mask[available] = True
            selected_chunks.append(available)
        selected_count = sum(int(chunk.numel()) for chunk in selected_chunks)
        remaining = max_tokens - selected_count
        if remaining > 0:
            available = candidates[~chosen_mask.index_select(0, candidates)]
            if available.numel() > remaining:
                available_scores = flat_dense_scores.index_select(0, available)
                available = available.index_select(
                    0, available_scores.topk(remaining, sorted=True).indices
                )
            if available.numel():
                selected_chunks.append(available)
        candidates = torch.cat(selected_chunks) if selected_chunks else candidates[:max_tokens]
        count = min(int(candidates.numel()), max_tokens)
        candidates = candidates[:count]
        if count:
            selected_features[batch_index, :count] = token_features[batch_index].index_select(
                0, candidates
            )
            selected_position[batch_index, :count] = token_position[batch_index].index_select(
                0, candidates
            )
            selected_padding[batch_index, :count] = False
    routing = torch.cat((coordinates.to(scores.dtype), scores[..., None]), dim=-1)
    return selected_features, selected_position, selected_padding, routing


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def test_gain_map_head_shape_and_expectation_use_squared_support() -> None:
    head = GainMapHead(4, bin_limit=3, group_norm_groups=3)
    assert head(torch.randn(2, 4, 3, 5)).shape == (2, 4, 3, 5)

    probabilities = torch.tensor([0.25, 0.25, 0.50])
    logits = probabilities.log().reshape(1, 3, 1, 1).requires_grad_()
    expectation = gain_map_expectation(logits)

    # Support is {0^2, 1^2, 2^2}, not the uniformly spaced {0, 1, 2}.
    assert expectation.shape == (1, 1, 1)
    assert expectation.item() == pytest.approx(2.25)
    expectation.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_gain_map_targets_compute_iof_sum_handle_empty_targets_and_cap() -> None:
    targets = [
        {"boxes": torch.tensor([[0.25, 0.25, 0.50, 0.50]])},
        {"boxes": torch.empty((0, 4))},
    ]
    gain = build_gain_map_targets(
        targets,
        spatial_size=(2, 2),
        patch_fraction=(0.5, 0.5),
        bin_limit=2,
    )

    assert gain.shape == (2, 2, 2)
    assert torch.equal(gain[0], torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
    assert torch.count_nonzero(gain[1]) == 0

    repeated_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]]).repeat(5, 1)
    capped = build_gain_map_targets(
        [{"boxes": repeated_boxes}],
        spatial_size=(1, 1),
        patch_fraction=(1.0, 1.0),
        bin_limit=2,
    )
    assert capped.item() == pytest.approx(4.0)


def test_distribution_focal_loss_interpolates_on_non_uniform_squared_bins() -> None:
    logits = torch.tensor([0.2, 1.4, -0.7]).reshape(1, 3, 1, 1).requires_grad_()
    target = torch.tensor([[[2.0]]])

    loss = distribution_focal_gain_loss(logits, target)
    log_probability = logits.detach().flatten().log_softmax(dim=0)
    # 2 lies one third of the way from 1^2 to 2^2.
    expected = -(2.0 / 3.0) * log_probability[1] - (1.0 / 3.0) * log_probability[2]
    assert loss.item() == pytest.approx(expected.item())

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    masked_logits = torch.randn(1, 3, 1, 1, requires_grad=True)
    masked_loss = distribution_focal_gain_loss(
        masked_logits,
        target,
        valid_mask=torch.zeros_like(target, dtype=torch.bool),
    )
    assert masked_loss.item() == 0.0
    masked_loss.backward()
    assert masked_logits.grad is not None
    assert torch.count_nonzero(masked_logits.grad) == 0


def test_local_peak_margin_enforces_four_neighbours_and_plateau_is_deterministic() -> None:
    target = torch.zeros((1, 3, 3))
    target[0, 1, 1] = 1.0

    separated = torch.zeros((1, 3, 3))
    separated[0, 1, 1] = 1.0
    assert local_peak_margin_loss(separated, target, margin=0.2).item() == 0.0

    violating = torch.zeros((1, 3, 3), requires_grad=True)
    with torch.no_grad():
        violating[0, 1, 1] = 0.1
        violating[0, 0, 1] = 0.3
        violating[0, 2, 1] = 0.3
        violating[0, 1, 0] = 0.3
        violating[0, 1, 2] = 0.3
    loss = local_peak_margin_loss(violating, target, margin=0.2)
    # Four violating neighbours, each with hinge value 0.4, normalized by the
    # single GT peak as in paper Eq. (5).
    assert loss.item() == pytest.approx(1.6)
    loss.backward()
    assert violating.grad is not None
    assert violating.grad[0, 1, 1].item() < 0.0
    assert violating.grad[0, 0, 1].item() > 0.0
    assert violating.grad[0, 2, 1].item() > 0.0
    assert violating.grad[0, 1, 0].item() > 0.0
    assert violating.grad[0, 1, 2].item() > 0.0

    plateau_peaks = ground_truth_peak_mask(torch.ones((1, 2, 2)))
    expected_peak = torch.tensor([[[True, False], [False, False]]])
    assert torch.equal(plateau_peaks, expected_peak)

    separated_plateaus = torch.tensor([[[1.0, 1.0, 0.0, 2.0, 2.0]]])
    separated_peaks = ground_truth_peak_mask(separated_plateaus)
    assert torch.equal(
        separated_peaks,
        torch.tensor([[[True, False, False, True, False]]]),
    )


def test_issga_is_batched_non_mutating_and_deterministic() -> None:
    gain = torch.zeros((2, 4, 4))
    gain[0, 1, 1] = 5.0
    gain[0, 3, 3] = 4.0
    gain[1, 0, 0] = 2.0
    gain[1, 0, 1] = 2.0
    original = gain.clone()

    coordinates, scores = iterative_soft_subtraction(
        gain,
        patch_shape=(2, 2),
        budget=2,
    )
    repeated_coordinates, repeated_scores = iterative_soft_subtraction(
        gain,
        patch_shape=(2, 2),
        budget=2,
    )

    assert coordinates.shape == (2, 2, 2)
    assert scores.shape == (2, 2)
    assert coordinates[0].tolist() == [[1, 1], [3, 3]]
    assert coordinates[1, 0].tolist() == [0, 0]
    assert torch.all(scores[:, 1:] <= scores[:, :-1])
    assert torch.equal(coordinates, repeated_coordinates)
    assert torch.equal(scores, repeated_scores)
    assert torch.equal(gain, original)


def test_issga_excludes_padding_and_handles_budget_above_valid_grid() -> None:
    gain = torch.tensor([[[100.0, 3.0], [2.0, 1.0]]])
    valid = torch.tensor([[[False, True], [True, True]]])
    coordinates, scores = iterative_soft_subtraction(
        gain,
        patch_shape=(1, 1),
        budget=5,
        valid_mask=valid,
    )

    assert [0, 0] not in coordinates[0].tolist()
    assert coordinates.shape == (1, 5, 2)
    assert scores.shape == (1, 5)
    assert torch.isfinite(scores).all()
    assert scores[0, -1].item() == 0.0


def test_sparse_local_tokens_have_bounded_shapes_padding_and_determinism() -> None:
    local_features = torch.arange(1.0, 17.0).reshape(1, 1, 4, 4)
    local_position = local_features + 100.0
    local_padding_mask = torch.zeros((1, 4, 4), dtype=torch.bool)
    local_padding_mask[0, 0, 2] = True
    gain_map = torch.tensor([[[1.0, 9.0], [2.0, 4.0]]])

    first = select_sparse_local_tokens(
        local_features,
        local_position,
        local_padding_mask,
        gain_map,
        patch_fraction=(0.5, 0.5),
        patch_budget=1,
        max_tokens=5,
    )
    second = select_sparse_local_tokens(
        local_features,
        local_position,
        local_padding_mask,
        gain_map,
        patch_fraction=(0.5, 0.5),
        patch_budget=1,
        max_tokens=5,
    )
    features, positions, padding, routing = first

    assert features.shape == (1, 5, 1)
    assert positions.shape == (1, 5, 1)
    assert padding.shape == (1, 5)
    assert routing.shape == (1, 1, 3)
    assert torch.equal(features[0, :, 0], torch.tensor([4.0, 7.0, 8.0, 0.0, 0.0]))
    assert torch.equal(positions[0, :, 0], torch.tensor([104.0, 107.0, 108.0, 0.0, 0.0]))
    assert torch.equal(padding, torch.tensor([[False, False, False, True, True]]))
    assert torch.equal(routing, torch.tensor([[[0.0, 1.0, 9.0]]]))
    for left, right in zip(first, second):
        assert torch.equal(left, right)


def test_sparse_local_tokens_reserve_capacity_for_each_route() -> None:
    local_features = torch.arange(1.0, 33.0).reshape(1, 1, 4, 8)
    local_position = torch.zeros_like(local_features)
    local_padding_mask = torch.zeros((1, 4, 8), dtype=torch.bool)
    gain_map = torch.tensor([[[9.0, 1.0]]])

    features, _, padding, routing = select_sparse_local_tokens(
        local_features,
        local_position,
        local_padding_mask,
        gain_map,
        patch_fraction=(1.0, 0.5),
        patch_budget=2,
        max_tokens=2,
    )

    flattened_indices = features[0, :, 0].to(torch.long) - 1
    columns = flattened_indices.remainder(8)
    assert (columns < 4).any()
    assert (columns >= 4).any()
    assert not padding.any()
    assert routing.shape == (1, 2, 3)


def test_sparse_local_tokens_all_padding_uses_finite_sentinel() -> None:
    features, positions, padding, _ = select_sparse_local_tokens(
        torch.randn(1, 2, 2, 2),
        torch.randn(1, 2, 2, 2),
        torch.ones((1, 2, 2), dtype=torch.bool),
        torch.ones((1, 1, 1)),
        gain_valid_mask=torch.zeros((1, 1, 1), dtype=torch.bool),
        patch_fraction=(1.0, 1.0),
        patch_budget=1,
        max_tokens=2,
    )

    assert not padding[0, 0]
    assert padding[0, 1]
    assert torch.equal(features[0, 0], torch.zeros(2))
    assert torch.equal(positions[0, 0], torch.zeros(2))


@pytest.mark.parametrize("device_name", _DEVICES)
@pytest.mark.parametrize(
    ("patch_fraction", "patch_budget", "max_tokens"),
    [
        ((0.65, 0.55), 3, 7),
        ((0.75, 0.70), 4, 13),
        ((0.55, 0.45), 5, 3),
        ((0.80, 0.80), 4, 70),
    ],
)
def test_batched_sparse_selector_matches_dynamic_reference(
    device_name: str,
    patch_fraction: tuple[float, float],
    patch_budget: int,
    max_tokens: int,
) -> None:
    """Compare quota, ordering, fill and padding against the old selector.

    Seeded continuous Gain values intentionally avoid undefined ``topk`` tie
    ordering; the batch includes random padding, overlapping routes and a fully
    padded sample.  ``max_tokens=70`` also covers capacity above H*W.
    """

    generator = torch.Generator().manual_seed(9100 + patch_budget * 100 + max_tokens)
    batch, channels, height, width = 3, 4, 7, 9
    local_features = torch.randn(batch, channels, height, width, generator=generator)
    local_position = torch.randn(batch, channels, height, width, generator=generator)
    padding = torch.rand(batch, height, width, generator=generator) < 0.22
    padding[0, 0, 0] = False
    padding[1] = True
    padding[1, -1, -1] = False
    padding[2] = True
    gain_map = torch.rand(batch, 4, 5, generator=generator)
    # A tiny unique row-major offset makes exact interpolation ties vanishingly
    # unlikely without materially changing any route score.
    gain_map += torch.arange(20, dtype=gain_map.dtype).reshape(1, 4, 5) * 1e-5
    gain_valid = torch.rand(batch, 4, 5, generator=generator) > 0.15
    gain_valid[2] = False

    device = torch.device(device_name)
    arguments = (
        local_features.to(device),
        local_position.to(device),
        padding.to(device),
        gain_map.to(device),
    )
    keywords = {
        "gain_valid_mask": gain_valid.to(device),
        "patch_fraction": patch_fraction,
        "patch_budget": patch_budget,
        "max_tokens": max_tokens,
    }
    optimized = select_sparse_local_tokens(*arguments, **keywords)
    reference = _dynamic_sparse_token_reference(*arguments, **keywords)

    for optimized_tensor, reference_tensor in zip(optimized, reference):
        assert torch.equal(optimized_tensor, reference_tensor)


def test_uhr_model_criterion_backward_has_finite_new_module_gradients() -> None:
    pytest.importorskip("scipy")
    try:
        import torchvision  # noqa: F401
    except (ImportError, OSError) as error:  # pragma: no cover - environment dependent
        pytest.skip(f"torchvision is unavailable: {error}")

    from rsdet.models.bhcdetr import BHCDetr, BHCDetrConfig
    from rsdet.models.detection_loss import BHCDetrCriterion

    config = BHCDetrConfig(
        image_size=64,
        backbone_pretrained=False,
        hidden_dim=32,
        num_queries=5,
        encoder_layers=1,
        decoder_layers=1,
        nheads=4,
        dim_feedforward=64,
        dropout=0.0,
        projection_dim=16,
        uhr_enabled=True,
        uhr_patch_size=32,
        uhr_patch_budget=2,
        uhr_max_local_tokens=8,
        uhr_gain_head_groups=8,
    )
    model = BHCDetr(config).train()
    criterion = BHCDetrCriterion(
        num_classes=config.num_classes,
        projection_dim=config.projection_dim,
        decoder_layers=config.decoder_layers,
    ).train()
    images = torch.randn(1, 3, 64, 64)
    padding = torch.zeros((1, 64, 64), dtype=torch.bool)
    padding[:, 32:] = True
    targets = [
        {
            # A roughly one-pixel target verifies the float32 Gain target path.
            "boxes": torch.tensor([[0.5, 0.25, 1.0 / 64.0, 1.0 / 64.0]]),
            "labels": torch.tensor([24], dtype=torch.long),
        }
    ]

    outputs = model(images, padding)
    assert (outputs["routing"][..., 0] == 0).all()
    losses = criterion(outputs, targets)
    assert torch.isfinite(losses["loss_total"])
    assert torch.isfinite(losses["loss_gain_map"])
    assert torch.isfinite(losses["loss_gain_lpm"])
    losses["loss_total"].backward()

    assert model.gain_map_head is not None
    gain_gradient = model.gain_map_head.layers[0].weight.grad
    assert gain_gradient is not None and torch.isfinite(gain_gradient).all()
    local_attention = model.decoder.layers[0].classification_branch.local_cross_attn
    assert local_attention is not None
    assert local_attention.in_proj_weight.grad is not None
    assert torch.isfinite(local_attention.in_proj_weight.grad).all()
