"""Small deterministic checks for BHC-DETR horizontal box math."""

from __future__ import annotations

import pytest

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - machine dependent
    pytest.skip("PyTorch is unavailable", allow_module_level=True)

from rsdet.models.detection_loss import (
    BHCDetrLossConfig,
    HungarianMatcher,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    generalized_box_iou,
)


def test_box_conversion_round_trip() -> None:
    boxes = torch.tensor([[0.5, 0.4, 0.2, 0.6]])
    assert torch.allclose(box_xyxy_to_cxcywh(box_cxcywh_to_xyxy(boxes)), boxes)


def test_generalized_iou_is_one_for_identical_boxes() -> None:
    boxes = torch.tensor([[0.1, 0.2, 0.8, 0.9]])
    assert generalized_box_iou(boxes, boxes).item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable",
            ),
        ),
    ],
)
def test_batched_layer_matching_equals_independent_scipy_assignments(
    device: str,
) -> None:
    scipy_optimize = pytest.importorskip("scipy.optimize")
    generator = torch.Generator().manual_seed(23)
    matcher = HungarianMatcher(BHCDetrLossConfig())
    outputs = []
    for _ in range(3):
        outputs.append(
            {
                "pred_logits": torch.randn(3, 7, 25, generator=generator).to(device),
                "pred_boxes": torch.rand(3, 7, 4, generator=generator).to(device),
            }
        )
    targets = [
        {
            "labels": torch.tensor([2, 7, 19], device=device),
            "boxes": torch.rand(3, 4, generator=generator).to(device),
        },
        {
            "labels": torch.empty(0, dtype=torch.long, device=device),
            "boxes": torch.empty(0, 4, device=device),
        },
        {
            "labels": torch.tensor([1, 24], device=device),
            "boxes": torch.rand(2, 4, generator=generator).to(device),
        },
    ]

    expected = []
    for layer_output in outputs:
        layer_expected = []
        for batch_index, target in enumerate(targets):
            if target["labels"].numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=device)
                layer_expected.append((empty, empty.clone()))
                continue
            cost = matcher._cost_matrix(layer_output, target, batch_index)
            source, target_indices = scipy_optimize.linear_sum_assignment(cost.cpu().numpy())
            layer_expected.append(
                (
                    torch.from_numpy(source).to(device),
                    torch.from_numpy(target_indices).to(device),
                )
            )
        expected.append(layer_expected)

    actual = matcher.match_layers(outputs, targets)

    assert len(actual) == len(expected)
    for actual_layer, expected_layer in zip(actual, expected):
        assert len(actual_layer) == len(expected_layer)
        for actual_pair, expected_pair in zip(actual_layer, expected_layer):
            assert torch.equal(actual_pair[0], expected_pair[0])
            assert torch.equal(actual_pair[1], expected_pair[1])
