from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rsdet.innovation.yolo_feature_quality import (  # noqa: E402
    MultiScaleROIFeatureEncoder,
    expand_xyxy,
)


def test_expand_boxes_clips_to_model_input() -> None:
    boxes = torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0]])
    expanded = expand_xyxy(boxes, ratio=2.0, image_height=20, image_width=20)
    assert torch.allclose(expanded, torch.tensor([[0.0, 0.0, 0.0, 15.0, 15.0]]))


def test_multiscale_roi_encoder_shape() -> None:
    encoder = MultiScaleROIFeatureEncoder([8, 16], [4, 8], projection_dim=12)
    features = [torch.randn(1, 8, 16, 16), torch.randn(1, 16, 8, 8)]
    boxes = torch.tensor([[0.0, 8.0, 8.0, 40.0, 40.0]])
    output = encoder(features, boxes, image_height=64, image_width=64)
    assert output.shape == (1, 24)
    assert torch.isfinite(output).all()


def test_multiscale_roi_encoder_accepts_feature_geometry_dtype_mismatch() -> None:
    encoder = MultiScaleROIFeatureEncoder([8], [4], projection_dim=6)
    # Float64 exercises the same ROIAlign dtype constraint as CUDA AMP FP16 on CPU,
    # while the adapter intentionally remains in its frozen FP32 contract.
    features = [torch.randn(1, 8, 16, 16, dtype=torch.float64)]
    boxes = torch.tensor([[0.0, 8.0, 8.0, 40.0, 40.0]], dtype=torch.float32)
    output = encoder(features, boxes, image_height=64, image_width=64)
    assert output.shape == (1, 6)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
