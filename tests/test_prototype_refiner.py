"""HPR 模型前向、损失反向和检测融合测试。"""

import numpy as np
import pytest

from rsdet.contracts import InferenceSample, Prediction
from rsdet.models.prototype_refiner import (
    HierarchicalPrototypeLoss,
    HierarchicalPrototypeRefiner,
)
from rsdet.models.ultralytics_adapter import UltralyticsDetector

torch = pytest.importorskip("torch")


CLASS_COUNTS = [
    17,
    30,
    641,
    1994,
    1317,
    1297,
    998,
    500,
    1017,
    361,
    547,
    750,
    895,
    762,
    432,
    583,
    1265,
    1424,
    493,
    2147,
    1114,
    262,
    933,
    752,
    402,
]


def test_refiner_forward_backward_and_prototype_update() -> None:
    model = HierarchicalPrototypeRefiner(embedding_dim=64)
    criterion = HierarchicalPrototypeLoss(CLASS_COUNTS)
    images = torch.randn(6, 3, 64, 64)
    labels = torch.tensor([0, 0, 4, 4, 21, 24])

    outputs = model(images)
    losses = criterion(outputs, labels)
    losses["loss"].backward()
    model.update_prototypes(outputs["embeddings"], labels)

    assert outputs["fine_logits"].shape == (6, 25)
    assert outputs["coarse_logits"].shape == (6, 3)
    assert torch.isfinite(losses["loss"])
    assert int((model.prototype_counts > 0).sum()) == 4


class _FakeRefiner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, images):
        logits = torch.zeros(len(images), 25, device=images.device)
        logits[:, 5] = 10.0
        return {"fine_logits": logits}

    def fused_logits(self, outputs, *, prototype_weight=0.35):
        return outputs["fine_logits"]


def test_adapter_only_refines_uncertain_same_coarse_candidate() -> None:
    detector = UltralyticsDetector(
        refiner={
            "coarse_classes": ["aircraft"],
            "max_base_confidence": 0.75,
            "score_blend": 0.0,
        }
    )
    detector._refiner = _FakeRefiner()
    sample = InferenceSample(
        image_id=1,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        width=64,
        height=64,
    )
    prediction = Prediction(
        image_id=1,
        boxes_xyxy=[[10, 10, 40, 40], [20, 20, 50, 50]],
        scores=[0.5, 0.9],
        labels=[4, 4],
    )

    refined = detector._refine_predictions([sample], [prediction])[0]

    assert refined.labels == [5, 4]
    assert refined.scores == [0.5, 0.9]


def test_adapter_preserves_base_label_when_refiner_gate_rejects_change() -> None:
    detector = UltralyticsDetector(
        refiner={
            "coarse_classes": ["aircraft"],
            "max_base_confidence": 0.75,
            "min_refined_confidence": 0.9999,
            "min_refined_margin": 0.5,
        }
    )
    detector._refiner = _FakeRefiner()
    sample = InferenceSample(
        image_id=1,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        width=64,
        height=64,
    )
    prediction = Prediction(1, [[10, 10, 40, 40]], [0.5], [4])

    refined = detector._refine_predictions([sample], [prediction])[0]

    assert refined.labels == [4]
