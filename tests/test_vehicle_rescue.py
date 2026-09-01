from collections.abc import Sequence

import numpy as np

from rsdet.contracts import InferenceSample, Prediction
from rsdet.models.base import BaseDetector
from rsdet.submission.vehicle_rescue import (
    SelectiveVehicleRescueDetector,
    VehicleRescueConfig,
    apply_vehicle_reject_rescue,
)


class _Detector(BaseDetector):
    def __init__(self, outputs: dict[int, Prediction]) -> None:
        self.outputs = outputs
        self.calls: list[list[int]] = []

    def load(self, checkpoint_path: str) -> None:
        pass

    def predict(self, batch: Sequence[InferenceSample]) -> list[Prediction]:
        self.calls.append([item.image_id for item in batch])
        return [self.outputs[item.image_id] for item in batch]

    def to(self, device: str) -> None:
        pass

    def eval(self) -> None:
        pass


def test_vehicle_rescue_preserves_non_vehicle_and_primary_geometry() -> None:
    primary = Prediction(1, [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], [0.7, 0.1, 0.1], [0, 24, 24])
    specialist = Prediction(1, [[20, 20, 30, 30]], [0.8], [24])
    result = apply_vehicle_reject_rescue(
        primary, specialist,
        config=VehicleRescueConfig(core_threshold=0.2, rescue_product_threshold=0.05, promoted_score=0.200001)
    )
    assert result.boxes_xyxy == [[0, 0, 10, 10], [20, 20, 30, 30]]
    assert result.labels == [0, 24]
    assert result.scores[0] == 0.7


def test_selective_detector_skips_tiles_without_vehicle_tail() -> None:
    samples = [InferenceSample(index, np.zeros((4, 4, 3), dtype=np.uint8), 4, 4) for index in (1, 2)]
    primary = _Detector({
        1: Prediction(1, [[0, 0, 2, 2]], [0.8], [0]),
        2: Prediction(2, [[0, 0, 2, 2]], [0.1], [24]),
    })
    specialist = _Detector({2: Prediction(2, [[0, 0, 2, 2]], [0.9], [24])})
    detector = SelectiveVehicleRescueDetector(
        primary, specialist,
        config=VehicleRescueConfig(core_threshold=0.2, rescue_product_threshold=0.05, promoted_score=0.200001),
    )
    outputs = detector.predict(samples)
    assert specialist.calls == [[2]]
    assert outputs[0] is primary.outputs[1]
    assert outputs[1].scores == [0.200001]
