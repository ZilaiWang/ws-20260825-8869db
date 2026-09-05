"""One lossless 90-degree tile view with boxes returned in original pixels."""
from __future__ import annotations

import numpy as np

from rsdet.contracts import InferenceSample, Prediction
from rsdet.models.base import BaseDetector


class Rot90ViewDetector(BaseDetector):
    def __init__(self, detector: BaseDetector):
        self.detector = detector

    def load(self, checkpoint_path):
        self.detector.load(checkpoint_path)

    def to(self, device):
        self.detector.to(device)

    def eval(self):
        self.detector.eval()

    def predict(self, batch):
        rotated = [InferenceSample(
            image_id=s.image_id, image=np.ascontiguousarray(np.rot90(s.image)),
            width=s.height, height=s.width, metadata=dict(s.metadata),
        ) for s in batch]
        predictions = self.detector.predict(rotated)
        if len(predictions) != len(batch):
            raise ValueError("rot90 detector batch coverage mismatch")
        output = []
        for sample, pred in zip(batch, predictions, strict=True):
            if pred.image_id != sample.image_id:
                raise ValueError("rot90 detector changed image id/order")
            boxes = [[sample.width-y2, x1, sample.width-y1, x2]
                     for x1, y1, x2, y2 in pred.boxes_xyxy]
            output.append(Prediction(sample.image_id, boxes, pred.scores, pred.labels))
        return output
