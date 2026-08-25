"""Mock 检测器：从 InferenceSample.metadata["gt_boxes"] 读取真值返回。

完全替代尚不存在的 M1 adapter，不依赖 GPU / ultralytics / 任何权重文件。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, List

import numpy as np

from rsdet.contracts import InferenceSample, Prediction
from rsdet.models.base import BaseDetector
from rsdet.models.registry import register_model


@register_model("mock")
class MockDetector(BaseDetector):
    """占位检测器：把 sample.metadata["gt_boxes"] 作为预测结果。

    对每个 tile，metadata["gt_boxes"] 应为 list[dict]：
        {"bbox": [x1,y1,x2,y2], "category_id": int, "score": float}

    如果 metadata 中无 gt_boxes，该 tile 返回空 Prediction。

    可选参数：
        noise_std: 坐标噪声标准差（像素），默认 0.0（完美检测）。
        score_offset: 分数减量，默认 0.0（分数不变）。
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        score_offset: float = 0.0,
        **kwargs: Any,
    ):
        self._noise_std = float(noise_std)
        self._score_offset = float(score_offset)
        self._rng: np.random.RandomState | None = None

    def load(self, checkpoint_path: str) -> None:
        pass

    def to(self, device: str) -> None:
        pass

    def eval(self) -> None:
        self._rng = np.random.RandomState(42)

    def predict(self, batch: Sequence[InferenceSample]) -> List[Prediction]:
        if self._rng is None:
            self._rng = np.random.RandomState(42)

        results: List[Prediction] = []
        for sample in batch:
            gt_boxes: list = sample.metadata.get("gt_boxes", [])
            boxes: List[List[float]] = []
            scores: List[float] = []
            labels: List[int] = []

            for entry in gt_boxes:
                box = list(entry["bbox"])  # [x1,y1,x2,y2] 局部坐标
                if self._noise_std > 0.0:
                    box[0] += float(self._rng.normal(0, self._noise_std))
                    box[1] += float(self._rng.normal(0, self._noise_std))
                    box[2] += float(self._rng.normal(0, self._noise_std))
                    box[3] += float(self._rng.normal(0, self._noise_std))
                score = max(0.0, min(1.0, float(entry.get("score", 1.0)) - self._score_offset))
                boxes.append(box)
                scores.append(score)
                labels.append(int(entry["category_id"]))

            results.append(
                Prediction(
                    image_id=sample.image_id,
                    boxes_xyxy=boxes,
                    scores=scores,
                    labels=labels,
                )
            )
        return results
