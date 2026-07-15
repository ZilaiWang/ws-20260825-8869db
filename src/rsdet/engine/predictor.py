"""轻量、模型无关的批量推理编排。

本模块只负责分批调用模型适配器并校验公共输出，不接管不同框架的预处理、
训练或大图切片逻辑。
"""

from collections.abc import Iterable, Sequence

from rsdet.contracts import InferenceSample, Prediction
from rsdet.models.base import BaseDetector
from rsdet.predictions import validate_prediction


def predict_batches(
    detector: BaseDetector,
    samples: Sequence[InferenceSample],
    *,
    batch_size: int = 1,
    allowed_category_ids: Iterable[int] | None = None,
) -> list[Prediction]:
    """按批调用 detector，并确保每个输入恰好对应一个合法输出。"""
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须 > 0，当前为 {batch_size}")

    allowed = None if allowed_category_ids is None else tuple(allowed_category_ids)
    detector.eval()
    outputs: list[Prediction] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        batch_outputs = list(detector.predict(batch))
        if len(batch_outputs) != len(batch):
            raise ValueError(
                f"模型返回 {len(batch_outputs)} 个预测，但当前 batch 有 {len(batch)} 个输入"
            )
        for sample, prediction in zip(batch, batch_outputs):
            validate_prediction(
                prediction,
                expected_image_id=sample.image_id,
                allowed_category_ids=allowed,
                image_size=(sample.width, sample.height),
            )
            outputs.append(prediction)
    return outputs
