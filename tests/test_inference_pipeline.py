"""统一滑窗推理核心测试。"""

import numpy as np

from rsdet.engine.inference import predict_image
from rsdet.evaluation.runtime import RuntimeBreakdown
from rsdet.models.registry import build_model


def test_dummy_detector_runs_through_tiled_pipeline() -> None:
    detector = build_model("dummy", {})
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    runtime = RuntimeBreakdown()

    prediction = predict_image(
        detector,
        image_id=23,
        image=image,
        batch_size=2,
        tiling_config={"enabled": True, "tile_size": 600, "overlap": 100},
        runtime=runtime,
    )

    assert prediction.image_id == 23
    assert prediction.boxes_xyxy == []
    assert runtime.tiling >= 0.0
    assert runtime.model >= 0.0
