"""M1 (YOLO26-s/1024) 接入封装。

C 交付 adapter 后，只需确认注册名和权重路径，其余不用改。

用法（C 交付权重后）：
    from rsdet.pipeline.m1_wrapper import M1Wrapper

    detector = M1Wrapper(weights_path="path/to/m1_best.pt")
    detector.load("path/to/m1_best.pt")
    detector.to("cuda")
    detector.eval()

然后直接传入 run_pipeline():
    from rsdet.pipeline.large_image import run_pipeline, PipelineConfig
    pred, timing = run_pipeline(image, detector, config=PipelineConfig(...))
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import List

import numpy as np

from rsdet.contracts import InferenceSample, Prediction
from rsdet.models.base import BaseDetector
from rsdet.models.registry import register_model


@register_model("m1_yolo26s")
class M1Wrapper(BaseDetector):
    """封装 Ultralytics YOLO26-s 推理，符合 BaseDetector 接口。

    Args:
        weights_path: M1 权重文件路径 (.pt)。
        imgsz: 推理分辨率，默认 1024。
        conf: 最低置信度阈值（推理阶段），默认 0.001。
        iou: NMS IoU 阈值（Ultralytics 内部），默认 0.7。
        max_det: 每张图最大检测数，默认 500。
    """

    def __init__(
        self,
        weights_path: str = "",
        imgsz: int = 1024,
        conf: float = 0.001,
        iou: float = 0.7,
        max_det: int = 500,
        **kwargs,
    ):
        self._weights_path = weights_path
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou
        self._max_det = max_det
        self._model = None
        self._device = "cpu"

    def load(self, checkpoint_path: str = "") -> None:
        """加载 YOLO 权重。"""
        from ultralytics import YOLO

        path = checkpoint_path or self._weights_path
        if not path:
            raise ValueError("M1Wrapper: 需要 weights_path 或 checkpoint_path")
        self._model = YOLO(path)

    def to(self, device: str) -> None:
        """将模型移至指定设备。"""
        self._device = device
        if self._model is not None:
            self._model.to(device)

    def eval(self) -> None:
        """Ultralytics 在 predict() 内自动设置 eval 模式，此处无操作。"""
        pass

    def predict(self, batch: Sequence[InferenceSample]) -> List[Prediction]:
        """对一批 tile 执行 YOLO 推理。

        每个 InferenceSample.image 预期为 numpy HWC RGB uint8。
        Ultralytics 原生支持该格式，无需额外转换。
        """
        if self._model is None:
            raise RuntimeError("M1Wrapper: 请先调用 load() 加载权重")

        results: List[Prediction] = []
        for sample in batch:
            img = sample.image
            if not isinstance(img, np.ndarray):
                img = np.array(img, dtype=np.uint8)

            out = self._model.predict(
                img,
                imgsz=self._imgsz,
                conf=self._conf,
                iou=self._iou,
                max_det=self._max_det,
                device=self._device,
                verbose=False,
            )[0]

            if out.boxes is not None and len(out.boxes) > 0:
                boxes_xyxy = out.boxes.xyxy.cpu().numpy().tolist()
                scores = out.boxes.conf.cpu().numpy().tolist()
                labels = out.boxes.cls.cpu().numpy().astype(int).tolist()
            else:
                boxes_xyxy, scores, labels = [], [], []

            results.append(
                Prediction(
                    image_id=sample.image_id,
                    boxes_xyxy=boxes_xyxy,
                    scores=scores,
                    labels=labels,
                )
            )
        return results
