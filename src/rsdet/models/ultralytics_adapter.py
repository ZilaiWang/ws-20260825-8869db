"""Ultralytics YOLO/RT-DETR 的统一推理适配器。

依赖采用延迟导入，因此没有安装 PyTorch/Ultralytics 时，数据、评测和单元测试
仍可正常使用。只有真正构建或加载该检测器时才要求模型环境。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rsdet.contracts import InferenceSample, Prediction
from rsdet.data.xh_dataset import coarse_name
from rsdet.models.base import BaseDetector
from rsdet.models.registry import register_model


def create_ultralytics_model(family: str, weights: str | Path):
    """按模型族创建 Ultralytics 模型，集中处理可选依赖错误。"""
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError as error:
        raise ImportError('Ultralytics 模型依赖未安装。请执行 pip install -e ".[model]"') from error

    normalized = family.strip().lower().replace("-", "")
    if normalized == "yolo":
        return YOLO(str(weights))
    if normalized in {"rtdetr", "detr"}:
        return RTDETR(str(weights))
    raise ValueError(f"不支持的 Ultralytics 模型族: {family!r}")


@register_model("ultralytics")
class UltralyticsDetector(BaseDetector):
    """把 Ultralytics 结果转换为项目统一 ``Prediction``。"""

    def __init__(
        self,
        *,
        family: str = "yolo",
        imgsz: int = 1024,
        confidence: float = 0.05,
        iou: float = 0.6,
        max_detections: int = 500,
        half: bool = True,
        agnostic_nms: bool = False,
        refiner: Mapping[str, Any] | None = None,
    ) -> None:
        if imgsz <= 0:
            raise ValueError("imgsz 必须 > 0")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence 必须在 [0, 1]")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou 必须在 [0, 1]")
        if max_detections <= 0:
            raise ValueError("max_detections 必须 > 0")
        self.family = family
        self.imgsz = int(imgsz)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.max_detections = int(max_detections)
        self.half = bool(half)
        self.agnostic_nms = bool(agnostic_nms)
        self.refiner_config = dict(refiner or {})
        self._device = "cpu"
        self._model: Any | None = None
        self._refiner: Any | None = None
        self._refiner_metadata: dict[str, Any] = {}

    def load(self, checkpoint_path: str) -> None:
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"模型权重不存在: {checkpoint}")
        self._model = create_ultralytics_model(self.family, checkpoint)
        if self.refiner_config:
            from rsdet.models.prototype_refiner import load_refiner_checkpoint

            refiner_checkpoint = str(self.refiner_config.get("checkpoint", "")).strip()
            if not refiner_checkpoint:
                raise ValueError("model.refiner 已启用但 checkpoint 为空")
            self._refiner, self._refiner_metadata = load_refiner_checkpoint(
                refiner_checkpoint, map_location="cpu"
            )

    def to(self, device: str) -> None:
        self._device = str(device)
        if self._model is not None and hasattr(self._model, "to"):
            self._model.to(self._device)
        if self._refiner is not None:
            self._refiner.to(self._device)

    def eval(self) -> None:
        if self._model is None:
            return
        torch_model = getattr(self._model, "model", None)
        if torch_model is not None and hasattr(torch_model, "eval"):
            torch_model.eval()
        if self._refiner is not None:
            self._refiner.eval()

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    @staticmethod
    def _prepare_source(image: Any) -> Any:
        """Convert the project's RGB ndarray contract to Ultralytics BGR input."""
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Ultralytics 输入必须是 HWC RGB: {image.shape}")
            return np.ascontiguousarray(image[..., ::-1])
        return image

    def predict(self, batch: Sequence[InferenceSample]) -> list[Prediction]:
        if self._model is None:
            raise RuntimeError("模型尚未 load()")
        if not batch:
            return []

        results = self._model.predict(
            source=[self._prepare_source(sample.image) for sample in batch],
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            max_det=self.max_detections,
            device=self._device,
            quantize=16 if self.half and self._device != "cpu" else None,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
            stream=False,
        )
        results = list(results)
        if len(results) != len(batch):
            raise ValueError(f"Ultralytics 返回 {len(results)} 个结果，输入 batch 为 {len(batch)}")

        predictions: list[Prediction] = []
        for sample, result in zip(batch, results):
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                predictions.append(Prediction(sample.image_id, [], [], []))
                continue
            raw_boxes = [
                [float(coordinate) for coordinate in box] for box in self._as_list(boxes.xyxy)
            ]
            raw_scores = [float(score) for score in self._as_list(boxes.conf)]
            raw_labels = [int(label) for label in self._as_list(boxes.cls)]
            valid = [
                index
                for index, box in enumerate(raw_boxes)
                if len(box) == 4
                and all(math.isfinite(value) for value in box)
                and box[2] > box[0]
                and box[3] > box[1]
            ]
            predictions.append(
                Prediction(
                    image_id=sample.image_id,
                    boxes_xyxy=[raw_boxes[index] for index in valid],
                    scores=[raw_scores[index] for index in valid],
                    labels=[raw_labels[index] for index in valid],
                )
            )
        return self._refine_predictions(batch, predictions)

    def _refine_predictions(
        self,
        batch: Sequence[InferenceSample],
        predictions: list[Prediction],
    ) -> list[Prediction]:
        """仅对不确定的舰船/飞机候选启动 HPR，避免固定增加全部候选时延。"""
        if self._refiner is None:
            return predictions
        import torch

        from rsdet.data.object_crops import (
            as_rgb_image,
            crop_and_resize,
            crop_to_tensor,
        )

        enabled_coarse = set(self.refiner_config.get("coarse_classes", ["ship", "aircraft"]))
        min_confidence = float(self.refiner_config.get("min_base_confidence", 0.0))
        max_confidence = float(self.refiner_config.get("max_base_confidence", 0.75))
        prototype_weight = float(self.refiner_config.get("prototype_weight", 0.35))
        score_blend = float(self.refiner_config.get("score_blend", 0.0))
        min_refined_confidence = float(self.refiner_config.get("min_refined_confidence", 0.0))
        min_refined_margin = float(self.refiner_config.get("min_refined_margin", 0.0))
        refiner_batch_size = int(self.refiner_config.get("batch_size", 128))
        input_size = int(
            self.refiner_config.get("input_size", self._refiner_metadata.get("input_size", 128))
        )
        context_ratio = float(
            self.refiner_config.get(
                "context_ratio", self._refiner_metadata.get("context_ratio", 0.15)
            )
        )
        if not 0.0 <= min_confidence <= max_confidence <= 1.0:
            raise ValueError(
                "refiner 置信度范围必须满足 0 <= min_base_confidence <= max_base_confidence <= 1"
            )
        if not 0.0 <= score_blend <= 1.0:
            raise ValueError("refiner.score_blend 必须在 [0, 1]")
        if not 0.0 <= min_refined_confidence <= 1.0:
            raise ValueError("refiner.min_refined_confidence 必须在 [0, 1]")
        if not 0.0 <= min_refined_margin <= 1.0:
            raise ValueError("refiner.min_refined_margin 必须在 [0, 1]")
        if refiner_batch_size <= 0:
            raise ValueError("refiner.batch_size 必须 > 0")

        refined = [
            Prediction(
                prediction.image_id,
                [list(box) for box in prediction.boxes_xyxy],
                list(prediction.scores),
                list(prediction.labels),
            )
            for prediction in predictions
        ]
        candidates: list[tuple[int, int, str]] = []
        crops = []
        for prediction_index, (sample, prediction) in enumerate(zip(batch, predictions)):
            source_image = as_rgb_image(sample.image)
            for detection_index, (box, score, label) in enumerate(
                zip(prediction.boxes_xyxy, prediction.scores, prediction.labels)
            ):
                group = coarse_name(int(label))
                if (
                    group not in enabled_coarse
                    or float(score) < min_confidence
                    or float(score) > max_confidence
                ):
                    continue
                crop = crop_and_resize(
                    source_image,
                    box,
                    output_size=input_size,
                    context_ratio=context_ratio,
                )
                crops.append(crop_to_tensor(crop))
                candidates.append((prediction_index, detection_index, group))
        if not crops:
            return refined

        device = next(self._refiner.parameters()).device
        for start in range(0, len(crops), refiner_batch_size):
            crop_batch = torch.stack(crops[start : start + refiner_batch_size]).to(device)
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=self.half and device.type == "cuda",
                ),
            ):
                outputs = self._refiner(crop_batch)
                logits = self._refiner.fused_logits(outputs, prototype_weight=prototype_weight)
            for local_index, row in enumerate(logits.float().cpu()):
                prediction_index, detection_index, group = candidates[start + local_index]
                if group == "ship":
                    allowed = list(range(0, 4))
                elif group == "aircraft":
                    allowed = list(range(4, 24))
                else:
                    allowed = [24]
                probabilities = row[allowed].softmax(dim=0)
                best_local = int(probabilities.argmax().item())
                refined_label = allowed[best_local]
                refined_confidence = float(probabilities[best_local].item())
                base_label = int(refined[prediction_index].labels[detection_index])
                base_local = allowed.index(base_label)
                base_probability = float(probabilities[base_local].item())
                if refined_label != base_label and (
                    refined_confidence < min_refined_confidence
                    or refined_confidence - base_probability < min_refined_margin
                ):
                    continue
                base_score = float(refined[prediction_index].scores[detection_index])
                refined[prediction_index].labels[detection_index] = refined_label
                if score_blend > 0.0:
                    refined[prediction_index].scores[detection_index] = (
                        base_score ** (1.0 - score_blend)
                        * max(refined_confidence, 1e-6) ** score_blend
                    )
        return refined


__all__ = ["UltralyticsDetector", "create_ultralytics_model"]
