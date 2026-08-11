"""Project inference adapter for the paper-aligned BHC-DETR model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from rsdet.contracts import InferenceSample, Prediction
from rsdet.models.base import BaseDetector
from rsdet.models.bhcdetr import BHCDetr, BHCDetrConfig
from rsdet.models.detection_loss import box_cxcywh_to_xyxy
from rsdet.models.registry import register_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _model_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint must contain a model/state_dict mapping")
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def _letterbox_rgb(
    image: np.ndarray, image_size: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"BHC-DETR input must be uint8 HWC RGB, found {image.shape}/{image.dtype}")
    height, width = image.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, min(image_size, int(round(width * scale))))
    resized_height = max(1, min(image_size, int(round(height * scale))))
    pad_x = (image_size - resized_width) // 2
    pad_y = (image_size - resized_height) // 2
    resized = np.asarray(
        Image.fromarray(image, mode="RGB").resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    array = np.ascontiguousarray(canvas.transpose(2, 0, 1))
    tensor = torch.from_numpy(array).float().div_(255.0)
    mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
    tensor = (tensor - mean) / std
    mask = torch.ones((image_size, image_size), dtype=torch.bool)
    mask[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = False
    return (
        tensor,
        mask,
        {
            "scale_x": resized_width / width,
            "scale_y": resized_height / height,
            "pad_x": float(pad_x),
            "pad_y": float(pad_y),
        },
    )


@register_model("bhcdetr")
class BHCDetrDetector(BaseDetector):
    """Convert BHC-DETR set predictions to the repository ``Prediction`` contract."""

    def __init__(
        self,
        *,
        image_size: int | None = None,
        confidence: float = 0.001,
        max_detections: int = 300,
        half: bool = True,
    ) -> None:
        if image_size is not None and image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if max_detections <= 0:
            raise ValueError("max_detections must be positive")
        self.image_size_override = image_size
        self.confidence = float(confidence)
        self.max_detections = int(max_detections)
        self.half = bool(half)
        self._device = torch.device("cpu")
        self._model: BHCDetr | None = None
        self._config: BHCDetrConfig | None = None

    def load(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("BHC-DETR checkpoint root must be a mapping")
        raw_config = checkpoint.get("model_config")
        if not isinstance(raw_config, Mapping):
            raise ValueError("checkpoint is missing model_config")
        config_values = dict(raw_config)
        config_values["backbone_pretrained"] = False
        config_values["backbone_weights"] = None
        if self.image_size_override is not None:
            config_values["image_size"] = int(self.image_size_override)
        self._config = BHCDetrConfig.from_mapping(config_values)
        if self._config.num_classes != 25:
            raise ValueError("checkpoint must predict the 25 official fine classes")
        self._model = BHCDetr(self._config)
        missing, unexpected = self._model.load_state_dict(_model_state(checkpoint), strict=False)
        if missing or unexpected:
            raise ValueError(
                f"checkpoint architecture mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

    def to(self, device: str) -> None:
        self._device = torch.device(device)
        if self._model is not None:
            self._model.to(self._device)

    def eval(self) -> None:
        if self._model is not None:
            self._model.eval()

    def predict(self, batch: Sequence[InferenceSample]) -> list[Prediction]:
        if self._model is None or self._config is None:
            raise RuntimeError("BHC-DETR has not been loaded")
        if not batch:
            return []
        prepared = [
            _letterbox_rgb(np.asarray(sample.image), self._config.image_size) for sample in batch
        ]
        images = torch.stack([item[0] for item in prepared]).to(self._device, non_blocking=True)
        masks = torch.stack([item[1] for item in prepared]).to(self._device, non_blocking=True)
        autocast_enabled = self.half and self._device.type == "cuda"
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self._device.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ),
        ):
            outputs = self._model(images, masks, return_aux=False)
        probabilities = outputs["pred_logits"].float().sigmoid()
        normalized_boxes = outputs["pred_boxes"].float()
        query_scores, query_labels = probabilities.max(dim=-1)
        square_boxes = box_cxcywh_to_xyxy(normalized_boxes) * self._config.image_size

        predictions: list[Prediction] = []
        for index, sample in enumerate(batch):
            scores = query_scores[index]
            labels = query_labels[index]
            boxes = square_boxes[index]
            keep = torch.nonzero(scores >= self.confidence, as_tuple=False).flatten()
            if keep.numel():
                keep = keep[scores[keep].argsort(descending=True)[: self.max_detections]]
            transform = prepared[index][2]
            if keep.numel():
                selected_boxes = boxes.index_select(0, keep).clone()
                selected_scores = scores.index_select(0, keep)
                selected_labels = labels.index_select(0, keep)
                selected_boxes[:, (0, 2)] = (
                    selected_boxes[:, (0, 2)] - transform["pad_x"]
                ) / transform["scale_x"]
                selected_boxes[:, (1, 3)] = (
                    selected_boxes[:, (1, 3)] - transform["pad_y"]
                ) / transform["scale_y"]
                selected_boxes[:, (0, 2)] = selected_boxes[:, (0, 2)].clamp(
                    0.0, float(sample.width)
                )
                selected_boxes[:, (1, 3)] = selected_boxes[:, (1, 3)].clamp(
                    0.0, float(sample.height)
                )
                valid = (selected_boxes[:, 2] > selected_boxes[:, 0]) & (
                    selected_boxes[:, 3] > selected_boxes[:, 1]
                )
                # Transfer once per tile instead of synchronizing CUDA once
                # for every query via .item()/.tolist().
                packed = torch.cat(
                    (
                        selected_boxes[valid],
                        selected_scores[valid, None],
                        selected_labels[valid, None].to(selected_boxes.dtype),
                    ),
                    dim=1,
                ).cpu()
                rows = packed.tolist()
            else:
                rows = []
            output_boxes = [row[:4] for row in rows]
            output_scores = [float(row[4]) for row in rows]
            output_labels = [int(row[5]) for row in rows]
            predictions.append(
                Prediction(
                    image_id=sample.image_id,
                    boxes_xyxy=output_boxes,
                    scores=output_scores,
                    labels=output_labels,
                )
            )
        return predictions


__all__ = ["BHCDetrDetector"]
