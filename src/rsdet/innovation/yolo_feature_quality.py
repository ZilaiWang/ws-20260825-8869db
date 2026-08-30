"""Detector-feature extraction primitives for the official-match quality head.

Boxes passed to :class:`MultiScaleROIFeatureEncoder` must already be expressed
in the model-input coordinate system (after the same resize/letterbox used by
YOLO).  For the competition's preferred 1024 tile -> 1024 input path this is an
identity transform.  For Normal-CV3 images, record the exact letterbox scale and
padding instead of approximating it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn
from torchvision.ops import roi_align


class YoloPyramidTap:
    """Forward-hook selected Ultralytics DetectionModel layers.

    Example::

        tap = YoloPyramidTap(yolo.model, layer_indices=(15, 18, 21))
        results = yolo.predict(...)
        p3, p4, p5 = tap.features()

    Layer indices are deliberately configuration, not guessed in code.  Audit
    them once for each frozen YOLO architecture and record output shapes.
    """

    def __init__(self, detection_model: nn.Module, layer_indices: Iterable[int]) -> None:
        layers = getattr(detection_model, "model", None)
        if layers is None:
            raise ValueError("Ultralytics DetectionModel lacks .model layers")
        self._layers = layers
        self._indices = tuple(int(value) for value in layer_indices)
        if not self._indices or len(set(self._indices)) != len(self._indices):
            raise ValueError("layer_indices must be a non-empty unique sequence")
        self._latest: dict[int, torch.Tensor] = {}
        self._handles = []
        for index in self._indices:
            if index < 0 or index >= len(layers):
                raise IndexError(f"YOLO feature layer index out of range: {index}")
            self._handles.append(layers[index].register_forward_hook(self._hook(index)))

    def _hook(self, index: int):
        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = self._first_tensor(output)
            if tensor is None:
                raise RuntimeError(f"YOLO layer {index} did not return a tensor")
            self._latest[index] = tensor

        return capture

    @staticmethod
    def _first_tensor(value: Any) -> torch.Tensor | None:
        if torch.is_tensor(value):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                result = YoloPyramidTap._first_tensor(item)
                if result is not None:
                    return result
        if isinstance(value, dict):
            for item in value.values():
                result = YoloPyramidTap._first_tensor(item)
                if result is not None:
                    return result
        return None

    def clear(self) -> None:
        self._latest.clear()

    def features(self, *, detach: bool = False) -> tuple[torch.Tensor, ...]:
        missing = [index for index in self._indices if index not in self._latest]
        if missing:
            raise RuntimeError(f"feature hooks did not fire for layers: {missing}")
        values = tuple(self._latest[index] for index in self._indices)
        return tuple(value.detach() for value in values) if detach else values

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._latest.clear()

    def __enter__(self) -> "YoloPyramidTap":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def expand_xyxy(
    boxes_with_batch: torch.Tensor,
    *,
    ratio: float,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Expand ``[batch,x1,y1,x2,y2]`` boxes around their centers and clip."""

    if boxes_with_batch.ndim != 2 or boxes_with_batch.shape[1] != 5:
        raise ValueError("boxes must have shape [K, 5]")
    if ratio < 1.0:
        raise ValueError("ratio must be >= 1")
    output = boxes_with_batch.clone()
    x1, y1, x2, y2 = (boxes_with_batch[:, index] for index in range(1, 5))
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_width = (x2 - x1) * 0.5 * ratio
    half_height = (y2 - y1) * 0.5 * ratio
    output[:, 1] = (cx - half_width).clamp(0.0, float(image_width))
    output[:, 2] = (cy - half_height).clamp(0.0, float(image_height))
    output[:, 3] = (cx + half_width).clamp(0.0, float(image_width))
    output[:, 4] = (cy + half_height).clamp(0.0, float(image_height))
    return output


class MultiScaleROIFeatureEncoder(nn.Module):
    """Pool core and context-difference evidence from detector FPN features."""

    def __init__(
        self,
        in_channels: Sequence[int],
        strides: Sequence[int],
        *,
        projection_dim: int = 64,
        output_size: int = 3,
        context_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if len(in_channels) != len(strides) or not in_channels:
            raise ValueError("in_channels and strides must be aligned non-empty sequences")
        if any(value <= 0 for value in in_channels) or any(value <= 0 for value in strides):
            raise ValueError("channels and strides must be positive")
        if projection_dim <= 0 or output_size <= 0 or context_ratio < 1.0:
            raise ValueError("invalid ROI encoder dimensions")
        self.strides = tuple(float(value) for value in strides)
        self.output_size = int(output_size)
        self.context_ratio = float(context_ratio)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(channels) * 2, projection_dim),
                    nn.LayerNorm(projection_dim),
                    nn.GELU(),
                )
                for channels in in_channels
            ]
        )
        self.output_dim = len(in_channels) * projection_dim

    def forward(
        self,
        feature_maps: Sequence[torch.Tensor],
        boxes_with_batch: torch.Tensor,
        *,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        if len(feature_maps) != len(self.projections):
            raise ValueError("feature_maps do not match configured levels")
        if boxes_with_batch.ndim != 2 or boxes_with_batch.shape[1] != 5:
            raise ValueError("boxes_with_batch must have shape [K, 5]")
        if boxes_with_batch.numel() == 0:
            return feature_maps[0].new_zeros((0, self.output_dim))
        context_boxes = expand_xyxy(
            boxes_with_batch,
            ratio=self.context_ratio,
            image_height=image_height,
            image_width=image_width,
        )
        outputs = []
        for feature, stride, projection in zip(
            feature_maps, self.strides, self.projections, strict=True
        ):
            core = roi_align(
                feature,
                boxes_with_batch,
                output_size=self.output_size,
                spatial_scale=1.0 / stride,
                sampling_ratio=2,
                aligned=True,
            ).mean(dim=(2, 3))
            context = roi_align(
                feature,
                context_boxes,
                output_size=self.output_size,
                spatial_scale=1.0 / stride,
                sampling_ratio=2,
                aligned=True,
            ).mean(dim=(2, 3))
            # Context-minus-core is a detector-feature analogue of a ring crop,
            # without running another image backbone per proposal.
            outputs.append(projection(torch.cat((core, context - core), dim=1)))
        return torch.cat(outputs, dim=1)


__all__ = [
    "MultiScaleROIFeatureEncoder",
    "YoloPyramidTap",
    "expand_xyxy",
]
