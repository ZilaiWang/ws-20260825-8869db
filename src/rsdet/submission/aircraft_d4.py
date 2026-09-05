"""Full-data aircraft-only D4 classifier used after detector fusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rsdet.contracts import Prediction
from rsdet.data.crop_classification import render_crop
from rsdet.features.p04_inputs import D4_VIEW_IDS, apply_d4_view
from rsdet.models.crop_classifier import (
    build_convnext_tiny_architecture,
    build_convnext_tiny_classifier,
)
from rsdet.postprocess.nms import nms

AIRCRAFT_LABELS = frozenset(range(4, 24))


def filter_prediction_by_score(prediction: Prediction, threshold: float) -> Prediction:
    """Return a stable score-filtered copy of a fused prediction."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    keep = [
        index
        for index, score in enumerate(prediction.scores)
        if float(score) >= threshold
    ]
    return Prediction(
        image_id=prediction.image_id,
        boxes_xyxy=[list(prediction.boxes_xyxy[index]) for index in keep],
        scores=[float(prediction.scores[index]) for index in keep],
        labels=[int(prediction.labels[index]) for index in keep],
    )


def _normalize(image: Image.Image) -> Any:
    import torch
    from torchvision.transforms import functional

    tensor = functional.to_tensor(image)
    return functional.normalize(
        tensor,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ).to(dtype=torch.float32)


def _tensorized_d4_views(images: Any) -> Any:
    """Create object-major D4 views without repeating pixel normalization.

    The ordering and transforms exactly match ``D4_VIEW_IDS``: optional
    horizontal flip first, followed by a counter-clockwise rotation.
    """

    import torch

    if images.ndim != 4:
        raise ValueError("D4 base images must have shape [N,C,H,W]")
    flipped = torch.flip(images, dims=(-1,))
    views = (
        images,
        torch.rot90(images, 1, dims=(-2, -1)),
        torch.rot90(images, 2, dims=(-2, -1)),
        torch.rot90(images, 3, dims=(-2, -1)),
        flipped,
        torch.rot90(flipped, 1, dims=(-2, -1)),
        torch.rot90(flipped, 2, dims=(-2, -1)),
        torch.rot90(flipped, 3, dims=(-2, -1)),
    )
    return torch.stack(views, dim=1).flatten(0, 1)


def apply_aircraft_probabilities(
    prediction: Prediction,
    probabilities: Sequence[Sequence[float]],
    *,
    min_probability: float,
    nms_iou: float,
) -> Prediction:
    """Relabel only aircraft proposals, then repeat aircraft same-class NMS."""

    aircraft_indices = [
        index for index, label in enumerate(prediction.labels) if int(label) in AIRCRAFT_LABELS
    ]
    if len(aircraft_indices) != len(probabilities):
        raise ValueError("aircraft probability rows do not match routed proposals")
    labels = [int(value) for value in prediction.labels]
    for index, values in zip(aircraft_indices, probabilities, strict=True):
        row = np.asarray(values, dtype=np.float64)
        if row.shape != (20,) or not np.all(np.isfinite(row)):
            raise ValueError("aircraft probability row must contain 20 finite values")
        if np.any(row < 0.0) or not np.isclose(row.sum(), 1.0, atol=1e-5):
            raise ValueError("aircraft probability row must be normalized")
        best = int(row.argmax())
        if float(row[best]) >= min_probability:
            labels[index] = best + 4

    keep = [index for index, label in enumerate(labels) if label not in AIRCRAFT_LABELS]
    for label in sorted(AIRCRAFT_LABELS):
        indices = [index for index, value in enumerate(labels) if value == label]
        selected = nms(
            [prediction.boxes_xyxy[index] for index in indices],
            [prediction.scores[index] for index in indices],
            nms_iou,
        )
        keep.extend(indices[local_index] for local_index in selected)
    keep.sort(key=lambda index: (-float(prediction.scores[index]), index))
    return Prediction(
        image_id=prediction.image_id,
        boxes_xyxy=[list(prediction.boxes_xyxy[index]) for index in keep],
        scores=[float(prediction.scores[index]) for index in keep],
        labels=[labels[index] for index in keep],
    )


class AircraftD4ClassifierRuntime:
    """Load one audited full checkpoint and relabel fused aircraft proposals."""

    def __init__(self, config: Mapping[str, Any], device: str) -> None:
        import torch

        self.config = dict(config)
        self.device = torch.device(device)
        checkpoint_path = Path(str(self.config["weight_path"]))
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        resolved = payload.get("resolved_config", {})
        expected = {
            "contract_version": "r1_aircraft_view_consistency_full_v1",
            "experiment_id": "R1-5-AIRCRAFT-VIEW-CONSISTENCY-FULL",
            "fold": "full",
            "method": "view_consistency",
            "epochs": 5,
            "checkpoint_selection": "fixed_epoch_last_no_validation",
        }
        mismatches = {
            key: {"actual": resolved.get(key), "expected": value}
            for key, value in expected.items()
            if resolved.get(key) != value
        }
        if mismatches:
            raise ValueError(f"aircraft full classifier contract mismatch: {mismatches}")
        if bool(self.config.get("checkpoint_contains_full_state", False)):
            self.model = build_convnext_tiny_architecture(25, regime="fine_tune")
        else:
            imagenet_path = Path(str(self.config["imagenet_weight_path"]))
            self.model = build_convnext_tiny_classifier(
                25, weight_path=imagenet_path, regime="fine_tune"
            )
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.channels_last = bool(self.config.get("channels_last", False))
        self.tensorized_views = bool(self.config.get("tensorized_views", False))
        self.model.to(self.device)
        if self.channels_last:
            self.model.to(memory_format=torch.channels_last)
        self.model.eval()

    def refine(self, rgb: np.ndarray, prediction: Prediction) -> Prediction:
        import torch

        indices = [
            index
            for index, label in enumerate(prediction.labels)
            if int(label) in AIRCRAFT_LABELS
        ]
        if not indices:
            return prediction
        source = Image.fromarray(rgb, mode="RGB")
        probabilities: list[list[float]] = []
        object_batch = int(self.config.get("batch_objects", 16))
        for start in range(0, len(indices), object_batch):
            batch_indices = indices[start : start + object_batch]
            crops = [
                render_crop(source, prediction.boxes_xyxy[index], 224)
                for index in batch_indices
            ]
            if self.tensorized_views:
                base_images = torch.stack([_normalize(crop) for crop in crops]).to(
                    self.device, non_blocking=True
                )
                images = _tensorized_d4_views(base_images)
            else:
                tensors = [
                    _normalize(apply_d4_view(crop, view))
                    for crop in crops
                    for view in D4_VIEW_IDS
                ]
                images = torch.stack(tensors).to(self.device, non_blocking=True)
            if self.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                logits = self.model(images).reshape(len(batch_indices), len(D4_VIEW_IDS), 25)
            values = logits.float()[:, :, 4:24].softmax(dim=2).mean(dim=1)
            probabilities.extend(values.cpu().tolist())
        return apply_aircraft_probabilities(
            prediction,
            probabilities,
            min_probability=float(self.config.get("relabel_min_probability", 0.9)),
            nms_iou=float(self.config.get("nms_iou", 0.5)),
        )


__all__ = [
    "AIRCRAFT_LABELS",
    "AircraftD4ClassifierRuntime",
    "_tensorized_d4_views",
    "apply_aircraft_probabilities",
    "filter_prediction_by_score",
]
