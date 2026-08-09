"""Restore tile coordinates and remove duplicate cross-tile detections."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from numbers import Integral

from rsdet.contracts import Prediction, TileRecord
from rsdet.data.xh_dataset import coarse_name
from rsdet.postprocess.nms import nms
from rsdet.tiling.coordinates import clip_bbox, tile_to_full


def _grouped_nms(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    iou_threshold: float,
    coarse: bool,
) -> list[int]:
    """Run NMS independently for each fine class or official coarse class."""
    groups: dict[int | str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[coarse_name(label) if coarse else label].append(index)

    keep: list[int] = []
    for indices in groups.values():
        local_keep = nms(
            [boxes[index] for index in indices],
            [scores[index] for index in indices],
            iou_threshold,
        )
        keep.extend(indices[local_index] for local_index in local_keep)
    return sorted(keep, key=lambda index: (-float(scores[index]), index))


def _validated_label(label: object, *, location: str) -> int:
    if isinstance(label, bool) or not isinstance(label, Integral):
        raise ValueError(f"{location} must be an integer category id")
    numeric_label = int(label)
    # coarse_name is the canonical validation for the official 0..24 label space.
    coarse_name(numeric_label)
    return numeric_label


def _validated_score(score: object, *, location: str) -> float:
    try:
        numeric_score = float(score)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{location} must be numeric") from error
    if not math.isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
        raise ValueError(f"{location} must be finite and within [0, 1]")
    return numeric_score


def _validate_tile(tile: TileRecord, *, parent_image_id: int, index: int) -> None:
    if tile.parent_image_id != parent_image_id:
        raise ValueError(
            f"tiles[{index}].parent_image_id={tile.parent_image_id} does not match "
            f"parent_image_id={parent_image_id}"
        )
    if tile.width <= 0 or tile.height <= 0:
        raise ValueError(f"tiles[{index}] dimensions must be > 0")
    if tile.x_offset < 0 or tile.y_offset < 0:
        raise ValueError(f"tiles[{index}] offsets must be >= 0")


def fuse_tile_predictions(
    tile_predictions: Sequence[Prediction],
    tiles: Sequence[TileRecord],
    *,
    parent_image_id: int,
    image_width: int,
    image_height: int,
    fine_nms_iou: float = 0.55,
    coarse_nms_iou: float | None = 0.85,
    max_detections: int | None = None,
) -> Prediction:
    """Fuse predictions from tiles into one prediction for the parent image.

    Tile-local boxes are translated to absolute coordinates and clipped to the
    parent image. Degenerate boxes created by clipping are discarded. NMS first
    removes duplicates within each fine category. When ``coarse_nms_iou`` is not
    ``None``, a second, normally much stricter pass removes near-identical boxes
    assigned to different fine categories within the same official coarse class.
    """
    if len(tile_predictions) != len(tiles):
        raise ValueError("tile_predictions and tiles must have the same length")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("parent image dimensions must be > 0")
    if isinstance(max_detections, bool) or (
        max_detections is not None
        and (not isinstance(max_detections, Integral) or max_detections <= 0)
    ):
        raise ValueError("max_detections must be a positive integer or None")

    # Validate thresholds even when every tile has no detections.
    nms([], [], fine_nms_iou)
    if coarse_nms_iou is not None:
        nms([], [], coarse_nms_iou)

    tile_ids: set[int] = set()
    boxes: list[list[float]] = []
    scores: list[float] = []
    labels: list[int] = []
    for index, (prediction, tile) in enumerate(zip(tile_predictions, tiles)):
        _validate_tile(tile, parent_image_id=parent_image_id, index=index)
        if tile.tile_id in tile_ids:
            raise ValueError(f"duplicate tile_id: {tile.tile_id}")
        tile_ids.add(tile.tile_id)
        if prediction.image_id != tile.tile_id:
            raise ValueError(
                f"tile_predictions[{index}].image_id={prediction.image_id} does not match "
                f"tiles[{index}].tile_id={tile.tile_id}"
            )
        if not (
            len(prediction.boxes_xyxy)
            == len(prediction.scores)
            == len(prediction.labels)
        ):
            raise ValueError(
                f"tile_predictions[{index}] boxes, scores, and labels must have equal lengths"
            )

        for detection_index, (box, score, label) in enumerate(
            zip(prediction.boxes_xyxy, prediction.scores, prediction.labels)
        ):
            location = f"tile_predictions[{index}] detection[{detection_index}]"
            numeric_score = _validated_score(score, location=f"{location}.score")
            numeric_label = _validated_label(label, location=f"{location}.label")
            try:
                restored = tile_to_full(box, tile.x_offset, tile.y_offset)
                clipped = [
                    float(value)
                    for value in clip_bbox(restored, image_width, image_height)
                ]
            except (TypeError, ValueError) as error:
                raise ValueError(f"{location}.box is invalid: {error}") from error
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            boxes.append(clipped)
            scores.append(numeric_score)
            labels.append(numeric_label)

    if not boxes:
        return Prediction(parent_image_id, [], [], [])

    keep = _grouped_nms(
        boxes,
        scores,
        labels,
        iou_threshold=fine_nms_iou,
        coarse=False,
    )
    boxes = [boxes[index] for index in keep]
    scores = [scores[index] for index in keep]
    labels = [labels[index] for index in keep]

    if coarse_nms_iou is not None:
        keep = _grouped_nms(
            boxes,
            scores,
            labels,
            iou_threshold=coarse_nms_iou,
            coarse=True,
        )
        boxes = [boxes[index] for index in keep]
        scores = [scores[index] for index in keep]
        labels = [labels[index] for index in keep]

    if max_detections is not None:
        limit = int(max_detections)
        boxes = boxes[:limit]
        scores = scores[:limit]
        labels = labels[:limit]
    return Prediction(parent_image_id, boxes, scores, labels)


__all__ = ["fuse_tile_predictions"]
