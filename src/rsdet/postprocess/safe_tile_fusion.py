"""Conservative, class-consistent fusion for overlapping image tiles.

The legacy global aggregator deliberately performs cross-tile class voting.  That
is useful for object-level evidence studies, but it is unsafe as a submission
default because a class label can inherit another proposal's box and score.  This
module implements a smaller deployment contract:

* proposals below the frozen operating point are removed before fusion;
* only detections from different tiles with the same fine class may merge;
* clustering is anchor-greedy and therefore non-transitive;
* the output label, score and box always come from one real proposal;
* a proposal away from an internal tile edge is preferred as the canonical box.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from rsdet.contracts import Prediction, TileRecord
from rsdet.data.xh_dataset import coarse_name
from rsdet.postprocess.nms import nms
from rsdet.tiling.coordinates import clip_bbox, tile_to_full


@dataclass(frozen=True)
class _Candidate:
    box: tuple[float, float, float, float]
    score: float
    label: int
    tile_id: int
    touches_internal_border: bool


def _area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def _overlap(first: Sequence[float], second: Sequence[float]) -> tuple[float, float]:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = _area(first)
    second_area = _area(second)
    union = first_area + second_area - intersection
    iou = intersection / union if union > 0.0 else 0.0
    smaller = min(first_area, second_area)
    ios = intersection / smaller if smaller > 0.0 else 0.0
    return iou, ios


def _touches_internal_border(
    local_box: Sequence[float],
    tile: TileRecord,
    *,
    image_width: int,
    image_height: int,
    margin: float,
) -> bool:
    left = tile.x_offset > 0 and float(local_box[0]) <= margin
    top = tile.y_offset > 0 and float(local_box[1]) <= margin
    right_is_internal = tile.x_offset + tile.width < image_width
    bottom_is_internal = tile.y_offset + tile.height < image_height
    right = right_is_internal and float(local_box[2]) >= float(tile.width) - margin
    bottom = bottom_is_internal and float(local_box[3]) >= float(tile.height) - margin
    return left or top or right or bottom


def _validate_threshold(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return numeric


def _restore_candidates(
    tile_predictions: Sequence[Prediction],
    tiles: Sequence[TileRecord],
    *,
    parent_image_id: int,
    image_width: int,
    image_height: int,
    score_threshold: float,
    border_margin: float,
) -> list[_Candidate]:
    if len(tile_predictions) != len(tiles):
        raise ValueError("tile_predictions and tiles must have the same length")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("parent image dimensions must be > 0")
    if border_margin < 0.0 or not math.isfinite(border_margin):
        raise ValueError("border_margin must be finite and >= 0")

    candidates: list[_Candidate] = []
    seen_tiles: set[int] = set()
    for index, (prediction, tile) in enumerate(zip(tile_predictions, tiles)):
        if tile.parent_image_id != parent_image_id:
            raise ValueError(f"tiles[{index}] parent_image_id mismatch")
        if tile.tile_id in seen_tiles:
            raise ValueError(f"duplicate tile_id: {tile.tile_id}")
        seen_tiles.add(tile.tile_id)
        if prediction.image_id != tile.tile_id:
            raise ValueError(f"tile_predictions[{index}] image_id mismatch")
        if not (len(prediction.boxes_xyxy) == len(prediction.scores) == len(prediction.labels)):
            raise ValueError(f"tile_predictions[{index}] arrays must have equal lengths")

        for detection_index, (box, score, label) in enumerate(
            zip(prediction.boxes_xyxy, prediction.scores, prediction.labels)
        ):
            numeric_score = _validate_threshold(
                float(score), f"tile_predictions[{index}][{detection_index}].score"
            )
            if numeric_score < score_threshold:
                continue
            if isinstance(label, bool) or not isinstance(label, Integral):
                raise ValueError("fine category id must be an integer")
            numeric_label = int(label)
            coarse_name(numeric_label)
            if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
                raise ValueError("box must contain four finite values")
            restored = tile_to_full(box, tile.x_offset, tile.y_offset)
            clipped = tuple(
                float(value) for value in clip_bbox(restored, image_width, image_height)
            )
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            candidates.append(
                _Candidate(
                    box=clipped,
                    score=numeric_score,
                    label=numeric_label,
                    tile_id=int(tile.tile_id),
                    touches_internal_border=_touches_internal_border(
                        box,
                        tile,
                        image_width=image_width,
                        image_height=image_height,
                        margin=border_margin,
                    ),
                )
            )
    return candidates


def _candidate_sort_key(candidate: _Candidate) -> tuple[float, int, tuple[float, ...], int]:
    return (-candidate.score, candidate.label, candidate.box, candidate.tile_id)


def _canonical(candidates: Sequence[_Candidate]) -> _Candidate:
    # Completeness is more informative than a small score difference for objects
    # cut by an internal tile boundary.  Every returned field still comes from
    # this one selected proposal.
    return max(
        candidates,
        key=lambda item: (
            not item.touches_internal_border,
            item.score,
            _area(item.box),
            -item.tile_id,
        ),
    )


def _grid_cells(
    box: Sequence[float], cell_size: float = 256.0
) -> tuple[tuple[int, int], ...]:
    """Return every half-open spatial cell touched by a positive-area box.

    ``nextafter`` keeps a box whose right/bottom edge lies exactly on a cell
    boundary out of the adjacent cell.  Two boxes with positive intersection
    must therefore share at least one returned cell; merely touching edges is
    not treated as overlap.
    """

    x1, y1, x2, y2 = (float(value) for value in box)
    right = math.nextafter(x2, -math.inf)
    bottom = math.nextafter(y2, -math.inf)
    column_start = math.floor(x1 / cell_size)
    column_end = math.floor(right / cell_size)
    row_start = math.floor(y1 / cell_size)
    row_end = math.floor(bottom / cell_size)
    return tuple(
        (column, row)
        for row in range(row_start, row_end + 1)
        for column in range(column_start, column_end + 1)
    )


def _anchor_greedy_canonical(
    candidates: Sequence[_Candidate], merge_iou: float, merge_ios: float
) -> list[_Candidate]:
    """Run the frozen anchor-greedy contract with an exact spatial index.

    Fine labels are independent under the safe-fusion contract.  Indexing each
    label separately avoids comparing every RT-DETR proposal with proposals of
    the other 24 classes.  The grid only removes pairs that have zero positive
    intersection, so the selected clusters are identical to the former
    all-pairs implementation for positive merge thresholds.
    """

    by_label: dict[int, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_label[candidate.label].append(candidate)

    canonical: list[_Candidate] = []
    for label in sorted(by_label):
        ordered = sorted(by_label[label], key=_candidate_sort_key)
        assigned = [False] * len(ordered)
        spatial: dict[tuple[int, int], list[int]] = defaultdict(list)
        if merge_iou > 0.0 and merge_ios > 0.0:
            for index, candidate in enumerate(ordered):
                for cell in _grid_cells(candidate.box):
                    spatial[cell].append(index)

        for anchor_index, anchor in enumerate(ordered):
            if assigned[anchor_index]:
                continue
            assigned[anchor_index] = True
            cluster = [anchor]
            if merge_iou == 0.0 or merge_ios == 0.0:
                possible = range(anchor_index + 1, len(ordered))
            else:
                possible = sorted(
                    {
                        index
                        for cell in _grid_cells(anchor.box)
                        for index in spatial.get(cell, ())
                        if index > anchor_index
                    }
                )
            for candidate_index in possible:
                if assigned[candidate_index]:
                    continue
                candidate = ordered[candidate_index]
                if candidate.tile_id == anchor.tile_id:
                    continue
                iou, ios = _overlap(anchor.box, candidate.box)
                if iou >= merge_iou or ios >= merge_ios:
                    assigned[candidate_index] = True
                    cluster.append(candidate)
            canonical.append(_canonical(cluster))
    return sorted(canonical, key=_candidate_sort_key)


def _fine_nms(candidates: Sequence[_Candidate], iou_threshold: float) -> list[_Candidate]:
    groups: dict[int, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.label].append(candidate)
    kept: list[_Candidate] = []
    for group in groups.values():
        ordered = sorted(group, key=_candidate_sort_key)
        indices = nms(
            [list(item.box) for item in ordered],
            [item.score for item in ordered],
            iou_threshold,
        )
        kept.extend(ordered[index] for index in indices)
    return sorted(kept, key=_candidate_sort_key)


def fuse_safe_tile_predictions(
    tile_predictions: Sequence[Prediction],
    tiles: Sequence[TileRecord],
    *,
    parent_image_id: int,
    image_width: int,
    image_height: int,
    score_threshold: float = 0.0,
    merge_iou: float = 0.50,
    merge_ios: float = 0.75,
    fine_nms_iou: float = 0.70,
    border_margin: float = 8.0,
    max_detections: int | None = None,
) -> Prediction:
    """Fuse duplicates without changing fine classes or propagating clusters."""
    score_threshold = _validate_threshold(score_threshold, "score_threshold")
    merge_iou = _validate_threshold(merge_iou, "merge_iou")
    merge_ios = _validate_threshold(merge_ios, "merge_ios")
    _validate_threshold(fine_nms_iou, "fine_nms_iou")
    if isinstance(max_detections, bool) or (
        max_detections is not None
        and (not isinstance(max_detections, Integral) or max_detections <= 0)
    ):
        raise ValueError("max_detections must be a positive integer or None")

    candidates = _restore_candidates(
        tile_predictions,
        tiles,
        parent_image_id=parent_image_id,
        image_width=image_width,
        image_height=image_height,
        score_threshold=score_threshold,
        border_margin=border_margin,
    )
    canonical = _anchor_greedy_canonical(candidates, merge_iou, merge_ios)

    canonical = _fine_nms(canonical, fine_nms_iou)
    if max_detections is not None:
        canonical = canonical[: int(max_detections)]
    return Prediction(
        parent_image_id,
        [list(item.box) for item in canonical],
        [item.score for item in canonical],
        [item.label for item in canonical],
    )


__all__ = ["fuse_safe_tile_predictions"]
