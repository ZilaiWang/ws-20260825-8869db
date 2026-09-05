"""Conservative, class-consistent fusion for overlapping image tiles.

The legacy global aggregator deliberately performs cross-tile class voting.  That
is useful for object-level evidence studies, but it is unsafe as a submission
default because a class label can inherit another proposal's box and score.  This
module implements a smaller deployment contract:

* proposals below the candidate floor are removed before fusion;
* only detections from different tiles with the same fine class may merge;
* clustering is anchor-greedy and therefore non-transitive;
* the output label, score and box always come from one real proposal;
* a proposal away from an internal tile edge is preferred as the canonical box.

The optional BATIS contract separates the candidate floor from the final
operating point.  Low-score members may support clustering and audits, but can
never replace a member that is eligible for final output.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral

from rsdet.contracts import Prediction, TileRecord
from rsdet.postprocess.nms import nms
from rsdet.postprocess.thresholds import effective_threshold, normalize_fine_thresholds
from rsdet.tiling.boundary_geometry import locate_owner_tile, tile_owner_lookup
from rsdet.tiling.coordinates import clip_bbox, tile_to_full


@dataclass(frozen=True)
class _Candidate:
    box: tuple[float, float, float, float]
    score: float
    label: int
    tile_id: int
    touches_internal_border: bool
    local_box: tuple[float, float, float, float]
    tile_width: int
    tile_height: int
    internal_edges: tuple[bool, bool, bool, bool]
    owns_center: bool


@dataclass(frozen=True)
class SafeFusionAudit:
    """Mechanism counts emitted by threshold-safe BATIS replay."""

    candidate_count: int
    cluster_count: int
    canonical_count_before_nms: int
    output_count: int
    threshold_inversion_count: int
    threshold_inversion_applied_count: int
    threshold_safe_cluster_count: int
    legacy_score_drop_cluster_count: int
    owner_selected_count: int
    no_output_eligible_member_count: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


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
    score_threshold_by_coarse: Mapping[str, float] | None,
    border_margin: float,
    score_threshold_by_fine: Mapping[int, float] | None = None,
) -> list[_Candidate]:
    if len(tile_predictions) != len(tiles):
        raise ValueError("tile_predictions and tiles must have the same length")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("parent image dimensions must be > 0")
    if border_margin < 0.0 or not math.isfinite(border_margin):
        raise ValueError("border_margin must be finite and >= 0")

    x_cores, y_cores, owner_lookup = tile_owner_lookup(
        tiles,
        image_width=image_width,
        image_height=image_height,
    )
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
            if isinstance(label, bool) or not isinstance(label, Integral):
                raise ValueError("fine category id must be an integer")
            numeric_label = int(label)
            threshold = effective_threshold(
                numeric_label,
                global_threshold=score_threshold,
                coarse_thresholds=score_threshold_by_coarse,
                fine_thresholds=score_threshold_by_fine,
            )
            if numeric_score < threshold:
                continue
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
                    local_box=tuple(float(value) for value in box),
                    tile_width=int(tile.width),
                    tile_height=int(tile.height),
                    internal_edges=(
                        tile.x_offset > 0,
                        tile.y_offset > 0,
                        tile.x_offset + tile.width < image_width,
                        tile.y_offset + tile.height < image_height,
                    ),
                    owns_center=locate_owner_tile(
                        (clipped[0] + clipped[2]) / 2.0,
                        (clipped[1] + clipped[3]) / 2.0,
                        x_cores=x_cores,
                        y_cores=y_cores,
                        lookup=owner_lookup,
                    )
                    == int(tile.tile_id),
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


def _logit(value: float) -> float:
    clipped = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def _context_margin(candidate: _Candidate) -> float:
    """Return context margin relative to object size on avoidable tile edges."""
    x1, y1, x2, y2 = candidate.local_box
    left, top, right, bottom = candidate.internal_edges
    margins = []
    if left:
        margins.append(x1)
    if top:
        margins.append(y1)
    if right:
        margins.append(float(candidate.tile_width) - x2)
    if bottom:
        margins.append(float(candidate.tile_height) - y2)
    if not margins:
        return math.inf
    return min(margins) / max(x2 - x1, y2 - y1, 1e-9)


def _threshold_safe_canonical(
    candidates: Sequence[_Candidate],
    *,
    output_score_threshold: float,
    owner_logit_slack: float | None,
) -> _Candidate | None:
    """Choose a real proposal without allowing support-only score inversion."""
    eligible = [item for item in candidates if item.score >= output_score_threshold]
    if not eligible:
        return None
    if owner_logit_slack is None:
        # H1 changes only the permission boundary; legacy completeness ordering
        # remains intact among proposals that can actually reach final output.
        return _canonical(eligible)
    best_score = max(item.score for item in eligible)
    near_best = [
        item
        for item in eligible
        if _logit(item.score) >= _logit(best_score) - owner_logit_slack
    ]
    owners = [item for item in near_best if item.owns_center]
    pool = owners if owners else near_best
    return max(
        pool,
        key=lambda item: (
            _context_margin(item),
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
    candidates: Sequence[_Candidate],
    merge_iou: float,
    merge_ios: float,
    *,
    output_score_threshold: float | None = None,
    owner_logit_slack: float | None = None,
    threshold_safe_category_ids: frozenset[int] | None = None,
    audit_score_threshold: float | None = None,
) -> tuple[list[_Candidate], dict[str, int]]:
    """Run the frozen anchor-greedy contract with an exact spatial index.

    Fine labels are independent under the safe-fusion contract.  Indexing each
    label separately avoids comparing every RT-DETR proposal with proposals of
    the other 24 classes.  The grid only removes pairs that have zero positive
    intersection, so the selected clusters are identical to the former
    all-pairs implementation.  Public entry points prohibit zero merge
    thresholds because the ``or`` contract would otherwise merge every
    cross-tile same-class pair.
    """

    by_label: dict[int, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_label[candidate.label].append(candidate)

    canonical: list[_Candidate] = []
    audit = {
        "cluster_count": 0,
        "threshold_inversion_count": 0,
        "threshold_inversion_applied_count": 0,
        "threshold_safe_cluster_count": 0,
        "legacy_score_drop_cluster_count": 0,
        "owner_selected_count": 0,
        "no_output_eligible_member_count": 0,
    }
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
            audit["cluster_count"] += 1
            legacy = _canonical(cluster)
            maximum = max(item.score for item in cluster)
            if legacy.score + 1e-12 < maximum:
                audit["legacy_score_drop_cluster_count"] += 1
            diagnostic_threshold = (
                output_score_threshold
                if audit_score_threshold is None
                else audit_score_threshold
            )
            inverted = (
                diagnostic_threshold is not None
                and maximum >= diagnostic_threshold
                and legacy.score < diagnostic_threshold
            )
            if inverted:
                audit["threshold_inversion_count"] += 1
            apply_threshold_safe = output_score_threshold is not None and (
                threshold_safe_category_ids is None or label in threshold_safe_category_ids
            )
            if not apply_threshold_safe:
                selected = legacy
            else:
                audit["threshold_safe_cluster_count"] += 1
                if inverted:
                    audit["threshold_inversion_applied_count"] += 1
                selected = _threshold_safe_canonical(
                    cluster,
                    output_score_threshold=output_score_threshold,
                    owner_logit_slack=owner_logit_slack,
                )
                if selected is None:
                    audit["no_output_eligible_member_count"] += 1
                    continue
            if selected.owns_center:
                audit["owner_selected_count"] += 1
            canonical.append(selected)
    return sorted(canonical, key=_candidate_sort_key), audit


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
    score_threshold_by_coarse: Mapping[str, float] | None = None,
    score_threshold_by_fine: Mapping[int, float] | None = None,
    merge_iou: float = 0.50,
    merge_ios: float = 0.75,
    fine_nms_iou: float = 0.70,
    border_margin: float = 8.0,
    max_detections: int | None = None,
    output_score_threshold: float | None = None,
    owner_logit_slack: float | None = None,
    threshold_safe_category_ids: Sequence[int] | None = None,
    audit_score_threshold: float | None = None,
    return_audit: bool = False,
) -> Prediction | tuple[Prediction, SafeFusionAudit]:
    """Fuse duplicates without changing fine classes or propagating clusters."""
    score_threshold = _validate_threshold(score_threshold, "score_threshold")
    if score_threshold_by_coarse is not None:
        expected = {"ship", "aircraft", "vehicle"}
        if set(score_threshold_by_coarse) != expected:
            raise ValueError(
                "score_threshold_by_coarse must contain exactly ship, aircraft, vehicle"
            )
        score_threshold_by_coarse = {
            name: _validate_threshold(value, f"score_threshold_by_coarse.{name}")
            for name, value in score_threshold_by_coarse.items()
        }
    score_threshold_by_fine = normalize_fine_thresholds(score_threshold_by_fine)
    if output_score_threshold is not None:
        output_score_threshold = _validate_threshold(
            output_score_threshold, "output_score_threshold"
        )
        if output_score_threshold < score_threshold:
            raise ValueError("output_score_threshold must be >= score_threshold")
    if audit_score_threshold is not None:
        audit_score_threshold = _validate_threshold(
            audit_score_threshold, "audit_score_threshold"
        )
        if audit_score_threshold < score_threshold:
            raise ValueError("audit_score_threshold must be >= score_threshold")
    normalized_safe_ids: frozenset[int] | None = None
    if threshold_safe_category_ids is not None:
        if output_score_threshold is None:
            raise ValueError(
                "threshold_safe_category_ids requires output_score_threshold"
            )
        values: set[int] = set()
        for value in threshold_safe_category_ids:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(
                    "threshold_safe_category_ids must contain non-negative integers"
                )
            values.add(int(value))
        if not values:
            raise ValueError("threshold_safe_category_ids must not be empty")
        normalized_safe_ids = frozenset(values)
    if owner_logit_slack is not None:
        if output_score_threshold is None:
            raise ValueError("owner_logit_slack requires output_score_threshold")
        if not math.isfinite(owner_logit_slack) or owner_logit_slack < 0.0:
            raise ValueError("owner_logit_slack must be finite and >= 0")
    merge_iou = _validate_threshold(merge_iou, "merge_iou")
    merge_ios = _validate_threshold(merge_ios, "merge_ios")
    if merge_iou <= 0.0 or merge_ios <= 0.0:
        raise ValueError("merge_iou and merge_ios must both be > 0")
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
        score_threshold_by_coarse=score_threshold_by_coarse,
        score_threshold_by_fine=score_threshold_by_fine,
        border_margin=border_margin,
    )
    canonical, audit_counts = _anchor_greedy_canonical(
        candidates,
        merge_iou,
        merge_ios,
        output_score_threshold=output_score_threshold,
        owner_logit_slack=owner_logit_slack,
        threshold_safe_category_ids=normalized_safe_ids,
        audit_score_threshold=audit_score_threshold,
    )

    canonical_count = len(canonical)
    canonical = _fine_nms(canonical, fine_nms_iou)
    if max_detections is not None:
        canonical = canonical[: int(max_detections)]
    prediction = Prediction(
        parent_image_id,
        [list(item.box) for item in canonical],
        [item.score for item in canonical],
        [item.label for item in canonical],
    )
    if not return_audit:
        return prediction
    audit = SafeFusionAudit(
        candidate_count=len(candidates),
        canonical_count_before_nms=canonical_count,
        output_count=len(canonical),
        **audit_counts,
    )
    return prediction, audit


__all__ = ["SafeFusionAudit", "fuse_safe_tile_predictions"]
