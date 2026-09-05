"""Pure decision primitives. No alternative implementation of official matching."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

AIRCRAFT = frozenset(range(4, 24))
WEAK_LABELS = frozenset({0, 1, 2, 3, 24})


def validate_rate(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite in [0,1]")
    return value


def certify_keep(
    first_probs: Any,
    original_air_labels: Sequence[int],
    *,
    threshold: float = 0.9,
    views: int = 8,
    margin: float = 2e-4,
) -> np.ndarray:
    """Certify that NO alternative class can reach the full-view threshold.

    `original_air_labels` are local aircraft class IDs 0..19. This bounds the
    mean of view probabilities, NOT softmax(mean(logits)). The returned mask
    only permits KEEP; it never performs an early relabel.

    Mathematical statement assumes unchanged per-view probabilities. CUDA
    batch-size-dependent numerical drift must separately pass output parity.
    `margin` is conservative, not an a priori bound on arbitrary GPU drift.
    """
    p = np.asarray(first_probs, dtype=np.float64)
    labels = np.asarray(original_air_labels)
    tau = validate_rate(threshold, "threshold")
    if isinstance(views, bool) or not isinstance(views, int) or views < 1:
        raise ValueError("views must be a positive integer")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    if p.ndim != 2 or p.shape[1] != 20 or labels.shape != (len(p),):
        raise ValueError("expected [N,20] probabilities and [N] labels")
    if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels >= 20)):
        raise ValueError("original labels must be integer aircraft-local IDs")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid probabilities")
    if not np.allclose(p.sum(1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("probability rows must sum to one")
    # Round conservatively upward, including arithmetic at the exact .2/.9 boundary.
    upper = np.nextafter((p + (views - 1)) / views, np.inf)
    upper += 8 * np.finfo(np.float64).eps
    upper[np.arange(len(p)), labels] = -np.inf
    return upper.max(1) < tau - margin


def full_view_labels(
    probabilities: Any, original_labels: Sequence[int], threshold: float = 0.9
) -> np.ndarray:
    """Return local class decisions from REAL completed D4 probabilities."""
    p = np.asarray(probabilities, dtype=np.float64)
    original = np.asarray(original_labels, dtype=np.int64)
    validate_rate(threshold, "threshold")
    if p.ndim != 2 or p.shape[1] != 20 or len(p) != len(original):
        raise ValueError("shape mismatch")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("invalid completed probabilities")
    best = p.argmax(1)
    return np.where(p[np.arange(len(p)), best] >= threshold, best, original)


def prediction_fingerprint(
    prediction: Any, labels: set[int] | frozenset[int] | None = None
) -> Counter:
    """Order-insensitive MULTISET. Duplicates must not be silently discarded."""
    if not (len(prediction.boxes_xyxy) == len(prediction.scores) == len(prediction.labels)):
        raise ValueError("inconsistent Prediction lengths")
    return Counter(
        (int(label), float(score), *map(float, box))
        for box, score, label in zip(
            prediction.boxes_xyxy, prediction.scores, prediction.labels, strict=True
        )
        if labels is None or int(label) in labels
    )


def route_after_fusion(
    primary: Any,
    alternative: Any,
    *,
    alternative_labels: Sequence[int],
    primary_threshold: float,
    alternative_threshold: float,
) -> Any:
    """Mutually exclusive ownership AFTER each full fusion and its own top-k.

    This does not rescore, union two heads within a class, invent boxes, or run
    a new cross-class NMS. Output uses the actual originating head's score.
    """
    if primary.image_id != alternative.image_id:
        raise ValueError("parent image IDs do not match")
    labels = tuple(alternative_labels)
    if any(isinstance(x, bool) or not isinstance(x, int) for x in labels):
        raise ValueError("ownership labels must be integers")
    if len(labels) != len(set(labels)) or not set(labels) <= WEAK_LABELS:
        raise ValueError("alternative ownership must be a unique subset of Ship/FSC")
    own = set(labels)
    t0 = validate_rate(primary_threshold, "primary_threshold")
    t1 = validate_rate(alternative_threshold, "alternative_threshold")
    rows = []
    for origin, prediction, threshold in ((0, primary, t0), (1, alternative, t1)):
        prediction_fingerprint(prediction)
        for index, (box, score, label) in enumerate(
            zip(prediction.boxes_xyxy, prediction.scores, prediction.labels, strict=True)
        ):
            label, score = int(label), float(score)
            if not 0 <= label < 25 or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid detection")
            if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
                raise ValueError("invalid box")
            if (label in own) != bool(origin) or score < threshold:
                continue
            rows.append((score, origin, index, label, list(box)))
    rows.sort(key=lambda x: (-x[0], x[1], x[2]))
    return type(primary)(
        image_id=primary.image_id,
        boxes_xyxy=[x[4] for x in rows],
        scores=[x[0] for x in rows],
        labels=[x[3] for x in rows],
    )


def apply_aircraft_labels(prediction: Any, local_labels: Sequence[int]) -> Any:
    """Apply class decisions, then the ORIGINAL repository same-class NMS.

    Called for both early-kept and fully evaluated objects. It does NOT create
    fake probabilities for objects whose remaining D4 views were skipped.
    """
    from rsdet.postprocess.nms import nms

    return _apply_aircraft_labels(prediction, local_labels, nms)


def _apply_aircraft_labels(
    prediction: Any, local_labels: Sequence[int], nms: Any, nms_iou: float = 0.5
) -> Any:
    indices = [i for i, label in enumerate(prediction.labels) if int(label) in AIRCRAFT]
    if len(indices) != len(local_labels):
        raise ValueError("aircraft labels do not align")
    labels = list(map(int, prediction.labels))
    for index, value in zip(indices, local_labels, strict=True):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not 0 <= value < 20
        ):
            raise ValueError("invalid aircraft-local label")
        labels[index] = int(value) + 4
    keep = [i for i, label in enumerate(labels) if label not in AIRCRAFT]
    for label in sorted(AIRCRAFT):
        members = [i for i, value in enumerate(labels) if value == label]
        selected = nms(
            [prediction.boxes_xyxy[i] for i in members],
            [prediction.scores[i] for i in members],
            nms_iou,
        )
        keep.extend(members[i] for i in selected)
    keep.sort(key=lambda i: (-float(prediction.scores[i]), i))
    return type(prediction)(
        image_id=prediction.image_id,
        boxes_xyxy=[list(prediction.boxes_xyxy[i]) for i in keep],
        scores=[float(prediction.scores[i]) for i in keep],
        labels=[labels[i] for i in keep],
    )
