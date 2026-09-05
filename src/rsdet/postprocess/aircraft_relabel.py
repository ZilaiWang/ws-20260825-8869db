"""Fixed aircraft-only relabeling, shared by cached replay and runtime checks."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import numpy as np

from rsdet.postprocess.nms import class_aware_nms_predictions


def relabel_aircraft(
    predictions: dict[Any, list[dict]],
    bundles: list[dict],
    *,
    min_probability: float = 0.90,
    nms_iou: float = 0.50,
) -> dict[Any, list[dict]]:
    """Consume one within-aircraft probability vector per routed input box.

    Probability is only a relabeling gate, never a replacement detector score.
    Require exact bundle coverage so missing, duplicate, or stale rows fail closed.
    Ship and vehicle records bypass unchanged, including auxiliary metadata.
    """
    if not math.isfinite(min_probability) or not 0 <= min_probability <= 1:
        raise ValueError("invalid relabel probability threshold")
    output = deepcopy(predictions)
    expected = {
        (i, j)
        for i, rows in predictions.items()
        for j, row in enumerate(rows)
        if 4 <= row["category_id"] < 24
    }
    seen = set()
    for bundle in bundles:
        key = (bundle["image_id"], bundle["prediction_index"])
        if key not in expected or key in seen:
            raise ValueError("unknown or duplicate aircraft bundle")
        seen.add(key)
        row = output[key[0]][key[1]]
        if bundle["old_category"] != row["category_id"]:
            raise ValueError("stale aircraft bundle category")
        probs = np.asarray(bundle["probabilities"], dtype=np.float64)
        if (
            probs.shape != (20,)
            or not np.isfinite(probs).all()
            or np.any(probs < 0)
            or np.any(probs > 1)
            or not np.isclose(probs.sum(), 1.0, atol=1e-5, rtol=0)
        ):
            raise ValueError("expected normalized within-aircraft probabilities")
        if float(probs.max()) >= min_probability:
            row["category_id"] = int(probs.argmax()) + 4
    if seen != expected:
        raise ValueError("incomplete aircraft bundle coverage")
    return class_aware_nms_predictions(output, nms_iou, category_ids=list(range(4, 24)))
