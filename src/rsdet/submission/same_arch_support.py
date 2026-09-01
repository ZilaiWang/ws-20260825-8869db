"""Fixed same-architecture support rescoring for existing COCO detections."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _xywh_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax0, ay0, aw, ah = map(float, first)
    bx0, by0, bw, bh = map(float, second)
    ax1, ay1 = ax0 + max(0.0, aw), ay0 + max(0.0, ah)
    bx1, by1 = bx0 + max(0.0, bw), by0 + max(0.0, bh)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - intersection
    return 0.0 if union <= 0.0 else intersection / union


def rescore_same_fine_support(
    primary: Sequence[Mapping[str, Any]],
    specialist: Sequence[Mapping[str, Any]],
    *,
    label_iou_thresholds: Mapping[int, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Multiply selected primary scores by best same-fine specialist score.

    Geometry, category, order, and candidate count remain unchanged.  A selected
    proposal with no sufficient support receives score zero.  Unselected labels
    bypass the module exactly.
    """
    thresholds = {int(label): float(value) for label, value in label_iou_thresholds.items()}
    if not thresholds:
        raise ValueError("label_iou_thresholds must be non-empty")
    if any(not 0.0 <= value <= 1.0 for value in thresholds.values()):
        raise ValueError("IoU thresholds must be within [0, 1]")
    support_index: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    support_image_ids: set[int] = set()
    for row in specialist:
        support_image_ids.add(int(row["image_id"]))
        support_index[(int(row["image_id"]), int(row["category_id"]))].append(row)

    output: list[dict[str, Any]] = []
    selected = supported = 0
    per_label: dict[int, dict[str, int]] = defaultdict(lambda: {"selected": 0, "supported": 0})
    for original in primary:
        row = dict(original)
        label = int(row["category_id"])
        if label not in thresholds or int(row["image_id"]) not in support_image_ids:
            output.append(row)
            continue
        selected += 1
        per_label[label]["selected"] += 1
        best = max(
            (
                float(item["score"])
                for item in support_index.get((int(row["image_id"]), label), ())
                if _xywh_iou(row["bbox"], item["bbox"]) >= thresholds[label]
            ),
            default=0.0,
        )
        if best > 0.0:
            supported += 1
            per_label[label]["supported"] += 1
        row["score"] = float(row["score"]) * best
        output.append(row)
    return output, {
        "protocol": "same_arch_same_fine_product_support_v1",
        "input_count": len(primary),
        "output_count": len(output),
        "scoped_image_count": len(support_image_ids),
        "selected_count": selected,
        "supported_count": supported,
        "unsupported_count": selected - supported,
        "label_iou_thresholds": {str(key): value for key, value in thresholds.items()},
        "per_label": {str(key): value for key, value in sorted(per_label.items())},
    }


__all__ = ["rescore_same_fine_support"]
