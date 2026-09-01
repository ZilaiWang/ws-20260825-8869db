"""Frozen Background-100MP false-positive stress metric."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BackgroundStressMetrics:
    image_count: int
    megapixels: float
    false_positives: int
    false_positives_per_100mp: float
    per_coarse_false_positives: dict[str, int]
    per_coarse_false_positives_per_100mp: dict[str, float]


def evaluate_background_stress(
    manifest: Sequence[Mapping[str, object]],
    predictions: Mapping[int, Sequence[Mapping[str, object]]],
    *,
    category_mapping: Mapping[int, str],
) -> BackgroundStressMetrics:
    """Count every prediction as FP on an audited object-free manifest."""

    ids = [int(row["image_id"]) for row in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("background manifest image_id values must be unique")
    unexpected = set(predictions) - set(ids)
    if unexpected:
        raise ValueError(f"predictions contain images outside manifest: {sorted(unexpected)[:10]}")
    pixels = sum(int(row["width"]) * int(row["height"]) for row in manifest)
    if pixels <= 0:
        raise ValueError("background manifest must contain positive pixel area")
    per_coarse = {name: 0 for name in sorted(set(category_mapping.values()))}
    for image_id in ids:
        for prediction in predictions.get(image_id, ()):
            category_id = int(prediction["category_id"])
            if category_id not in category_mapping:
                raise ValueError(f"unknown prediction category_id: {category_id}")
            per_coarse[category_mapping[category_id]] += 1
    total = sum(per_coarse.values())
    megapixels = pixels / 1_000_000.0
    scale = 100.0 / megapixels
    return BackgroundStressMetrics(
        image_count=len(ids),
        megapixels=megapixels,
        false_positives=total,
        false_positives_per_100mp=total * scale,
        per_coarse_false_positives=per_coarse,
        per_coarse_false_positives_per_100mp={
            name: count * scale for name, count in per_coarse.items()
        },
    )


__all__ = ["BackgroundStressMetrics", "evaluate_background_stress"]
