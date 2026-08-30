"""Deterministic ambiguity routing for bounded PAV inference cost."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class RoutedCandidate:
    candidate_id: int
    image_id: int
    ambiguity: float
    reasons: tuple[str, ...]


def ambiguity_score(record: Mapping[str, object]) -> tuple[float, tuple[str, ...]]:
    """Combine deployable disagreement, entropy, border and quality signals."""

    score = 0.0
    reasons: list[str] = []
    entropy = float(record.get("crop_entropy", 0.0))
    if math.isfinite(entropy) and entropy > 0:
        score += min(entropy / math.log(25.0), 1.0)
        reasons.append("fine_entropy")
    if int(record.get("detector_crop_agree", 1)) == 0:
        score += 1.0
        reasons.append("detector_crop_disagreement")
    if bool(record.get("border_risk", False)):
        score += 0.75
        reasons.append("border_risk")
    quality = float(record.get("localization_quality", 1.0))
    if math.isfinite(quality):
        score += max(0.0, min(1.0 - quality, 1.0))
        if quality < 0.75:
            reasons.append("low_localization_quality")
    return score, tuple(reasons)


def route_ambiguous_candidates(
    records: Iterable[Mapping[str, object]],
    *,
    max_per_image: int,
    minimum_ambiguity: float = 0.5,
) -> tuple[RoutedCandidate, ...]:
    """Select at most K candidates per image with stable candidate-ID ties."""

    if max_per_image < 0 or minimum_ambiguity < 0:
        raise ValueError("invalid routing limits")
    by_image: dict[int, list[RoutedCandidate]] = defaultdict(list)
    seen: set[int] = set()
    for record in records:
        candidate_id = int(record["candidate_id"])
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        image_id = int(record["image_id"])
        score, reasons = ambiguity_score(record)
        if score >= minimum_ambiguity:
            by_image[image_id].append(RoutedCandidate(candidate_id, image_id, score, reasons))
    selected: list[RoutedCandidate] = []
    for image_id in sorted(by_image):
        ranked = sorted(by_image[image_id], key=lambda row: (-row.ambiguity, row.candidate_id))
        selected.extend(ranked[:max_per_image])
    return tuple(selected)


__all__ = ["RoutedCandidate", "ambiguity_score", "route_ambiguous_candidates"]
