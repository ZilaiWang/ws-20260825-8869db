"""PAV manifest geometry and metadata contracts."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

PAV_MANIFEST_VERSION = "hera_pav_manifest_v1"
PAV_METADATA_COLUMNS = (
    "meta_detector_score",
    "meta_crop_top1",
    "meta_crop_margin",
    "meta_crop_entropy_norm",
    "meta_detector_crop_agree",
    "meta_log_short_edge",
    "meta_log_area",
    "meta_log_aspect",
    "meta_log_density",
    "meta_coarse_aircraft",
    "meta_coarse_ship",
    "meta_coarse_vehicle",
)


def square_crop_box(box_xyxy: Sequence[float], *, scale: float) -> tuple[float, ...]:
    if len(box_xyxy) != 4 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid proposal box or crop scale")
    x0, y0, x1, y1 = (float(value) for value in box_xyxy)
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise ValueError("proposal box contains non-finite values")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("proposal box has non-positive extent")
    side = max(x1 - x0, y1 - y0) * scale
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)


def metadata_from_node(
    node: Mapping[str, object],
    *,
    coarse_name: str,
) -> dict[str, float]:
    """Orient and compress deployable scalar evidence into 12 stable fields."""

    if coarse_name not in {"aircraft", "ship", "vehicle"}:
        raise ValueError(f"unknown coarse class: {coarse_name}")
    short_edge = max(float(node["short_edge"]), 0.0)
    area = max(float(node["area"]), 0.0)
    aspect = max(float(node["aspect"]), 1.0)
    density = max(float(node["local_density"]), 0.0)
    result = {
        "meta_detector_score": float(node["y5_score"]),
        "meta_crop_top1": float(node["crop_top1"]),
        "meta_crop_margin": float(node["crop_margin"]),
        "meta_crop_entropy_norm": float(node["crop_entropy"]) / math.log(25.0),
        "meta_detector_crop_agree": float(node["detector_crop_agree"]),
        "meta_log_short_edge": math.log1p(short_edge),
        "meta_log_area": math.log1p(area),
        "meta_log_aspect": math.log(aspect),
        "meta_log_density": math.log1p(density),
        "meta_coarse_aircraft": float(coarse_name == "aircraft"),
        "meta_coarse_ship": float(coarse_name == "ship"),
        "meta_coarse_vehicle": float(coarse_name == "vehicle"),
    }
    if tuple(result) != PAV_METADATA_COLUMNS or not all(
        math.isfinite(value) for value in result.values()
    ):
        raise RuntimeError("PAV metadata contract violation")
    return result


__all__ = [
    "PAV_MANIFEST_VERSION",
    "PAV_METADATA_COLUMNS",
    "metadata_from_node",
    "square_crop_box",
]
