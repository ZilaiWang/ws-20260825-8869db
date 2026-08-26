from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .actions import Candidate


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_relation_features(candidates: Sequence[Candidate]) -> np.ndarray:
    """Reference relation tensor [N,N,8]; extend with project-specific flags."""

    n = len(candidates)
    out = np.zeros((n, n, 8), dtype=np.float32)
    for i, a in enumerate(candidates):
        ax1, ay1, ax2, ay2 = a.box_xyxy
        acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
        aw, ah = max(ax2 - ax1, 1e-6), max(ay2 - ay1, 1e-6)
        aa = aw * ah
        for j, b in enumerate(candidates):
            bx1, by1, bx2, by2 = b.box_xyxy
            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            bw, bh = max(bx2 - bx1, 1e-6), max(by2 - by1, 1e-6)
            ba = bw * bh
            out[i, j] = np.asarray(
                [
                    box_iou(a.box_xyxy, b.box_xyxy),
                    (bcx - acx) / math.sqrt(aa),
                    (bcy - acy) / math.sqrt(aa),
                    math.log(bw / aw),
                    math.log(bh / ah),
                    math.log(ba / aa),
                    float(a.cls_id == b.cls_id),
                    b.score - a.score,
                ],
                dtype=np.float32,
            )
    return out
