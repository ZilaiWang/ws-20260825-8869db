"""官方 frontier 评分器适配器（SCOPE 的 ExactScorer）。

把「候选集合」映射为官方核心指标 Recall@FDR=0.12 前沿分数，完全复用
``scripts/a5_oto_oer.py`` 的 OER + 同类 NMS + 贪心匹配 + fixed-risk frontier 逻辑。

设计要点：
- 候选只含可部署字段（image_id / category_id / score / bbox_xyxy），不触碰 GT 字段；
- GT 只在本模块离线标签阶段使用，deploy 包严禁 import 本模块；
- frontier 分阶段计算：per-image 的 NMS + 贪心匹配（可增量），再全局按 score 扫描。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from rsdet.evaluation.official_frontier import official_fixed_risk_frontier


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """向量化 IoU 矩阵。boxes_a [N,4], boxes_b [M,4] -> [N,M]。"""
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


@dataclass
class CandidateView:
    """单个候选的可部署视图（与 scope_router.actions.Candidate 对齐）。"""

    _idx: int
    image_id: int
    category_id: int
    score: float
    bbox_xyxy: list[float]


@dataclass
class FrontierResult:
    recall_at_fdr: dict[float, float]
    n_gt: int
    n_kept: int
    # Active counts at the primary (first requested) fixed-risk point.
    n_tp: int
    n_fp: int
    # Counts after including every NMS-kept prediction.  These must never be
    # used as active-FP/protected-TP labels for a fixed-risk resolver.
    n_tp_total: int = 0
    n_fp_total: int = 0
    score_threshold_at_fdr: dict[float, float | None] = field(default_factory=dict)


class FrontierScorer:
    """在固定 GT 作用域上计算 Recall@FDR 前沿的评分器。

    Args:
        proto: 解析后的评估协议（含 category_mapping / iou_thresholds）。
        gt_boxes: ``{image_id: [{bbox_xyxy, category_id}]}``。
        image_ids: 参与评估的图像集合；``None`` 表示全部 GT 图像。
        iou_thresholds: 可选覆盖。
    """

    def __init__(
        self,
        proto: Any,
        gt_boxes: dict[int, list[dict[str, Any]]],
        image_ids: set[int] | None = None,
    ) -> None:
        self.proto = proto
        self.gt_boxes = gt_boxes
        self.image_ids = set(gt_boxes.keys()) if image_ids is None else set(image_ids)
        self.n_gt = sum(
            len(gts) for i, gts in gt_boxes.items() if i in self.image_ids
        )

    def frontier(
        self, candidates: Sequence[CandidateView], fdr_levels: Sequence[float] = (0.12, 0.11, 0.10)
    ) -> FrontierResult:
        levels = tuple(float(value) for value in fdr_levels)
        result = official_fixed_risk_frontier(
            gt_boxes=self.gt_boxes,
            predictions=(
                {
                    "image_id": candidate.image_id,
                    "category_id": candidate.category_id,
                    "score": candidate.score,
                    "bbox_xyxy": candidate.bbox_xyxy,
                    "source_prediction_index": candidate._idx,
                }
                for candidate in candidates
            ),
            category_mapping=self.proto.category_mapping,
            iou_thresholds=self.proto.iou_thresholds,
            image_ids=self.image_ids,
            fdr_levels=levels,
            nms_iou=0.50,
        )
        primary = result.points[levels[0]]
        return FrontierResult(
            recall_at_fdr={level: result.points[level].recall for level in levels},
            n_gt=result.n_gt,
            n_kept=result.n_kept,
            n_tp=primary.tp,
            n_fp=primary.fp,
            n_tp_total=result.total_tp,
            n_fp_total=result.total_fp,
            score_threshold_at_fdr={
                level: result.points[level].score_threshold for level in levels
            },
        )

    def score(self, candidates: Sequence[CandidateView], fdr_level: float = 0.12) -> float:
        """返回单个 FDR 水平的 Recall 前沿分数。"""
        return self.frontier(candidates).recall_at_fdr.get(fdr_level, 0.0)


def candidates_from_rows(rows: Iterable[dict[str, Any]]) -> list[CandidateView]:
    """从 ``{_idx, image_id, category_id, score, bbox_xyxy}`` 行构造候选视图。"""
    out: list[CandidateView] = []
    for r in rows:
        out.append(
            CandidateView(
                _idx=int(r["_idx"]),
                image_id=int(r["image_id"]),
                category_id=int(r["category_id"]),
                score=float(r["score"]),
                bbox_xyxy=[float(v) for v in r["bbox_xyxy"]],
            )
        )
    return out
