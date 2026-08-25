"""官方 frontier 评分器适配器（SCOPE 的 ExactScorer）。

把「候选集合」映射为官方核心指标 Recall@FDR=0.12 前沿分数，完全复用
``scripts/a5_oto_oer.py`` 的 OER + 同类 NMS + 贪心匹配 + fixed-risk frontier 逻辑。

设计要点：
- 候选只含可部署字段（image_id / category_id / score / bbox_xyxy），不触碰 GT 字段；
- GT 只在本模块离线标签阶段使用，deploy 包严禁 import 本模块；
- frontier 分阶段计算：per-image 的 NMS + 贪心匹配（可增量），再全局按 score 扫描。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from rsdet.evaluation.official_metric import compute_iou


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
    n_tp: int
    n_fp: int


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

    def _nms(self, candidates: Sequence[CandidateView]) -> list[CandidateView]:
        """同类 IoU>0.5 贪心 NMS（按 score 降序，per-image）。"""
        by_img: dict[int, list[CandidateView]] = defaultdict(list)
        for p in candidates:
            by_img[p.image_id].append(p)
        kept: list[CandidateView] = []
        for img, pl in by_img.items():
            od = sorted(pl, key=lambda p: -p.score)
            sup: set[int] = set()
            for p in od:
                if p._idx in sup:
                    continue
                kept.append(p)
                for q in od:
                    if q._idx == p._idx or q._idx in sup:
                        continue
                    if q.category_id != p.category_id:
                        continue
                    if compute_iou(p.bbox_xyxy, q.bbox_xyxy) > 0.5:
                        sup.add(q._idx)
        return kept

    def _match(self, kept: Sequence[CandidateView]) -> set[int]:
        """贪心匹配，返回 TP 候选的 _idx 集合。"""
        by_img: dict[int, list[CandidateView]] = defaultdict(list)
        for p in kept:
            by_img[p.image_id].append(p)
        used: dict[int, set[int]] = defaultdict(set)
        tp: set[int] = set()
        for img, gts in self.gt_boxes.items():
            if img not in self.image_ids:
                continue
            od = sorted(by_img.get(img, []), key=lambda p: -p.score)
            for g in gts:
                cid = int(g["category_id"])
                thr = self.proto.iou_thresholds[self.proto.category_mapping[cid]]
                bi, bix = 0.0, None
                for p in od:
                    if p._idx in used[img] or p.category_id != cid:
                        continue
                    iou = compute_iou(p.bbox_xyxy, g["bbox_xyxy"])
                    if iou > bi:
                        bi, bix = iou, p
                if bix is not None and bi >= thr:
                    used[img].add(bix._idx)
                    tp.add(bix._idx)
        return tp

    def frontier(
        self, candidates: Sequence[CandidateView], fdr_levels: Sequence[float] = (0.12, 0.11, 0.10)
    ) -> FrontierResult:
        # 只保留作用域内的候选（否则单折评估会把其他折候选计入 FP）
        candidates = [c for c in candidates if c.image_id in self.image_ids]
        kept = self._nms(candidates)
        tp = self._match(kept)
        od = sorted(kept, key=lambda p: -p.score)
        tp_ = fp = 0
        br: dict[float, float] = {v: 0.0 for v in fdr_levels}
        for p in od:
            if p._idx in tp:
                tp_ += 1
            else:
                fp += 1
            rec = tp_ / self.n_gt
            fdr = fp / (tp_ + fp) if tp_ + fp else 0.0
            for v in fdr_levels:
                if fdr <= v:
                    br[v] = max(br[v], rec)
        return FrontierResult(
            recall_at_fdr=br,
            n_gt=self.n_gt,
            n_kept=len(kept),
            n_tp=tp_,
            n_fp=fp,
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
