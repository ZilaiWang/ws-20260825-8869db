"""增量 frontier 评分器（反事实标签生成的快速版本）。

关键优化：单候选动作（DROP/RELABEL）只影响该候选所在 image 的 NMS + 匹配，
其他 image 的 kept/TP 状态不变。因此：
1. base 状态预计算所有 image 的 NMS + 匹配（一次）；
2. 单候选动作只重算该 image（~十几候选），其他 image 复用；
3. 全局 frontier 扫描用 numpy 向量化。

这比完整重算（每次 O(N log N) + O(N×G)）快约 10~50 倍。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from rsdet.evaluation.official_frontier import scan_fixed_risk_points

from .official_scorer import CandidateView, box_iou_matrix


class IncrementalFrontierScorer:
    def __init__(
        self,
        proto: Any,
        gt_boxes: dict[int, list[dict[str, Any]]],
        image_ids: set[int],
        candidates: Sequence[CandidateView],
    ) -> None:
        self.proto = proto
        self.gt_boxes = gt_boxes
        self.image_ids = image_ids
        self.n_gt = sum(len(g) for i, g in gt_boxes.items() if i in image_ids)
        # 候选按 image 分组（可变副本）
        self.cand_by_img: dict[int, list[CandidateView]] = defaultdict(list)
        self.idx_to_img: dict[int, int] = {}
        for c in candidates:
            if c.image_id in image_ids:
                self.cand_by_img[c.image_id].append(c)
                self.idx_to_img[c._idx] = c.image_id
        # base 状态
        self.kept_by_img: dict[int, list[CandidateView]] = {}
        self.tp_by_img: dict[int, set[int]] = {}
        self._recompute_base()

    def _nms_match(self, img: int, cands: list[CandidateView]):
        """单 image 的 NMS + 匹配，返回 (kept, tp_idx_set)。"""
        od = sorted(cands, key=lambda p: -p.score)
        n = len(od)
        if n == 0:
            return [], set()
        boxes = np.asarray([p.bbox_xyxy for p in od], dtype=np.float64)
        ious = box_iou_matrix(boxes, boxes)
        sup: set[int] = set()
        kept: list[CandidateView] = []
        for i, p in enumerate(od):
            if i in sup:
                continue
            kept.append(p)
            for j in range(i + 1, n):
                if j in sup or od[j].category_id != p.category_id:
                    continue
                if ious[i, j] > 0.5:
                    sup.add(j)
        # 匹配
        gts = self.gt_boxes.get(img, [])
        tp: set[int] = set()
        if kept and gts:
            kept_boxes = np.asarray([p.bbox_xyxy for p in kept], dtype=np.float64)
            gt_arr = np.asarray([g["bbox_xyxy"] for g in gts], dtype=np.float64)
            m_ious = box_iou_matrix(kept_boxes, gt_arr)
            # Official semantics are prediction-first, score-descending.  Each
            # prediction claims its best still-unmatched same-fine GT.
            used_gt: set[int] = set()
            for ci, p in enumerate(kept):
                thr = self.proto.iou_thresholds[
                    self.proto.category_mapping[p.category_id]
                ]
                best_gi, best_iou = -1, -1.0
                for gi, g in enumerate(gts):
                    if gi in used_gt or int(g["category_id"]) != p.category_id:
                        continue
                    iou = float(m_ious[ci, gi])
                    if iou >= thr and iou > best_iou:
                        best_iou, best_gi = iou, gi
                if best_gi >= 0:
                    used_gt.add(best_gi)
                    tp.add(p._idx)
        return kept, tp

    def _recompute_base(self) -> None:
        self.kept_by_img = {}
        self.tp_by_img = {}
        for img, cands in self.cand_by_img.items():
            k, t = self._nms_match(img, cands)
            self.kept_by_img[img] = k
            self.tp_by_img[img] = t
        self._scan()

    def _scan(self) -> None:
        """全局 frontier 扫描（numpy 向量化）。"""
        all_kept = [c for k in self.kept_by_img.values() for c in k]
        if not all_kept:
            self.recall_at_fdr_012 = 0.0
            return
        rows = [
            (
                p.score,
                p._idx,
                p._idx in self.tp_by_img.get(p.image_id, set()),
            )
            for p in all_kept
        ]
        point = scan_fixed_risk_points(
            scored_tp_rows=rows,
            n_gt=self.n_gt,
            fdr_levels=(0.12,),
        )[0.12]
        self.recall_at_fdr_012 = point.recall
        self.n_tp = point.tp
        self.n_fp = point.fp
        self.n_tp_total = sum(bool(row[2]) for row in rows)
        self.n_fp_total = len(rows) - self.n_tp_total

    def score(self) -> float:
        return self.recall_at_fdr_012

    def score_after(self, candidate_idx: int, action_kind: str, new_cls: int | None = None) -> tuple[float, int, int]:
        """动作后的 (frontier 分数, tp 数, fp 数)。不改变内部状态。"""
        return self.score_after_multi({candidate_idx: (action_kind, new_cls)})

    def score_after_multi(self, edits: dict[int, tuple[str, int | None]]) -> tuple[float, int, int]:
        """多候选动作后的 (frontier, tp, fp)。只重算受影响的 image。

        edits: {candidate_idx: (action_kind, new_cls)}。所有候选须在同一 image
        （pairwise 交互场景），跨 image 则逐 image 分组重算（通用但稍慢）。
        不改变内部状态。
        """
        if not edits:
            return self.recall_at_fdr_012, self.n_tp, self.n_fp
        # 按 image 分组
        edits_by_img: dict[int, dict[int, tuple[str, int | None]]] = defaultdict(dict)
        for idx, (ak, nc) in edits.items():
            img = self.idx_to_img.get(idx)
            if img is None:
                continue
            edits_by_img[img][idx] = (ak, nc)

        kept_pool: list[CandidateView] = []
        tp_pool: set[int] = set()
        touched_imgs = set(edits_by_img.keys())
        for img, ok in self.kept_by_img.items():
            if img in touched_imgs:
                cands = list(self.cand_by_img[img])
                drops = {idx for idx, (ak, _) in edits_by_img[img].items() if ak == "drop"}
                relabels = {idx: nc for idx, (ak, nc) in edits_by_img[img].items()
                            if ak == "relabel" and nc is not None}
                new_cands = []
                for c in cands:
                    if c._idx in drops:
                        continue
                    if c._idx in relabels:
                        new_cands.append(CandidateView(c._idx, c.image_id, relabels[c._idx],
                                                       c.score, c.bbox_xyxy))
                    else:
                        new_cands.append(c)
                k, t = self._nms_match(img, new_cands)
                kept_pool.extend(k)
                tp_pool |= t
            else:
                kept_pool.extend(ok)
                tp_pool |= self.tp_by_img[img]

        if not kept_pool:
            return 0.0, 0, 0
        point = scan_fixed_risk_points(
            scored_tp_rows=(
                (p.score, p._idx, p._idx in tp_pool) for p in kept_pool
            ),
            n_gt=self.n_gt,
            fdr_levels=(0.12,),
        )[0.12]
        return point.recall, point.tp, point.fp
