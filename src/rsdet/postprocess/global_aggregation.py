"""全局对象重构（主线 3 核心）：把一个目标在整图/跨 tile 上被报成的多个框、
多个细类聚合为一个对象，并选出最可靠的类别。

机制转译自 Group Evidence Matters (arXiv:2509.10779) 的推理期后处理，
取其中与任务直接相关的两层门 + 投票 + NMS：

1. Spatial Gate（``spatial_cluster``）——按框中心距离聚类（union-find 连通分量）。
   同一目标被 tile 切成多框时中心仍在同一邻域；相邻的不同目标中心也近，
   因此仅靠空间门不够，需要第二层门分开它们。
2. Semantic Gate（``_iou_subcluster``）——空间簇内按 IoU 连通分量再划分子簇。
   同一目标被切出的框高度重叠（IoU 高 → 合并），相邻不同目标几乎不重叠
   （IoU 低 → 分开），由此避免 false merge。用 IoU 连通代替原文的 embedding 聚类，
   便于最小机制落地，后续可替换为更强的语义特征。
3. 细类投票（``class_vote``）——子簇内按 score 加权投票决定唯一细类，
   解决"同一目标被预测成多个型号"的冲突；平手时取较小 category_id 保证确定性。
4. canonical 框选择——取子簇内最高分框的坐标作为对象框。
5. 同类 NMS（``class_aware_nms``）——对聚合结果做同类内 NMS，处理跨簇残留
   的同框重叠；跨细类不互压（跨类合并由投票负责，而非 NMS）。

proposal 统一为 dict：{'x','y','width','height','category_id','score'}，
xywh 格式，与 OOF proposal CSV 同构。

用法::

    objects = aggregate(proposals, cluster_eps=50.0, merge_iou=0.3, nms_iou=0.5)

输出每个对象一个框：{'x','y','width','height','category_id','score','evidence'}，
``score`` 为子簇内最高分，``evidence`` 为子簇内候选数（聚合证据量）。
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, List, Tuple

import numpy as np

from rsdet.contracts import InferenceSample, Prediction, TileRecord
from rsdet.tiling.coordinates import clip_bbox, tile_to_full, xywh_to_xyxy

logger = logging.getLogger(__name__)


def _iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """计算两个 xywh 框的 IoU。"""
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["width"], a["y"] + a["height"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def spatial_cluster(proposals: List[Dict[str, Any]], eps: float = 50.0) -> List[List[int]]:
    """按框中心距离做 union-find 聚类（Spatial Gate，DBSCAN 的连通分量简化）。

    用网格哈希加速：把框中心按 eps 划入格点，只检查同格与 8 邻域格内的配对。
    与 O(n^2) 逐对版本语义完全一致（同一判据 dx^2+dy^2 <= eps^2、同一 union-find），
    但把配对量从 O(n^2) 降到近线性，支撑 10K 大图数千 proposal 的 20s 预算。

    Args:
        proposals: xywh proposal 列表。
        eps: 中心欧氏距离阈值，低于该值的框连通。

    Returns:
        簇列表，每个簇是 proposals 的下标列表。
    """
    n = len(proposals)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    cell = eps if eps > 0 else 1e-9
    cells: Dict[Tuple[int, int], List[Tuple[int, float, float]]] = {}
    for i in range(n):
        cx = proposals[i]["x"] + proposals[i]["width"] / 2
        cy = proposals[i]["y"] + proposals[i]["height"] / 2
        gx = math.floor(cx / cell)
        gy = math.floor(cy / cell)
        cells.setdefault((gx, gy), []).append((i, cx, cy))

    eps_sq = eps * eps
    for (gx, gy), members in cells.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor = cells.get((gx + dx, gy + dy))
                if neighbor is None:
                    continue
                for ai, ax, ay in members:
                    for bi, bx, by in neighbor:
                        if bi <= ai:  # 每对只检查一次，且避免自配对
                            continue
                        ddx = ax - bx
                        ddy = ay - by
                        if ddx * ddx + ddy * ddy <= eps_sq:
                            union(ai, bi)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _iou_subcluster(
    indices: List[int],
    proposals: List[Dict[str, Any]],
    merge_iou: float,
) -> List[List[int]]:
    """空间簇内按 IoU 连通分量划分子簇（Semantic Gate 的最小替代）。

    同一目标被 tile 切出的框互连，相邻的不同目标互不连接，由此分开。
    """
    n = len(indices)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if _iou(proposals[indices[i]], proposals[indices[j]]) >= merge_iou:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(indices[i])
    return list(groups.values())


def _class_vote_scores(
    indices: List[int], proposals: List[Dict[str, Any]]
) -> Dict[int, float]:
    """子簇内各细类的 score 加权票数（evidence 明细，主线 2 消费）。"""
    votes: Dict[int, float] = {}
    for i in indices:
        p = proposals[i]
        votes[p["category_id"]] = votes.get(p["category_id"], 0.0) + p["score"]
    return votes


def class_vote(indices: List[int], proposals: List[Dict[str, Any]]) -> int:
    """子簇内 score 加权细类投票；分数平手时取 category_id 较小者（确定性）。"""
    votes = _class_vote_scores(indices, proposals)
    return max(votes, key=lambda c: (votes[c], -c))


def class_aware_nms(boxes: List[Dict[str, Any]], iou_thr: float = 0.5) -> List[Dict[str, Any]]:
    """同类内 NMS：高分框保留，压制同类且 IoU 超阈值的低分框；跨类不互压。

    按类分组后用 numpy 向量化 IoU（逐行计算，内存 O(m)），避免纯 Python
    双循环在数千对象下的 O(k^2) 开销，支撑 10K 大图聚合的 20s 预算。
    结果与逐对版本语义一致。
    """
    n = len(boxes)
    suppressed = [False] * n
    kept_idx: List[int] = []

    groups: Dict[int, List[int]] = defaultdict(list)
    for i, b in enumerate(boxes):
        groups[b["category_id"]].append(i)

    for indices in groups.values():
        if len(indices) <= 1:
            kept_idx.extend(indices)
            continue
        indices.sort(
            key=lambda i: (boxes[i]["score"], boxes[i]["category_id"]),
            reverse=True,
        )
        xyxy = np.array(
            [
                [
                    boxes[i]["x"],
                    boxes[i]["y"],
                    boxes[i]["x"] + boxes[i]["width"],
                    boxes[i]["y"] + boxes[i]["height"],
                ]
                for i in indices
            ],
            dtype=np.float64,
        )
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        m = len(indices)
        for a in range(m):
            ia = indices[a]
            if suppressed[ia]:
                continue
            kept_idx.append(ia)
            if a + 1 >= m:
                break
            # 一次向量化算 box a 与所有低分同类的 IoU，标记需压制的
            x1 = np.maximum(xyxy[a, 0], xyxy[a + 1:, 0])
            y1 = np.maximum(xyxy[a, 1], xyxy[a + 1:, 1])
            x2 = np.minimum(xyxy[a, 2], xyxy[a + 1:, 2])
            y2 = np.minimum(xyxy[a, 3], xyxy[a + 1:, 3])
            iw = np.maximum(0.0, x2 - x1)
            ih = np.maximum(0.0, y2 - y1)
            inter = iw * ih
            union = area[a] + area[a + 1:] - inter
            ious = np.where(union > 0.0, inter / union, 0.0)
            for offset in np.flatnonzero(ious >= iou_thr):
                suppressed[indices[a + 1 + offset]] = True

    # 与逐对版本一致：按 (score, category_id) 全局降序返回，保证输出顺序确定
    kept_idx.sort(
        key=lambda i: (boxes[i]["score"], boxes[i]["category_id"]),
        reverse=True,
    )
    return [boxes[i] for i in kept_idx]


def aggregate(
    proposals: List[Dict[str, Any]],
    cluster_eps: float = 50.0,
    merge_iou: float = 0.3,
    nms_iou: float = 0.5,
) -> List[Dict[str, Any]]:
    """完整聚合流水线：Spatial Gate → IoU 子簇 → 细类投票 → canonical 框 → 同类 NMS。

    Args:
        proposals: xywh proposal 列表（可来自整图或跨 tile 的坐标恢复结果）。
            proposal 可携带可选字段 ``source_tile_id``（来源 tile），
            输出对象将据此带出 ``source_tile_ids``（主线 2 契约需要）。
        cluster_eps: Spatial Gate 中心距离阈值。
        merge_iou: 空间簇内判定"同一对象"的 IoU 阈值（区分相邻不同目标）。
        nms_iou: 同类 NMS 的 IoU 阈值。

    Returns:
        每个对象一个框：{'x','y','width','height','category_id','score','evidence'}；
        当 proposal 带 ``source_tile_id`` 时额外带 ``source_tile_ids``（去重排序）
        与 ``category_votes``（各细类票数明细，{细类: score 加权和}）。
    """
    clusters = spatial_cluster(proposals, eps=cluster_eps)
    aggregated: List[Dict[str, Any]] = []
    for cl in clusters:
        for sub in _iou_subcluster(cl, proposals, merge_iou):
            best = max(sub, key=lambda i: proposals[i]["score"])
            cat = class_vote(sub, proposals)
            p = proposals[best]
            obj: Dict[str, Any] = {
                "x": p["x"],
                "y": p["y"],
                "width": p["width"],
                "height": p["height"],
                "category_id": cat,
                "score": p["score"],
                "evidence": len(sub),
            }
            if "source_tile_id" in p:
                source_tiles = {
                    int(proposals[i]["source_tile_id"]) for i in sub
                }
                obj["source_tile_ids"] = sorted(source_tiles)
                obj["category_votes"] = _class_vote_scores(sub, proposals)
            aggregated.append(obj)
    return class_aware_nms(aggregated, iou_thr=nms_iou)


# ---------------------------------------------------------------------------
# tile 流水线桥接：fuse_global_predictions
# ---------------------------------------------------------------------------

def _validated_label(label: object, *, location: str) -> int:
    """校验细类 id 为 [0, 24] 的整数。"""
    if isinstance(label, bool) or not isinstance(label, Integral):
        raise ValueError(f"{location} must be an integer category id")
    numeric = int(label)
    if not 0 <= numeric <= 24:
        raise ValueError(f"{location} category id {numeric} out of range [0, 24]")
    return numeric


def _validated_score(score: object, *, location: str) -> float:
    """校验 score 为 [0, 1] 的有限数值。"""
    try:
        numeric = float(score)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{location} must be numeric") from error
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{location} must be finite and within [0, 1]")
    return numeric


def _restore_proposals(
    tile_predictions: Sequence[Prediction],
    tiles: Sequence[TileRecord],
    *,
    parent_image_id: int,
    image_width: int,
    image_height: int,
    max_detections: int | None = None,
) -> List[Dict[str, Any]]:
    """校验 tile 预测并把局部框恢复到整图坐标，产出聚合 proposal。

    每个 proposal：{'x','y','width','height','category_id','score','source_tile_id'}。
    裁剪后退化的框（零面积）被丢弃。

    Args:
        见 ``fuse_global_predictions``。

    Returns:
        整图坐标下的 proposal 列表（含 ``source_tile_id``，供对象契约溯源）。
    """
    if len(tile_predictions) != len(tiles):
        raise ValueError("tile_predictions and tiles must have the same length")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("parent image dimensions must be > 0")
    if isinstance(max_detections, bool) or (
        max_detections is not None
        and (not isinstance(max_detections, Integral) or max_detections <= 0)
    ):
        raise ValueError("max_detections must be a positive integer or None")

    tile_ids: set[int] = set()
    proposals: List[Dict[str, Any]] = []
    for index, (prediction, tile) in enumerate(zip(tile_predictions, tiles)):
        if tile.parent_image_id != parent_image_id:
            raise ValueError(
                f"tiles[{index}].parent_image_id={tile.parent_image_id} does not match "
                f"parent_image_id={parent_image_id}"
            )
        if tile.width <= 0 or tile.height <= 0:
            raise ValueError(f"tiles[{index}] dimensions must be > 0")
        if tile.x_offset < 0 or tile.y_offset < 0:
            raise ValueError(f"tiles[{index}] offsets must be >= 0")
        if tile.tile_id in tile_ids:
            raise ValueError(f"duplicate tile_id: {tile.tile_id}")
        tile_ids.add(tile.tile_id)
        if prediction.image_id != tile.tile_id:
            raise ValueError(
                f"tile_predictions[{index}].image_id={prediction.image_id} does not match "
                f"tiles[{index}].tile_id={tile.tile_id}"
            )
        if not (
            len(prediction.boxes_xyxy)
            == len(prediction.scores)
            == len(prediction.labels)
        ):
            raise ValueError(
                f"tile_predictions[{index}] boxes, scores, and labels must have equal lengths"
            )

        for detection_index, (box, score, label) in enumerate(
            zip(prediction.boxes_xyxy, prediction.scores, prediction.labels)
        ):
            location = f"tile_predictions[{index}] detection[{detection_index}]"
            numeric_score = _validated_score(score, location=f"{location}.score")
            numeric_label = _validated_label(label, location=f"{location}.label")
            try:
                restored = tile_to_full(box, tile.x_offset, tile.y_offset)
                clipped = clip_bbox(restored, image_width, image_height)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{location}.box is invalid: {error}") from error
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            proposals.append({
                "x": clipped[0],
                "y": clipped[1],
                "width": clipped[2] - clipped[0],
                "height": clipped[3] - clipped[1],
                "category_id": numeric_label,
                "score": numeric_score,
                "source_tile_id": tile.tile_id,
            })
    return proposals


def _aggregate_objects(
    proposals: List[Dict[str, Any]],
    *,
    cluster_eps: float,
    merge_iou: float,
    nms_iou: float,
    score_threshold: float,
    max_detections: int | None,
) -> List[Dict[str, Any]]:
    """聚合 + 按分值过滤 + 排序 + 截断，返回对象 dict 列表。"""
    objects = aggregate(
        proposals,
        cluster_eps=cluster_eps,
        merge_iou=merge_iou,
        nms_iou=nms_iou,
    )
    if score_threshold > 0.0:
        objects = [o for o in objects if o["score"] >= score_threshold]
    objects.sort(key=lambda o: (-o["score"], o["category_id"], o["x"], o["y"]))
    if max_detections is not None:
        objects = objects[: int(max_detections)]
    return objects


def fuse_global_predictions(
    tile_predictions: Sequence[Prediction],
    tiles: Sequence[TileRecord],
    *,
    parent_image_id: int,
    image_width: int,
    image_height: int,
    cluster_eps: float = 50.0,
    merge_iou: float = 0.3,
    nms_iou: float = 0.5,
    score_threshold: float = 0.0,
    max_detections: int | None = None,
) -> Prediction:
    """全局对象重构融合：把各 tile 预测在整图坐标下聚合为对象。

    与 tile_fusion.fuse_tile_predictions 接口对齐，可直接替换；区别是：
    它用证据投票统一同一目标被切出的多框 / 被报成的多型号，
    而非仅靠粗类 NMS 压近似重叠框。

    流程：
        1. ``_restore_proposals``：坐标恢复 + 裁剪 + 校验 + 丢弃退化框；
        2. ``aggregate()``：空间聚类 → IoU 子簇 → 细类投票 → canonical → 同类 NMS；
        3. ``_aggregate_objects``：按分值过滤、按分数降序、``max_detections`` 截断；
        4. 投影为 ``Prediction``（丢弃 evidence / 来源，仅保留框/分/类）。

    需要对象级证据（evidence / 来源 tile / 细类投票）时用
    ``global_object_manifest()``。

    Args:
        tile_predictions: 各 tile 推理结果（tile 局部坐标），同序于 tiles。
        tiles: 切片记录（含 offset / parent_image_id）。
        parent_image_id: 原图 image_id，写入输出 Prediction。
        image_width / image_height: 原图尺寸，用于裁剪。
        cluster_eps / merge_iou / nms_iou: 透传给 ``aggregate()``。
        score_threshold: 聚合后对象的分值过滤（低于该分丢弃）。
        max_detections: 最多保留的对象数（按分数降序取前 N）。

    Returns:
        全局坐标下的 Prediction；每对象一个框。
    """
    proposals = _restore_proposals(
        tile_predictions,
        tiles,
        parent_image_id=parent_image_id,
        image_width=image_width,
        image_height=image_height,
        max_detections=max_detections,
    )
    if not proposals:
        return Prediction(parent_image_id, [], [], [])
    objects = _aggregate_objects(
        proposals,
        cluster_eps=cluster_eps,
        merge_iou=merge_iou,
        nms_iou=nms_iou,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )

    boxes: List[List[float]] = []
    scores: List[float] = []
    labels: List[int] = []
    for o in objects:
        boxes.append(xywh_to_xyxy([o["x"], o["y"], o["width"], o["height"]]))
        scores.append(o["score"])
        labels.append(o["category_id"])
    return Prediction(parent_image_id, boxes, scores, labels)


@dataclass
class GlobalObject:
    """聚合后的全局对象（主线 2 输出契约）。

    每对象恰对应一个目标一次；下游（容易对象校准输出 / 困难对象完整重裁）
    直接消费该结构，无需再处理跨 tile 重复或跨细类冲突。

    Attributes:
        object_id: 图内对象序号（0 起，唯一）。
        parent_image_id: 原图 image_id。
        bbox_xyxy: 整图坐标框 [x1, y1, x2, y2]（像素）。
        category_id: 聚合后细类 0-24。
        score: canonical 框分数（子簇最高分）。
        evidence: 合并的候选检测数（证据量）。
        source_tile_ids: 贡献证据的 tile id 列表（去重升序）。
        category_votes: 各细类 score 加权票数明细，{细类: 分值}，
            供主线 2 质量门控 / 困难判断使用。
    """

    object_id: int
    parent_image_id: int
    bbox_xyxy: List[float]
    category_id: int
    score: float
    evidence: int
    source_tile_ids: List[int]
    category_votes: Dict[int, float]


def global_object_manifest(
    tile_predictions: Sequence[Prediction],
    tiles: Sequence[TileRecord],
    *,
    parent_image_id: int,
    image_width: int,
    image_height: int,
    cluster_eps: float = 50.0,
    merge_iou: float = 0.3,
    nms_iou: float = 0.5,
    score_threshold: float = 0.0,
    max_detections: int | None = None,
) -> List[GlobalObject]:
    """主线 2 输出契约：聚合后每对象一个 ``GlobalObject``，可溯源、带证据。

    与 ``fuse_global_predictions`` 同源同参数，但保留对象级证据
    （evidence、来源 tile、各细类投票），供下游"容易对象直接输出 /
    困难对象完整重裁"两条路直接消费。

    Returns:
        按分数降序的 ``GlobalObject`` 列表；``object_id`` 为该顺序的下标。
    """
    proposals = _restore_proposals(
        tile_predictions,
        tiles,
        parent_image_id=parent_image_id,
        image_width=image_width,
        image_height=image_height,
        max_detections=max_detections,
    )
    if not proposals:
        return []
    objects = _aggregate_objects(
        proposals,
        cluster_eps=cluster_eps,
        merge_iou=merge_iou,
        nms_iou=nms_iou,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )
    return [
        GlobalObject(
            object_id=index,
            parent_image_id=parent_image_id,
            bbox_xyxy=xywh_to_xyxy([o["x"], o["y"], o["width"], o["height"]]),
            category_id=o["category_id"],
            score=o["score"],
            evidence=o["evidence"],
            source_tile_ids=o.get("source_tile_ids", []),
            category_votes=o.get("category_votes", {}),
        )
        for index, o in enumerate(objects)
    ]


# ---------------------------------------------------------------------------
# WP6：条件计算 / 困难对象二次检测（复用同一检测器，不设计新网络）
# ---------------------------------------------------------------------------

@dataclass
class HardObjectCriteria:
    """困难对象判定与二次检测参数。

    Attributes:
        hard_score_threshold: score 低于该值判困难（低置信）。
        hard_evidence_threshold: evidence 低于该值判困难（跨 tile 证据不足）。
        crop_padding: 围绕对象框外扩的裁剪边长（像素）。
        match_iou: 重检测框与对象框的 IoU 关联阈值。
        max_crops: 单图最多二次检测的困难对象数（时延预算上限）。
        min_score: 只采纳 score 不低于该值的重检测证据。
    """

    hard_score_threshold: float = 0.3
    hard_evidence_threshold: int = 2
    crop_padding: float = 64.0
    match_iou: float = 0.3
    max_crops: int = 8
    min_score: float = 0.0


@dataclass
class RefinementTiming:
    """二次检测计时（纳入 20s 端到端预算检查）。"""

    n_hard: int = 0      # 被判困难的对象数
    n_crops: int = 0     # 实际重裁检测的 crop 数
    refine_s: float = 0.0  # 二次检测总耗时


def gate_hard_objects(
    objects: Sequence[GlobalObject],
    criteria: HardObjectCriteria,
) -> List[int]:
    """返回被判为"困难"的对象下标（低置信 / 证据不足），最弱优先。"""
    hard: List[tuple[int, float]] = []
    for index, obj in enumerate(objects):
        if (
            obj.score < criteria.hard_score_threshold
            or obj.evidence < criteria.hard_evidence_threshold
        ):
            hard.append((index, obj.score))
    hard.sort(key=lambda pair: pair[1])  # score 最低者最先被重检
    return [index for index, _ in hard[: criteria.max_crops]]


def _crop_region(
    bbox_xyxy: List[float],
    pad: float,
    width: int,
    height: int,
) -> List[float]:
    """对象框外扩 pad 后的完整重裁区域，并裁剪到图像边界内。"""
    return [
        max(0.0, bbox_xyxy[0] - pad),
        max(0.0, bbox_xyxy[1] - pad),
        min(float(width), bbox_xyxy[2] + pad),
        min(float(height), bbox_xyxy[3] + pad),
    ]


def _iou_xyxy(a: List[float], b: List[float]) -> float:
    """xyxy 框 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _extract_crop(image: np.ndarray, region: List[float]) -> np.ndarray:
    """从原图裁出 [x1,y1,x2,y2] 区域，保证返回 3 通道 RGB。"""
    x1, y1, x2, y2 = (int(round(v)) for v in region)
    patch = image[y1:y2, x1:x2]
    if patch.ndim == 2:
        patch = np.stack([patch] * 3, axis=-1)
    elif patch.shape[2] == 1:
        patch = np.broadcast_to(patch, (patch.shape[0], patch.shape[1], 3))
    elif patch.shape[2] >= 4:
        patch = patch[:, :, :3]
    return patch.copy()


def re_detect_hard_objects(
    image: np.ndarray,
    objects: Sequence[GlobalObject],
    *,
    detect_batch: Callable[[Sequence[InferenceSample]], Sequence[Prediction]],
    crop_metadata_fn: Callable[[List[float], int], Dict[str, Any]],
    criteria: HardObjectCriteria | None = None,
) -> tuple[List[GlobalObject], RefinementTiming]:
    """对困难对象从原图完整重裁、二次检测、证据融合回写（条件计算）。

    只对 ``gate_hard_objects`` 选出的困难对象（低置信 / 证据不足）付出重检成本，
    容易对象直接通过——这就是"条件计算"：时延只花在值得再看一眼的对象上。

    Args:
        image: 原图 numpy 数组 (H, W, 3) uint8。
        objects: 聚合后的对象清单（主线 2 契约输出）。
        detect_batch: 检测回调，签名同 ``predict_batches(detector, samples, batch_size)``，
            接收各二次检测 crop 的 ``InferenceSample``，返回 crop 局部坐标
            ``Prediction`` 列表（与 samples 同序）。
        crop_metadata_fn: (crop_bbox_xyxy, crop_id) -> metadata dict，
            为每个 crop 生成模型需要的 metadata（如 mock 真值注入）。
        criteria: 困难判定与裁剪参数；缺省用 ``HardObjectCriteria()``。

    Returns:
        (更新后的对象清单, 二次检测计时)。每个困难对象合并重检测证据后
        重新加权投票选类；若重检测得到更高分框，则采纳其更完整的坐标。
    """
    timing = RefinementTiming()
    if not objects:
        return [], timing
    if criteria is None:
        criteria = HardObjectCriteria()

    width, height = image.shape[1], image.shape[0]
    hard_indices = gate_hard_objects(objects, criteria)
    timing.n_hard = len(hard_indices)
    if not hard_indices:
        return list(objects), timing

    samples: List[InferenceSample] = []
    hard_regions: List[tuple[int, List[float]]] = []
    for index in hard_indices:
        obj = objects[index]
        region = _crop_region(obj.bbox_xyxy, criteria.crop_padding, width, height)
        crop_id = obj.parent_image_id * 1_000_000 + index
        patch = _extract_crop(image, region)
        samples.append(
            InferenceSample(
                image_id=crop_id,
                image=patch,
                width=patch.shape[1],
                height=patch.shape[0],
                metadata=crop_metadata_fn(region, crop_id),
            )
        )
        hard_regions.append((index, region))
    timing.n_crops = len(samples)

    t0 = time.perf_counter()
    predictions = list(detect_batch(samples))
    timing.refine_s = time.perf_counter() - t0

    updates: Dict[int, Dict[str, Any]] = {}
    for (index, region), prediction in zip(hard_regions, predictions):
        obj = objects[index]
        ox, oy = float(region[0]), float(region[1])
        votes = dict(obj.category_votes)
        evidence = int(obj.evidence)
        best_score = float(obj.score)
        best_cat = obj.category_id
        best_box: List[float] = list(obj.bbox_xyxy)
        for box, score, label in zip(
            prediction.boxes_xyxy, prediction.scores, prediction.labels
        ):
            if score < criteria.min_score:
                continue
            global_box = [box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy]
            if _iou_xyxy(global_box, obj.bbox_xyxy) < criteria.match_iou:
                continue
            votes[label] = votes.get(label, 0.0) + float(score)
            evidence += 1
            if float(score) > best_score:
                best_score = float(score)
                best_cat = int(label)
                best_box = global_box
        if evidence > obj.evidence:
            # 证据增加才更新；重投票选类，平手取较小 category_id
            winner = max(votes, key=lambda c: (votes[c], -c))
            updates[index] = {
                "bbox_xyxy": best_box if best_score > obj.score else obj.bbox_xyxy,
                "category_id": winner,
                "score": best_score if best_score > obj.score else obj.score,
                "evidence": evidence,
                "category_votes": votes,
            }

    result: List[GlobalObject] = []
    for index, obj in enumerate(objects):
        if index in updates:
            u = updates[index]
            result.append(
                GlobalObject(
                    object_id=obj.object_id,
                    parent_image_id=obj.parent_image_id,
                    bbox_xyxy=u["bbox_xyxy"],
                    category_id=u["category_id"],
                    score=u["score"],
                    evidence=u["evidence"],
                    source_tile_ids=list(obj.source_tile_ids),
                    category_votes=u["category_votes"],
                )
            )
        else:
            result.append(obj)
    return result, timing


__all__ = [
    "GlobalObject",
    "HardObjectCriteria",
    "RefinementTiming",
    "aggregate",
    "class_aware_nms",
    "class_vote",
    "fuse_global_predictions",
    "gate_hard_objects",
    "global_object_manifest",
    "re_detect_hard_objects",
    "spatial_cluster",
]
