#!/usr/bin/env python3
"""验证 E 主线 3 核心场景：跨 tile 切分 + 多细类冲突的聚合归并。

材料 37（E 主线 3 审查）整改第 4 项：用合成 10K 大图跑通 fusion="global"
路径（跨 tile），记录跨细类冲突归并的实际触发情况——这是主线 3 的核心证据。

此前 test_e_boundary_accuracy.py 已用小图（3k~5k）+ mock 检测器验证了
"跨 tile 切分"，但：① 不是完整 10K / 100 tiles；② mock 返回正确细类，
未触发"同一目标跨 tile 被报成多个细类"的跨细类冲突归并。

本脚本：
  - 完整 10000×10000 + tile 1280/overlap 256（E 任务单冻结几何）；
  - 对每个跨 tile 目标，在其每个来源 tile 生成"切分碎片框"（目标 bbox
    裁剪到 tile 边界，模拟每个 tile 只看到目标的一部分）；
  - 故意制造多细类冲突：主 tile 给正确细类（高分），其余 tile 给同粗类内
    的扰动细类（低分）；
  - 调 fuse_global_predictions 聚合，验证归并 + score 加权细类投票。

纯 CPU，不依赖 GPU / 真实模型权重。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rsdet.contracts import Prediction
from rsdet.postprocess.global_aggregation import fuse_global_predictions
from rsdet.tiling.slicer import generate_tiles
from rsdet.tiling.synthetic import generate_synthetic_scene


def _coarse(category_id: int) -> str:
    if 0 <= category_id <= 3:
        return "ship"
    if 4 <= category_id <= 23:
        return "aircraft"
    if category_id == 24:
        return "vehicle"
    raise ValueError(f"非法细类: {category_id}")


def _distorted_class(category_id: int, offset: int) -> int:
    """同粗类内偏移出错误细类（制造跨细类冲突）。vehicle 单细类不变。"""
    coarse = _coarse(category_id)
    if coarse == "ship":
        return (category_id + offset) % 4
    if coarse == "aircraft":
        return 4 + (category_id - 4 + offset) % 20
    return category_id


def _center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _center_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def build_fragments(
    scene: Any,
    tiles: list[Any],
) -> tuple[list[Prediction], dict[int, dict[str, Any]]]:
    """为每个 tile 构造碎片预测（切分 + 多细类冲突）。

    Returns:
        (tile_predictions, truth): truth 记录每个 object_id 的真实细类与来源 tile 数。
    """
    per_tile: dict[int, list[tuple[list[float], int, float]]] = {}
    truth: dict[int, dict[str, Any]] = {}
    for oid, obj in enumerate(scene.objects):
        truth[oid] = {
            "true_category": obj.category_id,
            "coarse": _coarse(obj.category_id),
            "n_tiles": len(obj.tile_ids),
            "cross_tile": len(obj.tile_ids) > 1,
        }
        # 先为每个来源 tile 生成"切分碎片"（裁剪到 tile 边界），记录面积。
        fragments: list[tuple[int, list[float], float]] = []  # (tile_id, local_xyxy, area)
        for tid in obj.tile_ids:
            tile = next(t for t in tiles if t.tile_id == tid)
            gx1, gy1, gx2, gy2 = obj.bbox
            tx1, ty1 = float(tile.x_offset), float(tile.y_offset)
            tx2, ty2 = tx1 + tile.width, ty1 + tile.height
            ix1, iy1 = max(gx1, tx1), max(gy1, ty1)
            ix2, iy2 = min(gx2, tx2), min(gy2, ty2)
            if ix2 - ix1 < 1.0 or iy2 - iy1 < 1.0:
                continue
            lx1, ly1 = ix1 - tx1, iy1 - ty1
            lx2, ly2 = ix2 - tx1, iy2 - ty1
            area = (ix2 - ix1) * (iy2 - iy1)
            fragments.append((tid, [lx1, ly1, lx2, ly2], area))
        if not fragments:
            continue
        # 选"可见面积最大"的碎片作为主证据（正确细类 + 高分），
        # 其余碎片模拟"多细类冲突"（同粗类扰动细类 + 低分）。
        fragments.sort(key=lambda f: f[2], reverse=True)
        for rank, (tid, box, _area) in enumerate(fragments):
            if rank == 0:
                cat, score = obj.category_id, 0.95
            else:
                cat, score = _distorted_class(obj.category_id, rank), 0.2
            per_tile.setdefault(tid, []).append((box, cat, score))

    tile_predictions: list[Prediction] = []
    for tile in tiles:
        frags = per_tile.get(tile.tile_id, [])
        tile_predictions.append(
            Prediction(
                tile.tile_id,
                [f[0] for f in frags],
                [f[2] for f in frags],
                [f[1] for f in frags],
            )
        )
    return tile_predictions, truth


def _overlapping_object_ids(gt_objects: list[Any], center_eps: float = 100.0) -> set[int]:
    """检测 synthetic 生成缺陷：不同目标空间几乎重叠（真实遥感不会发生）。

    synthetic 只保证"同细类目标不重叠"，但不同细类目标可能被放到几乎同一
    位置（如船与飞机中心距仅几十像素）。这类目标对的聚合归并并非本验证要
    覆盖的"单目标切分"场景，予以排除。
    """
    excluded: set[int] = set()
    n = len(gt_objects)
    for i in range(n):
        ci = _center(gt_objects[i].bbox)
        for j in range(i + 1, n):
            cj = _center(gt_objects[j].bbox)
            if _center_dist(ci, cj) < center_eps:
                excluded.add(i)
                excluded.add(j)
    return excluded


def evaluate(scene: Any, fused: Prediction, truth: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """匹配聚合对象与原始目标，统计归并正确率与细类投票正确率。"""
    gt_objects = scene.objects
    excluded = _overlapping_object_ids(gt_objects)
    # 每个 GT 目标 → 最佳匹配聚合对象（中心距离最近）
    per_object_match: list[tuple[int, float] | None] = []
    for obj in gt_objects:
        gc = _center(obj.bbox)
        best_idx, best_dist = None, float("inf")
        for j, pb in enumerate(fused.boxes_xyxy):
            d = _center_dist(gc, _center(pb))
            if d < best_dist:
                best_idx, best_dist = j, d
        per_object_match.append((best_idx, best_dist) if best_idx is not None else None)

    # 1:1 归并检查：每个聚合对象被至多 1 个目标"主要占用"
    used_pred: set[int] = set()
    cross_total = 0
    merge_ok = 0
    class_ok = 0
    class_conflict_total = 0
    matched_obj_ious: list[float] = []
    for oid, obj in enumerate(gt_objects):
        if oid in excluded:
            continue
        m = per_object_match[oid]
        if m is None:
            continue
        pidx, dist = m
        # 中心距离足够近才视为"匹配到"（避免把远距离对象误配）
        matched = dist < 200.0 and pidx not in used_pred
        if not matched:
            continue
        used_pred.add(pidx)
        if truth[oid]["cross_tile"]:
            cross_total += 1
            merge_ok += 1
            matched_obj_ious.append(_iou(obj.bbox, fused.boxes_xyxy[pidx]))
            # 细类投票正确性（只对制造了多细类冲突的目标统计）
            if truth[oid]["coarse"] != "vehicle" and truth[oid]["n_tiles"] > 1:
                class_conflict_total += 1
                if fused.labels[pidx] == obj.category_id:
                    class_ok += 1

    cross_tile_gt = sum(
        1 for oid, v in truth.items() if v["cross_tile"] and oid not in excluded
    )
    conflict_gt = sum(
        1
        for oid, v in truth.items()
        if v["cross_tile"] and v["coarse"] != "vehicle" and oid not in excluded
    )
    return {
        "n_objects_total": len(gt_objects),
        "n_excluded_overlapping": len(excluded),
        "n_cross_tile_objects": cross_tile_gt,
        "n_cross_tile_matched": cross_total,
        "n_conflict_objects": conflict_gt,
        "merge_match_rate": (cross_total / cross_tile_gt) if cross_tile_gt else None,
        "class_vote_correct": class_ok,
        "class_vote_total": class_conflict_total,
        "class_vote_rate": (class_ok / class_conflict_total) if class_conflict_total else None,
        "n_fused_objects": len(fused.boxes_xyxy),
        "mean_matched_iou": (
            sum(matched_obj_ious) / len(matched_obj_ious) if matched_obj_ious else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=10000)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--num-ships", type=int, default=12)
    parser.add_argument("--num-aircraft", type=int, default=30)
    parser.add_argument("--num-vehicles", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    scene = generate_synthetic_scene(
        image_size=args.image_size,
        tile_size=args.tile_size,
        overlap=args.overlap,
        num_ships=args.num_ships,
        num_aircraft=args.num_aircraft,
        num_vehicles=args.num_vehicles,
        seed=args.seed,
    )
    tiles = generate_tiles(args.image_size, args.image_size, args.tile_size, args.overlap)
    tile_predictions, truth = build_fragments(scene, tiles)

    fused = fuse_global_predictions(
        tile_predictions,
        tiles,
        parent_image_id=scene.image_id,
        image_width=args.image_size,
        image_height=args.image_size,
    )

    report = {
        "config": {
            "image_size": args.image_size,
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "stride": args.tile_size - args.overlap,
            "seed": args.seed,
        },
        "n_tiles": len(tiles),
        **evaluate(scene, fused, truth),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # 门禁：归并匹配率与细类投票率必须为 100%（合成场景是可控的）
    ok = (
        report["merge_match_rate"] is not None
        and report["merge_match_rate"] >= 0.99
        and (report["class_vote_rate"] is None or report["class_vote_rate"] >= 0.99)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
