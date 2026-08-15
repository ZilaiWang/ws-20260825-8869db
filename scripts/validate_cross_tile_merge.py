#!/usr/bin/env python3
"""E 主线 3：合成 10K 大图跨 tile 冲突归并验证（整改清单第 4 条）。

目标：验证主线 3 核心场景——同一目标在 10K 大图上被多个 tile 同时检测到、
且被报成不同细类时，``fusion="global"`` 聚合能否把它归并为一个对象，
并通过 score 加权细类投票选出最可靠的类别。

方法：
  1. ``generate_synthetic_scene`` 生成含跨 tile 目标的 10K 合成图；
  2. 自定义 ``tile_metadata_fn``：对每个 tile，把落在其中的真值目标，
     按目标在不同 tile 里**随机漂移到相邻细类**（模拟真实跨 tile 细类冲突）；
  3. 用 mock 检测器 + ``run_pipeline(fusion="global", collect_objects=True)``
     跑通完整链路；
  4. 逐对象核对：跨 tile 目标是否归并为单对象、evidence 是否=出现 tile 数、
     投票选出的细类是否=真值（或至少落在投票集内）。

用法：
    python scripts/validate_cross_tile_merge.py [--image-size 10000] [--seed 42]
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import rsdet.pipeline.mock_model  # noqa: F401  注册 mock
from rsdet.contracts import InferenceSample, TileRecord
from rsdet.engine.predictor import predict_batches
from rsdet.models.registry import build_model
from rsdet.pipeline.large_image import PipelineConfig, run_pipeline
from rsdet.postprocess.global_aggregation import GlobalObject
from rsdet.tiling.synthetic import generate_synthetic_scene


def _neighbor_classes(cid: int) -> list[int]:
    """同一大类内的相邻细类（模拟跨 tile 细类漂移）。"""
    if 0 <= cid <= 3:      # ship 大类
        return [0, 1, 2, 3]
    if 4 <= cid <= 23:     # aircraft 大类
        lo = max(4, cid - 1)
        hi = min(23, cid + 1)
        return [lo, cid, hi]
    return [24]            # vehicle


def _tile_metadata_with_conflict(scene, rng: np.random.RandomState, drift_prob: float = 0.5):
    """每个 tile 报告落在其中的真值；跨 tile 目标按 drift_prob 漂移细类。"""

    def _fn(tile: TileRecord) -> dict[str, Any]:
        boxes = []
        tx1, ty1 = float(tile.x_offset), float(tile.y_offset)
        tx2, ty2 = tx1 + tile.width, ty1 + tile.height
        for obj in scene.objects:
            if tile.tile_id not in obj.tile_ids:
                continue
            gx1, gy1, gx2, gy2 = obj.bbox
            lx1 = max(0.0, gx1 - tx1)
            ly1 = max(0.0, gy1 - ty1)
            lx2 = min(float(tile.width), gx2 - tx1)
            ly2 = min(float(tile.height), gy2 - ty1)
            if lx2 <= lx1 or ly2 <= ly1:
                continue
            cid = obj.category_id
            if len(obj.tile_ids) > 1 and rng.random() < drift_prob:
                cid = int(rng.choice(_neighbor_classes(cid)))
            boxes.append(
                {
                    "bbox": [lx1, ly1, lx2, ly2],
                    "category_id": cid,
                    "score": float(rng.uniform(0.5, 0.95)),
                }
            )
        return {"gt_boxes": boxes}

    return _fn


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-size", type=int, default=10000)
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drift-prob", type=float, default=0.5)
    ap.add_argument("--output", type=str, default="outputs/e_wp3/cross_tile_merge.json")
    args = ap.parse_args(argv)

    rng = np.random.RandomState(args.seed)
    scene = generate_synthetic_scene(
        image_size=args.image_size,
        tile_size=args.tile_size,
        overlap=args.overlap,
        num_ships=8,
        num_aircraft=20,
        num_vehicles=4,
        seed=args.seed,
    )
    cross_tile = [o for o in scene.objects if len(o.tile_ids) > 1]
    print(f"合成图 {args.image_size}x{args.image_size}: {len(scene.objects)} 目标, "
          f"其中跨 tile 目标 {len(cross_tile)} ({len(cross_tile)/max(1,len(scene.objects))*100:.0f}%)")

    detector = build_model("mock", {"init_args": {}})
    detector.eval()

    config = PipelineConfig(
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        fusion="global",
        cluster_eps=50.0,
        merge_iou=0.3,
        nms_iou=0.5,
    )

    t0 = time.perf_counter()
    prediction, timing, objects = run_pipeline(
        scene.image,
        detector,
        config=config,
        tile_metadata_fn=_tile_metadata_with_conflict(scene, rng, args.drift_prob),
        collect_objects=True,
    )
    wall_s = time.perf_counter() - t0

    print(f"pipeline: {timing.n_tiles} tiles, {len(scene.objects)} GT, "
          f"{len(prediction.boxes_xyxy)} 输出框, {len(objects)} 对象, {wall_s:.2f}s")

    # ---- 逐 GT 目标核对 ----
    n_total = len(scene.objects)
    n_merged = 0          # 跨 tile 目标被归并为单对象
    n_single = 0          # 单 tile 目标保持单对象
    merge_errors = []     # 跨 tile 目标没被合并（>1 对象覆盖它）
    class_correct = 0
    vote_multi = 0        # 触发多细类投票的对象数
    evidence_ok = 0

    gt_matched: dict[int, list[int]] = collections.defaultdict(list)  # obj_idx -> object idxs
    for gi, obj in enumerate(scene.objects):
        for oi, o in enumerate(objects):
            if _iou(obj.bbox, o.bbox_xyxy) >= 0.3:
                gt_matched[gi].append(oi)

    for gi, obj in enumerate(scene.objects):
        hit = gt_matched.get(gi, [])
        if len(hit) == 0:
            continue
        if len(obj.tile_ids) > 1:
            n_merged += 1
            if len(hit) > 1:
                merge_errors.append((gi, len(hit)))
            o = objects[hit[0]]
            if o.category_id == obj.category_id:
                class_correct += 1
            if o.evidence == len(obj.tile_ids):
                evidence_ok += 1
        else:
            n_single += 1

    for o in objects:
        votes = o.category_votes or {}
        if len(votes) > 1:
            vote_multi += 1

    print("\n===== 跨 tile 冲突归并验证 =====")
    print(f"跨 tile 目标数          : {n_total} (其中跨tile {len(cross_tile)})")
    print(f"跨tile归并为单对象      : {n_merged} / {len(cross_tile)}")
    print(f"跨tile归并错误(被拆开)  : {len(merge_errors)}")
    for gi, k in merge_errors[:5]:
        obj = scene.objects[gi]
        print(f"   - GT#{gi} cat={obj.category_id} 出现在 {len(obj.tile_ids)} tiles, 被拆成 {k} 个对象")
    print(f"投票选出细类=真值       : {class_correct} / {n_merged}")
    print(f"evidence=出现tile数     : {evidence_ok} / {n_merged}")
    print(f"触发多细类投票的对象数  : {vote_multi}")
    print(f"单tile目标保持单对象    : {n_single} / {len(scene.objects) - len(cross_tile)}")

    # 冲突对象样例
    conflict_samples = [o for o in objects if len((o.category_votes or {})) > 1][:3]
    for o in conflict_samples:
        print(f"   冲突样例: cat={o.category_id} votes={dict(o.category_votes)} "
              f"evidence={o.evidence} tiles={o.source_tile_ids}")

    payload = {
        "image_size": args.image_size,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "seed": args.seed,
        "drift_prob": args.drift_prob,
        "n_gt": n_total,
        "n_cross_tile": len(cross_tile),
        "n_objects": len(objects),
        "n_output_boxes": len(prediction.boxes_xyxy),
        "n_merged": n_merged,
        "n_merge_errors": len(merge_errors),
        "class_correct": class_correct,
        "evidence_ok": evidence_ok,
        "vote_multi": vote_multi,
        "n_single": n_single,
        "wall_s": wall_s,
        "timing": timing.to_dict(),
        "note": "mock 检测器 + 跨tile细类漂移模拟；真实 M1 权重待服务器验证",
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
