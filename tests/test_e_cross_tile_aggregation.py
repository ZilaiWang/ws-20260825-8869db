"""E 主线 3 核心场景回归测试：跨 tile 切分 + 多细类冲突的聚合归并。

对应材料 37（E 主线 3 审查）整改第 4 项——用合成 10K 大图跑通
fusion="global" 路径（跨 tile），验证"同一目标跨 tile 切成多块 + 被报成
多个细类"时，聚合能否正确归并成一个对象并投票出正确细类。

本测试为纯 CPU，构造逻辑复用 scripts/verify_e_cross_tile_aggregation.py。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_e_cross_tile_aggregation import (  # noqa: E402
    build_fragments,
    evaluate,
)

from rsdet.postprocess.global_aggregation import fuse_global_predictions  # noqa: E402
from rsdet.tiling.slicer import generate_tiles  # noqa: E402
from rsdet.tiling.synthetic import generate_synthetic_scene  # noqa: E402


def _run(seed: int) -> dict:
    scene = generate_synthetic_scene(
        image_size=10000,
        tile_size=1280,
        overlap=256,
        num_ships=12,
        num_aircraft=30,
        num_vehicles=6,
        seed=seed,
    )
    tiles = generate_tiles(10000, 10000, 1280, 256)
    tile_predictions, truth = build_fragments(scene, tiles)
    fused = fuse_global_predictions(
        tile_predictions,
        tiles,
        parent_image_id=scene.image_id,
        image_width=10000,
        image_height=10000,
    )
    return evaluate(scene, fused, truth)


def test_full_10k_tile_count_is_100():
    """冻结几何 1280/256/1024 恰好切出 100 个 tile（E 任务单 S0）。"""
    tiles = generate_tiles(10000, 10000, 1280, 256)
    assert len(tiles) == 100


def test_cross_tile_merge_and_class_vote_across_seeds():
    """跨 tile 切分归并与多细类投票，在多个 seed 下均为 100%。"""
    for seed in (1, 42, 99, 20260815):
        report = _run(seed)
        assert report["n_cross_tile_objects"] >= 20, (
            f"seed={seed} 跨 tile 目标过少，无法构成有效验证: {report}"
        )
        assert report["merge_match_rate"] == 1.0, f"seed={seed} 归并未达 100%: {report}"
        assert report["class_vote_rate"] in (None, 1.0), (
            f"seed={seed} 细类投票未达 100%: {report}"
        )
        assert report["mean_matched_iou"] > 0.9, f"seed={seed} 坐标恢复精度过低: {report}"
