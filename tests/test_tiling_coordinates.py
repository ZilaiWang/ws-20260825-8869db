"""切片和坐标转换测试。"""

import pytest

from rsdet.tiling.coordinates import (
    clip_bbox,
    full_to_tile,
    tile_to_full,
    xywh_to_xyxy,
    xyxy_to_xywh,
)
from rsdet.tiling.slicer import generate_tiles


class TestXYConversions:
    def test_xyxy_to_xywh(self):
        assert xyxy_to_xywh([10, 20, 110, 220]) == [10, 20, 100, 200]

    def test_xywh_to_xyxy(self):
        assert xywh_to_xyxy([10, 20, 100, 200]) == [10, 20, 110, 220]

    def test_roundtrip_xyxy(self):
        """xyxy → xywh → xyxy 往返一致。"""
        box = [15.0, 25.0, 115.0, 225.0]
        assert xywh_to_xyxy(xyxy_to_xywh(box)) == box

    def test_roundtrip_xywh(self):
        """xywh → xyxy → xywh 往返一致。"""
        box = [10, 20, 100, 200]
        assert xyxy_to_xywh(xywh_to_xyxy(box)) == box

    def test_invalid_xyxy(self):
        with pytest.raises(ValueError):
            xyxy_to_xywh([100, 20, 10, 220])  # x2 < x1

    def test_invalid_xywh(self):
        with pytest.raises(ValueError):
            xywh_to_xyxy([10, 20, -5, 200])  # w < 0

    def test_invalid_box_length(self):
        with pytest.raises(ValueError):
            xyxy_to_xywh([10, 20, 30])


class TestTileFullConversions:
    def test_tile_to_full_and_back(self):
        """tile 坐标 → 全图 → tile 往返一致。"""
        box_tile = [10, 20, 110, 220]
        box_full = tile_to_full(box_tile, 500, 300)
        assert box_full == [510, 320, 610, 520]
        box_back = full_to_tile(box_full, 500, 300)
        assert box_back == box_tile

    def test_full_to_tile_and_back(self):
        """全图坐标 → tile → 全图往返一致。"""
        box_full = [510, 320, 610, 520]
        box_tile = full_to_tile(box_full, 500, 300)
        assert box_tile == [10, 20, 110, 220]
        box_back = tile_to_full(box_tile, 500, 300)
        assert box_back == box_full


class TestClipBbox:
    def test_clip_partially_outside(self):
        """部分在图像外的 bbox 被裁剪。"""
        clipped = clip_bbox([-10, -5, 150, 250], 100, 200)
        assert clipped == [0, 0, 100, 200]

    def test_clip_fully_outside(self):
        """完全在图像外的 bbox 返回 [0,0,0,0]。"""
        clipped = clip_bbox([200, 300, 300, 400], 100, 200)
        assert clipped == [100, 200, 100, 200]

    def test_clip_fully_inside(self):
        """完全在图像内的 bbox 不变。"""
        clipped = clip_bbox([10, 20, 50, 80], 100, 200)
        assert clipped == [10, 20, 50, 80]

    def test_invalid_image_size(self):
        with pytest.raises(ValueError):
            clip_bbox([10, 20, 50, 80], -1, 0)


class TestGenerateTiles:
    def test_small_image_single_tile(self):
        """tile_size 大于图像时返回单张全图切片。"""
        tiles = generate_tiles(512, 512, 1024, 200)
        assert len(tiles) == 1
        assert tiles[0].width == 512
        assert tiles[0].height == 512

    def test_10k_image_coverage(self):
        """10000×10000 图像能被完整覆盖。"""
        tiles = generate_tiles(10000, 10000, 1024, 200)
        assert len(tiles) > 0
        # 检查最后一个 tile 覆盖右下角
        max_x = max(t.x_offset + t.width for t in tiles)
        max_y = max(t.y_offset + t.height for t in tiles)
        assert max_x >= 10000
        assert max_y >= 10000

    def test_edge_coverage(self):
        """图像边缘被正确覆盖。"""
        tiles = generate_tiles(1000, 1000, 600, 100)
        # 最右 tile 必须覆盖到 1000
        rightmost = max(t.x_offset + t.width for t in tiles)
        assert rightmost >= 1000
        # 最下 tile 必须覆盖到 1000
        bottommost = max(t.y_offset + t.height for t in tiles)
        assert bottommost >= 1000
        assert {t.width for t in tiles} == {600}
        assert {t.height for t in tiles} == {600}

    def test_no_gap_between_tiles(self):
        """相邻切片之间无未覆盖条带。"""
        tiles = generate_tiles(3000, 2000, 1024, 200)
        for y_start in range(0, 2000, 100):
            for x_start in range(0, 3000, 100):
                # 检查每个采样点至少被一个 tile 覆盖
                covered = any(
                    t.x_offset <= x_start < t.x_offset + t.width
                    and t.y_offset <= y_start < t.y_offset + t.height
                    for t in tiles
                )
                assert covered, f"点 ({x_start}, {y_start}) 未被覆盖"

    def test_no_duplicate_tiles(self):
        """不生成完全相同的 tile。"""
        tiles = generate_tiles(2000, 2000, 1024, 200)
        coords = [(t.x_offset, t.y_offset) for t in tiles]
        assert len(coords) == len(set(coords))

    def test_invalid_tile_size(self):
        with pytest.raises(ValueError):
            generate_tiles(1000, 1000, 0, 100)

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            generate_tiles(1000, 1000, 1024, -1)

    def test_overlap_too_large(self):
        with pytest.raises(ValueError):
            generate_tiles(1000, 1000, 500, 600)

    def test_invalid_image_size(self):
        with pytest.raises(ValueError):
            generate_tiles(0, 1000, 512, 100)
