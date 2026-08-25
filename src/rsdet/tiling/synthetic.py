"""合成 10K 大图生成 + 真值。

不依赖任何外部数据或模型权重，纯 numpy 即可。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rsdet.tiling.coordinates import xyxy_to_xywh


@dataclass
class SyntheticObject:
    """合成大图中的单个目标。"""

    bbox: List[float]  # [x1, y1, x2, y2] 全局像素坐标
    category_id: int  # 0-24 细类
    tile_ids: List[int] = field(default_factory=list)  # 出现在哪些 tile 中


@dataclass
class SyntheticScene:
    """合成 10K 大图及其真值。"""

    image: np.ndarray  # (H, W, 3) uint8, RGB
    objects: List[SyntheticObject]
    image_id: int = 0

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    def to_gt_coco(self) -> Dict[str, Any]:
        """导出为 COCO GT JSON 对象（含 images + annotations + categories）。"""
        images = [
            {
                "id": self.image_id,
                "width": self.width,
                "height": self.height,
                "file_name": "synthetic_10k.png",
            }
        ]
        annotations: List[Dict[str, Any]] = []
        for anno_id, obj in enumerate(self.objects):
            x, y, w, h = xyxy_to_xywh(obj.bbox)
            if w <= 0 or h <= 0:
                continue
            annotations.append(
                {
                    "id": anno_id,
                    "image_id": self.image_id,
                    "category_id": obj.category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
        categories = [
            {"id": cid, "name": str(cid), "supercategory": _supercategory(cid)} for cid in range(25)
        ]
        return {"images": images, "annotations": annotations, "categories": categories}

    def save_gt(self, path: str | Path) -> None:
        """将 GT COCO JSON 写入文件。"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_gt_coco(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _supercategory(category_id: int) -> str:
    if 0 <= category_id <= 3:
        return "ship"
    if 4 <= category_id <= 23:
        return "aircraft"
    if category_id == 24:
        return "vehicle"
    return "unknown"


def _ensure_no_overlap_same_class(
    box: Tuple[float, float, float, float],
    placed: List[Tuple[float, float, float, float, int]],
    category_id: int,
    min_gap: float = 4.0,
) -> bool:
    """检查 box 是否与同类别已放置框重叠。返回 True 表示无冲突。"""
    x1, y1, x2, y2 = box
    for px1, py1, px2, py2, pcid in placed:
        if pcid != category_id:
            continue
        ix1 = max(x1, px1)
        iy1 = max(y1, py1)
        ix2 = min(x2, px2)
        iy2 = min(y2, py2)
        if ix2 > ix1 - min_gap and iy2 > iy1 - min_gap:
            return False
    return True


def generate_synthetic_scene(
    *,
    image_size: int = 10000,
    tile_size: int = 1024,
    overlap: int = 128,
    num_ships: int = 10,
    num_aircraft: int = 30,
    num_vehicles: int = 5,
    seed: int = 42,
) -> SyntheticScene:
    """生成一张合成 10K 大图。

    目标按以下策略放置：
      - 一部分放在 tile 内部（不会出现在边界/重叠区）
      - 一部分横跨两个相邻 tile 的重叠区
      - 一部分横跨 tile 边界（水平边 + 垂直边 + 角）

    所有目标保证不与其他同细类目标重叠。

    Args:
        image_size: 图像边长（像素），默认 10000。
        tile_size: 切片边长。
        overlap: 相邻切片重叠量。
        num_ships: 舰船类目标总数（均匀填充 0-3 四个细类）。
        num_aircraft: 飞机类目标总数（均匀填充 4-23 二十个细类）。
        num_vehicles: 车辆类目标总数（固定细类 24）。
        seed: 随机种子，确保可复现。

    Returns:
        SyntheticScene，含图像 numpy 数组 + 目标列表 + 每个目标所在的 tile 列表。
    """
    rng = np.random.RandomState(seed)

    # 背景：暗灰底 + 轻微噪声
    image: np.ndarray = (
        np.clip(
            rng.randint(30, 60, (image_size, image_size, 3), dtype=np.uint8).astype(np.int32)
            + rng.randint(0, 15, (image_size, image_size, 3), dtype=np.uint8).astype(np.int32),
            0,
            255,
        )
    ).astype(np.uint8)

    # ------ 颜色映射 ------
    class_colors: Dict[int, Tuple[int, int, int]] = {}
    for cid in range(0, 4):
        class_colors[cid] = (180, 200, 220)  # 浅蓝灰 - ship
    for cid in range(4, 24):
        class_colors[cid] = (220, 200, 180)  # 浅暖色 - aircraft
    class_colors[24] = (180, 220, 180)  # 浅绿 - vehicle

    # ------ 计算 tile 位置 ------
    stride = tile_size - overlap
    x_starts: List[int] = []
    pos = 0
    while pos < image_size:
        x_starts.append(pos)
        pos += stride
    # 最后一块对齐到边缘
    if x_starts[-1] + tile_size < image_size:
        x_starts.append(image_size - tile_size)
    y_starts: List[int] = []
    pos = 0
    while pos < image_size:
        y_starts.append(pos)
        pos += stride
    if y_starts[-1] + tile_size < image_size:
        y_starts.append(image_size - tile_size)

    tiles_info = []
    tid = 0
    for ys in y_starts:
        for xs in x_starts:
            tiles_info.append(
                {
                    "tid": tid,
                    "x": xs,
                    "y": ys,
                    "w": min(tile_size, image_size - xs),
                    "h": min(tile_size, image_size - ys),
                }
            )
            tid += 1

    # ------ 分配细类 ------
    ship_classes = list(range(0, 4))
    aircraft_classes = list(range(4, 24))
    vehicle_class = 24

    def _pick_classes(pool: List[int], count: int) -> List[int]:
        return [pool[i % len(pool)] for i in range(count)]

    target_specs: List[Tuple[float, float, int]] = []  # (min_w, max_w, category_id)

    for cid in _pick_classes(ship_classes, num_ships):
        target_specs.append((80, 160, cid))  # 中型舰船
    for cid in _pick_classes(aircraft_classes, num_aircraft):
        target_specs.append((40, 90, cid))  # 小型飞机
    for _ in range(num_vehicles):
        target_specs.append((24, 48, vehicle_class))  # 更小车

    rng.shuffle(target_specs)

    # ------ 放置策略 ------
    # 三类位置：interior（完全在 1 个 tile 内）、overlap（横跨重叠区）、edge（沿但不过 tile 边）
    # 每种策略的比例
    placement_modes = ["interior"] * 6 + ["overlap_x"] * 1 + ["overlap_y"] * 1 + ["edge"] * 2

    placed: List[Tuple[float, float, float, float, int]] = []
    objects: List[SyntheticObject] = []
    max_attempts = 200

    for spec_idx, (min_w, max_w, cid) in enumerate(target_specs):
        mode = placement_modes[spec_idx % len(placement_modes)]
        bbox: Optional[Tuple[float, float, float, float]] = None

        for _attempt in range(max_attempts):
            # 随机框尺寸
            bw = rng.uniform(min_w, max_w)
            bh = rng.uniform(min_w, max_w)

            if mode == "interior":
                # 放在某个随机 tile 的安全室内
                tile = tiles_info[rng.randint(0, len(tiles_info))]
                margin = 32
                x1 = rng.uniform(tile["x"] + margin, tile["x"] + tile["w"] - bw - margin)
                y1 = rng.uniform(tile["y"] + margin, tile["y"] + tile["h"] - bh - margin)
                bbox = (x1, y1, x1 + bw, y1 + bh)

            elif mode == "overlap_x":
                # 横跨两个水平相邻 tile 的重叠区
                # 找有水平邻居的 tile
                candidates_x = [
                    t
                    for t in tiles_info
                    if any(t2["x"] == t["x"] + stride and t2["y"] == t["y"] for t2 in tiles_info)
                ]
                if not candidates_x:
                    candidates_x = tiles_info
                tile = candidates_x[rng.randint(0, len(candidates_x))]
                # 框中心放在重叠区
                overlap_center = tile["x"] + tile["w"] - overlap / 2.0
                x1 = overlap_center - bw / 2.0
                y1 = rng.uniform(tile["y"] + 16, tile["y"] + tile["h"] - bh - 16)
                bbox = (x1, y1, x1 + bw, y1 + bh)

            elif mode == "overlap_y":
                # 横跨两个垂直相邻 tile 的重叠区
                candidates_y = [
                    t
                    for t in tiles_info
                    if any(t2["y"] == t["y"] + stride and t2["x"] == t["x"] for t2 in tiles_info)
                ]
                if not candidates_y:
                    candidates_y = tiles_info
                tile = candidates_y[rng.randint(0, len(candidates_y))]
                overlap_center = tile["y"] + tile["h"] - overlap / 2.0
                x1 = rng.uniform(tile["x"] + 16, tile["x"] + tile["w"] - bw - 16)
                y1 = overlap_center - bh / 2.0
                bbox = (x1, y1, x1 + bw, y1 + bh)

            else:  # edge — 靠近 tile 边界但不过
                tile = tiles_info[rng.randint(0, len(tiles_info))]
                edge_choice = rng.randint(0, 4)
                margin = 8.0
                if edge_choice == 0:
                    x1 = tile["x"] + margin
                    y1 = rng.uniform(tile["y"] + margin, tile["y"] + tile["h"] - bh - margin)
                elif edge_choice == 1:
                    x1 = tile["x"] + tile["w"] - bw - margin
                    y1 = rng.uniform(tile["y"] + margin, tile["y"] + tile["h"] - bh - margin)
                elif edge_choice == 2:
                    x1 = rng.uniform(tile["x"] + margin, tile["x"] + tile["w"] - bw - margin)
                    y1 = tile["y"] + margin
                else:
                    x1 = rng.uniform(tile["x"] + margin, tile["x"] + tile["w"] - bw - margin)
                    y1 = tile["y"] + tile["h"] - bh - margin
                bbox = (x1, y1, x1 + bw, y1 + bh)

            x1, y1, x2, y2 = bbox
            # 边界检查
            if x1 < 0 or y1 < 0 or x2 > image_size or y2 > image_size:
                continue
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            # 同类别不重叠
            if not _ensure_no_overlap_same_class(bbox, placed, cid):
                continue
            break
        else:
            # 尝试耗尽 → 跳过这个目标
            continue

        x1, y1, x2, y2 = bbox
        placed.append((x1, y1, x2, y2, cid))

        # 画矩形到图像
        color = class_colors.get(cid, (255, 255, 255))
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        # 填充
        image[y1i:y2i, x1i:x2i] = np.array(color, dtype=np.uint8)
        # 边框（稍亮）
        border = np.clip(np.array(color, dtype=np.int32) + 40, 0, 255).astype(np.uint8)
        image[y1i:y2i, x1i : x1i + 2] = border
        image[y1i:y2i, x2i - 2 : x2i] = border
        image[y1i : y1i + 2, x1i:x2i] = border
        image[y2i - 2 : y2i, x1i:x2i] = border

        # 找这个目标落在哪些 tile 中（基于全局 IOU 判定）
        in_tiles: List[int] = []
        for tile in tiles_info:
            tx1, ty1 = tile["x"], tile["y"]
            tx2, ty2 = tx1 + tile["w"], ty1 + tile["h"]
            ix1 = max(x1, tx1)
            iy1 = max(y1, ty1)
            ix2 = min(x2, tx2)
            iy2 = min(y2, ty2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area = (x2 - x1) * (y2 - y1)
                if inter / area > 0.3:  # 至少 30% 在 tile 内
                    in_tiles.append(tile["tid"])

        objects.append(
            SyntheticObject(
                bbox=[x1, y1, x2, y2],
                category_id=cid,
                tile_ids=in_tiles,
            )
        )

    return SyntheticScene(image=image, objects=objects, image_id=0)
