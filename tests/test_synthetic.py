"""合成大图生成测试。"""

import json

import numpy as np

from rsdet.tiling.synthetic import (
    SyntheticObject,
    SyntheticScene,
    generate_synthetic_scene,
)


class TestSyntheticScene:
    def test_image_shape(self):
        """合成图尺寸正确。"""
        scene = generate_synthetic_scene(image_size=5000, seed=42)
        assert scene.image.shape == (5000, 5000, 3)
        assert scene.image.dtype == np.uint8

    def test_has_objects(self):
        """合成图至少生成了一些目标。"""
        scene = generate_synthetic_scene(
            image_size=5000,
            num_ships=10,
            num_aircraft=30,
            num_vehicles=5,
            seed=42,
        )
        assert len(scene.objects) > 0

    def test_objects_within_bounds(self):
        """所有生成的 object bbox 在图像范围内。"""
        scene = generate_synthetic_scene(image_size=8000, seed=123)
        for obj in scene.objects:
            x1, y1, x2, y2 = obj.bbox
            assert 0 <= x1 < x2 <= 8000, f"bbox x 越界: {obj.bbox}"
            assert 0 <= y1 < y2 <= 8000, f"bbox y 越界: {obj.bbox}"

    def test_no_same_class_overlap(self):
        """同细类目标不重叠。"""
        scene = generate_synthetic_scene(
            image_size=5000,
            num_ships=20,
            num_aircraft=50,
            num_vehicles=10,
            seed=42,
        )
        by_class: dict[int, list[list[float]]] = {}
        for obj in scene.objects:
            by_class.setdefault(obj.category_id, []).append(obj.bbox)

        for cid, boxes in by_class.items():
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    px1, py1, px2, py2 = boxes[j]
                    ix1 = max(x1, px1)
                    iy1 = max(y1, py1)
                    ix2 = min(x2, px2)
                    iy2 = min(y2, py2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        area1 = (x2 - x1) * (y2 - y1)
                        area2 = (px2 - px1) * (py2 - py1)
                        iou = inter / (area1 + area2 - inter)
                        assert iou < 0.3, (
                            f"同细类 {cid} 目标重叠: {boxes[i]} vs {boxes[j]}, IoU={iou:.3f}"
                        )

    def test_deterministic(self):
        """同 seed 两次生成结果一致。"""
        scene1 = generate_synthetic_scene(
            image_size=5000, num_ships=5, num_aircraft=10, num_vehicles=2, seed=99
        )
        scene2 = generate_synthetic_scene(
            image_size=5000, num_ships=5, num_aircraft=10, num_vehicles=2, seed=99
        )
        assert np.array_equal(scene1.image, scene2.image)
        assert len(scene1.objects) == len(scene2.objects)
        for o1, o2 in zip(scene1.objects, scene2.objects):
            assert o1.bbox == o2.bbox
            assert o1.category_id == o2.category_id

    def test_different_seeds_different(self):
        """不同 seed 生成不同结果。"""
        scene1 = generate_synthetic_scene(
            image_size=5000, num_ships=5, num_aircraft=10, num_vehicles=2, seed=1
        )
        scene2 = generate_synthetic_scene(
            image_size=5000, num_ships=5, num_aircraft=10, num_vehicles=2, seed=2
        )
        # 图像应不同
        assert not np.array_equal(scene1.image, scene2.image)

    def test_objects_tile_tracking(self):
        """每个目标至少追踪到一个 tile ID。"""
        scene = generate_synthetic_scene(image_size=5000, tile_size=1024, overlap=128, seed=42)
        for obj in scene.objects:
            assert len(obj.tile_ids) > 0, f"目标 {obj.bbox} 未落入任何 tile"


class TestGtCocoFormat:
    def test_gt_coco_format(self):
        """GT COCO JSON 字段完整。"""
        scene = generate_synthetic_scene(
            image_size=5000,
            num_ships=3,
            num_aircraft=5,
            num_vehicles=2,
            seed=42,
        )
        gt = scene.to_gt_coco()

        assert isinstance(gt, dict)
        assert "images" in gt
        assert "annotations" in gt
        assert "categories" in gt

        # images
        assert len(gt["images"]) == 1
        img = gt["images"][0]
        assert img["id"] == scene.image_id
        assert img["width"] == 5000
        assert img["height"] == 5000

        # annotations
        for anno in gt["annotations"]:
            assert "id" in anno
            assert "image_id" in anno
            assert "category_id" in anno
            assert "bbox" in anno
            assert "area" in anno
            assert "iscrowd" in anno
            # bbox 是 xywh, w>0, h>0
            _, _, w, h = anno["bbox"]
            assert w > 0, f"bbox width <= 0: {anno}"
            assert h > 0, f"bbox height <= 0: {anno}"
            # category_id 在 0-24
            assert 0 <= anno["category_id"] <= 24

        # categories
        assert len(gt["categories"]) == 25
        for cat in gt["categories"]:
            assert "id" in cat
            assert "name" in cat
            assert "supercategory" in cat

    def test_gt_coco_bbox_xywh_format(self):
        """GT bbox 确实从 xyxy 转换为了 xywh。"""
        # 用一个已知的 object 验证
        obj = SyntheticObject(bbox=[100.0, 200.0, 300.0, 500.0], category_id=4, tile_ids=[0])
        scene = SyntheticScene(
            image=np.zeros((1000, 1000, 3), dtype=np.uint8),
            objects=[obj],
        )
        gt = scene.to_gt_coco()
        bbox = gt["annotations"][0]["bbox"]
        # xywh = [100, 200, 200, 300]
        assert bbox == [100.0, 200.0, 200.0, 300.0]

    def test_save_gt_writes_file(self, tmp_path):
        """save_gt 写入合法 JSON。"""
        scene = generate_synthetic_scene(
            image_size=1000, num_ships=1, num_aircraft=1, num_vehicles=1, seed=42
        )
        path = tmp_path / "gt.json"
        scene.save_gt(path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "images" in data
        assert "annotations" in data
