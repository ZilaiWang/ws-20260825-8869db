"""COCO GT 导出和冻结 manifest 过滤测试。"""

import json
from pathlib import Path

from PIL import Image

from scripts.export_coco import export_coco


def test_export_only_manifest_validation_subset(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    image_dir = data_root / "images" / "train"
    label_dir = data_root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    for stem, class_id in (("a", 0), ("b", 24)):
        Image.new("RGB", (100, 80), color=(10, 20, 30)).save(image_dir / f"{stem}.jpg")
        (label_dir / f"{stem}.txt").write_text(f"{class_id} 0.5 0.5 0.2 0.25\n", encoding="utf-8")

    manifest = tmp_path / "dev_v1.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "dev_v1",
                "samples": [
                    {
                        "image_id": 1,
                        "relative_path": "images/train/a.jpg",
                        "split": "train",
                        "group_id": "g1",
                    },
                    {
                        "image_id": 2,
                        "relative_path": "images/train/b.jpg",
                        "split": "val",
                        "group_id": "g2",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "dev_v1_gt.json"
    document = export_coco(
        data_root,
        "train",
        output,
        manifest_path=manifest,
        manifest_split="val",
    )

    assert [image["id"] for image in document["images"]] == [2]
    assert document["annotations"] == [
        {
            "id": 1,
            "image_id": 2,
            "category_id": 24,
            "bbox": [40.0, 30.0, 20.0, 20.0],
            "area": 400.0,
            "iscrowd": 0,
        }
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == document
