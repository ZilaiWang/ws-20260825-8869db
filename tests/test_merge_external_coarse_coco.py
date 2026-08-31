from __future__ import annotations

from scripts.merge_external_coarse_coco import merge_sources


def _payload(file_name: str) -> dict:
    return {
        "images": [{"id": 1, "file_name": file_name, "width": 10, "height": 10}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 2, 3, 4]}
        ],
        "categories": [{"id": 0, "name": "aircraft"}],
    }


def test_merge_rebases_colliding_train_val_ids() -> None:
    output, audit = merge_sources(
        [("train", _payload("P0001.png")), ("val", _payload("P0001.png"))]
    )
    assert [row["id"] for row in output["images"]] == [1, 2]
    assert [row["file_name"] for row in output["images"]] == [
        "train/P0001.png",
        "val/P0001.png",
    ]
    assert [row["image_id"] for row in output["annotations"]] == [1, 2]
    assert audit["image_ids_contiguous"] is True
