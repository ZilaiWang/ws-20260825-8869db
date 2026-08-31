from scripts.select_external_coco_smoke_subset import select_subset


def test_select_subset_covers_categories() -> None:
    payload = {
        "images": [{"id": index, "file_name": f"{index}.png"} for index in range(1, 6)],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0},
            {"id": 2, "image_id": 2, "category_id": 1},
            {"id": 3, "image_id": 3, "category_id": 2},
            {"id": 4, "image_id": 4, "category_id": 3},
        ],
        "categories": [
            {"id": 0, "name": "a"},
            {"id": 1, "name": "b"},
            {"id": 2, "name": "c"},
            {"id": 3, "name": "d"},
        ],
    }
    output, audit = select_subset(payload, 4)
    assert len(output["images"]) == 4
    assert audit["all_categories_covered"] is True
