from __future__ import annotations

from scripts.build_pseudo_source_projection_probe import build
from scripts.project_source_predictions_to_pseudo10k import project


def test_source_projection_preserves_primary_detection_fields() -> None:
    images = [
        {"id": index + 1, "file_name": f"images/train/{index}.jpg", "width": 500, "height": 1000}
        for index in range(100)
    ]
    annotations = [
        {"id": index + 1, "image_id": index + 1, "category_id": 2, "bbox": [1, 2, 3, 4]}
        for index in range(100)
    ]
    full = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index, "name": str(index)} for index in range(25)],
    }
    pseudo = {
        "images": [
            {
                "id": 900,
                "file_name": "pseudo.jpg",
                "width": 10_000,
                "height": 10_000,
                "source_images": [row["file_name"] for row in images],
            }
        ],
        "annotations": [
            {"id": index + 1, "image_id": 900, "category_id": 2, "bbox": [1, 2, 3, 4]}
            for index in range(100)
        ],
        "categories": full["categories"],
    }
    source, mapping = build(full, pseudo)
    assert len(source["images"]) == 100
    predictions = [{"image_id": 1, "category_id": 2, "bbox": [10, 20, 30, 40], "score": 0.7}]
    output = project(predictions, mapping)
    assert output == [
        {
            "image_id": 900,
            "category_id": 2,
            "bbox": [260.0, 20.0, 30.0, 40.0],
            "score": 0.7,
        }
    ]
