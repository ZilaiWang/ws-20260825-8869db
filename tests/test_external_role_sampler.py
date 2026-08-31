from pathlib import Path

from scripts.build_external_role_sampler import build_role_list


def test_external_role_sampler_changes_only_role_repetition(tmp_path: Path) -> None:
    for name in ("a.png", "b.png", "c.png"):
        (tmp_path / name).write_bytes(b"x")
    coco = {
        "images": [
            {"id": 1, "file_name": "a.png"},
            {"id": 2, "file_name": "b.png"},
            {"id": 3, "file_name": "c.png"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 0},
            {"image_id": 2, "category_id": 2},
            {"image_id": 3, "category_id": 3},
        ],
    }
    general, _ = build_role_list(coco, tmp_path, "EXT-G")
    vehicle, _ = build_role_list(coco, tmp_path, "EXT-V")
    assert len(general) == 7
    assert len(vehicle) == 7
