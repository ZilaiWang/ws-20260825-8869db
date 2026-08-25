"""P04 mock teacher 的小型端到端缓存测试。"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image

from rsdet.features.p04_cache import FeatureCache


def _load_script(name: str = "extract_p04_features.py"):
    path = Path(__file__).resolve().parent.parent / "scripts" / name
    module_name = f"p04_{path.stem}_test_script"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(root: Path) -> Path:
    image_dir = root / "images"
    image_dir.mkdir()
    fields = [
        "manifest_version",
        "crop_id",
        "annotation_uid",
        "source_image_id",
        "source_relative_path",
        "source_width",
        "source_height",
        "source_checksum_sha256",
        "class_id",
        "class_name",
        "major_class",
        "crop_policy",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
        "fold",
        "leakage_group_id",
        "color_mode",
        "outside_policy",
        "resize_semantics",
    ]
    rows = []
    for uid, fold, color in (
        ("a", 0, (255, 0, 0)),
        ("b", 1, (0, 255, 0)),
        ("c", 2, (0, 0, 255)),
    ):
        Image.new("RGB", (4, 4), color).save(image_dir / f"{uid}.png")
        rows.append(
            {
                "manifest_version": "v1",
                "crop_id": f"crop_{uid}",
                "annotation_uid": uid,
                "source_image_id": f"image_{uid}",
                "source_relative_path": f"images/{uid}.png",
                "source_width": 4,
                "source_height": 4,
                "source_checksum_sha256": "0" * 64,
                "class_id": 0,
                "class_name": "HM",
                "major_class": "ship",
                "crop_policy": "tight",
                "crop_x0": 0,
                "crop_y0": 0,
                "crop_x1": 4,
                "crop_y1": 4,
                "fold": fold,
                "leakage_group_id": f"g{fold}",
                "color_mode": "RGB",
                "outside_policy": "pad",
                "resize_semantics": "direct_square_resize",
            }
        )
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_mock_extraction_writes_resumable_d4_cache(tmp_path: Path) -> None:
    script = _load_script()
    manifest = _fixture(tmp_path)
    output = tmp_path / "cache"
    arguments = [
        "--manifest",
        str(manifest),
        "--data-root",
        str(tmp_path),
        "--output-dir",
        str(output),
        "--teacher",
        "mock",
        "--views",
        "d4",
        "--batch-size",
        "5",
        "--shard-size",
        "7",
        "--device",
        "cpu",
    ]
    assert script.main(arguments) == 0
    cache = FeatureCache(output)
    assert cache.audit()["row_count"] == 24
    summary = json.loads((output / "extraction_summary.json").read_text(encoding="utf-8"))
    assert summary["object_count"] == 3
    assert summary["feature_names"] == ["mock_stats"]
    original_summary = (output / "extraction_summary.json").read_bytes()
    assert script.main(arguments) == 0
    assert (output / "extraction_summary.json").read_bytes() == original_summary
    resume = json.loads((output / "resume_check_summary.json").read_text(encoding="utf-8"))
    assert resume["computed_rows_this_invocation"] == 0
    assert resume["skipped_rows_this_invocation"] == 24


def test_cache_audit_expected_shape_is_a_hard_gate(tmp_path: Path) -> None:
    extract = _load_script()
    audit = _load_script("audit_p04_feature_cache.py")
    manifest = _fixture(tmp_path)
    output = tmp_path / "cache"
    assert (
        extract.main(
            [
                "--manifest",
                str(manifest),
                "--data-root",
                str(tmp_path),
                "--output-dir",
                str(output),
                "--teacher",
                "mock",
                "--views",
                "identity",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (
        audit.main(
            [
                "--cache-dir",
                str(output),
                "--expected-rows",
                "3",
                "--expected-feature",
                "mock_stats=8",
            ]
        )
        == 0
    )
    assert (
        audit.main(
            [
                "--cache-dir",
                str(output),
                "--expected-rows",
                "4",
                "--expected-feature",
                "mock_stats=8",
            ]
        )
        == 2
    )
