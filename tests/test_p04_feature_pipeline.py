"""P04 输入、缓存、manifest 对齐和稳定性的无 GPU 测试。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rsdet.analysis.p04_features import (
    analyze_raw_ensemble,
    compare_cache_overlap,
    compare_repeat_caches,
)
from rsdet.features.p04_cache import FeatureCache, FeatureCacheWriter
from rsdet.features.p04_inputs import (
    D4_VIEW_IDS,
    apply_d4_view,
    canonical_image_sha256,
    load_all_crop_records,
)
from rsdet.features.p04_probe import deterministic_view_index, load_probe_data


def _manifest(path: Path) -> None:
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
    for uid, fold in (("a", 0), ("b", 1), ("c", 2)):
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
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_cache(
    path: Path,
    *,
    feature_names: tuple[str, ...],
    rows: dict[str, list],
    features: dict[str, np.ndarray],
    metadata: dict | None = None,
) -> FeatureCache:
    writer = FeatureCacheWriter(
        path,
        metadata=metadata or {"teacher": "unit-test"},
        feature_names=feature_names,
        storage_dtype="float32",
    )
    writer.write_shard(0, rows, features)
    writer.finalize(1, len(rows["annotation_uid"]))
    return FeatureCache(path)


def test_d4_views_are_complete_and_canonical_hash_is_pixel_sensitive() -> None:
    array = np.zeros((3, 3, 3), dtype=np.uint8)
    for y in range(3):
        for x in range(3):
            array[y, x] = (y * 30 + x, y * 20 + 2 * x, y * 10 + 3 * x)
    image = Image.fromarray(array, mode="RGB")
    variants = {apply_d4_view(image, view).tobytes() for view in D4_VIEW_IDS}
    assert len(variants) == 8
    assert canonical_image_sha256(image) == canonical_image_sha256(image.copy())
    changed = image.copy()
    changed.putpixel((0, 0), (255, 0, 0))
    assert canonical_image_sha256(image) != canonical_image_sha256(changed)


def test_cache_writer_resume_audit_and_feature_load(tmp_path: Path) -> None:
    rows = {
        "annotation_uid": ["a", "b"],
        "crop_id": ["crop_a", "crop_b"],
        "canonical_input_sha256": ["1" * 64, "2" * 64],
        "view_id": ["r0", "r0"],
    }
    features = {"f": np.asarray([[1, 2], [3, 4]], dtype=np.float32)}
    cache = _write_cache(
        tmp_path / "cache", feature_names=("f",), rows=rows, features=features
    )
    assert cache.audit()["status"] == "pass"
    assert np.array_equal(cache.load_feature("f")["f"], features["f"])
    writer = FeatureCacheWriter(
        tmp_path / "cache",
        metadata={"teacher": "unit-test"},
        feature_names=("f",),
        storage_dtype="float32",
    )
    assert writer.valid_existing_shard(0, 2)


def test_repeat_and_subset_overlap_compare_use_annotation_view_keys(
    tmp_path: Path,
) -> None:
    full_rows = {
        "annotation_uid": ["a", "a", "b"],
        "crop_id": ["crop_a", "crop_a", "crop_b"],
        "canonical_input_sha256": ["1" * 64, "1" * 64, "2" * 64],
        "view_id": ["r0", "r90", "r0"],
    }
    full_features = {
        "f": np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    }
    _write_cache(
        tmp_path / "full-a",
        feature_names=("f",),
        rows=full_rows,
        features=full_features,
    )
    _write_cache(
        tmp_path / "full-b",
        feature_names=("f",),
        rows=full_rows,
        features=full_features,
    )
    subset_rows = {key: values[:2] for key, values in full_rows.items()}
    _write_cache(
        tmp_path / "subset",
        feature_names=("f",),
        rows=subset_rows,
        features={"f": full_features["f"][:2]},
    )

    repeat = compare_repeat_caches(str(tmp_path / "full-a"), str(tmp_path / "full-b"))
    assert repeat["status"] == "pass"
    assert repeat["features"]["f"]["common_row_count"] == 3
    overlap = compare_cache_overlap(str(tmp_path / "full-a"), str(tmp_path / "subset"))
    assert overlap["status"] == "pass"
    assert overlap["features"]["f"]["common_row_count"] == 2

    with pytest.raises(ValueError, match="行 key 不一致"):
        compare_repeat_caches(str(tmp_path / "full-a"), str(tmp_path / "subset"))


def test_cache_audit_rejects_cross_view_input_contract_drift(tmp_path: Path) -> None:
    rows = {
        "annotation_uid": ["a", "a"],
        "crop_id": ["crop_a", "crop_a"],
        "canonical_input_sha256": ["1" * 64, "2" * 64],
        "view_id": ["r0", "r90"],
    }
    cache = _write_cache(
        tmp_path / "drift",
        feature_names=("f",),
        rows=rows,
        features={"f": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="跨视图 crop/input 合同不一致"):
        cache.audit()


def test_probe_alignment_uses_manifest_fold_and_rejects_changed_crop(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest)
    records = load_all_crop_records(manifest)
    rows = {
        "annotation_uid": [item.annotation_uid for item in records],
        "crop_id": [item.crop_id for item in records],
        "canonical_input_sha256": ["1" * 64] * 3,
        "view_id": ["r0"] * 3,
    }
    _write_cache(
        tmp_path / "cache",
        feature_names=("f",),
        rows=rows,
        features={"f": np.eye(3, dtype=np.float32)},
    )
    data = load_probe_data(tmp_path / "cache", feature_name="f", manifest_path=manifest)
    assert data.folds.tolist() == [0, 1, 2]
    assert data.class_ids.tolist() == [0, 0, 0]
    assert data.features.shape == (3, 1, 3)

    text = manifest.read_text(encoding="utf-8").replace("crop_a", "crop_a_changed")
    changed = tmp_path / "changed.csv"
    changed.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="crop_id"):
        load_probe_data(tmp_path / "cache", feature_name="f", manifest_path=changed)


def test_probe_rejects_unverified_cross_manifest_cache_reuse(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest)
    records = load_all_crop_records(manifest)
    rows = {
        "annotation_uid": [item.annotation_uid for item in records],
        "crop_id": [item.crop_id for item in records],
        "canonical_input_sha256": ["1" * 64] * 3,
        "view_id": ["r0"] * 3,
    }
    _write_cache(
        tmp_path / "cache-mismatch",
        feature_names=("f",),
        rows=rows,
        features={"f": np.eye(3, dtype=np.float32)},
        metadata={"manifest_sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="P04-TASK-05"):
        load_probe_data(
            tmp_path / "cache-mismatch", feature_name="f", manifest_path=manifest
        )


def test_raw_ensemble_gate_uses_nested_e4_and_e8(tmp_path: Path) -> None:
    names = tuple(
        f"raw_{location}_e{size}"
        for location in ("map0_t100", "map6_t261")
        for size in (1, 4, 8)
    )
    rows = {
        "annotation_uid": ["a", "b"],
        "crop_id": ["crop_a", "crop_b"],
        "canonical_input_sha256": ["1" * 64, "2" * 64],
        "view_id": ["r0", "r0"],
    }
    base = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    features = {name: base.copy() for name in names}
    _write_cache(
        tmp_path / "raw", feature_names=names, rows=rows, features=features
    )
    report = analyze_raw_ensemble(str(tmp_path / "raw"))
    assert report["ensemble4_gate"]["status"] == "pass"
    assert report["comparisons"]["map0_t100_e4_vs_e8"]["median"] == pytest.approx(1.0)


def test_view_selection_is_stable_and_epoch_dependent() -> None:
    first = [deterministic_view_index("uid", epoch, 42, 8) for epoch in range(20)]
    second = [deterministic_view_index("uid", epoch, 42, 8) for epoch in range(20)]
    assert first == second
    assert len(set(first)) > 1
