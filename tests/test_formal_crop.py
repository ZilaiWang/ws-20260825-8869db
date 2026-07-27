"""Metadata-only formal crop-manifest rehang tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rsdet.analysis.formal_crop import (
    HISTORICAL_ASSIGNMENT_COLUMNS,
    build_formal_crop_manifest,
)
from rsdet.data.formal_cv3 import load_formal_cv3_manifest, sha256_file
from rsdet.data.xh_dataset import FINE_NAMES, coarse_name

POLICIES = ("tight", "context_1p25", "jitter_light")


def _write_cv3(path: Path) -> str:
    payload = {
        "version": "formal_test_v1",
        "data_version": "test_data_v1",
        "fold_count": 3,
        "scientific_contract": {
            "group_indivisible": True,
            "every_image_validates_exactly_once": True,
            "every_class_has_validation_source_in_every_fold": True,
        },
        "samples": [
            {
                "image_id": fold + 1,
                "relative_path": f"images/train/source_{fold}.jpg",
                "fold": fold,
                "group_id": f"formal_group_{fold}",
                "group_rule": "test_group",
            }
            for fold in range(3)
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def _load_cv3(path: Path, digest: str):
    return load_formal_cv3_manifest(
        path,
        expected_sha256=digest,
        expected_version="formal_test_v1",
        expected_data_version="test_data_v1",
        expected_fold_count=3,
        expected_sample_count=3,
        expected_group_count=3,
        expected_fold_image_counts=(1, 1, 1),
        expected_fold_group_counts=(1, 1, 1),
    )


def _write_exploratory(path: Path) -> str:
    fieldnames = [
        "manifest_version",
        "schema_version",
        "crop_id",
        "annotation_uid",
        "source_image_id",
        "source_relative_path",
        "source_width",
        "source_height",
        "source_checksum_sha256",
        "original_main_split",
        "original_fold",
        "main_split",
        "fold",
        "main_split_changed",
        "fold_changed",
        "main_assignment_reason",
        "fold_assignment_reason",
        "group_id",
        "leakage_group_id",
        "leakage_group_image_count",
        "estimated_group_uid",
        "estimated_group_count_in_leakage_group",
        "group_basis",
        "group_confidence",
        "near_duplicate_edge_count",
        "class_id",
        "class_name",
        "major_class",
        "crop_policy",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
        "color_mode",
        "outside_policy",
        "resize_semantics",
    ]
    rows = []
    for fold in range(3):
        for class_id, class_name in enumerate(FINE_NAMES):
            annotation_uid = f"ann_{fold}_{class_id}"
            for policy in POLICIES:
                scale = {
                    "tight": 1.0,
                    "context_1p25": 1.25,
                    "jitter_light": 1.0,
                }[policy]
                rows.append(
                    {
                        "manifest_version": "exploratory_test_v1",
                        "schema_version": "crop_manifest_v1",
                        "crop_id": f"crop_{fold}_{class_id}_{policy}",
                        "annotation_uid": annotation_uid,
                        "source_image_id": f"audit_image_{fold}",
                        "source_relative_path": f"images/train/source_{fold}.jpg",
                        "source_width": "100",
                        "source_height": "100",
                        "source_checksum_sha256": str(fold) * 64,
                        "original_main_split": "train",
                        "original_fold": str((fold + 1) % 3),
                        "main_split": "train",
                        "fold": str((fold + 1) % 3),
                        "main_split_changed": "False",
                        "fold_changed": "False",
                        "main_assignment_reason": "exploratory",
                        "fold_assignment_reason": "exploratory",
                        "group_id": f"old_group_{fold}",
                        "leakage_group_id": f"old_group_{fold}",
                        "leakage_group_image_count": "1",
                        "estimated_group_uid": f"estimated_{fold}",
                        "estimated_group_count_in_leakage_group": "1",
                        "group_basis": "exploratory",
                        "group_confidence": "low",
                        "near_duplicate_edge_count": "0",
                        "class_id": str(class_id),
                        "class_name": class_name,
                        "major_class": coarse_name(class_id),
                        "crop_policy": policy,
                        "crop_x0": "10",
                        "crop_y0": "20",
                        "crop_x1": str(10 + 40 * scale),
                        "crop_y1": str(20 + 40 * scale),
                        "color_mode": "RGB",
                        "outside_policy": "pad",
                        "resize_semantics": "direct_square_resize",
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _config(source_sha256: str) -> dict[str, object]:
    return {
        "version": "formal_crop_manifest_v2",
        "schema_version": "formal_crop_manifest_v2",
        "source_manifest_version": "exploratory_test_v1",
        "source_manifest_schema_version": "crop_manifest_v1",
        "source_manifest_sha256": source_sha256,
        "expected_rows": 225,
        "expected_annotations": 75,
        "expected_source_images": 3,
        "expected_policies": list(POLICIES),
        "expected_fold_object_counts": [25, 25, 25],
        "preserve_crop_id": True,
        "preserve_non_assignment_fields": True,
        "pixel_io_policy": "forbidden_metadata_only",
    }


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_rehang_preserves_crop_identity_and_uses_only_formal_assignment(
    tmp_path: Path,
) -> None:
    cv3_path = tmp_path / "cv3.json"
    formal_cv3 = _load_cv3(cv3_path, _write_cv3(cv3_path))
    exploratory = tmp_path / "crop.csv"
    source_sha = _write_exploratory(exploratory)
    output = tmp_path / "formal"

    audit = build_formal_crop_manifest(
        exploratory_manifest_path=exploratory,
        formal_cv3=formal_cv3,
        config=_config(source_sha),
        output_dir=output,
    )

    source_rows = _read(exploratory)
    formal_rows = _read(output / "formal_crop_manifest.csv")
    assert audit["formal_crop_admission"] is True
    assert audit["pixels_read"] == 0
    assert audit["geometry_recomputed"] is False
    assert audit["historical_assignment_fields_preserved"] is True
    assert [row["crop_id"] for row in formal_rows] == [
        row["crop_id"] for row in source_rows
    ]
    assert [row["crop_x0"] for row in formal_rows] == [
        row["crop_x0"] for row in source_rows
    ]
    assert "historical_p02_fold" in formal_rows[0]
    assert "historical_p02_group_id" in formal_rows[0]
    assert "main_split" not in formal_rows[0]
    for source_row, row in zip(source_rows, formal_rows):
        for field in HISTORICAL_ASSIGNMENT_COLUMNS:
            assert row[f"historical_p02_{field}"] == source_row[field]
        fold = int(row["formal_image_id"]) - 1
        assert int(row["fold"]) == fold
        assert row["group_id"] == f"formal_group_{fold}"
        assert row["leakage_group_id"] == row["group_id"]
        assert row["manifest_version"] == "formal_crop_manifest_v2"
        assert row["assignment_scope"] == "formal_cv3_fold_only"


def test_rehang_is_byte_deterministic(tmp_path: Path) -> None:
    cv3_path = tmp_path / "cv3.json"
    formal_cv3 = _load_cv3(cv3_path, _write_cv3(cv3_path))
    exploratory = tmp_path / "crop.csv"
    source_sha = _write_exploratory(exploratory)

    for output in (tmp_path / "first", tmp_path / "second"):
        build_formal_crop_manifest(
            exploratory_manifest_path=exploratory,
            formal_cv3=formal_cv3,
            config=_config(source_sha),
            output_dir=output,
        )

    assert (tmp_path / "first/formal_crop_manifest.csv").read_bytes() == (
        tmp_path / "second/formal_crop_manifest.csv"
    ).read_bytes()


def test_rehang_refuses_to_overwrite_accepted_output(tmp_path: Path) -> None:
    cv3_path = tmp_path / "cv3.json"
    formal_cv3 = _load_cv3(cv3_path, _write_cv3(cv3_path))
    exploratory = tmp_path / "crop.csv"
    source_sha = _write_exploratory(exploratory)
    output = tmp_path / "formal"
    kwargs = {
        "exploratory_manifest_path": exploratory,
        "formal_cv3": formal_cv3,
        "config": _config(source_sha),
        "output_dir": output,
    }

    build_formal_crop_manifest(**kwargs)
    with pytest.raises(FileExistsError, match="immutable"):
        build_formal_crop_manifest(**kwargs)


def test_rehang_rejects_source_sha_mismatch_without_output(tmp_path: Path) -> None:
    cv3_path = tmp_path / "cv3.json"
    formal_cv3 = _load_cv3(cv3_path, _write_cv3(cv3_path))
    exploratory = tmp_path / "crop.csv"
    _write_exploratory(exploratory)
    output = tmp_path / "formal"

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_formal_crop_manifest(
            exploratory_manifest_path=exploratory,
            formal_cv3=formal_cv3,
            config=_config("0" * 64),
            output_dir=output,
        )
    assert not (output / "formal_crop_manifest.csv").exists()


def test_rehang_rejects_source_path_not_in_cv3(tmp_path: Path) -> None:
    cv3_path = tmp_path / "cv3.json"
    formal_cv3 = _load_cv3(cv3_path, _write_cv3(cv3_path))
    exploratory = tmp_path / "crop.csv"
    _write_exploratory(exploratory)
    rows = _read(exploratory)
    rows[0]["source_relative_path"] = "images/train/unknown.jpg"
    with exploratory.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "formal"

    with pytest.raises(ValueError, match="absent from formal CV3"):
        build_formal_crop_manifest(
            exploratory_manifest_path=exploratory,
            formal_cv3=formal_cv3,
            config=_config(sha256_file(exploratory)),
            output_dir=output,
        )
    assert not (output / "formal_crop_manifest.csv").exists()
