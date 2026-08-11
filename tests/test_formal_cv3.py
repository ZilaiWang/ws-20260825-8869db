"""Common formal-CV3 consumer gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsdet.data.formal_cv3 import (
    formal_cv3_audit_payload,
    load_formal_cv3_manifest,
    sha256_file,
    validate_written_view,
    write_formal_cv3_views,
)


def _payload() -> dict[str, object]:
    samples = []
    image_id = 1
    for fold, count in enumerate((2, 2, 2)):
        for offset in range(count):
            samples.append(
                {
                    "image_id": image_id,
                    "relative_path": f"images/train/image_{image_id}.jpg",
                    "fold": fold,
                    "group_id": f"group_{fold}_{offset}",
                    "group_rule": "test_source_group",
                }
            )
            image_id += 1
    return {
        "version": "formal_test_v1",
        "data_version": "test_data_v1",
        "fold_count": 3,
        "scientific_contract": {
            "group_indivisible": True,
            "every_image_validates_exactly_once": True,
            "every_class_has_validation_source_in_every_fold": True,
        },
        "samples": samples,
    }


def _write(path: Path, payload: dict[str, object]) -> str:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def _load(path: Path, digest: str):
    return load_formal_cv3_manifest(
        path,
        expected_sha256=digest,
        expected_version="formal_test_v1",
        expected_data_version="test_data_v1",
        expected_fold_count=3,
        expected_sample_count=6,
        expected_group_count=6,
        expected_fold_image_counts=(2, 2, 2),
        expected_fold_group_counts=(2, 2, 2),
    )


def test_consumer_builds_complete_group_isolated_views(tmp_path: Path) -> None:
    path = tmp_path / "cv3.json"
    manifest = _load(path, _write(path, _payload()))

    for held_out_fold in range(3):
        view = manifest.view(held_out_fold)
        assert len(view.train) == 4
        assert len(view.val) == 2
        assert {item.group_id for item in view.train}.isdisjoint(item.group_id for item in view.val)

    audit = formal_cv3_audit_payload(manifest)
    assert audit["formal_cv3_admission"] is True
    assert audit["group_cross_fold_count"] == 0
    assert audit["fold_image_counts"] == [2, 2, 2]


def test_consumer_writes_deterministic_framework_neutral_views(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cv3.json"
    manifest = _load(path, _write(path, _payload()))
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_hashes = write_formal_cv3_views(manifest, first)
    second_hashes = write_formal_cv3_views(manifest, second)

    assert first_hashes == second_hashes
    for held_out_fold in range(3):
        for split in ("train", "val"):
            filename = f"formal_cv3_fold{held_out_fold}_{split}.csv"
            assert (first / filename).read_bytes() == (second / filename).read_bytes()
            validate_written_view(
                first / filename,
                manifest=manifest,
                held_out_fold=held_out_fold,
                split=split,
            )


def test_consumer_refuses_to_overwrite_accepted_views(tmp_path: Path) -> None:
    path = tmp_path / "cv3.json"
    manifest = _load(path, _write(path, _payload()))
    output = tmp_path / "views"
    write_formal_cv3_views(manifest, output)

    with pytest.raises(FileExistsError, match="immutable"):
        write_formal_cv3_views(manifest, output)


def test_consumer_rejects_sha_mismatch_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "cv3.json"
    _write(path, _payload())

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _load(path, "0" * 64)


def test_consumer_rejects_cross_fold_group_even_with_matching_sha(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["samples"][1]["group_id"] = payload["samples"][2]["group_id"]
    path = tmp_path / "cv3.json"
    digest = _write(path, payload)

    with pytest.raises(ValueError, match="cross-fold groups"):
        _load(path, digest)


def test_consumer_rejects_non_contiguous_image_coverage(tmp_path: Path) -> None:
    payload = _payload()
    payload["samples"][-1]["image_id"] = 99
    path = tmp_path / "cv3.json"
    digest = _write(path, payload)

    with pytest.raises(ValueError, match="image_id coverage mismatch"):
        _load(path, digest)
