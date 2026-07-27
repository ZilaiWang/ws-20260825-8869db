from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from rsdet.experiments.detection_data_lock import (
    LOCK_VERSION,
    atomic_write_new_json,
    build_lock_payload,
    canonical_json_sha256,
    load_and_validate_spec,
    sha256_file,
    verify_existing_lock,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    image_dir = data_root / "images" / "train"
    label_dir = data_root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image_specs = [
        ("a.jpg", (100, 80), 0, (10, 20, 30, 40)),
        ("b.jpg", (120, 100), 1, (30, 10, 90, 70)),
    ]
    source_rows = []
    for image_index, (filename, size, class_id, box) in enumerate(image_specs, start=1):
        image_path = image_dir / filename
        Image.new("RGB", size, color=(20 * image_index, 40, 60)).save(
            image_path,
            format="JPEG",
            quality=95,
        )
        width, height = size
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / (2 * width)
        cy = (y0 + y1) / (2 * height)
        box_width = (x1 - x0) / width
        box_height = (y1 - y0) / height
        (label_dir / Path(filename).with_suffix(".txt")).write_text(
            f"{class_id} {cx:.6f} {cy:.6f} {box_width:.6f} {box_height:.6f}\n",
            encoding="utf-8",
        )
        image_relative = f"images/train/{filename}"
        common = {
            "annotation_uid": f"ann-{image_index}",
            "source_image_id": f"source-{image_index}",
            "source_relative_path": image_relative,
            "source_width": str(width),
            "source_height": str(height),
            "source_checksum_sha256": sha256_file(image_path),
            "class_id": str(class_id),
            "class_name": f"class-{class_id}",
            "major_class": "aircraft",
            "modality": "rgb",
            "gt_x0": str(x0),
            "gt_y0": str(y0),
            "gt_x1": str(x1),
            "gt_y1": str(y1),
            "gt_width": str(x1 - x0),
            "gt_height": str(y1 - y0),
            "gt_short_edge": str(min(x1 - x0, y1 - y0)),
        }
        source_rows.append(common)

    p02_rows: list[dict[str, str]] = []
    for source in source_rows:
        for policy in ("tight", "context_1p25", "jitter_light"):
            p02_rows.append(
                {
                    "crop_id": f"{source['annotation_uid']}--{policy}",
                    **source,
                    "crop_policy": policy,
                }
            )
    p02_path = tmp_path / "crop_manifest.csv"
    _write_csv(p02_path, p02_rows)
    p02_sha = sha256_file(p02_path)
    formal_rows = [
        {**row, "source_crop_manifest_sha256": p02_sha} for row in p02_rows
    ]
    formal_path = tmp_path / "formal_crop_manifest.csv"
    _write_csv(formal_path, formal_rows)

    cv3_payload = {
        "version": "cv3-test",
        "data_version": "data-test",
        "fold_count": 2,
        "scientific_contract": {
            "group_indivisible": True,
            "every_image_validates_exactly_once": True,
            "every_class_has_validation_source_in_every_fold": True,
        },
        "samples": [
            {
                "image_id": index,
                "relative_path": f"images/train/{filename}",
                "fold": index - 1,
                "group_id": f"group-{index}",
                "group_rule": "test",
            }
            for index, (filename, *_rest) in enumerate(image_specs, start=1)
        ],
    }
    cv3_path = tmp_path / "cv3.json"
    cv3_path.write_text(
        json.dumps(cv3_payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    spec = {
        "schema_version": "formal_detection_data_lock_spec_v1",
        "cv3_manifest": {
            "sha256": sha256_file(cv3_path),
            "version": "cv3-test",
            "data_version": "data-test",
            "fold_count": 2,
            "sample_count": 2,
            "source_group_count": 2,
            "fold_image_counts": [1, 1],
            "fold_group_counts": [1, 1],
        },
        "p02_manifest": {
            "sha256": p02_sha,
            "row_count": 6,
            "annotation_count": 2,
        },
        "formal_crop_manifest": {
            "sha256": sha256_file(formal_path),
            "row_count": 6,
            "annotation_count": 2,
        },
        "dataset": {
            "image_prefix": "images",
            "label_prefix": "labels",
            "class_count": 2,
            "object_count": 2,
            "crop_policies": ["tight", "context_1p25", "jitter_light"],
            "tight_policy": "tight",
            "coordinate_tolerance": 5e-6,
        },
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return {
        "data_root": data_root,
        "cv3": cv3_path,
        "p02": p02_path,
        "formal": formal_path,
        "spec": spec_path,
    }


def _build(paths: dict[str, Path]) -> dict:
    return build_lock_payload(
        spec_path=paths["spec"],
        data_root=paths["data_root"],
        cv3_manifest_path=paths["cv3"],
        p02_manifest_path=paths["p02"],
        formal_crop_manifest_path=paths["formal"],
    )


def test_build_write_and_verify_data_lock(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    payload = _build(paths)
    assert payload["schema_version"] == LOCK_VERSION
    assert payload["summary"]["image_count"] == 2
    assert payload["summary"]["object_count"] == 2
    assert payload["summary"]["class_counts"] == {"0": 1, "1": 1}
    assert payload["summary"]["p02_formal_gt_equivalence"] is True
    assert payload["summary"]["yolo_formal_gt_equivalence"] is True
    body = dict(payload)
    fingerprint = body.pop("lock_fingerprint")
    assert fingerprint == canonical_json_sha256(body)

    lock_path = tmp_path / "FORMAL_DETECTION_DATA_LOCK.json"
    atomic_write_new_json(lock_path, payload)
    assert lock_path.stat().st_mode & 0o222 == 0
    report = verify_existing_lock(
        spec_path=paths["spec"],
        data_root=paths["data_root"],
        cv3_manifest_path=paths["cv3"],
        p02_manifest_path=paths["p02"],
        formal_crop_manifest_path=paths["formal"],
        lock_path=lock_path,
        expected_lock_sha256=sha256_file(lock_path),
    )
    assert report["status"] == "pass"
    assert report["image_count"] == 2
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        atomic_write_new_json(lock_path, payload)


def test_image_or_label_byte_drift_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    payload = _build(paths)
    lock_path = tmp_path / "lock.json"
    atomic_write_new_json(lock_path, payload)

    label = paths["data_root"] / "labels/train/a.txt"
    label.write_text("0 0.3 0.4 0.21 0.25\n", encoding="utf-8")
    with pytest.raises(ValueError, match="coordinates differ"):
        verify_existing_lock(
            spec_path=paths["spec"],
            data_root=paths["data_root"],
            cv3_manifest_path=paths["cv3"],
            p02_manifest_path=paths["p02"],
            formal_crop_manifest_path=paths["formal"],
            lock_path=lock_path,
        )

    paths = _fixture(tmp_path / "image-drift")
    image = paths["data_root"] / "images/train/a.jpg"
    image.write_bytes(image.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="image bytes differ"):
        _build(paths)


def test_p02_and_formal_gt_difference_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with paths["formal"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["class_name"] = "wrong-class"
    _write_csv(paths["formal"], rows)
    spec = json.loads(paths["spec"].read_text(encoding="utf-8"))
    spec["formal_crop_manifest"]["sha256"] = sha256_file(paths["formal"])
    paths["spec"].write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="P0-2/formal GT differs"):
        _build(paths)


def test_writable_or_tampered_lock_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    lock_path = tmp_path / "lock.json"
    atomic_write_new_json(lock_path, _build(paths))
    lock_path.chmod(0o644)
    with pytest.raises(PermissionError, match="write permission"):
        verify_existing_lock(
            spec_path=paths["spec"],
            data_root=paths["data_root"],
            cv3_manifest_path=paths["cv3"],
            p02_manifest_path=paths["p02"],
            formal_crop_manifest_path=paths["formal"],
            lock_path=lock_path,
        )
    lock_path.write_text('{"schema_version":"tampered"}\n', encoding="utf-8")
    lock_path.chmod(0o444)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_existing_lock(
            spec_path=paths["spec"],
            data_root=paths["data_root"],
            cv3_manifest_path=paths["cv3"],
            p02_manifest_path=paths["p02"],
            formal_crop_manifest_path=paths["formal"],
            lock_path=lock_path,
        )


def test_project_spec_freezes_all_formal_dataset_counts() -> None:
    spec = load_and_validate_spec(
        Path(__file__).parents[1]
        / "configs"
        / "experiments"
        / "formal_detection_data_lock.json"
    )
    assert spec["cv3_manifest"]["sample_count"] == 4481
    assert spec["cv3_manifest"]["source_group_count"] == 255
    assert spec["p02_manifest"]["row_count"] == 62799
    assert spec["formal_crop_manifest"]["annotation_count"] == 20933
    assert spec["dataset"]["class_count"] == 25
    assert spec["dataset"]["object_count"] == 20933
