"""MAR20 分组批次 A：registry、背景视图和缓存合同。"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rsdet.grouping.cache import PlaceFeatureCache, PlaceFeatureCacheWriter
from rsdet.grouping.contracts import (
    STRICT_LABELS,
    EvidenceLabel,
    canonical_node_uid,
    canonical_pair_uid,
    canonical_pixel_sha256,
)
from rsdet.grouping.descriptors import MockPlaceEncoder
from rsdet.grouping.masks import (
    apply_mask_fill,
    build_foreground_mask,
    render_place_inputs,
    select_background_tiles,
)
from rsdet.grouping.registry import build_registry, load_annotations, load_registry, write_registry


def _xml(number: int, *, width: int, height: int, class_name: str = "A1") -> str:
    return f"""<annotation>
<filename>{number}.jpg</filename>
<size><width>{width}</width><height>{height}</height><depth>3</depth></size>
<object><name>{class_name}</name><bndbox>
<xmin>10</xmin><ymin>12</ymin><xmax>30</xmax><ymax>32</ymax>
</bndbox></object></annotation>"""


def _mini_dataset(tmp_path: Path) -> tuple[Path, Path]:
    competition = tmp_path / "competition"
    mar20 = tmp_path / "MAR20"
    for path in (
        competition / "images/train",
        competition / "labels/train",
        competition / "images/val",
        competition / "labels/val",
        mar20 / "JPEGImages",
        mar20 / "Annotations/Horizontal Bounding Boxes",
        mar20 / "ImageSets/Main",
    ):
        path.mkdir(parents=True, exist_ok=True)
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    array[..., 0] = np.arange(64, dtype=np.uint8)[None, :]
    array[..., 1] = np.arange(64, dtype=np.uint8)[:, None]
    image1 = mar20 / "JPEGImages/1.jpg"
    Image.fromarray(array, mode="RGB").save(image1, quality=95)
    array[..., 2] = 90
    Image.fromarray(array, mode="RGB").save(mar20 / "JPEGImages/2.jpg", quality=95)
    shutil.copyfile(image1, competition / "images/train/MAR20_1.jpg")
    # class=4(A1), bbox=(10,12,30,32) in 64x64.
    (competition / "labels/train/MAR20_1.txt").write_text(
        "4 0.3125 0.34375 0.3125 0.3125\n", encoding="utf-8"
    )
    (mar20 / "Annotations/Horizontal Bounding Boxes/1.xml").write_text(
        _xml(1, width=64, height=64), encoding="utf-8"
    )
    # 第二张故意保留原数据中真实存在的 XML 0x0 尺寸问题。
    (mar20 / "Annotations/Horizontal Bounding Boxes/2.xml").write_text(
        _xml(2, width=0, height=0), encoding="utf-8"
    )
    (mar20 / "ImageSets/Main/train.txt").write_text("1\n", encoding="utf-8")
    (mar20 / "ImageSets/Main/test.txt").write_text("2\n", encoding="utf-8")
    return competition, mar20


def test_evidence_and_pair_contract_prevent_likely_union() -> None:
    assert canonical_node_uid(9) == "mar20:9"
    assert canonical_pair_uid("mar20:20", "mar20:3") == "mar20:3--mar20:20"
    assert EvidenceLabel.LIKELY_SAME_AIRPORT not in STRICT_LABELS
    assert EvidenceLabel.SAME_LOCAL_SITE in STRICT_LABELS
    with pytest.raises(ValueError, match="不得相同"):
        canonical_pair_uid("mar20:1", "mar20:1")


def test_registry_maps_target_and_records_xml_size_fallback(tmp_path: Path) -> None:
    competition, mar20 = _mini_dataset(tmp_path)
    records, annotations, summary = build_registry(
        competition_root=competition,
        mar20_root=mar20,
        expected_all=2,
        expected_target=1,
        expected_official_train=1,
        expected_official_test=1,
    )
    assert summary["status"] == "pass"
    assert summary["counts"] == {
        "all": 2,
        "target": 1,
        "bridge": 1,
        "official_train": 1,
        "official_test": 1,
    }
    assert summary["xml_size_missing_count"] == 1
    target = next(item for item in records if item.is_target)
    assert target.file_bytes_equal and target.pixel_equal
    assert target.annotation_class_hist_equal
    output = tmp_path / "registry"
    write_registry(output, records, annotations, summary)
    assert len(load_registry(output / "image_registry.csv", expected_rows=2)) == 2
    assert len(load_annotations(output / "image_annotations.jsonl", expected_rows=2)) == 2


def test_registry_reports_pixel_mismatch_instead_of_silent_mapping(tmp_path: Path) -> None:
    competition, mar20 = _mini_dataset(tmp_path)
    changed = np.full((64, 64, 3), 255, dtype=np.uint8)
    Image.fromarray(changed, mode="RGB").save(competition / "images/train/MAR20_1.jpg")
    _, _, summary = build_registry(
        competition_root=competition,
        mar20_root=mar20,
        expected_all=2,
        expected_target=1,
        expected_official_train=1,
        expected_official_test=1,
    )
    assert summary["status"] == "fail"
    assert summary["target_alignment"]["pixel_mismatch_count"] == 1


def test_mask_fill_preserves_every_background_pixel() -> None:
    array = np.arange(80 * 80 * 3, dtype=np.uint32).reshape(80, 80, 3) % 255
    image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    mask = build_foreground_mask(image.size, [(25, 25, 45, 45)], dilation_ratio=0.10)
    filled = apply_mask_fill(image, mask, method="local_mean")
    foreground = np.asarray(mask) > 0
    before = np.asarray(image)
    after = np.asarray(filled)
    assert np.array_equal(before[~foreground], after[~foreground])
    assert not np.array_equal(before[foreground], after[foreground])
    assert canonical_pixel_sha256(image) != canonical_pixel_sha256(filled)


def test_background_tiles_respect_valid_fraction_and_are_deterministic() -> None:
    image = Image.new("RGB", (96, 96), (10, 20, 30))
    mask = build_foreground_mask(image.size, [(32, 32, 64, 64)], dilation_ratio=0.0)
    first = select_background_tiles(
        image,
        mask,
        tile_size=32,
        stride=16,
        min_valid_fraction=1.0,
        max_tiles=5,
    )
    second = select_background_tiles(
        image,
        mask,
        tile_size=32,
        stride=16,
        min_valid_fraction=1.0,
        max_tiles=5,
    )
    assert [item[1] for item in first] == [item[1] for item in second]
    assert len(first) == 5
    foreground = np.asarray(mask) > 0
    for _, (x1, y1, x2, y2), valid in first:
        assert valid == 1.0
        assert not foreground[y1:y2, x1:x2].any()


def test_render_place_inputs_and_mock_encoder() -> None:
    image = Image.new("RGB", (96, 96), (100, 110, 120))
    rendered = render_place_inputs(
        node_uid="mar20:1",
        image=image,
        boxes=[(35, 35, 55, 55)],
        view_types=("original", "masked_inpaint", "background_tiles"),
        rotations=(0, 90),
        input_size=56,
        dilation_ratio=0.1,
        fill_method="local_mean",
        tile_size=32,
        tile_stride=16,
        tile_valid_fraction=1.0,
        max_tiles=2,
    )
    assert len(rendered) == 8
    assert all(item.image.size == (56, 56) for item in rendered)
    encoder = MockPlaceEncoder()
    features = encoder.extract([item.image for item in rendered[:3]])
    assert features["mock_rgb_stats"].shape == (3, 6)
    assert np.isfinite(features["mock_rgb_stats"]).all()


def test_place_feature_cache_resume_and_audit(tmp_path: Path) -> None:
    rows = {
        "node_uid": ["mar20:1", "mar20:2"],
        "view_type": ["masked_inpaint", "masked_inpaint"],
        "rotation": [0, 0],
        "item_index": [0, 0],
        "input_sha256": ["1" * 64, "2" * 64],
    }
    features = {"f": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)}
    writer = PlaceFeatureCacheWriter(
        tmp_path / "cache",
        metadata={"test": True},
        feature_names=("f",),
        storage_dtype="float32",
    )
    writer.write_shard(0, rows=rows, features=features)
    writer.finalize(expected_shards=1, expected_rows=2)
    cache = PlaceFeatureCache(tmp_path / "cache")
    assert cache.audit()["status"] == "pass"
    assert writer.valid_existing_shard(0, 2)
    loaded = cache.load_all()
    assert np.array_equal(loaded["feature__f"], features["f"])


def test_place_cache_rejects_duplicate_row_keys(tmp_path: Path) -> None:
    rows = {
        "node_uid": ["mar20:1", "mar20:1"],
        "view_type": ["original", "original"],
        "rotation": [0, 0],
        "item_index": [0, 0],
        "input_sha256": ["1" * 64, "1" * 64],
    }
    writer = PlaceFeatureCacheWriter(
        tmp_path / "cache",
        metadata={"test": True},
        feature_names=("f",),
        storage_dtype="float32",
    )
    writer.write_shard(
        0,
        rows=rows,
        features={"f": np.asarray([[1.0], [1.0]], dtype=np.float32)},
    )
    writer.finalize(expected_shards=1, expected_rows=2)
    with pytest.raises(ValueError, match="重复"):
        PlaceFeatureCache(tmp_path / "cache").audit()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_script(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{root / 'src'}{os.pathsep}{existing}" if existing else str(root / "src")
    )
    return subprocess.run(
        [sys.executable, str(root / "scripts" / script), *arguments],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_compile_blind_review_checks_repeat_and_writes_pair_contract(tmp_path: Path) -> None:
    mapping = [
        {
            "card_id": "CAL-0001",
            "pair_uid": "mar20:1--mar20:2",
            "node_u": "mar20:1",
            "node_v": "mar20:2",
            "route": "candidate",
            "duplicate_group": "mar20:1--mar20:2",
            "swapped": 0,
        },
        {
            "card_id": "CAL-0002",
            "pair_uid": "mar20:3--mar20:4",
            "node_u": "mar20:3",
            "node_v": "mar20:4",
            "route": "negative",
            "duplicate_group": "mar20:3--mar20:4",
            "swapped": 0,
        },
        {
            "card_id": "CAL-0003",
            "pair_uid": "mar20:1--mar20:2",
            "node_u": "mar20:2",
            "node_v": "mar20:1",
            "route": "candidate",
            "duplicate_group": "mar20:1--mar20:2",
            "swapped": 1,
        },
    ]
    decisions = [
        {
            "card_id": row["card_id"],
            "label": "same_local_site" if index != 1 else "different_airport",
            "confidence": "0.9",
            "supporting_evidence": "runway geometry",
            "counter_evidence": "",
            "notes": "",
        }
        for index, row in enumerate(mapping)
    ]
    mapping_path = tmp_path / "mapping.csv"
    decisions_path = tmp_path / "decisions.csv"
    _write_csv(mapping_path, mapping)
    _write_csv(decisions_path, decisions)
    output_dir = tmp_path / "compiled"
    result = _run_script(
        "compile_mar20_calibration_review.py",
        "--mapping",
        str(mapping_path),
        "--decisions",
        str(decisions_path),
        "--output-dir",
        str(output_dir),
        "--minimum-positive-pairs",
        "1",
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(
        (output_dir / "calibration_compile_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "pass"
    assert summary["repeat_agreement"] == 1.0
    assert summary["formal_threshold_admission"] is True


def test_compile_view_review_separates_methods_and_background_route(tmp_path: Path) -> None:
    audit_path = tmp_path / "view_audit.csv"
    review_path = tmp_path / "manual_view_review.csv"
    _write_csv(
        audit_path,
        [
            {
                "node_uid": node,
                "fill_method": method,
                "dilation_ratio": 0.15,
                "background_tile_count": 0 if node == "mar20:2" else 8,
            }
            for node in ("mar20:1", "mar20:2")
            for method in ("blur", "local_mean", "telea")
        ],
    )
    reviews = [
        {
            "node_uid": node,
            "valid": 1,
            "blur_aircraft_remnant": 1,
            "blur_inpaint_artifact": 0,
            "local_mean_aircraft_remnant": 0,
            "local_mean_inpaint_artifact": 0,
            "telea_aircraft_remnant": 0,
            "telea_inpaint_artifact": 0,
            "background_tile_available": 0 if node == "mar20:2" else 1,
            "background_tile_aircraft": "" if node == "mar20:2" else 0,
            "notes": "",
        }
        for node in ("mar20:1", "mar20:2")
    ]
    _write_csv(review_path, reviews)
    output = tmp_path / "view_review.json"
    result = _run_script(
        "compile_mar20_view_review.py",
        "--view-audit",
        str(audit_path),
        "--review",
        str(review_path),
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["formal_view_admission"] is True
    assert report["admissible_fill_methods"] == ["local_mean", "telea"]
    assert report["metrics"]["method_metrics"]["blur"]["admitted"] is False
    reviews[0]["background_tile_aircraft"] = 1
    _write_csv(review_path, reviews)
    result = _run_script(
        "compile_mar20_view_review.py",
        "--view-audit",
        str(audit_path),
        "--review",
        str(review_path),
        "--output",
        str(output),
    )
    assert result.returncode == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["formal_view_admission"] is True
    assert report["background_tile_admission"] is False


def test_descriptor_bakeoff_reports_foreground_influence_and_selects_masked(
    tmp_path: Path,
) -> None:
    nodes = [f"mar20:{number}" for number in range(1, 7)]
    original = np.asarray(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )
    masked = np.asarray(
        [
            [1, 0, 0, 0],
            [0.8, 0.6, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0.8, 0.6],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )
    background = masked.copy()
    features = np.concatenate([original, masked, background], axis=0)
    rows = {
        "node_uid": nodes * 3,
        "view_type": ["original"] * 6 + ["masked_inpaint"] * 6 + ["background_tiles"] * 6,
        "rotation": [0] * 18,
        "item_index": [0] * 18,
        "input_sha256": [f"{index:064x}" for index in range(18)],
    }
    cache_dir = tmp_path / "cache"
    writer = PlaceFeatureCacheWriter(
        cache_dir,
        metadata={"test": True},
        feature_names=("f",),
        storage_dtype="float32",
    )
    writer.write_shard(0, rows=rows, features={"f": features})
    writer.finalize(expected_shards=1, expected_rows=18)
    pairs = [
        {
            "pair_uid": "mar20:1--mar20:2",
            "node_u": "mar20:1",
            "node_v": "mar20:2",
            "binary_role": "positive",
            "split": "held_out_audit",
        },
        {
            "pair_uid": "mar20:1--mar20:3",
            "node_u": "mar20:1",
            "node_v": "mar20:3",
            "binary_role": "negative",
            "split": "held_out_audit",
        },
        {
            "pair_uid": "mar20:4--mar20:5",
            "node_u": "mar20:4",
            "node_v": "mar20:5",
            "binary_role": "positive",
            "split": "calibration",
        },
        {
            "pair_uid": "mar20:4--mar20:6",
            "node_u": "mar20:4",
            "node_v": "mar20:6",
            "binary_role": "negative",
            "split": "calibration",
        },
    ]
    pairs_path = tmp_path / "pairs.csv"
    _write_csv(pairs_path, pairs)
    output_dir = tmp_path / "bakeoff"
    result = _run_script(
        "analyze_mar20_descriptor_bakeoff.py",
        "--cache-dir",
        str(cache_dir),
        "--calibration-pairs",
        str(pairs_path),
        "--output-dir",
        str(output_dir),
        "--k-values",
        "1",
        "--minimum-heldout-positive-directions",
        "2",
        "--heldout-recall-target",
        "1.0",
    )
    assert result.returncode == 0, result.stderr
    selection = json.loads((output_dir / "selected_descriptor.json").read_text(encoding="utf-8"))
    assert selection["formal_retrieval_admission"] is True
    assert selection["selected_round_a"] == {
        "feature_name": "f",
        "view_type": "masked_inpaint",
    }
    with (output_dir / "descriptor_bakeoff.csv").open(encoding="utf-8", newline="") as file:
        rows_out = list(csv.DictReader(file))
    masked_row = next(row for row in rows_out if row["view_type"] == "masked_inpaint")
    assert float(masked_row["foreground_influence__held_out_audit__positive_median"]) > 0
