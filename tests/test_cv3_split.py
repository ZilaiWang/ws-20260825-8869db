import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from rsdet.data.cv3_split import (
    EXPECTED_K60_GROUPS,
    EXPECTED_MAR20_IMAGES,
    FOLD_COUNT,
    audit_near_duplicates,
    load_airport_groups,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
GROUP_MAP = ROOT / "data/groups/mar20_airport_proxy_k60_for_b.csv"
ASSIGNMENT = ROOT / "data/splits/cv3_airport_proxy_k60_v2_groups.json"
MANIFEST = ROOT / "data/splits/cv3_airport_proxy_k60_v2.json"
AUDIT = ROOT / "reports/data/cv3_airport_proxy_k60_v2_audit.json"
DATA_ROOT = ROOT.parent / "data"
EXPECTED_GROUP_MAP_SHA256 = "3cd2fdb1db0b95ec3069db569fe56019942b9760a3a261f44800164135256f00"
EXPECTED_MANIFEST_SHA256 = "27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331"


def _raw_group_map() -> dict[str, str]:
    with GROUP_MAP.open(encoding="utf-8-sig", newline="") as handle:
        return {Path(row["image_name"]).stem: row["group_id"] for row in csv.DictReader(handle)}


def test_k60_csv_contract_is_frozen() -> None:
    raw = _raw_group_map()
    loaded = load_airport_groups(GROUP_MAP, expected_stems=set(raw))

    assert loaded == raw
    assert len(loaded) == EXPECTED_MAR20_IMAGES
    assert len(set(loaded.values())) == EXPECTED_K60_GROUPS
    assert sha256_file(GROUP_MAP) == EXPECTED_GROUP_MAP_SHA256


@pytest.mark.parametrize("mutation", ["duplicate", "empty", "missing", "extra"])
def test_k60_loader_rejects_contract_violations(tmp_path: Path, mutation: str) -> None:
    rows = [
        {"image_name": "MAR20_1.jpg", "group_id": "mar20-airport-proxy-001"},
        {"image_name": "MAR20_2.jpg", "group_id": "mar20-airport-proxy-002"},
    ]
    expected_stems = {"MAR20_1", "MAR20_2"}
    if mutation == "duplicate":
        rows[1]["image_name"] = "MAR20_1.jpg"
    elif mutation == "empty":
        rows[1]["group_id"] = ""
    elif mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(
            {
                "image_name": "MAR20_3.jpg",
                "group_id": "mar20-airport-proxy-002",
            }
        )

    path = tmp_path / "groups.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "group_id"])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError):
        load_airport_groups(
            path,
            expected_stems=expected_stems,
            expected_image_count=2,
            expected_group_count=2,
        )


def test_legacy_dhash_is_audit_only_and_does_not_mutate_k60(
    tmp_path: Path,
) -> None:
    mapping = {
        "MAR20_1": "mar20-airport-proxy-001",
        "MAR20_2": "mar20-airport-proxy-002",
    }
    original = dict(mapping)
    path = tmp_path / "near_duplicates.json"
    path.write_text(
        json.dumps({"duplicate_groups": [["MAR20_1", "MAR20_2"]]}),
        encoding="utf-8",
    )

    result = audit_near_duplicates(path, mapping)

    assert mapping == original
    assert result["policy"] == "audit_only_never_merge_k60"
    assert result["cross_k60_candidate_groups"] == 1


def test_frozen_v2_manifest_preserves_k60_and_all_groups() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assignment_payload = json.loads(ASSIGNMENT.read_text(encoding="utf-8"))
    assignment = assignment_payload["groups"]
    samples = manifest["samples"]

    assert manifest["version"] == "cv3_airport_proxy_k60_v2"
    assert manifest["fold_count"] == FOLD_COUNT
    assert len(samples) == 4481
    assert len({sample["image_id"] for sample in samples}) == len(samples)
    assert len({sample["relative_path"] for sample in samples}) == len(samples)
    assert set(assignment) == {sample["group_id"] for sample in samples}

    folds_by_group: defaultdict[str, set[int]] = defaultdict(set)
    for sample in samples:
        assert sample["fold"] in range(FOLD_COUNT)
        assert sample["fold"] == assignment[sample["group_id"]]
        folds_by_group[sample["group_id"]].add(sample["fold"])
    assert all(len(folds) == 1 for folds in folds_by_group.values())

    k60 = _raw_group_map()
    mar20_samples = {
        Path(sample["relative_path"]).stem: sample
        for sample in samples
        if Path(sample["relative_path"]).stem.startswith("MAR20_")
    }
    assert set(mar20_samples) == set(k60)
    assert all(mar20_samples[stem]["group_id"] == group_id for stem, group_id in k60.items())
    assert len({sample["group_id"] for sample in mar20_samples.values()}) == 60


def test_frozen_v2_preserves_non_mar20_group_fields_from_dev_v1() -> None:
    v2 = json.loads(MANIFEST.read_text(encoding="utf-8"))
    dev_v1 = json.loads((ROOT / "data/splits/dev_v1.json").read_text(encoding="utf-8"))
    baseline = {
        sample["image_id"]: sample
        for sample in dev_v1["samples"]
        if not Path(sample["relative_path"]).stem.startswith("MAR20_")
    }
    actual = {
        sample["image_id"]: sample
        for sample in v2["samples"]
        if not Path(sample["relative_path"]).stem.startswith("MAR20_")
    }

    assert set(actual) == set(baseline)
    for image_id, sample in actual.items():
        old = baseline[image_id]
        assert sample["relative_path"] == old["relative_path"]
        assert sample["group_id"] == old["group_id"]
        assert sample["group_rule"] == old["group_rule"]


def test_frozen_v2_scientific_audit_passes() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    scientific = audit["scientific_audit"]

    assert audit["formal_cv3_admission"] is True
    assert scientific["status"] == "pass"
    assert scientific["hard_gate_violations"] == []
    assert scientific["group_cross_fold_count"] == 0
    assert scientific["mar20_image_count"] == 3073
    assert scientific["mar20_group_count"] == 60
    assert scientific["k60_mapping_mismatch_count"] == 0
    assert len(scientific["per_class"]) == 25
    for row in scientific["per_class"]:
        assert all(value > 0 for value in row["val_boxes"])
        assert all(value > 0 for value in row["val_groups"])
        assert min(row["train_boxes"]) >= row["minimum_train_boxes"]

    tu160 = scientific["tu160_stress_fold"]
    assert sorted(tu160["val_boxes"]) == [1, 8, 352]
    assert sorted(tu160["train_boxes"]) == [9, 353, 360]
    assert set(tu160["group_folds"].values()) == {0, 1, 2}
    assert sha256_file(MANIFEST) == EXPECTED_MANIFEST_SHA256


@pytest.mark.skipif(
    not (DATA_ROOT / "images/train").is_dir(),
    reason="full competition data are not present",
)
def test_v2_cli_is_byte_deterministic_and_failure_does_not_write(
    tmp_path: Path,
) -> None:
    output_a = tmp_path / "a.json"
    audit_a = tmp_path / "a.audit.json"
    output_b = tmp_path / "b.json"
    audit_b = tmp_path / "b.audit.json"
    base_command = [
        sys.executable,
        str(ROOT / "scripts/build_cv3.py"),
        "--data-root",
        str(DATA_ROOT),
        "--airport-groups",
        str(GROUP_MAP),
        "--assignment",
        str(ASSIGNMENT),
        "--near-duplicates",
        str(ROOT / "reports/data/near_duplicates_mar20.json"),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    for seed, output, audit in (
        ("0", output_a, audit_a),
        ("777", output_b, audit_b),
    ):
        environment["PYTHONHASHSEED"] = seed
        subprocess.run(
            [
                *base_command,
                "--output",
                str(output),
                "--audit-output",
                str(audit),
            ],
            check=True,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
    assert output_a.read_bytes() == output_b.read_bytes() == MANIFEST.read_bytes()

    invalid_assignment = json.loads(ASSIGNMENT.read_text(encoding="utf-8"))
    invalid_assignment["groups"]["mar20-airport-proxy-019"] = 0
    invalid_path = tmp_path / "invalid_assignment.json"
    invalid_path.write_text(
        json.dumps(invalid_assignment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    failed_output = tmp_path / "must_not_exist.json"
    failed_audit = tmp_path / "must_not_exist.audit.json"
    failed = subprocess.run(
        [
            *base_command[: base_command.index("--assignment") + 1],
            str(invalid_path),
            *base_command[base_command.index("--assignment") + 2 :],
            "--output",
            str(failed_output),
            "--audit-output",
            str(failed_audit),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert not failed_output.exists()
    assert not failed_audit.exists()
