"""P0-2 crop manifest CLI 小数据端到端测试。"""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_small_audit_package(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    image_specs = [
        ("img_0", "train", 0, "HM", "ship"),
        ("img_1", "val", 4, "A1_SU-35", "aircraft"),
        ("img_2", "train", 24, "FSC", "vehicle"),
    ]
    image_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    domain_rows: list[dict[str, object]] = []
    bbox_rows: list[dict[str, object]] = []
    for index, (uid, split, category_id, category_name, macro) in enumerate(image_specs):
        relative_path = f"images/{split}/{uid}.png"
        image_path = data_root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 80), color=(20 + index * 20, 40, 60)).save(image_path)
        checksum = hashlib.sha256(image_path.read_bytes()).hexdigest()
        group = f"group_{index}"
        image_rows.append(
            {
                "image_uid": uid,
                "relative_path": relative_path,
                "width": 100,
                "height": 80,
                "sha256": checksum,
                "estimated_group_uid": group,
                "group_basis": "test",
                "group_confidence": "high",
            }
        )
        split_rows.append(
            {
                "image_uid": uid,
                "relative_path": relative_path,
                "estimated_group_uid": group,
                "main_split": split,
                "fold3": index,
                "difficult_quality_subset": False,
                "edge_target_subset": False,
                "pure_background_subset": False,
                "speed_proxy_subset": False,
            }
        )
        domain_rows.append(
            {"image_uid": uid, "domain_cluster": index, "estimated_group_uid": group}
        )
        bbox_rows.append(
            {
                "annotation_uid": f"ann_{index}",
                "image_uid": uid,
                "category_id": category_id,
                "category_name": category_name,
                "macro_category": macro,
                "x0": 20,
                "y0": 20,
                "x1": 60,
                "y1": 50,
                "width_px": 40,
                "height_px": 30,
                "width_norm": 0.4,
                "height_norm": 0.375,
                "issue_flags": "",
            }
        )
    image_stats = tmp_path / "image_stats.csv"
    split_candidates = tmp_path / "split_candidates.csv"
    domain_assignments = tmp_path / "domain_assignments.csv"
    bbox_statistics = tmp_path / "bbox_statistics.csv"
    near_duplicates = tmp_path / "near_duplicates.csv"
    _write_csv(image_stats, list(image_rows[0]), image_rows)
    _write_csv(split_candidates, list(split_rows[0]), split_rows)
    _write_csv(domain_assignments, list(domain_rows[0]), domain_rows)
    _write_csv(bbox_statistics, list(bbox_rows[0]), bbox_rows)
    _write_csv(
        near_duplicates,
        ["image_uid_1", "image_uid_2", "status"],
        [],
    )
    return {
        "data_root": data_root,
        "image_stats": image_stats,
        "split_candidates": split_candidates,
        "domain_assignments": domain_assignments,
        "bbox_statistics": bbox_statistics,
        "near_duplicates": near_duplicates,
    }


def _run(
    repo_root: Path, package: dict[str, Path], output: Path
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    return subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_crop_manifest.py"),
            "--bbox-statistics",
            str(package["bbox_statistics"]),
            "--image-stats",
            str(package["image_stats"]),
            "--split-candidates",
            str(package["split_candidates"]),
            "--domain-assignments",
            str(package["domain_assignments"]),
            "--near-duplicates",
            str(package["near_duplicates"]),
            "--data-root",
            str(package["data_root"]),
            "--config",
            str(repo_root / "configs" / "analysis" / "exploratory_crop_manifest.yaml"),
            "--output-dir",
            str(output),
            "--skip-previews",
            "--skip-figures",
            "--verbose",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_generates_reproducible_numeric_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    package = _write_small_audit_package(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    first_run = _run(repo_root, package, first)
    second_run = _run(repo_root, package, second)
    assert first_run.returncode == 0, first_run.stdout + first_run.stderr
    assert second_run.returncode == 0, second_run.stdout + second_run.stderr
    expected = {
        "crop_manifest.csv",
        "image_assignments.csv",
        "leakage_components.csv",
        "class_distribution.csv",
        "policy_geometry_summary.csv",
        "validation_report.json",
        "manifest_summary.json",
        "resolved_config.yaml",
        "meta.json",
        "report.md",
        "run.log",
    }
    assert expected <= {path.name for path in first.iterdir()}
    for filename in (
        "crop_manifest.csv",
        "image_assignments.csv",
        "leakage_components.csv",
        "class_distribution.csv",
        "policy_geometry_summary.csv",
        "validation_report.json",
        "manifest_summary.json",
        "resolved_config.yaml",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_cli_missing_column_returns_nonzero_without_report(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    package = _write_small_audit_package(tmp_path)
    package["bbox_statistics"].write_text(
        "annotation_uid,image_uid\nann_0,img_0\n", encoding="utf-8"
    )
    output = tmp_path / "failed"
    result = _run(repo_root, package, output)
    assert result.returncode == 1
    assert not (output / "report.md").exists()
