"""对象几何可见性 CLI 小数据端到端测试。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _write_small_audit_tables(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "image_stats.csv"
    images.write_text("\ufeffimage_uid,width,height\nimg_1,200,200\n", encoding="utf-8")
    boxes = tmp_path / "bbox_statistics.csv"
    header = (
        "\ufeffannotation_uid,image_uid,category_id,category_name,macro_category,"
        "width_norm,height_norm,x0,y0,x1,y1,width_px,height_px,issue_flags\n"
    )
    rows = [
        "ann_1,img_1,0,HM,ship,0.5,0.25,50,75,150,125,100,50,",
        "ann_2,img_1,4,A1_SU-35,aircraft,0.25,0.25,10,10,60,60,50,50,",
        "ann_3,img_1,24,FSC,vehicle,0.1,0.1,-5,0,15,20,20,20,out_of_bounds;touches_edge_within_1px",
    ]
    boxes.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    return boxes, images


def _run(repo_root: Path, boxes: Path, images: Path, output: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    return subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_object_visibility.py"),
            "--bbox-statistics",
            str(boxes),
            "--image-stats",
            str(images),
            "--config",
            str(repo_root / "configs" / "analysis" / "object_visibility.yaml"),
            "--output-dir",
            str(output),
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
    boxes, images = _write_small_audit_tables(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    first_run = _run(repo_root, boxes, images, first)
    second_run = _run(repo_root, boxes, images, second)
    assert first_run.returncode == 0, first_run.stdout + first_run.stderr
    assert second_run.returncode == 0, second_run.stdout + second_run.stderr
    expected = {
        "resolved_config.yaml",
        "meta.json",
        "recommendations.yaml",
        "object_visibility.csv",
        "visibility_summary.csv",
        "scenario_comparison.csv",
        "report.md",
        "run.log",
    }
    assert expected <= {path.name for path in first.iterdir()}
    for filename in (
        "object_visibility.csv",
        "visibility_summary.csv",
        "scenario_comparison.csv",
        "recommendations.yaml",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_cli_missing_column_returns_nonzero_without_report(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    boxes, images = _write_small_audit_tables(tmp_path)
    boxes.write_text("annotation_uid,image_uid\nann_1,img_1\n", encoding="utf-8")
    output = tmp_path / "failed"
    result = _run(repo_root, boxes, images, output)
    assert result.returncode == 1
    assert not (output / "report.md").exists()
