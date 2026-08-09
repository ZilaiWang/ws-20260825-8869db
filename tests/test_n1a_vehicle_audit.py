"""A 主线 1 车辆后处理审计测试。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rsdet.analysis.crossfit_thresholds import load_cv3_aggregate


def _make_aggregate(tmp_path: Path):
    """构造小型 OOF aggregate 用于审计测试。"""
    root = tmp_path / "agg"
    root.mkdir(parents=True)
    (root / "oof_metadata.json").write_text(
        json.dumps(
            {
                "model_key": "M1",
                "seed": 42,
                "low_score_threshold": 0.001,
                "source_manifest_sha256": "sha",
            }
        ),
        encoding="utf-8",
    )
    predictions = []
    # 图1: 1 个车辆 GT 匹配的高分候选 + 1 个低分干扰
    predictions.append(
        {"image_id": 1, "category_id": 24, "bbox": [0, 0, 10, 10], "score": 0.9}
    )
    # 图2: 车辆 GT 有一个低分候选(0.01) → low_score_only
    predictions.append(
        {"image_id": 2, "category_id": 24, "bbox": [0, 0, 10, 10], "score": 0.01}
    )
    # 图3: 车辆 GT 完全无候选(只有飞机候选)
    predictions.append(
        {"image_id": 3, "category_id": 4, "bbox": [0, 0, 10, 10], "score": 0.8}
    )
    (root / "predictions_oof_low.json").write_text(
        json.dumps(predictions), encoding="utf-8"
    )
    (root / "oof_images.csv").write_text(
        "image_id,relative_path,fold,group_id\n"
        "1,images/train/a.jpg,0,g1\n"
        "2,images/train/b.jpg,1,g1\n"
        "3,images/train/c.jpg,2,g1\n",
        encoding="utf-8",
    )
    return root


def _make_manifest(tmp_path: Path):
    """formal manifest：3 个车辆 GT。"""
    path = tmp_path / "formal.csv"
    path.write_text(
        "crop_policy,annotation_uid,formal_image_id,class_id,"
        "gt_x0,gt_y0,gt_x1,gt_y1,group_id,fold,source_edge_risk\n"
        "tight,ann_1,1,24,1,1,10,10,g1,0,False\n"
        "tight,ann_2,2,24,1,1,10,10,g1,1,False\n"
        "tight,ann_3,3,24,1,1,10,10,g1,2,False\n",
        encoding="utf-8",
    )
    return path


def test_audit_buckets(tmp_path: Path):
    """验证三类分桶正确。"""
    aggregate = _make_aggregate(tmp_path)
    manifest = _make_manifest(tmp_path)
    output = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/n1_a_postprocess_audit.py",
            "--aggregate",
            str(aggregate),
            "--formal-manifest",
            str(manifest),
            "--output-dir",
            str(output),
        ],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads((output / "vehicle_audit_result.json").read_text(encoding="utf-8"))
    summary = data["summary"]
    assert summary["vehicle_gt_total"] == 3
    assert summary["matched"] == 1   # 图1 高分
    assert summary["low_score_only"] == 1  # 图2 低分
    assert summary["no_candidate"] == 1    # 图3 无车辆候选
    assert summary["work_recall"] == pytest.approx(1 / 3, abs=0.001)


def test_iou_threshold_respected(tmp_path: Path):
    """IoU<0.35 的候选不算命中。"""
    aggregate = _make_aggregate(tmp_path)
    # 修改图1 的候选位置使其 IoU 不足
    root = aggregate / "predictions_oof_low.json"
    data = json.loads(root.read_text(encoding="utf-8"))
    data[0]["bbox"] = [100, 100, 5, 5]  # 远离 GT [1,1,10,10]
    root.write_text(json.dumps(data), encoding="utf-8")

    manifest = _make_manifest(tmp_path)
    output = tmp_path / "out2"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/n1_a_postprocess_audit.py",
            "--aggregate",
            str(aggregate),
            "--formal-manifest",
            str(manifest),
            "--output-dir",
            str(output),
        ],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads((output / "vehicle_audit_result.json").read_text(encoding="utf-8"))
    assert data["summary"]["matched"] == 0  # IoU 不足不算
