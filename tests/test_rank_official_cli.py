"""官方 V1.6 排名 CLI（scripts/rank_official.py）端到端测试。"""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_rank_from_teams_file(tmp_path: Path) -> None:
    """从队伍汇总 JSON 计算排名。"""
    root = Path(__file__).parent.parent
    teams_path = tmp_path / "teams.json"
    output_path = tmp_path / "rank.json"
    teams_path.write_text(
        json.dumps(
            [
                {
                    "team_id": "M1",
                    "ship_recall": 0.7235,
                    "ship_fdr": 0.5201,
                    "aircraft_recall": 0.9076,
                    "aircraft_fdr": 0.1571,
                    "vehicle_recall": 0.6169,
                    "vehicle_fdr": 0.6161,
                    "latency_seconds": 6.2,
                },
                {
                    "team_id": "M3",
                    "ship_recall": 0.8010,
                    "ship_fdr": 0.3100,
                    "aircraft_recall": 0.9300,
                    "aircraft_fdr": 0.1200,
                    "vehicle_recall": 0.5900,
                    "vehicle_fdr": 0.6600,
                    "latency_seconds": 9.5,
                },
            ]
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/rank_official.py",
            "--teams",
            str(teams_path),
            "--output",
            str(output_path),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["contract"] == "official_ranking_v1_6"
    teams = {team["team_id"]: team for team in data["teams"]}
    # M3 全面优于 M1（除时延更慢），7 项排名应更靠前
    assert teams["M3"]["second_order_position"] < teams["M1"]["second_order_position"]
    assert teams["M1"]["rank_count"] == 7
    assert teams["M3"]["rank_count"] == 7


def test_rank_from_evaluate_outputs(tmp_path: Path) -> None:
    """从 evaluate.py 产物（含 seven_ranking_metrics 块）计算排名。"""
    root = Path(__file__).parent.parent
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"
    result_path = tmp_path / "result.json"
    rank_path = tmp_path / "rank.json"
    gt_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 24, "bbox": [0, 0, 100, 100]}
                ],
                "categories": [{"id": 24, "name": "FSC"}],
            }
        ),
        encoding="utf-8",
    )
    pred_path.write_text(
        json.dumps([{"image_id": 1, "category_id": 24, "bbox": [25, 25, 100, 100], "score": 0.9}]),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    eval_run = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--gt",
            str(gt_path),
            "--pred",
            str(pred_path),
            "--project-config",
            "configs/project.yaml",
            "--latency-seconds",
            "5.0",
            "--output",
            str(result_path),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert eval_run.returncode == 0, eval_run.stdout + eval_run.stderr

    rank_run = subprocess.run(
        [
            sys.executable,
            "scripts/rank_official.py",
            "--results",
            str(result_path),
            "--tag",
            "M1",
            "--output",
            str(rank_path),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rank_run.returncode == 0, rank_run.stdout + rank_run.stderr
    data = json.loads(rank_path.read_text(encoding="utf-8"))
    assert len(data["teams"]) == 1
    team = data["teams"][0]
    assert team["team_id"] == "M1"
    assert team["metric_values"]["vehicle_recall"] == 1.0
    assert team["metric_values"]["latency_seconds"] == 5.0
    # 单大类（vehicle）评估：只有 vehicle 两项 + 时延共 3 项参与排名；
    # ship/aircraft 为 None 被排除。
    assert team["rank_count"] == 3
    assert "ship_recall" not in team["metric_values"]


def test_missing_seven_ranking_block_rejected(tmp_path: Path) -> None:
    """没有 seven_ranking_metrics 块的 JSON 应被拒绝。"""
    root = Path(__file__).parent.parent
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"official_ranking": {"per_coarse": {}}}), encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/rank_official.py",
            "--results",
            str(bad_path),
            "--tag",
            "X",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "seven_ranking_metrics" in result.stdout + result.stderr


def test_tag_count_mismatch_rejected(tmp_path: Path) -> None:
    """--results 与 --tag 数量不一致应报错。"""
    root = Path(__file__).parent.parent
    result_path = tmp_path / "r.json"
    result_path.write_text(json.dumps({}), encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/rank_official.py",
            "--results",
            str(result_path),
            "--tag",
            "M1",
            "--tag",
            "M3",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "数量必须一致" in result.stdout + result.stderr
