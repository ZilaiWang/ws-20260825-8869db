import json
from pathlib import Path

import pytest

from scripts.compare_pseudo_frontiers import _row


def test_row_extracts_fixed_risk_and_candidate_metrics(tmp_path: Path) -> None:
    payload = {
        "status": "complete",
        "candidate_floor": {
            "recall": 0.97,
            "fdr": 0.90,
            "per_coarse": {
                name: {"recall": value}
                for name, value in {"ship": 0.96, "aircraft": 0.99, "vehicle": 0.91}.items()
            },
        },
        "frontiers": {
            "0.150": {
                "crossfit": {
                    "recall": 0.82,
                    "fdr": 0.149,
                    "macro_recall": 0.80,
                    "macro_fdr": 0.16,
                    "per_coarse": {
                        name: {"recall": recall, "fdr": fdr}
                        for name, (recall, fdr) in {
                            "ship": (0.80, 0.15),
                            "aircraft": (0.90, 0.10),
                            "vehicle": (0.55, 0.30),
                        }.items()
                    },
                }
            }
        },
    }
    path = tmp_path / "frontier.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    row = _row("candidate", path, "0.150")
    assert row["candidate_recall"] == pytest.approx(0.97)
    assert row["recall"] == pytest.approx(0.82)
    assert row["vehicle_fdr"] == pytest.approx(0.30)


def test_row_rejects_incomplete_artifact(tmp_path: Path) -> None:
    path = tmp_path / "frontier.json"
    path.write_text('{"status":"failed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not complete"):
        _row("bad", path, "0.150")
