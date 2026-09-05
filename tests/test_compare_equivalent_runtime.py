from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_compare_equivalent_runtime_reports_exact_speedup(tmp_path: Path) -> None:
    predictions = [{"image_id": 1, "category_id": 2, "bbox": [1, 2, 3, 4], "score": 0.9}]
    paths = {}
    for name, payload in {
        "reference_predictions": predictions,
        "candidate_predictions": predictions,
        "reference_summary": {"images": 1, "predictions": 1, "mean_image_seconds": 2.0},
        "candidate_summary": {"images": 1, "predictions": 1, "mean_image_seconds": 1.25},
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    output = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_equivalent_runtime.py",
            "--reference-predictions",
            str(paths["reference_predictions"]),
            "--candidate-predictions",
            str(paths["candidate_predictions"]),
            "--reference-summary",
            str(paths["reference_summary"]),
            "--candidate-summary",
            str(paths["candidate_summary"]),
            "--protocol",
            "unit_test_v1",
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["predictions_exact"] is True
    assert payload["candidate_speedup"] == 1.6
    assert payload["protocol"] == "unit_test_v1"
