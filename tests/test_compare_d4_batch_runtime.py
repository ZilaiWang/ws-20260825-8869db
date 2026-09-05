from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_compare_d4_batch_runtime_accepts_exact_predictions(tmp_path: Path) -> None:
    predictions = [{"image_id": 1, "category_id": 4, "bbox": [1, 2, 3, 4], "score": 0.9}]
    summary32 = {"images": 1, "predictions": 1, "mean_image_seconds": 2.0}
    summary64 = {"images": 1, "predictions": 1, "mean_image_seconds": 1.5}
    paths = {}
    for name, payload in {
        "pred32": predictions,
        "pred64": predictions,
        "sum32": summary32,
        "sum64": summary64,
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    output = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_d4_batch_runtime.py",
            "--batch32-predictions",
            str(paths["pred32"]),
            "--batch64-predictions",
            str(paths["pred64"]),
            "--batch32-summary",
            str(paths["sum32"]),
            "--batch64-summary",
            str(paths["sum64"]),
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["predictions_exact"] is True
    assert payload["batch64_speedup"] == 4 / 3
