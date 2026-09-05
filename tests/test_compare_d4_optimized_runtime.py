from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_compare_d4_optimized_runtime_accepts_exact_predictions(tmp_path: Path) -> None:
    predictions = [{"image_id": 1, "category_id": 4, "bbox": [1, 2, 3, 4], "score": 0.9}]
    reference = {"images": 1, "predictions": 1, "mean_image_seconds": 2.0}
    optimized = {"images": 1, "predictions": 1, "mean_image_seconds": 1.0}
    paths = {}
    for name, payload in {
        "ref_pred": predictions,
        "opt_pred": predictions,
        "ref_sum": reference,
        "opt_sum": optimized,
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    output = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_d4_optimized_runtime.py",
            "--reference-predictions",
            str(paths["ref_pred"]),
            "--optimized-predictions",
            str(paths["opt_pred"]),
            "--reference-summary",
            str(paths["ref_sum"]),
            "--optimized-summary",
            str(paths["opt_sum"]),
            "--optimization-label",
            "tensorized_views=true,channels_last=false",
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["predictions_exact"] is True
    assert payload["optimized_speedup"] == 2.0
    assert payload["optimization"] == "tensorized_views=true,channels_last=false"
