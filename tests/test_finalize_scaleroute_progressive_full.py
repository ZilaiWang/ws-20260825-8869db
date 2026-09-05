import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_finalize_progressive_full_candidate(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    training = tmp_path / "training.json"
    write_json(
        training,
        {
            "status": "complete_candidate",
            "imgsz": 1280,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
        },
    )
    results = tmp_path / "results.csv"
    with results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows({"epoch": index, "loss": 1.0} for index in range(1, 41))
    background = tmp_path / "background.json"
    write_json(
        background,
        {"route": {"false_positives_per_100mp": 2.0, "false_positives": 2}},
    )
    background_runtime = tmp_path / "background_runtime.json"
    write_json(background_runtime, {"image_count": 400, "confidence": 0.001})
    timing = tmp_path / "timing.json"
    write_json(
        timing,
        {
            "status": "cv3_oof_pseudo_inference_complete",
            "folds": [
                {"wall_seconds": 30.0, "image_timings": [{"wall_seconds": 5.0}]},
                {"wall_seconds": 40.0, "image_timings": [{"wall_seconds": 6.0}]},
                {"wall_seconds": 50.0, "image_timings": [{"wall_seconds": 7.0}]},
            ],
        },
    )
    contract = tmp_path / "contract.json"
    write_json(contract, {"deployment_threshold": 0.536})
    output = tmp_path / "summary.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "finalize_scaleroute_progressive_full.py"),
            "--training-result",
            str(training),
            "--results-csv",
            str(results),
            "--background-result",
            str(background),
            "--background-runtime",
            str(background_runtime),
            "--timing-summary",
            str(timing),
            "--input-contract",
            str(contract),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "complete_pending_submission_decision"
    assert payload["timing_only_hard_pseudo10k"]["mean_wall_seconds"] == 6.0
    assert payload["background_100mp"]["false_positives_per_100mp"] == 2.0
