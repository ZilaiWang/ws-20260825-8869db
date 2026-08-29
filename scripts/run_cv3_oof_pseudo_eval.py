#!/usr/bin/env python3
"""Run one frozen deployment config on formal-CV3 held-out pseudo-10K folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.submission.competition import run_submission


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    combined_gt = json.loads(
        (args.pseudo_root / "ground_truth.json").read_text(encoding="utf-8")
    )
    image_id_by_name = {
        str(item["file_name"]): int(item["id"]) for item in combined_gt["images"]
    }

    all_predictions: list[dict[str, object]] = []
    fold_summaries: list[dict[str, object]] = []
    for fold, weight in enumerate(args.weights):
        if not weight.is_file():
            raise FileNotFoundError(weight)
        fold_dir = args.pseudo_root / f"fold_{fold}"
        input_dir = fold_dir / "images"
        config = json.loads(json.dumps(base_config))
        config["model"]["weight_path"] = str(weight.resolve())
        config["model"]["expected_sha256"] = _sha256(weight)
        config["workpoint_id"] = f"{base_config['workpoint_id']}__oof_fold{fold}"
        run_dir = args.output_dir / f"fold_{fold}"
        run_dir.mkdir(parents=True, exist_ok=True)
        materialized = run_dir / "config.materialized.json"
        materialized.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        started = time.perf_counter()
        payload = run_submission(input_dir, run_dir, materialized)
        elapsed = time.perf_counter() - started
        fold_predictions: list[dict[str, object]] = []
        for image in payload["images"]:
            filename = str(image["file_name"])
            image_id = image_id_by_name[filename]
            for item in image["objects"]:
                x1, y1, x2, y2 = item["bbox"]
                fold_predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(item["category_id"]),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(item["score"]),
                        "source_fold": fold,
                    }
                )
        prediction_path = run_dir / "predictions.json"
        prediction_path.write_text(
            json.dumps(fold_predictions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        all_predictions.extend(fold_predictions)
        fold_summaries.append(
            {
                "fold": fold,
                "weight": str(weight),
                "weight_sha256": _sha256(weight),
                "images": len(payload["images"]),
                "predictions": len(fold_predictions),
                "wall_seconds": elapsed,
                "average_seconds": elapsed / max(1, len(payload["images"])),
            }
        )

    combined_path = args.output_dir / "predictions.json"
    combined_path.write_text(
        json.dumps(all_predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "cv3_oof_pseudo_inference_complete",
        "protocol": "each_fold_checkpoint_only_on_its_heldout_pseudo10k_v1",
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "predictions": len(all_predictions),
        "predictions_sha256": _sha256(combined_path),
        "folds": fold_summaries,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
