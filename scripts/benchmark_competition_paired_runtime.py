#!/usr/bin/env python3
"""AB/BA paired runtime benchmark for two CompetitionDetector configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from PIL import Image

from rsdet.submission.competition import CompetitionDetector, load_submission_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_image(root: Path, image: dict[str, Any]) -> Path:
    file_name = str(image["file_name"])
    candidates = [root / "images" / file_name, root / file_name]
    if "fold" in image:
        candidates.insert(0, root / f"fold_{int(image['fold'])}" / "images" / file_name)
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(f"expected one image for {file_name}, found {existing}")
    return existing[0]


def _synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _predict(detector: CompetitionDetector, image: Image.Image) -> tuple[float, int]:
    _synchronize()
    started = time.perf_counter()
    rows = detector.predict(image)
    _synchronize()
    return time.perf_counter() - started, len(rows)


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p25": statistics.quantiles(ordered, n=4, method="inclusive")[0],
        "median": statistics.median(ordered),
        "p75": statistics.quantiles(ordered, n=4, method="inclusive")[2],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-a", type=Path, required=True)
    parser.add_argument("--config-b", type=Path, required=True)
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-images", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 2 or args.warmup_images < 1:
        raise ValueError("repeats must be >=2 and warmup-images must be >=1")
    if args.output.exists():
        raise FileExistsError(args.output)

    gt_path = args.ground_truth or args.pseudo_root / "ground_truth.json"
    document = json.loads(gt_path.read_text(encoding="utf-8"))
    images = document.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("ground truth must contain images")
    paths = [_resolve_image(args.pseudo_root, row) for row in images]
    config_a = load_submission_config(args.config_a)
    config_b = load_submission_config(args.config_b)
    config_a["device"] = args.device
    config_b["device"] = args.device
    detector_a = CompetitionDetector(config_a)
    detector_b = CompetitionDetector(config_b)
    Image.MAX_IMAGE_PIXELS = None

    for path in paths[: args.warmup_images]:
        with Image.open(path) as source:
            image = source.convert("RGB")
        _predict(detector_a, image)
        _predict(detector_b, image)

    records: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        for image_index, path in enumerate(paths):
            with Image.open(path) as source:
                image = source.convert("RGB")
            order = ("a", "b") if (repeat + image_index) % 2 == 0 else ("b", "a")
            measured: dict[str, tuple[float, int]] = {}
            for token in order:
                detector = detector_a if token == "a" else detector_b
                measured[token] = _predict(detector, image)
            records.append(
                {
                    "repeat": repeat,
                    "image_index": image_index,
                    "file_name": str(images[image_index]["file_name"]),
                    "order": "".join(order),
                    "a_seconds": measured["a"][0],
                    "b_seconds": measured["b"][0],
                    "delta_b_minus_a_seconds": measured["b"][0] - measured["a"][0],
                    "a_predictions": measured["a"][1],
                    "b_predictions": measured["b"][1],
                }
            )
            print(
                f"repeat={repeat + 1}/{args.repeats} image={image_index + 1}/{len(paths)} "
                f"a={measured['a'][0]:.4f}s b={measured['b'][0]:.4f}s",
                flush=True,
            )

    a_times = [float(row["a_seconds"]) for row in records]
    b_times = [float(row["b_seconds"]) for row in records]
    deltas = [float(row["delta_b_minus_a_seconds"]) for row in records]
    by_order = {
        order: _quantiles(
            [float(row["delta_b_minus_a_seconds"]) for row in records if row["order"] == order]
        )
        for order in ("ab", "ba")
    }
    payload = {
        "status": "complete",
        "protocol": "competition_detector_abba_paired_runtime_v1",
        "config_a": str(args.config_a),
        "config_a_sha256": _sha256(args.config_a),
        "config_b": str(args.config_b),
        "config_b_sha256": _sha256(args.config_b),
        "ground_truth": str(gt_path),
        "ground_truth_sha256": _sha256(gt_path),
        "device": args.device,
        "images": len(paths),
        "repeats": args.repeats,
        "warmup_images": args.warmup_images,
        "measurements": len(records),
        "a_seconds": _quantiles(a_times),
        "b_seconds": _quantiles(b_times),
        "paired_delta_b_minus_a_seconds": _quantiles(deltas),
        "paired_delta_by_order": by_order,
        "records": records,
        "timing_is_proxy_diagnostic_not_official_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
