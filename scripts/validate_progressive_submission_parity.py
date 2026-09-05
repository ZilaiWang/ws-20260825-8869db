#!/usr/bin/env python3
"""Run the delivered official entry point on all 12 frozen pseudo images.

Launch with PYTHONPATH=<delivery>/app to validate the exact delivered source.
No threshold selection, training, packaging, or official submission is performed.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from rsdet.submission import competition


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    delivery, root, out = args.delivery.resolve(), args.root, args.output
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    if not Path(competition.__file__).resolve().is_relative_to(delivery / "app"):
        raise ValueError("must import the delivered source, not the working repository")
    for entry in read(delivery / "BUILD_MANIFEST.json")["files"]:
        if sha(delivery / entry["path"]) != entry["sha256"]:
            raise ValueError(f"delivery SHA mismatch: {entry['path']}")
    config = read(delivery / "app/config.json")
    config["model"]["weight_path"] = str(delivery / "models/model.pt")
    materialized = out / "gpu_path_config.json"
    materialized.write_text(json.dumps(config, indent=2) + "\n")
    inputs = out / "input"
    inputs.mkdir()
    references = {}
    diag = root / "results/SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-SEEN-DIAGNOSTIC-V1"
    for name, directory in [("hard", "pseudo10k-trial-mix-local"),
                             ("sentinel", "pseudo10k-trial-mix-sentinel-v1")]:
        gt = read(root / directory / "ground_truth.json")
        pred = read(diag / name / "predictions.json")
        for image in gt["images"]:
            filename = f"{name}_{image['file_name']}"
            source = root / directory / f"fold_{image['fold']}" / "images" / image["file_name"]
            (inputs / filename).symlink_to(source)
            references[filename] = sorted([
                [p["category_id"], *p["bbox"][:2],
                 p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3], p["score"]]
                for p in pred if p["image_id"] == image["id"] and p["score"] >= 0.536
            ])
    durations = []

    class TimedDetector(competition.CompetitionDetector):
        def predict(self, image):
            start = time.perf_counter()
            result = super().predict(image)
            durations.append(time.perf_counter() - start)
            return result

    start = time.perf_counter()
    payload = competition.run_submission(inputs, out, materialized, detector_factory=TimedDetector)
    checks = []
    for item, seconds in zip(payload["images"], durations, strict=True):
        actual = sorted([o["category_id"], *o["bbox"], o["score"]] for o in item["objects"])
        expected = references[item["file_name"]]
        if len(actual) != len(expected):
            raise ValueError(f"object count mismatch: {item['file_name']}: {len(actual)} != {len(expected)}")
        max_box, max_score = 0.0, 0.0
        for a, b in zip(actual, expected, strict=True):
            if a[0] != b[0]:
                raise ValueError("category mismatch")
            max_box = max(max_box, max(abs(x - y) for x, y in zip(a[1:5], b[1:5])))
            max_score = max(max_score, abs(a[5] - b[5]))
        if max_box > 0.001 or max_score > 0.00001:
            raise ValueError(f"per-box parity failed: {item['file_name']}: {max_box}, {max_score}")
        checks.append({"image": item["file_name"], "objects": len(actual),
                       "max_box_delta_pixels": max_box, "max_score_delta": max_score,
                       "seconds": seconds})
    summary = {
        "status": "pass", "check": "delivered_entrypoint_vs_frozen_offline_ledger",
        "container_gpu_run": False,
        "note": "Linux RTX3090 runs the exact delivered Python source; Docker itself is tested separately.",
        "weight_sha256": sha(delivery / "models/model.pt"),
        "original_container_config_sha256": sha(delivery / "app/config.json"),
        "delivery_manifest_sha256": sha(delivery / "BUILD_MANIFEST.json"),
        "result_sha256": sha(out / "result.json"),
        "counts": competition.validate_result_payload(payload),
        "cold_end_to_end_seconds": time.perf_counter() - start,
        "mean_image_seconds": sum(durations) / len(durations), "images": checks,
        "source_seen_diagnostic_only": True,
    }
    (out / "parity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
