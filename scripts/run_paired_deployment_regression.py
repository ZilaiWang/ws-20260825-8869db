#!/usr/bin/env python3
"""Fixed B: known Background-100MP + two native-pixel 100MP seam/latency canvases.

Engineering regression only. No data-distribution simulation, full training,
container build, official submission or threshold fitting.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from rsdet.experiments.paired_trend import read, safe_path, sha, validate_bundle, write
from scripts.run_paired_trend import PROJECT, infer, validate_lineage


def freeze_canvases(bundle: Path, output: Path) -> dict:
    validate_bundle(bundle, PROJECT)
    gt = read(bundle / "train_gt.json")
    by_image = defaultdict(list)
    for a in gt["annotations"]:
        by_image[a["image_id"]].append(a)
    selected = []
    for labels in (set(range(4)), set(range(4, 24)), {24}):
        rows = [
            i for i in gt["images"] if any(a["category_id"] in labels for a in by_image[i["id"]])
        ]
        rows.sort(key=lambda i: (len(by_image[i["id"]]), i["id"]))
        selected.extend([rows[0], rows[-1]])
    if len({i["id"] for i in selected}) != 6:
        raise ValueError("six unique sparse/dense source images required")
    placements = []
    for index, i in enumerate(selected):
        if max(i["width"], i["height"]) > 1800:
            raise ValueError("source cannot fit native-pixel cell without cropping")
        placements.append(
            {
                "source": i,
                "annotations": by_image[i["id"]],
                "x": 300 + (index % 3) * 2800,
                "y": 300 + (index // 3) * 3500,
            }
        )
    config_path = ROOT / "submission/docker/configs/progressive40_full_s1280_frozen0536_v1.json"
    config = read(config_path)
    for key in ("source_training_checkpoint_sha256", "workpoint_id", "deployment_role"):
        config.pop(key, None)
    config["model"]["weight_path"] = "RUNTIME_CHECKPOINT"
    config["model"]["expected_sha256"] = "RUNTIME_CHECKPOINT_SHA"
    config["post_fusion_score_threshold"] = None
    result = {
        "version": "paired_deployment_regression_v1",
        "engineering_only": True,
        "bundle_sha256": sha(bundle / "contract.json"),
        "placements": placements,
        "canvas_size": [10000, 10000],
        "translations": [[0, 0], [383, 511]],
        "background_fill": [0, 0, 0],
        "native_pixels_no_resize_no_crop": True,
        "purpose": "sparse/dense + changed tile seams; not hidden-set score prediction",
        "submission_config": config,
        "submission_template_sha256": sha(config_path),
        "background_manifest_sha256": read(bundle / "contract.json")["deployment_regression"][
            "background_manifest_sha256"
        ],
    }
    write(output, result)
    return result


def canvas(contract: dict, data_root: Path, index: int) -> Image.Image:
    image = Image.new("RGB", tuple(contract["canvas_size"]), tuple(contract["background_fill"]))
    dx, dy = contract["translations"][index]
    for p in contract["placements"]:
        with Image.open(safe_path(data_root, p["source"]["file_name"])) as source:
            if source.size != (p["source"]["width"], p["source"]["height"]):
                raise ValueError("source dimension changed")
            image.paste(source.convert("RGB"), (p["x"] + dx, p["y"] + dy))
    return image


def execute(
    bundle: Path,
    config_path: Path,
    data_root: Path,
    background: Path,
    checkpoint: Path,
    review: Path,
    output: Path,
    device: str,
) -> dict:
    frozen = validate_bundle(bundle, PROJECT, data_root)
    config = read(config_path)
    r = read(review / "review.json")
    if (
        config["bundle_sha256"] != sha(bundle / "contract.json")
        or r["bundle_sha256"] != config["bundle_sha256"]
    ):
        raise ValueError("regression/review bundle mismatch")
    for rel, digest in r["artifacts"].items():
        if sha(safe_path(review, rel)) != digest:
            raise ValueError("review artifact tampered")
    validate_lineage(read(review / "lineage.json"), bundle, checkpoint)
    threshold = read(review / "threshold.json")["threshold"]
    manifest = background / "background_100mp_manifest.jsonl"
    if sha(manifest) != config["background_manifest_sha256"]:
        raise ValueError("background manifest changed")
    bg = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    for row in bg:
        if sha(safe_path(background, row["file_name"])) != row["sha256"]:
            raise ValueError("background image changed")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    bg_gt = output / "background_gt.json"
    write(
        bg_gt,
        {
            "images": [{"id": row["image_id"], "file_name": row["file_name"]} for row in bg],
            "annotations": [],
        },
    )
    predictions = infer(
        checkpoint, bg_gt, background, output / "background", frozen["inference"], device
    )
    retained = [p for p in read(predictions) if p["score"] >= threshold]
    megapixels = sum(row["width"] * row["height"] for row in bg) / 1e6

    import torch

    from rsdet.evaluation.official_metric import evaluate_predictions
    from rsdet.evaluation.protocol import parse_evaluation_protocol
    from rsdet.pipeline.large_image import run_pipeline
    from rsdet.submission.competition import CompetitionDetector, validate_result_payload
    from rsdet.utils.config import load_config

    runtime_config = config["submission_config"]
    runtime_config["device"] = device
    runtime_config["model"].update(
        weight_path=str(checkpoint.resolve()), expected_sha256=sha(checkpoint)
    )
    runtime_config["post_fusion_score_threshold"] = threshold
    write(output / "runtime_config.json", runtime_config)
    detector = CompetitionDetector(runtime_config)
    with Image.open(safe_path(data_root, config["placements"][0]["source"]["file_name"])) as warmup:
        detector.predict(warmup.convert("RGB"))
    timings, checks, result_images = [], [], []
    for index in range(2):
        image = canvas(config, data_root, index)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        objects = detector.predict(image)
        item = {
            "image_id": f"regression_{index}",
            "file_name": f"regression_{index}.png",
            "width": 10000,
            "height": 10000,
            "objects": objects,
            "run_end_timestamp": int(time.time() * 1000),
        }
        json.dumps(item, allow_nan=False)
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
        result_images.append(item)
        # Independent offline pipeline call, then compare exact exported boxes.
        raw, _ = run_pipeline(
            np.asarray(image), detector.detector, config=detector.pipeline_config, parent_image_id=0
        )
        expected = []
        for box, score, label in zip(raw.boxes_xyxy, raw.scores, raw.labels):
            x1, y1, x2, y2 = np.clip(np.array(box, dtype=float), 0, 10000)
            if score >= threshold and x2 > x1 and y2 > y1:
                expected.append([int(label), x1, y1, x2, y2, float(score)])
        actual = [[o["category_id"], *o["bbox"], o["score"]] for o in objects]
        expected.sort()
        actual.sort()
        if len(expected) != len(actual) or (
            actual and not np.allclose(actual, expected, atol=1e-5, rtol=0)
        ):
            raise ValueError("offline/entry per-box parity failed")
        dx, dy = config["translations"][index]
        gt_rows = []
        for placement in config["placements"]:
            for a in placement["annotations"]:
                x, y, w, h = a["bbox"]
                x += placement["x"] + dx
                y += placement["y"] + dy
                gt_rows.append({"category_id": a["category_id"], "bbox_xyxy": [x, y, x + w, y + h]})
        protocol = parse_evaluation_protocol(load_config(PROJECT))
        matched = evaluate_predictions(
            {0: gt_rows},
            {
                0: [
                    {"category_id": o["category_id"], "bbox_xyxy": o["bbox"], "score": o["score"]}
                    for o in objects
                ]
            },
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
        )
        checks.append(
            {
                "canvas": index,
                "objects": len(actual),
                "per_box_parity": True,
                "known_gt": len(gt_rows),
                "tp": matched.details["tp"],
                "fp": matched.details["fp"],
                "fn": matched.details["fn"],
                "partial_taxonomy_engineering_counts_not_platform_score": True,
            }
        )
        image.close()
    payload = {"status": "success", "images": result_images}
    counts = validate_result_payload(payload)
    write(output / "result.json", payload)
    gpu = torch.cuda.get_device_properties(device)
    result = {
        "status": "pass",
        "engineering_only": True,
        "container_gpu_tested": False,
        "bundle_sha256": sha(bundle / "contract.json"),
        "config_sha256": sha(config_path),
        "checkpoint_sha256": sha(checkpoint),
        "threshold": threshold,
        "background_fp": len(retained),
        "background_fp_per_100mp": 100 * len(retained) / megapixels,
        "background_fp_by_fine": dict(Counter(p["category_id"] for p in retained)),
        "background_megapixels": megapixels,
        "entrypoint_parity": checks,
        "output_counts": counts,
        "latency_seconds": sum(timings) / len(timings),
        "timings": timings,
        "timing_scope": "warm loaded model; preprocessing+safe tiling+inference+fusion+serialization; excludes image IO",
        "gpu": gpu.name,
        "gpu_uuid": str(getattr(gpu, "uuid", "unavailable")),
        "torch": torch.__version__,
        "automatic_full_or_submission_admission": False,
    }
    write(output / "regression.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "run"))
    p.add_argument("--bundle", type=Path, default=ROOT / "data/splits/paired_trend_v1")
    p.add_argument(
        "--config", type=Path, default=ROOT / "configs/experiments/paired_deployment_v1.json"
    )
    p.add_argument("--data-root", type=Path, default=ROOT.parent / "data")
    p.add_argument(
        "--background", type=Path, default=ROOT / "outputs/MACROSHIFT-BACKGROUND-100MP-FROZEN"
    )
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--review", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.command == "freeze":
        result = freeze_canvases(a.bundle, a.config)
    else:
        if not all((a.checkpoint, a.review, a.output)):
            p.error("run requires --checkpoint, --review, --output")
        result = execute(
            a.bundle,
            a.config,
            a.data_root,
            a.background,
            a.checkpoint,
            a.review,
            a.output,
            a.device,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
