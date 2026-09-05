#!/usr/bin/env python3
"""Measure resident CE identity/D4 cost, requiring exact frozen per-box replay."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_p40_vehicle_zoom import read, sha
from r1_aircraft_refinement import _load_adapted_model, _normalize

from rsdet.data.crop_classification import render_crop
from rsdet.evaluation.coco import load_coco_predictions
from rsdet.features.p04_inputs import D4_VIEW_IDS, apply_d4_view
from rsdet.postprocess.aircraft_relabel import relabel_aircraft


def classify(model, source, image_id, rows, views, batch_objects):
    import torch

    indices = [j for j, row in enumerate(rows) if 4 <= row["category_id"] < 24]
    bundles = []
    for start in range(0, len(indices), batch_objects):
        selected = indices[start : start + batch_objects]
        tensors = []
        for j in selected:
            crop = render_crop(source, rows[j]["bbox_xyxy"], 224)
            tensors.extend(_normalize(apply_d4_view(crop, view)) for view in views)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(torch.stack(tensors).cuda()).reshape(len(selected), len(views), 25)
        probabilities = logits.float()[:, :, 4:24].softmax(2).mean(1).cpu().numpy()
        bundles.extend(
            {
                "image_id": image_id,
                "prediction_index": j,
                "old_category": rows[j]["category_id"],
                "probabilities": probs.tolist(),
            }
            for j, probs in zip(selected, probabilities, strict=True)
        )
    return bundles


def canonical(rows):
    return sorted((row["category_id"], row["score"], tuple(row["bbox_xyxy"])) for row in rows)


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = Path("configs/experiments/p40_aircraft_runtime_v1.json")
    cfg = read(config_path)
    parent = read("configs/experiments/p40_rot90_complement_v1.json")
    imagenet = Path("/root/autodl-tmp/pretrained/convnext_tiny-983f1562.pth")
    if sha(imagenet) != "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d":
        raise ValueError("ImageNet asset mismatch")
    datasets, hashes = {}, {"imagenet": sha(imagenet)}
    for condition, root_name in (
        ("hard", "pseudo10k-trial-mix-local"),
        ("sentinel", "pseudo10k-trial-mix-sentinel-v1"),
    ):
        root = Path("/root/autodl-tmp") / root_name
        gt_path = root / "ground_truth.json"
        if sha(gt_path) != parent["ground_truth_sha256"][condition]:
            raise ValueError("GT lineage mismatch")
        images = read(gt_path)["images"]
        cache_root = (
            args.results
            / "SCALEROUTE-PLAN15-PROGRESSIVE40-FIXED-BENCHMARKS-V1"
            / condition
            / "progressive_i1280"
        )
        pred_path = cache_root / "predictions.json"
        raw = load_coco_predictions(pred_path)
        cache = {}
        for mode in cfg["views"]:
            folder = (
                args.results / "P40-AIRCRAFT-CE-IDENTITY-V1" / condition
                if mode == "identity"
                else args.results
                / "P40-AIRCRAFT-CE-D4-V1"
                / ("sentinel" if condition == "sentinel" else "")
            )
            preflight = read(folder / "preflight.json")
            if sha(pred_path) != preflight["pred_sha256"] or sha(gt_path) != preflight["gt_sha256"]:
                raise ValueError("prediction/GT differs from classified cache")
            hashes[f"{condition}/{mode}/preflight"] = sha(folder / "preflight.json")
            hashes[f"{condition}/{mode}/probabilities"] = sha(folder / "probabilities.json")
            cache[mode] = {"preflight": preflight, "bundles": read(folder / "probabilities.json")}
        if cache["identity"]["preflight"]["thresholds"] != cache["d4"]["preflight"]["thresholds"]:
            raise ValueError("paired threshold mismatch")
        thresholds = cache["identity"]["preflight"]["thresholds"]
        base = {
            image["id"]: [
                row
                for row in raw.get(image["id"], [])
                if row["score"] >= thresholds[str(image["fold"])]
            ]
            for image in images
        }
        datasets[condition] = dict(root=root, images=images, base=base, cache=cache)
    (args.output / "preflight.json").write_text(
        json.dumps(
            {
                "config": cfg,
                "config_sha256": sha(config_path),
                "inputs": hashes,
                "code_sha256": {
                    str(path): sha(path)
                    for directory in ("src", "scripts", "configs")
                    for path in sorted(Path(directory).rglob("*"))
                    if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
                },
            },
            indent=2,
        )
        + "\n"
    )
    Image.MAX_IMAGE_PIXELS = None
    records, model_loads, image_hashes = [], [], {}
    for fold in range(3):
        asset = datasets["hard"]["cache"]["identity"]["preflight"]["classifier_assets"][str(fold)]
        checkpoint = Path(asset["path"])
        if sha(checkpoint) != asset["sha256"]:
            raise ValueError("classifier checkpoint changed")
        for dataset in datasets.values():
            for cache in dataset["cache"].values():
                if cache["preflight"]["classifier_assets"][str(fold)]["sha256"] != asset["sha256"]:
                    raise ValueError("nonpaired classifier assets")
        start = time.perf_counter()
        model, _ = _load_adapted_model(checkpoint, imagenet, fold=fold, method="ce")
        model.cuda().eval()
        torch.cuda.synchronize()
        model_loads.append(time.perf_counter() - start)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for _ in range(cfg["warmup_batches_per_fold"]):
                model(torch.zeros((cfg["batch_objects"] * 8, 3, 224, 224), device="cuda"))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for condition, dataset in datasets.items():
            for image in dataset["images"]:
                if image["fold"] != fold:
                    continue
                i = image["id"]
                path = dataset["root"] / f"fold_{fold}/images" / image["file_name"]
                image_hashes[f"{condition}/{i}"] = sha(path)
                rows = dataset["base"][i]
                expected = {
                    mode: relabel_aircraft(
                        {i: rows}, [b for b in cache["bundles"] if b["image_id"] == i]
                    )
                    for mode, cache in dataset["cache"].items()
                }
                for repeat in range(cfg["timed_repeats_per_image"]):
                    for mode in cfg["views"][:: (-1 if repeat % 2 else 1)]:
                        views = D4_VIEW_IDS[:1] if mode == "identity" else D4_VIEW_IDS
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        with Image.open(path) as source:
                            source.load()
                            bundles = classify(model, source, i, rows, views, cfg["batch_objects"])
                        actual = relabel_aircraft({i: rows}, bundles)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - started
                        if canonical(actual[i]) != canonical(expected[mode][i]):
                            raise AssertionError(f"cached decisions differ: {condition}/{i}/{mode}")
                        records.append(
                            dict(
                                condition=condition,
                                image_id=i,
                                fold=fold,
                                mode=mode,
                                repeat=repeat,
                                aircraft_boxes=len(bundles),
                                seconds=elapsed,
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                exact_decisions=True,
                            )
                        )
                (args.output / "status.txt").write_text(f"complete_fold_{fold}_{condition}_{i}\n")
        del model
        torch.cuda.empty_cache()
    summary = {}
    for condition in datasets:
        summary[condition] = {}
        for mode in cfg["views"]:
            rows = [r for r in records if r["condition"] == condition and r["mode"] == mode]
            seconds = [r["seconds"] for r in rows]
            summary[condition][mode] = {
                "mean_seconds": float(np.mean(seconds)),
                "p50_seconds": float(np.median(seconds)),
                "p95_seconds": float(np.quantile(seconds, 0.95)),
                "max_seconds": max(seconds),
                "n_image_repeats": len(seconds),
            }
    result = {
        "status": "complete",
        "experiment_id": cfg["experiment_id"],
        "summary": summary,
        "records": records,
        "cold_model_load_seconds_per_fold": model_loads,
        "image_sha256": image_hashes,
        "all_cached_decisions_exact": True,
        "scope": cfg["timing_scope"],
        "not_docker_or_full_pipeline_latency": True,
        "filesystem_cache_may_be_warm": True,
        "formal_admission": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "threads": torch.get_num_threads(),
        },
    }
    (args.output / "runtime.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "status.txt").write_text("complete_exact_replay_and_resident_cost\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
