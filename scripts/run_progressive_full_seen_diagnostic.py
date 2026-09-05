#!/usr/bin/env python3
"""Replay the frozen P40 benchmarks with its full weight; NEVER an OOF result.

This diagnostic is explicitly authorized to contain training-source overlap.
It cannot select thresholds, admit a model, train, package, or submit anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.platform_protocol import (
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
FULL_NAME = "SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-3GPU-R2"
CV3_NAME = "SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1"
WEIGHT_SHA = "904c4935a85484a83d98930b0862bd1b5a1b0e9e7c6ed4eea7525391d383123f"
THRESHOLD = 0.536
SAFETY = {
    "data_leakage": True,
    "is_oof": False,
    "eligible_for_admission": False,
    "official_score_prediction": False,
    "warning": "Full training saw all Normal images and the source images of Hard/Sentinel. "
    "Scores are seen-source diagnostics, not generalization or official score forecasts.",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def metrics(gt_path: Path, pred_path: Path, latency: float | None) -> dict:
    document = read(gt_path)
    image_ids = {int(row["id"]) for row in document["images"]}
    if len(image_ids) != len(document["images"]):
        raise ValueError("duplicate GT image IDs")
    for row in read(pred_path):
        if int(row["image_id"]) not in image_ids or int(row["category_id"]) not in range(25):
            raise ValueError("prediction identity/taxonomy mismatch")
        if not all(math.isfinite(float(v)) for v in [row["score"], *row["bbox"]]):
            raise ValueError("nonfinite prediction")
        if not 0 <= float(row["score"]) <= 1 or min(row["bbox"][2:]) <= 0:
            raise ValueError("invalid score/box")
    protocol = parse_evaluation_protocol(load_config(REPO / "configs/project.yaml"))
    gt = load_coco_ground_truth(gt_path)
    raw = load_coco_predictions(pred_path)
    pred = {i: [r for r in raw.get(i, []) if r["score"] >= THRESHOLD] for i in gt}
    kwargs = dict(class_names=protocol.class_names, category_mapping=protocol.category_mapping,
                  iou_thresholds=protocol.iou_thresholds)
    pooled = evaluate_predictions(gt, pred, **kwargs)
    ranking = evaluate_ranking_metrics(gt, pred, require_complete_taxonomy=True, **kwargs)
    platform = platform_metrics_payload(build_platform_observed_metrics(
        ranking, recall_min=protocol.recall_min, fdr_max=protocol.fdr_max,
        latency_seconds=latency, latency_max_seconds=protocol.latency_max_seconds,
    ))
    return {
        "images": len(image_ids), "gt_objects": len(document["annotations"]),
        "gt_sha256": sha(gt_path), "prediction_sha256": sha(pred_path),
        "threshold": THRESHOLD, "threshold_retuned": False,
        "predictions_after_threshold": sum(map(len, pred.values())),
        "platform": platform, "latency_seconds": latency,
        "pooled_diagnostic": {"recall": pooled.recall, "fdr": pooled.fdr},
        "per_fine": {str(k): {**asdict(v), "recall": v.recall, "fdr": v.fdr}
                     for k, v in ranking.per_fine.items()},
    }


def pseudo_provenance(folder: Path) -> float:
    summary = read(folder / "run_summary.json")
    if sha(folder / "predictions.json") != summary["predictions_sha256"]:
        raise ValueError("pseudo prediction SHA mismatch")
    if (summary["family"] != "yolo" or summary["score_floor"] != 0.001
            or summary.get("score_transform") is not None or summary["coarse_label_space"]):
        raise ValueError("wrong pseudo inference configuration")
    p = summary["pipeline"]
    for key, value in {"tile_size": 1024, "overlap": 256, "batch_size": 4,
                       "fusion": "safe", "max_detections": 4000}.items():
        if p[key] != value:
            raise ValueError(f"wrong pipeline {key}")
    if sorted(f["fold"] for f in summary["folds"]) != [0, 1, 2]:
        raise ValueError("missing pseudo partition")
    times = []
    for f in summary["folds"]:
        if f["weight_sha256"] != WEIGHT_SHA or f["images"] != 2:
            raise ValueError("wrong checkpoint or pseudo image coverage")
        if f.get("agreement_adapter") is not None:
            raise ValueError("unexpected agreement adapter")
        times.extend(x["wall_seconds"] for x in f["image_timings"])
    if len(times) != 6:
        raise ValueError("wrong timing coverage")
    return fmean(times)


def run(command: list[str], log: Path, gpu: int | None = None):
    env = {**os.environ, "PYTHONPATH": f"{REPO}:{REPO / 'src'}",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        return subprocess.Popen([sys.executable, *command], cwd=REPO, env=env,
                                stdout=handle, stderr=subprocess.STDOUT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root, args.output
    if out.exists():
        raise FileExistsError("refusing to overwrite an existing diagnostic")
    out.mkdir(parents=True)
    full, cv3 = root / "results" / FULL_NAME, root / "results" / CV3_NAME
    weight = full / "adaptation/runs/resolution_adaptation/weights/last.pt"
    jobs = []
    try:
        if sha(weight) != WEIGHT_SHA:
            raise ValueError("wrong full checkpoint")
        validation = read(full / "validation_summary.json")
        if validation["deployment_threshold"] != THRESHOLD:
            raise ValueError("wrong frozen threshold")
        frontier = read(cv3 / "aggregate/crossfit_frontier.json")
        full_names = {Path(s).name for s in (
            root / "results/Y5-FULL-S1280-3GPU-R1/all_train_images.txt"
        ).read_text().splitlines() if s.strip()}
        if len(full_names) != 4481:
            raise ValueError("wrong full training inventory")
        audit = {**SAFETY, "checkpoint": str(weight), "checkpoint_sha256": sha(weight),
                 "threshold": THRESHOLD, "hardware_migration": validation["hardware_migration"],
                 "full_training_history": validation["effective_training_history"],
                 "normal_partitions": [], "pseudo_sources": {}, "code_sha256": {}}
        files = ["scripts/train_progressive_resolution_adaptation.py",
                 "scripts/resume_progressive_resolution_ddp.py",
                 "src/rsdet/innovation/progressive_resume.py", "src/rsdet/innovation/trainers.py",
                 "src/rsdet/innovation/rotate90.py", "src/rsdet/models/ultralytics_adapter.py",
                 "scripts/infer_cv3_oof.py", "scripts/run_multifamily_cv3_pseudo_eval.py",
                 "src/rsdet/pipeline/large_image.py", "src/rsdet/evaluation/official_metric.py",
                 "src/rsdet/evaluation/platform_protocol.py", "configs/project.yaml",
                 "scripts/run_progressive_full_seen_diagnostic.py"]
        audit["code_sha256"] = {f: sha(REPO / f) for f in files}
        seen = set()
        normal = out / "normal"
        normal.mkdir()
        for fold in range(3):
            assets = root / ("capscale-assets" if fold == 0 else f"capscale-cv3-assets/fold_{fold}")
            manifest, gt_path = assets / "split_view.json", assets / "instances_val.json"
            gt = read(gt_path)
            rows = [s for s in read(manifest)["samples"] if s["split"] == "val"]
            ids = {int(s["image_id"]) for s in rows}
            if (ids != {int(i["id"]) for i in gt["images"]} or ids & seen
                    or ids != set(map(int, frontier["fold_image_ids"][str(fold)]))):
                raise ValueError("Normal manifest/GT/frontier identity mismatch")
            source = cv3 / f"fold_{fold}/resolved_infer.yaml"
            config = yaml.safe_load(source.read_text())
            old_manifest = config["input"]["manifest"]
            model = config["model"]
            for k, value in {"family": "yolo", "imgsz": 1280, "confidence": 0.001,
                             "max_detections": 500, "iou": 0.7, "half": True}.items():
                if model.get(k) != value:
                    raise ValueError(f"unexpected model field {k}")
            if config["tiling"]["enabled"] or any(model.get(k) for k in
                    ["label_map", "drop_labels", "score_transform", "agnostic_nms"]):
                raise ValueError("unexpected Normal routing or label transformation")
            for s in rows:
                image = Path(config["input"]["data_root"]) / s["relative_path"]
                canonical = root / "data" / s["relative_path"]
                if image.name not in full_names or not image.is_file():
                    raise ValueError("missing/wrong Normal input")
                if not image.samefile(canonical) and sha(image) != sha(canonical):
                    raise ValueError("fold data image differs from full training image")
            model["checkpoint"] = str(weight)
            config["input"]["manifest"] = str(manifest)
            config["output_json"] = str(normal / f"fold_{fold}/predictions_low.json")
            config["device"] = "cuda:0"
            folder = normal / f"fold_{fold}"
            folder.mkdir()
            (folder / "resolved_infer.yaml").write_text(yaml.safe_dump(config))
            shutil.copy2(gt_path, folder / "ground_truth.json")
            audit["normal_partitions"].append({"fold": fold, "images": len(ids),
                "source_config_sha256": sha(source), "manifest_sha256": sha(manifest),
                "gt_sha256": sha(gt_path), "original_manifest": old_manifest,
                "resolved_manifest": str(manifest), "all_images_equal_full_training": True})
            seen |= ids
        if len(seen) != 4481:
            raise ValueError("Normal must cover all 4481 images exactly once")
        for name, directory, expected in [
            ("hard", "pseudo10k-trial-mix-local", (6, 2158)),
            ("sentinel", "pseudo10k-trial-mix-sentinel-v1", (6, 1969)),
        ]:
            gt_path = root / directory / "ground_truth.json"
            gt = read(gt_path)
            if (len(gt["images"]), len(gt["annotations"])) != expected:
                raise ValueError("pseudo GT inventory changed")
            sources = {Path(s).name for i in gt["images"] for s in i["source_images"]}
            if not sources <= full_names or len(sources) != 600:
                raise ValueError("unexpected pseudo source inventory")
            audit["pseudo_sources"][name] = {"sources": len(sources), "full_seen": len(sources),
                                             "gt_sha256": sha(gt_path)}
        pseudo_provenance(full / "timing_only_hard")
        write(out / "audit.json", audit)
        write(out / "status.json", {"status": "running_normal", **SAFETY})
        for fold in range(3):
            folder = normal / f"fold_{fold}"
            jobs.append(run(["scripts/infer_cv3_oof.py", "--config",
                             str(folder / "resolved_infer.yaml")], folder / "inference.log", fold))
        # Wait for every partition even if one fails; never leave untracked children.
        codes = [p.wait() for p in jobs]
        if any(codes):
            raise RuntimeError(f"Normal inference failed: {codes}")
        predictions, images, annotations = [], [], []
        for fold in range(3):
            folder = normal / f"fold_{fold}"
            pred_path = folder / "predictions_low.json"
            runtime = read(folder / "predictions_low.runtime.json")
            if (runtime["artifacts"]["checkpoint"]["sha256"] != WEIGHT_SHA
                    or runtime["artifacts"]["predictions"]["sha256"] != sha(pred_path)
                    or runtime["images"] != audit["normal_partitions"][fold]["images"]):
                raise ValueError("Normal inference provenance mismatch")
            gt = read(folder / "ground_truth.json")
            images.extend(gt["images"])
            annotations.extend(gt["annotations"])
            predictions.extend(read(pred_path))
        write(normal / "ground_truth.json", {"images": images, "annotations": annotations,
                                               "categories": gt["categories"], **SAFETY})
        write(normal / "predictions.json", predictions)
        write(out / "status.json", {"status": "running_sentinel", **SAFETY})
        process = run(["scripts/run_multifamily_cv3_pseudo_eval.py", "--pseudo-root",
            str(root / "pseudo10k-trial-mix-sentinel-v1"), "--family", "yolo", "--weights",
            *[str(weight)] * 3, "--output-dir", str(out / "sentinel"), "--score-floor", "0.001",
            "--batch-size", "4", "--device", "cuda:0", "--imgsz", "1280",
            "--tile-size", "1024", "--overlap", "256"], out / "sentinel/inference.log", 0)
        jobs.append(process)
        if process.wait():
            raise RuntimeError("Sentinel inference failed")
        results = {}
        for name, folder, gt_path in [
            ("normal", normal, normal / "ground_truth.json"),
            ("hard", full / "timing_only_hard", root / "pseudo10k-trial-mix-local/ground_truth.json"),
            ("sentinel", out / "sentinel", root / "pseudo10k-trial-mix-sentinel-v1/ground_truth.json"),
        ]:
            latency = None if name == "normal" else pseudo_provenance(folder)
            result = {**SAFETY, "benchmark": name,
                      "hard_predictions_reused": name == "hard",
                      "legacy_engine_oof_label_is_not_applicable": True,
                      **metrics(gt_path, folder / "predictions.json", latency)}
            write(out / name / "seen_diagnostic_metrics.json", result)
            results[name] = result
            if name != "normal":
                shutil.copy2(gt_path, out / name / "ground_truth.json")
        shutil.copy2(full / "timing_only_hard/predictions.json", out / "hard/predictions.json")
        shutil.copy2(full / "timing_only_hard/run_summary.json", out / "hard/run_summary.json")
        write(out / "summary.json", {"status": "complete_seen_diagnostic_only", **SAFETY,
              "checkpoint_sha256": WEIGHT_SHA, "threshold": THRESHOLD, "benchmarks": results})
        write(out / "status.json", {"status": "complete_seen_diagnostic_only", **SAFETY})
        paths = sorted(p for p in out.rglob("*") if p.is_file())
        (out / "SHA256SUMS.txt").write_text("".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in paths))
        print(json.dumps({name: r["platform"] for name, r in results.items()}, indent=2))
    except BaseException as exc:
        for process in jobs:
            if process.poll() is None:
                process.terminate()
        for process in jobs:
            process.wait()
        write(out / "status.json", {"status": "failed", "error": str(exc), **SAFETY})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
