#!/usr/bin/env python3
"""One entry point: freeze -> prepare/train baseline -> paired candidate review.

No Docker build, full-data fit, submission, threshold scan on confirmation,
or automatic resume. Training is opt-in via --execute; preflight is CPU-only.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rsdet.analysis.oof_detection import build_threshold_curve
from rsdet.data.xh_dataset import FINE_NAMES
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.experiments.paired_trend import (
    OFFICIAL_WEIGHT_SHA,
    VERSION,
    freeze,
    read,
    safe_path,
    sha,
    validate_bundle,
    write,
)
from rsdet.postprocess.calibration import build_threshold_grid
from rsdet.utils.config import load_config
from scripts.compare_candidate_trend import compare
from scripts.evaluate_fixed_score_threshold import _metrics_payload

PROJECT = ROOT / "configs/project.yaml"


def environment_versions() -> dict:
    result = {"python": sys.version.split()[0]}
    for package in ("torch", "torchvision", "ultralytics", "albumentations", "numpy", "Pillow"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not_installed"
    return result


def code_fingerprint() -> dict:
    paths = list((ROOT / "src").rglob("*.py")) + [
        Path(__file__).resolve(),
        ROOT / "scripts/compare_candidate_trend.py",
        ROOT / "scripts/evaluate_fixed_score_threshold.py",
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def materialize(bundle: Path, data_root: Path, weights: Path, out: Path, device: str) -> dict:
    contract = validate_bundle(bundle, PROJECT, data_root)
    if sha(weights) != OFFICIAL_WEIGHT_SHA:
        raise ValueError(
            "baseline requires audited official YOLO26s initialization, not any full weight"
        )
    if "," in device:
        raise ValueError("v1 reference is single GPU; no implicit DDP/batch change")
    samples = read(bundle / "manifest.json")["samples"]
    train = [s for s in samples if s["split"] == "train"]
    plan = {
        "version": VERSION,
        "bundle_sha256": sha(bundle / "contract.json"),
        "data_root": str(data_root.resolve()),
        "weights": str(weights.resolve()),
        "weight_sha256": sha(weights),
        "device": device,
        "recipe": contract["baseline"],
        "code": code_fingerprint(),
        "train_image_ids": sorted(s["image_id"] for s in train),
        "holdouts_used_for_training_or_checkpoint_selection": False,
    }
    if (out / "plan.json").exists():
        if read(out / "plan.json") != plan:
            raise ValueError("prepared plan differs; use a fresh run directory")
        if sha(out / "train.txt") != read(out / "materialized.json")["train_sha256"]:
            raise ValueError("training list modified")
        if sha(out / "dataset.yaml") != read(out / "materialized.json")["dataset_sha256"]:
            raise ValueError("dataset YAML modified")
        return plan
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    write(out / "plan.json", plan)
    (out / "train.txt").write_text(
        "\n".join(str(safe_path(data_root, s["relative_path"])) for s in train) + "\n"
    )
    # Ultralytics may final-validate even with val=False. It must never see holdouts.
    dataset = {
        "path": str(data_root.resolve()),
        "train": str((out / "train.txt").resolve()),
        "val": str((out / "train.txt").resolve()),
        "nc": 25,
        "names": list(FINE_NAMES),
    }
    (out / "dataset.yaml").write_text(yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False))
    write(
        out / "materialized.json",
        {
            "train_sha256": sha(out / "train.txt"),
            "dataset_sha256": sha(out / "dataset.yaml"),
            "train_images": len(train),
        },
    )
    return plan


def check_epochs(path: Path, epochs: int) -> None:
    with path.open() as handle:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(handle)]
    if len(rows) != epochs or [int(r["epoch"]) for r in rows] != list(range(1, epochs + 1)):
        raise ValueError("checkpoint stage is not exactly the frozen epoch sequence")
    loss_keys = [k for k in rows[0] if k.startswith("train/") and "loss" in k]
    if not loss_keys or any(not math.isfinite(float(r[k])) for r in rows for k in loss_keys):
        raise ValueError("missing or nonfinite training losses")


def train_baseline(bundle: Path, data_root: Path, weights: Path, out: Path, device: str) -> Path:
    plan = materialize(bundle, data_root, weights, out, device)
    import torch
    from ultralytics import YOLO

    from rsdet.innovation.trainers import rotate90_augmentations

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: preflight ready, real baseline training not started")
    checkpoint = weights
    ancestors = [
        {"checkpoint_sha256": OFFICIAL_WEIGHT_SHA, "kind": "audited_official_initialization"}
    ]
    for name in ("foundation", "adaptation"):
        config = plan["recipe"][name]
        stage = out / name
        last = stage / "run/weights/last.pt"
        receipt = stage / "complete.json"
        if receipt.exists():
            r = read(receipt)
            if (
                r["checkpoint_sha256"] != sha(last)
                or r["initial_sha256"] != sha(checkpoint)
                or r["plan_sha256"] != sha(out / "plan.json")
                or r["results_sha256"] != sha(stage / "run/results.csv")
                or r["environment"] != environment_versions()
            ):
                raise ValueError("completed stage cache mismatch")
            check_epochs(stage / "run/results.csv", config["epochs"])
        else:
            if stage.exists():
                raise FileExistsError(f"partial stage preserved; no implicit resume: {stage}")
            stage.mkdir()
            model = YOLO(str(checkpoint.resolve()))
            model.train(
                data=str((out / "dataset.yaml").resolve()),
                project=str(stage.resolve()),
                name="run",
                exist_ok=False,
                augmentations=rotate90_augmentations(p=1.0),
                optimizer="AdamW",
                weight_decay=0.0005,
                cos_lr=True,
                amp=True,
                deterministic=True,
                seed=42,
                patience=0,
                val=False,
                plots=False,
                save=True,
                workers=4,
                device=device,
                **config,
            )
            if list(model.names.values()) != list(FINE_NAMES):
                raise ValueError("wrong trained fine taxonomy")
            check_epochs(stage / "run/results.csv", config["epochs"])
            write(
                receipt,
                {
                    "checkpoint_sha256": sha(last),
                    "initial_sha256": sha(checkpoint),
                    "plan_sha256": sha(out / "plan.json"),
                    "results_sha256": sha(stage / "run/results.csv"),
                    "stage": name,
                    "config": config,
                    "environment": environment_versions(),
                },
            )
        ancestors.append(
            {
                "checkpoint_sha256": sha(last),
                "kind": "trained_on_frozen_train",
                "train_image_ids": plan["train_image_ids"],
            }
        )
        checkpoint = last
    lineage = {
        "bundle_sha256": plan["bundle_sha256"],
        "checkpoint_sha256": sha(checkpoint),
        "train_image_ids": plan["train_image_ids"],
        "ancestors": ancestors,
        "training_recipe": plan["recipe"],
        "training_plan_sha256": sha(out / "plan.json"),
        "external_training_data": False,
    }
    if (out / "lineage.json").exists():
        if read(out / "lineage.json") != lineage:
            raise ValueError("baseline lineage changed")
    else:
        write(out / "lineage.json", lineage)
    return checkpoint


def validate_lineage(lineage: dict, bundle: Path, checkpoint: Path) -> None:
    if lineage["bundle_sha256"] != sha(bundle / "contract.json") or lineage[
        "checkpoint_sha256"
    ] != sha(checkpoint):
        raise ValueError("lineage checkpoint/split SHA mismatch")
    allowed = {
        s["image_id"] for s in read(bundle / "manifest.json")["samples"] if s["split"] == "train"
    }
    if set(lineage["train_image_ids"]) != allowed:
        raise ValueError("candidate must use exactly the frozen training universe")
    ancestors = lineage["ancestors"]
    if not ancestors or ancestors[0] != {
        "checkpoint_sha256": OFFICIAL_WEIGHT_SHA,
        "kind": "audited_official_initialization",
    }:
        raise ValueError("unverified initialization ancestry")
    if lineage.get("external_training_data") is not False or len(ancestors) < 2:
        raise ValueError("unverified training lineage")
    for index, row in enumerate(ancestors[1:], 1):
        if set(row["train_image_ids"]) != allowed:
            raise ValueError("initialization contains heldout/unverified training sources")
        if row["kind"] == "trained_on_frozen_train":
            continue
        if row["kind"] != "bn_recalibrated_on_frozen_train":
            raise ValueError("initialization contains heldout/unverified training sources")
        if (index != len(ancestors) - 1 or index < 2
                or ancestors[index - 1]["kind"] != "trained_on_frozen_train"
                or row.get("parent_checkpoint_sha256") != ancestors[index - 1]["checkpoint_sha256"]
                or row.get("non_bn_state_bitwise_equal") is not True
                or lineage.get("post_training_transform") != "PAIRED-P40-BN-TRAINONLY-V1"
                or any(len(row.get(k, "")) != 64 or any(c not in "0123456789abcdef" for c in row[k])
                       for k in ("plan_sha256", "receipt_sha256"))):
            raise ValueError("unverified BN transformation ancestry")
    if ancestors[-1]["checkpoint_sha256"] != sha(checkpoint):
        raise ValueError("lineage does not terminate at evaluated checkpoint")


def infer(
    checkpoint: Path, gt_path: Path, data_root: Path, output: Path, config: dict, device: str
) -> Path:
    """Record complete attempted image IDs, including images with zero detections."""
    pred = output / "predictions_low.json"
    request = {
        "checkpoint_sha256": sha(checkpoint),
        "gt_sha256": sha(gt_path),
        "config": config,
        "device": device,
        "environment": environment_versions(),
    }
    if (output / "inference.json").exists():
        receipt = read(output / "inference.json")
        if receipt["request"] != request or receipt["predictions_sha256"] != sha(pred):
            raise ValueError("inference cache mismatch")
        if receipt["image_ids"] != sorted(i["id"] for i in read(gt_path)["images"]):
            raise ValueError("inference cache coverage mismatch")
        return pred
    if output.exists():
        raise FileExistsError(f"partial inference preserved: {output}")
    import numpy as np
    from PIL import Image

    from rsdet.contracts import InferenceSample
    from rsdet.models.ultralytics_adapter import UltralyticsDetector

    detector = UltralyticsDetector(
        family="yolo",
        imgsz=config["imgsz"],
        confidence=config["score_floor"],
        iou=config["iou"],
        max_detections=config["max_det"],
        half=config["half"],
        agnostic_nms=False,
    )
    detector.load(str(checkpoint.resolve()))
    raw_names = detector._model.names
    if tuple(raw_names[i] for i in range(len(raw_names))) != FINE_NAMES:
        raise ValueError("inference checkpoint does not contain the frozen 25-fine taxonomy")
    detector.to(device)
    detector.eval()
    images = read(gt_path)["images"]
    records, visited = [], []
    for offset in range(0, len(images), config["batch"]):
        samples = []
        for row in images[offset : offset + config["batch"]]:
            with Image.open(safe_path(data_root, row["file_name"])) as img:
                rgb = np.asarray(img.convert("RGB"))
            samples.append(InferenceSample(row["id"], rgb, rgb.shape[1], rgb.shape[0]))
        outputs = detector.predict(samples)
        if sorted(p.image_id for p in outputs) != sorted(s.image_id for s in samples):
            raise ValueError("inference did not return every input image")
        for p in outputs:
            visited.append(p.image_id)
            for b, score, label in zip(p.boxes_xyxy, p.scores, p.labels):
                x1, y1, x2, y2 = map(float, b)
                records.append(
                    {
                        "image_id": p.image_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                        "category_id": int(label),
                    }
                )
    write(pred, records)
    write(
        output / "inference.json",
        {
            "request": request,
            "image_ids": sorted(visited),
            "predictions_sha256": sha(pred),
            "latency_seconds": None,
            "note": "crop batch timing is not official 100MP latency",
        },
    )
    return pred


def load_predictions(gt_path: Path, pred_path: Path) -> tuple[dict, dict]:
    gt, pred = load_coco_ground_truth(gt_path), load_coco_predictions(pred_path)
    if set(pred) - set(gt):
        raise ValueError("predictions outside frozen image universe")
    for rows in pred.values():
        for row in rows:
            box = row["bbox_xyxy"]
            if (
                row["category_id"] not in range(25)
                or not all(math.isfinite(x) for x in box)
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                raise ValueError("invalid prediction taxonomy/box")
    return gt, pred


def select_threshold(gt_path: Path, pred_path: Path, grid: dict) -> tuple[float, dict]:
    gt, pred = load_predictions(gt_path, pred_path)
    thresholds = sorted(set(build_threshold_grid(**grid) + [1.0]))
    protocol = parse_evaluation_protocol(load_config(PROJECT))
    curve, audit = build_threshold_curve(gt, pred, thresholds=thresholds, protocol=protocol)
    selected = max(
        curve,
        key=lambda p: (
            p["platform_quality_score"],
            p["platform_gate_recall"],
            -p["platform_gate_fdr"],
            p["threshold"],
        ),
    )
    return float(selected["threshold"]), {
        "selected": selected,
        "trace_audit": audit,
        "grid": grid,
        "selection": "development_only_max_six_quality_sum",
        "development_gt_sha256": sha(gt_path),
        "development_predictions_sha256": sha(pred_path),
    }


def metrics(gt_path: Path, pred_path: Path, threshold: float) -> dict:
    gt, pred = load_predictions(gt_path, pred_path)
    pred = {i: [p for p in pred.get(i, []) if p["score"] >= threshold] for i in gt}
    protocol = parse_evaluation_protocol(load_config(PROJECT))
    kwargs = dict(
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    pooled = evaluate_predictions(gt, pred, **kwargs)
    ranking = evaluate_ranking_metrics(gt, pred, require_complete_taxonomy=True, **kwargs)
    return {
        "input_sha256": {"gt": sha(gt_path), "pred": sha(pred_path)},
        "threshold": threshold,
        "latency_seconds": None,
        "image_coverage": {
            "negative_coverage_known": True,
            "evaluated_images": len(gt),
            "empty_gt_images": sum(not g for g in gt.values()),
        },
        **_metrics_payload(pooled, ranking, protocol),
        "per_fine": {
            str(k): {"tp": r.tp, "fp": r.fp, "fn": r.fn, "recall": r.recall, "fdr": r.fdr}
            for k, r in ranking.per_fine.items()
        },
    }


def evaluate(
    bundle: Path,
    data_root: Path,
    checkpoint: Path,
    lineage_path: Path,
    output: Path,
    device: str,
    baseline: Path | None = None,
) -> dict:
    contract = validate_bundle(bundle, PROJECT, data_root)
    lineage = read(lineage_path)
    validate_lineage(lineage, bundle, checkpoint)
    if lineage.get("post_training_transform") == "PAIRED-P40-BN-TRAINONLY-V1":
        from scripts.run_paired_bn_recalibration import validate_candidate

        if baseline is None:
            raise ValueError("BN candidate requires its original baseline, not a new baseline role")
        validate_candidate(lineage_path.parent, bundle, baseline.parent, checkpoint)
    if output.exists():
        raise FileExistsError(output)
    if baseline:
        b = read(baseline / "review.json")
        if b["role"] != "baseline" or b["bundle_sha256"] != sha(bundle / "contract.json"):
            raise ValueError("wrong baseline benchmark")
        for name, digest in b["artifacts"].items():
            if sha(safe_path(baseline, name)) != digest:
                raise ValueError("baseline cache tampered")
    output.mkdir(parents=True)
    write(output / "lineage.json", lineage)
    write(output / "code.json", code_fingerprint())
    pred = infer(
        checkpoint,
        bundle / "development_gt.json",
        data_root,
        output / "development",
        contract["inference"],
        device,
    )
    threshold, selected = select_threshold(
        bundle / "development_gt.json", pred, contract["threshold_grid"]
    )
    write(
        output / "threshold.json",
        {
            **selected,
            "threshold": threshold,
            "checkpoint_sha256": sha(checkpoint),
            "bundle_sha256": sha(bundle / "contract.json"),
        },
    )
    dev = metrics(bundle / "development_gt.json", pred, threshold)
    write(output / "development_metrics.json", dev)
    dev_comparison = compare(read(baseline / "development_metrics.json"), dev) if baseline else None
    confirmed = baseline is None or dev_comparison["direction"] == "positive"
    confirmation_comparison = None
    # Do not even open confirmation GT on the negative/small candidate path.
    if confirmed:
        pred = infer(
            checkpoint,
            bundle / "confirmation_gt.json",
            data_root,
            output / "confirmation",
            contract["inference"],
            device,
        )
        confirmation = metrics(bundle / "confirmation_gt.json", pred, threshold)
        write(output / "confirmation_metrics.json", confirmation)
        if baseline:
            confirmation_comparison = compare(
                read(baseline / "confirmation_metrics.json"), confirmation
            )
    direction_ok = bool(
        confirmation_comparison and confirmation_comparison["delta_quality_contribution"] > 0
    )
    result = {
        "version": VERSION,
        "role": "candidate" if baseline else "baseline",
        "bundle_sha256": sha(bundle / "contract.json"),
        "threshold": threshold,
        "development": dev_comparison,
        "confirmation": confirmation_comparison,
        "confirmation_evaluated": confirmed,
        "lineage_verified_against_recorded_sources": True,
        "post_training_transform": lineage.get("post_training_transform"),
        "recipe_matches_baseline": None
        if not baseline
        else (read(baseline / "lineage.json")["training_recipe"] == lineage["training_recipe"]),
        "next_action": "deployment_regression"
        if direction_ok
        else ("baseline_cached" if baseline is None else "stop_or_analyze"),
        "quality_only_no_measured_100mp_latency": True,
        "environment": environment_versions(),
        "environment_matches_baseline": None
        if not baseline
        else (read(baseline / "review.json")["environment"] == environment_versions()),
        "official_score_forecast": False,
        "automatic_full_or_submission_admission": False,
        "small_support_warning": "HM and TU-160 have very small held-out support; inspect fine TP/FP/FN, not just the sign",
        "deployment_regression": "not_run_by_quality_review; use run_paired_deployment_regression.py",
        "artifacts": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*.json"))},
    }
    write(output / "review.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "prepare", "baseline", "candidate", "verify"))
    p.add_argument("--bundle", type=Path, default=ROOT / "data/splits/paired_trend_v1")
    p.add_argument("--data-root", type=Path, default=ROOT.parent / "data")
    p.add_argument("--weights", type=Path, default=ROOT / "outputs/yolo26s.pt")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/PAIRED-TREND-BASELINE-V1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--lineage", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument(
        "--source-groups", type=Path, default=ROOT / "data/splits/cv3_airport_proxy_k60_v2.json"
    )
    p.add_argument(
        "--source-gt",
        type=Path,
        default=ROOT
        / "outputs/SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-SEEN-DIAGNOSTIC-V1/normal/ground_truth.json",
    )
    p.add_argument(
        "--background", type=Path, default=ROOT / "outputs/MACROSHIFT-BACKGROUND-100MP-FROZEN"
    )
    a = p.parse_args()
    if a.command == "freeze":
        result = freeze(a.source_groups, a.source_gt, a.data_root, PROJECT, a.bundle, a.background)
    elif a.command == "verify":
        validate_bundle(a.bundle, PROJECT, a.data_root)
        result = {"status": "all_frozen_data_verified"}
    elif a.command in ("prepare", "baseline"):
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            materialize(a.bundle, a.data_root, a.weights, a.output, a.device)
            if a.command == "baseline" and a.execute:
                checkpoint = train_baseline(a.bundle, a.data_root, a.weights, a.output, a.device)
                result = evaluate(
                    a.bundle,
                    a.data_root,
                    checkpoint,
                    a.output / "lineage.json",
                    a.output / "evaluation",
                    a.device,
                )
            else:
                result = {
                    "status": "preflight_complete_training_not_started",
                    "plan": str(a.output / "plan.json"),
                }
    else:
        if not all((a.checkpoint, a.lineage, a.baseline)):
            p.error("candidate requires --checkpoint, --lineage, --baseline")
        validate_bundle(a.bundle, PROJECT, a.data_root)
        validate_lineage(read(a.lineage), a.bundle, a.checkpoint)
        result = (
            evaluate(a.bundle, a.data_root, a.checkpoint, a.lineage, a.output, a.device, a.baseline)
            if a.execute
            else {"status": "candidate_preflight_complete_inference_not_started"}
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
