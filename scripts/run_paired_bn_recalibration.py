#!/usr/bin/env python3
"""Prepare -> recalibrate -> reuse A review for exactly one P40 BN candidate.

No GPU work without --execute. Run from a separate code snapshot after the
reference/anchor chains finish. This never overwrites or resumes its parent.
Evaluation reuses platform_observed_20260831 through run_paired_trend.evaluate.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from rsdet.data.xh_dataset import FINE_NAMES
from rsdet.experiments.bn_recalibration import (
    bn_buffer_names,
    copy_bn_buffers,
    reestimate_bn,
    verify_only_bn_changed,
)
from rsdet.experiments.paired_trend import read, safe_path, sha, validate_bundle, write
from scripts import run_paired_trend as pipeline

CONFIG = ROOT / "configs/experiments/paired_bn_recalibration_v1.json"


def ordered_training(samples: list[dict], batch_size: int, seed: int) -> list[dict]:
    ids = [s["image_id"] for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate image identity")
    train = [s for s in samples if s["split"] == "train"]
    if not train or len(train) % batch_size:
        raise ValueError("training universe must form complete equal batches; no dropped tail")
    if len({s["relative_path"] for s in train}) != len(train):
        raise ValueError("duplicate training path")
    return sorted(train, key=lambda s: hashlib.sha256(f"{seed}:{s['image_id']}".encode()).hexdigest())


def build_plan(bundle: Path, baseline: Path) -> dict:
    contract = validate_bundle(bundle, pipeline.PROJECT)
    recipe = read(CONFIG)
    implemented = {
        "passes": 1, "batch": 8, "imgsz": 1280, "order_seed": 42,
        "expected_train_images": 3136, "reset_running_stats": True,
        "momentum": "cumulative_equal_batches", "augmentation": "none",
        "only_train_sources": True,
        "sampling": "all_training_images_once_natural_frequency_sha256_order",
        "preprocessing": "RGB_uint8_square_LetterBox_scaleup_pad114_then_float32_div255",
    }
    if any(recipe.get(k) != v for k, v in implemented.items()):
        raise ValueError("unsupported change to the fixed BN recipe")
    train = ordered_training(read(bundle / "manifest.json")["samples"], recipe["batch"], recipe["order_seed"])
    parent = read(baseline / "plan.json")
    ids = sorted(s["image_id"] for s in train)
    if len(train) != recipe["expected_train_images"]:
        raise ValueError("not the declared training population")
    if (parent["bundle_sha256"] != sha(bundle / "contract.json")
            or parent["recipe"] != contract["baseline"] or parent["train_image_ids"] != ids):
        raise ValueError("wrong baseline plan/recipe/split")
    return {
        "schema": "paired_bn_trainonly_v1",
        "recipe": recipe,
        "recipe_sha256": sha(CONFIG),
        "baseline_plan_sha256": sha(baseline / "plan.json"),
        "bundle_sha256": sha(bundle / "contract.json"),
        "source_code": {**pipeline.code_fingerprint(), str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
        "training_samples": train,
        "expected_batches": len(train) // recipe["batch"],
        "holdouts_or_background_used_for_calibration": False,
        "status": "prepared_not_executed",
    }


def verify_parent(bundle: Path, baseline: Path) -> tuple[Path, dict]:
    plan = read(baseline / "plan.json")
    previous = plan["weight_sha256"]
    for stage in ("foundation", "adaptation"):
        cp = baseline / stage / "run/weights/last.pt"
        receipt = read(baseline / stage / "complete.json")
        result_csv = baseline / stage / "run/results.csv"
        if (receipt["initial_sha256"] != previous or receipt["checkpoint_sha256"] != sha(cp)
                or receipt["plan_sha256"] != sha(baseline / "plan.json")
                or receipt["results_sha256"] != sha(result_csv)
                or receipt["config"] != plan["recipe"][stage]):
            raise ValueError("baseline completion receipt mismatch")
        pipeline.check_epochs(result_csv, plan["recipe"][stage]["epochs"])
        previous = sha(cp)
    lin = read(baseline / "lineage.json")
    pipeline.validate_lineage(lin, bundle, cp)
    review = read(baseline / "evaluation/review.json")
    if review["role"] != "baseline" or review["bundle_sha256"] != sha(bundle / "contract.json"):
        raise ValueError("baseline A evaluation not ready")
    if read(baseline / "evaluation/lineage.json") != lin:
        raise ValueError("baseline evaluation uses a different checkpoint/lineage")
    for name, digest in review["artifacts"].items():
        if sha(safe_path(baseline / "evaluation", name)) != digest:
            raise ValueError("baseline evaluation artifact changed")
    return cp, lin


def image_batches(samples: list[dict], data_root: Path, recipe: dict, device: str):
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics.data.augment import LetterBox

    transform = LetterBox(new_shape=(recipe["imgsz"], recipe["imgsz"]), auto=False,
                          scale_fill=False, scaleup=True, stride=32, padding_value=114)
    for start in range(0, len(samples), recipe["batch"]):
        batch = []
        for row in samples[start:start + recipe["batch"]]:
            path = safe_path(data_root, row["relative_path"])
            if sha(path) != row["image_sha256"]:
                raise ValueError("calibration image changed")
            with Image.open(path) as img:
                rgb = np.asarray(img.convert("RGB"))
            # LetterBox is channel-order agnostic. No accidental RGB/BGR reversal.
            rgb = transform(image=rgb)
            batch.append(torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))))
        yield torch.stack(batch).to(device=device, dtype=torch.float32).div_(255.0)


def derive_lineage(parent: dict, *, checkpoint_sha: str, plan_sha: str, receipt_sha: str) -> dict:
    lin = copy.deepcopy(parent)
    lin["checkpoint_sha256"] = checkpoint_sha
    lin["ancestors"].append({
        "kind": "bn_recalibrated_on_frozen_train",
        "checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "train_image_ids": parent["train_image_ids"],
        "plan_sha256": plan_sha,
        "receipt_sha256": receipt_sha,
        "non_bn_state_bitwise_equal": True,
    })
    lin["post_training_transform"] = "PAIRED-P40-BN-TRAINONLY-V1"
    return lin


def calibrate(bundle: Path, data_root: Path, baseline: Path, out: Path, device: str) -> dict:
    import torch

    # Import own checkpoint class registrations; only trusted local project weights.
    import ultralytics  # noqa: F401

    plan = read(out / "plan.json")
    if plan != build_plan(bundle, baseline):
        raise ValueError("prepared plan or code changed; use a new experiment directory")
    validate_bundle(bundle, pipeline.PROJECT, data_root)
    checkpoint, lineage = verify_parent(bundle, baseline)
    run = out / "calibration"
    run.mkdir()  # Fail on completed or partial work. No implicit repeat/resume.
    source_sha = sha(checkpoint)
    write(run / "started.json", {"parent_checkpoint_sha256": source_sha, "device": device,
                                "environment": pipeline.environment_versions()})
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_key = "ema" if raw.get("ema") is not None else "model"
    original = raw[source_key].cpu().eval()
    names = original.names
    if tuple(names[i] for i in range(len(names))) != FINE_NAMES:
        raise ValueError("wrong fine taxonomy")
    allowed = bn_buffer_names(original)
    before = {k: v.detach().clone() for k, v in original.state_dict().items()}
    work = copy.deepcopy(original).float().to(device)
    audit = reestimate_bn(work, image_batches(plan["training_samples"], data_root, plan["recipe"], device),
                          batch_size=plan["recipe"]["batch"], expected_batches=plan["expected_batches"])
    copied = copy_bn_buffers(work.cpu(), original)
    output = run / "bn_last.pt"
    # Unlike Model.save, explicitly clear stale EMA so loaders cannot ignore the
    # modified model. Preserve source dtype rather than silently half-round all weights.
    payload = {"model": original, "ema": None, "optimizer": None, "epoch": -1,
               "train_args": raw.get("train_args", {}), "version": raw.get("version"),
               "bn_recalibration": {"parent_sha256": source_sha, "plan_sha256": sha(out / "plan.json")}}
    with output.open("xb") as handle:
        torch.save(payload, handle)
    saved = torch.load(output, map_location="cpu", weights_only=False)
    serialized_audit = verify_only_bn_changed(before, saved["model"].state_dict(), allowed)
    if saved["ema"] is not None or sha(checkpoint) != source_sha:
        raise ValueError("EMA shadow or changed source checkpoint")
    result = {"status": "complete_candidate_not_admitted", "source_model_key": source_key,
              "parent_checkpoint_sha256": source_sha, "checkpoint_sha256": sha(output),
              "plan_sha256": sha(out / "plan.json"), "runtime": audit,
              "copy_back": copied, "serialization": serialized_audit,
              "environment": pipeline.environment_versions(), "accuracy_evaluated": False}
    write(run / "complete.json", result)
    write(out / "lineage.json", derive_lineage(lineage, checkpoint_sha=sha(output),
          plan_sha=sha(out / "plan.json"), receipt_sha=sha(run / "complete.json")))
    return result


def validate_candidate(out: Path, bundle: Path, baseline: Path, checkpoint: Path) -> None:
    if checkpoint.resolve() != (out / "calibration/bn_last.pt").resolve():
        raise ValueError("BN checkpoint is not the audited output")
    receipt_path = out / "calibration/complete.json"
    receipt = read(receipt_path)
    plan = read(out / "plan.json")
    parent, parent_lineage = verify_parent(bundle, baseline)
    expected = derive_lineage(parent_lineage, checkpoint_sha=sha(checkpoint),
                              plan_sha=sha(out / "plan.json"), receipt_sha=sha(receipt_path))
    if (plan != build_plan(bundle, baseline)
            or receipt["status"] != "complete_candidate_not_admitted"
            or receipt["checkpoint_sha256"] != sha(checkpoint)
            or receipt["parent_checkpoint_sha256"] != sha(parent)
            or receipt["plan_sha256"] != sha(out / "plan.json")
            or receipt["runtime"]["batches"] != plan["expected_batches"]
            or receipt["runtime"]["images"] != len(plan["training_samples"])
            or any(receipt[k]["all_non_bn_state_bitwise_equal"] is not True
                   for k in ("runtime", "copy_back", "serialization"))
            or receipt["copy_back"]["non_bn_state_sha256"] != receipt["serialization"]["non_bn_state_sha256"]
            or read(out / "lineage.json") != expected):
        raise ValueError("candidate receipt/provenance changed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "calibrate", "evaluate"))
    p.add_argument("--bundle", type=Path, default=ROOT / "data/splits/paired_trend_v1")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=ROOT.parent / "data")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "prepare":
            if args.output.exists():
                raise FileExistsError(args.output)
            result = build_plan(args.bundle, args.baseline)
            write(args.output / "plan.json", result)
        elif not args.execute:
            result = {"status": "not_executed", "reason": "--execute required"}
        elif args.command == "calibrate":
            result = calibrate(args.bundle, args.data_root, args.baseline, args.output, args.device)
        else:
            cp = args.output / "calibration/bn_last.pt"
            validate_candidate(args.output, args.bundle, args.baseline, cp)
            result = pipeline.evaluate(args.bundle, args.data_root, cp, args.output / "lineage.json",
                                       args.output / "evaluation", args.device, args.baseline / "evaluation")
    print(json.dumps(result if args.command != "prepare" else {
        "status": result["status"], "training_images": len(result["training_samples"]),
        "batches": result["expected_batches"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
