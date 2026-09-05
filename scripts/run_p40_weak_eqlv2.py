#!/usr/bin/env python3
"""Frozen P40 targeted-EQLv2 CV3 ablation; no full or deployment action."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import fcntl
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_p40_weak_rfs import (
    GTS,
    PSEUDO,
    SOURCES,
    audit_inputs,
    call,
    evaluate,
    read,
    sha,
    write,
)

CONFIG = ROOT / "configs/experiments/p40_weak_eqlv2_v1.json"


def validate_eqlv2_audit(path: Path) -> dict:
    audit = read(path)
    if set(audit) not in ({"detection"}, {"one2many", "one2one"}):
        raise ValueError("unexpected EQLv2 criterion branches")
    for branch in audit.values():
        if branch["focus_class_indices"] != [0, 1, 2, 3, 24] or branch["updates"] <= 0:
            raise ValueError("EQLv2 focus or update count differs from contract")
        for key in (
            "positive_gradient",
            "negative_gradient",
            "positive_negative_ratio",
            "positive_weight",
            "negative_weight",
        ):
            values = branch[key]
            if len(values) != 25 or not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"invalid EQLv2 {key}")
        for index in set(range(25)) - {0, 1, 2, 3, 24}:
            if branch["positive_weight"][index] != 1.0 or branch["negative_weight"][index] != 1.0:
                raise ValueError("non-focus class loss differs from BCE")
    return audit


def validate_completed_fold(out: Path, fold: int, frozen: dict) -> Path:
    run = out / f"fold_{fold}/adaptation/runs/resolution_adaptation"
    rows = list(csv.DictReader((run / "results.csv").open()))
    if [int(row["epoch"]) for row in rows] != list(range(1, 41)) or not all(
        math.isfinite(float(value)) for row in rows for key, value in row.items() if "loss" in key
    ):
        raise ValueError("training epoch count/nonfinite loss")
    weight = run / "weights/last.pt"
    training = read(out / f"fold_{fold}/adaptation/training_result.json")
    if sha(weight) != training["checkpoint_sha256"]:
        raise ValueError("training weight receipt mismatch")
    if training.get("classification_loss") != frozen["config"]["classification_loss"]:
        raise ValueError("training loss contract differs from preflight")
    validate_eqlv2_audit(run / "eqlv2_audit.json")
    return weight


def run(out: Path, frozen: dict) -> None:
    if audit_inputs(CONFIG) != frozen:
        raise ValueError("frozen assets/code changed")
    weights = []
    for fold, source in enumerate(SOURCES):
        fold_out = out / f"fold_{fold}"
        call(
            out,
            f"train_fold_{fold}",
            [
                "scripts/train_progressive_resolution_adaptation.py",
                "--weak-eqlv2",
                "--weights",
                source / "runs/foundation/weights/last.pt",
                "--data",
                source / "dataset.yaml",
                "--output",
                fold_out / "adaptation",
                "--epochs",
                40,
                "--imgsz",
                1280,
                "--batch",
                8,
                "--workers",
                4,
                "--device",
                "cuda:0",
                "--seed",
                42,
                "--lr0",
                0.0002,
                "--lrf",
                0.10,
                "--rotate90-p",
                1.0,
            ],
        )
        weights.append(validate_completed_fold(out, fold, frozen))
        call(
            out,
            f"materialize_fold_{fold}",
            [
                "scripts/materialize_yolo_cross_resolution_infer_config.py",
                "--source",
                source / "resolved_infer.yaml",
                "--output",
                fold_out / "resolved_infer.yaml",
                "--predictions",
                fold_out / "predictions_low.json",
                "--imgsz",
                1280,
                "--checkpoint",
                weights[-1],
            ],
        )
        call(out, f"infer_fold_{fold}", ["scripts/infer_cv3_oof.py", "--config", fold_out / "resolved_infer.yaml"])
    call(
        out,
        "aggregate",
        [
            "scripts/merge_cv3_coco_ledgers.py",
            "--gt",
            *GTS,
            "--pred",
            *[out / f"fold_{fold}/predictions_low.json" for fold in range(3)],
            "--output-gt",
            out / "aggregate/ground_truth.json",
            "--output-pred",
            out / "aggregate/predictions_low.json",
        ],
    )
    call(
        out,
        "calibrate_normal",
        [
            "scripts/analyze_cv3_oof_pseudo_frontier.py",
            "--gt",
            out / "aggregate/ground_truth.json",
            "--pred",
            out / "aggregate/predictions_low.json",
            "--output",
            out / "aggregate/crossfit_frontier.json",
            "--threshold-step",
            0.005,
        ],
    )
    for condition in ("hard", "sentinel"):
        call(
            out,
            f"infer_{condition}",
            [
                "scripts/run_multifamily_cv3_pseudo_eval.py",
                "--pseudo-root",
                PSEUDO[condition],
                "--family",
                "yolo",
                "--weights",
                *weights,
                "--output-dir",
                out / condition,
                "--score-floor",
                0.001,
                "--batch-size",
                4,
                "--device",
                "cuda:0",
                "--imgsz",
                1280,
                "--tile-size",
                1024,
                "--overlap",
                256,
            ],
        )
        if not evaluate(out, condition, CONFIG):
            (out / "status.txt").write_text(f"complete_stopped_{condition}\n")
            return
    (out / "status.txt").write_text("complete_positive_review_required_no_full\n")


def recover_infer_import(out: Path) -> None:
    """Continue after the exact post-fold-0 PYTHONPATH import failure.

    The completed fold-0 training is immutable.  Recovery repeats inference only,
    then continues the originally frozen fold-1/fold-2 and evaluation sequence.
    """
    if (out / "status.txt").read_text().strip() != "failed_CalledProcessError":
        raise ValueError("recovery requires the recorded subprocess failure")
    infer_log = (out / "infer_fold_0.log").read_text()
    if "ModuleNotFoundError: No module named 'rsdet'" not in infer_log:
        raise ValueError("recovery is limited to the recorded rsdet import failure")
    if (out / "fold_0/predictions_low.json").exists():
        raise ValueError("fold-0 prediction unexpectedly exists; refusing recovery")

    frozen = read(out / "preflight.json")
    if frozen["config_sha256"] != sha(CONFIG) or frozen["config"] != read(CONFIG):
        raise ValueError("frozen experiment configuration changed")
    driver_relative = str(Path(__file__).resolve().relative_to(ROOT))
    caller_relative = str((ROOT / "scripts/run_p40_weak_rfs.py").relative_to(ROOT))
    changed = {driver_relative, caller_relative}
    for relative, expected in frozen["code_sha256"].items():
        if relative not in changed and sha(ROOT / relative) != expected:
            raise ValueError(f"non-recovery code changed: {relative}")

    fold0_weight = validate_completed_fold(out, 0, frozen)
    write(
        out / "recovery_infer_import.json",
        {
            "status": "post_fold_0_inference_import_recovery",
            "reason": "subprocess lacked the frozen repository src directory on PYTHONPATH",
            "training_not_restarted": True,
            "fold_0_last_pt_sha256": sha(fold0_weight),
            "frozen_driver_sha256": frozen["code_sha256"][driver_relative],
            "recovery_driver_sha256": sha(Path(__file__)),
            "frozen_caller_sha256": frozen["code_sha256"][caller_relative],
            "recovery_caller_sha256": sha(ROOT / caller_relative),
        },
    )

    weights = [fold0_weight]
    fold0_out = out / "fold_0"
    call(
        out,
        "recover_infer_fold_0",
        ["scripts/infer_cv3_oof.py", "--config", fold0_out / "resolved_infer.yaml"],
    )
    for fold in (1, 2):
        source = SOURCES[fold]
        fold_out = out / f"fold_{fold}"
        call(
            out,
            f"train_fold_{fold}",
            [
                "scripts/train_progressive_resolution_adaptation.py",
                "--weak-eqlv2",
                "--weights",
                source / "runs/foundation/weights/last.pt",
                "--data",
                source / "dataset.yaml",
                "--output",
                fold_out / "adaptation",
                "--epochs",
                40,
                "--imgsz",
                1280,
                "--batch",
                8,
                "--workers",
                4,
                "--device",
                "cuda:0",
                "--seed",
                42,
                "--lr0",
                0.0002,
                "--lrf",
                0.10,
                "--rotate90-p",
                1.0,
            ],
        )
        weights.append(validate_completed_fold(out, fold, frozen))
        call(
            out,
            f"materialize_fold_{fold}",
            [
                "scripts/materialize_yolo_cross_resolution_infer_config.py",
                "--source",
                source / "resolved_infer.yaml",
                "--output",
                fold_out / "resolved_infer.yaml",
                "--predictions",
                fold_out / "predictions_low.json",
                "--imgsz",
                1280,
                "--checkpoint",
                weights[-1],
            ],
        )
        call(out, f"infer_fold_{fold}", ["scripts/infer_cv3_oof.py", "--config", fold_out / "resolved_infer.yaml"])

    call(
        out,
        "aggregate",
        [
            "scripts/merge_cv3_coco_ledgers.py",
            "--gt",
            *GTS,
            "--pred",
            *[out / f"fold_{fold}/predictions_low.json" for fold in range(3)],
            "--output-gt",
            out / "aggregate/ground_truth.json",
            "--output-pred",
            out / "aggregate/predictions_low.json",
        ],
    )
    call(
        out,
        "calibrate_normal",
        [
            "scripts/analyze_cv3_oof_pseudo_frontier.py",
            "--gt",
            out / "aggregate/ground_truth.json",
            "--pred",
            out / "aggregate/predictions_low.json",
            "--output",
            out / "aggregate/crossfit_frontier.json",
            "--threshold-step",
            0.005,
        ],
    )
    for condition in ("hard", "sentinel"):
        call(
            out,
            f"infer_{condition}",
            [
                "scripts/run_multifamily_cv3_pseudo_eval.py",
                "--pseudo-root",
                PSEUDO[condition],
                "--family",
                "yolo",
                "--weights",
                *weights,
                "--output-dir",
                out / condition,
                "--score-floor",
                0.001,
                "--batch-size",
                4,
                "--device",
                "cuda:0",
                "--imgsz",
                1280,
                "--tile-size",
                1024,
                "--overlap",
                256,
            ],
        )
        if not evaluate(out, condition, CONFIG):
            (out / "status.txt").write_text(f"complete_stopped_{condition}\n")
            return
    (out / "status.txt").write_text("complete_positive_review_required_no_full\n")


def smoke(out: Path) -> None:
    audit_inputs(CONFIG)
    out.mkdir(parents=True, exist_ok=False)
    data = yaml.safe_load((SOURCES[0] / "dataset.yaml").read_text())
    train = Path(data["train"]).read_text().splitlines()
    subset = [train[min(len(train) - 1, index * len(train) // 32)] for index in range(32)]
    (out / "train.txt").write_text("\n".join(subset) + "\n")
    (out / "val.txt").write_text("\n".join(subset[:8]) + "\n")
    data.update(train=str(out / "train.txt"), val=str(out / "val.txt"))
    (out / "dataset.yaml").write_text(yaml.safe_dump(data))
    call(
        out,
        "smoke_one_epoch_not_scientific",
        [
            "scripts/train_progressive_resolution_adaptation.py",
            "--weak-eqlv2",
            "--weights",
            SOURCES[0] / "runs/foundation/weights/last.pt",
            "--data",
            out / "dataset.yaml",
            "--output",
            out / "adaptation",
            "--epochs",
            1,
            "--imgsz",
            1280,
            "--batch",
            8,
            "--workers",
            0,
            "--device",
            "cuda:0",
            "--seed",
            42,
            "--lr0",
            0.0002,
            "--lrf",
            0.10,
            "--rotate90-p",
            1.0,
        ],
    )
    validate_eqlv2_audit(out / "adaptation/runs/resolution_adaptation/eqlv2_audit.json")
    (out / "status.txt").write_text("smoke_pass_not_a_candidate\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "smoke", "recover-infer-import"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "smoke":
        smoke(args.output)
    elif args.action == "prepare":
        frozen = audit_inputs(CONFIG)
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / "preflight.json", frozen)
        (args.output / "status.txt").write_text("prepared_not_started\n")
    elif args.action == "run":
        with (args.output / "execution.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (args.output / "status.txt").read_text().strip() != "prepared_not_started":
                raise ValueError("refusing duplicate/restart of non-prepared experiment")
            try:
                run(args.output, read(args.output / "preflight.json"))
            except BaseException as exc:
                (args.output / "status.txt").write_text(f"failed_{type(exc).__name__}\n")
                raise
    else:
        with (args.output / "execution.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                recover_infer_import(args.output)
            except BaseException as exc:
                (args.output / "status.txt").write_text(f"failed_recovery_{type(exc).__name__}\n")
                raise


if __name__ == "__main__":
    main()
