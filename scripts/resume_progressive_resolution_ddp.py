#!/usr/bin/env python3
"""Resume the verified epoch-2 Progressive-40 full checkpoint on three GPUs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_payload(payload: dict) -> None:
    if payload.get("epoch") != 1:
        raise ValueError("expected epoch index 1 (two completed epochs)")
    if not payload.get("optimizer") or payload.get("ema") is None:
        raise ValueError("resume checkpoint is missing optimizer or EMA")
    frozen = {
        "epochs": 40, "imgsz": 1280, "batch": 8, "seed": 42,
        "optimizer": "AdamW", "lr0": 0.0002, "lrf": 0.1,
        "mosaic": 0.0, "close_mosaic": 40, "cos_lr": True,
        "nbs": 64, "val": False, "deterministic": True,
    }
    for key, value in frozen.items():
        if payload.get("train_args", {}).get(key) != value:
            raise ValueError(f"source recipe mismatch: {key}")
    if payload.get("train_results", {}).get("epoch") != [1, 2]:
        raise ValueError("checkpoint must contain the exact epoch-1/2 history")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--original-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if sha256(args.checkpoint) != args.expected_sha256:
        raise ValueError("resume checkpoint SHA mismatch")
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")

    import torch

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_payload(payload)
    contract = json.loads(args.original_contract.read_text(encoding="utf-8"))
    contract.update({
        "schema_version": "progressive_resolution_ddp_migration_v1",
        "resume_source": str(args.checkpoint.resolve()),
        "resume_checkpoint_sha256": args.expected_sha256,
        "resume_completed_epochs": 2, "epochs": 40,
        "batch": 24, "device": "0,1,2", "workers": 8,
        "previous_global_batch": 8, "per_rank_batch": 8,
        "previous_nominal_effective_batch": 64, "nominal_effective_batch": 72,
        "batch_change_authority": "user_requested_three_gpu_accelerated_migration",
        "not_bitwise_equivalent_to_single_gpu": True,
        "optimizer_lr_scaling": "none",
        "optimizer_scaler_ema": "restored_from_epoch2_checkpoint",
        "rng_state": "rank_seed_reinitialized_by_upstream_resume_not_exact_rng_replay",
        "cache": False,
    })
    if args.dry_run:
        print(json.dumps(contract, indent=2))
        print("PROGRESSIVE_DDP_RESUME_PREFLIGHT_PASS")
        return 0

    args.output.mkdir(parents=True)
    resume_input = args.output / "resume_input.pt"
    shutil.copy2(args.checkpoint, resume_input)
    if sha256(resume_input) != args.expected_sha256:
        raise ValueError("copied resume input SHA mismatch")
    run_dir = args.output / "runs" / "resolution_adaptation"
    run_dir.mkdir(parents=True)
    history = payload["train_results"]
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history))
        writer.writeheader()
        for index in range(2):
            writer.writerow({key: value[index] for key, value in history.items()})
    (args.output / "training_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    os.environ["SCALEROUTE_RESUME_AUDIT_DIR"] = str((args.output / "resume_audits").resolve())
    del payload

    from ultralytics import YOLO

    from rsdet.innovation.progressive_resume import AuditedProgressiveResumeTrainer
    from rsdet.innovation.trainers import rotate90_augmentations

    model = YOLO(str(resume_input.resolve()))
    model.train(
        trainer=AuditedProgressiveResumeTrainer,
        resume=str(resume_input.resolve()), save_dir=str(run_dir.resolve()),
        device="0,1,2", batch=24, workers=8, cache=False,
        augmentations=rotate90_augmentations(p=1.0),
    )
    checkpoint = run_dir / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise RuntimeError("resume completed without last.pt")
    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in rows] != list(range(1, 41)):
        raise RuntimeError("full epoch history must be exactly 1..40")
    for rank in range(3):
        audit_path = args.output / "resume_audits" / f"rank_{rank}.json"
        if json.loads(audit_path.read_text())["status"] != "pass":
            raise RuntimeError(f"missing valid restoration audit for rank {rank}")
    result = {
        **contract, "status": "complete_candidate",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
    }
    (args.output / "training_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("PROGRESSIVE_DDP_RESUME_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
