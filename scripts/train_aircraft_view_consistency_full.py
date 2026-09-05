#!/usr/bin/env python3
"""Train the admitted aircraft D4 view-consistency adapter on all formal data.

The CV3 experiment already fixed every hyperparameter.  This command only
materializes the deployable full-data checkpoint: it starts from the frozen
P03 full checkpoint, uses every audited aircraft crop, performs five epochs,
does not evaluate validation data, and saves fixed-last only.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import yaml
from r1_aircraft_refinement import (
    _require_stack,
    _seed,
    _symmetric_view_consistency,
    _TrainDataset,
)

from rsdet.analysis.aircraft_refinement import (
    AIRCRAFT_CLASS_IDS,
    audit_aircraft_training_rows,
    load_aircraft_training_rows,
)
from rsdet.analysis.proposal_reranking import sha256_file
from rsdet.models.crop_classifier import build_convnext_tiny_classifier


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--p03-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_full_p03(
    checkpoint_path: Path,
    weights_path: Path,
    expected_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    import torch

    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"P03 full checkpoint SHA mismatch: {actual_sha256}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resolved = payload.get("resolved_config", {})
    expected = {
        "experiment_id": "P03-APEX-FULL-V1-CONVNEXT-T224",
        "fold": "full",
        "policy": "tight",
        "resolution": 224,
        "regime": "fine_tune",
        "sampler": "natural",
        "seed": 42,
        "checkpoint_selection": "fixed_epoch_last",
    }
    mismatches = {
        key: {"actual": resolved.get(key), "expected": value}
        for key, value in expected.items()
        if resolved.get(key) != value
    }
    if mismatches or int(payload.get("epoch", -1)) != 30:
        raise ValueError(f"P03 full embedded contract mismatch: {mismatches}")
    model = build_convnext_tiny_classifier(25, weight_path=weights_path, regime="fine_tune")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, {"checkpoint_sha256": actual_sha256, "embedded_config": resolved}


def main(argv: Sequence[str] | None = None) -> int:
    import torch
    import torch.nn.functional as functional
    import torchvision

    args = _parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "R1-5-AIRCRAFT-VIEW-CONSISTENCY-FULL":
        raise ValueError("unexpected full-data experiment contract")
    _require_stack(torch, torchvision, config)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if sha256_file(args.training_manifest) != config["inputs"]["training_manifest_sha256"]:
        raise ValueError("training manifest SHA mismatch")
    if sha256_file(args.weights) != config["inputs"]["convnext_weight_sha256"]:
        raise ValueError("ConvNeXt-Tiny weight SHA mismatch")

    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    seed = int(config["training"]["seed"])
    _seed(torch, seed)
    rows = load_aircraft_training_rows(args.training_manifest)
    audit = audit_aircraft_training_rows(rows)
    expected_rows = int(config["inputs"]["expected_training_aircraft_rows"])
    if audit["row_count"] != expected_rows:
        raise ValueError(f"aircraft row count mismatch: {audit['row_count']}")

    student, source_meta = _load_full_p03(
        args.p03_checkpoint,
        args.weights,
        str(config["inputs"]["p03_full_checkpoint_sha256"]),
    )
    student.to(device)
    dataset = _TrainDataset(rows, args.data_root.expanduser(), paired_d4_views=True)
    runtime = config["runtime"]
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size or runtime["train_batch_size"]),
        shuffle=True,
        num_workers=int(
            args.num_workers if args.num_workers is not None else runtime["num_workers"]
        ),
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=False,
        persistent_workers=False,
    )

    training = config["training"]
    head_parameters = list(student.classifier.parameters())
    head_ids = {id(item) for item in head_parameters}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [item for item in student.parameters() if id(item) not in head_ids],
                "lr": float(training["backbone_lr"]),
            },
            {"params": head_parameters, "lr": float(training["head_lr"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * len(loader))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        student.train()
        totals = {"samples": 0, "loss": 0.0, "ce": 0.0, "consistency": 0.0, "correct": 0}
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels20 = labels.to(device, non_blocking=True) - AIRCRAFT_CLASS_IDS[0]
            if images.ndim != 5 or images.shape[1] != 2:
                raise RuntimeError("view-consistency batch must be Bx2xCxHxW")
            optimizer.zero_grad(set_to_none=True)
            batch_size = images.shape[0]
            images = images.reshape(batch_size * 2, *images.shape[2:])
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = student(images)
                logits20 = logits[:, AIRCRAFT_CLASS_IDS[0] : AIRCRAFT_CLASS_IDS[-1] + 1]
                paired = logits20.reshape(batch_size, 2, -1)
                ce = functional.cross_entropy(
                    logits20,
                    labels20.repeat_interleave(2),
                    label_smoothing=float(training["label_smoothing"]),
                )
                consistency = _symmetric_view_consistency(
                    paired[:, 0, :],
                    paired[:, 1, :],
                    temperature=float(training["view_consistency"]["temperature"]),
                )
                loss = ce + float(training["view_consistency"]["loss_weight"]) * consistency
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                student.parameters(), float(training["grad_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count = int(labels20.numel())
            totals["samples"] += count
            totals["loss"] += float(loss.detach()) * count
            totals["ce"] += float(ce.detach()) * count
            totals["consistency"] += float(consistency.detach()) * count
            totals["correct"] += int(paired.mean(dim=1).argmax(dim=1).eq(labels20).sum())
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / totals["samples"],
            "train_ce": totals["ce"] / totals["samples"],
            "train_view_consistency": totals["consistency"] / totals["samples"],
            "train_accuracy": totals["correct"] / totals["samples"],
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)

    resolved = {
        "contract_version": "r1_aircraft_view_consistency_full_v1",
        "experiment_id": config["experiment_id"],
        "fold": "full",
        "selection_folds": [0, 1, 2],
        "method": "view_consistency",
        "epochs": epochs,
        "source_p03_checkpoint_sha256": source_meta["checkpoint_sha256"],
        "training_manifest_sha256": sha256_file(args.training_manifest),
        "training_rows": len(rows),
        "validation_rows_loaded_or_evaluated": False,
        "checkpoint_selection": "fixed_epoch_last_no_validation",
        "augmentation": "two_distinct_uniform_random_D4_views",
        "training_only_view_consistency": dict(training["view_consistency"]),
        "aircraft_class_ids": list(AIRCRAFT_CLASS_IDS),
    }
    checkpoint_path = args.output_dir / "final_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "resolved_config": resolved,
            "epoch": epochs,
            "selection_metric": "fixed_epoch_last_no_validation",
        },
        checkpoint_path,
    )
    summary = {
        "status": "full_data_checkpoint_ready",
        "resolved_config": resolved,
        "training_audit": audit,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_metrics": None,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    _write_json(args.output_dir / "run_summary.json", summary)
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "status.txt").write_text("complete\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
