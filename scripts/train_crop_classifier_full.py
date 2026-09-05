#!/usr/bin/env python3
"""Train the unique full-data P03 crop backbone for admitted APEX modules.

This command deliberately has no validation or checkpoint selection.  The
architecture and hyperparameters were selected by the earlier grouped CV3
experiment; every formal tight crop is used once per epoch and the fixed final
epoch is the only output checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_crop_classifier import (
    _build_transforms,
    _environment_meta,
    _make_loader,
    _optimizer_and_scheduler,
    _require_gpu_stack,
    _sampler_audit,
    _seed_everything,
    _train_one_epoch,
    _write_history,
)

from rsdet.data.crop_classification import CropClassificationDataset, class_counts
from rsdet.data.xh_dataset import FINE_NAMES
from rsdet.features.p04_inputs import load_all_crop_records
from rsdet.models.crop_classifier import (
    CONVNEXT_TINY_WEIGHT_SHA256,
    build_convnext_tiny_classifier,
    parameter_summary,
    sha256_file,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if config["experiment"]["id"] != "P03-APEX-FULL-V1-CONVNEXT-T224":
        raise ValueError("unexpected full-data experiment contract")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if sha256_file(args.manifest) != config["data"]["manifest_sha256"]:
        raise ValueError("formal crop manifest SHA mismatch")
    if sha256_file(args.weights) != CONVNEXT_TINY_WEIGHT_SHA256:
        raise ValueError("ImageNet ConvNeXt-Tiny SHA mismatch")

    args.output.mkdir(parents=True)
    torch, torchvision = _require_gpu_stack(args.device)
    seed = 42
    _seed_everything(torch, seed)
    device = torch.device(args.device)
    records = load_all_crop_records(args.manifest, crop_policy="tight", task="all25")
    if {item.manifest_version for item in records} != {config["data"]["manifest_version"]}:
        raise ValueError("formal crop manifest version mismatch")

    training = config["training"]["fine_tune"]
    runtime = config["runtime"]
    resolved = {
        "experiment_id": config["experiment"]["id"],
        "fold": "full",
        "policy": "tight",
        "resolution": 224,
        "regime": "fine_tune",
        "sampler": "natural",
        "seed": seed,
        "epochs": int(training["epochs"]),
        "batch_size": int(runtime["batch_size"]),
        "num_workers": int(runtime["num_workers"]),
        "n_train": len(records),
        "manifest": str(args.manifest.resolve()),
        "data_root": str(args.data_root.resolve()),
        "weights": str(args.weights.resolve()),
        "training_class_counts": class_counts(records),
        "checkpoint_selection": "fixed_epoch_last",
    }
    (args.output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False)
    )

    model = build_convnext_tiny_classifier(
        len(FINE_NAMES), weight_path=args.weights, regime="fine_tune"
    ).to(device)
    dataset = CropClassificationDataset(
        records,
        args.data_root,
        224,
        transform=_build_transforms(torchvision, train=True),
        image_cache_size=int(runtime["image_cache_size_per_worker"]),
    )
    loader = _make_loader(
        torch,
        dataset,
        records,
        batch_size=resolved["batch_size"],
        num_workers=resolved["num_workers"],
        train=True,
        sampler_name="natural",
        sampler_config=config["sampler"]["sqrt_inverse"],
        seed=seed,
    )
    optimizer, scheduler = _optimizer_and_scheduler(
        torch, model, "fine_tune", config, len(loader), resolved["epochs"]
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(training["label_smoothing"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    for epoch in range(1, resolved["epochs"] + 1):
        metrics = _train_one_epoch(
            torch,
            model,
            loader,
            criterion,
            optimizer,
            scheduler,
            device,
            scaler,
            float(training["grad_clip_norm"]),
            "fine_tune",
        )
        row = {"epoch": epoch, **metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(json.dumps(row), flush=True)
    _write_history(args.output / "history.csv", history)
    checkpoint = args.output / "final_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "resolved_config": resolved,
            "epoch": resolved["epochs"],
            "selection_metric": "fixed_epoch_last",
            "selection_value": None,
        },
        checkpoint,
    )
    meta = _environment_meta(torch, torchvision, device)
    meta.update(
        {
            "manifest_sha256": sha256_file(args.manifest),
            "weight_sha256": sha256_file(args.weights),
            "model_parameters": parameter_summary(model),
        }
    )
    _write_json(args.output / "meta.json", meta)
    _write_json(
        args.output / "summary.json",
        {
            "status": "full_data_checkpoint_ready",
            "resolved_config": resolved,
            "sampler_audit": _sampler_audit(
                records, "natural", config["sampler"]["sqrt_inverse"]
            ),
            "final_train_metrics": history[-1],
            "checkpoint_sha256": sha256_file(checkpoint),
            "validation_metrics": None,
            "validation_note": "no held-out data used for full-data checkpoint selection",
        },
    )
    (args.output / "status.txt").write_text("complete\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
