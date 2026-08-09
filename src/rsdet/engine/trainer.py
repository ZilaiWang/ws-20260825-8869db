"""Executable training engine for the paper-aligned BHC-DETR model."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

logger = logging.getLogger(__name__)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = int(np.random.get_state()[1][0]) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _to_device(
    batch: Mapping[str, Any],
    device: Any,
    *,
    channels_last: bool = False,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    images = batch["images"].to(device, non_blocking=True)
    if channels_last:
        import torch

        images = images.contiguous(memory_format=torch.channels_last)
    masks = batch["padding_masks"].to(device, non_blocking=True)
    targets = [
        {
            key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
            for key, value in target.items()
        }
        for target in batch["targets"]
    ]
    return images, masks, targets


def _accumulate_metrics(
    sums: dict[str, Any],
    losses: Mapping[str, Any],
) -> None:
    """Accumulate detached scalars without synchronizing CUDA every batch."""

    for name, value in losses.items():
        detached = value.detach().float()
        if detached.numel() != 1:
            raise ValueError(
                f"loss metric {name!r} must contain exactly one value, "
                f"got shape {tuple(detached.shape)}"
            )
        # Some detection losses are represented as one-element tensors [1],
        # while BHCL/Gain losses are zero-dimensional tensors [].  Normalizing
        # both to [] is metadata-only and therefore does not synchronize CUDA.
        detached = detached.reshape(())
        if name in sums:
            sums[name].add_(detached)
        else:
            sums[name] = detached.clone()


def _mean_metrics(sums: Mapping[str, Any], steps: int) -> dict[str, float]:
    divisor = max(steps, 1)
    names = sorted(sums)
    if not names:
        return {}
    values = [sums[name] for name in names]
    if all(hasattr(value, "detach") for value in values):
        import torch

        # One device-to-host transfer per epoch replaces one .item() call for
        # every loss in every batch, which otherwise serializes the GPU.
        averaged = (
            torch.stack([value.detach().float().reshape(()) for value in values])
            .div(float(divisor))
            .cpu()
            .tolist()
        )
        return dict(zip(names, (float(value) for value in averaged)))
    return {name: float(sums[name]) / divisor for name in names}


def _run_validation(
    model: Any,
    criterion: Any,
    loader: Any,
    device: Any,
    amp: bool,
    *,
    channels_last: bool = False,
    max_batches: int | None = None,
) -> dict[str, float]:
    import torch

    model.eval()
    criterion.eval()
    sums: dict[str, Any] = {}
    steps = 0
    with torch.inference_mode():
        for batch in loader:
            images, masks, targets = _to_device(
                batch,
                device,
                channels_last=channels_last,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                losses = criterion(model(images, masks), targets)
            _accumulate_metrics(sums, losses)
            steps += 1
            if max_batches is not None and steps >= max_batches:
                break
    return _mean_metrics(sums, steps)


def _save_checkpoint(
    path: Path,
    *,
    model: Any,
    criterion: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    seed: int,
    model_config: Any,
    loss_config: Any,
    train_config: Mapping[str, Any],
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "format_version": "bhcdetr_checkpoint_v1",
        "paper": "arXiv:2512.24074v1",
        "papers": ["arXiv:2512.24074v1", "arXiv:2604.21435v1"],
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "seed": seed,
        "model_config": model_config.to_dict(),
        "loss_config": loss_config.to_dict(),
        "train_config": dict(train_config),
        "model": model.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
    }
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_compatible_state(module: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    """Load only name/shape-compatible tensors for architecture warm-starting."""

    current = module.state_dict()
    compatible = {
        str(name): value
        for name, value in state.items()
        if name in current
        and hasattr(value, "shape")
        and tuple(value.shape) == tuple(current[name].shape)
    }
    incompatible_shape = sorted(
        str(name)
        for name, value in state.items()
        if name in current
        and hasattr(value, "shape")
        and tuple(value.shape) != tuple(current[name].shape)
    )
    missing, unexpected = module.load_state_dict(compatible, strict=False)
    return {
        "source_tensors": len(state),
        "loaded_tensors": len(compatible),
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
        "shape_mismatch": incompatible_shape,
        "missing_preview": list(missing[:20]),
        "unexpected_preview": list(unexpected[:20]),
    }


def _warm_start(
    path: Path,
    *,
    model: Any,
    criterion: Any,
) -> dict[str, Any]:
    """Initialize shared BHC weights without restoring optimizer/run state."""

    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("initial checkpoint root must be a mapping")
    model_state = checkpoint.get("model", checkpoint.get("state_dict"))
    if not isinstance(model_state, Mapping):
        raise ValueError("initial checkpoint does not contain model weights")
    report = {
        "checkpoint": str(path),
        "model": _load_compatible_state(model, model_state),
        "criterion": None,
    }
    criterion_state = checkpoint.get("criterion")
    if isinstance(criterion_state, Mapping):
        report["criterion"] = _load_compatible_state(criterion, criterion_state)
    return report


def _restore_checkpoint(
    path: Path,
    *,
    model: Any,
    criterion: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
) -> tuple[int, int, float]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != "bhcdetr_checkpoint_v1":
        raise ValueError("unsupported resume checkpoint")
    model.load_state_dict(checkpoint["model"], strict=True)
    criterion.load_state_dict(checkpoint["criterion"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("best_val_loss", math.inf)),
    )


def train(
    config: dict[str, Any],
    *,
    resume: str | Path | None = None,
    device_override: str | None = None,
    dry_run: bool = False,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Train BHC-DETR from a YAML-derived configuration mapping."""

    try:
        import torch
        from torch.utils.data import DataLoader
    except (ImportError, OSError) as error:
        raise ImportError(
            "BHC-DETR training requires a working CUDA-compatible PyTorch installation"
        ) from error

    from rsdet.data.bhcdetr_dataset import BHCDetrDataset, bhcdetr_collate
    from rsdet.models.bhcdetr import BHCDetr, BHCDetrConfig
    from rsdet.models.detection_loss import BHCDetrCriterion, BHCDetrLossConfig

    if str(config.get("model", "bhcdetr")) != "bhcdetr":
        raise ValueError("the active training engine only accepts model: bhcdetr")
    seed = int(config.get("seed", 42))
    _seed_everything(seed, torch)
    model_config = BHCDetrConfig.from_mapping(_mapping(config.get("architecture", {}), "architecture"))
    loss_config = BHCDetrLossConfig.from_mapping(_mapping(config.get("loss", {}), "loss"))
    data_config = _mapping(config.get("data"), "data")
    train_config = _mapping(config.get("train"), "train")
    augmentation = _mapping(train_config.get("augmentation", {}), "train.augmentation")

    resume_value = resume or config.get("resume")
    init_value = config.get("init_checkpoint")
    resume_path = (
        Path(str(resume_value)).expanduser()
        if resume_value is not None and str(resume_value).strip()
        else None
    )
    init_path = (
        Path(str(init_value)).expanduser()
        if init_value is not None and str(init_value).strip()
        else None
    )
    if resume_path is not None and init_path is not None:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
    if init_path is not None and not init_path.is_file():
        raise FileNotFoundError(f"init_checkpoint does not exist: {init_path}")

    data_root = Path(str(data_config.get("root", ""))).expanduser()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data.root does not exist: {data_root}")
    manifest_value = str(data_config.get("manifest", "")).strip()
    manifest = Path(manifest_value).expanduser() if manifest_value else None
    if manifest is not None and not manifest.is_file():
        raise FileNotFoundError(f"data.manifest does not exist: {manifest}")
    held_out_fold = data_config.get("held_out_fold")
    views_per_image = _positive_int(int(train_config.get("views_per_image", 2)), "views_per_image")
    dataset_kwargs = {
        "data_root": data_root,
        "manifest": manifest,
        "held_out_fold": held_out_fold,
        "image_size": model_config.image_size,
        "num_classes": model_config.num_classes,
        "flip_probability": float(augmentation.get("flip_probability", 0.5)),
        "max_shift_ratio": float(augmentation.get("max_shift_ratio", 0.1)),
    }
    train_dataset = BHCDetrDataset(
        split="train",
        training=True,
        views_per_image=views_per_image,
        **dataset_kwargs,
    )
    val_dataset = BHCDetrDataset(
        split="val",
        training=False,
        views_per_image=1,
        **dataset_kwargs,
    )
    if not train_dataset or not val_dataset:
        raise ValueError(
            f"empty train/val dataset: train={len(train_dataset)}, val={len(val_dataset)}"
        )
    output_dir = Path(str(config.get("output_dir", "outputs/bhcdetr"))).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "model": "bhcdetr",
        "paper": "arXiv:2512.24074v1",
        "papers": ["arXiv:2512.24074v1", "arXiv:2604.21435v1"],
        "uhr_small_object_enabled": model_config.uhr_enabled,
        "train_source_images": len(train_dataset),
        "train_views_per_source": views_per_image,
        "val_source_images": len(val_dataset),
        "image_size": model_config.image_size,
        "fine_classes": model_config.num_classes,
        "seed": seed,
        "resume": str(resume_path) if resume_path is not None else None,
        "init_checkpoint": str(init_path) if init_path is not None else None,
    }
    (output_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if dry_run:
        logger.info("BHC-DETR dry-run passed: %s", audit)
        return {"status": "dry_run_pass", **audit}

    requested_device = device_override or str(train_config.get("device", "cuda:0"))
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested_device}")
    amp = bool(train_config.get("amp", True)) and device.type == "cuda"
    deterministic = bool(train_config.get("deterministic", True))
    channels_last = bool(train_config.get("channels_last", False)) and device.type == "cuda"
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    matmul_precision = str(train_config.get("matmul_precision", "highest")).strip().lower()
    if matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError("train.matmul_precision must be highest, high, or medium")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)

    batch_size = _positive_int(int(train_config.get("batch_size", 2)), "batch_size")
    workers = max(0, int(train_config.get("workers", 4)))
    prefetch_factor = _positive_int(
        int(train_config.get("prefetch_factor", 2)),
        "prefetch_factor",
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": bhcdetr_collate,
        "worker_init_fn": _seed_worker,
        "generator": generator,
        # Worker-local Dataset copies must be rebuilt after set_epoch() so
        # their deterministic augmentation stream advances every epoch.
        "persistent_workers": False,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_options)

    model = BHCDetr(model_config).to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    criterion = BHCDetrCriterion(
        num_classes=model_config.num_classes,
        projection_dim=model_config.projection_dim,
        decoder_layers=model_config.decoder_layers,
        config=loss_config,
    ).to(device)
    if init_path is not None:
        init_report = _warm_start(
            init_path,
            model=model,
            criterion=criterion,
        )
        (output_dir / "init_checkpoint_report.json").write_text(
            json.dumps(init_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "warm-started BHC-DETR: model=%d/%d criterion=%s",
            init_report["model"]["loaded_tensors"],
            init_report["model"]["source_tensors"],
            (
                f'{init_report["criterion"]["loaded_tensors"]}/'
                f'{init_report["criterion"]["source_tensors"]}'
                if init_report["criterion"] is not None
                else "not available"
            ),
        )
    learning_rate = float(train_config.get("learning_rate", 5e-5))
    backbone_learning_rate = float(
        train_config.get("backbone_learning_rate", learning_rate)
    )
    weight_decay = float(train_config.get("weight_decay", 1e-4))
    if (
        not math.isfinite(learning_rate)
        or not math.isfinite(backbone_learning_rate)
        or learning_rate <= 0.0
        or backbone_learning_rate <= 0.0
    ):
        raise ValueError("learning rates must be positive")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("train.weight_decay must be finite and non-negative")
    use_backbone_parameter_group = "backbone_learning_rate" in train_config
    backbone_parameter_ids = (
        {
            id(parameter)
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        }
        if use_backbone_parameter_group
        else set()
    )
    main_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_parameter_ids
    ]
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in backbone_parameter_ids
    ]
    parameter_groups = []
    if main_parameters:
        parameter_groups.append(
            {"params": main_parameters, "lr": learning_rate, "name": "main"}
        )
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": backbone_learning_rate,
                "name": "backbone",
            }
        )
    if not parameter_groups:
        raise ValueError("model has no trainable parameters")
    optimizer_options = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }
    fused_optimizer = bool(train_config.get("fused_optimizer", False)) and device.type == "cuda"
    fused_optimizer_active = fused_optimizer
    if fused_optimizer:
        optimizer_options["fused"] = True
    try:
        optimizer = torch.optim.AdamW(parameter_groups, **optimizer_options)
    except (RuntimeError, TypeError, ValueError) as error:
        if not fused_optimizer:
            raise
        logger.warning("fused AdamW unavailable, falling back to standard AdamW: %s", error)
        optimizer_options.pop("fused", None)
        optimizer = torch.optim.AdamW(parameter_groups, **optimizer_options)
        fused_optimizer_active = False
    epochs = _positive_int(int(train_config.get("epochs", 120)), "epochs")
    accumulation_steps = _positive_int(
        int(train_config.get("gradient_accumulation_steps", 2)),
        "gradient_accumulation_steps",
    )
    updates_per_epoch = max(1, math.ceil(len(train_loader) / accumulation_steps))
    scheduler_name = str(train_config.get("scheduler", "none")).strip().lower()
    scheduler_interval = "epoch"
    if scheduler_name == "none":
        scheduler = None
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "warmup_multistep":
        warmup_steps = int(train_config.get("warmup_steps", 500))
        if warmup_steps < 0:
            raise ValueError("train.warmup_steps must be non-negative")
        raw_milestones = train_config.get("lr_milestones", [8, 11])
        if not isinstance(raw_milestones, (list, tuple)):
            raise ValueError("train.lr_milestones must be a sequence of epochs")
        milestones = sorted(
            {
                _positive_int(int(value), "train.lr_milestones item")
                for value in raw_milestones
            }
        )
        if any(value >= epochs for value in milestones):
            raise ValueError("train.lr_milestones items must be smaller than epochs")
        lr_gamma = float(train_config.get("lr_gamma", 0.1))
        if not math.isfinite(lr_gamma) or not 0.0 < lr_gamma <= 1.0:
            raise ValueError("train.lr_gamma must be in (0, 1]")
        milestone_steps = [value * updates_per_epoch for value in milestones]

        def lr_multiplier(update_index: int) -> float:
            warmup = (
                min(float(update_index + 1) / float(warmup_steps), 1.0)
                if warmup_steps > 0
                else 1.0
            )
            decay_count = sum(update_index >= step for step in milestone_steps)
            return warmup * (lr_gamma**decay_count)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
        scheduler_interval = "update"
    else:
        raise ValueError(
            "train.scheduler must be 'none', 'cosine', or 'warmup_multistep'"
        )
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
    except (AttributeError, TypeError):  # PyTorch 2.1 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    clip_max_norm = float(train_config.get("clip_max_norm", 0.1))
    if not math.isfinite(clip_max_norm) or clip_max_norm < 0.0:
        raise ValueError("clip_max_norm must be non-negative")
    validation_interval = _positive_int(
        int(train_config.get("validation_interval", 1)),
        "validation_interval",
    )
    log_interval = int(train_config.get("log_interval", 0))
    if log_interval < 0:
        raise ValueError("train.log_interval must be non-negative")
    start_epoch = 0
    global_step = 0
    best_val_loss = math.inf
    if resume_path is not None:
        start_epoch, global_step, best_val_loss = _restore_checkpoint(
            resume_path,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        logger.info("resumed BHC-DETR at epoch=%d step=%d", start_epoch, global_step)

    logger.info(
        "throughput profile: source_batch=%d model_views=%d effective_views_per_update=%d "
        "workers=%d prefetch=%d channels_last=%s matmul=%s fused_adamw=%s "
        "val_interval=%d log_interval=%d",
        batch_size,
        batch_size * views_per_image,
        batch_size * views_per_image * accumulation_steps,
        workers,
        prefetch_factor,
        channels_last,
        matmul_precision,
        fused_optimizer_active,
        validation_interval,
        log_interval,
    )

    history_path = output_dir / "metrics.jsonl"
    stop_requested = False
    for epoch in range(start_epoch, epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, Any] = {}
        epoch_steps = 0
        epoch_source_images = 0
        epoch_started_at = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            images, masks, targets = _to_device(
                batch,
                device,
                channels_last=channels_last,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                losses = criterion(model(images, masks), targets)
                scaled_loss = losses["loss_total"] / accumulation_steps
            scaler.scale(scaled_loss).backward()
            should_step = (batch_index + 1) % accumulation_steps == 0 or (
                batch_index + 1 == len(train_loader)
            )
            if should_step:
                if clip_max_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if scheduler is not None and scheduler_interval == "update":
                    scheduler.step()
                if max_steps is not None and global_step >= max_steps:
                    stop_requested = True
            _accumulate_metrics(sums, losses)
            epoch_steps += 1
            epoch_source_images += max(1, len(targets) // views_per_image)
            if log_interval and (
                (batch_index + 1) % log_interval == 0
                or batch_index + 1 == len(train_loader)
            ):
                elapsed = max(time.perf_counter() - epoch_started_at, 1e-9)
                memory = ""
                if device.type == "cuda":
                    memory = (
                        f" gpu_reserved_gib="
                        f"{torch.cuda.max_memory_reserved(device) / (1024**3):.2f}"
                    )
                logger.info(
                    "epoch=%d batch=%d/%d source_images_per_s=%.2f%s",
                    epoch,
                    batch_index + 1,
                    len(train_loader),
                    epoch_source_images / elapsed,
                    memory,
                )
            if stop_requested:
                break
        if scheduler is not None and scheduler_interval == "epoch":
            scheduler.step()
        train_metrics = _mean_metrics(sums, epoch_steps)
        epoch_train_seconds = time.perf_counter() - epoch_started_at
        validation_ran = (
            (epoch + 1) % validation_interval == 0
            or epoch + 1 == epochs
            or stop_requested
        )
        val_metrics = (
            _run_validation(
                model,
                criterion,
                val_loader,
                device,
                amp,
                channels_last=channels_last,
                max_batches=1 if max_steps is not None else None,
            )
            if validation_ran
            else {}
        )
        val_loss = float(val_metrics.get("loss_total", math.nan))
        is_best = validation_ran and math.isfinite(val_loss) and val_loss <= best_val_loss
        if is_best:
            best_val_loss = val_loss
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "learning_rates": {
                str(group.get("name", index)): float(group["lr"])
                for index, group in enumerate(optimizer.param_groups)
            },
            "validation_ran": validation_ran,
            "train_seconds": epoch_train_seconds,
            "train_source_images_per_second": (
                epoch_source_images / max(epoch_train_seconds, 1e-9)
            ),
            "train": train_metrics,
            "val": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _save_checkpoint(
            output_dir / "last.pt",
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            seed=seed,
            model_config=model_config,
            loss_config=loss_config,
            train_config=train_config,
        )
        if is_best:
            _save_checkpoint(
                output_dir / "best.pt",
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                seed=seed,
                model_config=model_config,
                loss_config=loss_config,
                train_config=train_config,
            )
        logger.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f train_seconds=%.1f source_images_per_s=%.2f",
            epoch,
            train_metrics.get("loss_total", math.nan),
            val_loss,
            epoch_train_seconds,
            epoch_source_images / max(epoch_train_seconds, 1e-9),
        )
        if stop_requested:
            break
    return {
        "status": "training_complete",
        "output_dir": str(output_dir),
        "last_checkpoint": str(output_dir / "last.pt"),
        "best_checkpoint": str(output_dir / "best.pt"),
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }


__all__ = ["train"]
