#!/usr/bin/env python3
"""Prepare, infer and evaluate the R1-0 proposal crop reranking screen."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from PIL import Image

from rsdet.analysis.proposal_reranking import (
    build_variants,
    load_logits,
    load_proposal_manifest,
    logits_to_probabilities,
    prepare_proposal_manifest,
    run_reranking_crossfit,
    sha256_file,
)
from rsdet.data.crop_classification import render_crop
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.models.crop_classifier import build_convnext_tiny_classifier
from rsdet.utils.config import load_config


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract_version") != "r1_proposal_reranking_v1":
        raise ValueError("R1 config contract_version 非法")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R1-0 formal OOF proposal crop reranking")
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--aggregate-dir", type=Path)
    prepare.add_argument("--output-dir", type=Path, required=True)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--manifest", type=Path, required=True)
    infer.add_argument("--data-root", type=Path, required=True)
    infer.add_argument("--weights", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--batch-size", type=int)
    infer.add_argument("--num-workers", type=int)
    infer.add_argument("--smoke", action="store_true")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--logits-dir", type=Path, required=True)
    evaluate.add_argument("--aggregate-dir", type=Path)
    evaluate.add_argument("--formal-crop-manifest", type=Path)
    evaluate.add_argument("--y1-calibration-result", type=Path)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    return parser


class _ProposalDataset:
    def __init__(self, records: Sequence[Any], data_root: Path, resolution: int) -> None:
        self.records = tuple(records)
        self.data_root = data_root.resolve()
        self.resolution = resolution
        self.cache: OrderedDict[Path, Image.Image] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, str]:
        import torch
        from torchvision.transforms import functional as functional

        record = self.records[index]
        source_path = (self.data_root / record.relative_path).resolve()
        try:
            source_path.relative_to(self.data_root)
        except ValueError as error:
            raise ValueError(f"source path 逃逸 data root: {source_path}") from error
        image = self.cache.pop(source_path, None)
        if image is None:
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            with Image.open(source_path) as source:
                image = source.convert("RGB")
        self.cache[source_path] = image
        while len(self.cache) > 8:
            self.cache.popitem(last=False)
        crop = render_crop(image, record.bbox_xyxy, self.resolution)
        tensor = functional.to_tensor(crop)
        tensor = functional.normalize(
            tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return tensor.to(dtype=torch.float32), record.proposal_uid


def _run_prepare(args: argparse.Namespace, config: dict[str, Any]) -> int:
    destination = args.output_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"prepare 输出目录非空: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    aggregate = args.aggregate_dir or Path(config["inputs"]["aggregate_dir"])
    audit = prepare_proposal_manifest(
        aggregate,
        destination / "proposal_inference_manifest.csv",
        expected_proposals=int(config["inputs"]["expected_proposals"]),
        expected_artifact_sha256=config["inputs"]["aggregate_artifact_sha256"],
    )
    _write_json(destination / "prepare_audit.json", audit)
    return 0


def _run_infer(args: argparse.Namespace, config: dict[str, Any]) -> int:
    import torch
    import torchvision

    if torch.__version__.split("+")[0] != "2.5.1":
        raise RuntimeError(f"torch 必须为 2.5.1，当前 {torch.__version__}")
    if torchvision.__version__.split("+")[0] != "0.20.1":
        raise RuntimeError(f"torchvision 必须为 0.20.1，当前 {torchvision.__version__}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    expected_checkpoint = config["inputs"]["p03_checkpoint_sha256"][str(args.fold)]
    actual_checkpoint = sha256_file(args.checkpoint)
    if actual_checkpoint != expected_checkpoint:
        raise ValueError(f"fold{args.fold} checkpoint SHA 不匹配: {actual_checkpoint}")

    records = [item for item in load_proposal_manifest(args.manifest) if item.fold == args.fold]
    if args.smoke:
        records = records[: min(32, len(records))]
    if not records:
        raise ValueError(f"fold{args.fold} 无 proposal")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("resolved_config", {})
    expected_fields = {
        "fold": args.fold,
        "policy": "tight",
        "resolution": 224,
        "regime": "fine_tune",
        "sampler": "natural",
        "seed": 42,
        "checkpoint_selection": "fixed_epoch_last",
    }
    mismatches = {
        key: {"actual": checkpoint_config.get(key), "expected": value}
        for key, value in expected_fields.items()
        if checkpoint_config.get(key) != value
    }
    if mismatches or int(checkpoint.get("epoch", -1)) != 30:
        raise ValueError(f"P03 checkpoint embedded contract 不匹配: {mismatches}")

    device = torch.device(args.device)
    model = build_convnext_tiny_classifier(
        25,
        weight_path=args.weights,
        regime="fine_tune",
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    dataset = _ProposalDataset(records, args.data_root.expanduser(), 224)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size or config["runtime"]["batch_size"]),
        shuffle=False,
        num_workers=int(
            args.num_workers if args.num_workers is not None else config["runtime"]["num_workers"]
        ),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / (
        f"fold_{args.fold}_smoke_logits.npz" if args.smoke else f"fold_{args.fold}_logits.npz"
    )
    if output_path.exists():
        raise FileExistsError(f"禁止覆盖 logits: {output_path}")
    all_logits: list[np.ndarray] = []
    all_uids: list[str] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for images, uids in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
            all_logits.append(logits.float().cpu().numpy())
            all_uids.extend(str(uid) for uid in uids)
    elapsed = time.perf_counter() - started
    matrix = np.concatenate(all_logits, axis=0).astype(np.float32, copy=False)
    if matrix.shape != (len(records), 25) or not np.isfinite(matrix).all():
        raise ValueError(f"logits 输出非法: shape={matrix.shape}")
    np.savez_compressed(
        output_path,
        logits=matrix,
        proposal_uids=np.asarray(all_uids, dtype=f"<U{max(map(len, all_uids))}"),
    )
    runtime = {
        "contract_version": "r1_proposal_reranking_v1",
        "status": "smoke_pass" if args.smoke else "complete",
        "fold": args.fold,
        "proposal_count": len(records),
        "checkpoint_sha256": actual_checkpoint,
        "manifest_sha256": sha256_file(args.manifest),
        "logits_sha256": sha256_file(output_path),
        "elapsed_seconds": elapsed,
        "proposals_per_second": len(records) / elapsed,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    suffix = "smoke_runtime" if args.smoke else "runtime"
    _write_json(destination / f"fold_{args.fold}_{suffix}.json", runtime)
    return 0


def _build_decision(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    merged = result["merged"]
    raw_delta = merged["delta_r1_vs_d0"]
    c2_delta = merged["delta_c2_r1_vs_c2"]
    raw_transition = merged["paired_transition_r1_vs_d0"]
    c2_transition = merged["paired_transition_c2_r1_vs_c2"]
    gate = config["decision_gates"]

    raw_signal = bool(
        merged["R1_selected"]["recall"] >= float(gate["official_recall_min"])
        and merged["R1_selected"]["fdr"] <= float(gate["official_fdr_max"])
        and raw_delta["recall"] >= -float(gate["maximum_recall_drop"])
        and raw_delta["fdr"] <= float(gate["maximum_fdr_increase"])
        and raw_delta["macro_recall"] > 0.0
        and raw_transition["new_tp"] >= raw_transition["broken_tp"]
    )
    incremental = bool(
        merged["C2_frozen_after_r1"]["recall"] >= float(gate["official_recall_min"])
        and merged["C2_frozen_after_r1"]["fdr"] <= float(gate["official_fdr_max"])
        and c2_delta["recall"] >= -float(gate["maximum_recall_drop"])
        and c2_delta["fdr"] <= float(gate["maximum_fdr_increase"])
        and c2_delta["macro_recall"] >= float(gate["minimum_incremental_macro_recall"])
        and c2_transition["new_tp"] >= c2_transition["broken_tp"]
    )
    if incremental:
        next_action = "admit_r1_inference_and_run_short_proposal_domain_finetune"
    elif raw_signal:
        next_action = "run_short_proposal_domain_finetune_before_operational_admission"
    else:
        next_action = "stop_r1_and_prioritize_background_rejection"
    return {
        "contract_version": "r1_proposal_reranking_decision_v1",
        "status": "complete",
        "formal_admission": False,
        "inference_only_signal_vs_d0": raw_signal,
        "incremental_signal_vs_frozen_c2": incremental,
        "next_action": next_action,
        "reason": {
            "delta_r1_vs_d0": raw_delta,
            "delta_c2_r1_vs_c2": c2_delta,
            "paired_transition_r1_vs_d0": raw_transition,
            "paired_transition_c2_r1_vs_c2": c2_transition,
        },
        "gates": gate,
    }


def _run_evaluate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    destination = args.output_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"evaluate 输出目录非空: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    records = load_proposal_manifest(args.manifest)
    if len(records) != int(config["inputs"]["expected_proposals"]):
        raise ValueError("R1 manifest proposal 数不符合冻结配置")
    logits = load_logits(args.logits_dir, records)
    probabilities = logits_to_probabilities(logits)
    project_config = load_config(config["project_config"])
    protocol = parse_evaluation_protocol(project_config)
    y1_path = args.y1_calibration_result or Path(config["inputs"]["y1_calibration_result"])
    if sha256_file(y1_path) != config["inputs"]["y1_calibration_result_sha256"]:
        raise ValueError("Y1 calibration_result SHA 不匹配")
    formal_manifest = args.formal_crop_manifest or Path(
        config["inputs"]["formal_crop_manifest"]
    )
    if sha256_file(formal_manifest) != config["inputs"]["formal_crop_manifest_sha256"]:
        raise ValueError("formal crop manifest SHA 不匹配")
    y1 = json.loads(y1_path.read_text(encoding="utf-8"))
    if y1["merged_held_out"]["C2_prior"]["official_gate_passed"] is not True:
        raise ValueError("冻结 Y1 C2 baseline 未通过官方门禁")
    search = config["search"]
    result, selected_raw = run_reranking_crossfit(
        records=records,
        probabilities=probabilities,
        aggregate_dir=args.aggregate_dir or config["inputs"]["aggregate_dir"],
        formal_crop_manifest_path=formal_manifest,
        y1_calibration_result=y1,
        protocol=protocol,
        variants=build_variants(search),
        threshold_start=float(search["threshold_start"]),
        threshold_stop=float(search["threshold_stop"]),
        threshold_step=float(search["threshold_step"]),
        maximum_selection_recall_drop=float(search["maximum_selection_recall_drop"]),
    )
    parity = result["merged"]["C2_frozen_reconstruction_parity"]
    tolerance = float(config["inputs"]["y1_reconstruction_tolerance"])
    if max(abs(float(value)) for value in parity.values()) > tolerance:
        raise ValueError(f"冻结 Y1 C2 重建未通过 parity gate: {parity}")
    decision = _build_decision(result, config)
    _write_json(destination / "reranking_result.json", result)
    _write_json(destination / "decision.json", decision)
    predictions = []
    for image_id in sorted(selected_raw):
        for record in selected_raw[image_id]:
            x0, y0, x1, y1_value = record["bbox_xyxy"]
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(record["category_id"]),
                    "bbox": [x0, y0, x1 - x0, y1_value - y0],
                    "score": float(record["score"]),
                    "proposal_uid": record["proposal_uid"],
                }
            )
    _write_json(destination / "selected_crossfit_raw_predictions.json", predictions)
    summary = {
        "status": "complete",
        "next_action": decision["next_action"],
        "selected_variants": [item["selected_variant"] for item in result["per_fold"]],
        "merged": result["merged"],
    }
    _write_json(destination / "summary.json", summary)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _load_config(args.config.expanduser().resolve())
        if args.command == "prepare":
            return _run_prepare(args, config)
        if args.command == "infer":
            return _run_infer(args, config)
        if args.command == "evaluate":
            return _run_evaluate(args, config)
        raise AssertionError(args.command)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as error:
        print(f"R1-0 failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
