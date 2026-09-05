#!/usr/bin/env python3
"""P40 aircraft relabel-only transfer of existing R1 CE+D4; no score fusion."""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_p40_vehicle_zoom import metrics, read, sha
from r1_aircraft_refinement import _load_adapted_model, _normalize

from rsdet.data.crop_classification import render_crop
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.experiments.fixed_proxy import review_quality_delta
from rsdet.features.p04_inputs import D4_VIEW_IDS, apply_d4_view
from rsdet.models.crop_classifier import build_convnext_tiny_classifier
from rsdet.postprocess.nms import class_aware_nms_predictions
from rsdet.utils.config import load_config


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=("hard", "sentinel", "normal_diagnostic"), required=True)
    p.add_argument("--normal-gt", type=Path)
    p.add_argument("--pseudo-root", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--frontier", type=Path, required=True)
    p.add_argument("--classifier-root", type=Path)
    p.add_argument(
        "--full-classifier",
        type=Path,
        help="Full-data fixed-last classifier used only for deployment diagnostics",
    )
    p.add_argument("--imagenet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-comparison", type=Path,
                   help="Optional completed positive classifier, not the bare detector, as comparator")
    p.add_argument("--config", type=Path, default=Path("configs/experiments/p40_aircraft_ce_d4_v1.json"))
    args = p.parse_args()
    if (args.classifier_root is None) == (args.full_classifier is None):
        raise ValueError("provide exactly one of --classifier-root or --full-classifier")
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = args.config
    config = read(config_path)
    method = config.get("classifier_method", "ce")
    if method not in {"ce", "view_consistency"}:
        raise ValueError("unknown frozen classifier method")
    if config.get("require_positive_reference", False) and args.reference_comparison is None:
        raise ValueError("this incremental experiment requires the frozen positive reference")
    view_ids = D4_VIEW_IDS[:1] if config.get("view_mode") == "identity" else D4_VIEW_IDS
    parent = read(config["parent_p40_contract"])
    classifier = load_config(config["classifier_contract"])
    is_normal = args.condition == "normal_diagnostic"
    gt_path = args.normal_gt if is_normal else args.pseudo_root / "ground_truth.json"
    expected_gt = (read("configs/experiments/p40_rot90_supported_rescue_v1.json")["normal_gt_sha256"]
                   if is_normal else parent["ground_truth_sha256"][args.condition])
    if gt_path is None or sha(gt_path) != expected_gt or sha(args.frontier) != parent["frontier_sha256"]:
        raise ValueError("proxy or calibration source changed")
    if sha(args.imagenet) != classifier["inputs"]["convnext_weight_sha256"]:
        raise ValueError("ImageNet initialization asset mismatch")
    document = read(gt_path)
    gt = load_coco_ground_truth(gt_path)
    thresholds = {int(k): v for k, v in read(args.frontier)["frontiers"]["0.150"]["crossfit_thresholds"].items()}
    images = {r["id"]: r for r in document["images"]}
    raw = load_coco_predictions(args.pred)
    if set(raw) - set(images):
        raise ValueError("prediction image coverage mismatch")
    for row in read(args.pred):
        if not is_normal and row["source_fold"] != images[row["image_id"]]["fold"]:
            raise ValueError("wrong detector heldout fold")
    baseline = {i: [r for r in raw.get(i, []) if r["score"] >= thresholds[m["fold"]]]
                for i, m in images.items()}
    candidate = deepcopy(baseline)
    assets = {}
    for fold in range(3):
        path = (
            args.full_classifier
            if args.full_classifier is not None
            else args.classifier_root / f"fold_{fold}/final_checkpoint.pt"
        )
        assets[str(fold)] = {"path": str(path), "sha256": sha(path)}
    (args.output / "preflight.json").write_text(json.dumps({"contract": config,
        "contract_sha256": sha(config_path), "classifier_assets": assets,
        "pred_sha256": sha(args.pred), "gt_sha256": sha(gt_path), "thresholds": thresholds,
        "code_sha256": {str(path): sha(path) for directory in ("src", "scripts", "configs")
            for path in sorted(Path(directory).rglob("*")) if path.is_file()
            and "__pycache__" not in path.parts and path.suffix != ".pyc"}}, indent=2)+"\n")
    started = time.perf_counter()
    changed, bundles, audited = 0, [], {}
    Image.MAX_IMAGE_PIXELS = None
    for fold in range(3):
        if args.full_classifier is None:
            model, meta = _load_adapted_model(
                Path(assets[str(fold)]["path"]), args.imagenet, fold=fold, method=method
            )
            if (
                meta["source_p03_checkpoint_sha256"]
                != classifier["inputs"]["p03_checkpoint_sha256"][str(fold)]
            ):
                raise ValueError("classifier source fold checkpoint mismatch")
        else:
            payload = torch.load(
                Path(assets[str(fold)]["path"]), map_location="cpu", weights_only=False
            )
            resolved = payload.get("resolved_config", {})
            expected = {
                "contract_version": "r1_aircraft_view_consistency_full_v1",
                "experiment_id": "R1-5-AIRCRAFT-VIEW-CONSISTENCY-FULL",
                "fold": "full",
                "method": "view_consistency",
                "epochs": 5,
                "checkpoint_selection": "fixed_epoch_last_no_validation",
            }
            mismatches = {
                key: {"actual": resolved.get(key), "expected": value}
                for key, value in expected.items()
                if resolved.get(key) != value
            }
            if mismatches:
                raise ValueError(f"full classifier contract mismatch: {mismatches}")
            model = build_convnext_tiny_classifier(
                25, weight_path=args.imagenet, regime="fine_tune"
            )
            model.load_state_dict(payload["model_state_dict"], strict=True)
            meta = {
                "checkpoint_sha256": assets[str(fold)]["sha256"],
                "source_p03_checkpoint_sha256": resolved[
                    "source_p03_checkpoint_sha256"
                ],
                "embedded_config": resolved,
                "diagnostic_full_data_classifier": True,
            }
        audited[str(fold)] = meta
        model.cuda().eval()
        for i, image in images.items():
            if image["fold"] != fold:
                continue
            indices = [idx for idx, r in enumerate(candidate[i]) if 4 <= r["category_id"] < 24]
            path = (args.pseudo_root / image["file_name"] if is_normal else
                    args.pseudo_root / f"fold_{fold}/images" / image["file_name"])
            if not indices:
                continue
            with Image.open(path) as source:
                source.load()
                for offset in range(0, len(indices), 16):
                    batch_indices = indices[offset:offset+16]
                    tensors = []
                    for idx in batch_indices:
                        crop = render_crop(source, candidate[i][idx]["bbox_xyxy"], 224)
                        tensors.extend(_normalize(apply_d4_view(crop, view)) for view in view_ids)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        logits = model(torch.stack(tensors).cuda()).reshape(len(batch_indices), len(view_ids), 25)
                    probabilities = logits.float()[:, :, 4:24].softmax(2).mean(1).cpu().numpy()
                    for idx, probs in zip(batch_indices, probabilities, strict=True):
                        old = candidate[i][idx]["category_id"]
                        label = int(np.argmax(probs)) + 4
                        confidence = float(np.max(probs))
                        if confidence >= config["relabel_min_probability"]:
                            candidate[i][idx]["category_id"] = label
                            changed += int(label != old)
                        bundles.append({"image_id": i, "prediction_index": idx, "old_category": old,
                                        "probabilities": probs.tolist()})
            (args.output / "status.txt").write_text(f"classified_fold_{fold}_image_{i}\n")
        del model
        torch.cuda.empty_cache()
    elapsed = time.perf_counter()-started
    control = class_aware_nms_predictions(baseline, .5, category_ids=list(range(4, 24)))
    candidate = class_aware_nms_predictions(candidate, .5, category_ids=list(range(4, 24)))
    def key(r):
        return (r["category_id"], r["score"], tuple(r["bbox_xyxy"]))
    for i in images:
        if sorted((key(r) for r in candidate[i] if r["category_id"] not in range(4,24))) != sorted(
            key(r) for r in baseline[i] if r["category_id"] not in range(4,24)
        ):
            raise AssertionError("Ship/Vehicle bypass changed")
    protocol = parse_evaluation_protocol(load_config("configs/project.yaml"))
    b, n, c = (metrics(gt, pred, protocol) for pred in (baseline, control, candidate))

    def coco_rows(predictions):
        rows = []
        for image_id in sorted(predictions):
            for item in predictions[image_id]:
                x1, y1, x2, y2 = (float(value) for value in item["bbox_xyxy"])
                row = {
                    "image_id": int(image_id),
                    "category_id": int(item["category_id"]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(item["score"]),
                }
                if "source_fold" in item:
                    row["source_fold"] = int(item["source_fold"])
                rows.append(row)
        return rows

    (args.output / "baseline_predictions.json").write_text(
        json.dumps(coco_rows(control), separators=(",", ":")) + "\n"
    )
    (args.output / "candidate_predictions.json").write_text(
        json.dumps(coco_rows(candidate), separators=(",", ":")) + "\n"
    )
    reference = n
    reference_sha256 = None
    if args.reference_comparison is not None:
        previous = read(args.reference_comparison)
        expected_reference = config.get("reference_sha256", {}).get(args.condition)
        reference_sha256 = sha(args.reference_comparison)
        if expected_reference != reference_sha256 or previous["status"] != "complete":
            raise ValueError("positive reference result changed")
        if previous["original_baseline"] != b or previous["nms_only_control"] != n:
            raise ValueError("positive reference uses different input/control/metric")
        reference = previous["candidate"]
    review = (dict(next_action="record_normal_diagnostic_not_an_additional_gate", formal_admission=False)
              if is_normal else review_quality_delta(reference["platform"], c["platform"], stage=args.condition,
                  minimum=config[f"{args.condition}_quality_delta_min_exclusive"]))
    result = {"status": "complete", "experiment_id": config["experiment_id"],
              "original_baseline": b, "nms_only_control": n, "candidate": c,
              "review": review, "changed_fine_labels": changed, "aircraft_proposals": len(bundles),
              "views": list(view_ids),
              "reference_comparison_sha256": reference_sha256,
              "elapsed_load_decode_classify_seconds": elapsed, "not_end_to_end_latency": True,
              "classifier_provenance": audited, "ship_vehicle_exact_bypass": True}
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (args.output / "probabilities.json").write_text(json.dumps(bundles)+"\n")
    (args.output / "status.txt").write_text("complete_"+review["next_action"]+"\n")
    print(json.dumps({"review": review, "changed": changed}), flush=True)


if __name__ == "__main__":
    main()
