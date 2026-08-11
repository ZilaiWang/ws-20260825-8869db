"""R1-0 leakage-safe proposal crop reranking.

The experiment reuses the formal M1 OOF proposals and the fold-specific P03-F
ConvNeXt classifiers.  A crop classifier may only change a fine category inside
the detector's original coarse category.  Hyperparameters and score thresholds
are selected on the other two folds and applied once to the held-out fold.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsdet.analysis.crossfit_thresholds import (
    _filter_by_score,
    evaluate_ranking_workpoint,
    evaluate_workpoint,
    load_cv3_aggregate,
    load_gt_from_formal_crop_manifest,
    scan_global_threshold,
)
from rsdet.data.xh_dataset import coarse_name
from rsdet.evaluation.official_metric import evaluate_predictions_with_trace
from rsdet.evaluation.protocol import EvaluationProtocol
from rsdet.postprocess.yolo_calibration import (
    ClassSpatialStatistics,
    calibrate_predictions,
)

R1_CONTRACT_VERSION = "r1_proposal_reranking_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProposalRecord:
    proposal_uid: str
    image_id: int
    fold: int
    relative_path: str
    category_id: int
    bbox_xyxy: tuple[float, float, float, float]
    detector_score: float
    source_prediction_index: int


@dataclass(frozen=True)
class RerankVariant:
    name: str
    relabel: bool
    min_top_probability: float = 0.0
    min_margin: float = 0.0
    fusion_alpha: float = 0.0


def _coarse_ids(category_id: int) -> tuple[int, ...]:
    name = coarse_name(category_id)
    if name == "ship":
        return tuple(range(0, 4))
    if name == "aircraft":
        return tuple(range(4, 24))
    return (24,)


def build_variants(config: Mapping[str, Any]) -> list[RerankVariant]:
    """Build the compact, predeclared R1-0 screen grid."""

    variants = [RerankVariant("D0_detector", relabel=False)]
    variants.append(RerankVariant("R1_hard_same_coarse", relabel=True))
    for top in config["min_top_probabilities"]:
        for margin in config["min_margins"]:
            variants.append(
                RerankVariant(
                    f"R2_gate_p{float(top):.2f}_m{float(margin):.2f}",
                    relabel=True,
                    min_top_probability=float(top),
                    min_margin=float(margin),
                )
            )
    for alpha in config["fusion_alphas"]:
        variants.append(
            RerankVariant(
                f"R3_fuse_a{float(alpha):.2f}",
                relabel=False,
                fusion_alpha=float(alpha),
            )
        )
    fixed_alpha = float(config["gated_fusion_alpha"])
    for top in config["min_top_probabilities"]:
        for margin in config["min_margins"]:
            variants.append(
                RerankVariant(
                    f"R4_gate_fuse_p{float(top):.2f}_m{float(margin):.2f}"
                    f"_a{fixed_alpha:.2f}",
                    relabel=True,
                    min_top_probability=float(top),
                    min_margin=float(margin),
                    fusion_alpha=fixed_alpha,
                )
            )
    if len({variant.name for variant in variants}) != len(variants):
        raise ValueError("R1 variants 名称重复")
    return variants


def prepare_proposal_manifest(
    aggregate_dir: str | Path,
    output_path: str | Path,
    *,
    expected_proposals: int,
    expected_artifact_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Join the frozen OOF image/proposal tables into an inference manifest."""

    root = Path(aggregate_dir).expanduser().resolve()
    artifact_paths = {
        "oof_metadata": root / "oof_metadata.json",
        "oof_images": root / "oof_images.csv",
        "oof_proposals": root / "oof_proposals.csv",
        "predictions_oof_low": root / "predictions_oof_low.json",
    }
    actual_sha: dict[str, str] = {}
    for name, path in artifact_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"M1 aggregate 缺少 {path}")
        actual_sha[name] = sha256_file(path)
        expected = expected_artifact_sha256.get(name)
        if expected and actual_sha[name] != expected:
            raise ValueError(f"{name} SHA 不匹配: {actual_sha[name]} != {expected}")

    image_rows: dict[int, dict[str, str]] = {}
    with artifact_paths["oof_images"].open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = int(row["image_id"])
            if image_id in image_rows:
                raise ValueError(f"重复 image_id={image_id}")
            image_rows[image_id] = row

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "proposal_uid",
        "image_id",
        "fold",
        "relative_path",
        "category_id",
        "x0",
        "y0",
        "x1",
        "y1",
        "detector_score",
        "source_prediction_index",
    ]
    seen: set[str] = set()
    fold_counts = {0: 0, 1: 0, 2: 0}
    rows_written = 0
    with (
        artifact_paths["oof_proposals"].open("r", encoding="utf-8", newline="") as source,
        output.open("w", encoding="utf-8", newline="") as target,
    ):
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            uid = row["proposal_uid"].strip()
            image_id = int(row["image_id"])
            fold = int(row["fold"])
            if not uid or uid in seen:
                raise ValueError(f"proposal_uid 为空或重复: {uid!r}")
            if image_id not in image_rows:
                raise ValueError(f"proposal 指向未知 image_id={image_id}")
            image_row = image_rows[image_id]
            if fold != int(image_row["fold"]) or fold not in fold_counts:
                raise ValueError(f"proposal {uid} fold 与 image 表不一致")
            category_id = int(row["category_id"])
            if category_id not in range(25):
                raise ValueError(f"proposal {uid} category_id 非法")
            x0, y0 = float(row["x"]), float(row["y"])
            width, height = float(row["width"]), float(row["height"])
            score = float(row["score"])
            values = (x0, y0, width, height, score)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"proposal {uid} 包含非有限数")
            if width <= 0.0 or height <= 0.0 or not 0.0 <= score <= 1.0:
                raise ValueError(f"proposal {uid} 几何或 score 非法")
            writer.writerow(
                {
                    "proposal_uid": uid,
                    "image_id": image_id,
                    "fold": fold,
                    "relative_path": image_row["relative_path"],
                    "category_id": category_id,
                    "x0": format(x0, ".10g"),
                    "y0": format(y0, ".10g"),
                    "x1": format(x0 + width, ".10g"),
                    "y1": format(y0 + height, ".10g"),
                    "detector_score": format(score, ".10g"),
                    "source_prediction_index": int(row["source_prediction_index"]),
                }
            )
            seen.add(uid)
            fold_counts[fold] += 1
            rows_written += 1
    if rows_written != expected_proposals:
        raise ValueError(f"proposal 数 {rows_written} != expected {expected_proposals}")
    return {
        "contract_version": R1_CONTRACT_VERSION,
        "status": "pass",
        "proposal_count": rows_written,
        "image_count": len(image_rows),
        "fold_counts": fold_counts,
        "manifest_path": str(output),
        "manifest_sha256": sha256_file(output),
        "source_sha256": actual_sha,
    }


def load_proposal_manifest(path: str | Path) -> list[ProposalRecord]:
    records: list[ProposalRecord] = []
    seen: set[str] = set()
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            uid = row["proposal_uid"]
            if uid in seen:
                raise ValueError(f"重复 proposal_uid={uid}")
            record = ProposalRecord(
                proposal_uid=uid,
                image_id=int(row["image_id"]),
                fold=int(row["fold"]),
                relative_path=row["relative_path"],
                category_id=int(row["category_id"]),
                bbox_xyxy=tuple(float(row[key]) for key in ("x0", "y0", "x1", "y1")),
                detector_score=float(row["detector_score"]),
                source_prediction_index=int(row["source_prediction_index"]),
            )
            if record.fold not in {0, 1, 2} or record.category_id not in range(25):
                raise ValueError(f"manifest record 非法: {record}")
            seen.add(uid)
            records.append(record)
    if not records:
        raise ValueError("proposal manifest 为空")
    return records


def load_logits(logits_dir: str | Path, records: Sequence[ProposalRecord]) -> dict[str, np.ndarray]:
    """Load three safe NPZ files and require exact one-to-one UID coverage."""

    root = Path(logits_dir).expanduser().resolve()
    result: dict[str, np.ndarray] = {}
    for fold in (0, 1, 2):
        path = root / f"fold_{fold}_logits.npz"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 fold logits: {path}")
        with np.load(path, allow_pickle=False) as payload:
            logits = np.asarray(payload["logits"], dtype=np.float32)
            uids = np.asarray(payload["proposal_uids"])
        if logits.ndim != 2 or logits.shape[1] != 25 or logits.shape[0] != len(uids):
            raise ValueError(f"{path} logits/UID shape 非法")
        if not np.isfinite(logits).all():
            raise ValueError(f"{path} 包含 NaN/Inf")
        for uid, vector in zip(uids.tolist(), logits, strict=True):
            key = str(uid)
            if key in result:
                raise ValueError(f"logits proposal_uid 重复: {key}")
            result[key] = vector
    expected = {record.proposal_uid for record in records}
    actual = set(result)
    if actual != expected:
        raise ValueError(
            f"logits coverage 不完整: missing={len(expected-actual)}, extra={len(actual-expected)}"
        )
    return result


def logits_to_probabilities(logits: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    probabilities: dict[str, np.ndarray] = {}
    for uid, vector in logits.items():
        shifted = np.asarray(vector, dtype=np.float64) - float(np.max(vector))
        values = np.exp(shifted)
        probabilities[uid] = values / float(values.sum())
    return probabilities


def _candidate_category_and_margin(
    probabilities: np.ndarray,
    original_category: int,
) -> tuple[int, float, float]:
    allowed = _coarse_ids(original_category)
    ranked = sorted(allowed, key=lambda index: float(probabilities[index]), reverse=True)
    selected = ranked[0]
    top = float(probabilities[selected])
    second = float(probabilities[ranked[1]]) if len(ranked) > 1 else 0.0
    return selected, top, top - second


def apply_variant(
    records: Sequence[ProposalRecord],
    probabilities: Mapping[str, np.ndarray],
    variant: RerankVariant,
    *,
    image_ids: Sequence[int],
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, int]]:
    """Apply a variant without changing boxes or crossing coarse categories."""

    predictions: dict[int, list[dict[str, Any]]] = {int(image_id): [] for image_id in image_ids}
    relabelled = gated_out = 0
    for record in records:
        probability = probabilities[record.proposal_uid]
        selected, top, margin = _candidate_category_and_margin(probability, record.category_id)
        category_id = record.category_id
        if variant.relabel and selected != record.category_id:
            if top >= variant.min_top_probability and margin >= variant.min_margin:
                category_id = selected
                relabelled += 1
            else:
                gated_out += 1
        score_probability = float(probability[category_id])
        score = record.detector_score
        if variant.fusion_alpha > 0.0:
            epsilon = 1e-8
            score = math.exp(
                (1.0 - variant.fusion_alpha) * math.log(max(score, epsilon))
                + variant.fusion_alpha * math.log(max(score_probability, epsilon))
            )
        if coarse_name(category_id) != coarse_name(record.category_id):
            raise AssertionError("R1 variant 跨越了大类边界")
        predictions[record.image_id].append(
            {
                "image_id": record.image_id,
                "category_id": category_id,
                "score": score,
                "bbox_xyxy": list(record.bbox_xyxy),
                "proposal_uid": record.proposal_uid,
            }
        )
    return predictions, {"relabelled": relabelled, "gated_out": gated_out}


def _metric_payload(metrics: Any, ranking: Any) -> dict[str, Any]:
    return {
        "recall": metrics.recall,
        "fdr": metrics.fdr,
        "tp": int(metrics.details.get("tp", 0)),
        "fp": int(metrics.details.get("fp", 0)),
        "fn": int(metrics.details.get("fn", 0)),
        "macro_recall": ranking.overall_recall,
        "macro_fdr": ranking.overall_fdr,
        "per_coarse": {
            name: {
                "macro_recall": item.macro_recall,
                "macro_fdr": item.macro_fdr,
                "pooled_recall": item.pooled_recall,
                "pooled_fdr": item.pooled_fdr,
            }
            for name, item in ranking.per_coarse.items()
        },
    }


def _matched_gt_keys(trace: Any) -> set[tuple[int, int]]:
    return {(item.image_id, item.ground_truth_index) for item in trace.matches}


def paired_transition(
    gt: dict[int, list[dict[str, Any]]],
    baseline: dict[int, list[dict[str, Any]]],
    candidate: dict[int, list[dict[str, Any]]],
    protocol: EvaluationProtocol,
) -> dict[str, Any]:
    baseline_metrics, baseline_trace = evaluate_predictions_with_trace(
        gt,
        baseline,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    candidate_metrics, candidate_trace = evaluate_predictions_with_trace(
        gt,
        candidate,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    base_keys = _matched_gt_keys(baseline_trace)
    candidate_keys = _matched_gt_keys(candidate_trace)
    return {
        "new_tp": len(candidate_keys - base_keys),
        "broken_tp": len(base_keys - candidate_keys),
        "retained_tp": len(base_keys & candidate_keys),
        "net_tp": len(candidate_keys) - len(base_keys),
        "fp_delta": int(candidate_metrics.details["fp"] - baseline_metrics.details["fp"]),
        "fn_delta": int(candidate_metrics.details["fn"] - baseline_metrics.details["fn"]),
    }


def apply_frozen_c2(
    calibration_result: Mapping[str, Any],
    predictions: Mapping[int, list[dict[str, Any]]],
    image_folds: Mapping[int, int],
) -> dict[int, list[dict[str, Any]]]:
    """Apply the already selected Y1 C2 parameters; never re-fit on R1 output."""

    output: dict[int, list[dict[str, Any]]] = {}
    fold_blocks = {
        int(item["held_out_fold"]): item for item in calibration_result["per_fold"]
    }
    if set(fold_blocks) != {0, 1, 2}:
        raise ValueError("Y1 calibration_result 缺少三折")
    for fold, block in fold_blocks.items():
        statistic_payload = block["statistics"]
        statistics = ClassSpatialStatistics(
            counts=tuple(int(value) for value in statistic_payload["counts"]),
            priors=tuple(float(value) for value in statistic_payload["priors"]),
            fractal_dimensions=tuple(
                float(value) for value in statistic_payload["fractal_dimensions"]
            ),
            source_folds=tuple(int(value) for value in block["selection_folds"]),
        )
        parameters = block["methods"]["C2_prior"]["selected"]
        fold_predictions = {
            image_id: list(predictions.get(image_id, []))
            for image_id, image_fold in image_folds.items()
            if image_fold == fold
        }
        adjusted = calibrate_predictions(
            fold_predictions,
            statistics=statistics,
            beta=float(parameters["beta"]),
            spatial_lambda=float(parameters["spatial_lambda"]),
        )
        output.update(_filter_by_score(adjusted, float(parameters["threshold"])))
    return output


def run_reranking_crossfit(
    *,
    records: Sequence[ProposalRecord],
    probabilities: Mapping[str, np.ndarray],
    aggregate_dir: str | Path,
    formal_crop_manifest_path: str | Path,
    y1_calibration_result: Mapping[str, Any],
    protocol: EvaluationProtocol,
    variants: Sequence[RerankVariant],
    threshold_start: float,
    threshold_stop: float,
    threshold_step: float,
    maximum_selection_recall_drop: float,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    """Select a variant and threshold on two folds, evaluate on the third."""

    metadata, _, image_folds = load_cv3_aggregate(aggregate_dir, candidate_floor=0.001)
    gt = load_gt_from_formal_crop_manifest(
        formal_crop_manifest_path,
        expected_images=4481,
        expected_annotations=20933,
    )
    all_image_ids = sorted(image_folds)
    record_folds = {record.fold for record in records}
    if len(records) != int(metadata["proposal_count"]) or record_folds != {0, 1, 2}:
        raise ValueError("R1 manifest 与 M1 aggregate proposal/fold 不一致")

    transformed: dict[str, dict[int, list[dict[str, Any]]]] = {}
    audits: dict[str, dict[str, int]] = {}
    for variant in variants:
        transformed[variant.name], audits[variant.name] = apply_variant(
            records,
            probabilities,
            variant,
            image_ids=all_image_ids,
        )

    per_fold: list[dict[str, Any]] = []
    merged_gt: dict[int, list[dict[str, Any]]] = {}
    merged_baseline: dict[int, list[dict[str, Any]]] = {}
    merged_candidate: dict[int, list[dict[str, Any]]] = {}
    merged_candidate_raw: dict[int, list[dict[str, Any]]] = {}

    for held_out in (0, 1, 2):
        selection_images = {
            image_id for image_id, fold in image_folds.items() if fold != held_out
        }
        heldout_images = {image_id for image_id, fold in image_folds.items() if fold == held_out}
        selection_gt = {image_id: gt[image_id] for image_id in selection_images}
        heldout_gt = {image_id: gt[image_id] for image_id in heldout_images}
        candidate_results: list[dict[str, Any]] = []
        for variant in variants:
            selection_predictions = {
                image_id: transformed[variant.name][image_id] for image_id in selection_images
            }
            threshold, metrics, _ = scan_global_threshold(
                selection_gt,
                selection_predictions,
                protocol=protocol,
                threshold_start=threshold_start,
                threshold_stop=threshold_stop,
                threshold_step=threshold_step,
                internal_recall_min=protocol.recall_min,
                internal_fdr_max=protocol.fdr_max,
            )
            ranking = evaluate_ranking_workpoint(
                selection_gt,
                selection_predictions,
                threshold=threshold,
                protocol=protocol,
                require_complete_taxonomy=True,
            )
            candidate_results.append(
                {
                    "variant": variant.name,
                    "threshold": threshold,
                    "selection": _metric_payload(metrics, ranking),
                }
            )
        baseline = next(item for item in candidate_results if item["variant"] == "D0_detector")
        for item in candidate_results:
            payload = item["selection"]
            eligible = bool(
                payload["recall"] >= baseline["selection"]["recall"]
                - maximum_selection_recall_drop
                and payload["fdr"] <= baseline["selection"]["fdr"] + 1e-12
                and payload["recall"] >= protocol.recall_min
                and payload["fdr"] <= protocol.fdr_max
            )
            item["selection_eligible"] = eligible
            item["selection_rank"] = [
                int(eligible),
                payload["macro_recall"],
                -payload["macro_fdr"],
                payload["recall"],
                -payload["fdr"],
                int(item["variant"] == "D0_detector"),
            ]
        selected = max(candidate_results, key=lambda item: tuple(item["selection_rank"]))
        baseline_raw = {
            image_id: transformed["D0_detector"][image_id] for image_id in heldout_images
        }
        selected_raw = {
            image_id: transformed[selected["variant"]][image_id] for image_id in heldout_images
        }
        baseline_selected = _filter_by_score(baseline_raw, float(baseline["threshold"]))
        candidate_selected = _filter_by_score(selected_raw, float(selected["threshold"]))
        baseline_metrics = evaluate_workpoint(
            heldout_gt, baseline_selected, threshold=0.0, protocol=protocol
        )
        baseline_ranking = evaluate_ranking_workpoint(
            heldout_gt,
            baseline_selected,
            threshold=0.0,
            protocol=protocol,
            require_complete_taxonomy=False,
        )
        candidate_metrics = evaluate_workpoint(
            heldout_gt, candidate_selected, threshold=0.0, protocol=protocol
        )
        candidate_ranking = evaluate_ranking_workpoint(
            heldout_gt,
            candidate_selected,
            threshold=0.0,
            protocol=protocol,
            require_complete_taxonomy=False,
        )
        merged_gt.update(heldout_gt)
        merged_baseline.update(baseline_selected)
        merged_candidate.update(candidate_selected)
        merged_candidate_raw.update(selected_raw)
        per_fold.append(
            {
                "held_out_fold": held_out,
                "selection_folds": sorted({0, 1, 2} - {held_out}),
                "selected_variant": selected["variant"],
                "selected_threshold": selected["threshold"],
                "candidate_grid": candidate_results,
                "held_out_baseline": _metric_payload(baseline_metrics, baseline_ranking),
                "held_out_candidate": _metric_payload(candidate_metrics, candidate_ranking),
                "held_out_transition": paired_transition(
                    heldout_gt, baseline_selected, candidate_selected, protocol
                ),
            }
        )

    def evaluate_merged(predictions: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
        metrics = evaluate_workpoint(merged_gt, predictions, threshold=0.0, protocol=protocol)
        ranking = evaluate_ranking_workpoint(
            merged_gt,
            predictions,
            threshold=0.0,
            protocol=protocol,
            require_complete_taxonomy=True,
        )
        return _metric_payload(metrics, ranking)

    baseline_payload = evaluate_merged(merged_baseline)
    candidate_payload = evaluate_merged(merged_candidate)
    c2_baseline = apply_frozen_c2(y1_calibration_result, transformed["D0_detector"], image_folds)
    c2_candidate = apply_frozen_c2(y1_calibration_result, merged_candidate_raw, image_folds)
    # Include explicit empty images for consistent auditing; evaluator semantics are unchanged.
    c2_baseline_full = {image_id: c2_baseline.get(image_id, []) for image_id in all_image_ids}
    c2_candidate_full = {image_id: c2_candidate.get(image_id, []) for image_id in all_image_ids}
    c2_baseline_payload = evaluate_merged(c2_baseline_full)
    c2_candidate_payload = evaluate_merged(c2_candidate_full)
    expected_c2 = y1_calibration_result["merged_held_out"]["C2_prior"]
    c2_parity = {
        "recall_delta": c2_baseline_payload["recall"] - float(expected_c2["recall"]),
        "fdr_delta": c2_baseline_payload["fdr"] - float(expected_c2["fdr"]),
        "macro_recall_delta": (
            c2_baseline_payload["macro_recall"]
            - float(expected_c2["official_ranking"]["overall_macro_recall"])
        ),
        "macro_fdr_delta": (
            c2_baseline_payload["macro_fdr"]
            - float(expected_c2["official_ranking"]["overall_macro_fdr"])
        ),
    }

    def delta(candidate: Mapping[str, Any], baseline_value: Mapping[str, Any]) -> dict[str, float]:
        return {
            "recall": float(candidate["recall"] - baseline_value["recall"]),
            "fdr": float(candidate["fdr"] - baseline_value["fdr"]),
            "macro_recall": float(candidate["macro_recall"] - baseline_value["macro_recall"]),
            "macro_fdr": float(candidate["macro_fdr"] - baseline_value["macro_fdr"]),
        }

    result = {
        "contract_version": R1_CONTRACT_VERSION,
        "scientific_scope": "inference_only_p03_teacher_on_formal_m1_oof_proposals",
        "leakage_control": "fold_specific_teacher_and_outer_two_fold_variant_selection",
        "proposal_count": len(records),
        "variant_audits": audits,
        "per_fold": per_fold,
        "merged": {
            "D0_detector": baseline_payload,
            "R1_selected": candidate_payload,
            "delta_r1_vs_d0": delta(candidate_payload, baseline_payload),
            "C2_frozen_original": c2_baseline_payload,
            "C2_frozen_after_r1": c2_candidate_payload,
            "C2_frozen_reconstruction_parity": c2_parity,
            "delta_c2_r1_vs_c2": delta(c2_candidate_payload, c2_baseline_payload),
            "paired_transition_r1_vs_d0": paired_transition(
                merged_gt, merged_baseline, merged_candidate, protocol
            ),
            "paired_transition_c2_r1_vs_c2": paired_transition(
                merged_gt, c2_baseline_full, c2_candidate_full, protocol
            ),
        },
    }
    return result, merged_candidate_raw


__all__ = [
    "R1_CONTRACT_VERSION",
    "ProposalRecord",
    "RerankVariant",
    "apply_frozen_c2",
    "apply_variant",
    "build_variants",
    "load_logits",
    "load_proposal_manifest",
    "logits_to_probabilities",
    "paired_transition",
    "prepare_proposal_manifest",
    "run_reranking_crossfit",
    "sha256_file",
]
