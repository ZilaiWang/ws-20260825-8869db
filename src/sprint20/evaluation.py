"""Call the repository's matcher/scorer; never choose policy on evaluation GT."""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

MAPPING = {c: "ship" if c < 4 else "aircraft" if c < 24 else "vehicle" for c in range(25)}
COARSE_IDS = {"ship": list(range(4)), "aircraft": list(range(4, 24)), "vehicle": [24]}


def gt_from_coco(payload):
    gt = {int(image["id"]): [] for image in payload["images"]}
    for annotation in payload["annotations"]:
        if annotation.get("iscrowd", 0) or annotation.get("ignore", 0):
            raise ValueError(
                "Crowd/ignore GT needs an explicitly audited contract; not silently dropped"
            )
        image_id, label = int(annotation["image_id"]), int(annotation["category_id"])
        if image_id not in gt or label not in MAPPING:
            raise ValueError("GT uses unknown image/category IDs")
        x, y, w, h = map(float, annotation["bbox"])
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError("Invalid COCO bbox")
        gt[image_id].append({"bbox_xyxy": [x, y, x + w, y + h], "category_id": label})
    return gt


def cache_predictions(cache, threshold):
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    result = {}
    for image in cache["images"]:
        image_id = int(image["image_id"])
        if image_id in result:
            raise ValueError("Duplicate image IDs in prediction cache")
        p = image["prediction"]
        result[image_id] = [
            {"bbox_xyxy": list(b), "score": float(s), "category_id": int(c)}
            for b, s, c in zip(p["boxes_xyxy"], p["scores"], p["labels"], strict=True)
            if float(s) >= threshold
        ]
    return result


def route_dicts(base, alternate, alternate_labels):
    labels = set(alternate_labels)
    if not labels <= {0, 1, 2, 3, 24} or set(base) != set(alternate):
        raise ValueError("Invalid ownership/image IDs")
    return {
        i: [p for p in base[i] if p["category_id"] not in labels]
        + [p for p in alternate[i] if p["category_id"] in labels]
        for i in base
    }


def evaluate(gt, pred, latency):
    from rsdet.evaluation.absolute_score import platform_confirmed_score
    from rsdet.evaluation.official_metric import evaluate_ranking_metrics

    ranking = evaluate_ranking_metrics(
        gt, pred, category_mapping=MAPPING, require_complete_taxonomy=True
    )
    rows = {
        coarse: {"recall": row.macro_recall, "fdr": row.macro_fdr}
        for coarse, row in ranking.per_coarse.items()
    }
    return {
        "score": platform_confirmed_score(rows, latency),
        "per_fine": {str(c): asdict(v) for c, v in ranking.per_fine.items()},
        "matching": ranking.details,
    }


def group_counts(gt, pred, image_to_group):
    from rsdet.evaluation.official_metric import evaluate_predictions_with_trace

    if set(gt) != set(pred) or set(gt) - set(image_to_group):
        raise ValueError("All images, INCLUDING negative images, need a group and prediction")
    _, trace = evaluate_predictions_with_trace(gt, pred, category_mapping=MAPPING)
    groups = sorted(set(image_to_group[i] for i in gt))
    indices = {g: i for i, g in enumerate(groups)}
    counts = np.zeros((len(groups), 25, 3), dtype=np.int64)  # TP,FP,FN
    for column, events in enumerate(
        (trace.matches, trace.unmatched_predictions, trace.unmatched_ground_truths)
    ):
        for event in events:
            counts[indices[image_to_group[event.image_id]], event.category_id, column] += 1
    return groups, counts


def rates_from_counts(counts):
    counts = np.asarray(counts)
    if counts.shape != (25, 3) or np.any(counts < 0):
        raise ValueError("Expected nonnegative [25,3] TP/FP/FN counts")
    tp, fp, fn = counts.T
    if np.any(tp + fn == 0):
        return None  # Never give an absent rare class an invented perfect Recall.
    recall = tp / (tp + fn)
    fdr = np.divide(fp, tp + fp, out=np.zeros(25, dtype=float), where=(tp + fp) != 0)
    return {
        name: {"recall": float(recall[ids].mean()), "fdr": float(fdr[ids].mean())}
        for name, ids in COARSE_IDS.items()
    }


def paired_bootstrap(
    base_counts, alt_counts, base_latency, alt_latency, *, repetitions=3000, seed=42, scorer=None
):
    """Resample whole source groups; flag missing-taxonomy replicates.

    Does not invent another final lockbox, remove previous adaptive selection,
    compensate for a mismatched training lineage, or predict the hidden score.
    """
    if scorer is None:
        from rsdet.evaluation.absolute_score import platform_confirmed_score

        def scorer(rows, latency):
            return platform_confirmed_score(rows, latency)["total_score"]

    b, a = np.asarray(base_counts), np.asarray(alt_counts)
    if b.shape != a.shape or b.ndim != 3 or b.shape[1:] != (25, 3) or len(b) < 2:
        raise ValueError("Need paired arrays [at least 2 groups,25,3]")
    if not np.array_equal(b[:, :, 0] + b[:, :, 2], a[:, :, 0] + a[:, :, 2]):
        raise ValueError("Baseline and candidate do not have identical per-group GT counts")
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    rng = np.random.default_rng(seed)
    deltas = []
    missing = 0
    for _ in range(repetitions):
        sample = rng.integers(0, len(b), len(b))
        rb, ra = rates_from_counts(b[sample].sum(0)), rates_from_counts(a[sample].sum(0))
        if rb is None or ra is None:
            missing += 1
            continue
        deltas.append(scorer(ra, alt_latency) - scorer(rb, base_latency))
    output = {
        "repetitions": repetitions,
        "valid_repetitions": len(deltas),
        "missing_taxonomy_repetitions": missing,
        "source_groups": len(b),
        "conditional_on_complete_taxonomy": True,
        "valid_fraction": len(deltas) / repetitions,
    }
    if deltas:
        output.update(
            mean=float(np.mean(deltas)),
            p10=float(np.quantile(deltas, 0.10)),
            p50=float(np.quantile(deltas, 0.50)),
            p90=float(np.quantile(deltas, 0.90)),
            probability_positive=float(np.mean(np.array(deltas) > 0)),
        )
    output["interpretation"] = (
        "inconclusive_sparse_taxonomy"
        if len(deltas) / repetitions < 0.8
        else "paired_development_evidence_not_hidden_score_prediction"
    )
    return output


def validate_frozen_policy(policy):
    """Check declared selection/evaluation separation; not a substitute for lineage audits."""
    if policy.get("frozen_before_evaluation") is not True:
        raise ValueError("Freeze the WHOLE policy before evaluation")
    selected = policy.get("selection_groups", [])
    evaluated = policy.get("evaluation_groups", [])
    if not isinstance(selected, list) or not isinstance(evaluated, list):
        raise ValueError("Policy groups must be lists")
    if set(map(str, selected)) & set(map(str, evaluated)):
        raise ValueError("Policy selection groups overlap evaluation groups")
    for name in ("primary_threshold", "alternative_threshold"):
        value = policy.get(name)
        if (
            value is None
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise ValueError(f"Invalid/missing {name}")
    for name in ("baseline_latency_seconds", "candidate_latency_seconds"):
        value = policy.get(name)
        if (
            value is None
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"Invalid/missing {name}")
    return {
        "selection_groups": list(map(str, selected)),
        "evaluation_groups": list(map(str, evaluated)),
        "declared_groups_nonempty": bool(selected) and bool(evaluated),
    }
