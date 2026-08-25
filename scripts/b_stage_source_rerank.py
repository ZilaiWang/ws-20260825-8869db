#!/usr/bin/env python3
"""B-stage source-aware candidate audit and cross-fit reranking.

This script is diagnostic.  It reuses the project's official matcher, keeps a
fixed per-image candidate budget, and fits source-family score transforms only
on the other folds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsdet.analysis.oof_detection import (  # noqa: E402
    decompose_official_errors,
    load_formal_ground_truth,
    load_oof_aggregate,
)
from rsdet.evaluation.official_metric import (  # noqa: E402
    evaluate_predictions,
    evaluate_predictions_with_trace,
    evaluate_ranking_metrics,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol  # noqa: E402
from rsdet.postprocess.calibration import (  # noqa: E402
    filter_predictions_by_score,
)
from rsdet.utils.config import load_config  # noqa: E402


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B-stage source audit and cross-fit candidate reranking"
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--project", type=Path, default=PROJECT_ROOT)
    p.add_argument("--seed", type=int, default=20260813)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def size_bin(short_edge: float) -> str:
    if short_edge < 32:
        return "<32"
    if short_edge < 64:
        return "32-64"
    if short_edge < 128:
        return "64-128"
    if short_edge < 256:
        return "128-256"
    return ">=256"


def source_family(group: str) -> str:
    if group.startswith("mar20-airport-proxy-"):
        return "aircraft_source_family"
    if group.startswith("scene_"):
        return "ship_source_family"
    if group.startswith("site_"):
        return "vehicle_source_family"
    return "unknown_source_family"


def main() -> int:
    a = args()
    root = a.root.resolve()
    out = a.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    aggregate = root / "M1-CV3-OOF-aggregate"
    manifest = root / "formal_crop_manifest.csv"
    project_config = load_config(a.project / "configs/project.yaml")
    protocol = parse_evaluation_protocol(project_config)
    formal = load_formal_ground_truth(manifest)
    metadata, predictions = load_oof_aggregate(
        aggregate,
        expected_model_key="M1",
        expected_manifest_sha256="27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331",
        expected_formal_crop_sha256="a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128",
        expected_images=4481,
        candidate_floor=0.001,
    )
    image_rows = {int(r["image_id"]): r for r in read_csv(aggregate / "oof_images.csv")}
    proposal_rows = read_csv(aggregate / "oof_proposals.csv")
    if len(proposal_rows) != metadata["proposal_count"]:
        raise ValueError("proposal count mismatch")

    # Build a strict proposal-to-per-image-index map. source_prediction_index is
    # fold-global and therefore cannot index predictions[image_id] directly.
    local_counts: Counter[int] = Counter()
    local_index_by_uid: dict[str, int] = {}
    for row in proposal_rows:
        image_id = int(row["image_id"])
        local_index_by_uid[row["proposal_uid"]] = local_counts[image_id]
        local_counts[image_id] += 1
    expected_counts = {image_id: len(rows) for image_id, rows in predictions.items()}
    if dict(local_counts) != expected_counts:
        raise ValueError("proposal per-image order/count mismatch")

    # Official matching at the exploratory workpoint supplies stable labels.
    selected = filter_predictions_by_score(predictions, 0.051)
    selected_local_index = {
        (image_id, id(item)): index
        for image_id, items in predictions.items()
        for index, item in enumerate(items)
    }
    _, trace = evaluate_predictions_with_trace(
        formal.boxes,
        selected,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    labels: dict[tuple[int, int], str] = {}
    for item in trace.matches:
        labels[
            (
                item.image_id,
                selected_local_index[
                    (item.image_id, id(selected[item.image_id][item.prediction_index]))
                ],
            )
        ] = "TP"
    for item in trace.unmatched_predictions:
        labels[
            (
                item.image_id,
                selected_local_index[
                    (item.image_id, id(selected[item.image_id][item.prediction_index]))
                ],
            )
        ] = "FP"
    _, floor_cases, _ = decompose_official_errors(
        formal, predictions, threshold=0.001, protocol=protocol, model_key="M1", include_cases=True
    )
    floor_error_map: dict[tuple[int, int], str] = {}
    for case in floor_cases:
        if case["case_side"] == "prediction":
            floor_error_map[
                (int(case["image_id"]), int(str(case["item_uid"]).rsplit("-p", 1)[1]))
            ] = str(case["reason"])
    _, floor_trace = evaluate_predictions_with_trace(
        formal.boxes,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    floor_tp = {(x.image_id, x.prediction_index) for x in floor_trace.matches}

    # The proposal ledger and prediction JSON have identical per-image order.
    audit: list[dict[str, Any]] = []
    for row in proposal_rows:
        image_id = int(row["image_id"])
        pred_index = local_index_by_uid[row["proposal_uid"]]
        prediction = predictions[image_id][pred_index]
        score = float(prediction["score"])
        width = float(row["width"])
        height = float(row["height"])
        group = image_rows[image_id]["group_id"]
        label = labels.get((image_id, pred_index), "BELOW_WORKPOINT")
        # Detailed FP attribution is available from the official error CSV; here
        # preserve a conservative label and use exact official matching for rerank.
        audit.append(
            {
                "proposal_uid": row["proposal_uid"],
                "image_id": image_id,
                "fold": int(image_rows[image_id]["fold"]),
                "group_id": group,
                "relative_path": image_rows[image_id].get("relative_path", ""),
                "source_family": source_family(group),
                "category_id": int(row["category_id"]),
                "score": score,
                "width": width,
                "height": height,
                "bbox_xyxy": " ".join(f"{float(row[k]):.10g}" for k in ("x", "y"))
                + " "
                + " ".join(
                    f"{float(float(row[k]) + float(row[dim])):.10g}"
                    for k, dim in (("x", "width"), ("y", "height"))
                ),
                "short_edge": min(width, height),
                "size_bin": size_bin(min(width, height)),
                "score_bin": "<0.051" if score < 0.051 else ">=0.051",
                "official_label_at_0.051": label,
                "official_label_at_0.001": (
                    "TP"
                    if (image_id, pred_index) in floor_tp
                    else floor_error_map.get((image_id, pred_index), "UNATTRIBUTED")
                ),
                "local_prediction_index": pred_index,
            }
        )
    fields = list(audit[0])
    with (out / "candidate_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)

    # Full diagnostic error decomposition is imported from the project's analysis entry point.
    for row in audit:
        key = (row["image_id"], row["local_prediction_index"])
        # Full-floor attribution uses the original per-image candidate index;
        # this is the stable audit label used by the source/size tables.
        row["error_type"] = row["official_label_at_0.001"]
    fields = list(audit[0])
    with (out / "candidate_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)

    def write_summary(name: str, keys: tuple[str, ...]) -> None:
        counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        for row in audit:
            k = tuple(str(row[x]) for x in keys)
            counts[k]["proposals"] += 1
            counts[k][str(row["error_type"])] += 1
        rows = []
        for k, c in sorted(counts.items()):
            item = {key: value for key, value in zip(keys, k)}
            item.update(c)
            rows.append(item)
        (out / name).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    write_summary("source_family_error_summary.json", ("source_family",))
    write_summary("exact_group_error_summary.json", ("group_id",))
    write_summary("source_size_error_summary.json", ("source_family", "size_bin"))
    write_summary("fold_source_error_summary.json", ("fold", "source_family"))

    # Build fixed-budget rerank candidates. Each held-out fold keeps exactly the
    # number of candidates kept by raw score >= 0.051 in that fold.
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in audit:
        by_image[row["image_id"]].append(row)
    result_rows: list[dict[str, Any]] = []
    methods = ("raw_score", "source_quantile", "source_size_quantile")
    for heldout in range(3):
        train = [r for r in audit if r["fold"] != heldout and r["score"] >= 0.001]
        global_values = sorted(r["score"] for r in train)
        source_values: dict[str, list[float]] = defaultdict(list)
        source_size_values: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in train:
            source_values[r["source_family"]].append(r["score"])
            source_size_values[(r["source_family"], r["size_bin"])].append(r["score"])
        source_values = {k: sorted(v) for k, v in source_values.items()}
        source_size_values = {k: sorted(v) for k, v in source_size_values.items()}
        raw_budget_by_image = {
            image_id: sum(r["score"] >= 0.051 for r in rows)
            for image_id, rows in by_image.items()
            if int(image_rows[image_id]["fold"]) == heldout
        }
        fixed_budget = sum(raw_budget_by_image.values())
        for method in methods:
            reranked: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for image_id, image_rows_for_image in by_image.items():
                if int(image_rows[image_id]["fold"]) != heldout:
                    continue
                scored_rows: list[tuple[float, float, str, dict[str, Any]]] = []
                for r in image_rows_for_image:
                    transformed = r["score"]
                    if method != "raw_score":
                        source_vals = source_values.get(r["source_family"], global_values)
                        source_q = bisect_right(source_vals, r["score"]) / max(1, len(source_vals))
                        transformed = source_q
                        if method == "source_size_quantile":
                            cell_vals = source_size_values.get(
                                (r["source_family"], r["size_bin"]), []
                            )
                            if cell_vals:
                                cell_q = bisect_right(cell_vals, r["score"]) / len(cell_vals)
                                weight = len(cell_vals) / (len(cell_vals) + 200.0)
                                transformed = weight * cell_q + (1.0 - weight) * source_q
                    scored_rows.append((float(transformed), r["score"], r["proposal_uid"], r))
                chosen = sorted(scored_rows, key=lambda x: (-x[0], -x[1], x[2]))[
                    : raw_budget_by_image[image_id]
                ]
                for transformed, _, _, r in chosen:
                    item = predictions[image_id][r["local_prediction_index"]].copy()
                    item["score"] = transformed
                    reranked[image_id].append(item)
            for image_id in (i for i, r in image_rows.items() if int(r["fold"]) == heldout):
                reranked.setdefault(image_id, [])
            gt = {i: formal.boxes.get(i, []) for i in reranked}
            metrics = evaluate_predictions(
                gt,
                reranked,
                class_names=protocol.class_names,
                category_mapping=protocol.category_mapping,
                iou_thresholds=protocol.iou_thresholds,
            )
            ranking = evaluate_ranking_metrics(
                gt,
                reranked,
                class_names=protocol.class_names,
                category_mapping=protocol.category_mapping,
                iou_thresholds=protocol.iou_thresholds,
            )
            result_rows.append(
                {
                    "heldout_fold": heldout,
                    "method": method,
                    "candidate_budget": fixed_budget,
                    "recall": metrics.recall,
                    "fdr": metrics.fdr,
                    "tp": metrics.details["tp"],
                    "fp": metrics.details["fp"],
                    "fn": metrics.details["fn"],
                    "official_macro_recall": ranking.overall_recall,
                    "official_macro_fdr": ranking.overall_fdr,
                }
            )
    with (out / "rerank_fold_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_rows)

    # Stratified FP_BG review sample (deterministic, no image pixels copied).
    bg = [r for r in audit if r["error_type"] == "FP_BG"]
    rng = random.Random(a.seed)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in bg:
        key = (r["source_family"], str(r["category_id"]), r["score_bin"])
        buckets[key].append(r)
    sample: list[dict[str, Any]] = []
    for key, rows in sorted(buckets.items()):
        rng.shuffle(rows)
        sample.extend(rows[: min(10, len(rows))])
    sample_fields = [
        "proposal_uid",
        "image_id",
        "fold",
        "relative_path",
        "group_id",
        "source_family",
        "category_id",
        "score",
        "bbox_xyxy",
        "width",
        "height",
        "size_bin",
        "score_bin",
        "manual_label",
    ]
    with (out / "fp_bg_review_sample.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sample_fields, lineterminator="\n")
        writer.writeheader()
        for r in sample:
            writer.writerow(
                {
                    **{key: r.get(key, "") for key in sample_fields if key != "manual_label"},
                    "manual_label": "",
                }
            )
    metadata_out = {
        "status": "complete_b1_b2_exploratory",
        "input_sha256": {
            "formal_crop_manifest": sha256(manifest),
            "oof_proposals": sha256(aggregate / "oof_proposals.csv"),
        },
        "counts": {
            "images": len(image_rows),
            "proposals": len(audit),
            "fp_bg_review_sample": len(sample),
        },
        "methods": list(methods),
        "official_workpoint": 0.051,
        "budget_policy": "per-image raw score >= 0.051 candidate count",
        "scientific_scope": {
            "cross_fit": True,
            "fixed_candidate_budget": True,
            "deployment_final": False,
        },
    }
    (out / "b_stage_metadata.json").write_text(
        json.dumps(metadata_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"B_STAGE_PASS images={len(image_rows)} proposals={len(audit)} sample={len(sample)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
