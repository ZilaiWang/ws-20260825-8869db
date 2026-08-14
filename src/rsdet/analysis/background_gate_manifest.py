"""N2-CFG 训练 manifest 构建。

从 R1-6 最终链候选 + formal GT + image ledger + N0 盲审白名单构造
``context_1.25`` 前景门控的训练样本清单。

样本构造严格遵循《改进方案 1》第 2.4 节：

- 正样本：R1-6 候选中的官方 TP（deployable_positive）。oracle_positive 依赖
  N0-3 证据层与低分 reserve pool，属于 N2-P 范畴，首轮不进入；
- 负样本：只允许 N0 盲审白名单中的 ``clear_background``；
- 未确认的 ``FP_BG``、``FP_CLS``、``FP_DUP``、``FP_LOC`` 一律跳过（ignore），
  不进入训练；
- 标签是二分类 ``is_foreground``（1/0），不改候选类别、不改框。

本模块只做 CPU 归因与坐标扩展，不渲染像素（渲染在训练/评估阶段进行）。
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsdet.analysis.background_gate import (
    CONTEXT_EXPANSION,
    coarse_of_category_id,
    expand_context_bbox,
)
from rsdet.analysis.oof_detection import (
    decompose_official_errors,
    load_formal_ground_truth,
)
from rsdet.analysis.proposal_crops import load_image_path_mapping
from rsdet.evaluation.official_metric import evaluate_predictions_with_trace
from rsdet.evaluation.protocol import EvaluationProtocol

# 背景哨兵标签（is_foreground=0 的负样本）。
BACKGROUND_CLASS_ID = 25
BACKGROUND_CLASS_NAME = "background"

GATE_MANIFEST_COLUMNS: tuple[str, ...] = (
    "proposal_uid",
    "image_id",
    "fold",
    "leakage_group_id",
    "coarse",
    "class_id",
    "class_name",
    "view",
    "is_foreground",
    "score",
    "source_prediction_index",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "context_x0",
    "context_y0",
    "context_x1",
    "context_y1",
    "source_relative_path",
    "source_width",
    "source_height",
)


def _load_selected_predictions(
    path: str | Path,
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    """加载 R1-6 最终链候选，按 image_id 分组并保持列表顺序。"""
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("selected_predictions 必须是列表")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    flat: list[dict[str, Any]] = []
    for item in payload:
        image_id = int(item["image_id"])
        record = dict(item)
        record["_image_id"] = image_id
        grouped[image_id].append(record)
        flat.append(record)
    return dict(grouped), flat


def _load_image_ledger(path: str | Path) -> dict[int, dict[str, Any]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        ledger: dict[int, dict[str, Any]] = {}
        for row in reader:
            image_id = int(row["image_id"])
            if image_id in ledger:
                raise ValueError(f"image ledger 重复 image_id={image_id}")
            ledger[image_id] = {
                "fold": int(row["fold"]),
                "group_id": str(row["group_id"]).strip(),
                "relative_path": str(row["relative_path"]).strip(),
            }
    return ledger


def _load_clear_background_uids(path: str | Path) -> set[str]:
    """加载 N0 盲审白名单，返回 clear_background 的 proposal_uid 集合。"""
    if path is None:
        return set()
    with Path(path).expanduser().resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        uids = {str(row["proposal_uid"]).strip() for row in reader}
    if not uids:
        raise ValueError(f"clear_background 白名单为空: {path}")
    return uids


def build_foreground_gate_manifest(
    *,
    selected_predictions_path: str | Path,
    formal_crop_manifest_path: str | Path,
    image_ledger_path: str | Path,
    clear_background_whitelist_path: str | Path | None,
    protocol: EvaluationProtocol,
    formal_manifest_sha256: str,
    context_ratio: float = CONTEXT_EXPANSION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """构造前景门控训练 manifest 行与统计。"""
    predictions, flat = _load_selected_predictions(selected_predictions_path)
    formal = load_formal_ground_truth(
        formal_crop_manifest_path,
        expected_sha256=formal_manifest_sha256,
        expected_images=4481,
        expected_annotations=20933,
    )
    ledger = _load_image_ledger(image_ledger_path)
    if set(ledger) != set(formal.image_ids):
        raise ValueError("image ledger 与 formal GT 的 image_id 集合不一致")
    clear_uids = _load_clear_background_uids(clear_background_whitelist_path)

    # 官方匹配（TP）与错误归因（FP_BG 等）。
    _, trace = evaluate_predictions_with_trace(
        formal.boxes,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    matched_gt_by_pred: dict[tuple[int, int], int] = {}
    for match in trace.matches:
        gt_object = formal.objects[(match.image_id, match.ground_truth_index)]
        matched_gt_by_pred[(match.image_id, match.prediction_index)] = gt_object.category_id

    _, cases, _ = decompose_official_errors(
        formal,
        predictions,
        threshold=0.0,
        protocol=protocol,
        model_key="R1_6_FINAL_CHAIN",
        include_cases=True,
    )
    fp_reason_by_pred: dict[tuple[int, int], str] = {}
    for case in cases:
        if case["case_side"] != "prediction":
            continue
        image_id = int(case["image_id"])
        prediction_index = int(str(case["item_uid"]).rsplit("-p", 1)[1])
        fp_reason_by_pred[(image_id, prediction_index)] = case["reason"]

    path_map = load_image_path_mapping(formal_crop_manifest_path)

    rows: list[dict[str, Any]] = []
    view_counts: defaultdict[str, int] = defaultdict(int)
    skipped: defaultdict[str, int] = defaultdict(int)
    for image_id, candidates in predictions.items():
        for prediction_index, candidate in enumerate(candidates):
            proposal_uid = str(candidate.get("proposal_uid") or f"r1-i{image_id}-p{prediction_index}")
            key = (image_id, prediction_index)
            coarse = coarse_of_category_id(int(candidate["category_id"]))

            view: str | None = None
            is_foreground: int | None = None
            class_id: int | None = None
            class_name: str | None = None
            if key in matched_gt_by_pred:
                view = "deployable_positive"
                is_foreground = 1
                class_id = matched_gt_by_pred[key]
            elif fp_reason_by_pred.get(key) == "FP_BG":
                if proposal_uid in clear_uids:
                    view = "clear_background"
                    is_foreground = 0
                    class_id = BACKGROUND_CLASS_ID
                    class_name = BACKGROUND_CLASS_NAME
                else:
                    skipped["unconfirmed_fp_bg"] += 1
                    continue
            else:
                reason = fp_reason_by_pred.get(key, "UNKNOWN")
                skipped[f"ignored_{reason.lower()}"] += 1
                continue

            if class_name is None:
                from rsdet.data.xh_dataset import FINE_NAMES

                class_name = FINE_NAMES[class_id]

            bbox = [float(value) for value in candidate["bbox_xyxy"]]
            context = expand_context_bbox(bbox, ratio=context_ratio)
            ledger_row = ledger[image_id]
            path_info = path_map[image_id]

            rows.append(
                {
                    "proposal_uid": proposal_uid,
                    "image_id": image_id,
                    "fold": ledger_row["fold"],
                    "leakage_group_id": ledger_row["group_id"],
                    "coarse": coarse,
                    "class_id": class_id,
                    "class_name": class_name,
                    "view": view,
                    "is_foreground": is_foreground,
                    "score": float(candidate["score"]),
                    "source_prediction_index": prediction_index,
                    "bbox_x0": bbox[0],
                    "bbox_y0": bbox[1],
                    "bbox_x1": bbox[2],
                    "bbox_y1": bbox[3],
                    "context_x0": context[0],
                    "context_y0": context[1],
                    "context_x1": context[2],
                    "context_y1": context[3],
                    "source_relative_path": path_info["source_relative_path"],
                    "source_width": path_info["source_width"],
                    "source_height": path_info["source_height"],
                }
            )
            view_counts[view] += 1

    summary = {
        "candidate_count": len(flat),
        "rows_written": len(rows),
        "view_counts": dict(view_counts),
        "skipped": dict(skipped),
        "clear_background_uids": len(clear_uids),
        "context_ratio": context_ratio,
    }
    return rows, summary


def write_gate_manifest(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """写前景门控 manifest CSV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GATE_MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in GATE_MANIFEST_COLUMNS})


__all__ = [
    "BACKGROUND_CLASS_ID",
    "BACKGROUND_CLASS_NAME",
    "GATE_MANIFEST_COLUMNS",
    "build_foreground_gate_manifest",
    "write_gate_manifest",
]
