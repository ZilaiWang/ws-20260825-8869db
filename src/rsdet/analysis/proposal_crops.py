"""N2：从对象证据 manifest 生成 Pred-OOF proposal crop manifest。

背景
----
N0-3 已生成统一对象证据层（pred_oof_evidence.json），每个候选含
proposal_uid、bbox_xyxy、fold、官方状态、GT 匹配、oracle 命中。N2 对象学生
需要把这些候选渲染成 crop 用于训练/重分类。

本模块把对象证据 manifest + formal crop manifest（提供原图路径）合并为
**proposal crop manifest**（CSV），每行一条候选 crop：

- ``crop_xyxy``：候选框（可直接喂 ``render_crop``）；
- ``fold``：候选所在 CV3 fold；
- ``leakage_group_id``：source_group（防泄漏按来源组划分）；
- 训练标签：对官方 TP 用 ``matched_gt_category``（真值细类，学生学"纠正细类"），
  对 FP 用预测类别（学生学"保持但不强化错误"），hard_negative 用背景标签；
- ``view``：oracle_positive / deployable_positive / hard_negative；
- ``is_background``：hard_negative 中明确标为背景的候选（N0-4 人工审计后才可
  作为真实背景样本；默认仅用 FP_BG 且无 oracle hit 的）。

防泄漏
------
每个 held-out fold 的对象学生只能用另外两个 fold 的候选训练；本模块只负责
生成 manifest，训练时的 fold 划分由调用方（train_object_student）按
``leakage_group_id`` 控制，与 N0-3 的 fold 归属一致。
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

# 背景标签：用一个超出 25 细类范围的哨兵值。
BACKGROUND_CLASS_ID = 25
BACKGROUND_CLASS_NAME = "background"

ALLOWED_VIEWS: frozenset[str] = frozenset(
    {"oracle_positive", "deployable_positive", "hard_negative"}
)

PROPOSAL_CROP_COLUMNS: tuple[str, ...] = (
    "proposal_uid",
    "image_id",
    "annotation_uid",
    "source_relative_path",
    "source_width",
    "source_height",
    "fold",
    "leakage_group_id",
    "class_id",
    "class_name",
    "major_class",
    "crop_policy",
    "crop_xyxy",
    "outside_policy",
    "resize_semantics",
    "color_mode",
    "view",
    "is_background",
    "official_status",
    "score",
    "oracle_hit",
    "matched_gt_category",
)


def _parse_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} 非整数: {value!r}") from error


def _parse_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} 非数值: {value!r}") from error
    return result


def load_image_path_mapping(
    formal_manifest_path: str | Path,
) -> dict[int, dict[str, Any]]:
    """从 formal crop manifest 构建 ``{formal_image_id: {path,w,h}}``。"""
    manifest = Path(formal_manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"formal manifest 不存在: {manifest}")
    mapping: dict[int, dict[str, Any]] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("crop_policy", "").strip() != "tight":
                continue
            image_id = _parse_int(row["formal_image_id"], "formal_image_id")
            if image_id in mapping:
                continue
            mapping[image_id] = {
                "source_relative_path": row["source_relative_path"].strip(),
                "source_width": _parse_int(row["source_width"], "source_width"),
                "source_height": _parse_int(row["source_height"], "source_height"),
            }
    return mapping


def _major_class_of(class_id: int) -> str:
    if class_id < 4:
        return "ship"
    if class_id < 24:
        return "aircraft"
    if class_id == 24:
        return "vehicle"
    return "background"


def _class_name_of(class_id: int) -> str:
    if class_id == 25:
        return BACKGROUND_CLASS_NAME
    from rsdet.data.crop_classification import FINE_NAMES

    return FINE_NAMES[class_id]


def build_proposal_crop_manifest(
    *,
    evidence_manifest: Mapping[str, Any],
    formal_manifest_path: str | Path,
    include_views: Sequence[str] | None = None,
    background_candidates: str = "fp_bg_no_oracle",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """从对象证据 manifest 生成 proposal crop 记录。

    Args:
        evidence_manifest: N0-3 pred_oof_evidence.json（dict）。
        formal_manifest_path: formal crop manifest CSV（提供原图路径）。
        include_views: 包含哪些视图（默认全部三种）。
        background_candidates: hard_negative 中哪些算背景：
            "fp_bg_no_oracle"（默认）＝ official_status==FP_BG 且无 oracle hit。

    Returns:
        (rows, summary)。rows 可直接写 CSV。
    """
    if include_views is None:
        include_views = list(ALLOWED_VIEWS)
    for view in include_views:
        if view not in ALLOWED_VIEWS:
            raise ValueError(f"未知视图 {view!r}（合法: {sorted(ALLOWED_VIEWS)}）")

    path_map = load_image_path_mapping(formal_manifest_path)
    records = evidence_manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("evidence manifest 的 records 必须是列表")

    rows: list[dict[str, Any]] = []
    missing_images: set[int] = set()
    background_count = 0
    view_counts: defaultdict[str, int] = defaultdict(int)

    for record in records:
        image_id = _parse_int(record["image_id"], "image_id")
        path_info = path_map.get(image_id)
        if path_info is None:
            missing_images.add(image_id)
            continue
        official_status = str(record.get("official_status", ""))
        view: str | None = None
        if official_status == "TP":
            view = "deployable_positive"
        elif record.get("oracle_hit"):
            view = "oracle_positive"
        else:
            view = "hard_negative"
        if view not in include_views:
            continue
        view_counts[view] += 1

        matched_gt_category = record.get("matched_gt_category")
        is_background = False
        if view == "hard_negative":
            if background_candidates == "fp_bg_no_oracle":
                is_background = (
                    official_status == "FP_BG" and not bool(record.get("oracle_hit"))
                )
            else:
                raise ValueError(
                    f"未知 background_candidates={background_candidates!r}"
                )
            if is_background:
                background_count += 1

        # 训练标签：TP→真值细类；FP→预测类别（保持原类别，学生可纠错）；
        # 背景→哨兵 25。
        if is_background:
            class_id = BACKGROUND_CLASS_ID
        elif matched_gt_category is not None:
            class_id = _parse_int(matched_gt_category, "matched_gt_category")
        else:
            class_id = _parse_int(record["category_id"], "category_id")

        bbox = [_parse_float(v, "bbox") for v in record["bbox_xyxy"]]
        if len(bbox) != 4:
            raise ValueError(f"proposal {record['proposal_uid']} 的 bbox 必须 4 个数")

        rows.append(
            {
                "proposal_uid": str(record["proposal_uid"]),
                "image_id": image_id,
                "annotation_uid": f"pred-{record['proposal_uid']}",
                "source_relative_path": path_info["source_relative_path"],
                "source_width": path_info["source_width"],
                "source_height": path_info["source_height"],
                "fold": _parse_int(record["fold"], "fold"),
                "leakage_group_id": str(record.get("source_group", "")),
                "class_id": class_id,
                "class_name": _class_name_of(class_id),
                "major_class": _major_class_of(class_id),
                "crop_policy": "tight",
                "crop_xyxy": ",".join(f"{value:.4f}" for value in bbox),
                "outside_policy": "pad",
                "resize_semantics": "direct_square_resize",
                "color_mode": "RGB",
                "view": view,
                "is_background": str(is_background).lower(),
                "official_status": official_status,
                "score": _parse_float(record["score"], "score"),
                "oracle_hit": str(bool(record.get("oracle_hit"))).lower(),
                "matched_gt_category": (
                    str(matched_gt_category)
                    if matched_gt_category is not None
                    else ""
                ),
            }
        )

    summary = {
        "total_candidates": len(records),
        "rows_written": len(rows),
        "missing_image_paths": len(missing_images),
        "background_candidates": background_count,
        "view_counts": dict(view_counts),
    }
    if missing_images:
        summary["missing_image_ids_sample"] = sorted(missing_images)[:10]
    return rows, summary


def write_proposal_crop_manifest(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    """写 proposal crop manifest CSV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROPOSAL_CROP_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in PROPOSAL_CROP_COLUMNS})


def split_by_fold(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """按 fold 拆分 proposal crop 行。"""
    folded: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        folded[int(row["fold"])].append(dict(row))
    return dict(folded)


def stats_by_view(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """按视图统计（对象数、背景数、各类计数）。"""
    stats: dict[str, dict[str, int]] = {}
    for row in rows:
        view = str(row["view"])
        entry = stats.setdefault(view, {"n": 0, "background": 0, "classes": {}})
        entry["n"] += 1
        if str(row["is_background"]).lower() == "true":
            entry["background"] += 1
        class_id = int(row["class_id"])
        entry["classes"][class_id] = entry["classes"].get(class_id, 0) + 1
    return stats
