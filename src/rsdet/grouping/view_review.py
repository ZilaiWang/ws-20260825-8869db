"""MAR20 背景视图人工复核的宽表合同。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

REVIEW_METHODS = ("blur", "local_mean", "telea")


def build_view_review_rows(
    audit_rows: Iterable[dict[str, Any]],
    *,
    primary_dilation: float,
    methods: Sequence[str] = REVIEW_METHODS,
) -> list[dict[str, Any]]:
    methods = tuple(methods)
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("review methods 必须非空且唯一")
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    nodes: set[str] = set()
    for row in audit_rows:
        if abs(float(row["dilation_ratio"]) - primary_dilation) > 1e-12:
            continue
        node_uid = str(row["node_uid"])
        fill_method = str(row["fill_method"])
        if fill_method not in methods:
            continue
        key = (node_uid, fill_method)
        if key in selected:
            raise ValueError(f"view audit primary key 重复: {key}")
        selected[key] = row
        nodes.add(node_uid)
    if not nodes:
        raise ValueError("view audit 中没有 primary dilation 行")
    result = []
    for node_uid in sorted(nodes, key=lambda value: int(value.split(":")[1])):
        missing = [method for method in methods if (node_uid, method) not in selected]
        if missing:
            raise ValueError(f"{node_uid}: 缺少 fill methods={missing}")
        tile_counts = {
            int(selected[(node_uid, method)]["background_tile_count"]) for method in methods
        }
        if len(tile_counts) != 1:
            raise ValueError(f"{node_uid}: 同一 dilation 的背景 tile 数随 fill method 改变")
        row: dict[str, Any] = {
            "node_uid": node_uid,
            "valid": "",
        }
        for method in methods:
            row[f"{method}_aircraft_remnant"] = ""
            row[f"{method}_inpaint_artifact"] = ""
        row["background_tile_available"] = int(next(iter(tile_counts)) > 0)
        row["background_tile_aircraft"] = ""
        row["notes"] = ""
        result.append(row)
    return result


def write_view_review_template(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("view review template rows 不能为空")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
