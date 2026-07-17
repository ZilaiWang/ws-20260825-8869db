"""对象尺度与 feature-grid 几何可见性审计。

本模块只回答“一个完整 bbox 在固定预处理后占据多少连续网格跨度”。
它不模拟 tile 截断，不测量真实前景面积，也不把网格跨度解释为模型精度。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.sax.saxutils import escape

import numpy as np
import yaml

from rsdet.data.xh_dataset import COARSE_NAMES, FINE_NAMES, coarse_name

REQUIRED_BBOX_COLUMNS = {
    "annotation_uid",
    "image_uid",
    "category_id",
    "category_name",
    "macro_category",
    "width_norm",
    "height_norm",
    "x0",
    "y0",
    "x1",
    "y1",
    "width_px",
    "height_px",
    "issue_flags",
}
REQUIRED_IMAGE_COLUMNS = {"image_uid", "width", "height"}


@dataclass(frozen=True)
class ObjectBox:
    """通过完整性和类别映射校验的原始 bbox。"""

    annotation_uid: str
    image_uid: str
    category_id: int
    category_name: str
    macro_category: str
    image_width: float
    image_height: float
    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    height: float
    issue_flags: str

    @property
    def native_short(self) -> float:
        return min(self.width, self.height)

    @property
    def native_long(self) -> float:
        return max(self.width, self.height)

    @property
    def source_edge_risk(self) -> bool:
        return bool(self.issue_flags)


@dataclass(frozen=True)
class Representation:
    id: str
    family: str
    step: float


@dataclass(frozen=True)
class Scenario:
    id: str
    kind: str
    tile_size: float | None = None
    input_size: float | None = None
    crop_resolution: float | None = None
    context_scale: float | None = None

    @property
    def scale_factor(self) -> float | None:
        if self.kind == "tile":
            assert self.tile_size is not None and self.input_size is not None
            return self.input_size / self.tile_size
        return None


@dataclass(frozen=True)
class ScenarioValues:
    resized_width: np.ndarray
    resized_height: np.ndarray
    resized_short: np.ndarray
    resized_long: np.ndarray
    allocation_gain: np.ndarray
    crop_nominal_side: np.ndarray
    padding_fraction: np.ndarray
    bbox_visible_fraction: np.ndarray
    grid_width: dict[str, np.ndarray]
    grid_height: dict[str, np.ndarray]
    grid_short: dict[str, np.ndarray]
    grid_area: dict[str, np.ndarray]


def _finite_positive(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} 必须是数值，实际为 {value!r}") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} 必须是有限正数，实际为 {value!r}")
    return number


def _finite(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} 必须是数值，实际为 {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} 必须是有限数，实际为 {value!r}")
    return number


def _require_columns(fieldnames: Sequence[str] | None, required: set[str], path: Path) -> None:
    actual = set(fieldnames or ())
    missing = sorted(required - actual)
    if missing:
        raise ValueError(f"{path} 缺少必需列: {', '.join(missing)}")


def _read_image_dimensions(path: Path) -> dict[str, tuple[float, float]]:
    dimensions: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        _require_columns(reader.fieldnames, REQUIRED_IMAGE_COLUMNS, path)
        for line_number, row in enumerate(reader, start=2):
            uid = (row.get("image_uid") or "").strip()
            if not uid:
                raise ValueError(f"{path}:{line_number}: image_uid 为空")
            if uid in dimensions:
                raise ValueError(f"{path}:{line_number}: image_uid 重复: {uid}")
            dimensions[uid] = (
                _finite_positive(row.get("width"), f"{path}:{line_number}:width"),
                _finite_positive(row.get("height"), f"{path}:{line_number}:height"),
            )
    if not dimensions:
        raise ValueError(f"{path} 没有图像记录")
    return dimensions


def load_object_boxes(bbox_path: str | Path, image_stats_path: str | Path) -> list[ObjectBox]:
    """读取审计 CSV，保留贴边/越界框并严格校验类别映射。"""

    bbox_path = Path(bbox_path).expanduser().resolve()
    image_stats_path = Path(image_stats_path).expanduser().resolve()
    dimensions = _read_image_dimensions(image_stats_path)
    objects: list[ObjectBox] = []
    seen: set[str] = set()

    with bbox_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        _require_columns(reader.fieldnames, REQUIRED_BBOX_COLUMNS, bbox_path)
        for line_number, row in enumerate(reader, start=2):
            prefix = f"{bbox_path}:{line_number}"
            annotation_uid = (row.get("annotation_uid") or "").strip()
            image_uid = (row.get("image_uid") or "").strip()
            if not annotation_uid or not image_uid:
                raise ValueError(f"{prefix}: annotation_uid/image_uid 不得为空")
            if annotation_uid in seen:
                raise ValueError(f"{prefix}: annotation_uid 重复: {annotation_uid}")
            if image_uid not in dimensions:
                raise ValueError(f"{prefix}: image_uid 未在 image_stats 中找到: {image_uid}")

            try:
                category_id = int(row.get("category_id", ""))
            except ValueError as error:
                raise ValueError(f"{prefix}: category_id 非法") from error
            if not 0 <= category_id < len(FINE_NAMES):
                raise ValueError(f"{prefix}: category_id 超出 0--24: {category_id}")
            category_name = (row.get("category_name") or "").strip()
            macro_category = (row.get("macro_category") or "").strip()
            expected_name = FINE_NAMES[category_id]
            expected_macro = coarse_name(category_id)
            if category_name != expected_name or macro_category != expected_macro:
                raise ValueError(
                    f"{prefix}: 类别映射冲突，实际=({category_id},{category_name},"
                    f"{macro_category})，期望=({category_id},{expected_name},{expected_macro})"
                )

            image_width, image_height = dimensions[image_uid]
            width = _finite_positive(row.get("width_px"), f"{prefix}:width_px")
            height = _finite_positive(row.get("height_px"), f"{prefix}:height_px")
            width_norm = _finite_positive(row.get("width_norm"), f"{prefix}:width_norm")
            height_norm = _finite_positive(row.get("height_norm"), f"{prefix}:height_norm")
            if not math.isclose(width, width_norm * image_width, rel_tol=1e-5, abs_tol=0.02):
                raise ValueError(f"{prefix}: width_px 与 width_norm * image_width 不一致")
            if not math.isclose(height, height_norm * image_height, rel_tol=1e-5, abs_tol=0.02):
                raise ValueError(f"{prefix}: height_px 与 height_norm * image_height 不一致")

            x0 = _finite(row.get("x0"), f"{prefix}:x0")
            y0 = _finite(row.get("y0"), f"{prefix}:y0")
            x1 = _finite(row.get("x1"), f"{prefix}:x1")
            y1 = _finite(row.get("y1"), f"{prefix}:y1")
            if not math.isclose(x1 - x0, width, rel_tol=1e-5, abs_tol=0.02):
                raise ValueError(f"{prefix}: x1-x0 与 width_px 不一致")
            if not math.isclose(y1 - y0, height, rel_tol=1e-5, abs_tol=0.02):
                raise ValueError(f"{prefix}: y1-y0 与 height_px 不一致")

            seen.add(annotation_uid)
            objects.append(
                ObjectBox(
                    annotation_uid=annotation_uid,
                    image_uid=image_uid,
                    category_id=category_id,
                    category_name=category_name,
                    macro_category=macro_category,
                    image_width=image_width,
                    image_height=image_height,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    width=width,
                    height=height,
                    issue_flags=(row.get("issue_flags") or "").strip(),
                )
            )

    if not objects:
        raise ValueError(f"{bbox_path} 没有 bbox 记录")
    return sorted(objects, key=lambda item: item.annotation_uid)


def load_analysis_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError("分析配置顶层必须是映射")
    return config


def parse_analysis_config(
    config: Mapping[str, Any],
) -> tuple[list[Scenario], list[Representation], tuple[float, float, float]]:
    """将 YAML 配置展开为稳定有序的场景和表征网格。"""

    scenarios: list[Scenario] = []
    for raw in config.get("tile_scenarios", []):
        if not isinstance(raw, Mapping):
            raise TypeError("tile_scenarios 每项必须是映射")
        scenarios.append(
            Scenario(
                id=str(raw.get("id", "")).strip(),
                kind="tile",
                tile_size=_finite_positive(raw.get("tile_size"), "tile_size"),
                input_size=_finite_positive(raw.get("input_size"), "input_size"),
            )
        )

    crop_config = config.get("crop_scenarios", {})
    if not isinstance(crop_config, Mapping):
        raise TypeError("crop_scenarios 必须是映射")
    for resolution_raw in crop_config.get("resolutions", []):
        resolution = _finite_positive(resolution_raw, "crop resolution")
        for context_raw in crop_config.get("context_scales", []):
            context = _finite_positive(context_raw, "context_scale")
            context_id = f"{context:g}".replace(".", "p")
            scenarios.append(
                Scenario(
                    id=f"crop_{resolution:g}_ctx{context_id}",
                    kind="crop",
                    crop_resolution=resolution,
                    context_scale=context,
                )
            )

    representations: list[Representation] = []
    for raw in config.get("representations", []):
        if not isinstance(raw, Mapping):
            raise TypeError("representations 每项必须是映射")
        representations.append(
            Representation(
                id=str(raw.get("id", "")).strip(),
                family=str(raw.get("family", "")).strip(),
                step=_finite_positive(raw.get("step"), "representation step"),
            )
        )

    if not scenarios or not representations:
        raise ValueError("至少需要一个场景和一个表征网格")
    scenario_ids = [item.id for item in scenarios]
    representation_ids = [item.id for item in representations]
    if any(not item for item in scenario_ids + representation_ids):
        raise ValueError("场景和表征 ID 不得为空")
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("场景 ID 重复")
    if len(representation_ids) != len(set(representation_ids)):
        raise ValueError("表征 ID 重复")

    thresholds_config = config.get("risk_thresholds", {})
    if not isinstance(thresholds_config, Mapping):
        raise TypeError("risk_thresholds 必须是映射")
    thresholds = (
        _finite_positive(thresholds_config.get("severe_upper"), "severe_upper"),
        _finite_positive(
            thresholds_config.get("constrained_upper"), "constrained_upper"
        ),
        _finite_positive(thresholds_config.get("moderate_upper"), "moderate_upper"),
    )
    if not thresholds[0] < thresholds[1] < thresholds[2]:
        raise ValueError("风险边界必须严格递增")

    primary = config.get("primary_representations", {})
    if not isinstance(primary, Mapping):
        raise TypeError("primary_representations 必须是映射")
    missing_primary = sorted(set(primary.values()) - set(representation_ids))
    if missing_primary:
        raise ValueError(f"主要表征未定义: {', '.join(missing_primary)}")
    return scenarios, representations, thresholds


def classify_risk(values: np.ndarray, thresholds: tuple[float, float, float]) -> np.ndarray:
    """返回 0/1/2/3，分别对应 <2、[2,4)、[4,8)、>=8 类型的配置档。"""

    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("grid span 必须是有限非负数")
    return np.digitize(values, thresholds, right=False).astype(np.int8)


def _visible_fraction(objects: Sequence[ObjectBox]) -> np.ndarray:
    result = np.empty(len(objects), dtype=np.float64)
    for index, obj in enumerate(objects):
        clipped_width = max(0.0, min(obj.x1, obj.image_width) - max(obj.x0, 0.0))
        clipped_height = max(0.0, min(obj.y1, obj.image_height) - max(obj.y0, 0.0))
        result[index] = np.clip(clipped_width * clipped_height / (obj.width * obj.height), 0, 1)
    return result


def compute_scenario_values(
    objects: Sequence[ObjectBox],
    scenario: Scenario,
    representations: Sequence[Representation],
) -> ScenarioValues:
    """计算逐对象几何跨度；crop 超出原图时保持名义方窗并 padding。"""

    if not objects:
        raise ValueError("objects 不得为空")
    width = np.asarray([item.width for item in objects], dtype=np.float64)
    height = np.asarray([item.height for item in objects], dtype=np.float64)
    native_short = np.minimum(width, height)
    crop_nominal_side = np.full(len(objects), np.nan, dtype=np.float64)
    padding_fraction = np.zeros(len(objects), dtype=np.float64)

    if scenario.kind == "tile":
        if scenario.scale_factor is None:
            raise ValueError("tile 场景缺少缩放因子")
        resized_width = width * scenario.scale_factor
        resized_height = height * scenario.scale_factor
    elif scenario.kind == "crop":
        if scenario.crop_resolution is None or scenario.context_scale is None:
            raise ValueError("crop 场景缺少 resolution/context")
        native_long = np.maximum(width, height)
        crop_nominal_side = scenario.context_scale * native_long
        # 先固定 resolution/context 比例，保证 224/1.0 与 336/1.5
        # 这类数学等价场景在风险边界上也字节级一致。
        content_scale = scenario.crop_resolution / scenario.context_scale
        resized_width = content_scale * width / native_long
        resized_height = content_scale * height / native_long
        for index, obj in enumerate(objects):
            side = crop_nominal_side[index]
            center_x = (obj.x0 + obj.x1) / 2.0
            center_y = (obj.y0 + obj.y1) / 2.0
            crop_x0, crop_x1 = center_x - side / 2.0, center_x + side / 2.0
            crop_y0, crop_y1 = center_y - side / 2.0, center_y + side / 2.0
            valid_width = max(0.0, min(crop_x1, obj.image_width) - max(crop_x0, 0.0))
            valid_height = max(0.0, min(crop_y1, obj.image_height) - max(crop_y0, 0.0))
            valid_fraction = np.clip(valid_width * valid_height / (side * side), 0, 1)
            padding_fraction[index] = 1.0 - valid_fraction
    else:
        raise ValueError(f"未知场景类型: {scenario.kind}")

    resized_short = np.minimum(resized_width, resized_height)
    resized_long = np.maximum(resized_width, resized_height)
    allocation_gain = resized_short / native_short
    grid_width = {item.id: resized_width / item.step for item in representations}
    grid_height = {item.id: resized_height / item.step for item in representations}
    grid_short = {
        item.id: np.minimum(grid_width[item.id], grid_height[item.id])
        for item in representations
    }
    grid_area = {
        item.id: grid_width[item.id] * grid_height[item.id] for item in representations
    }
    return ScenarioValues(
        resized_width=resized_width,
        resized_height=resized_height,
        resized_short=resized_short,
        resized_long=resized_long,
        allocation_gain=allocation_gain,
        crop_nominal_side=crop_nominal_side,
        padding_fraction=padding_fraction,
        bbox_visible_fraction=_visible_fraction(objects),
        grid_width=grid_width,
        grid_height=grid_height,
        grid_short=grid_short,
        grid_area=grid_area,
    )


def _format_float(value: float) -> str:
    if math.isnan(value):
        return ""
    # 17 位有效数可对 float64 无损往返，确保从 CSV 重算 2/4/8
    # 边界分档时与内存中的原始结果一致。
    return format(float(value), ".17g")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
    }


def _scope_masks(objects: Sequence[ObjectBox]) -> dict[str, np.ndarray]:
    edge = np.asarray([item.source_edge_risk for item in objects], dtype=bool)
    out_of_bounds = np.asarray(
        ["out_of_bounds" in item.issue_flags.split(";") for item in objects], dtype=bool
    )
    touches_only = np.asarray(
        [item.issue_flags == "touches_edge_within_1px" for item in objects], dtype=bool
    )
    return {
        "all": np.ones(len(objects), dtype=bool),
        "clean_in_bounds": ~edge,
        "source_edge_risk": edge,
        "touches_edge_in_bounds": touches_only,
        "out_of_bounds": out_of_bounds,
    }


def _group_masks(objects: Sequence[ObjectBox]) -> list[tuple[str, str, np.ndarray]]:
    rows: list[tuple[str, str, np.ndarray]] = [
        ("overall", "all_objects", np.ones(len(objects), dtype=bool))
    ]
    for macro in COARSE_NAMES:
        rows.append(
            (
                "macro",
                macro,
                np.asarray([item.macro_category == macro for item in objects], dtype=bool),
            )
        )
    for category_id, name in enumerate(FINE_NAMES):
        rows.append(
            (
                "fine",
                name,
                np.asarray([item.category_id == category_id for item in objects], dtype=bool),
            )
        )
    return rows


def _percentile_columns(
    prefix: str, values: np.ndarray, percentiles: Sequence[float]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for percentile, value in zip(percentiles, np.percentile(values, percentiles)):
        result[f"{prefix}_p{int(percentile):02d}"] = float(value)
    return result


def summarize_scenarios(
    objects: Sequence[ObjectBox],
    scenarios: Sequence[Scenario],
    values_by_scenario: Mapping[str, ScenarioValues],
    representations: Sequence[Representation],
    thresholds: tuple[float, float, float],
    percentiles: Sequence[float],
    scopes: Sequence[str],
) -> list[dict[str, Any]]:
    """生成按场景、数据质量、大类/细类分层的可追溯汇总。"""

    native_short = np.asarray([item.native_short for item in objects], dtype=np.float64)
    image_uids = np.asarray([item.image_uid for item in objects], dtype=object)
    scope_masks = _scope_masks(objects)
    unknown_scopes = sorted(set(scopes) - set(scope_masks))
    if unknown_scopes:
        raise ValueError(f"未知 summary scope: {', '.join(unknown_scopes)}")
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        values = values_by_scenario[scenario.id]
        for scope in scopes:
            for group_level, group_name, group_mask in _group_masks(objects):
                mask = scope_masks[scope] & group_mask
                count = int(mask.sum())
                if count == 0:
                    continue
                row: dict[str, Any] = {
                    "scenario_id": scenario.id,
                    "scenario_type": scenario.kind,
                    "scope": scope,
                    "group_level": group_level,
                    "group_name": group_name,
                    "n_boxes": count,
                    "n_images": len(set(image_uids[mask].tolist())),
                    "scale_factor": scenario.scale_factor,
                    "tile_size": scenario.tile_size,
                    "input_size": scenario.input_size,
                    "crop_resolution": scenario.crop_resolution,
                    "context_scale": scenario.context_scale,
                    "padding_fraction_mean": float(values.padding_fraction[mask].mean()),
                    "bbox_visible_fraction_mean": float(
                        values.bbox_visible_fraction[mask].mean()
                    ),
                }
                row.update(_percentile_columns("native_short_px", native_short[mask], percentiles))
                row.update(
                    _percentile_columns("resized_short_px", values.resized_short[mask], percentiles)
                )
                row.update(
                    _percentile_columns(
                        "allocation_gain", values.allocation_gain[mask], percentiles
                    )
                )
                row.update(
                    _percentile_columns(
                        "padding_fraction", values.padding_fraction[mask], percentiles
                    )
                )
                row.update(
                    _percentile_columns(
                        "bbox_visible_fraction",
                        values.bbox_visible_fraction[mask],
                        percentiles,
                    )
                )
                for representation in representations:
                    grid_values = values.grid_short[representation.id][mask]
                    row.update(
                        _percentile_columns(
                            f"{representation.id}_short_span", grid_values, percentiles
                        )
                    )
                    row[f"{representation.id}_lt2_count"] = int(
                        (grid_values < thresholds[0]).sum()
                    )
                    row[f"{representation.id}_lt2_pct"] = float(
                        100.0 * (grid_values < thresholds[0]).mean()
                    )
                    row[f"{representation.id}_lt4_count"] = int(
                        (grid_values < thresholds[1]).sum()
                    )
                    row[f"{representation.id}_lt4_pct"] = float(
                        100.0 * (grid_values < thresholds[1]).mean()
                    )
                    row[f"{representation.id}_lt8_count"] = int(
                        (grid_values < thresholds[2]).sum()
                    )
                    row[f"{representation.id}_lt8_pct"] = float(
                        100.0 * (grid_values < thresholds[2]).mean()
                    )
                    row[f"{representation.id}_ge8_count"] = int(
                        (grid_values >= thresholds[2]).sum()
                    )
                    row[f"{representation.id}_ge8_pct"] = float(
                        100.0 * (grid_values >= thresholds[2]).mean()
                    )
                rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            formatted = {
                key: _format_float(value) if isinstance(value, (float, np.floating)) else value
                for key, value in row.items()
            }
            writer.writerow(formatted)


def write_object_details(
    path: Path,
    objects: Sequence[ObjectBox],
    scenarios: Sequence[Scenario],
    values_by_scenario: Mapping[str, ScenarioValues],
    representations: Sequence[Representation],
    thresholds: tuple[float, float, float],
) -> None:
    """流式写出 object x scenario 明细，避免在内存中构造宽表副本。"""

    base_fields = [
        "annotation_uid",
        "image_uid",
        "category_id",
        "category_name",
        "macro_category",
        "issue_flags",
        "source_edge_risk",
        "image_width",
        "image_height",
        "native_width_px",
        "native_height_px",
        "native_short_px",
        "native_long_px",
        "aspect_ratio_wh",
        "scenario_id",
        "scenario_type",
        "tile_size",
        "input_size",
        "scale_factor",
        "crop_resolution",
        "context_scale",
        "crop_nominal_side_px",
        "padding_fraction",
        "bbox_visible_fraction",
        "resized_width_px",
        "resized_height_px",
        "resized_short_px",
        "resized_long_px",
        "allocation_gain",
    ]
    grid_fields: list[str] = []
    for item in representations:
        grid_fields.extend(
            [
                f"{item.id}_width_span",
                f"{item.id}_height_span",
                f"{item.id}_short_span",
                f"{item.id}_area",
                f"{item.id}_risk_bin",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=base_fields + grid_fields)
        writer.writeheader()
        for scenario in scenarios:
            values = values_by_scenario[scenario.id]
            for index, obj in enumerate(objects):
                row: dict[str, Any] = {
                    "annotation_uid": obj.annotation_uid,
                    "image_uid": obj.image_uid,
                    "category_id": obj.category_id,
                    "category_name": obj.category_name,
                    "macro_category": obj.macro_category,
                    "issue_flags": obj.issue_flags,
                    "source_edge_risk": obj.source_edge_risk,
                    "image_width": obj.image_width,
                    "image_height": obj.image_height,
                    "native_width_px": obj.width,
                    "native_height_px": obj.height,
                    "native_short_px": obj.native_short,
                    "native_long_px": obj.native_long,
                    "aspect_ratio_wh": obj.width / obj.height,
                    "scenario_id": scenario.id,
                    "scenario_type": scenario.kind,
                    "tile_size": scenario.tile_size,
                    "input_size": scenario.input_size,
                    "scale_factor": scenario.scale_factor,
                    "crop_resolution": scenario.crop_resolution,
                    "context_scale": scenario.context_scale,
                    "crop_nominal_side_px": values.crop_nominal_side[index],
                    "padding_fraction": values.padding_fraction[index],
                    "bbox_visible_fraction": values.bbox_visible_fraction[index],
                    "resized_width_px": values.resized_width[index],
                    "resized_height_px": values.resized_height[index],
                    "resized_short_px": values.resized_short[index],
                    "resized_long_px": values.resized_long[index],
                    "allocation_gain": values.allocation_gain[index],
                }
                for representation in representations:
                    rep_id = representation.id
                    short_span = values.grid_short[rep_id][index]
                    if short_span < thresholds[0]:
                        risk_bin = 0
                    elif short_span < thresholds[1]:
                        risk_bin = 1
                    elif short_span < thresholds[2]:
                        risk_bin = 2
                    else:
                        risk_bin = 3
                    row.update(
                        {
                            f"{rep_id}_width_span": values.grid_width[rep_id][index],
                            f"{rep_id}_height_span": values.grid_height[rep_id][index],
                            f"{rep_id}_short_span": values.grid_short[rep_id][index],
                            f"{rep_id}_area": values.grid_area[rep_id][index],
                            f"{rep_id}_risk_bin": risk_bin,
                        }
                    )
                writer.writerow(
                    {
                        key: _format_float(value)
                        if isinstance(value, (float, np.floating))
                        else value
                        for key, value in row.items()
                    }
                )


def build_scenario_comparison(
    summary_rows: Sequence[Mapping[str, Any]], primary_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """构造不强行排名的 Pareto 对照表，并补充三大类等权平均。"""

    selected: list[dict[str, Any]] = []
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for source in summary_rows:
        if source["scope"] != "all" or source["group_level"] not in {"overall", "macro"}:
            continue
        row = {
            "scenario_id": source["scenario_id"],
            "scenario_type": source["scenario_type"],
            "group_name": source["group_name"],
            "n_boxes": source["n_boxes"],
            "scale_factor": source["scale_factor"],
            "tile_size": source["tile_size"],
            "input_size": source["input_size"],
            "crop_resolution": source["crop_resolution"],
            "context_scale": source["context_scale"],
            "native_short_px_p50": source["native_short_px_p50"],
            "resized_short_px_p50": source["resized_short_px_p50"],
            "allocation_gain_p50": source["allocation_gain_p50"],
            "padding_fraction_mean": source["padding_fraction_mean"],
        }
        for rep_id in primary_ids:
            for suffix in ("short_span_p05", "short_span_p50", "lt2_pct", "lt4_pct", "lt8_pct", "ge8_pct"):
                row[f"{rep_id}_{suffix}"] = source[f"{rep_id}_{suffix}"]
        selected.append(row)
        if source["group_level"] == "macro":
            by_scenario.setdefault(str(source["scenario_id"]), []).append(row)

    for scenario_id, macro_rows in by_scenario.items():
        if len(macro_rows) != len(COARSE_NAMES):
            continue
        template = macro_rows[0]
        average: dict[str, Any] = {
            "scenario_id": scenario_id,
            "scenario_type": template["scenario_type"],
            "group_name": "macro_average_equal_weight",
            "n_boxes": sum(int(item["n_boxes"]) for item in macro_rows),
            "scale_factor": template["scale_factor"],
            "tile_size": template["tile_size"],
            "input_size": template["input_size"],
            "crop_resolution": template["crop_resolution"],
            "context_scale": template["context_scale"],
        }
        numeric_keys = [
            key
            for key in template
            if key not in average and key not in {"scenario_id", "scenario_type", "group_name"}
        ]
        for key in numeric_keys:
            average[key] = float(np.mean([float(item[key]) for item in macro_rows]))
        selected.append(average)
    group_order = {"all_objects": 0, "ship": 1, "aircraft": 2, "vehicle": 3, "macro_average_equal_weight": 4}
    return sorted(selected, key=lambda row: (str(row["scenario_id"]), group_order[row["group_name"]]))


def _summary_lookup(
    summary_rows: Sequence[Mapping[str, Any]], scenario_id: str, group_name: str
) -> Mapping[str, Any]:
    for row in summary_rows:
        if (
            row["scenario_id"] == scenario_id
            and row["scope"] == "all"
            and row["group_name"] == group_name
        ):
            return row
    raise KeyError(f"未找到汇总行: {scenario_id}/{group_name}")


def derive_recommendations(
    config: Mapping[str, Any],
    scenarios: Sequence[Scenario],
    summary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """按预注册阈值生成保守的 probe 建议，不输出模型准入结论。"""

    rules = config.get("decision_heuristics", {})
    primary = config["primary_representations"]
    rep_id = str(primary["discriminative"])
    reference_tile = str(rules["reference_tile_scenario"])
    reference_crop = str(rules["reference_crop_scenario"])
    high_crop = str(rules["high_resolution_crop_scenario"])
    wide_crop = str(rules["wide_context_crop_scenario"])
    reduction_threshold = float(rules["crop_preferred_lt4_reduction_pp"])

    crop_deltas: dict[str, float] = {}
    significant_macros: list[str] = []
    for macro in COARSE_NAMES:
        tile_row = _summary_lookup(summary_rows, reference_tile, macro)
        crop_row = _summary_lookup(summary_rows, reference_crop, macro)
        tile_risk = float(tile_row[f"{rep_id}_lt4_pct"])
        crop_risk = float(crop_row[f"{rep_id}_lt4_pct"])
        crop_deltas[macro] = tile_risk - crop_risk
        if tile_risk >= reduction_threshold:
            significant_macros.append(macro)
    macros_meeting_reduction = [
        macro for macro, value in crop_deltas.items() if value >= reduction_threshold
    ]
    crop_preferred = bool(significant_macros) and bool(macros_meeting_reduction) and all(
        value >= -1e-9 for value in crop_deltas.values()
    )

    high_resolution_gain = {
        macro: float(_summary_lookup(summary_rows, high_crop, macro)[f"{rep_id}_ge8_pct"])
        - float(_summary_lookup(summary_rows, reference_crop, macro)[f"{rep_id}_ge8_pct"])
        for macro in COARSE_NAMES
    }
    high_resolution_probe = any(
        value >= float(rules["high_resolution_ge8_gain_pp"])
        for value in high_resolution_gain.values()
    )
    context_risk_increase = {
        macro: float(_summary_lookup(summary_rows, wide_crop, macro)[f"{rep_id}_lt4_pct"])
        - float(_summary_lookup(summary_rows, reference_crop, macro)[f"{rep_id}_lt4_pct"])
        for macro in COARSE_NAMES
    }

    risky_tiles: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario.kind != "tile":
            continue
        vehicle = _summary_lookup(summary_rows, scenario.id, "vehicle")
        lt2 = float(vehicle[f"{rep_id}_lt2_pct"])
        lt4 = float(vehicle[f"{rep_id}_lt4_pct"])
        if lt2 >= float(rules["risky_tile_vehicle_lt2_pct"]) or lt4 >= float(
            rules["risky_tile_vehicle_lt4_pct"]
        ):
            risky_tiles.append(
                {
                    "scenario_id": scenario.id,
                    "vehicle_lt2_pct": round(lt2, 6),
                    "vehicle_lt4_pct": round(lt4, 6),
                }
            )

    scale_groups: dict[str, list[str]] = {}
    for scenario in scenarios:
        if scenario.kind == "tile":
            key = f"{scenario.scale_factor:.12g}"
            scale_groups.setdefault(key, []).append(scenario.id)
    equivalent_groups = [
        {"scale_factor": float(scale), "scenarios": ids}
        for scale, ids in scale_groups.items()
        if len(ids) > 1
    ]
    return {
        "evidence_type": "derived_geometric_proxy",
        "teacher_location": {
            "recommendation": "prioritize_object_crop_probe"
            if crop_preferred
            else "no_geometric_preference_established",
            "reference_tile": reference_tile,
            "reference_crop": reference_crop,
            "primary_representation": rep_id,
            "lt4_reduction_percentage_points": {
                key: round(value, 6) for key, value in crop_deltas.items()
            },
            "macros_with_significant_tile_bottleneck": significant_macros,
            "macros_meeting_reduction_threshold": macros_meeting_reduction,
            "decision_rule": (
                "at least one macro improves by the preregistered threshold, "
                "no macro worsens, and the reference tile has a significant bottleneck"
            ),
            "requires_accuracy_validation": True,
        },
        "crop_resolution": {
            "recommendation": "probe_224_and_336"
            if high_resolution_probe
            else "probe_224_first",
            "ge8_gain_336_vs_224_percentage_points": {
                key: round(value, 6) for key, value in high_resolution_gain.items()
            },
            "selection_requires_accuracy_and_cost_validation": True,
        },
        "crop_context": {
            "recommendation": "probe_tight_and_1p25_defer_1p5",
            "lt4_increase_1p5_vs_1p25_percentage_points": {
                key: round(value, 6) for key, value in context_risk_increase.items()
            },
            "reason": "geometry cannot measure semantic value of context",
        },
        "tile_scenarios_flagged_for_vehicle_grid_risk": risky_tiles,
        "scale_equivalent_tile_groups": equivalent_groups,
        "non_claims": [
            "grid span is not accuracy",
            "crop upsampling does not recover native detail",
            "tile truncation and overlap are not modeled",
            "future 10K imagery native scale is assumed transferable but unverified",
        ],
    }


def _write_svg_bar_chart(
    path: Path,
    title: str,
    labels: Sequence[str],
    series: Mapping[str, Sequence[float]],
    y_label: str,
) -> None:
    """无绘图依赖的简单 SVG 分组柱图，图内只用 ASCII 避免 CI 字体问题。"""

    width = max(900, 120 + len(labels) * 110)
    height = 520
    left, top, bottom, right = 75, 55, 155, 30
    chart_width, chart_height = width - left - right, height - top - bottom
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f59e0b"]
    max_value = max([100.0, *(value for values in series.values() for value in values)])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{escape(title)}</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + chart_height * (1 - tick / max_value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick}</text>')
    group_width = chart_width / max(1, len(labels))
    bar_width = group_width * 0.72 / max(1, len(series))
    for series_index, (name, values) in enumerate(series.items()):
        color = colors[series_index % len(colors)]
        for index, value in enumerate(values):
            x = left + index * group_width + group_width * 0.14 + series_index * bar_width
            bar_height = chart_height * float(value) / max_value
            y = top + chart_height - bar_height
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width-2:.2f}" height="{bar_height:.2f}" fill="{color}"/>')
    for index, label in enumerate(labels):
        x = left + (index + 0.5) * group_width
        parts.append(f'<text x="{x:.2f}" y="{top+chart_height+18}" transform="rotate(38 {x:.2f} {top+chart_height+18})" text-anchor="start" font-family="sans-serif" font-size="10">{escape(label)}</text>')
    legend_x = left
    for series_index, name in enumerate(series):
        x = legend_x + series_index * 140
        color = colors[series_index % len(colors)]
        parts.append(f'<rect x="{x}" y="{height-24}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{x+18}" y="{height-13}" font-family="sans-serif" font-size="11">{escape(name)}</text>')
    parts.append(f'<text x="18" y="{top+chart_height/2}" transform="rotate(-90 18 {top+chart_height/2})" text-anchor="middle" font-family="sans-serif" font-size="12">{escape(y_label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_figures(
    output_dir: Path,
    scenarios: Sequence[Scenario],
    summary_rows: Sequence[Mapping[str, Any]],
    primary: Mapping[str, str],
) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    tile_scenarios = [item for item in scenarios if item.kind == "tile"]
    crop_scenarios = [item for item in scenarios if item.kind == "crop"]
    generated: list[str] = []
    for filename, rep_id in (
        ("tile_vit14_lt4_risk.svg", str(primary["discriminative"])),
        ("tile_diffusion_s8_lt4_risk.svg", str(primary["diffusion"])),
    ):
        series = {
            macro: [
                float(_summary_lookup(summary_rows, scenario.id, macro)[f"{rep_id}_lt4_pct"])
                for scenario in tile_scenarios
            ]
            for macro in COARSE_NAMES
        }
        _write_svg_bar_chart(
            figure_dir / filename,
            f"Tile full-object grid risk: {rep_id}",
            [item.id for item in tile_scenarios],
            series,
            "boxes with short span <4 (%)",
        )
        generated.append(f"figures/{filename}")

    rep_id = str(primary["discriminative"])
    crop_series = {
        macro: [
            float(_summary_lookup(summary_rows, scenario.id, macro)[f"{rep_id}_ge8_pct"])
            for scenario in crop_scenarios
        ]
        for macro in COARSE_NAMES
    }
    filename = "crop_vit14_ge8_allocation.svg"
    _write_svg_bar_chart(
        figure_dir / filename,
        f"Object-crop relatively well-sampled proxy: {rep_id}",
        [item.id for item in crop_scenarios],
        crop_series,
        "boxes with short span >=8 (%)",
    )
    generated.append(f"figures/{filename}")

    padding_series = {
        macro: [
            100.0
            * float(_summary_lookup(summary_rows, scenario.id, macro)["padding_fraction_mean"])
            for scenario in crop_scenarios
        ]
        for macro in COARSE_NAMES
    }
    filename = "crop_padding_fraction.svg"
    _write_svg_bar_chart(
        figure_dir / filename,
        "Object-crop source-boundary padding",
        [item.id for item in crop_scenarios],
        padding_series,
        "mean padding area (%)",
    )
    generated.append(f"figures/{filename}")
    return generated


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    scenarios: Sequence[Scenario],
    summary_rows: Sequence[Mapping[str, Any]],
    recommendations: Mapping[str, Any],
) -> str:
    """生成一份强调证据边界、公式口径与后续准入条件的中文报告。"""

    primary = config["primary_representations"]
    vit = str(primary["discriminative"])
    diffusion = str(primary["diffusion"])
    rules = config["decision_heuristics"]
    tile_ref = str(rules["reference_tile_scenario"])
    crop_ref = str(rules["reference_crop_scenario"])
    high_crop = str(rules["high_resolution_crop_scenario"])

    raw_rows = []
    for macro in COARSE_NAMES:
        row = _summary_lookup(summary_rows, tile_ref, macro)
        raw_rows.append(
            (
                macro,
                int(row["n_boxes"]),
                f"{float(row['native_short_px_p05']):.1f}",
                f"{float(row['native_short_px_p50']):.1f}",
                f"{float(row['native_short_px_p95']):.1f}",
            )
        )
    comparison_rows = []
    diffusion_rows = []
    for macro in COARSE_NAMES:
        tile = _summary_lookup(summary_rows, tile_ref, macro)
        crop = _summary_lookup(summary_rows, crop_ref, macro)
        crop_hi = _summary_lookup(summary_rows, high_crop, macro)
        comparison_rows.append(
            (
                macro,
                f"{float(tile[f'{vit}_short_span_p50']):.2f}",
                f"{float(tile[f'{vit}_lt4_pct']):.1f}%",
                f"{float(crop[f'{vit}_short_span_p50']):.2f}",
                f"{float(crop[f'{vit}_lt4_pct']):.1f}%",
                f"{float(crop_hi[f'{vit}_ge8_pct']):.1f}%",
            )
        )
        diffusion_rows.append(
            (
                macro,
                f"{float(tile[f'{diffusion}_short_span_p05']):.2f}",
                f"{float(tile[f'{diffusion}_short_span_p50']):.2f}",
                f"{float(tile[f'{diffusion}_lt4_pct']):.1f}%",
                f"{float(crop[f'{diffusion}_short_span_p05']):.2f}",
                f"{float(crop[f'{diffusion}_short_span_p50']):.2f}",
                f"{float(crop[f'{diffusion}_lt4_pct']):.1f}%",
            )
        )

    context_rows = []
    for scenario_id in ("crop_224_ctx1", "crop_224_ctx1p25", "crop_224_ctx1p5"):
        for macro in COARSE_NAMES:
            row = _summary_lookup(summary_rows, scenario_id, macro)
            context_rows.append(
                (
                    scenario_id,
                    macro,
                    f"{float(row[f'{vit}_short_span_p50']):.2f}",
                    f"{float(row[f'{vit}_lt4_pct']):.1f}%",
                    f"{float(row[f'{vit}_ge8_pct']):.1f}%",
                    f"{100 * float(row['padding_fraction_mean']):.1f}%",
                )
            )

    edge_rows = []
    for scope in (
        "all",
        "clean_in_bounds",
        "source_edge_risk",
        "touches_edge_in_bounds",
        "out_of_bounds",
    ):
        row = next(
            (
                item
                for item in summary_rows
                if item["scenario_id"] == crop_ref
                and item["scope"] == scope
                and item["group_name"] == "all_objects"
            ),
            None,
        )
        if row is None:
            continue
        edge_rows.append(
            (
                scope,
                int(row["n_boxes"]),
                f"{float(row['bbox_visible_fraction_p05']):.6f}",
                f"{float(row['bbox_visible_fraction_mean']):.6f}",
                f"{100 * float(row['padding_fraction_mean']):.1f}%",
                f"{100 * float(row['padding_fraction_p95']):.1f}%",
            )
        )

    fine_rows = []
    available_fine_names = {
        str(row["group_name"])
        for row in summary_rows
        if row["scenario_id"] == tile_ref
        and row["scope"] == "all"
        and row["group_level"] == "fine"
    }
    for name in FINE_NAMES:
        if name not in available_fine_names:
            continue
        tile = _summary_lookup(summary_rows, tile_ref, name)
        crop = _summary_lookup(summary_rows, crop_ref, name)
        crop_hi = _summary_lookup(summary_rows, high_crop, name)
        fine_rows.append(
            (
                name,
                int(tile["n_boxes"]),
                f"{float(tile['native_short_px_p50']):.1f}",
                f"{float(tile[f'{vit}_lt4_pct']):.1f}%",
                f"{float(crop[f'{vit}_lt4_pct']):.1f}%",
                f"{float(crop_hi[f'{vit}_ge8_pct']):.1f}%",
            )
        )
    fine_rows.sort(key=lambda row: (-float(str(row[3]).rstrip("%")), int(row[1]), row[0]))
    low_support = sorted(
        ((str(row[0]), int(row[1])) for row in fine_rows if int(row[1]) < 50),
        key=lambda item: (item[1], item[0]),
    )
    low_support_text = "、".join(f"{name} {count} 例" for name, count in low_support) or "无"
    macro_counts = {str(row[0]): int(row[1]) for row in raw_rows}

    counts = metadata["validation"]
    rec_teacher = recommendations["teacher_location"]
    rec_resolution = recommendations["crop_resolution"]
    risky_tiles = recommendations["tile_scenarios_flagged_for_vehicle_grid_risk"]
    equivalent_groups = recommendations["scale_equivalent_tile_groups"]
    equivalent_text = "; ".join(
        f"s={item['scale_factor']:g}: {', '.join(item['scenarios'])}"
        for item in equivalent_groups
    ) or "无"
    risky_text = "、".join(item["scenario_id"] for item in risky_tiles) or "无"

    return f"""# X-CROP-00 对象尺度与表征网格几何可见性报告

## 1. 结论摘要

本实验是“几何表征分配/架构瓶颈审计”，不是模型精度实验。全量统计了 {counts['n_boxes']:,} 个 bbox，其中 {counts['n_clean_in_bounds']:,} 个无审计标记，{counts['n_source_edge_risk']:,} 个有原图贴边/越界风险；后者未被删除。

按预注册几何规则，当前建议是：

- 教师输入：`{rec_teacher['recommendation']}`，即优先做对象 crop 实证，但不宣称整 tile 教师无效。
- crop 分辨率：`{rec_resolution['recommendation']}`，224 作成本基线，336 作一档高分辨率对照；最终取舍必须看 P0-3/P0-4 精度、显存和吞吐。
- crop 上下文：首轮保留 tight 和 1.25x，1.5x 先延后；本实验只能量化主体稀释和 padding，不能量化语义上下文的价值。
- 按车辆 `{vit}` 短边网格跨度规则标记的高风险 tile 场景：{risky_text}。这些只是后续实验的重点对照，不是直接禁用。

## 2. 证据范围与数据校验

直接观测证据是当前训练集审计表中的 bbox 像素尺度、图像尺寸、类别和边界标记。派生证据是按固定 resize/crop 公式计算的输入像素跨度和 feature-grid 跨度。“未来 10K 测试图与当前数据的原生目标像素尺度近似可迁移”只是待验证假设。

{_markdown_table(['macro', 'n', 'native short P5', 'P50', 'P95'], raw_rows)}

数据校验已确认：annotation_uid 唯一，所有 image_uid 成功连接，25 个细类与 `rsdet.data.xh_dataset` 的大类映射一致，宽高均为有限正数，且像素宽高与归一化标注反算一致。

## 3. 计算口径

tile 场景假设目标完整位于固定 tile canvas：`s=I/T`，`w_in=s*w`，`h_in=s*h`。crop 场景使用不缩短的名义方窗：`L=context*max(w,h)`，`w_out=R*w/L`，`h_out=R*h/L`，超出原图时 padding 到 L 后直接 resize。对 step/patch 为 r 的表征，主指标是短边连续跨度 `min(w_out/r,h_out/r)`，而不是易被细长 HBB 高估的面积。

短边跨度预注册分档为 `<2`、`[2,4)`、`[4,8)`、`>=8`。它们的含义分别是 severe-grid-bottleneck、constrained、moderate 和 relatively-well-sampled 几何代理，不是能力定律。`{diffusion}` 与 `{vit}` 的数字不能横向解释成相同表征能力。

## 4. 主要结果

下表对比参考 tile `{tile_ref}`、对象 crop `{crop_ref}` 和高分辨率 crop `{high_crop}`。“放大”只增加网格分配，不增加原图里已经存在的信息量。

{_markdown_table(['macro', 'tile median span', 'tile <4', 'crop224 median span', 'crop224 <4', 'crop336 >=8'], comparison_rows)}

对候选扩散 `/8` 网格，可见性数值如下。它只代表一个候选潜空间分配；具体教师不一定使用该步长。

{_markdown_table(['macro', 'tile P5', 'tile P50', 'tile <4', 'crop224 P5', 'crop224 P50', 'crop224 <4'], diffusion_rows)}

### 4.1 224 输入的上下文稀释

{_markdown_table(['scenario', 'macro', 'vit14 P50', '<4', '>=8', 'mean padding'], context_rows)}

tight crop 在纯几何指标上必然最好，因为主体占比最大；这不能证明它的分类精度最高。1.25x 保留为语义上下文对照；1.5x 使舰船 `{vit}<4` 风险相对 1.25x 再增加 {recommendations['crop_context']['lt4_increase_1p5_vs_1p25_percentage_points']['ship']:.1f} 个百分点，所以不进首轮。

### 4.2 细类分层

{_markdown_table(['fine class', 'n', 'native short P50', 'tile vit14 <4', 'crop224 <4', 'crop336 >=8'], fine_rows)}

表中 `n` 必须与百分比同时阅读。当前低于 50 例的细类为：{low_support_text}，其分位数和风险比例稳定性明显低于高频类。overall 结果也会被 {macro_counts['aircraft']:,} 个飞机框主导，因此本实验的决策一律检查三大类和细类，不只看实例加权总体。

### 4.3 原图边界风险

{_markdown_table(['scope', 'n', 'bbox visible P5', 'bbox visible mean', 'mean padding', 'padding P95'], edge_rows)}

同一缩放因子的 tile 在 P0-1 中完全等价：{equivalent_text}。tile 尺寸对上下文、tile 数和截断概率仍有影响，但这些不属于本指标。

## 5. 边界对象与限制

- {counts['n_source_edge_risk']:,} 个 `source_edge_risk` 对象单独出表；其中原图外已缺失的内容无法由 padding 恢复。
- crop 的 `allocation_gain` 是表征网格预算增益，不是超分或新细节。
- HBB 对旋转飞机/舰船含有背景，所以 bbox 跨度仍是结构可见性的上界代理。
- feature step 不等于感受野；具体扩散教师确定后，必须用它的真实 VAE factor、patchification 和抽取层更新配置。
- P0-1 不回答扩散是否优于 DINOv2、最优时间步/层、最终 tile 和官方 Recall/FDR。

## 6. 后续执行准入

1. P0-3 用 GT crop 比较 tight/1.25x 和 224/336，同时报精度、吞吐和显存。
2. P0-4 对比普通 crop classifier、DINOv2 教师和扩散特征教师；本报告只用来固定它们的输入候选。
3. 最终 tile 需联合独立的截断仿真、真实检测指标以及 10K/RTX 3090 端到端时延，不得由本表单独选定。
4. 当教师架构确定时，更新 `representations` 并重跑本实验；不直接沿用通用 `/8` 假设。

## 7. 复现信息

- analysis version: `{metadata['analysis_version']}`
- generated at: `{metadata['generated_at']}`
- git commit: `{metadata['git']['commit']}`
- bbox SHA-256: `{metadata['inputs']['bbox_statistics']['sha256']}`
- image stats SHA-256: `{metadata['inputs']['image_stats']['sha256']}`
- config SHA-256: `{metadata['inputs']['config']['sha256']}`
- analysis module SHA-256: `{metadata['inputs']['implementation']['analysis_module']['sha256']}`
- CLI SHA-256: `{metadata['inputs']['implementation']['cli_script']['sha256']}`
- 完整数值产物：`object_visibility.csv`、`visibility_summary.csv`、`scenario_comparison.csv`、`recommendations.yaml`、`meta.json`
"""


def run_visibility_analysis(
    *,
    bbox_path: str | Path,
    image_stats_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    command: Sequence[str] | None = None,
    repo_root: str | Path | None = None,
    write_charts: bool = True,
) -> dict[str, Any]:
    """运行全量分析并写出冻结配置、明细、汇总、图表、建议和报告。"""

    bbox_path = Path(bbox_path).expanduser().resolve()
    image_stats_path = Path(image_stats_path).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_analysis_config(config_path)
    scenarios, representations, thresholds = parse_analysis_config(config)
    objects = load_object_boxes(bbox_path, image_stats_path)
    values_by_scenario = {
        scenario.id: compute_scenario_values(objects, scenario, representations)
        for scenario in scenarios
    }
    summary_config = config.get("summary", {})
    percentiles = [float(item) for item in summary_config.get("percentiles", [5, 25, 50, 75, 95])]
    if any(item < 0 or item > 100 for item in percentiles) or sorted(percentiles) != percentiles:
        raise ValueError("summary percentiles 必须在 0--100 内且递增")
    scopes = [str(item) for item in summary_config.get("scopes", ["all"])]
    summary_rows = summarize_scenarios(
        objects,
        scenarios,
        values_by_scenario,
        representations,
        thresholds,
        percentiles,
        scopes,
    )
    primary_ids = [str(value) for value in config["primary_representations"].values()]
    comparison_rows = build_scenario_comparison(summary_rows, primary_ids)
    recommendations = derive_recommendations(config, scenarios, summary_rows)

    write_object_details(
        output_dir / "object_visibility.csv",
        objects,
        scenarios,
        values_by_scenario,
        representations,
        thresholds,
    )
    summary_fields = list(summary_rows[0].keys())
    comparison_fields = list(comparison_rows[0].keys())
    _write_csv(output_dir / "visibility_summary.csv", summary_rows, summary_fields)
    _write_csv(output_dir / "scenario_comparison.csv", comparison_rows, comparison_fields)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    with (output_dir / "recommendations.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(recommendations, file, allow_unicode=True, sort_keys=False)

    figure_paths = (
        write_figures(
            output_dir,
            scenarios,
            summary_rows,
            config["primary_representations"],
        )
        if write_charts
        else []
    )
    issue_counts: dict[str, int] = {}
    for obj in objects:
        issue_counts[obj.issue_flags or "clean"] = issue_counts.get(obj.issue_flags or "clean", 0) + 1
    analysis_meta = config.get("analysis", {})
    repo_root_path = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
    implementation_inputs: dict[str, Any] = {
        "analysis_module": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        }
    }
    cli_path = repo_root_path / "scripts" / "analyze_object_visibility.py"
    if cli_path.is_file():
        implementation_inputs["cli_script"] = {
            "path": str(cli_path),
            "sha256": _sha256(cli_path),
        }
    metadata: dict[str, Any] = {
        "experiment_id": analysis_meta.get("experiment_id"),
        "analysis_version": analysis_meta.get("analysis_version"),
        "status": analysis_meta.get("status"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": list(command or ()),
        "git": _git_metadata(repo_root_path),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pyyaml": yaml.__version__,
        },
        "inputs": {
            "bbox_statistics": {"path": str(bbox_path), "sha256": _sha256(bbox_path)},
            "image_stats": {"path": str(image_stats_path), "sha256": _sha256(image_stats_path)},
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "implementation": implementation_inputs,
        },
        "validation": {
            "n_boxes": len(objects),
            "n_images_with_boxes": len({item.image_uid for item in objects}),
            "n_categories": len({item.category_id for item in objects}),
            "n_clean_in_bounds": sum(not item.source_edge_risk for item in objects),
            "n_source_edge_risk": sum(item.source_edge_risk for item in objects),
            "issue_flag_counts": issue_counts,
            "all_annotation_uids_unique": True,
            "all_image_uids_joined": True,
            "all_class_mappings_valid": True,
            "all_box_dimensions_finite_positive": True,
        },
        "method": {
            "semantics": "full_object_intrinsic_visibility",
            "tile_formula": "s=I/T; w_in=s*w; h_in=s*h",
            "crop_formula": "L=context*max(w,h); w_out=R*w/L; h_out=R*h/L",
            "crop_boundary_policy": "fixed nominal square with padding; direct square resize",
            "primary_metric": "continuous feature-grid short-axis span",
            "risk_bins": ["<2", "[2,4)", "[4,8)", ">=8"],
            "not_official_evaluation": True,
        },
        "scenario_count": len(scenarios),
        "detail_row_count": len(objects) * len(scenarios),
        "summary_row_count": len(summary_rows),
        "figures": figure_paths,
        "recommendations": recommendations,
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    report = build_report(metadata, config, scenarios, summary_rows, recommendations)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return metadata
