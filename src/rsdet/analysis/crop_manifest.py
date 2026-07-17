"""生成防泄漏的探索性对象 crop manifest。

本模块只生成来源、分组、划分和 crop 几何契约。除少量 QA 预览外，
不物化全量 crop，不修改原图，也不将审计阶段启发式 split 声称为正式划分。
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.sax.saxutils import escape

import numpy as np
import PIL
import yaml
from PIL import Image, ImageDraw

from rsdet.data.xh_dataset import COARSE_NAMES, FINE_NAMES, MODALITY, coarse_name

REQUIRED_IMAGE_COLUMNS = {
    "image_uid",
    "relative_path",
    "width",
    "height",
    "sha256",
    "estimated_group_uid",
    "group_basis",
    "group_confidence",
}
REQUIRED_SPLIT_COLUMNS = {
    "image_uid",
    "relative_path",
    "estimated_group_uid",
    "main_split",
    "fold3",
    "difficult_quality_subset",
    "edge_target_subset",
    "pure_background_subset",
    "speed_proxy_subset",
}
REQUIRED_DOMAIN_COLUMNS = {"image_uid", "domain_cluster", "estimated_group_uid"}
REQUIRED_BBOX_COLUMNS = {
    "annotation_uid",
    "image_uid",
    "category_id",
    "category_name",
    "macro_category",
    "x0",
    "y0",
    "x1",
    "y1",
    "width_px",
    "height_px",
    "width_norm",
    "height_norm",
    "issue_flags",
}
REQUIRED_NEAR_DUPLICATE_COLUMNS = {"image_uid_1", "image_uid_2", "status"}


@dataclass(frozen=True)
class ImageSource:
    image_uid: str
    relative_path: str
    width: float
    height: float
    sha256: str
    estimated_group_uid: str
    group_basis: str
    group_confidence: str
    original_main_split: str
    original_fold: int
    difficult_quality_subset: bool
    edge_target_subset: bool
    pure_background_subset: bool
    speed_proxy_subset: bool
    domain_cluster: int


@dataclass(frozen=True)
class ObjectAnnotation:
    annotation_uid: str
    image_uid: str
    category_id: int
    category_name: str
    macro_category: str
    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    height: float
    issue_flags: str


@dataclass(frozen=True)
class CropPolicy:
    id: str
    kind: str
    role: str
    context_scale: float
    fingerprint: str
    jitter: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ImageAssignment:
    image_uid: str
    leakage_group_id: str
    leakage_group_image_count: int
    estimated_group_count: int
    near_duplicate_edge_count: int
    main_split: str
    fold: int
    main_assignment_reason: str
    fold_assignment_reason: str
    main_split_changed: bool
    fold_changed: bool


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        root_first, root_second = self.find(first), self.find(second)
        if root_first == root_second:
            return
        if root_first > root_second:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        actual = set(reader.fieldnames or ())
        missing = sorted(required - actual)
        if missing:
            raise ValueError(f"{path} 缺少必需列: {', '.join(missing)}")
        return [dict(row) for row in reader]


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} 必须是数值，实际为 {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} 必须是有限数，实际为 {value!r}")
    return result


def _positive(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result <= 0:
        raise ValueError(f"{field} 必须大于 0，实际为 {value!r}")
    return result


def _parse_bool(value: Any, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} 必须是 True/False，实际为 {value!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(prefix: str, *parts: str, length: int = 16) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:length]}"


def _format_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return format(float(value), ".17g")


def _unit_fraction(value: float) -> float:
    """将比例截断到 [0,1]，并消除端点附近的浮点尾差。"""

    clipped = float(np.clip(value, 0.0, 1.0))
    if math.isclose(clipped, 0.0, abs_tol=1e-12):
        return 0.0
    if math.isclose(clipped, 1.0, abs_tol=1e-12):
        return 1.0
    return clipped


def load_crop_manifest_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError("crop manifest 配置顶层必须是映射")
    return config


def parse_crop_policies(config: Mapping[str, Any]) -> list[CropPolicy]:
    policies: list[CropPolicy] = []
    for raw in config.get("crop_policies", []):
        if not isinstance(raw, Mapping):
            raise TypeError("crop_policies 每项必须是映射")
        policy_id = str(raw.get("id", "")).strip()
        kind = str(raw.get("kind", "")).strip()
        role = str(raw.get("role", "")).strip()
        context_scale = _positive(raw.get("context_scale"), "context_scale")
        if kind not in {"gt_square", "jittered_square"}:
            raise ValueError(f"未知 crop policy kind: {kind}")
        jitter = raw.get("jitter")
        if kind == "jittered_square":
            if not isinstance(jitter, Mapping):
                raise ValueError(f"{policy_id} 缺少 jitter 配置")
            if jitter.get("algorithm") != "sha256_uniform_v1":
                raise ValueError("当前只支持 sha256_uniform_v1 jitter")
            shift = _finite(jitter.get("center_shift_fraction_max"), "center shift")
            if not 0 <= shift <= 0.5:
                raise ValueError("center_shift_fraction_max 必须在 [0,0.5]")
            for axis in ("width", "height"):
                lower = _positive(jitter.get(f"{axis}_scale_min"), f"{axis}_scale_min")
                upper = _positive(jitter.get(f"{axis}_scale_max"), f"{axis}_scale_max")
                if lower > upper:
                    raise ValueError(f"{axis} scale 下界不得大于上界")
        canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        policies.append(
            CropPolicy(
                id=policy_id,
                kind=kind,
                role=role,
                context_scale=context_scale,
                fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
                jitter=dict(jitter) if isinstance(jitter, Mapping) else None,
            )
        )
    ids = [item.id for item in policies]
    if not policies or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("crop policy ID 必须非空且唯一")
    required = {"tight", "context_1p25", "jitter_light"}
    if set(ids) != required:
        raise ValueError(f"首轮 crop policy 必须恰为 {sorted(required)}")
    return policies


def load_image_sources(
    image_stats_path: str | Path,
    split_candidates_path: str | Path,
    domain_assignments_path: str | Path,
) -> dict[str, ImageSource]:
    image_stats_path = Path(image_stats_path).resolve()
    split_candidates_path = Path(split_candidates_path).resolve()
    domain_assignments_path = Path(domain_assignments_path).resolve()
    image_rows = _read_csv(image_stats_path, REQUIRED_IMAGE_COLUMNS)
    split_rows = _read_csv(split_candidates_path, REQUIRED_SPLIT_COLUMNS)
    domain_rows = _read_csv(domain_assignments_path, REQUIRED_DOMAIN_COLUMNS)
    split_by_uid = {row["image_uid"]: row for row in split_rows}
    domain_by_uid = {row["image_uid"]: row for row in domain_rows}
    if len(split_by_uid) != len(split_rows) or len(domain_by_uid) != len(domain_rows):
        raise ValueError("split/domain 表中 image_uid 重复")

    sources: dict[str, ImageSource] = {}
    for line_number, row in enumerate(image_rows, start=2):
        uid = row["image_uid"].strip()
        if not uid or uid in sources:
            raise ValueError(f"{image_stats_path}:{line_number}: image_uid 为空或重复")
        if uid not in split_by_uid or uid not in domain_by_uid:
            raise ValueError(f"{uid} 未能完整连接 split/domain 表")
        split_row, domain_row = split_by_uid[uid], domain_by_uid[uid]
        if split_row["relative_path"] != row["relative_path"]:
            raise ValueError(f"{uid} 的 relative_path 在输入表中不一致")
        if not (
            split_row["estimated_group_uid"]
            == row["estimated_group_uid"]
            == domain_row["estimated_group_uid"]
        ):
            raise ValueError(f"{uid} 的 estimated_group_uid 在输入表中不一致")
        main_split = split_row["main_split"].strip()
        if main_split not in {"train", "val"}:
            raise ValueError(f"{uid} 的 main_split 非法: {main_split}")
        fold = int(split_row["fold3"])
        if fold not in {0, 1, 2}:
            raise ValueError(f"{uid} 的 fold3 非法: {fold}")
        checksum = row["sha256"].strip().lower()
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
            raise ValueError(f"{uid} 的 sha256 非法")
        relative_path = row["relative_path"].strip().replace("\\", "/")
        if not relative_path or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise ValueError(f"{uid} 的 relative_path 不安全: {relative_path}")
        sources[uid] = ImageSource(
            image_uid=uid,
            relative_path=relative_path,
            width=_positive(row["width"], f"{uid}:width"),
            height=_positive(row["height"], f"{uid}:height"),
            sha256=checksum,
            estimated_group_uid=row["estimated_group_uid"].strip(),
            group_basis=row["group_basis"].strip(),
            group_confidence=row["group_confidence"].strip(),
            original_main_split=main_split,
            original_fold=fold,
            difficult_quality_subset=_parse_bool(
                split_row["difficult_quality_subset"], f"{uid}:difficult_quality_subset"
            ),
            edge_target_subset=_parse_bool(
                split_row["edge_target_subset"], f"{uid}:edge_target_subset"
            ),
            pure_background_subset=_parse_bool(
                split_row["pure_background_subset"], f"{uid}:pure_background_subset"
            ),
            speed_proxy_subset=_parse_bool(
                split_row["speed_proxy_subset"], f"{uid}:speed_proxy_subset"
            ),
            domain_cluster=int(domain_row["domain_cluster"]),
        )
    if set(sources) != set(split_by_uid) or set(sources) != set(domain_by_uid):
        raise ValueError("image/split/domain 三张表的 image_uid 集合不一致")
    return sources


def load_annotations(
    bbox_statistics_path: str | Path, sources: Mapping[str, ImageSource]
) -> list[ObjectAnnotation]:
    path = Path(bbox_statistics_path).resolve()
    rows = _read_csv(path, REQUIRED_BBOX_COLUMNS)
    annotations: list[ObjectAnnotation] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        prefix = f"{path}:{line_number}"
        annotation_uid, image_uid = row["annotation_uid"].strip(), row["image_uid"].strip()
        if not annotation_uid or annotation_uid in seen:
            raise ValueError(f"{prefix}: annotation_uid 为空或重复")
        if image_uid not in sources:
            raise ValueError(f"{prefix}: image_uid 未连接: {image_uid}")
        try:
            category_id = int(row["category_id"])
        except ValueError as error:
            raise ValueError(f"{prefix}: category_id 非法") from error
        if not 0 <= category_id < len(FINE_NAMES):
            raise ValueError(f"{prefix}: category_id 超出 0--24")
        category_name, macro = row["category_name"].strip(), row["macro_category"].strip()
        if category_name != FINE_NAMES[category_id] or macro != coarse_name(category_id):
            raise ValueError(f"{prefix}: 类别映射冲突")
        source = sources[image_uid]
        width = _positive(row["width_px"], f"{prefix}:width_px")
        height = _positive(row["height_px"], f"{prefix}:height_px")
        width_norm = _positive(row["width_norm"], f"{prefix}:width_norm")
        height_norm = _positive(row["height_norm"], f"{prefix}:height_norm")
        if not math.isclose(width, width_norm * source.width, rel_tol=1e-5, abs_tol=0.02):
            raise ValueError(f"{prefix}: width_px 与归一化标注不一致")
        if not math.isclose(height, height_norm * source.height, rel_tol=1e-5, abs_tol=0.02):
            raise ValueError(f"{prefix}: height_px 与归一化标注不一致")
        x0, y0, x1, y1 = (_finite(row[key], f"{prefix}:{key}") for key in ("x0", "y0", "x1", "y1"))
        if not math.isclose(x1 - x0, width, rel_tol=1e-5, abs_tol=0.02):
            raise ValueError(f"{prefix}: x1-x0 与 width_px 不一致")
        if not math.isclose(y1 - y0, height, rel_tol=1e-5, abs_tol=0.02):
            raise ValueError(f"{prefix}: y1-y0 与 height_px 不一致")
        annotations.append(
            ObjectAnnotation(
                annotation_uid=annotation_uid,
                image_uid=image_uid,
                category_id=category_id,
                category_name=category_name,
                macro_category=macro,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                width=width,
                height=height,
                issue_flags=row["issue_flags"].strip(),
            )
        )
        seen.add(annotation_uid)
    if not annotations:
        raise ValueError("bbox 统计表为空")
    return sorted(annotations, key=lambda item: item.annotation_uid)


def load_near_duplicate_pairs(
    path: str | Path, sources: Mapping[str, ImageSource]
) -> list[tuple[str, str, str]]:
    path = Path(path).resolve()
    rows = _read_csv(path, REQUIRED_NEAR_DUPLICATE_COLUMNS)
    pairs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line_number, row in enumerate(rows, start=2):
        first, second = row["image_uid_1"].strip(), row["image_uid_2"].strip()
        if first not in sources or second not in sources or first == second:
            raise ValueError(f"{path}:{line_number}: 近重复图像对非法")
        pair = tuple(sorted((first, second)))
        if pair in seen:
            raise ValueError(f"{path}:{line_number}: 近重复图像对重复")
        seen.add(pair)
        pairs.append((pair[0], pair[1], row["status"].strip()))
    return sorted(pairs)


def _assignment_for_axis(
    components: Sequence[Sequence[ImageSource]],
    field: str,
    labels: Sequence[str],
    target_fractions: Mapping[str, float],
) -> tuple[dict[str, str], dict[str, str]]:
    """先最小化移动图像数，只在票数并列时以目标分布平衡破局。"""

    value_for = (
        (lambda source: source.original_main_split)
        if field == "main_split"
        else (lambda source: str(source.original_fold))
    )
    component_candidates: list[list[str]] = []
    counts_by_component: list[Counter[str]] = []
    for component in components:
        counts = Counter(value_for(source) for source in component)
        if not set(counts) <= set(labels):
            raise ValueError(f"{field} 包含未知标签")
        maximum = max(counts.values())
        component_candidates.append(sorted(label for label, count in counts.items() if count == maximum))
        counts_by_component.append(counts)

    combinations = math.prod(len(item) for item in component_candidates)
    if combinations > 100_000:
        raise ValueError(f"{field} 并列分配组合过多，拒绝隐式贪心分配")
    total = sum(len(component) for component in components)
    target = {label: total * float(target_fractions[label]) for label in labels}
    best_key: tuple[Any, ...] | None = None
    best_choice: tuple[str, ...] | None = None
    for choice in itertools.product(*component_candidates):
        final_counts = Counter()
        moved = 0
        for component, counts, selected in zip(components, counts_by_component, choice):
            final_counts[selected] += len(component)
            moved += len(component) - counts[selected]
        imbalance = sum((final_counts[label] - target[label]) ** 2 for label in labels)
        key = (moved, imbalance, choice)
        if best_key is None or key < best_key:
            best_key, best_choice = key, choice
    assert best_choice is not None

    assignments: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for component, counts, candidates, selected in zip(
        components, counts_by_component, component_candidates, best_choice
    ):
        if len(counts) == 1:
            reason = "unchanged_component"
        elif len(candidates) == 1:
            reason = "majority_repair"
        else:
            reason = "tie_balance_repair"
        for source in component:
            assignments[source.image_uid] = selected
            reasons[source.image_uid] = reason
    return assignments, reasons


def build_leakage_safe_assignments(
    sources: Mapping[str, ImageSource],
    near_duplicate_pairs: Sequence[tuple[str, str, str]],
    config: Mapping[str, Any],
) -> tuple[dict[str, ImageAssignment], list[dict[str, Any]]]:
    """合并启发式同源组和近重复连边，然后修复候选划分。"""

    union_find = UnionFind(sources)
    images_by_estimated_group: dict[str, list[str]] = defaultdict(list)
    for source in sources.values():
        images_by_estimated_group[source.estimated_group_uid].append(source.image_uid)
    for image_uids in images_by_estimated_group.values():
        first = image_uids[0]
        for image_uid in image_uids[1:]:
            union_find.union(first, image_uid)
    for first, second, _ in near_duplicate_pairs:
        union_find.union(first, second)

    component_uids: dict[str, list[str]] = defaultdict(list)
    for image_uid in sorted(sources):
        component_uids[union_find.find(image_uid)].append(image_uid)
    components = [
        [sources[image_uid] for image_uid in image_uids]
        for _, image_uids in sorted(component_uids.items(), key=lambda item: min(item[1]))
    ]
    leakage_config = config.get("leakage_control", {})
    if not isinstance(leakage_config, Mapping):
        raise TypeError("leakage_control 必须是映射")
    main_targets = leakage_config.get("main_split_target_fractions", {})
    if not isinstance(main_targets, Mapping) or set(main_targets) != {"train", "val"}:
        raise ValueError("main_split_target_fractions 必须定义 train/val")
    if not math.isclose(sum(float(value) for value in main_targets.values()), 1.0):
        raise ValueError("main split 目标比例之和必须为 1")
    fold_count = int(leakage_config.get("fold_count", 0))
    if fold_count != 3:
        raise ValueError("当前探索性 manifest 只支持 3 折")
    main_assignment, main_reason = _assignment_for_axis(
        components, "main_split", ("train", "val"), main_targets
    )
    fold_labels = tuple(str(index) for index in range(fold_count))
    fold_assignment, fold_reason = _assignment_for_axis(
        components,
        "fold",
        fold_labels,
        {label: 1.0 / fold_count for label in fold_labels},
    )

    edge_count_by_root: Counter[str] = Counter()
    for first, _, _ in near_duplicate_pairs:
        edge_count_by_root[union_find.find(first)] += 1
    assignments: dict[str, ImageAssignment] = {}
    component_rows: list[dict[str, Any]] = []
    for component in components:
        image_uids = sorted(source.image_uid for source in component)
        root = union_find.find(image_uids[0])
        leakage_group_id = _stable_hash("lgrp_", *image_uids, length=16)
        estimated_groups = {source.estimated_group_uid for source in component}
        assigned_main = main_assignment[image_uids[0]]
        assigned_fold = int(fold_assignment[image_uids[0]])
        original_main_counts = Counter(source.original_main_split for source in component)
        original_fold_counts = Counter(str(source.original_fold) for source in component)
        component_rows.append(
            {
                "leakage_group_id": leakage_group_id,
                "image_count": len(component),
                "estimated_group_count": len(estimated_groups),
                "near_duplicate_edge_count": edge_count_by_root[root],
                "contains_near_duplicate_union": edge_count_by_root[root] > 0,
                "original_main_split_counts": json.dumps(
                    dict(sorted(original_main_counts.items())), sort_keys=True
                ),
                "assigned_main_split": assigned_main,
                "main_assignment_reason": main_reason[image_uids[0]],
                "main_moved_images": sum(
                    source.original_main_split != assigned_main for source in component
                ),
                "original_fold_counts": json.dumps(
                    dict(sorted(original_fold_counts.items())), sort_keys=True
                ),
                "assigned_fold": assigned_fold,
                "fold_assignment_reason": fold_reason[image_uids[0]],
                "fold_moved_images": sum(
                    source.original_fold != assigned_fold for source in component
                ),
                "estimated_group_ids": ";".join(sorted(estimated_groups)),
                "image_uids": ";".join(image_uids),
            }
        )
        for source in component:
            assignments[source.image_uid] = ImageAssignment(
                image_uid=source.image_uid,
                leakage_group_id=leakage_group_id,
                leakage_group_image_count=len(component),
                estimated_group_count=len(estimated_groups),
                near_duplicate_edge_count=edge_count_by_root[root],
                main_split=assigned_main,
                fold=assigned_fold,
                main_assignment_reason=main_reason[source.image_uid],
                fold_assignment_reason=fold_reason[source.image_uid],
                main_split_changed=source.original_main_split != assigned_main,
                fold_changed=source.original_fold != assigned_fold,
            )
    return assignments, sorted(component_rows, key=lambda row: row["leakage_group_id"])


def _sha256_uniforms(key: str, count: int) -> tuple[int, list[float]]:
    if count > 4:
        raise ValueError("sha256_uniform_v1 一次最多生成 4 个均匀值")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    integers = [int.from_bytes(digest[index * 8 : (index + 1) * 8], "big") for index in range(count)]
    denominator = float(2**64)
    return integers[0], [(value + 0.5) / denominator for value in integers]


def compute_crop_geometry(
    annotation: ObjectAnnotation,
    source: ImageSource,
    policy: CropPolicy,
    manifest_version: str,
) -> dict[str, Any]:
    """计算名义 proposal 和方形 crop window，不对原图取整或裁剪。"""

    gt_center_x = (annotation.x0 + annotation.x1) / 2.0
    gt_center_y = (annotation.y0 + annotation.y1) / 2.0
    proposal_center_x, proposal_center_y = gt_center_x, gt_center_y
    proposal_width, proposal_height = annotation.width, annotation.height
    jitter_seed: int | None = None
    jitter_dx_fraction = jitter_dy_fraction = 0.0
    jitter_width_scale = jitter_height_scale = 1.0
    if policy.kind == "jittered_square":
        assert policy.jitter is not None
        jitter_key = (
            f"{policy.jitter['global_seed']}|{manifest_version}|"
            f"{annotation.annotation_uid}|{policy.id}|{policy.fingerprint}"
        )
        jitter_seed, uniforms = _sha256_uniforms(jitter_key, 4)
        shift = float(policy.jitter["center_shift_fraction_max"])
        jitter_dx_fraction = -shift + 2 * shift * uniforms[0]
        jitter_dy_fraction = -shift + 2 * shift * uniforms[1]
        jitter_width_scale = float(policy.jitter["width_scale_min"]) + (
            float(policy.jitter["width_scale_max"])
            - float(policy.jitter["width_scale_min"])
        ) * uniforms[2]
        jitter_height_scale = float(policy.jitter["height_scale_min"]) + (
            float(policy.jitter["height_scale_max"])
            - float(policy.jitter["height_scale_min"])
        ) * uniforms[3]
        proposal_center_x += jitter_dx_fraction * annotation.width
        proposal_center_y += jitter_dy_fraction * annotation.height
        proposal_width *= jitter_width_scale
        proposal_height *= jitter_height_scale

    proposal_x0 = proposal_center_x - proposal_width / 2.0
    proposal_y0 = proposal_center_y - proposal_height / 2.0
    proposal_x1 = proposal_center_x + proposal_width / 2.0
    proposal_y1 = proposal_center_y + proposal_height / 2.0
    crop_side = policy.context_scale * max(proposal_width, proposal_height)
    crop_x0 = proposal_center_x - crop_side / 2.0
    crop_y0 = proposal_center_y - crop_side / 2.0
    crop_x1 = proposal_center_x + crop_side / 2.0
    crop_y1 = proposal_center_y + crop_side / 2.0

    valid_width = max(0.0, min(crop_x1, source.width) - max(crop_x0, 0.0))
    valid_height = max(0.0, min(crop_y1, source.height) - max(crop_y0, 0.0))
    source_valid_fraction = _unit_fraction(valid_width * valid_height / (crop_side**2))
    gt_in_crop_width = max(0.0, min(annotation.x1, crop_x1) - max(annotation.x0, crop_x0))
    gt_in_crop_height = max(0.0, min(annotation.y1, crop_y1) - max(annotation.y0, crop_y0))
    gt_coverage_fraction = _unit_fraction(
        gt_in_crop_width * gt_in_crop_height / (annotation.width * annotation.height)
    )
    visible_gt_width = max(
        0.0, min(annotation.x1, source.width) - max(annotation.x0, 0.0)
    )
    visible_gt_height = max(
        0.0, min(annotation.y1, source.height) - max(annotation.y0, 0.0)
    )
    source_gt_visible_fraction = _unit_fraction(
        visible_gt_width * visible_gt_height / (annotation.width * annotation.height)
    )
    return {
        "proposal_x0": proposal_x0,
        "proposal_y0": proposal_y0,
        "proposal_x1": proposal_x1,
        "proposal_y1": proposal_y1,
        "crop_x0": crop_x0,
        "crop_y0": crop_y0,
        "crop_x1": crop_x1,
        "crop_y1": crop_y1,
        "crop_side_px": crop_side,
        "source_valid_fraction": source_valid_fraction,
        "padding_fraction": _unit_fraction(1.0 - source_valid_fraction),
        "gt_coverage_fraction": gt_coverage_fraction,
        "source_gt_visible_fraction": source_gt_visible_fraction,
        "jitter_seed_u64": jitter_seed,
        "jitter_dx_fraction": jitter_dx_fraction,
        "jitter_dy_fraction": jitter_dy_fraction,
        "jitter_width_scale": jitter_width_scale,
        "jitter_height_scale": jitter_height_scale,
    }


def build_manifest_rows(
    annotations: Sequence[ObjectAnnotation],
    sources: Mapping[str, ImageSource],
    assignments: Mapping[str, ImageAssignment],
    policies: Sequence[CropPolicy],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    manifest_config = config["manifest"]
    render_config = config["render_contract"]
    manifest_version = str(manifest_config["version"])
    schema_version = str(manifest_config["schema_version"])
    target_resolutions = ";".join(str(int(value)) for value in manifest_config["target_resolutions"])
    rows: list[dict[str, Any]] = []
    for annotation in annotations:
        source = sources[annotation.image_uid]
        assignment = assignments[annotation.image_uid]
        for policy in policies:
            geometry = compute_crop_geometry(annotation, source, policy, manifest_version)
            crop_id = _stable_hash(
                "crop_",
                manifest_version,
                annotation.annotation_uid,
                policy.id,
                policy.fingerprint,
                length=20,
            )
            row: dict[str, Any] = {
                "manifest_version": manifest_version,
                "schema_version": schema_version,
                "crop_id": crop_id,
                "annotation_uid": annotation.annotation_uid,
                "source_image_id": source.image_uid,
                "source_relative_path": source.relative_path,
                "source_width": source.width,
                "source_height": source.height,
                "source_checksum_sha256": source.sha256,
                "source_issue_flags": annotation.issue_flags,
                "source_edge_risk": bool(annotation.issue_flags),
                "original_main_split": source.original_main_split,
                "original_fold": source.original_fold,
                "main_split": assignment.main_split,
                "fold": assignment.fold,
                "main_split_changed": assignment.main_split_changed,
                "fold_changed": assignment.fold_changed,
                "main_assignment_reason": assignment.main_assignment_reason,
                "fold_assignment_reason": assignment.fold_assignment_reason,
                "group_id": assignment.leakage_group_id,
                "leakage_group_id": assignment.leakage_group_id,
                "leakage_group_image_count": assignment.leakage_group_image_count,
                "estimated_group_uid": source.estimated_group_uid,
                "estimated_group_count_in_leakage_group": assignment.estimated_group_count,
                "group_basis": source.group_basis,
                "group_confidence": source.group_confidence,
                "near_duplicate_edge_count": assignment.near_duplicate_edge_count,
                "domain_cluster": source.domain_cluster,
                "difficult_quality_subset": source.difficult_quality_subset,
                "edge_target_subset": source.edge_target_subset,
                "pure_background_subset": source.pure_background_subset,
                "speed_proxy_subset": source.speed_proxy_subset,
                "class_id": annotation.category_id,
                "class_name": annotation.category_name,
                "major_class": annotation.macro_category,
                "modality": MODALITY[annotation.macro_category],
                "gt_x0": annotation.x0,
                "gt_y0": annotation.y0,
                "gt_x1": annotation.x1,
                "gt_y1": annotation.y1,
                "gt_width": annotation.width,
                "gt_height": annotation.height,
                "gt_short_edge": min(annotation.width, annotation.height),
                "crop_policy": policy.id,
                "crop_policy_kind": policy.kind,
                "crop_policy_role": policy.role,
                "crop_policy_fingerprint": policy.fingerprint,
                "context_scale": policy.context_scale,
                "target_resolutions": target_resolutions,
                "color_mode": render_config["color_mode"],
                "outside_policy": render_config["square_window_outside_policy"],
                "resize_semantics": render_config["resize_semantics"],
                "coordinate_semantics": render_config["coordinate_semantics"],
            }
            row.update(geometry)
            rows.append(row)
    return rows


def build_image_assignment_rows(
    sources: Mapping[str, ImageSource], assignments: Mapping[str, ImageAssignment]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_uid in sorted(sources):
        source, assignment = sources[image_uid], assignments[image_uid]
        rows.append(
            {
                "image_uid": image_uid,
                "relative_path": source.relative_path,
                "source_checksum_sha256": source.sha256,
                "estimated_group_uid": source.estimated_group_uid,
                "group_basis": source.group_basis,
                "group_confidence": source.group_confidence,
                "leakage_group_id": assignment.leakage_group_id,
                "leakage_group_image_count": assignment.leakage_group_image_count,
                "estimated_group_count_in_leakage_group": assignment.estimated_group_count,
                "near_duplicate_edge_count": assignment.near_duplicate_edge_count,
                "original_main_split": source.original_main_split,
                "main_split": assignment.main_split,
                "main_split_changed": assignment.main_split_changed,
                "main_assignment_reason": assignment.main_assignment_reason,
                "original_fold": source.original_fold,
                "fold": assignment.fold,
                "fold_changed": assignment.fold_changed,
                "fold_assignment_reason": assignment.fold_assignment_reason,
                "domain_cluster": source.domain_cluster,
                "difficult_quality_subset": source.difficult_quality_subset,
                "edge_target_subset": source.edge_target_subset,
                "pure_background_subset": source.pure_background_subset,
                "speed_proxy_subset": source.speed_proxy_subset,
            }
        )
    return rows


def build_distribution_rows(
    annotations: Sequence[ObjectAnnotation],
    assignments: Mapping[str, ImageAssignment],
) -> list[dict[str, Any]]:
    partitions: list[tuple[str, str, np.ndarray]] = [
        ("all", "all", np.ones(len(annotations), dtype=bool))
    ]
    for split in ("train", "val"):
        partitions.append(
            (
                "main_split",
                split,
                np.asarray(
                    [assignments[item.image_uid].main_split == split for item in annotations],
                    dtype=bool,
                ),
            )
        )
    for fold in range(3):
        partitions.append(
            (
                "fold",
                str(fold),
                np.asarray(
                    [assignments[item.image_uid].fold == fold for item in annotations],
                    dtype=bool,
                ),
            )
        )
    groups: list[tuple[str, str, np.ndarray]] = [
        ("overall", "all_objects", np.ones(len(annotations), dtype=bool))
    ]
    for macro in COARSE_NAMES:
        groups.append(
            (
                "macro",
                macro,
                np.asarray([item.macro_category == macro for item in annotations], dtype=bool),
            )
        )
    for category_id, category_name in enumerate(FINE_NAMES):
        groups.append(
            (
                "fine",
                category_name,
                np.asarray([item.category_id == category_id for item in annotations], dtype=bool),
            )
        )
    image_uids = np.asarray([item.image_uid for item in annotations], dtype=object)
    group_ids = np.asarray(
        [assignments[item.image_uid].leakage_group_id for item in annotations], dtype=object
    )
    rows: list[dict[str, Any]] = []
    for partition_type, partition_value, partition_mask in partitions:
        for group_level, group_name, group_mask in groups:
            mask = partition_mask & group_mask
            if not mask.any():
                continue
            rows.append(
                {
                    "partition_type": partition_type,
                    "partition_value": partition_value,
                    "group_level": group_level,
                    "group_name": group_name,
                    "n_objects": int(mask.sum()),
                    "n_source_images": len(set(image_uids[mask].tolist())),
                    "n_leakage_groups": len(set(group_ids[mask].tolist())),
                }
            )
    return rows


def build_policy_summary(
    manifest_rows: Sequence[Mapping[str, Any]], policies: Sequence[CropPolicy]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        policy_rows = [row for row in manifest_rows if row["crop_policy"] == policy.id]
        for group_name in ("all_objects", *COARSE_NAMES):
            selected = (
                policy_rows
                if group_name == "all_objects"
                else [row for row in policy_rows if row["major_class"] == group_name]
            )
            padding = np.asarray([row["padding_fraction"] for row in selected], dtype=np.float64)
            coverage = np.asarray(
                [row["gt_coverage_fraction"] for row in selected], dtype=np.float64
            )
            side = np.asarray([row["crop_side_px"] for row in selected], dtype=np.float64)
            rows.append(
                {
                    "crop_policy": policy.id,
                    "group_name": group_name,
                    "n_objects": len(selected),
                    "crop_side_p05": float(np.percentile(side, 5)),
                    "crop_side_p50": float(np.percentile(side, 50)),
                    "crop_side_p95": float(np.percentile(side, 95)),
                    "padding_fraction_mean": float(padding.mean()),
                    "padding_fraction_p95": float(np.percentile(padding, 95)),
                    "gt_coverage_fraction_p05": float(np.percentile(coverage, 5)),
                    "gt_coverage_fraction_p50": float(np.percentile(coverage, 50)),
                    "gt_coverage_fraction_mean": float(coverage.mean()),
                    "gt_coverage_lt90_pct": float(100 * (coverage < 0.9).mean()),
                    "gt_coverage_lt75_pct": float(100 * (coverage < 0.75).mean()),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"拒绝写出空 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _format_float(value)
                    if isinstance(value, (float, np.floating))
                    else ""
                    if value is None
                    else value
                    for key, value in row.items()
                }
            )


def verify_source_files(
    data_root: str | Path,
    sources: Mapping[str, ImageSource],
    *,
    verify_checksums: bool,
) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在: {root}")
    verified = 0
    total_bytes = 0
    for image_uid in sorted(sources):
        source = sources[image_uid]
        path = (root / source.relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"数据路径逃逸根目录: {source.relative_path}") from error
        if not path.is_file():
            raise FileNotFoundError(f"源图像不存在: {path}")
        total_bytes += path.stat().st_size
        if verify_checksums:
            actual = _sha256_file(path)
            if actual != source.sha256:
                raise ValueError(
                    f"源图像 checksum 变化: {source.relative_path}; "
                    f"expected={source.sha256}, actual={actual}"
                )
        verified += 1
    ordered_sources = sorted(sources.values(), key=lambda item: item.relative_path)
    dataset_fingerprint = hashlib.sha256(
        "\n".join(
            f"{source.relative_path}\t{source.sha256}" for source in ordered_sources
        ).encode("utf-8")
    ).hexdigest()
    return {
        "data_root": str(root),
        "verified_source_count": verified,
        "verified_sha256": verify_checksums,
        "total_source_bytes": total_bytes,
        "source_set_fingerprint_sha256": dataset_fingerprint,
    }


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]],
    annotations: Sequence[ObjectAnnotation],
    sources: Mapping[str, ImageSource],
    assignments: Mapping[str, ImageAssignment],
    policies: Sequence[CropPolicy],
    near_duplicate_pairs: Sequence[tuple[str, str, str]],
) -> dict[str, Any]:
    expected_policy_ids = {policy.id for policy in policies}
    expected_count = len(annotations) * len(policies)
    if len(manifest_rows) != expected_count:
        raise ValueError(f"manifest 行数错误: {len(manifest_rows)} != {expected_count}")
    crop_ids = [str(row["crop_id"]) for row in manifest_rows]
    if len(crop_ids) != len(set(crop_ids)):
        raise ValueError("crop_id 不唯一")
    policies_by_annotation: dict[str, set[str]] = defaultdict(set)
    geometry_invalid = 0
    fraction_invalid = 0
    for row in manifest_rows:
        policies_by_annotation[str(row["annotation_uid"])].add(str(row["crop_policy"]))
        width = float(row["crop_x1"]) - float(row["crop_x0"])
        height = float(row["crop_y1"]) - float(row["crop_y0"])
        side = float(row["crop_side_px"])
        geometry_invalid += not (
            math.isfinite(width)
            and math.isfinite(height)
            and side > 0
            and math.isclose(width, side, rel_tol=1e-12, abs_tol=1e-9)
            and math.isclose(height, side, rel_tol=1e-12, abs_tol=1e-9)
        )
        for field in (
            "source_valid_fraction",
            "padding_fraction",
            "gt_coverage_fraction",
            "source_gt_visible_fraction",
        ):
            fraction = float(row[field])
            fraction_invalid += not (math.isfinite(fraction) and 0 <= fraction <= 1)
    if geometry_invalid or fraction_invalid:
        raise ValueError(
            f"manifest 几何失效: square={geometry_invalid}, fraction={fraction_invalid}"
        )
    if set(policies_by_annotation) != {item.annotation_uid for item in annotations}:
        raise ValueError("manifest annotation 集合不一致")
    missing_policy_sets = sum(
        policy_ids != expected_policy_ids for policy_ids in policies_by_annotation.values()
    )
    if missing_policy_sets:
        raise ValueError(f"{missing_policy_sets} 个对象的 crop policy 不完整")

    estimated_group_assignments: dict[str, set[tuple[str, int]]] = defaultdict(set)
    leakage_group_assignments: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for source in sources.values():
        assignment = assignments[source.image_uid]
        pair = (assignment.main_split, assignment.fold)
        estimated_group_assignments[source.estimated_group_uid].add(pair)
        leakage_group_assignments[assignment.leakage_group_id].add(pair)
    if any(len(values) != 1 for values in estimated_group_assignments.values()):
        raise ValueError("estimated source group 修复后仍跨 split/fold")
    if any(len(values) != 1 for values in leakage_group_assignments.values()):
        raise ValueError("leakage group 修复后仍跨 split/fold")
    post_near_duplicate_main_violations = 0
    post_near_duplicate_fold_violations = 0
    original_near_duplicate_main_violations = 0
    original_near_duplicate_fold_violations = 0
    for first, second, _ in near_duplicate_pairs:
        first_source, second_source = sources[first], sources[second]
        first_assignment, second_assignment = assignments[first], assignments[second]
        original_near_duplicate_main_violations += (
            first_source.original_main_split != second_source.original_main_split
        )
        original_near_duplicate_fold_violations += (
            first_source.original_fold != second_source.original_fold
        )
        post_near_duplicate_main_violations += (
            first_assignment.main_split != second_assignment.main_split
        )
        post_near_duplicate_fold_violations += first_assignment.fold != second_assignment.fold
    if post_near_duplicate_main_violations or post_near_duplicate_fold_violations:
        raise ValueError("近重复修复后仍跨 split/fold")

    return {
        "valid": True,
        "n_source_images": len(sources),
        "n_annotations": len(annotations),
        "n_policies": len(policies),
        "n_manifest_rows": len(manifest_rows),
        "n_unique_crop_ids": len(set(crop_ids)),
        "all_annotations_have_all_policies": True,
        "all_windows_square_finite_positive": True,
        "all_fractions_in_unit_interval": True,
        "estimated_group_leakage_count": 0,
        "leakage_group_leakage_count": 0,
        "near_duplicate_pair_count": len(near_duplicate_pairs),
        "original_near_duplicate_main_split_violations": int(
            original_near_duplicate_main_violations
        ),
        "original_near_duplicate_fold_violations": int(
            original_near_duplicate_fold_violations
        ),
        "post_repair_near_duplicate_main_split_violations": 0,
        "post_repair_near_duplicate_fold_violations": 0,
        "main_split_moved_images": sum(
            assignment.main_split_changed for assignment in assignments.values()
        ),
        "fold_moved_images": sum(assignment.fold_changed for assignment in assignments.values()),
    }


def _render_float_window(
    image_path: Path, window: tuple[float, float, float, float], size: int
) -> Image.Image:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        return rgb.transform(
            (size, size),
            Image.Transform.EXTENT,
            data=window,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0),
        )


def generate_preview_contact_sheets(
    output_dir: Path,
    data_root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    policies: Sequence[CropPolicy],
    preview_size: int,
) -> tuple[list[str], dict[str, str]]:
    """每类选一个接近本类原生短边中位数的 clean/train 对象做几何 QA。"""

    tight_rows = [
        row
        for row in manifest_rows
        if row["crop_policy"] == "tight"
        and row["main_split"] == "train"
        and not row["source_edge_risk"]
    ]
    selected_annotation_by_class: dict[str, str] = {}
    for category_name in FINE_NAMES:
        candidates = [row for row in tight_rows if row["class_name"] == category_name]
        if not candidates:
            candidates = [
                row
                for row in manifest_rows
                if row["crop_policy"] == "tight" and row["class_name"] == category_name
            ]
        short_edges = np.asarray([row["gt_short_edge"] for row in candidates], dtype=np.float64)
        median = float(np.median(short_edges))
        selected = min(
            candidates,
            key=lambda row: (abs(float(row["gt_short_edge"]) - median), row["annotation_uid"]),
        )
        selected_annotation_by_class[category_name] = str(selected["annotation_uid"])

    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    columns, rows_count = 5, 5
    label_height = 24
    paths: list[str] = []
    for policy in policies:
        canvas = Image.new(
            "RGB", (columns * preview_size, rows_count * (preview_size + label_height)), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for index, category_name in enumerate(FINE_NAMES):
            annotation_uid = selected_annotation_by_class[category_name]
            row = next(
                item
                for item in manifest_rows
                if item["annotation_uid"] == annotation_uid
                and item["crop_policy"] == policy.id
            )
            window = tuple(
                float(row[key]) for key in ("crop_x0", "crop_y0", "crop_x1", "crop_y1")
            )
            crop = _render_float_window(
                data_root / str(row["source_relative_path"]), window, preview_size
            )
            x = (index % columns) * preview_size
            y = (index // columns) * (preview_size + label_height)
            canvas.paste(crop, (x, y))
            draw.rectangle((x, y + preview_size, x + preview_size, y + preview_size + label_height), fill="white")
            draw.text((x + 3, y + preview_size + 5), category_name, fill="black")
        relative_path = f"previews/{policy.id}.png"
        canvas.save(output_dir / relative_path)
        paths.append(relative_path)
    return paths, selected_annotation_by_class


def _write_svg_bar_chart(
    path: Path,
    title: str,
    labels: Sequence[str],
    series: Mapping[str, Sequence[float]],
    y_label: str,
) -> None:
    width = max(760, 140 + len(labels) * 130)
    height = 500
    left, top, right, bottom = 80, 55, 30, 110
    chart_width, chart_height = width - left - right, height - top - bottom
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")
    maximum = max(value for values in series.values() for value in values)
    if maximum <= 100:
        y_max = max(1.0, math.ceil(maximum / 10.0) * 10.0)
    else:
        magnitude = 10 ** math.floor(math.log10(maximum))
        step = magnitude / 2
        y_max = math.ceil(maximum / step) * step
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{escape(title)}</text>',
    ]
    for tick_index in range(6):
        tick = y_max * tick_index / 5
        y = top + chart_height * (1 - tick / y_max)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick:.0f}</text>'
        )
    group_width = chart_width / len(labels)
    bar_width = group_width * 0.72 / len(series)
    for series_index, (name, values) in enumerate(series.items()):
        color = colors[series_index % len(colors)]
        for index, value in enumerate(values):
            x = left + index * group_width + group_width * 0.14 + series_index * bar_width
            bar_height = chart_height * value / y_max
            y = top + chart_height - bar_height
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width-2:.2f}" height="{bar_height:.2f}" fill="{color}"/>'
            )
    for index, label in enumerate(labels):
        x = left + (index + 0.5) * group_width
        parts.append(
            f'<text x="{x:.2f}" y="{top+chart_height+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{escape(label)}</text>'
        )
    for series_index, name in enumerate(series):
        x = left + series_index * 150
        color = colors[series_index % len(colors)]
        parts.append(f'<rect x="{x}" y="{height-25}" width="12" height="12" fill="{color}"/>')
        parts.append(
            f'<text x="{x+18}" y="{height-14}" font-family="sans-serif" font-size="11">{escape(name)}</text>'
        )
    parts.append(
        f'<text x="18" y="{top+chart_height/2}" transform="rotate(-90 18 {top+chart_height/2})" text-anchor="middle" font-family="sans-serif" font-size="12">{escape(y_label)}</text>'
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_figures(
    output_dir: Path,
    distribution_rows: Sequence[Mapping[str, Any]],
    policy_summary: Sequence[Mapping[str, Any]],
) -> list[str]:
    def object_count(partition_type: str, value: str, macro: str) -> float:
        row = _distribution_lookup(distribution_rows, partition_type, value, macro)
        return float(row["n_objects"])

    figure_dir = output_dir / "figures"
    fold_path = figure_dir / "fold_macro_object_distribution.svg"
    _write_svg_bar_chart(
        fold_path,
        "Exploratory leakage-safe fold distribution",
        ["fold_0", "fold_1", "fold_2"],
        {
            macro: [object_count("fold", str(fold), macro) for fold in range(3)]
            for macro in COARSE_NAMES
        },
        "object count",
    )
    split_path = figure_dir / "main_split_macro_object_distribution.svg"
    _write_svg_bar_chart(
        split_path,
        "Exploratory main split distribution",
        ["train", "val"],
        {
            macro: [object_count("main_split", split, macro) for split in ("train", "val")]
            for macro in COARSE_NAMES
        },
        "object count",
    )
    geometry_path = figure_dir / "policy_padding_and_coverage.svg"
    overall = {
        str(row["crop_policy"]): row
        for row in policy_summary
        if row["group_name"] == "all_objects"
    }
    labels = list(overall)
    _write_svg_bar_chart(
        geometry_path,
        "Crop policy geometry diagnostics",
        labels,
        {
            "mean_padding_pct": [
                100 * float(overall[label]["padding_fraction_mean"]) for label in labels
            ],
            "p05_missing_gt_pct": [
                100 * (1 - float(overall[label]["gt_coverage_fraction_p05"]))
                for label in labels
            ],
        },
        "percentage points",
    )
    return [
        "figures/fold_macro_object_distribution.svg",
        "figures/main_split_macro_object_distribution.svg",
        "figures/policy_padding_and_coverage.svg",
    ]


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _distribution_lookup(
    rows: Sequence[Mapping[str, Any]], partition_type: str, value: str, group: str
) -> Mapping[str, Any]:
    return next(
        (
            row
            for row in rows
            if row["partition_type"] == partition_type
            and row["partition_value"] == value
            and row["group_name"] == group
        ),
        {"n_objects": 0, "n_source_images": 0, "n_leakage_groups": 0},
    )


def build_report(
    metadata: Mapping[str, Any],
    validation: Mapping[str, Any],
    sources: Mapping[str, ImageSource],
    assignments: Mapping[str, ImageAssignment],
    _annotations: Sequence[ObjectAnnotation],
    distribution_rows: Sequence[Mapping[str, Any]],
    policy_summary: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
) -> str:
    image_split_counts = Counter(item.main_split for item in assignments.values())
    image_fold_counts = Counter(item.fold for item in assignments.values())
    object_split_rows = []
    for split in ("train", "val"):
        overall = _distribution_lookup(distribution_rows, "main_split", split, "all_objects")
        object_split_rows.append(
            (
                split,
                image_split_counts[split],
                overall["n_objects"],
                *(
                    _distribution_lookup(distribution_rows, "main_split", split, macro)[
                        "n_objects"
                    ]
                    for macro in COARSE_NAMES
                ),
            )
        )
    fold_rows = []
    for fold in range(3):
        overall = _distribution_lookup(distribution_rows, "fold", str(fold), "all_objects")
        fold_rows.append(
            (
                fold,
                image_fold_counts[fold],
                overall["n_objects"],
                *(
                    _distribution_lookup(distribution_rows, "fold", str(fold), macro)[
                        "n_objects"
                    ]
                    for macro in COARSE_NAMES
                ),
                *(
                    _distribution_lookup(distribution_rows, "fold", str(fold), name)[
                        "n_objects"
                    ]
                    for name in ("HM", "LQS", "FSC")
                ),
            )
        )
    policy_rows = []
    for row in policy_summary:
        if row["group_name"] != "all_objects":
            continue
        policy_rows.append(
            (
                row["crop_policy"],
                f"{float(row['crop_side_p50']):.1f}",
                f"{100 * float(row['padding_fraction_mean']):.1f}%",
                f"{100 * float(row['padding_fraction_p95']):.1f}%",
                f"{float(row['gt_coverage_fraction_p05']):.3f}",
                f"{float(row['gt_coverage_fraction_mean']):.3f}",
                f"{float(row['gt_coverage_lt90_pct']):.1f}%",
            )
        )
    rare_rows = []
    for name in ("HM", "LQS", "FSC"):
        overall = _distribution_lookup(distribution_rows, "all", "all", name)
        rare_rows.append(
            (
                name,
                overall["n_objects"],
                overall["n_source_images"],
                overall["n_leakage_groups"],
            )
        )
    rare_counts = {name: count for name, count, _, _ in rare_rows}
    merged_components = [
        row
        for row in component_rows
        if int(row["estimated_group_count"]) > 1
        or int(row["near_duplicate_edge_count"]) > 0
    ]
    source_confidence = Counter(source.group_confidence for source in sources.values())
    config_hash = metadata["inputs"]["config"]["sha256"]
    return f"""# P0-2 探索性 crop manifest 报告

## 1. 结论摘要

已生成 {validation['n_annotations']:,} 个真实对象、{validation['n_policies']} 种几何策略，共 {validation['n_manifest_rows']:,} 行 crop manifest。全部 crop 仅以源图路径、浮点方窗和渲染契约表示；未修改原图，也未物化全量 crop。

本 manifest 可直接用于 P0-3/P0-4 探索实验，但不是正式 split。原因是上游 `estimated_group_uid` 来自文件名启发式分组，置信度为 {dict(source_confidence)}；后续 B 冻结正式划分后必须以新版本重生成。

## 2. 防泄漏划分修复

审计候选 split 已保证同一启发式组同折，但 {validation['near_duplicate_pair_count']} 条近重复候选中，原候选仍有 {validation['original_near_duplicate_main_split_violations']} 条跨 main split、{validation['original_near_duplicate_fold_violations']} 条跨 fold。本次将启发式同源组与近重复连边做保守并集，得到 {len(component_rows)} 个 leakage group，其中 {len(merged_components)} 个含近重复合并证据。

分配先最小化移动图像数，只在票数并列时以 80/20 和三折等分为平衡目标破局。最终 main split 移动 {validation['main_split_moved_images']} 张图，fold 移动 {validation['fold_moved_images']} 张图；修复后近重复跨划分和 leakage group 泄漏均为 0。

{_markdown_table(['main split', 'images', 'objects', 'ship', 'aircraft', 'vehicle'], object_split_rows)}

{_markdown_table(['fold', 'images', 'objects', 'ship', 'aircraft', 'vehicle', 'HM', 'LQS', 'FSC'], fold_rows)}

## 3. crop 策略与几何结果

- `tight`：以 GT 长边为边长的方窗，短边方向保留源图上下文；只在方窗越出原图时补零。
- `context_1p25`：围绕 GT 中心扩张 1.25 倍。
- `jitter_light`：中心偏移最多为宽高的 ±8%，宽高独立缩放为 0.9–1.1，由 SHA-256 直接派生并把实现后的参数存入每行。

{_markdown_table(['policy', 'crop side P50', 'mean padding', 'padding P95', 'GT coverage P5', 'GT coverage mean', 'coverage <0.9'], policy_rows)}

Jitter 只是一个固定的轻度 proposal 鲁棒性诊断集，不代表真实 M1 OOF 误差分布。它的意义是先固定链路和衡量方式，待 M1 产出后必须用真实预测 crop 替换或校准。

224 和 336 不重复生成 manifest 行。它们共用同一浮点 crop window，仅在 loader 运行时直接 resize，从而避免“因分辨率不同而换了样本”的混杂。

## 4. 小样本覆盖

{_markdown_table(['class', 'objects', 'source images', 'leakage groups'], rare_rows)}

HM 和 LQS 仍分别只有 {rare_counts['HM']} 和 {rare_counts['LQS']} 个对象。manifest 不复制尾类行来伪造数据量；P0-3 若需类别均衡，应在 sampler 中实现，且验证集始终保持自然分布。

## 5. P0-3/P0-4 使用契约

1. 三折的第 k 折作验证时，必须先以 `fold == k` 按源图分区，再选 crop policy；不得先展开增强再随机分行。
2. 首轮 P0-3 主衡量只比较 `tight` 和 `context_1p25`，分辨率为 224/336；`jitter_light` 作固定鲁棒性诊断。
3. P0-4 的 ImageNet、DINOv2 和扩散教师必须共用相同 `crop_id`、fold 和输入分辨率。
4. 特征缓存键至少包含 `crop_id + resolution + teacher_weight_checksum + layer/timestep + preprocessing_fingerprint`。
5. 原图外方窗使用 padding，不缩短名义方窗，不因贴边而改变主体缩放比。

## 6. 局限与不得过度解释的内容

- 近重复连边仍是需要人工核验的候选；本 manifest 选择保守合并，可能降低一点数据利用率，但不冒泄漏风险。
- `domain_cluster` 是数值聚类，不是经人工确认的机场/港口/地面语义域。
- `jitter_light` 不能代替真实 OOF proposal。
- 本次只生成正样本对象 crop，没有构造背景类或困难负样本。
- 本 manifest 不改变类别分布，不可用其行数声称样本量扩大了三倍。

## 7. 复现和产物

- manifest version: `{metadata['manifest_version']}`
- generated at: `{metadata['generated_at']}`
- git commit: `{metadata['git']['commit']}`
- config SHA-256: `{config_hash}`
- source set fingerprint: `{metadata['source_verification']['source_set_fingerprint_sha256']}`
- 数值产物：`crop_manifest.csv`、`image_assignments.csv`、`leakage_components.csv`、`class_distribution.csv`、`policy_geometry_summary.csv`、`validation_report.json`、`meta.json`
- QA 产物：3 张几何统计 SVG 和 3 张每类一例的 crop 联系表。预览只做人工几何检查，不进入训练。
"""


def run_crop_manifest_analysis(
    *,
    bbox_statistics_path: str | Path,
    image_stats_path: str | Path,
    split_candidates_path: str | Path,
    domain_assignments_path: str | Path,
    near_duplicates_path: str | Path,
    data_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    command: Sequence[str] | None = None,
    repo_root: str | Path | None = None,
    verify_source_checksums: bool | None = None,
    generate_previews: bool | None = None,
    write_charts: bool = True,
) -> dict[str, Any]:
    bbox_statistics_path = Path(bbox_statistics_path).expanduser().resolve()
    image_stats_path = Path(image_stats_path).expanduser().resolve()
    split_candidates_path = Path(split_candidates_path).expanduser().resolve()
    domain_assignments_path = Path(domain_assignments_path).expanduser().resolve()
    near_duplicates_path = Path(near_duplicates_path).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    repo_root_path = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_crop_manifest_config(config_path)
    manifest_config = config.get("manifest", {})
    if not isinstance(manifest_config, Mapping):
        raise TypeError("manifest 配置必须是映射")
    for key in ("experiment_id", "version", "schema_version", "status"):
        if not str(manifest_config.get(key, "")).strip():
            raise ValueError(f"manifest.{key} 不得为空")
    resolutions = [int(value) for value in manifest_config.get("target_resolutions", [])]
    if resolutions != [224, 336]:
        raise ValueError("根据 P0-1，首轮 target_resolutions 必须固定为 [224, 336]")
    leakage_config = config.get("leakage_control", {})
    if not (
        isinstance(leakage_config, Mapping)
        and leakage_config.get("union_estimated_source_groups") is True
        and leakage_config.get("union_near_duplicate_candidates") is True
    ):
        raise ValueError("探索性 manifest 必须保守合并同源组和近重复候选")
    if leakage_config.get("assignment_policy") != "minimum_moved_images_then_balance":
        raise ValueError("assignment_policy 必须固定为 minimum_moved_images_then_balance")

    render_config = config.get("render_contract", {})
    expected_render_contract = {
        "color_mode": "RGB",
        "square_window_outside_policy": "pad",
        "preview_padding_mode": "constant_rgb_zero",
        "resize_semantics": "direct_square_resize",
        "coordinate_semantics": "continuous_float_xyxy_half_open",
    }
    if not isinstance(render_config, Mapping) or any(
        render_config.get(key) != value for key, value in expected_render_contract.items()
    ):
        raise ValueError("首轮 crop 渲染契约与预注册配置不一致")

    policies = parse_crop_policies(config)
    sources = load_image_sources(
        image_stats_path, split_candidates_path, domain_assignments_path
    )
    annotations = load_annotations(bbox_statistics_path, sources)
    near_duplicate_pairs = load_near_duplicate_pairs(near_duplicates_path, sources)
    assignments, component_rows = build_leakage_safe_assignments(
        sources, near_duplicate_pairs, config
    )
    manifest_rows = build_manifest_rows(
        annotations, sources, assignments, policies, config
    )
    validation = validate_manifest(
        manifest_rows,
        annotations,
        sources,
        assignments,
        policies,
        near_duplicate_pairs,
    )

    verification_config = config.get("verification", {})
    if not isinstance(verification_config, Mapping):
        raise TypeError("verification 必须是映射")
    if verification_config.get("verify_all_source_files_exist") is not True:
        raise ValueError("必须检查全部源图存在")
    if int(verification_config.get("preview_samples_per_class", 0)) != 1:
        raise ValueError("首轮 QA 固定每类一个预览对象")
    preview_size = int(verification_config.get("preview_size", 0))
    if preview_size <= 0:
        raise ValueError("preview_size 必须为正整数")
    should_verify_checksums = (
        bool(verification_config.get("verify_all_source_sha256", True))
        if verify_source_checksums is None
        else verify_source_checksums
    )
    source_verification = verify_source_files(
        data_root, sources, verify_checksums=should_verify_checksums
    )
    distribution_rows = build_distribution_rows(annotations, assignments)
    policy_summary = build_policy_summary(manifest_rows, policies)
    image_assignment_rows = build_image_assignment_rows(sources, assignments)

    _write_csv(output_dir / "crop_manifest.csv", manifest_rows)
    _write_csv(output_dir / "image_assignments.csv", image_assignment_rows)
    _write_csv(output_dir / "leakage_components.csv", component_rows)
    _write_csv(output_dir / "class_distribution.csv", distribution_rows)
    _write_csv(output_dir / "policy_geometry_summary.csv", policy_summary)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    with (output_dir / "validation_report.json").open("w", encoding="utf-8") as file:
        json.dump(validation, file, indent=2, ensure_ascii=False)

    should_generate_previews = (
        bool(verification_config.get("generate_previews", True))
        if generate_previews is None
        else generate_previews
    )
    preview_paths: list[str] = []
    preview_annotation_ids: dict[str, str] = {}
    if should_generate_previews:
        preview_paths, preview_annotation_ids = generate_preview_contact_sheets(
            output_dir,
            data_root,
            manifest_rows,
            policies,
            preview_size,
        )
        with (output_dir / "previews" / "selected_annotations.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(preview_annotation_ids, file, indent=2, ensure_ascii=False)
    figure_paths = (
        write_figures(output_dir, distribution_rows, policy_summary) if write_charts else []
    )

    main_image_counts = Counter(item.main_split for item in assignments.values())
    fold_image_counts = Counter(str(item.fold) for item in assignments.values())
    class_summary = {
        name: {
            "objects": sum(item.category_name == name for item in annotations),
            "source_images": len(
                {item.image_uid for item in annotations if item.category_name == name}
            ),
        }
        for name in FINE_NAMES
    }
    summary = {
        "manifest_version": manifest_config["version"],
        "schema_version": manifest_config["schema_version"],
        "source_images": len(sources),
        "annotations": len(annotations),
        "policies": [policy.id for policy in policies],
        "manifest_rows": len(manifest_rows),
        "leakage_groups": len(component_rows),
        "main_split_image_counts": dict(sorted(main_image_counts.items())),
        "fold_image_counts": dict(sorted(fold_image_counts.items())),
        "class_summary": class_summary,
    }
    with (output_dir / "manifest_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    input_paths = {
        "bbox_statistics": bbox_statistics_path,
        "image_stats": image_stats_path,
        "split_candidates": split_candidates_path,
        "domain_assignments": domain_assignments_path,
        "near_duplicates": near_duplicates_path,
        "config": config_path,
    }
    implementation: dict[str, Any] = {
        "analysis_module": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        }
    }
    cli_path = repo_root_path / "scripts" / "build_crop_manifest.py"
    if cli_path.is_file():
        implementation["cli_script"] = {
            "path": str(cli_path),
            "sha256": _sha256_file(cli_path),
        }
    metadata: dict[str, Any] = {
        "experiment_id": manifest_config["experiment_id"],
        "manifest_version": manifest_config["version"],
        "schema_version": manifest_config["schema_version"],
        "status": manifest_config["status"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": list(command or ()),
        "git": _git_metadata(repo_root_path),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pyyaml": yaml.__version__,
            "pillow": PIL.__version__,
        },
        "inputs": {
            **{
                name: {"path": str(path), "sha256": _sha256_file(path)}
                for name, path in input_paths.items()
            },
            "implementation": implementation,
        },
        "source_verification": source_verification,
        "validation": validation,
        "summary": summary,
        "figures": figure_paths,
        "previews": preview_paths,
        "preview_annotation_ids": preview_annotation_ids,
        "non_claims": [
            "exploratory split is not the formal B-owned split",
            "jitter distribution is not a real detector OOF error distribution",
            "three crop policies do not triple independent source information",
            "manifest contains no hard negatives or background class",
        ],
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    report = build_report(
        metadata,
        validation,
        sources,
        assignments,
        annotations,
        distribution_rows,
        policy_summary,
        component_rows,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return metadata
