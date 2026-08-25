"""MAR20 原始集与竞赛子集的可审计节点登记。"""

from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

from rsdet.grouping.contracts import (
    ANNOTATION_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    REGISTRY_SCHEMA_VERSION,
    atomic_write_json,
    canonical_node_uid,
    canonical_pixel_sha256,
    sha256_file,
    stable_json_sha256,
)

MAR20_TARGET_PATTERN = re.compile(r"^MAR20_([1-9][0-9]*)$")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
AIRCRAFT_CLASS_MIN = 4
AIRCRAFT_CLASS_MAX = 23

REGISTRY_FIELDS = (
    "schema_version",
    "protocol_version",
    "node_uid",
    "mar20_number",
    "is_target",
    "is_bridge",
    "official_side",
    "competition_image_id",
    "original_relative_path",
    "competition_relative_path",
    "original_file_sha256",
    "competition_file_sha256",
    "file_bytes_equal",
    "original_pixel_sha256",
    "competition_pixel_sha256",
    "pixel_equal",
    "width",
    "height",
    "bbox_count",
    "fine_class_hist_json",
    "original_annotation_sha256",
    "competition_annotation_sha256",
    "annotation_class_hist_equal",
)


@dataclass(frozen=True)
class BoxAnnotation:
    class_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class RegistryRecord:
    schema_version: str
    protocol_version: str
    node_uid: str
    mar20_number: int
    is_target: bool
    is_bridge: bool
    official_side: str
    competition_image_id: str
    original_relative_path: str
    competition_relative_path: str
    original_file_sha256: str
    competition_file_sha256: str
    file_bytes_equal: bool
    original_pixel_sha256: str
    competition_pixel_sha256: str
    pixel_equal: bool
    width: int
    height: int
    bbox_count: int
    fine_class_hist_json: str
    original_annotation_sha256: str
    competition_annotation_sha256: str
    annotation_class_hist_equal: bool


def _read_id_list(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values: list[int] = []
    for line_number, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        stripped = text.strip()
        if not stripped:
            continue
        try:
            value = int(stripped)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: MAR20 ID 必须是整数") from error
        if value <= 0:
            raise ValueError(f"{path}:{line_number}: MAR20 ID 必须为正整数")
        values.append(value)
    if len(values) != len(set(values)):
        raise ValueError(f"{path} 内有重复 ID")
    return values


def _parse_original_annotation(
    path: Path, *, image_size: tuple[int, int]
) -> tuple[list[BoxAnnotation], bool]:
    if not path.is_file():
        raise FileNotFoundError(path)
    root = ET.parse(path).getroot()
    xml_width = int(root.findtext("size/width", default="0"))
    xml_height = int(root.findtext("size/height", default="0"))
    width, height = image_size
    size_missing = xml_width <= 0 or xml_height <= 0
    if not size_missing and (xml_width, xml_height) != image_size:
        raise ValueError(f"{path}: XML 尺寸 {(xml_width, xml_height)} 与图像 {image_size} 不一致")
    boxes: list[BoxAnnotation] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if not re.fullmatch(r"A(?:[1-9]|1[0-9]|20)", name):
            raise ValueError(f"{path}: 非法 MAR20 类名 {name!r}")
        class_id = AIRCRAFT_CLASS_MIN + int(name[1:]) - 1
        bbox = obj.find("bndbox")
        if bbox is None:
            raise ValueError(f"{path}: object 缺少 bndbox")
        values = tuple(
            float(bbox.findtext(key, default="nan")) for key in ("xmin", "ymin", "xmax", "ymax")
        )
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"{path}: 非法 HBB {values}")
        boxes.append(BoxAnnotation(class_id=class_id, class_name=name, xyxy=values))
    if not boxes:
        raise ValueError(f"{path}: 不含 object")
    boxes.sort(key=lambda item: (item.class_id, *item.xyxy))
    return boxes, size_missing


def _parse_competition_label(path: Path, width: int, height: int) -> list[BoxAnnotation]:
    if not path.is_file():
        raise FileNotFoundError(path)
    boxes: list[BoxAnnotation] = []
    for line_number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = text.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: YOLO 标签必须为 5 列")
        try:
            class_id = int(parts[0])
            cx, cy, box_width, box_height = map(float, parts[1:])
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: 非法 YOLO 数值") from error
        if not AIRCRAFT_CLASS_MIN <= class_id <= AIRCRAFT_CLASS_MAX:
            raise ValueError(f"{path}:{line_number}: MAR20 图出现非飞机 class_id={class_id}")
        x1 = (cx - box_width / 2.0) * width
        y1 = (cy - box_height / 2.0) * height
        x2 = (cx + box_width / 2.0) * width
        y2 = (cy + box_height / 2.0) * height
        if not (x2 > x1 and y2 > y1):
            raise ValueError(f"{path}:{line_number}: YOLO 转换后为空框")
        boxes.append(
            BoxAnnotation(
                class_id=class_id,
                class_name=f"A{class_id - AIRCRAFT_CLASS_MIN + 1}",
                xyxy=(x1, y1, x2, y2),
            )
        )
    boxes.sort(key=lambda item: (item.class_id, *item.xyxy))
    return boxes


def _annotation_payload(boxes: Sequence[BoxAnnotation]) -> list[dict[str, Any]]:
    return [
        {
            "class_id": item.class_id,
            "class_name": item.class_name,
            "xyxy": [round(value, 6) for value in item.xyxy],
        }
        for item in boxes
    ]


def _class_hist(boxes: Sequence[BoxAnnotation]) -> dict[str, int]:
    counts = Counter(item.class_id for item in boxes)
    return {str(key): counts[key] for key in sorted(counts)}


def _scan_competition_images(data_root: Path) -> dict[int, tuple[str, Path, Path]]:
    result: dict[int, tuple[str, Path, Path]] = {}
    for split in ("train", "val"):
        image_dir = data_root / "images" / split
        label_dir = data_root / "labels" / split
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir(), key=lambda item: item.name):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            matched = MAR20_TARGET_PATTERN.fullmatch(image_path.stem)
            if not matched:
                continue
            number = int(matched.group(1))
            if number in result:
                raise ValueError(f"竞赛数据中 MAR20_{number} 重复")
            label_path = label_dir / f"{image_path.stem}.txt"
            result[number] = (split, image_path, label_path)
    return result


def build_registry(
    *,
    competition_root: str | Path,
    mar20_root: str | Path,
    expected_all: int = 3842,
    expected_target: int = 3073,
    expected_official_train: int = 1331,
    expected_official_test: int = 2511,
) -> tuple[list[RegistryRecord], list[dict[str, Any]], dict[str, Any]]:
    """构建 3,842 节点 registry；发现任何映射歧义即失败。"""

    competition = Path(competition_root).expanduser().resolve()
    mar20 = Path(mar20_root).expanduser().resolve()
    if not competition.is_dir() or not mar20.is_dir():
        raise FileNotFoundError("competition_root 或 mar20_root 不存在")
    train_path = mar20 / "ImageSets" / "Main" / "train.txt"
    test_path = mar20 / "ImageSets" / "Main" / "test.txt"
    train_ids = _read_id_list(train_path)
    test_ids = _read_id_list(test_path)
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"MAR20 train/test ID 重叠: {sorted(overlap)[:10]}")
    all_ids = train_ids + test_ids
    if len(train_ids) != expected_official_train or len(test_ids) != expected_official_test:
        raise ValueError(
            "MAR20 official side 数量错误: "
            f"train expected={expected_official_train}, actual={len(train_ids)}; "
            f"test expected={expected_official_test}, actual={len(test_ids)}"
        )
    if len(all_ids) != expected_all or len(set(all_ids)) != expected_all:
        raise ValueError(f"MAR20 train+test 应为 {expected_all}，实际 {len(set(all_ids))}")
    official_side = {value: "train" for value in train_ids}
    official_side.update({value: "test" for value in test_ids})
    target = _scan_competition_images(competition)
    if len(target) != expected_target:
        raise ValueError(f"竞赛 MAR20 图像应为 {expected_target}，实际 {len(target)}")
    unknown_target = set(target) - set(all_ids)
    if unknown_target:
        raise ValueError(f"竞赛 MAR20 编号不在原始全集: {sorted(unknown_target)[:10]}")

    records: list[RegistryRecord] = []
    annotations: list[dict[str, Any]] = []
    byte_mismatch: list[str] = []
    pixel_mismatch: list[str] = []
    annotation_mismatch: list[str] = []
    xml_size_missing: list[str] = []
    pixel_groups: dict[str, list[str]] = {}
    for number in sorted(all_ids):
        node_uid = canonical_node_uid(number)
        original_image = mar20 / "JPEGImages" / f"{number}.jpg"
        original_xml = mar20 / "Annotations" / "Horizontal Bounding Boxes" / f"{number}.xml"
        if not original_image.is_file():
            raise FileNotFoundError(original_image)
        with Image.open(original_image) as image:
            image.load()
            width, height = image.size
            original_pixel_sha = canonical_pixel_sha256(image)
        original_boxes, size_missing = _parse_original_annotation(
            original_xml, image_size=(width, height)
        )
        if size_missing:
            xml_size_missing.append(node_uid)
        original_file_sha = sha256_file(original_image)
        original_annotation_payload = _annotation_payload(original_boxes)
        original_annotation_sha = stable_json_sha256(original_annotation_payload)
        is_target = number in target
        competition_relative = ""
        competition_image_id = ""
        competition_file_sha = ""
        competition_pixel_sha = ""
        competition_annotation_sha = ""
        file_equal = False
        pixel_equal = False
        class_hist_equal = False
        competition_boxes: list[BoxAnnotation] = []
        if is_target:
            split, competition_image, label_path = target[number]
            competition_relative = competition_image.relative_to(competition).as_posix()
            competition_image_id = competition_image.stem
            with Image.open(competition_image) as image:
                image.load()
                if image.size != (width, height):
                    raise ValueError(f"{competition_image}: 与原 MAR20 尺寸不一致")
                competition_pixel_sha = canonical_pixel_sha256(image)
            competition_file_sha = sha256_file(competition_image)
            competition_boxes = _parse_competition_label(label_path, width, height)
            competition_annotation_sha = stable_json_sha256(_annotation_payload(competition_boxes))
            file_equal = competition_file_sha == original_file_sha
            pixel_equal = competition_pixel_sha == original_pixel_sha
            class_hist_equal = _class_hist(competition_boxes) == _class_hist(original_boxes)
            if not file_equal:
                byte_mismatch.append(node_uid)
            if not pixel_equal:
                pixel_mismatch.append(node_uid)
            if not class_hist_equal:
                annotation_mismatch.append(node_uid)
            # split 只用于追溯相对路径，不进入来源关系。
            if split not in {"train", "val"}:
                raise AssertionError(split)
        class_hist = _class_hist(original_boxes)
        record = RegistryRecord(
            schema_version=REGISTRY_SCHEMA_VERSION,
            protocol_version=PROTOCOL_VERSION,
            node_uid=node_uid,
            mar20_number=number,
            is_target=is_target,
            is_bridge=not is_target,
            official_side=official_side[number],
            competition_image_id=competition_image_id,
            original_relative_path=original_image.relative_to(mar20).as_posix(),
            competition_relative_path=competition_relative,
            original_file_sha256=original_file_sha,
            competition_file_sha256=competition_file_sha,
            file_bytes_equal=file_equal,
            original_pixel_sha256=original_pixel_sha,
            competition_pixel_sha256=competition_pixel_sha,
            pixel_equal=pixel_equal,
            width=width,
            height=height,
            bbox_count=len(original_boxes),
            fine_class_hist_json=json.dumps(class_hist, sort_keys=True, separators=(",", ":")),
            original_annotation_sha256=original_annotation_sha,
            competition_annotation_sha256=competition_annotation_sha,
            annotation_class_hist_equal=class_hist_equal,
        )
        records.append(record)
        annotations.append(
            {
                "schema_version": ANNOTATION_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "node_uid": node_uid,
                "width": width,
                "height": height,
                "boxes": original_annotation_payload,
                "source": "mar20_hbb_xml",
                "source_sha256": sha256_file(original_xml),
                "xml_size_missing_used_image_size": size_missing,
            }
        )
        pixel_groups.setdefault(original_pixel_sha, []).append(node_uid)

    duplicate_groups = [values for values in pixel_groups.values() if len(values) > 1]
    summary = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "pass" if not pixel_mismatch and not annotation_mismatch else "fail",
        "counts": {
            "all": len(records),
            "target": sum(item.is_target for item in records),
            "bridge": sum(item.is_bridge for item in records),
            "official_train": sum(item.official_side == "train" for item in records),
            "official_test": sum(item.official_side == "test" for item in records),
        },
        "target_alignment": {
            "file_byte_mismatch_count": len(byte_mismatch),
            "file_byte_mismatch_examples": byte_mismatch[:20],
            "pixel_mismatch_count": len(pixel_mismatch),
            "pixel_mismatch_examples": pixel_mismatch[:20],
            "class_hist_mismatch_count": len(annotation_mismatch),
            "class_hist_mismatch_examples": annotation_mismatch[:20],
        },
        "h0_duplicate_groups": duplicate_groups,
        "h0_duplicate_group_count": len(duplicate_groups),
        "h0_duplicate_image_count": sum(len(values) for values in duplicate_groups),
        "xml_size_missing_count": len(xml_size_missing),
        "xml_size_missing_examples": xml_size_missing[:20],
        "inputs": {
            "mar20_train_txt_sha256": sha256_file(train_path),
            "mar20_test_txt_sha256": sha256_file(test_path),
        },
    }
    return records, annotations, summary


def write_registry(
    output_dir: str | Path,
    records: Sequence[RegistryRecord],
    annotations: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "image_registry.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["is_target"] = int(record.is_target)
            row["is_bridge"] = int(record.is_bridge)
            row["file_bytes_equal"] = int(record.file_bytes_equal)
            row["pixel_equal"] = int(record.pixel_equal)
            row["annotation_class_hist_equal"] = int(record.annotation_class_hist_equal)
            writer.writerow(row)
    annotations_path = destination / "image_annotations.jsonl"
    temporary = annotations_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for item in annotations:
            file.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(annotations_path)
    result = dict(summary)
    result["artifacts"] = {
        "image_registry.csv": {
            "sha256": sha256_file(csv_path),
            "rows": len(records),
        },
        "image_annotations.jsonl": {
            "sha256": sha256_file(annotations_path),
            "rows": len(annotations),
        },
    }
    atomic_write_json(destination / "registry_summary.json", result)
    return result


def load_registry(path: str | Path, *, expected_rows: int | None = 3842) -> list[dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"registry 应为 {expected_rows} 行，实际 {len(rows)}")
    if any(row.get("schema_version") != REGISTRY_SCHEMA_VERSION for row in rows):
        raise ValueError("registry schema_version 不匹配")
    return rows


def load_annotations(
    path: str | Path, *, expected_rows: int | None = 3842
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line_number, text in enumerate(
        Path(path).expanduser().resolve().read_text(encoding="utf-8").splitlines(), 1
    ):
        if not text.strip():
            continue
        item = json.loads(text)
        if item.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(f"annotations:{line_number}: schema_version 不匹配")
        node_uid = item["node_uid"]
        if node_uid in result:
            raise ValueError(f"annotations node_uid 重复: {node_uid}")
        result[node_uid] = item
    if expected_rows is not None and len(result) != expected_rows:
        raise ValueError(f"annotations 应为 {expected_rows} 行，实际 {len(result)}")
    return result


def class_histogram(records: Iterable[RegistryRecord]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for record in records:
        total.update(json.loads(record.fine_class_hist_json))
    return {key: total[key] for key in sorted(total, key=int)}
