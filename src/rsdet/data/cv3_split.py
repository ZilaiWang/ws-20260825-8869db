"""Contracts and audits for the frozen three-fold grouped split.

The formal CV3 split treats every source group as an indivisible atom.  In
particular, the MAR20 K60 airport-proxy mapping is authoritative: legacy
near-duplicate candidates may be audited, but they must never merge or rewrite
K60 groups.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rsdet.data.xh_dataset import FINE_NAMES

FOLD_COUNT = 3
EXPECTED_MAR20_IMAGES = 3073
EXPECTED_K60_GROUPS = 60
K60_GROUP_RE = re.compile(r"^mar20-airport-proxy-(\d{3})$")
MAR20_RE = re.compile(r"^MAR20_(\d+)$")
SCENE_RE = re.compile(r"(L\d+A\d+|L\d{11})")
LATLON_RE = re.compile(r"(N[\d.]+-E[\d.]+)")
TU160_CLASS_ID = 9
TU160_EXPECTED_GROUP_BOXES = {
    "mar20-airport-proxy-018": 352,
    "mar20-airport-proxy-019": 1,
    "mar20-airport-proxy-020": 8,
}

# These are attainable, data-derived floors.  A universal 30-box floor is
# impossible for HM (17 boxes), LQS (30 boxes), and grouped TU-160 (352/8/1).
MIN_TRAIN_BOXES = {
    class_id: {0: 11, 1: 20, TU160_CLASS_ID: 9}.get(class_id, 30)
    for class_id in range(len(FINE_NAMES))
}


@dataclass(frozen=True)
class CV3ImageRecord:
    """One competition image and all information needed by the split audit."""

    image_id: int
    stem: str
    relative_path: str
    group_id: str
    group_rule: str
    class_ids: tuple[int, ...]
    coarse: str


@dataclass(frozen=True)
class CV3GroupRecord:
    """One indivisible source group."""

    group_id: str
    image_ids: tuple[int, ...]
    class_box_counts: Counter[int]
    coarse: str


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    """Hash a canonical JSON representation."""

    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write deterministic JSON only after the caller has passed all gates."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_airport_groups(
    path: Path,
    *,
    expected_stems: set[str],
    expected_image_count: int = EXPECTED_MAR20_IMAGES,
    expected_group_count: int = EXPECTED_K60_GROUPS,
) -> dict[str, str]:
    """Load and strictly validate the authoritative K60 two-column mapping."""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["image_name", "group_id"]:
            raise ValueError(
                "airport-group CSV header must be exactly image_name,group_id"
            )
        mapping: dict[str, str] = {}
        for line_number, row in enumerate(reader, 2):
            image_name = row["image_name"].strip()
            group_id = row["group_id"].strip()
            if not image_name:
                raise ValueError(f"{path}:{line_number}: empty image_name")
            if not group_id:
                raise ValueError(f"{path}:{line_number}: empty group_id")
            stem = Path(image_name).stem
            if Path(image_name).suffix.lower() not in {".jpg", ".jpeg"}:
                raise ValueError(
                    f"{path}:{line_number}: image_name must include a JPEG suffix"
                )
            if not MAR20_RE.fullmatch(stem):
                raise ValueError(f"{path}:{line_number}: invalid MAR20 name {image_name}")
            if stem in mapping:
                raise ValueError(f"{path}:{line_number}: duplicate assignment for {stem}")
            match = K60_GROUP_RE.fullmatch(group_id)
            if match is None or not 1 <= int(match.group(1)) <= expected_group_count:
                raise ValueError(f"{path}:{line_number}: invalid K60 group_id {group_id}")
            mapping[stem] = group_id

    if len(mapping) != expected_image_count:
        raise ValueError(
            f"K60 mapping must contain {expected_image_count} unique images, "
            f"found {len(mapping)}"
        )
    group_ids = set(mapping.values())
    expected_group_ids = {
        f"mar20-airport-proxy-{number:03d}"
        for number in range(1, expected_group_count + 1)
    }
    if group_ids != expected_group_ids:
        raise ValueError(
            "K60 group set mismatch: "
            f"missing={sorted(expected_group_ids - group_ids)}, "
            f"extra={sorted(group_ids - expected_group_ids)}"
        )
    if set(mapping) != expected_stems:
        raise ValueError(
            "K60 image coverage mismatch: "
            f"dataset_only={sorted(expected_stems - set(mapping))[:5]}, "
            f"mapping_only={sorted(set(mapping) - expected_stems)[:5]}"
        )
    return mapping


def derive_non_mar20_group(stem: str) -> tuple[str, str]:
    """Derive the authoritative ship/vehicle group from the competition name."""

    scene = SCENE_RE.search(stem)
    if scene:
        return f"scene_{scene.group(1)}", "ship_scene_id"
    latlon = LATLON_RE.search(stem)
    if latlon:
        return f"site_{latlon.group(1)}", "vehicle_latlon"
    if MAR20_RE.fullmatch(stem):
        raise ValueError(f"MAR20 image must come from the complete K60 map: {stem}")
    raise ValueError(f"cannot derive a non-MAR20 source group for {stem}")


def read_class_ids(label_path: Path | None) -> list[int]:
    """Read repeated fine-class IDs from one YOLO label file."""

    if label_path is None:
        return []
    result: list[int] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_number}: expected five YOLO columns"
            )
        result.append(int(parts[0]))
    return result


def aggregate_groups(records: Sequence[CV3ImageRecord]) -> list[CV3GroupRecord]:
    """Aggregate image records while rejecting mixed-domain source groups."""

    image_ids: defaultdict[str, list[int]] = defaultdict(list)
    box_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    coarse_by_group: dict[str, str] = {}
    for record in records:
        previous = coarse_by_group.setdefault(record.group_id, record.coarse)
        if previous != record.coarse:
            raise ValueError(
                f"group {record.group_id} mixes coarse domains {previous}/{record.coarse}"
            )
        image_ids[record.group_id].append(record.image_id)
        box_counts[record.group_id].update(record.class_ids)

    return [
        CV3GroupRecord(
            group_id=group_id,
            image_ids=tuple(sorted(image_ids[group_id])),
            class_box_counts=box_counts[group_id],
            coarse=coarse_by_group[group_id],
        )
        for group_id in sorted(image_ids)
    ]


def group_statistics_sha256(groups: Sequence[CV3GroupRecord]) -> str:
    """Fingerprint the exact group/image/label statistics used by the split."""

    payload = [
        {
            "group_id": group.group_id,
            "image_ids": list(group.image_ids),
            "class_box_counts": {
                str(class_id): group.class_box_counts[class_id]
                for class_id in sorted(group.class_box_counts)
            },
            "coarse": group.coarse,
        }
        for group in sorted(groups, key=lambda item: item.group_id)
    ]
    return sha256_json(payload)


def load_frozen_assignment(
    path: Path,
    *,
    expected_group_ids: set[str],
    expected_group_statistics_sha256: str,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Load a versioned, input-bound group-to-fold assignment."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != "cv3_airport_proxy_k60_v2_group_assignment_v1":
        raise ValueError("unexpected frozen assignment version")
    if payload.get("fold_count") != FOLD_COUNT:
        raise ValueError(f"frozen assignment must contain {FOLD_COUNT} folds")
    if payload.get("group_statistics_sha256") != expected_group_statistics_sha256:
        raise ValueError(
            "frozen assignment group-statistics fingerprint does not match current data"
        )
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, dict):
        raise ValueError("frozen assignment requires a groups object")
    if set(raw_groups) != expected_group_ids:
        raise ValueError(
            "frozen assignment group coverage mismatch: "
            f"missing={sorted(expected_group_ids - set(raw_groups))[:5]}, "
            f"extra={sorted(set(raw_groups) - expected_group_ids)[:5]}"
        )
    assignment: dict[str, int] = {}
    for group_id, fold in raw_groups.items():
        if type(fold) is not int or not 0 <= fold < FOLD_COUNT:
            raise ValueError(f"invalid fold for {group_id}: {fold!r}")
        assignment[group_id] = fold
    return assignment, payload


def audit_near_duplicates(
    path: Path | None, airport_groups: Mapping[str, str]
) -> dict[str, Any]:
    """Audit legacy dHash candidates without changing any K60 assignment."""

    result: dict[str, Any] = {
        "policy": "audit_only_never_merge_k60",
        "path": str(path) if path is not None else None,
        "sha256": None,
        "candidate_groups": 0,
        "cross_k60_candidate_groups": 0,
        "cross_k60_candidates": [],
    }
    if path is None:
        return result
    if not path.is_file():
        raise FileNotFoundError(path)
    result["sha256"] = sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    duplicate_groups = payload.get("duplicate_groups")
    if not isinstance(duplicate_groups, list):
        raise ValueError("near-duplicate audit file requires duplicate_groups")
    result["candidate_groups"] = len(duplicate_groups)
    cross: list[dict[str, Any]] = []
    for members in duplicate_groups:
        stems = [Path(str(member)).stem for member in members]
        group_ids = sorted(
            {airport_groups[stem] for stem in stems if stem in airport_groups}
        )
        if len(group_ids) > 1:
            cross.append({"stems": stems, "k60_group_ids": group_ids})
    result["cross_k60_candidate_groups"] = len(cross)
    result["cross_k60_candidates"] = cross
    return result


def label_set_sha256(records: Sequence[CV3ImageRecord]) -> str:
    """Fingerprint image IDs and their ordered class-label sequences."""

    payload = [
        {
            "image_id": record.image_id,
            "stem": record.stem,
            "class_ids": list(record.class_ids),
        }
        for record in sorted(records, key=lambda item: item.image_id)
    ]
    return sha256_json(payload)


def _sets_by_class_and_fold(
    records: Sequence[CV3ImageRecord], assignment: Mapping[str, int]
) -> tuple[
    Counter[tuple[int, int]],
    defaultdict[tuple[int, int], set[int]],
    defaultdict[tuple[int, int], set[str]],
]:
    boxes: Counter[tuple[int, int]] = Counter()
    images: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    groups: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
    for record in records:
        fold = assignment[record.group_id]
        for class_id in record.class_ids:
            boxes[(class_id, fold)] += 1
            images[(class_id, fold)].add(record.image_id)
            groups[(class_id, fold)].add(record.group_id)
    return boxes, images, groups


def build_scientific_audit(
    records: Sequence[CV3ImageRecord],
    groups: Sequence[CV3GroupRecord],
    assignment: Mapping[str, int],
    airport_groups: Mapping[str, str],
) -> dict[str, Any]:
    """Evaluate every hard gate before a formal manifest may be written."""

    violations: list[str] = []
    warnings: list[str] = []
    group_ids = {group.group_id for group in groups}
    if set(assignment) != group_ids:
        violations.append("assignment_group_coverage_mismatch")

    fold_image_counts = [0] * FOLD_COUNT
    fold_box_counts = [0] * FOLD_COUNT
    fold_group_counts = [0] * FOLD_COUNT
    fold_coarse_images: list[Counter[str]] = [Counter() for _ in range(FOLD_COUNT)]
    for group in groups:
        fold = assignment[group.group_id]
        fold_group_counts[fold] += 1
        fold_image_counts[fold] += len(group.image_ids)
        fold_box_counts[fold] += sum(group.class_box_counts.values())
    for record in records:
        fold_coarse_images[assignment[record.group_id]][record.coarse] += 1

    boxes, images, class_groups = _sets_by_class_and_fold(records, assignment)
    group_class_counts: Counter[int] = Counter()
    total_boxes: Counter[int] = Counter()
    for group in groups:
        total_boxes.update(group.class_box_counts)
        for class_id, count in group.class_box_counts.items():
            if count:
                group_class_counts[class_id] += 1

    per_class: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(FINE_NAMES):
        val_boxes = [boxes[(class_id, fold)] for fold in range(FOLD_COUNT)]
        val_images = [
            len(images[(class_id, fold)]) for fold in range(FOLD_COUNT)
        ]
        val_groups = [
            len(class_groups[(class_id, fold)]) for fold in range(FOLD_COUNT)
        ]
        train_boxes = [total_boxes[class_id] - count for count in val_boxes]
        minimum = MIN_TRAIN_BOXES[class_id]
        if total_boxes[class_id] <= 0:
            violations.append(f"class_{class_id}_absent_from_dataset")
            largest_share = 0.0
        else:
            largest_share = max(val_boxes) / total_boxes[class_id]
        if group_class_counts[class_id] < FOLD_COUNT:
            violations.append(
                f"class_{class_id}_has_only_{group_class_counts[class_id]}_source_groups"
            )
        for fold in range(FOLD_COUNT):
            if val_boxes[fold] <= 0 or val_groups[fold] <= 0:
                violations.append(f"class_{class_id}_fold_{fold}_validation_uncovered")
            if train_boxes[fold] < minimum:
                violations.append(
                    f"class_{class_id}_fold_{fold}_train_boxes_"
                    f"{train_boxes[fold]}_below_{minimum}"
                )
        if largest_share > 0.60:
            warnings.append(
                f"{class_name}: validation box share is group-dominated "
                f"({largest_share:.3f})"
            )
        if group_class_counts[class_id] >= 6 and min(val_groups) == 1:
            warnings.append(
                f"{class_name}: at least one fold has only one source group"
            )
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "total_boxes": total_boxes[class_id],
                "source_group_count": group_class_counts[class_id],
                "minimum_train_boxes": minimum,
                "val_boxes": val_boxes,
                "val_images": val_images,
                "val_groups": val_groups,
                "train_boxes": train_boxes,
                "minimum_observed_train_boxes": min(train_boxes),
                "largest_validation_box_share": largest_share,
            }
        )

    k60_records = [record for record in records if MAR20_RE.fullmatch(record.stem)]
    if len(k60_records) != EXPECTED_MAR20_IMAGES:
        violations.append(f"mar20_image_count_is_{len(k60_records)}")
    k60_group_ids = {record.group_id for record in k60_records}
    if len(k60_group_ids) != EXPECTED_K60_GROUPS:
        violations.append(f"k60_group_count_is_{len(k60_group_ids)}")
    mismatches = [
        record.stem
        for record in k60_records
        if airport_groups.get(record.stem) != record.group_id
    ]
    if mismatches:
        violations.append(f"k60_mapping_mismatches_{len(mismatches)}")

    tu_group_boxes = {
        group.group_id: group.class_box_counts[TU160_CLASS_ID]
        for group in groups
        if group.class_box_counts[TU160_CLASS_ID]
    }
    if tu_group_boxes != TU160_EXPECTED_GROUP_BOXES:
        violations.append(f"unexpected_tu160_group_boxes_{tu_group_boxes}")
    tu_group_folds = {
        group_id: assignment[group_id] for group_id in sorted(tu_group_boxes)
    }
    if set(tu_group_folds.values()) != set(range(FOLD_COUNT)):
        violations.append("tu160_source_groups_not_in_three_distinct_folds")
    tu_row = per_class[TU160_CLASS_ID]
    if sorted(tu_row["val_boxes"]) != [1, 8, 352]:
        violations.append(f"unexpected_tu160_val_boxes_{tu_row['val_boxes']}")
    if sorted(tu_row["train_boxes"]) != [9, 353, 360]:
        violations.append(f"unexpected_tu160_train_boxes_{tu_row['train_boxes']}")

    sample_group_folds: defaultdict[str, set[int]] = defaultdict(set)
    for record in records:
        sample_group_folds[record.group_id].add(assignment[record.group_id])
    crossing = sorted(
        group_id for group_id, folds in sample_group_folds.items() if len(folds) > 1
    )
    if crossing:
        violations.append(f"cross_fold_groups_{len(crossing)}")

    return {
        "status": "pass" if not violations else "fail",
        "formal_cv3_admission": not violations,
        "hard_gate_violations": violations,
        "warnings": warnings,
        "sample_count": len(records),
        "source_group_count": len(groups),
        "fold_image_counts": fold_image_counts,
        "fold_box_counts": fold_box_counts,
        "fold_group_counts": fold_group_counts,
        "fold_coarse_image_counts": [
            dict(sorted(counter.items())) for counter in fold_coarse_images
        ],
        "group_cross_fold_count": len(crossing),
        "mar20_image_count": len(k60_records),
        "mar20_group_count": len(k60_group_ids),
        "k60_mapping_mismatch_count": len(mismatches),
        "per_class": per_class,
        "tu160_stress_fold": {
            "group_boxes": dict(sorted(tu_group_boxes.items())),
            "group_folds": tu_group_folds,
            "val_boxes": tu_row["val_boxes"],
            "train_boxes": tu_row["train_boxes"],
            "minimum_train_boxes": min(tu_row["train_boxes"]),
            "interpretation": (
                "The 9-box training fold is an unavoidable grouped stress fold, "
                "not an optimization failure."
            ),
        },
    }


def make_manifest_samples(
    records: Iterable[CV3ImageRecord], assignment: Mapping[str, int]
) -> list[dict[str, Any]]:
    """Convert validated records to the integration-contract sample schema."""

    return [
        {
            "image_id": record.image_id,
            "relative_path": record.relative_path,
            "fold": assignment[record.group_id],
            "group_id": record.group_id,
            "group_rule": record.group_rule,
        }
        for record in sorted(records, key=lambda item: item.image_id)
    ]
