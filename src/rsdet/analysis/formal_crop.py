"""Rehang the frozen P0-2 crop rows onto the formal CV3 source groups.

The operation is deliberately metadata-only:

* source pixels are never opened;
* ``crop_id`` and every geometry/render field are copied byte-for-byte;
* source images join to CV3 by the exact ``source_relative_path``;
* exploratory assignments remain only in ``historical_p02_*`` columns;
* the active ``fold/group_id/leakage_group_id`` fields come exclusively from
  the frozen formal CV3 manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsdet.data.formal_cv3 import FormalCV3Manifest, sha256_file

REQUIRED_EXPLORATORY_COLUMNS = {
    "manifest_version",
    "schema_version",
    "crop_id",
    "annotation_uid",
    "source_image_id",
    "source_relative_path",
    "source_width",
    "source_height",
    "source_checksum_sha256",
    "class_id",
    "class_name",
    "major_class",
    "crop_policy",
    "crop_x0",
    "crop_y0",
    "crop_x1",
    "crop_y1",
    "fold",
    "group_id",
    "leakage_group_id",
    "color_mode",
    "outside_policy",
    "resize_semantics",
}

# These fields are tied to the P0-2 exploratory split/group heuristic.  They
# remain traceable but are never left under an active-looking formal name.
HISTORICAL_ASSIGNMENT_COLUMNS = {
    "original_main_split",
    "original_fold",
    "main_split",
    "fold",
    "main_split_changed",
    "fold_changed",
    "main_assignment_reason",
    "fold_assignment_reason",
    "group_id",
    "leakage_group_id",
    "leakage_group_image_count",
    "estimated_group_uid",
    "estimated_group_count_in_leakage_group",
    "group_basis",
    "group_confidence",
    "near_duplicate_edge_count",
}

FORMAL_COLUMNS = (
    "formal_image_id",
    "fold",
    "group_id",
    "leakage_group_id",
    "group_rule",
    "formal_group_image_count",
    "formal_cv3_version",
    "formal_cv3_sha256",
    "assignment_scope",
    "geometry_source_manifest_version",
    "crop_id_provenance",
    "source_crop_manifest_sha256",
)

ANNOTATION_INVARIANT_COLUMNS = (
    "source_image_id",
    "source_relative_path",
    "class_id",
    "class_name",
    "major_class",
)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
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


def _safe_relative_path(value: str, *, line_number: int) -> str:
    normalized = value.strip().replace("\\", "/")
    path = Path(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"exploratory crop row {line_number} has unsafe source path={value!r}"
        )
    return normalized


def _canonical_rows_fingerprint(
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    digest.update("\x1f".join(columns).encode("utf-8"))
    digest.update(b"\n")
    for row in rows:
        digest.update(
            "\x1f".join(str(row[column]) for column in columns).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _load_exploratory_rows(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        missing = sorted(REQUIRED_EXPLORATORY_COLUMNS - set(fieldnames))
        if missing:
            raise ValueError(
                f"exploratory crop manifest is missing columns: {missing}"
            )
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("exploratory crop manifest has duplicate columns")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("exploratory crop manifest is empty")
    return fieldnames, rows


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "version",
        "schema_version",
        "source_manifest_version",
        "source_manifest_schema_version",
        "source_manifest_sha256",
        "expected_rows",
        "expected_annotations",
        "expected_source_images",
        "expected_policies",
        "expected_fold_object_counts",
        "preserve_crop_id",
        "preserve_non_assignment_fields",
        "pixel_io_policy",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"formal_crop config is missing fields: {missing}")
    result = dict(config)
    for key in (
        "version",
        "schema_version",
        "source_manifest_version",
        "source_manifest_schema_version",
        "source_manifest_sha256",
    ):
        if not str(result[key]).strip():
            raise ValueError(f"formal_crop.{key} must not be empty")
    for key in ("expected_rows", "expected_annotations", "expected_source_images"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"formal_crop.{key} must be a positive integer")
    policies = tuple(str(value).strip() for value in result["expected_policies"])
    if not policies or any(not value for value in policies) or len(policies) != len(
        set(policies)
    ):
        raise ValueError("formal_crop.expected_policies must be unique and non-empty")
    fold_counts = tuple(int(value) for value in result["expected_fold_object_counts"])
    if len(fold_counts) != 3 or any(value <= 0 for value in fold_counts):
        raise ValueError(
            "formal_crop.expected_fold_object_counts must contain three positive values"
        )
    if sum(fold_counts) != result["expected_annotations"]:
        raise ValueError(
            "formal_crop fold object counts do not sum to expected_annotations"
        )
    if result["expected_rows"] != result["expected_annotations"] * len(policies):
        raise ValueError(
            "formal_crop expected_rows must equal annotations times policy count"
        )
    if result["preserve_crop_id"] is not True:
        raise ValueError("formal crop rehang must preserve crop_id")
    if result["preserve_non_assignment_fields"] is not True:
        raise ValueError("formal crop rehang must preserve non-assignment fields")
    if result["pixel_io_policy"] != "forbidden_metadata_only":
        raise ValueError("formal crop rehang must forbid pixel I/O")
    result["expected_policies"] = policies
    result["expected_fold_object_counts"] = fold_counts
    return result


def _output_fieldnames(input_fieldnames: Sequence[str]) -> list[str]:
    result: list[str] = []
    for field in input_fieldnames:
        if field in HISTORICAL_ASSIGNMENT_COLUMNS:
            result.append(f"historical_p02_{field}")
        else:
            result.append(field)
    collisions = set(result) & set(FORMAL_COLUMNS)
    if collisions:
        raise ValueError(f"formal crop output-column collision: {sorted(collisions)}")
    result.extend(FORMAL_COLUMNS)
    if len(result) != len(set(result)):
        raise ValueError("formal crop output columns are not unique")
    return result


def _write_csv_atomic(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _historical_value(row: Mapping[str, str], field: str) -> str:
    return str(row.get(field, ""))


def build_formal_crop_manifest(
    *,
    exploratory_manifest_path: str | Path,
    formal_cv3: FormalCV3Manifest,
    config: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create and audit ``formal_crop_manifest_v2`` without reading any pixels."""

    source_path = Path(exploratory_manifest_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output = Path(output_dir).expanduser().resolve()
    protected_outputs = (
        output / "formal_crop_manifest.csv",
        output / "formal_crop_audit.json",
    )
    existing = [path for path in protected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "formal crop outputs are immutable; rerun into a fresh output directory: "
            + ", ".join(str(path) for path in existing)
        )
    contract = _validate_config(config)
    source_sha256 = sha256_file(source_path)
    expected_source_sha256 = str(contract["source_manifest_sha256"]).lower()
    if source_sha256 != expected_source_sha256:
        raise ValueError(
            "exploratory crop manifest SHA256 mismatch: "
            f"expected={expected_source_sha256}, actual={source_sha256}"
        )
    input_fields, input_rows = _load_exploratory_rows(source_path)
    if len(input_rows) != contract["expected_rows"]:
        raise ValueError(
            f"exploratory crop row count changed: "
            f"{len(input_rows)} != {contract['expected_rows']}"
        )

    input_versions = {row["manifest_version"].strip() for row in input_rows}
    input_schemas = {row["schema_version"].strip() for row in input_rows}
    if input_versions != {contract["source_manifest_version"]}:
        raise ValueError(f"unexpected exploratory manifest versions: {input_versions}")
    if input_schemas != {contract["source_manifest_schema_version"]}:
        raise ValueError(f"unexpected exploratory schema versions: {input_schemas}")

    cv_by_path = {sample.relative_path: sample for sample in formal_cv3.samples}
    if len(cv_by_path) != len(formal_cv3.samples):
        raise ValueError("formal CV3 contains duplicate relative paths")
    source_path_by_id: dict[str, str] = {}
    source_id_by_path: dict[str, str] = {}
    crop_ids: set[str] = set()
    annotations: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for line_number, row in enumerate(input_rows, 2):
        source_id = row["source_image_id"].strip()
        annotation_uid = row["annotation_uid"].strip()
        crop_id = row["crop_id"].strip()
        source_relative_path = _safe_relative_path(
            row["source_relative_path"],
            line_number=line_number,
        )
        policy = row["crop_policy"].strip()
        if not source_id or not annotation_uid or not crop_id or not policy:
            raise ValueError(
                f"exploratory crop row {line_number} has an empty stable ID"
            )
        if crop_id in crop_ids:
            raise ValueError(f"exploratory crop_id is duplicated: {crop_id}")
        crop_ids.add(crop_id)
        previous_path = source_path_by_id.setdefault(source_id, source_relative_path)
        if previous_path != source_relative_path:
            raise ValueError(f"source_image_id {source_id} maps to multiple paths")
        previous_id = source_id_by_path.setdefault(source_relative_path, source_id)
        if previous_id != source_id:
            raise ValueError(
                f"source path {source_relative_path} maps to multiple source_image_id values"
            )
        if source_relative_path not in cv_by_path:
            raise ValueError(
                f"P0-2 source path is absent from formal CV3: {source_relative_path}"
            )
        annotations[annotation_uid].append(row)

    if len(annotations) != contract["expected_annotations"]:
        raise ValueError(
            f"exploratory annotation count changed: "
            f"{len(annotations)} != {contract['expected_annotations']}"
        )
    if len(source_path_by_id) != contract["expected_source_images"]:
        raise ValueError(
            f"exploratory source-image count changed: "
            f"{len(source_path_by_id)} != {contract['expected_source_images']}"
        )
    if set(source_id_by_path) != set(cv_by_path):
        raise ValueError(
            "P0-2/CV3 source-path coverage mismatch: "
            f"p0_only={sorted(set(source_id_by_path) - set(cv_by_path))[:3]}, "
            f"cv3_only={sorted(set(cv_by_path) - set(source_id_by_path))[:3]}"
        )

    expected_policies = set(contract["expected_policies"])
    annotation_source: dict[str, str] = {}
    for annotation_uid, rows in annotations.items():
        policies = {row["crop_policy"].strip() for row in rows}
        if len(rows) != len(expected_policies) or policies != expected_policies:
            raise ValueError(
                f"annotation {annotation_uid} does not have exactly "
                f"{sorted(expected_policies)}"
            )
        for field in ANNOTATION_INVARIANT_COLUMNS:
            values = {row[field] for row in rows}
            if len(values) != 1:
                raise ValueError(
                    f"annotation {annotation_uid} changes {field} across policies"
                )
        annotation_source[annotation_uid] = rows[0]["source_image_id"].strip()

    formal_group_images: Counter[str] = Counter(
        sample.group_id for sample in formal_cv3.samples
    )
    output_fields = _output_fieldnames(input_fields)
    output_rows: list[dict[str, str]] = []
    preserved_fields = [
        field
        for field in input_fields
        if field
        not in HISTORICAL_ASSIGNMENT_COLUMNS | {"manifest_version", "schema_version"}
    ]
    input_preserved_fingerprint = _canonical_rows_fingerprint(
        input_rows,
        preserved_fields,
    )
    policy_fold_objects: defaultdict[tuple[str, int], int] = defaultdict(int)
    policy_fold_sources: defaultdict[tuple[str, int], set[str]] = defaultdict(set)
    policy_fold_groups: defaultdict[tuple[str, int], set[str]] = defaultdict(set)
    policy_fold_classes: defaultdict[tuple[str, int], Counter[int]] = defaultdict(
        Counter
    )

    for row in input_rows:
        source_relative_path = row["source_relative_path"].strip().replace("\\", "/")
        sample = cv_by_path[source_relative_path]
        output_row: dict[str, str] = {}
        for field in input_fields:
            if field in HISTORICAL_ASSIGNMENT_COLUMNS:
                output_row[f"historical_p02_{field}"] = _historical_value(row, field)
            elif field == "manifest_version":
                output_row[field] = str(contract["version"])
            elif field == "schema_version":
                output_row[field] = str(contract["schema_version"])
            else:
                output_row[field] = row[field]
        output_row.update(
            {
                "formal_image_id": str(sample.image_id),
                "fold": str(sample.fold),
                "group_id": sample.group_id,
                "leakage_group_id": sample.group_id,
                "group_rule": sample.group_rule,
                "formal_group_image_count": str(formal_group_images[sample.group_id]),
                "formal_cv3_version": formal_cv3.version,
                "formal_cv3_sha256": formal_cv3.sha256,
                "assignment_scope": "formal_cv3_fold_only",
                "geometry_source_manifest_version": row["manifest_version"],
                "crop_id_provenance": "preserved_from_exploratory_crop_manifest",
                "source_crop_manifest_sha256": source_sha256,
            }
        )
        output_rows.append(output_row)
        key = (row["crop_policy"].strip(), sample.fold)
        policy_fold_objects[key] += 1
        policy_fold_sources[key].add(row["source_image_id"].strip())
        policy_fold_groups[key].add(sample.group_id)
        policy_fold_classes[key][int(row["class_id"])] += 1

    expected_fold_objects = tuple(contract["expected_fold_object_counts"])
    for policy in contract["expected_policies"]:
        actual = tuple(policy_fold_objects[(policy, fold)] for fold in range(3))
        if actual != expected_fold_objects:
            raise ValueError(
                f"{policy} formal fold object counts changed: "
                f"{actual} != {expected_fold_objects}"
            )
        for fold in range(3):
            if set(policy_fold_classes[(policy, fold)]) != set(range(25)):
                missing = set(range(25)) - set(policy_fold_classes[(policy, fold)])
                raise ValueError(
                    f"{policy}/fold{fold} lacks fine classes: {sorted(missing)}"
                )

    output_preserved_fingerprint = _canonical_rows_fingerprint(
        output_rows,
        preserved_fields,
    )
    if output_preserved_fingerprint != input_preserved_fingerprint:
        raise RuntimeError("non-assignment P0-2 fields changed during formal rehang")
    if [row["crop_id"] for row in input_rows] != [
        row["crop_id"] for row in output_rows
    ]:
        raise RuntimeError("crop_id changed during formal rehang")

    for held_out_fold in range(3):
        train_annotation_ids = {
            annotation_uid
            for annotation_uid, source_id in annotation_source.items()
            if cv_by_path[source_path_by_id[source_id]].fold != held_out_fold
        }
        val_annotation_ids = set(annotation_source) - train_annotation_ids
        if train_annotation_ids & val_annotation_ids:
            raise RuntimeError("formal crop train/val annotation overlap")
        train_source_ids = {
            annotation_source[uid] for uid in train_annotation_ids
        }
        val_source_ids = {annotation_source[uid] for uid in val_annotation_ids}
        if train_source_ids & val_source_ids:
            raise RuntimeError("formal crop train/val source-image overlap")
        train_groups = {
            cv_by_path[source_path_by_id[source_id]].group_id
            for source_id in train_source_ids
        }
        val_groups = {
            cv_by_path[source_path_by_id[source_id]].group_id
            for source_id in val_source_ids
        }
        if train_groups & val_groups:
            raise RuntimeError("formal crop train/val source-group overlap")

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "formal_crop_manifest.csv"
    _write_csv_atomic(
        manifest_path,
        fieldnames=output_fields,
        rows=output_rows,
    )
    output_sha256 = sha256_file(manifest_path)

    # Re-read the written file; compare preserved fields, crop IDs, active
    # assignments, and row order independently from the in-memory writer.
    written_fields, written_rows = _load_exploratory_rows(manifest_path)
    if written_fields != output_fields or len(written_rows) != len(output_rows):
        raise RuntimeError("written formal crop schema/row count changed")
    if _canonical_rows_fingerprint(
        written_rows,
        preserved_fields,
    ) != input_preserved_fingerprint:
        raise RuntimeError("written formal crop changed preserved P0-2 fields")
    for source, expected, written in zip(input_rows, output_rows, written_rows):
        for field in HISTORICAL_ASSIGNMENT_COLUMNS:
            historical_field = f"historical_p02_{field}"
            if written[historical_field] != _historical_value(source, field):
                raise RuntimeError(
                    f"written formal crop changed {historical_field}"
                )
        for field in (
            "crop_id",
            "annotation_uid",
            "source_image_id",
            "source_relative_path",
            "fold",
            "group_id",
            "leakage_group_id",
            "group_rule",
            "formal_image_id",
            "formal_cv3_sha256",
        ):
            if expected[field] != written[field]:
                raise RuntimeError(f"written formal crop changed {field}")

    fold_summary = []
    for fold in range(3):
        tight_key = (contract["expected_policies"][0], fold)
        fold_summary.append(
            {
                "fold": fold,
                "objects_per_policy": policy_fold_objects[tight_key],
                "source_images": len(policy_fold_sources[tight_key]),
                "source_groups": len(policy_fold_groups[tight_key]),
                "class_boxes": {
                    str(class_id): policy_fold_classes[tight_key][class_id]
                    for class_id in range(25)
                },
            }
        )
    audit = {
        "status": "formal_crop_manifest_v2_ready",
        "formal_crop_admission": True,
        "manifest_version": contract["version"],
        "schema_version": contract["schema_version"],
        "manifest": str(manifest_path),
        "manifest_sha256": output_sha256,
        "source_crop_manifest": str(source_path),
        "source_crop_manifest_sha256": source_sha256,
        "formal_cv3_manifest": str(formal_cv3.path),
        "formal_cv3_manifest_sha256": formal_cv3.sha256,
        "source_rows": len(input_rows),
        "source_annotations": len(annotations),
        "source_images": len(source_path_by_id),
        "policies": list(contract["expected_policies"]),
        "active_assignment_columns": [
            "fold",
            "group_id",
            "leakage_group_id",
            "group_rule",
        ],
        "historical_assignment_prefix": "historical_p02_",
        "historical_assignment_columns": sorted(HISTORICAL_ASSIGNMENT_COLUMNS),
        "historical_assignment_fields_preserved": True,
        "crop_id_preserved": True,
        "non_assignment_fields_preserved": True,
        "preserved_field_count": len(preserved_fields),
        "preserved_fields_sha256": input_preserved_fingerprint,
        "geometry_recomputed": False,
        "pixels_read": 0,
        "pixel_io_policy": "forbidden_metadata_only",
        "formal_fold_summary": fold_summary,
        "held_out_rule": "fold == held_out_fold is val; all other folds are train",
        "train_val_annotation_overlap": 0,
        "train_val_source_image_overlap": 0,
        "train_val_source_group_overlap": 0,
    }
    _atomic_write_json(output / "formal_crop_audit.json", audit)
    return audit
