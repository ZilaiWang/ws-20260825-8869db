"""Immutable byte-level lock for the formal CV3 detection dataset.

The formal split manifest freezes *which* source image belongs to each fold.
This module freezes the complementary fact that formal detection experiments
consume the same image and YOLO-label bytes.  It also proves that the YOLO
labels describe exactly the GT boxes inherited from P0-2 and the formal crop
manifest.

Only relative paths and deterministic measurements enter the lock.  Absolute
server paths, timestamps and host information are intentionally excluded so an
identical dataset produces byte-identical locks on different machines.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from rsdet.data.formal_cv3 import load_formal_cv3_manifest

SPEC_VERSION = "formal_detection_data_lock_spec_v1"
LOCK_VERSION = "formal_detection_data_lock_v1"
HEX_DIGITS = frozenset("0123456789abcdef")
GT_EQUIVALENCE_FIELDS = (
    "annotation_uid",
    "source_image_id",
    "source_relative_path",
    "source_width",
    "source_height",
    "source_checksum_sha256",
    "class_id",
    "class_name",
    "major_class",
    "modality",
    "gt_x0",
    "gt_y0",
    "gt_x1",
    "gt_y1",
    "gt_width",
    "gt_height",
    "gt_short_edge",
    "crop_policy",
)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    """Hash a JSON-compatible value using the repository's canonical encoding."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value: Any, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(char not in HEX_DIGITS for char in normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _strict_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{name} fields differ: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def load_and_validate_spec(path: str | Path) -> dict[str, Any]:
    """Load the reviewed scientific contract for the byte lock."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("data-lock spec root must be an object")
    _strict_keys(
        payload,
        {
            "schema_version",
            "cv3_manifest",
            "p02_manifest",
            "formal_crop_manifest",
            "dataset",
        },
        "data-lock spec",
    )
    if payload["schema_version"] != SPEC_VERSION:
        raise ValueError(f"schema_version must be {SPEC_VERSION}")

    cv3 = payload["cv3_manifest"]
    if not isinstance(cv3, dict):
        raise TypeError("cv3_manifest must be an object")
    _strict_keys(
        cv3,
        {
            "sha256",
            "version",
            "data_version",
            "fold_count",
            "sample_count",
            "source_group_count",
            "fold_image_counts",
            "fold_group_counts",
        },
        "cv3_manifest",
    )
    cv3["sha256"] = _sha256(cv3["sha256"], "cv3_manifest.sha256")
    for key in ("fold_count", "sample_count", "source_group_count"):
        cv3[key] = _positive_int(cv3[key], f"cv3_manifest.{key}")
    for key in ("fold_image_counts", "fold_group_counts"):
        values = cv3[key]
        if not isinstance(values, list) or len(values) != cv3["fold_count"]:
            raise ValueError(f"cv3_manifest.{key} must contain fold_count values")
        cv3[key] = [
            _positive_int(value, f"cv3_manifest.{key}[{index}]")
            for index, value in enumerate(values)
        ]

    for section in ("p02_manifest", "formal_crop_manifest"):
        item = payload[section]
        if not isinstance(item, dict):
            raise TypeError(f"{section} must be an object")
        _strict_keys(item, {"sha256", "row_count", "annotation_count"}, section)
        item["sha256"] = _sha256(item["sha256"], f"{section}.sha256")
        item["row_count"] = _positive_int(item["row_count"], f"{section}.row_count")
        item["annotation_count"] = _positive_int(
            item["annotation_count"], f"{section}.annotation_count"
        )

    dataset = payload["dataset"]
    if not isinstance(dataset, dict):
        raise TypeError("dataset must be an object")
    _strict_keys(
        dataset,
        {
            "image_prefix",
            "label_prefix",
            "class_count",
            "object_count",
            "crop_policies",
            "tight_policy",
            "coordinate_tolerance",
        },
        "dataset",
    )
    for key in ("image_prefix", "label_prefix", "tight_policy"):
        value = dataset[key]
        if not isinstance(value, str) or not value or Path(value).name != value:
            raise ValueError(f"dataset.{key} must be one safe path component")
    if dataset["image_prefix"] == dataset["label_prefix"]:
        raise ValueError("image_prefix and label_prefix must differ")
    dataset["class_count"] = _positive_int(dataset["class_count"], "dataset.class_count")
    dataset["object_count"] = _positive_int(dataset["object_count"], "dataset.object_count")
    policies = dataset["crop_policies"]
    if (
        not isinstance(policies, list)
        or not policies
        or any(not isinstance(value, str) or not value for value in policies)
        or len(set(policies)) != len(policies)
    ):
        raise ValueError("dataset.crop_policies must be a unique non-empty string list")
    if dataset["tight_policy"] not in policies:
        raise ValueError("dataset.tight_policy must occur in crop_policies")
    tolerance = dataset["coordinate_tolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (float, int))
        or not math.isfinite(float(tolerance))
        or not 0 < float(tolerance) <= 1e-3
    ):
        raise ValueError("dataset.coordinate_tolerance must be in (0, 1e-3]")
    dataset["coordinate_tolerance"] = float(tolerance)
    if payload["p02_manifest"]["row_count"] != payload["formal_crop_manifest"]["row_count"]:
        raise ValueError("P0-2 and formal crop row counts must match")
    annotation_counts = {
        payload["p02_manifest"]["annotation_count"],
        payload["formal_crop_manifest"]["annotation_count"],
        dataset["object_count"],
    }
    if len(annotation_counts) != 1:
        raise ValueError("annotation/object count contract differs")
    return payload


def _safe_resolve(root: Path, relative_path: str) -> Path:
    relative = Path(str(relative_path).replace("\\", "/"))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe dataset relative path: {relative_path!r}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"dataset path escapes root: {relative_path!r}") from error
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _label_relative_path(
    image_relative_path: str,
    *,
    image_prefix: str,
    label_prefix: str,
) -> str:
    image = Path(image_relative_path)
    if not image.parts or image.parts[0] != image_prefix:
        raise ValueError(
            f"image path must begin with {image_prefix!r}: {image_relative_path!r}"
        )
    label = Path(label_prefix, *image.parts[1:]).with_suffix(".txt")
    return label.as_posix()


def _load_crop_manifest(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    expected_annotations: int,
    expected_policies: Sequence[str],
    name: str,
) -> dict[str, dict[str, str]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{name} SHA-256 mismatch")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        required = {"crop_id", *GT_EQUIVALENCE_FIELDS}
        missing = required - columns
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        rows: dict[str, dict[str, str]] = {}
        annotation_policies: defaultdict[str, set[str]] = defaultdict(set)
        for index, row in enumerate(reader, start=2):
            crop_id = row["crop_id"].strip()
            if not crop_id or crop_id in rows:
                raise ValueError(f"{name}:{index} has empty/duplicate crop_id={crop_id!r}")
            if not row["annotation_uid"].strip():
                raise ValueError(f"{name}:{index} has empty annotation_uid")
            rows[crop_id] = row
            annotation_policies[row["annotation_uid"]].add(row["crop_policy"])
    if len(rows) != expected_rows:
        raise ValueError(f"{name} row count mismatch: {len(rows)} != {expected_rows}")
    if len(annotation_policies) != expected_annotations:
        raise ValueError(
            f"{name} annotation count mismatch: "
            f"{len(annotation_policies)} != {expected_annotations}"
        )
    wanted_policies = set(expected_policies)
    bad = [
        (annotation_uid, sorted(policies))
        for annotation_uid, policies in annotation_policies.items()
        if policies != wanted_policies
    ]
    if bad:
        raise ValueError(f"{name} per-annotation policy coverage differs: {bad[:3]}")
    return rows


def _prove_p02_formal_equivalence(
    p02_rows: Mapping[str, Mapping[str, str]],
    formal_rows: Mapping[str, Mapping[str, str]],
    *,
    p02_sha256: str,
) -> None:
    if set(p02_rows) != set(formal_rows):
        missing = sorted(set(p02_rows) - set(formal_rows))[:3]
        extra = sorted(set(formal_rows) - set(p02_rows))[:3]
        raise ValueError(f"P0-2/formal crop IDs differ: missing={missing}, extra={extra}")
    for crop_id in sorted(p02_rows):
        historical = p02_rows[crop_id]
        formal = formal_rows[crop_id]
        differences = [
            key
            for key in GT_EQUIVALENCE_FIELDS
            if historical[key] != formal[key]
        ]
        if differences:
            raise ValueError(
                f"P0-2/formal GT differs for crop_id={crop_id}: {differences}"
            )
        if formal.get("source_crop_manifest_sha256") != p02_sha256:
            raise ValueError(
                f"formal crop does not bind P0-2 SHA for crop_id={crop_id}"
            )


def _parse_yolo_label(
    path: Path,
    *,
    class_count: int,
    tolerance: float,
) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            raise ValueError(f"{path}:{line_number} contains a blank line")
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number} must contain exactly five fields")
        try:
            class_id = int(fields[0])
            values = tuple(float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"{path}:{line_number} is not valid YOLO text") from error
        if not 0 <= class_id < class_count:
            raise ValueError(f"{path}:{line_number} class_id is outside the frozen label set")
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number} contains NaN/Inf")
        cx, cy, width, height = values
        if (
            not -tolerance <= cx <= 1 + tolerance
            or not -tolerance <= cy <= 1 + tolerance
            or not 0 < width <= 1 + tolerance
            or not 0 < height <= 1 + tolerance
            or cx - width / 2 < -tolerance
            or cy - height / 2 < -tolerance
            or cx + width / 2 > 1 + tolerance
            or cy + height / 2 > 1 + tolerance
        ):
            raise ValueError(f"{path}:{line_number} box lies outside normalized image bounds")
        boxes.append((class_id, cx, cy, width, height))
    if not boxes:
        raise ValueError(f"{path} contains no objects")
    return boxes


def _expected_yolo_box(row: Mapping[str, str]) -> tuple[int, float, float, float, float]:
    source_width = float(row["source_width"])
    source_height = float(row["source_height"])
    x0, y0, x1, y1 = (
        float(row["gt_x0"]),
        float(row["gt_y0"]),
        float(row["gt_x1"]),
        float(row["gt_y1"]),
    )
    if (
        not all(math.isfinite(value) for value in (source_width, source_height, x0, y0, x1, y1))
        or source_width <= 0
        or source_height <= 0
        or not x0 < x1
        or not y0 < y1
    ):
        raise ValueError(f"invalid GT geometry for annotation {row['annotation_uid']}")
    return (
        int(row["class_id"]),
        (x0 + x1) / (2 * source_width),
        (y0 + y1) / (2 * source_height),
        (x1 - x0) / source_width,
        (y1 - y0) / source_height,
    )


def _match_boxes(
    actual: Sequence[tuple[int, float, float, float, float]],
    expected: Sequence[tuple[int, float, float, float, float]],
    *,
    tolerance: float,
    image_relative_path: str,
) -> float:
    if len(actual) != len(expected):
        raise ValueError(
            f"YOLO/formal object count differs for {image_relative_path}: "
            f"{len(actual)} != {len(expected)}"
        )
    unmatched = list(actual)
    maximum_error = 0.0
    for wanted in expected:
        candidates = [
            (
                max(abs(wanted[index] - found[index]) for index in range(1, 5)),
                index,
            )
            for index, found in enumerate(unmatched)
            if found[0] == wanted[0]
        ]
        if not candidates:
            raise ValueError(
                f"YOLO/formal class multiset differs for {image_relative_path}"
            )
        error, index = min(candidates)
        if error > tolerance:
            raise ValueError(
                f"YOLO/formal coordinates differ for {image_relative_path}: "
                f"max_abs_error={error:.12g} > {tolerance}"
            )
        maximum_error = max(maximum_error, error)
        unmatched.pop(index)
    return maximum_error


def _build_payload(
    *,
    spec: Mapping[str, Any],
    data_root: Path,
    cv3_manifest_path: Path,
    p02_manifest_path: Path,
    formal_crop_manifest_path: Path,
) -> dict[str, Any]:
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)
    cv3_contract = spec["cv3_manifest"]
    cv3 = load_formal_cv3_manifest(
        cv3_manifest_path,
        expected_sha256=cv3_contract["sha256"],
        expected_version=cv3_contract["version"],
        expected_data_version=cv3_contract["data_version"],
        expected_fold_count=cv3_contract["fold_count"],
        expected_sample_count=cv3_contract["sample_count"],
        expected_group_count=cv3_contract["source_group_count"],
        expected_fold_image_counts=cv3_contract["fold_image_counts"],
        expected_fold_group_counts=cv3_contract["fold_group_counts"],
    )
    dataset = spec["dataset"]
    p02_contract = spec["p02_manifest"]
    formal_contract = spec["formal_crop_manifest"]
    p02_rows = _load_crop_manifest(
        p02_manifest_path,
        expected_sha256=p02_contract["sha256"],
        expected_rows=p02_contract["row_count"],
        expected_annotations=p02_contract["annotation_count"],
        expected_policies=dataset["crop_policies"],
        name="P0-2 crop manifest",
    )
    formal_rows = _load_crop_manifest(
        formal_crop_manifest_path,
        expected_sha256=formal_contract["sha256"],
        expected_rows=formal_contract["row_count"],
        expected_annotations=formal_contract["annotation_count"],
        expected_policies=dataset["crop_policies"],
        name="formal crop manifest",
    )
    _prove_p02_formal_equivalence(
        p02_rows,
        formal_rows,
        p02_sha256=p02_contract["sha256"],
    )

    tight_rows: defaultdict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in formal_rows.values():
        if row["crop_policy"] == dataset["tight_policy"]:
            tight_rows[row["source_relative_path"]].append(row)
    if sum(map(len, tight_rows.values())) != dataset["object_count"]:
        raise ValueError("tight-policy object count differs from the frozen contract")
    cv3_paths = {sample.relative_path for sample in cv3.samples}
    if set(tight_rows) != cv3_paths:
        missing = sorted(cv3_paths - set(tight_rows))[:3]
        extra = sorted(set(tight_rows) - cv3_paths)[:3]
        raise ValueError(f"CV3/formal image coverage differs: missing={missing}, extra={extra}")

    inventory: list[dict[str, Any]] = []
    total_image_bytes = 0
    total_label_bytes = 0
    total_objects = 0
    class_counts: Counter[int] = Counter()
    maximum_coordinate_error = 0.0
    for sample in sorted(cv3.samples, key=lambda item: item.image_id):
        image_relative = sample.relative_path
        label_relative = _label_relative_path(
            image_relative,
            image_prefix=dataset["image_prefix"],
            label_prefix=dataset["label_prefix"],
        )
        image_path = _safe_resolve(data_root, image_relative)
        label_path = _safe_resolve(data_root, label_relative)
        source_rows = tight_rows[image_relative]
        source_widths = {int(row["source_width"]) for row in source_rows}
        source_heights = {int(row["source_height"]) for row in source_rows}
        source_checksums = {row["source_checksum_sha256"] for row in source_rows}
        if len(source_widths) != 1 or len(source_heights) != 1 or len(source_checksums) != 1:
            raise ValueError(f"formal source metadata conflicts for {image_relative}")
        expected_width = next(iter(source_widths))
        expected_height = next(iter(source_heights))
        image_sha = sha256_file(image_path)
        if image_sha != next(iter(source_checksums)):
            raise ValueError(f"image bytes differ from formal/P0-2 checksum: {image_relative}")
        with Image.open(image_path) as image:
            actual_width, actual_height = image.size
            image_format = image.format
        if (actual_width, actual_height) != (expected_width, expected_height):
            raise ValueError(
                f"image dimensions differ for {image_relative}: "
                f"{(actual_width, actual_height)} != {(expected_width, expected_height)}"
            )

        boxes = _parse_yolo_label(
            label_path,
            class_count=dataset["class_count"],
            tolerance=dataset["coordinate_tolerance"],
        )
        expected_boxes = [_expected_yolo_box(row) for row in source_rows]
        coordinate_error = _match_boxes(
            boxes,
            expected_boxes,
            tolerance=dataset["coordinate_tolerance"],
            image_relative_path=image_relative,
        )
        maximum_coordinate_error = max(maximum_coordinate_error, coordinate_error)
        class_counts.update(box[0] for box in boxes)
        image_size = image_path.stat().st_size
        label_size = label_path.stat().st_size
        total_image_bytes += image_size
        total_label_bytes += label_size
        total_objects += len(boxes)
        inventory.append(
            {
                "image_id": sample.image_id,
                "fold": sample.fold,
                "group_id": sample.group_id,
                "image": {
                    "relative_path": image_relative,
                    "size_bytes": image_size,
                    "sha256": image_sha,
                    "width": actual_width,
                    "height": actual_height,
                    "format": image_format,
                },
                "label": {
                    "relative_path": label_relative,
                    "size_bytes": label_size,
                    "sha256": sha256_file(label_path),
                    "object_count": len(boxes),
                },
            }
        )
    if total_objects != dataset["object_count"]:
        raise ValueError(f"YOLO object count mismatch: {total_objects}")
    if set(class_counts) != set(range(dataset["class_count"])):
        raise ValueError(
            f"YOLO class coverage differs: {sorted(class_counts)}"
        )

    contract = {
        "cv3_manifest_sha256": cv3_contract["sha256"],
        "p02_manifest_sha256": p02_contract["sha256"],
        "formal_crop_manifest_sha256": formal_contract["sha256"],
        "cv3_version": cv3_contract["version"],
        "data_version": cv3_contract["data_version"],
        "crop_policies": list(dataset["crop_policies"]),
        "tight_policy": dataset["tight_policy"],
        "class_count": dataset["class_count"],
        "coordinate_tolerance": dataset["coordinate_tolerance"],
        "gt_equivalence_fields": list(GT_EQUIVALENCE_FIELDS),
    }
    summary = {
        "image_count": len(inventory),
        "label_file_count": len(inventory),
        "object_count": total_objects,
        "class_counts": {str(key): class_counts[key] for key in sorted(class_counts)},
        "fold_image_counts": list(cv3.fold_image_counts),
        "source_group_count": cv3.source_group_count,
        "total_image_bytes": total_image_bytes,
        "total_label_bytes": total_label_bytes,
        "maximum_yolo_coordinate_error": maximum_coordinate_error,
        "p02_formal_gt_equivalence": True,
        "yolo_formal_gt_equivalence": True,
    }
    body: dict[str, Any] = {
        "schema_version": LOCK_VERSION,
        "contract": contract,
        "summary": summary,
        "inventory_fingerprint": canonical_json_sha256(inventory),
        "inventory": inventory,
    }
    body["lock_fingerprint"] = canonical_json_sha256(body)
    return body


def build_lock_payload(
    *,
    spec_path: str | Path,
    data_root: str | Path,
    cv3_manifest_path: str | Path,
    p02_manifest_path: str | Path,
    formal_crop_manifest_path: str | Path,
) -> dict[str, Any]:
    """Build and fully audit a deterministic formal detection-data lock."""

    spec = load_and_validate_spec(spec_path)
    return _build_payload(
        spec=spec,
        data_root=Path(data_root),
        cv3_manifest_path=Path(cv3_manifest_path).expanduser().resolve(),
        p02_manifest_path=Path(p02_manifest_path).expanduser().resolve(),
        formal_crop_manifest_path=Path(formal_crop_manifest_path).expanduser().resolve(),
    )


def atomic_write_new_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically create a read-only JSON artifact and refuse replacement."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable data lock: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        try:
            # A hard-link publication is an atomic no-clobber operation on the
            # same filesystem.  Unlike a second exists() check followed by
            # os.replace(), it has no race window in which another process
            # could create a lock and then have it overwritten.
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite immutable data lock: {target}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def verify_existing_lock(
    *,
    spec_path: str | Path,
    data_root: str | Path,
    cv3_manifest_path: str | Path,
    p02_manifest_path: str | Path,
    formal_crop_manifest_path: str | Path,
    lock_path: str | Path,
    expected_lock_sha256: str | None = None,
) -> dict[str, Any]:
    """Re-hash all inputs and require exact equality with an immutable lock."""

    lock_path = Path(lock_path).expanduser().resolve()
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    if lock_path.stat().st_mode & 0o222:
        raise PermissionError(f"data lock has write permission: {lock_path}")
    actual_lock_sha = sha256_file(lock_path)
    if expected_lock_sha256 is not None:
        wanted = _sha256(expected_lock_sha256, "expected_lock_sha256")
        if actual_lock_sha != wanted:
            raise ValueError(
                f"data-lock file SHA-256 mismatch: {actual_lock_sha} != {wanted}"
            )
    observed = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(observed, dict):
        raise TypeError("data lock root must be an object")
    fingerprint = observed.get("lock_fingerprint")
    body = dict(observed)
    body.pop("lock_fingerprint", None)
    if fingerprint != canonical_json_sha256(body):
        raise ValueError("data-lock internal fingerprint mismatch")
    rebuilt = build_lock_payload(
        spec_path=spec_path,
        data_root=data_root,
        cv3_manifest_path=cv3_manifest_path,
        p02_manifest_path=p02_manifest_path,
        formal_crop_manifest_path=formal_crop_manifest_path,
    )
    if rebuilt != observed:
        raise ValueError("live detection data does not reproduce the immutable lock")
    return {
        "status": "pass",
        "schema_version": observed["schema_version"],
        "lock_path": str(lock_path),
        "lock_file_sha256": actual_lock_sha,
        "lock_fingerprint": observed["lock_fingerprint"],
        "inventory_fingerprint": observed["inventory_fingerprint"],
        "image_count": observed["summary"]["image_count"],
        "label_file_count": observed["summary"]["label_file_count"],
        "object_count": observed["summary"]["object_count"],
        "p02_formal_gt_equivalence": observed["summary"]["p02_formal_gt_equivalence"],
        "yolo_formal_gt_equivalence": observed["summary"]["yolo_formal_gt_equivalence"],
    }


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a mutable verification report."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
