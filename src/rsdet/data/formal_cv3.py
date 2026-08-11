"""Strict consumer for the frozen formal CV3 image manifest.

This module does not solve or mutate a split.  It verifies one frozen manifest
against caller-provided fingerprints and counts, then exposes the only valid
runtime interpretation:

``fold == held_out_fold`` is validation and every other fold is training.

The implementation intentionally uses only the standard library so members can
run the common data gate before installing a model framework.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EXPECTED_SHA256_LENGTH = 64
ALLOWED_SPLITS = frozenset({"train", "val"})


@dataclass(frozen=True)
class FormalCV3Sample:
    """One immutable source-image assignment from a formal CV3 manifest."""

    image_id: int
    relative_path: str
    fold: int
    group_id: str
    group_rule: str


@dataclass(frozen=True)
class FormalCV3View:
    """One held-out-fold train/validation view."""

    held_out_fold: int
    train: tuple[FormalCV3Sample, ...]
    val: tuple[FormalCV3Sample, ...]

    def records(self, split: str) -> tuple[FormalCV3Sample, ...]:
        """Return the requested partition after validating its name."""

        if split not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}")
        return self.train if split == "train" else self.val


@dataclass(frozen=True)
class FormalCV3Manifest:
    """A fully audited formal CV3 manifest."""

    path: Path
    sha256: str
    version: str
    data_version: str
    fold_count: int
    samples: tuple[FormalCV3Sample, ...]
    fold_image_counts: tuple[int, ...]
    fold_group_counts: tuple[int, ...]
    source_group_count: int

    def view(self, held_out_fold: int) -> FormalCV3View:
        """Build and re-audit the runtime split for one held-out fold."""

        if type(held_out_fold) is not int or not 0 <= held_out_fold < self.fold_count:
            raise ValueError(
                f"held_out_fold must be in 0..{self.fold_count - 1}, found {held_out_fold!r}"
            )
        train = tuple(item for item in self.samples if item.fold != held_out_fold)
        val = tuple(item for item in self.samples if item.fold == held_out_fold)
        if not train or not val or len(train) + len(val) != len(self.samples):
            raise ValueError("held-out view does not cover the complete manifest")

        train_ids = {item.image_id for item in train}
        val_ids = {item.image_id for item in val}
        train_paths = {item.relative_path for item in train}
        val_paths = {item.relative_path for item in val}
        train_groups = {item.group_id for item in train}
        val_groups = {item.group_id for item in val}
        overlaps = {
            "image_id": train_ids & val_ids,
            "relative_path": train_paths & val_paths,
            "group_id": train_groups & val_groups,
        }
        leaking = {name: values for name, values in overlaps.items() if values}
        if leaking:
            examples = {name: sorted(values, key=str)[:3] for name, values in leaking.items()}
            raise ValueError(f"formal CV3 train/val leakage: {examples}")
        if len(val) != self.fold_image_counts[held_out_fold]:
            raise ValueError(
                f"fold {held_out_fold} validation count changed: "
                f"{len(val)} != {self.fold_image_counts[held_out_fold]}"
            )
        if len(val_groups) != self.fold_group_counts[held_out_fold]:
            raise ValueError(
                f"fold {held_out_fold} validation group count changed: "
                f"{len(val_groups)} != {self.fold_group_counts[held_out_fold]}"
            )
        return FormalCV3View(
            held_out_fold=held_out_fold,
            train=train,
            val=val,
        )


def sha256_file(path: str | Path) -> str:
    """Return the streaming SHA256 of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_expected_sha256(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != EXPECTED_SHA256_LENGTH or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError("expected_sha256 must be a 64-character lowercase hex digest")
    return normalized


def _safe_relative_path(value: Any, *, index: int) -> str:
    normalized = str(value).strip().replace("\\", "/")
    path = Path(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"sample {index} contains unsafe relative_path={value!r}")
    return normalized


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, found {value!r}")
    return value


def _expected_count_tuple(
    values: Sequence[int],
    *,
    fold_count: int,
    name: str,
) -> tuple[int, ...]:
    result = tuple(_positive_int(value, f"{name}[{index}]") for index, value in enumerate(values))
    if len(result) != fold_count:
        raise ValueError(f"{name} must contain {fold_count} values")
    return result


def load_formal_cv3_manifest(
    manifest_path: str | Path,
    *,
    expected_sha256: str,
    expected_version: str,
    expected_data_version: str,
    expected_fold_count: int,
    expected_sample_count: int,
    expected_group_count: int,
    expected_fold_image_counts: Sequence[int],
    expected_fold_group_counts: Sequence[int],
) -> FormalCV3Manifest:
    """Load a frozen CV3 manifest only after all identity and leakage gates pass."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_digest = _validate_expected_sha256(expected_sha256)
    actual_digest = sha256_file(path)
    if actual_digest != expected_digest:
        raise ValueError(
            f"formal CV3 SHA256 mismatch: expected={expected_digest}, actual={actual_digest}"
        )
    fold_count = _positive_int(expected_fold_count, "expected_fold_count")
    sample_count = _positive_int(expected_sample_count, "expected_sample_count")
    group_count = _positive_int(expected_group_count, "expected_group_count")
    image_count_contract = _expected_count_tuple(
        expected_fold_image_counts,
        fold_count=fold_count,
        name="expected_fold_image_counts",
    )
    group_count_contract = _expected_count_tuple(
        expected_fold_group_counts,
        fold_count=fold_count,
        name="expected_fold_group_counts",
    )
    if sum(image_count_contract) != sample_count:
        raise ValueError("expected fold image counts do not sum to expected_sample_count")
    if sum(group_count_contract) != group_count:
        raise ValueError("expected fold group counts do not sum to expected_group_count")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("formal CV3 manifest root must be an object")
    if payload.get("version") != expected_version:
        raise ValueError(
            f"formal CV3 version mismatch: {payload.get('version')!r} != {expected_version!r}"
        )
    if payload.get("data_version") != expected_data_version:
        raise ValueError(
            f"formal CV3 data_version mismatch: {payload.get('data_version')!r} "
            f"!= {expected_data_version!r}"
        )
    if payload.get("fold_count") != fold_count:
        raise ValueError(
            f"formal CV3 fold_count mismatch: {payload.get('fold_count')!r} != {fold_count}"
        )
    scientific_contract = payload.get("scientific_contract")
    if not isinstance(scientific_contract, Mapping):
        raise ValueError("formal CV3 manifest is missing scientific_contract")
    for key in (
        "group_indivisible",
        "every_image_validates_exactly_once",
        "every_class_has_validation_source_in_every_fold",
    ):
        if scientific_contract.get(key) is not True:
            raise ValueError(f"formal CV3 scientific contract does not assert {key}")

    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != sample_count:
        found = len(raw_samples) if isinstance(raw_samples, list) else type(raw_samples).__name__
        raise ValueError(f"formal CV3 sample count mismatch: {found} != {sample_count}")

    samples: list[FormalCV3Sample] = []
    seen_ids: set[int] = set()
    seen_paths: set[str] = set()
    folds_by_group: defaultdict[str, set[int]] = defaultdict(set)
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, Mapping):
            raise TypeError(f"sample {index} must be an object")
        missing = {
            "image_id",
            "relative_path",
            "fold",
            "group_id",
            "group_rule",
        } - set(raw)
        if missing:
            raise ValueError(f"sample {index} is missing fields: {sorted(missing)}")
        image_id = raw["image_id"]
        if type(image_id) is not int or image_id <= 0:
            raise ValueError(f"sample {index} has invalid image_id={image_id!r}")
        if image_id in seen_ids:
            raise ValueError(f"formal CV3 contains duplicate image_id={image_id}")
        relative_path = _safe_relative_path(raw["relative_path"], index=index)
        if relative_path in seen_paths:
            raise ValueError(f"formal CV3 contains duplicate path={relative_path}")
        fold = raw["fold"]
        if type(fold) is not int or not 0 <= fold < fold_count:
            raise ValueError(f"sample {index} has invalid fold={fold!r}")
        group_id = str(raw["group_id"]).strip()
        group_rule = str(raw["group_rule"]).strip()
        if not group_id or not group_rule:
            raise ValueError(f"sample {index} has an empty source-group field")
        samples.append(
            FormalCV3Sample(
                image_id=image_id,
                relative_path=relative_path,
                fold=fold,
                group_id=group_id,
                group_rule=group_rule,
            )
        )
        seen_ids.add(image_id)
        seen_paths.add(relative_path)
        folds_by_group[group_id].add(fold)

    expected_ids = set(range(1, sample_count + 1))
    if seen_ids != expected_ids:
        raise ValueError(
            "formal CV3 image_id coverage mismatch: "
            f"missing={sorted(expected_ids - seen_ids)[:5]}, "
            f"extra={sorted(seen_ids - expected_ids)[:5]}"
        )
    crossing = {
        group_id: sorted(folds) for group_id, folds in folds_by_group.items() if len(folds) != 1
    }
    if crossing:
        raise ValueError(
            f"formal CV3 has {len(crossing)} cross-fold groups; "
            f"examples={list(crossing.items())[:3]}"
        )
    if len(folds_by_group) != group_count:
        raise ValueError(
            f"formal CV3 source-group count mismatch: {len(folds_by_group)} != {group_count}"
        )

    fold_images = Counter(item.fold for item in samples)
    fold_groups = Counter(next(iter(folds)) for folds in folds_by_group.values())
    actual_fold_images = tuple(fold_images[fold] for fold in range(fold_count))
    actual_fold_groups = tuple(fold_groups[fold] for fold in range(fold_count))
    if actual_fold_images != image_count_contract:
        raise ValueError(
            f"formal CV3 fold image counts changed: {actual_fold_images} != {image_count_contract}"
        )
    if actual_fold_groups != group_count_contract:
        raise ValueError(
            f"formal CV3 fold group counts changed: {actual_fold_groups} != {group_count_contract}"
        )

    result = FormalCV3Manifest(
        path=path,
        sha256=actual_digest,
        version=expected_version,
        data_version=expected_data_version,
        fold_count=fold_count,
        samples=tuple(sorted(samples, key=lambda item: item.image_id)),
        fold_image_counts=actual_fold_images,
        fold_group_counts=actual_fold_groups,
        source_group_count=len(folds_by_group),
    )
    for held_out_fold in range(fold_count):
        result.view(held_out_fold)
    return result


def formal_cv3_audit_payload(manifest: FormalCV3Manifest) -> dict[str, Any]:
    """Build a deterministic machine-readable runtime audit."""

    views = []
    for held_out_fold in range(manifest.fold_count):
        view = manifest.view(held_out_fold)
        views.append(
            {
                "held_out_fold": held_out_fold,
                "train_images": len(view.train),
                "val_images": len(view.val),
                "train_groups": len({item.group_id for item in view.train}),
                "val_groups": len({item.group_id for item in view.val}),
                "image_overlap": 0,
                "path_overlap": 0,
                "group_overlap": 0,
            }
        )
    return {
        "status": "formal_cv3_consumer_pass",
        "formal_cv3_admission": True,
        "manifest": str(manifest.path),
        "manifest_version": manifest.version,
        "manifest_sha256": manifest.sha256,
        "data_version": manifest.data_version,
        "sample_count": len(manifest.samples),
        "source_group_count": manifest.source_group_count,
        "fold_count": manifest.fold_count,
        "fold_image_counts": list(manifest.fold_image_counts),
        "fold_group_counts": list(manifest.fold_group_counts),
        "group_cross_fold_count": 0,
        "views": views,
    }


def write_formal_cv3_views(
    manifest: FormalCV3Manifest,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write six deterministic CSV views after the complete consumer gate passes."""

    output = Path(output_dir).expanduser().resolve()
    expected_paths = tuple(
        output / f"formal_cv3_fold{held_out_fold}_{split}.csv"
        for held_out_fold in range(manifest.fold_count)
        for split in ("train", "val")
    )
    existing = [path for path in expected_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "formal CV3 outputs are immutable; rerun into a fresh output directory: "
            + ", ".join(str(path) for path in existing[:3])
        )
    output.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    fieldnames = [
        "image_id",
        "relative_path",
        "group_id",
        "group_rule",
        "fold",
        "held_out_fold",
        "split",
        "formal_cv3_version",
        "formal_cv3_sha256",
    ]
    for held_out_fold in range(manifest.fold_count):
        view = manifest.view(held_out_fold)
        for split in ("train", "val"):
            rows = view.records(split)
            path = output / f"formal_cv3_fold{held_out_fold}_{split}.csv"
            descriptor, temporary_name = tempfile.mkstemp(
                dir=output,
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for item in rows:
                        writer.writerow(
                            {
                                "image_id": item.image_id,
                                "relative_path": item.relative_path,
                                "group_id": item.group_id,
                                "group_rule": item.group_rule,
                                "fold": item.fold,
                                "held_out_fold": held_out_fold,
                                "split": split,
                                "formal_cv3_version": manifest.version,
                                "formal_cv3_sha256": manifest.sha256,
                            }
                        )
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(path)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            hashes[path.name] = sha256_file(path)
    return dict(sorted(hashes.items()))


def validate_written_view(
    path: str | Path,
    *,
    manifest: FormalCV3Manifest,
    held_out_fold: int,
    split: str,
) -> None:
    """Independently re-read one generated CSV and verify its frozen membership."""

    expected = manifest.view(held_out_fold).records(split)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    expected_ids = [item.image_id for item in expected]
    actual_ids = [int(row["image_id"]) for row in rows]
    if actual_ids != expected_ids:
        raise ValueError(f"written {split}/fold{held_out_fold} image membership changed")
    if any(
        int(row["held_out_fold"]) != held_out_fold
        or row["split"] != split
        or row["formal_cv3_sha256"] != manifest.sha256
        for row in rows
    ):
        raise ValueError(f"written {split}/fold{held_out_fold} metadata changed")


def finite_fraction(value: Any, *, name: str) -> float:
    """Parse a finite unit fraction for small config/CLI callers."""

    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return result


def groups_by_fold(samples: Iterable[FormalCV3Sample]) -> dict[int, set[str]]:
    """Return source-group sets, mainly for downstream audits."""

    result: defaultdict[int, set[str]] = defaultdict(set)
    for sample in samples:
        result[sample.fold].add(sample.group_id)
    return dict(result)
