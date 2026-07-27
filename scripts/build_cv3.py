#!/usr/bin/env python3
"""Build the frozen, source-isolated K60 CV3 manifest.

The K60 mapping is authoritative and immutable.  Legacy dHash candidates are
accepted only for an audit record and never merge or rewrite source groups.

Example:
    PYTHONPATH=src python scripts/build_cv3.py \
        --data-root ../data \
        --airport-groups data/groups/mar20_airport_proxy_k60_for_b.csv \
        --assignment data/splits/cv3_airport_proxy_k60_v2_groups.json \
        --near-duplicates reports/data/near_duplicates_mar20.json \
        --output data/splits/cv3_airport_proxy_k60_v2.json \
        --audit-output reports/data/cv3_airport_proxy_k60_v2_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsdet.data.cv3_split import (
    EXPECTED_K60_GROUPS,
    EXPECTED_MAR20_IMAGES,
    FOLD_COUNT,
    MAR20_RE,
    CV3ImageRecord,
    aggregate_groups,
    atomic_write_json,
    audit_near_duplicates,
    build_scientific_audit,
    derive_non_mar20_group,
    group_statistics_sha256,
    label_set_sha256,
    load_airport_groups,
    load_frozen_assignment,
    make_manifest_samples,
    read_class_ids,
    sha256_file,
)
from rsdet.data.splits import DATA_VERSION
from rsdet.data.xh_dataset import FINE_NAMES, XHDataset, coarse_name

VERSION = "cv3_airport_proxy_k60_v2"
PROTOCOL = "k60_immutable_hard_coverage_frozen_assignment_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--airport-groups",
        type=Path,
        required=True,
        help="authoritative two-column MAR20 K60 mapping",
    )
    parser.add_argument(
        "--assignment",
        type=Path,
        required=True,
        help="reviewed and frozen source-group-to-fold assignment",
    )
    parser.add_argument(
        "--near-duplicates",
        type=Path,
        default=None,
        help="legacy dHash candidates, audited only and never used to merge K60",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def build_records(
    dataset: XHDataset,
    airport_groups: dict[str, str],
) -> list[CV3ImageRecord]:
    """Apply K60 directly and derive only the non-MAR20 source groups."""

    records: list[CV3ImageRecord] = []
    seen_ids: set[int] = set()
    seen_stems: set[str] = set()
    for ref in dataset.refs:
        if ref.image_id in seen_ids:
            raise ValueError(f"duplicate image_id {ref.image_id}")
        if ref.stem in seen_stems:
            raise ValueError(f"duplicate image stem {ref.stem}")
        seen_ids.add(ref.image_id)
        seen_stems.add(ref.stem)

        if ref.stem in airport_groups:
            group_id = airport_groups[ref.stem]
            group_rule = "mar20_airport_proxy_k60"
        else:
            group_id, group_rule = derive_non_mar20_group(ref.stem)

        class_ids = tuple(read_class_ids(ref.label_path))
        if not class_ids:
            raise ValueError(f"image has no labeled objects: {ref.stem}")
        if any(not 0 <= class_id < len(FINE_NAMES) for class_id in class_ids):
            raise ValueError(f"image has an invalid class ID: {ref.stem}")
        coarse_names = {coarse_name(class_id) for class_id in class_ids}
        if len(coarse_names) != 1:
            raise ValueError(f"image mixes coarse domains: {ref.stem}")
        records.append(
            CV3ImageRecord(
                image_id=ref.image_id,
                stem=ref.stem,
                relative_path=f"images/train/{ref.image_path.name}",
                group_id=group_id,
                group_rule=group_rule,
                class_ids=class_ids,
                coarse=next(iter(coarse_names)),
            )
        )
    return records


def main() -> int:
    args = parse_args()
    if args.output.resolve() == args.audit_output.resolve():
        raise ValueError("output and audit-output must be different paths")
    if not args.airport_groups.is_file():
        raise FileNotFoundError(args.airport_groups)
    if not args.assignment.is_file():
        raise FileNotFoundError(args.assignment)

    dataset = XHDataset(args.data_root, "train", load_images=False)
    mar20_stems = {
        ref.stem for ref in dataset.refs if MAR20_RE.fullmatch(ref.stem)
    }
    airport_groups = load_airport_groups(
        args.airport_groups,
        expected_stems=mar20_stems,
        expected_image_count=EXPECTED_MAR20_IMAGES,
        expected_group_count=EXPECTED_K60_GROUPS,
    )
    records = build_records(dataset, airport_groups)
    groups = aggregate_groups(records)
    statistics_sha256 = group_statistics_sha256(groups)
    assignment, assignment_metadata = load_frozen_assignment(
        args.assignment,
        expected_group_ids={group.group_id for group in groups},
        expected_group_statistics_sha256=statistics_sha256,
    )
    near_duplicate_audit = audit_near_duplicates(
        args.near_duplicates, airport_groups
    )
    scientific_audit = build_scientific_audit(
        records, groups, assignment, airport_groups
    )
    if not scientific_audit["formal_cv3_admission"]:
        print(json.dumps(scientific_audit, ensure_ascii=False, indent=2))
        raise RuntimeError(
            "formal CV3 scientific gates failed; no output files were written"
        )

    group_map_sha256 = sha256_file(args.airport_groups)
    assignment_sha256 = sha256_file(args.assignment)
    manifest = {
        "version": VERSION,
        "data_version": DATA_VERSION,
        "protocol": PROTOCOL,
        "seed": assignment_metadata["seed"],
        "fold_count": FOLD_COUNT,
        "group_semantics": (
            "MAR20 K60 airport-proxy visual domains, not ground-truth airport IDs"
        ),
        "airport_group_map": args.airport_groups.name,
        "airport_group_map_sha256": group_map_sha256,
        "frozen_assignment": args.assignment.name,
        "frozen_assignment_sha256": assignment_sha256,
        "group_statistics_sha256": statistics_sha256,
        "label_set_sha256": label_set_sha256(records),
        "legacy_near_duplicate_policy": "audit_only_never_merge_k60",
        "scientific_contract": {
            "group_indivisible": True,
            "every_image_validates_exactly_once": True,
            "every_class_has_validation_source_in_every_fold": True,
            "minimum_train_boxes": {
                str(row["class_id"]): row["minimum_train_boxes"]
                for row in scientific_audit["per_class"]
            },
            "tu160_9shot_fold_is_preregistered": True,
        },
        "samples": make_manifest_samples(records, assignment),
    }

    # All input and scientific checks finish before either formal artifact exists.
    atomic_write_json(args.output, manifest)
    manifest_sha256 = sha256_file(args.output)
    audit = {
        "status": "cv3_airport_proxy_k60_v2_ready",
        "formal_cv3_admission": True,
        "version": VERSION,
        "protocol": PROTOCOL,
        "manifest": str(args.output),
        "manifest_sha256": manifest_sha256,
        "airport_group_map_sha256": group_map_sha256,
        "frozen_assignment_sha256": assignment_sha256,
        "group_statistics_sha256": statistics_sha256,
        "label_set_sha256": manifest["label_set_sha256"],
        "assignment_metadata": {
            key: value
            for key, value in assignment_metadata.items()
            if key != "groups"
        },
        "legacy_near_duplicate_audit": near_duplicate_audit,
        "scientific_audit": scientific_audit,
    }
    atomic_write_json(args.audit_output, audit)

    print(
        json.dumps(
            {
                "status": audit["status"],
                "manifest": str(args.output),
                "manifest_sha256": manifest_sha256,
                "audit": str(args.audit_output),
                "fold_image_counts": scientific_audit["fold_image_counts"],
                "fold_box_counts": scientific_audit["fold_box_counts"],
                "fold_group_counts": scientific_audit["fold_group_counts"],
                "tu160": scientific_audit["tu160_stress_fold"],
                "warnings": scientific_audit["warnings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
