#!/usr/bin/env python3
"""Build disjoint pseudo-10K mosaics for every formal CV3 held-out fold.

Each mosaic is made only from images marked ``split=val`` in one frozen
``split_view.json``.  The corresponding fold checkpoint is therefore the only
checkpoint allowed to evaluate it.  Images are not reused within a fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from scripts.build_pseudo_10k_mosaics import (
        GRID,
        _label_path,
        _labels,
        _select_images,
        _select_images_by_coarse_mix,
        build_mosaic_from_selected,
    )
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from build_pseudo_10k_mosaics import (
        GRID,
        _label_path,
        _labels,
        _select_images,
        _select_images_by_coarse_mix,
        build_mosaic_from_selected,
    )


def excluded_sources_from_ground_truth(paths: list[Path]) -> set[str]:
    """Read and normalize source-image identities from prior pseudo mosaics."""

    excluded: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for image in payload.get("images", []):
            for source in image.get("source_images", []):
                excluded.add(str(Path(source).expanduser().resolve()))
    return excluded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-views", type=Path, nargs=3, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count-per-fold", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--exclude-ground-truth",
        type=Path,
        action="append",
        default=[],
        help="prior pseudo ground_truth.json whose source_images must not be reused",
    )
    parser.add_argument(
        "--selection-profile",
        choices=("natural", "trial_mix_stress"),
        default="natural",
    )
    args = parser.parse_args()
    if args.count_per_fold <= 0:
        raise ValueError("--count-per-fold must be > 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_images: list[dict[str, object]] = []
    all_annotations: list[dict[str, object]] = []
    fold_records: list[dict[str, object]] = []
    seen_image_ids: set[int] = set()
    seen_sources: set[str] = set()
    excluded_sources = excluded_sources_from_ground_truth(args.exclude_ground_truth)

    for expected_fold, split_path in enumerate(args.split_views):
        view = json.loads(split_path.read_text(encoding="utf-8"))
        fold = int(view["held_out_fold"])
        if fold != expected_fold:
            raise ValueError(
                f"split view order mismatch: expected fold {expected_fold}, got {fold}"
            )
        val_paths = [
            args.data_root / str(sample["relative_path"])
            for sample in view["samples"]
            if sample.get("split") == "val"
        ]
        val_paths = [path for path in val_paths if path.is_file() and _label_path(path).is_file()]
        val_paths = [
            path for path in val_paths if str(path.expanduser().resolve()) not in excluded_sources
        ]
        required = args.count_per_fold * GRID * GRID
        if len(val_paths) < required:
            raise RuntimeError(f"fold {fold} has only {len(val_paths)} valid images; need {required}")

        remaining = sorted(val_paths, key=str)
        fold_dir = args.output_dir / f"fold_{fold}"
        image_dir = fold_dir / "images"
        fold_images: list[dict[str, object]] = []
        fold_annotations: list[dict[str, object]] = []
        for mosaic_index in range(args.count_per_fold):
            seed = args.seed + fold * 10_000 + mosaic_index
            available_classes = {
                label
                for path in remaining
                for label, *_ in _labels(_label_path(path))
            }
            selection_audit: dict[str, object] = {"profile": args.selection_profile}
            if args.selection_profile == "trial_mix_stress":
                selected, mix_audit = _select_images_by_coarse_mix(
                    remaining,
                    GRID * GRID,
                    seed,
                    {"ship": 0.533, "aircraft": 0.381, "vehicle": 0.086},
                    required_classes=available_classes,
                )
                selection_audit.update(mix_audit)
            else:
                selected = _select_images(
                    remaining,
                    GRID * GRID,
                    seed,
                    required_classes=available_classes,
                )
            selected_set = set(selected)
            remaining = [path for path in remaining if path not in selected_set]
            image_id = fold * args.count_per_fold + mosaic_index + 1
            output_image = image_dir / f"fold{fold}_pseudo_10k_{mosaic_index:02d}.jpg"
            image, annotations = build_mosaic_from_selected(
                selected, output_image, image_id
            )
            image["fold"] = fold
            image["seed"] = seed
            image["selection_audit"] = selection_audit
            fold_images.append(image)
            fold_annotations.extend(annotations)
            if image_id in seen_image_ids:
                raise RuntimeError(f"duplicate image_id: {image_id}")
            seen_image_ids.add(image_id)
            for source in image["source_images"]:
                source_text = str(source)
                if source_text in seen_sources:
                    raise RuntimeError(f"source image reused across pseudo mosaics: {source_text}")
                seen_sources.add(source_text)

        categories = [{"id": index, "name": str(index)} for index in range(25)]
        fold_gt = {
            "images": fold_images,
            "annotations": fold_annotations,
            "categories": categories,
        }
        (fold_dir / "ground_truth.json").write_text(
            json.dumps(fold_gt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        all_images.extend(fold_images)
        all_annotations.extend(fold_annotations)
        fold_records.append(
            {
                "fold": fold,
                "split_view": str(split_path),
                "split_view_sha256": _sha256(split_path),
                "held_out_images_available": len(val_paths),
                "mosaic_image_ids": [int(item["id"]) for item in fold_images],
                "source_images_used": len(fold_images) * GRID * GRID,
                "annotations": len(fold_annotations),
            }
        )

    combined = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": [{"id": index, "name": str(index)} for index in range(25)],
    }
    combined_path = args.output_dir / "ground_truth.json"
    combined_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "cv3_oof_pseudo_10k_ready",
        "protocol": "formal_cv3_heldout_only_disjoint_pseudo10k_v1",
        "selection_profile": args.selection_profile,
        "trial_mix_warning": (
            "trial_mix_stress uses observed prediction-trial coarse proportions and is a "
            "stress diagnostic, not an independent benchmark"
            if args.selection_profile == "trial_mix_stress"
            else None
        ),
        "folds": fold_records,
        "images": len(all_images),
        "source_images": len(seen_sources),
        "excluded_source_images": len(excluded_sources),
        "exclude_ground_truth": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in args.exclude_ground_truth
        ],
        "annotations": len(all_annotations),
        "ground_truth_sha256": _sha256(combined_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
