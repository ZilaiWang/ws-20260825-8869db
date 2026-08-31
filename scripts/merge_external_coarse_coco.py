#!/usr/bin/env python3
"""Merge external coarse COCO ledgers with disjoint IDs and path prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_sources(sources: list[tuple[str, dict]]) -> tuple[dict, dict]:
    if len(sources) < 2:
        raise ValueError("at least two external sources are required")
    categories = sources[0][1]["categories"]
    output_images = []
    output_annotations = []
    per_source = {}
    image_id = 1
    annotation_id = 1
    for source_name, payload in sources:
        if not source_name or "/" in source_name or ".." in source_name:
            raise ValueError(f"unsafe source prefix: {source_name!r}")
        if payload["categories"] != categories:
            raise ValueError("external source category ledgers differ")
        old_to_new: dict[int, int] = {}
        for row in sorted(payload["images"], key=lambda item: int(item["id"])):
            old_id = int(row["id"])
            if old_id in old_to_new:
                raise ValueError(f"duplicate image ID within source {source_name}: {old_id}")
            old_to_new[old_id] = image_id
            output_images.append(
                {
                    **row,
                    "id": image_id,
                    "file_name": f"{source_name}/{row['file_name']}",
                    "external_source": source_name,
                    "source_image_id": old_id,
                }
            )
            image_id += 1
        source_annotation_count = 0
        for row in sorted(payload["annotations"], key=lambda item: int(item["id"])):
            old_image_id = int(row["image_id"])
            if old_image_id not in old_to_new:
                raise ValueError("annotation references an unknown source image")
            output_annotations.append(
                {
                    **row,
                    "id": annotation_id,
                    "image_id": old_to_new[old_image_id],
                    "source_annotation_id": int(row["id"]),
                    "external_source": source_name,
                }
            )
            annotation_id += 1
            source_annotation_count += 1
        per_source[source_name] = {
            "images": len(old_to_new),
            "annotations": source_annotation_count,
        }
    output = {
        "images": output_images,
        "annotations": output_annotations,
        "categories": categories,
    }
    audit = {
        "status": "complete",
        "protocol": "external_coarse_coco_disjoint_rebase_merge_v1",
        "source_count": len(sources),
        "sources": per_source,
        "image_count": len(output_images),
        "annotation_count": len(output_annotations),
        "image_ids_contiguous": [row["id"] for row in output_images]
        == list(range(1, len(output_images) + 1)),
        "annotation_ids_contiguous": [row["id"] for row in output_annotations]
        == list(range(1, len(output_annotations) + 1)),
    }
    return output, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="PREFIX=COCO_JSON (repeat at least twice)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    sources = []
    source_paths = {}
    for spec in args.source:
        if "=" not in spec:
            raise ValueError("--source must be PREFIX=COCO_JSON")
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        sources.append((name, json.loads(path.read_text(encoding="utf-8"))))
        source_paths[name] = {"path": str(path.resolve()), "sha256": _sha256(path)}
    output, audit = merge_sources(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False) + "\n")
    audit["inputs"] = source_paths
    audit["output_sha256"] = _sha256(args.output)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
