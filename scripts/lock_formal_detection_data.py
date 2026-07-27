#!/usr/bin/env python3
"""Create or verify the formal CV3 image/YOLO byte lock."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rsdet.experiments.detection_data_lock import (
    atomic_write_json,
    atomic_write_new_json,
    build_lock_payload,
    sha256_file,
    verify_existing_lock,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bind formal CV3 image/YOLO bytes and prove GT equivalence",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--data-root", required=True, type=Path)
        child.add_argument("--cv3-manifest", required=True, type=Path)
        child.add_argument("--p02-manifest", required=True, type=Path)
        child.add_argument("--formal-crop-manifest", required=True, type=Path)
        if command == "create":
            child.add_argument("--output", required=True, type=Path)
        else:
            child.add_argument("--lock", required=True, type=Path)
            child.add_argument("--expected-lock-sha256")
            child.add_argument("--report", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    common = {
        "spec_path": args.config,
        "data_root": args.data_root,
        "cv3_manifest_path": args.cv3_manifest,
        "p02_manifest_path": args.p02_manifest,
        "formal_crop_manifest_path": args.formal_crop_manifest,
    }
    if args.command == "create":
        payload = build_lock_payload(**common)
        atomic_write_new_json(args.output, payload)
        summary = {
            "status": "created",
            "lock_path": str(args.output.expanduser().resolve()),
            "lock_file_sha256": sha256_file(args.output),
            "lock_fingerprint": payload["lock_fingerprint"],
            "inventory_fingerprint": payload["inventory_fingerprint"],
            "image_count": payload["summary"]["image_count"],
            "label_file_count": payload["summary"]["label_file_count"],
            "object_count": payload["summary"]["object_count"],
            "p02_formal_gt_equivalence": payload["summary"][
                "p02_formal_gt_equivalence"
            ],
            "yolo_formal_gt_equivalence": payload["summary"][
                "yolo_formal_gt_equivalence"
            ],
        }
    else:
        summary = verify_existing_lock(
            **common,
            lock_path=args.lock,
            expected_lock_sha256=args.expected_lock_sha256,
        )
        atomic_write_json(args.report, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
