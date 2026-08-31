#!/usr/bin/env python3
"""Download and recursively lock one official Google Drive dataset folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gdown", default="gdown")
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    folder_id = str(config.get("official_gdrive_folder_id", "")).strip()
    if not folder_id:
        raise ValueError("config lacks official_gdrive_folder_id")
    free = shutil.disk_usage(args.output_root.parent).free
    required = int(args.minimum_free_gib * 1024**3)
    plan = {
        "status": "dry_run" if args.dry_run else "download_requested",
        "protocol": "official_gdrive_folder_recursive_asset_lock_v1",
        "dataset_id": config.get("dataset_id"),
        "official_page": config.get("official_page"),
        "license_scope": config.get("license_scope"),
        "folder_id": folder_id,
        "config_sha256": _sha256(args.config),
        "free_bytes_before": free,
        "minimum_free_bytes": required,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "download_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    )
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if free < required:
        raise RuntimeError(f"insufficient free disk: free={free}, required={required}")
    subprocess.run(
        [
            args.gdown,
            "--folder",
            f"https://drive.google.com/drive/folders/{folder_id}",
            "--remaining-ok",
            "--continue",
            "-O",
            str(args.output_root),
        ],
        check=True,
    )
    rows = []
    for path in sorted(item for item in args.output_root.rglob("*") if item.is_file()):
        if path.name in {"download_plan.json", "ASSET_LOCK.json"}:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(args.output_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not rows:
        raise RuntimeError("Google Drive folder download produced no files")
    result = {
        **plan,
        "status": "complete",
        "file_count": len(rows),
        "files": rows,
        "free_bytes_after": shutil.disk_usage(args.output_root).free,
    }
    (args.output_root / "ASSET_LOCK.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
