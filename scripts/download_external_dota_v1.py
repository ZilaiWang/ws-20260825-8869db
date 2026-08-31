#!/usr/bin/env python3
"""Download the frozen official DOTA-v1 train/val assets with a disk gate."""

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
    parser.add_argument("--config", type=Path, default=Path("configs/external/dota_v1_coarse.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gdown", default="gdown")
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    assets = list(config.get("official_assets", []))
    if not assets:
        raise ValueError("external config has no official_assets")
    free_bytes = shutil.disk_usage(args.output_root.parent).free
    required_bytes = int(args.minimum_free_gib * 1024**3)
    plan = {
        "status": "dry_run" if args.dry_run else "download_requested",
        "protocol": "official_dota_v1_gdrive_asset_lock_v1",
        "config_sha256": _sha256(args.config),
        "output_root": str(args.output_root.resolve()),
        "free_bytes_before": free_bytes,
        "minimum_free_bytes": required_bytes,
        "assets": assets,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_root / "download_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"insufficient free disk: free={free_bytes}, required={required_bytes}; "
            "use the later training server instead of partially downloading"
        )
    rows = []
    for asset in assets:
        output = args.output_root / str(asset["relative_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.is_file():
            subprocess.run(
                [args.gdown, "--continue", str(asset["gdrive_id"]), "-O", str(output)],
                check=True,
            )
        rows.append(
            {
                **asset,
                "path": str(output.resolve()),
                "size": output.stat().st_size,
                "sha256": _sha256(output),
            }
        )
    result = {
        **plan,
        "status": "complete",
        "assets": rows,
        "free_bytes_after": shutil.disk_usage(args.output_root).free,
    }
    result_path = args.output_root / "ASSET_LOCK.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
