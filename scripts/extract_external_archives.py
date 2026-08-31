#!/usr/bin/env python3
"""Safely extract external-data ZIP archives and freeze a SHA manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_members(archive: zipfile.ZipFile, output_dir: Path) -> list[zipfile.ZipInfo]:
    root = output_dir.resolve()
    members = archive.infolist()
    for member in members:
        target = (output_dir / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"unsafe ZIP member: {member.filename}")
    return members


def extract_archives(archives: list[Path], output_dir: Path) -> dict:
    if not archives:
        raise ValueError("at least one archive is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    member_count = 0
    claimed_files: dict[str, Path] = {}
    members_by_archive: dict[Path, list[zipfile.ZipInfo]] = {}
    # Complete the collision/security audit before writing any extracted file.
    for archive_path in archives:
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            members = safe_members(archive, output_dir)
            for member in members:
                normalized = Path(member.filename).as_posix().rstrip("/")
                if member.is_dir() or not normalized:
                    continue
                previous = claimed_files.get(normalized)
                if previous is not None:
                    raise ValueError(
                        f"duplicate archive member {normalized}: {previous} / {archive_path}"
                    )
                claimed_files[normalized] = archive_path
            members_by_archive[archive_path] = members
        rows.append(
            {
                "path": str(archive_path.resolve()),
                "size": archive_path.stat().st_size,
                "sha256": _sha256(archive_path),
                "member_count": len(members),
            }
        )
        member_count += len(members)
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(output_dir, members=members_by_archive[archive_path])
    extracted_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    return {
        "status": "complete",
        "protocol": "safe_external_zip_extract_v1",
        "output_dir": str(output_dir.resolve()),
        "archives": rows,
        "archive_count": len(rows),
        "declared_member_count": member_count,
        "unique_file_member_count": len(claimed_files),
        "extracted_file_count": len(extracted_files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = extract_archives(args.archive, args.output_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
