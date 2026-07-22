#!/usr/bin/env python3
"""MG00：构建 MAR20 target/bridge 节点登记和输入门禁。"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from rsdet.grouping.contracts import PROTOCOL_VERSION, atomic_write_json, sha256_file
from rsdet.grouping.registry import build_registry, write_registry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 MAR20 来源分组 registry")
    parser.add_argument("--competition-root", required=True)
    parser.add_argument("--mar20-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(dirty), "dirty_count": len(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "dirty_count": None}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    implementation = {
        "src/rsdet/grouping/contracts.py": sha256_file(
            Path(__file__).resolve().parent.parent / "src/rsdet/grouping/contracts.py"
        ),
        "src/rsdet/grouping/registry.py": sha256_file(
            Path(__file__).resolve().parent.parent / "src/rsdet/grouping/registry.py"
        ),
        "scripts/build_mar20_source_registry.py": sha256_file(Path(__file__).resolve()),
    }
    atomic_write_json(
        output_dir / "environment.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "python": sys.version,
            "platform": platform.platform(),
            "git": _git_state(),
            "implementation_sha256": implementation,
        },
    )
    records, annotations, summary = build_registry(
        competition_root=args.competition_root,
        mar20_root=args.mar20_root,
    )
    result = write_registry(output_dir, records, annotations, summary)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
