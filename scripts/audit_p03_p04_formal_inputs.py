#!/usr/bin/env python3
"""审计 P03/P04 正式 crop 重挂与旧 P04 cache 的安全复用。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

from rsdet.analysis.formal_replay import (
    CV3_SHA256,
    audit_cache_reuse,
    audit_formal_crop_manifest,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P03/P04 formal replay input gate")
    parser.add_argument("--formal-manifest", required=True)
    parser.add_argument("--exploratory-manifest", required=True)
    parser.add_argument("--cv3-manifest", required=True)
    parser.add_argument("--expected-cv3-sha256", default=CV3_SHA256)
    parser.add_argument("--data-root")
    parser.add_argument("--asset-lock")
    parser.add_argument(
        "--cache",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="可重复；给出时必须同时给 --data-root，并逐对象核验 canonical224",
    )
    parser.add_argument(
        "--cache-identity",
        action="append",
        default=[],
        metavar="NAME=PREFIX=TEACHER_ID",
        help="每个正式 cache 的已审核 fingerprint 前缀与 teacher_id",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _cache_args(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--cache 必须为 NAME=DIR: {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"--cache 名称为空或重复: {name!r}")
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def _cache_identities(
    values: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    prefixes: dict[str, str] = {}
    teachers: dict[str, str] = {}
    for value in values:
        parts = value.split("=", 2)
        if len(parts) != 3:
            raise ValueError(f"--cache-identity 必须为 NAME=PREFIX=TEACHER_ID: {value!r}")
        name, prefix, teacher = (part.strip() for part in parts)
        if not name or name in prefixes or not prefix or not teacher:
            raise ValueError(f"--cache-identity 为空或重复: {value!r}")
        prefixes[name] = prefix
        teachers[name] = teacher
    return prefixes, teachers


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖 formal input audit: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"拒绝覆盖 formal input audit: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dirs = _cache_args(args.cache)
    fingerprint_prefixes, teacher_ids = _cache_identities(args.cache_identity)
    if cache_dirs and not args.data_root:
        raise ValueError("审计 cache 时必须给 --data-root")
    if cache_dirs and not args.asset_lock:
        raise ValueError("审计正式 cache 时必须给 --asset-lock")
    formal = audit_formal_crop_manifest(
        args.formal_manifest,
        args.exploratory_manifest,
        args.cv3_manifest,
        expected_cv3_sha256=args.expected_cv3_sha256,
    )
    caches = (
        audit_cache_reuse(
            args.formal_manifest,
            args.data_root,
            cache_dirs,
            expected_fingerprint_prefixes=fingerprint_prefixes,
            expected_teacher_ids=teacher_ids,
            asset_lock_path=args.asset_lock,
        )
        if cache_dirs
        else {}
    )
    payload = {
        **formal,
        "formal_crop_gate_status": formal["status"],
        "status": "formal_replay_inputs_ready",
        "caches": caches,
        "cache_count": len(caches),
    }
    output = Path(args.output).expanduser().resolve()
    _atomic_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
