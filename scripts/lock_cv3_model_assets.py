#!/usr/bin/env python3
"""Create or verify the immutable M1/M3 model asset and environment lock."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rsdet.experiments.model_asset_env_lock import (
    atomic_write_new_json,
    build_lock_payload,
    collect_environment,
    verify_existing_lock,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M1/M3 formal model asset and environment lock gate"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "verify"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", required=True)
        child.add_argument("--asset-root", required=True)
        child.add_argument("--expected-gpu", required=True)
        if name == "create":
            child.add_argument("--output", required=True)
        else:
            child.add_argument("--lock", required=True)
            child.add_argument("--report", required=True)
    return parser.parse_args(argv)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    environment = collect_environment()
    if args.command == "create":
        payload = build_lock_payload(
            spec_path=Path(args.config),
            asset_root=Path(args.asset_root),
            environment=environment,
            expected_gpu=args.expected_gpu,
        )
        atomic_write_new_json(Path(args.output), payload)
        summary = {
            "status": "created",
            "output": str(Path(args.output).expanduser().resolve()),
            "lock_fingerprint": payload["lock_fingerprint"],
            "asset_count": len(payload["assets"]),
            "gpu_name": payload["environment"]["torch_gpu"]["name"],
        }
    else:
        summary = verify_existing_lock(
            spec_path=Path(args.config),
            asset_root=Path(args.asset_root),
            lock_path=Path(args.lock),
            environment=environment,
            expected_gpu=args.expected_gpu,
        )
        _write_report(Path(args.report), summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
