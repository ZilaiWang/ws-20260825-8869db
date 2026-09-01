#!/usr/bin/env python3
"""Fail if an active formal entrypoint is not bound to the observed protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/evaluation/metric_protocol_registry.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    protocol = registry["formal_metric_protocol"]
    rows = []
    for raw_path in registry["active_formal_entrypoints"]:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        bound = (
            protocol in text
            or "parse_evaluation_protocol" in text
            or "PLATFORM_OBSERVED_PROTOCOL" in text
        )
        rows.append(
            {
                "path": raw_path,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "protocol_bound": bound,
            }
        )
    failed = [row["path"] for row in rows if not row["protocol_bound"]]
    payload = {
        "version": "metric_protocol_migration_audit_v1",
        "metric_protocol": protocol,
        "active_count": len(rows),
        "status": "pass" if not failed else "fail",
        "failed": failed,
        "files": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
