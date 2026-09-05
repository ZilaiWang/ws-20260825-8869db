#!/usr/bin/env python3
"""Split one deployment shared-forward cache into aligned OTO/OTM caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_shared(cache: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if cache.get("head") != "shared" or not cache.get("images"):
        raise ValueError("Expected one nonempty shared-forward cache")
    outputs: dict[str, dict[str, Any]] = {}
    for head in ("oto", "otm"):
        rows = []
        for row in cache["images"]:
            if "prediction" not in row or "otm_prediction" not in row:
                raise ValueError("Shared row lacks one of its aligned predictions")
            output = {key: value for key, value in row.items() if key != "otm_prediction"}
            if head == "otm":
                output["prediction"] = row["otm_prediction"]
            rows.append(output)
        outputs[head] = {
            **cache,
            "head": head,
            "cache_source": "deployment_shared_forward",
            "images": rows,
        }
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--output-oto", type=Path, required=True)
    parser.add_argument("--output-otm", type=Path, required=True)
    args = parser.parse_args()

    outputs = split_shared(_read(args.shared))
    _write(args.output_oto, outputs["oto"])
    _write(args.output_otm, outputs["otm"])
    print(
        json.dumps(
            {
                "status": "complete",
                "shared_sha256": _sha256(args.shared),
                "images": len(outputs["oto"]["images"]),
                "oto_sha256": _sha256(args.output_oto),
                "otm_sha256": _sha256(args.output_otm),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
