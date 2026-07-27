#!/usr/bin/env python3
"""Validate formal CV3 and materialize six framework-neutral CSV views."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

from rsdet.data.formal_cv3 import (
    formal_cv3_audit_payload,
    load_formal_cv3_manifest,
    validate_written_view,
    write_formal_cv3_views,
)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("formal CV3 config root must be a mapping")
    config = payload.get("formal_cv3")
    output = payload.get("output")
    if not isinstance(config, Mapping) or not isinstance(output, Mapping):
        raise ValueError("config must define formal_cv3 and output mappings")
    if output.get("write_csv_views") is not True:
        raise ValueError("formal CV3 task requires write_csv_views=true")
    if output.get("write_audit_json") is not True:
        raise ValueError("formal CV3 task requires write_audit_json=true")
    return {"formal_cv3": dict(config), "output": dict(output)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="严格校验正式 CV3，并输出 held_out_fold train/val 视图",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/cv3_airport_proxy_k60_v2.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/analysis/formal_cv3.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/FORMAL-CV3-CONSUMER-v1"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = _load_config(args.config.expanduser().resolve())
        contract = config["formal_cv3"]
        manifest = load_formal_cv3_manifest(
            args.manifest,
            expected_sha256=str(contract["manifest_sha256"]),
            expected_version=str(contract["version"]),
            expected_data_version=str(contract["data_version"]),
            expected_fold_count=int(contract["fold_count"]),
            expected_sample_count=int(contract["sample_count"]),
            expected_group_count=int(contract["source_group_count"]),
            expected_fold_image_counts=contract["fold_image_counts"],
            expected_fold_group_counts=contract["fold_group_counts"],
        )
        output = args.output_dir.expanduser().resolve()
        view_hashes = write_formal_cv3_views(manifest, output)
        for held_out_fold in range(manifest.fold_count):
            for split in ("train", "val"):
                validate_written_view(
                    output / f"formal_cv3_fold{held_out_fold}_{split}.csv",
                    manifest=manifest,
                    held_out_fold=held_out_fold,
                    split=split,
                )
        audit = formal_cv3_audit_payload(manifest)
        audit["view_sha256"] = view_hashes
        audit["config"] = str(args.config.expanduser().resolve())
        audit["output_schema_version"] = config["output"]["schema_version"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "formal_cv3_consumer_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"formal CV3 consumer failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
