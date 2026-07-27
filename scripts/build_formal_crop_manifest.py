#!/usr/bin/env python3
"""Rehang P0-2 crop records onto formal CV3 without reading source pixels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

from rsdet.analysis.formal_crop import build_formal_crop_manifest
from rsdet.data.formal_cv3 import load_formal_cv3_manifest


def _load_mapping(path: Path, key: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get(key), Mapping):
        raise ValueError(f"{path} must contain a {key} mapping")
    return dict(payload[key])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 source_relative_path 将 P0-2 记录重挂到正式 CV3",
    )
    parser.add_argument("--exploratory-manifest", type=Path, required=True)
    parser.add_argument(
        "--cv3-manifest",
        type=Path,
        default=Path("data/splits/cv3_airport_proxy_k60_v2.json"),
    )
    parser.add_argument(
        "--cv3-config",
        type=Path,
        default=Path("configs/analysis/formal_cv3.yaml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/analysis/formal_crop_manifest.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/P0-2-formal-crop-manifest-v2"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cv3_contract = _load_mapping(
            args.cv3_config.expanduser().resolve(),
            "formal_cv3",
        )
        formal_cv3 = load_formal_cv3_manifest(
            args.cv3_manifest,
            expected_sha256=str(cv3_contract["manifest_sha256"]),
            expected_version=str(cv3_contract["version"]),
            expected_data_version=str(cv3_contract["data_version"]),
            expected_fold_count=int(cv3_contract["fold_count"]),
            expected_sample_count=int(cv3_contract["sample_count"]),
            expected_group_count=int(cv3_contract["source_group_count"]),
            expected_fold_image_counts=cv3_contract["fold_image_counts"],
            expected_fold_group_counts=cv3_contract["fold_group_counts"],
        )
        crop_contract = _load_mapping(
            args.config.expanduser().resolve(),
            "formal_crop",
        )
        audit = build_formal_crop_manifest(
            exploratory_manifest_path=args.exploratory_manifest,
            formal_cv3=formal_cv3,
            config=crop_contract,
            output_dir=args.output_dir,
        )
        output = args.output_dir.expanduser().resolve()
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "formal_cv3": cv3_contract,
                    "formal_crop": crop_contract,
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"formal crop rehang failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
