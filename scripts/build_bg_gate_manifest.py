#!/usr/bin/env python3
"""构建 N2-CFG 前景门控训练 manifest（CLI 封装）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rsdet.analysis.background_gate_manifest import (
    build_foreground_gate_manifest,
    write_gate_manifest,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=Path("configs/experiments/n2_cfg_background_gate_v1.yaml"))
    value.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    value.add_argument("--clear-background-whitelist", type=Path, default=None)
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if config.get("contract_version") != "n2_cfg_background_gate_v1":
        raise ValueError("unexpected N2-CFG contract")
    root = Path.cwd().resolve()
    inputs = config["inputs"]
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    rows, summary = build_foreground_gate_manifest(
        selected_predictions_path=root / inputs["selected_predictions"],
        formal_crop_manifest_path=root / inputs["formal_crop_manifest"],
        image_ledger_path=root / inputs["image_ledger"],
        clear_background_whitelist_path=args.clear_background_whitelist,
        protocol=protocol,
        formal_manifest_sha256=inputs["formal_crop_manifest_sha256"],
        context_ratio=float(config["crop"]["expansion_ratio"]),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_gate_manifest(rows, output)
    summary["output_path"] = str(output)
    (output.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"BG_GATE_MANIFEST_PASS rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
