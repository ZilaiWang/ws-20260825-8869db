#!/usr/bin/env python3
"""Gate and launch the one allowed full fit from a frozen MacroShift recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    actual_recipe_sha = _sha256(args.recipe)
    if actual_recipe_sha != args.recipe_sha256.lower():
        raise ValueError("final recipe file SHA mismatch")
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    if recipe.get("metric_protocol") != "platform_observed_20260831":
        raise ValueError("final recipe uses a stale metric protocol")
    if recipe.get("unique_full_training_admission") is not True:
        raise RuntimeError("no independently admitted module; full fit is blocked")
    if not recipe.get("accepted_modules"):
        raise RuntimeError("final recipe accepted_modules is empty")
    contract = {
        "version": "macroshift_full_launcher_v1",
        "recipe": str(args.recipe.resolve()),
        "recipe_file_sha256": actual_recipe_sha,
        "recipe_content_sha256": recipe["recipe_sha256"],
        "accepted_modules": recipe["accepted_modules"],
        "single_full_fit": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "macroshift_launcher_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    command = [
        sys.executable,
        str(Path(__file__).with_name("train_full_y5.py")),
        "--manifest",
        str(args.manifest),
        "--data-root",
        str(args.data_root),
        "--weights",
        str(args.weights),
        "--expected-weight-sha256",
        args.expected_weight_sha256,
        "--output-dir",
        str(args.output_dir),
        "--model-key",
        "MACROSHIFT-FULL-S",
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
    ]
    if args.dry_run:
        command.append("--dry-run")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
