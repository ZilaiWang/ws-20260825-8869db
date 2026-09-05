#!/usr/bin/env python3
"""Create an audited YOLO class-row task-vector checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.experiments.class_task_vector import (
    assert_same_architecture,
    final_class_conv_candidates,
    interpolate_class_rows,
    resolve_inference_model,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint must be a dict: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--num-classes", type=int, default=25)
    parser.add_argument("--class-ids", default="24")
    parser.add_argument("--module-regex", required=True)
    parser.add_argument("--expected-module-count", type=int, default=6)
    parser.add_argument("--list-candidates", action="store_true")
    return parser.parse_args()


def main() -> int:
    import torch

    args = parse_args()
    base_checkpoint = load_checkpoint(args.base)
    donor_checkpoint = load_checkpoint(args.donor)
    model_key, base = resolve_inference_model(base_checkpoint)
    _, donor = resolve_inference_model(donor_checkpoint)
    assert_same_architecture(base, donor)
    candidates = final_class_conv_candidates(base, args.num_classes)
    if args.list_candidates:
        print(json.dumps({"candidate_modules": candidates}, indent=2))
        return 0
    if args.output is None:
        raise ValueError("--output is required")
    class_ids = [int(raw) for raw in args.class_ids.split(",") if raw.strip()]
    merged, matched = interpolate_class_rows(
        base,
        donor,
        alpha=args.alpha,
        num_classes=args.num_classes,
        class_ids=class_ids,
        module_regex=args.module_regex,
        expected_module_count=args.expected_module_count,
    )
    # Alpha zero is a strict checkpoint-state parity gate.
    if args.alpha == 0.0:
        for key, value in base.state_dict().items():
            if not torch.equal(value.detach().cpu(), merged.state_dict()[key].detach().cpu()):
                raise AssertionError(f"alpha=0 state mismatch: {key}")

    output_checkpoint = copy.deepcopy(base_checkpoint)
    output_checkpoint[model_key] = merged.half()
    for key in ("optimizer", "scaler"):
        if key in output_checkpoint:
            output_checkpoint[key] = None
    output_checkpoint["task_vector_audit"] = {
        "protocol": "lineage_constrained_class_row_task_vector_v1",
        "base_sha256": sha256_file(args.base),
        "donor_sha256": sha256_file(args.donor),
        "alpha": args.alpha,
        "class_ids": class_ids,
        "matched_modules": matched,
        "candidate_modules_seen": candidates,
        "inference_model_key": model_key,
        "coarse_ownership_preserved": class_ids == [24],
        "geometry_state_inherited_from_base": True,
        "non_target_rows_bitwise_inherited_from_base": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    audit = dict(output_checkpoint["task_vector_audit"])
    audit.update(
        {
            "base": str(args.base),
            "donor": str(args.donor),
            "output": str(args.output),
            "output_sha256": sha256_file(args.output),
        }
    )
    args.output.with_suffix(args.output.suffix + ".audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
