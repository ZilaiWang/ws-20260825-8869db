#!/usr/bin/env python3
"""Audit bounded spatial-classifier residual fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

BRANCH_PATTERN = re.compile(r"^model\.23\.(?:one2one_)?cv3\.")
FINAL_PATTERN = re.compile(
    r"^model\.23\.(?:one2one_)?cv3\.\d+\.2\.(?:weight|bias)$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-branch-relative-delta", type=float, required=True)
    parser.add_argument("--max-final-weight-relative-delta", type=float, required=True)
    parser.add_argument("--max-final-bias-delta", type=float, required=True)
    parser.add_argument(
        "--focus-class-indices", type=int, nargs="+", default=[0, 1, 2, 3, 24]
    )
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    base = YOLO(str(args.base.resolve())).model.cpu().state_dict()
    candidate = YOLO(str(args.candidate.resolve())).model.cpu().state_dict()
    if base.keys() != candidate.keys():
        raise ValueError("checkpoint state dictionaries expose different keys")
    focus = tuple(sorted(set(args.focus_class_indices)))
    changed: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    tolerance = 5e-4

    for name, base_tensor in base.items():
        current = candidate[name]
        if torch.equal(base_tensor, current):
            continue
        delta = current.float() - base_tensor.float()
        record: dict[str, object] = {
            "name": name,
            "shape": list(base_tensor.shape),
            "max_abs_delta": float(delta.abs().max().item()),
        }
        changed.append(record)
        if not BRANCH_PATTERN.match(name):
            violations.append({**record, "reason": "changed_outside_classifier_branches"})
            continue
        if FINAL_PATTERN.match(name):
            if base_tensor.shape[0] != 25:
                violations.append({**record, "reason": "unexpected_classifier_rows"})
                continue
            nonfocus = [index for index in range(25) if index not in focus]
            if not torch.equal(base_tensor[nonfocus], current[nonfocus]):
                violations.append({**record, "reason": "nonfocus_classifier_row_changed"})
            if name.endswith(".bias"):
                value = float(delta[list(focus)].abs().max().item())
                record["focused_max_bias_delta"] = value
                if value > args.max_final_bias_delta + tolerance:
                    violations.append({**record, "reason": "final_bias_bound_exceeded"})
            else:
                base_rows = base_tensor[list(focus)].float().flatten(1)
                delta_rows = delta[list(focus)].flatten(1)
                ratios = delta_rows.norm(dim=1) / base_rows.norm(dim=1).clamp_min(1e-12)
                value = float(ratios.max().item())
                record["focused_max_relative_l2_delta"] = value
                if value > args.max_final_weight_relative_delta + tolerance:
                    violations.append({**record, "reason": "final_weight_bound_exceeded"})
        elif base_tensor.is_floating_point():
            value = float(delta.norm().item() / base_tensor.float().norm().clamp_min(1e-12).item())
            record["relative_l2_delta"] = value
            if value > args.max_branch_relative_delta + tolerance:
                violations.append({**record, "reason": "branch_bound_exceeded"})
        else:
            violations.append({**record, "reason": "nonfloating_branch_buffer_changed"})

    status = "pass" if changed and not violations else "fail"
    payload = {
        "status": status,
        "protocol": "bounded_spatial_classifier_residual_checkpoint_audit_v1",
        "base": str(args.base.resolve()),
        "base_sha256": _sha256(args.base),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": _sha256(args.candidate),
        "focus_class_indices": list(focus),
        "bounds": {
            "max_branch_relative_delta": args.max_branch_relative_delta,
            "max_final_weight_relative_delta": args.max_final_weight_relative_delta,
            "max_final_bias_delta": args.max_final_bias_delta,
        },
        "state_tensor_count": len(base),
        "changed_tensor_count": len(changed),
        "changed_tensors": changed,
        "violations": violations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
