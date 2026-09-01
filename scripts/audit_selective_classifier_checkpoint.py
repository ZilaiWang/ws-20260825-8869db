#!/usr/bin/env python3
"""Verify that selective fine-tuning changed only permitted classifier rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

CLASSIFIER_PATTERN = re.compile(
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
    parser.add_argument(
        "--focus-class-indices", type=int, nargs="+", default=[0, 1, 2, 3, 24]
    )
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    base_model = YOLO(str(args.base.resolve())).model.cpu()
    candidate_model = YOLO(str(args.candidate.resolve())).model.cpu()
    base_state = base_model.state_dict()
    candidate_state = candidate_model.state_dict()
    if base_state.keys() != candidate_state.keys():
        raise ValueError("checkpoint state dictionaries expose different keys")
    focus = tuple(sorted(set(args.focus_class_indices)))
    changed_tensors: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    permitted_tensor_count = 0
    selected_rows_changed = 0

    for name, base_tensor in base_state.items():
        candidate_tensor = candidate_state[name]
        if base_tensor.shape != candidate_tensor.shape:
            violations.append({"name": name, "reason": "shape_mismatch"})
            continue
        if torch.equal(base_tensor, candidate_tensor):
            continue
        delta = (candidate_tensor.float() - base_tensor.float()).abs()
        item = {
            "name": name,
            "shape": list(base_tensor.shape),
            "max_abs_delta": float(delta.max().item()),
        }
        changed_tensors.append(item)
        if not CLASSIFIER_PATTERN.match(name):
            violations.append({**item, "reason": "changed_outside_final_classifiers"})
            continue
        permitted_tensor_count += 1
        if base_tensor.shape[0] != 25:
            violations.append({**item, "reason": "unexpected_classifier_rows"})
            continue
        nonfocus = [index for index in range(25) if index not in focus]
        if not torch.equal(base_tensor[nonfocus], candidate_tensor[nonfocus]):
            violations.append({**item, "reason": "nonfocus_classifier_row_changed"})
        if any(
            not torch.equal(base_tensor[index], candidate_tensor[index])
            for index in focus
        ):
            selected_rows_changed += 1

    status = "pass" if not violations and selected_rows_changed > 0 else "fail"
    payload = {
        "status": status,
        "protocol": "selective_classifier_checkpoint_exact_row_audit_v1",
        "base": str(args.base.resolve()),
        "base_sha256": _sha256(args.base),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": _sha256(args.candidate),
        "focus_class_indices": list(focus),
        "state_tensor_count": len(base_state),
        "changed_tensor_count": len(changed_tensors),
        "permitted_changed_tensor_count": permitted_tensor_count,
        "selected_rows_changed_tensor_count": selected_rows_changed,
        "changed_tensors": changed_tensors,
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
