#!/usr/bin/env python3
"""Fail-closed audit that a repeated-role list survives Ultralytics loading."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import yaml


def _multiplicity_histogram(values: list[str]) -> dict[str, int]:
    return {
        str(multiplicity): count
        for multiplicity, count in sorted(Counter(Counter(values).values()).items())
    }


def _class_counts(labels: list[dict]) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for row in labels:
        values = row.get("cls")
        if values is None:
            continue
        for value in values.reshape(-1).tolist():
            counts[int(value)] += 1
    return {str(key): value for key, value in sorted(counts.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=30)
    parser.add_argument("--world-size", type=int, default=3)
    args = parser.parse_args()

    import ultralytics
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.engine.trainer import BaseTrainer

    payload = yaml.safe_load(args.dataset.read_text(encoding="utf-8"))
    train_list = Path(str(payload["train"])).expanduser().resolve()
    expected_rows = [
        str(Path(line.strip()).expanduser().resolve())
        for line in train_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not expected_rows:
        raise ValueError("dataset train list is empty")
    hyp = get_cfg(
        overrides={
            "imgsz": args.imgsz,
            "batch": args.batch,
            "task": "detect",
            "mode": "train",
        }
    )
    dataset = YOLODataset(
        img_path=str(train_list),
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=True,
        hyp=hyp,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="role-audit: ",
        data=payload,
    )
    actual_rows = [str(Path(path).expanduser().resolve()) for path in dataset.im_files]
    if Counter(actual_rows) != Counter(expected_rows):
        missing = Counter(expected_rows) - Counter(actual_rows)
        extra = Counter(actual_rows) - Counter(expected_rows)
        raise RuntimeError(
            "Ultralytics changed the repeated role list: "
            f"missing_first={list(missing.items())[:5]}, extra_first={list(extra.items())[:5]}"
        )
    trainer_source = inspect.getsource(BaseTrainer._setup_train)
    expected_accumulate = max(round(float(hyp.nbs) / args.batch), 1)
    audit = {
        "status": "pass",
        "protocol": "ultralytics_repeated_role_dataset_runtime_audit_v1",
        "ultralytics_version": ultralytics.__version__,
        "dataset_yaml": str(args.dataset.resolve()),
        "dataset_yaml_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "train_list": str(train_list),
        "train_list_sha256": hashlib.sha256(train_list.read_bytes()).hexdigest(),
        "listed_row_count": len(expected_rows),
        "loaded_row_count": len(actual_rows),
        "unique_image_count": len(set(actual_rows)),
        "duplicate_multiplicity_histogram": _multiplicity_histogram(actual_rows),
        "loaded_class_instance_counts": _class_counts(dataset.labels),
        "requested_global_batch": args.batch,
        "world_size": args.world_size,
        "nominal_batch_per_rank": args.batch / args.world_size,
        "trainer_nbs": int(hyp.nbs),
        "expected_accumulate_from_installed_trainer_contract": expected_accumulate,
        "expected_effective_optimization_batch": args.batch * expected_accumulate,
        "base_trainer_setup_train_source_sha256": hashlib.sha256(
            trainer_source.encode("utf-8")
        ).hexdigest(),
        "note": (
            "Optimizer-group lr/weight_decay and runtime trainer.accumulate must also be "
            "copied from the completed training log into the final experiment audit."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
