#!/usr/bin/env python3
"""Build an auditable 50/50 CE + view-consistency aircraft bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"proposal_uids", "identity_logits", "d4_probabilities"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path} 缺少数组: {sorted(missing)}")
        return {key: np.asarray(payload[key]) for key in required}


def _aligned(source: dict[str, np.ndarray], target_uids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_uids = [str(value) for value in source["proposal_uids"].tolist()]
    if len(source_uids) != len(set(source_uids)):
        raise ValueError("bundle proposal UID 重复")
    index = {uid: offset for offset, uid in enumerate(source_uids)}
    target = [str(value) for value in target_uids.tolist()]
    if set(index) != set(target):
        raise ValueError("CE 与 consistency bundle UID 集合不同")
    order = np.asarray([index[uid] for uid in target], dtype=np.int64)
    return source["identity_logits"][order], source["d4_probabilities"][order]


def build_fold(ce_path: Path, consistency_path: Path, output_path: Path) -> dict[str, Any]:
    ce = _load(ce_path)
    consistency = _load(consistency_path)
    uids = np.asarray(ce["proposal_uids"])
    ce_identity = np.asarray(ce["identity_logits"], dtype=np.float64)
    ce_d4 = np.asarray(ce["d4_probabilities"], dtype=np.float64)
    vc_identity, vc_d4 = _aligned(consistency, uids)
    vc_identity = np.asarray(vc_identity, dtype=np.float64)
    vc_d4 = np.asarray(vc_d4, dtype=np.float64)
    if ce_identity.shape != vc_identity.shape or ce_d4.shape != vc_d4.shape:
        raise ValueError("CE 与 consistency bundle shape 不同")
    if ce_identity.shape != (len(uids), 25) or ce_d4.shape != (len(uids), 20):
        raise ValueError("bundle shape 不符合 25-way identity / 20-way D4 合同")
    if not all(
        np.isfinite(values).all()
        for values in (ce_identity, ce_d4, vc_identity, vc_d4)
    ):
        raise ValueError("bundle 包含 NaN/Inf")
    identity = 0.5 * ce_identity + 0.5 * vc_identity
    d4 = 0.5 * ce_d4 + 0.5 * vc_d4
    d4 /= d4.sum(axis=1, keepdims=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        proposal_uids=uids,
        identity_logits=identity.astype(np.float32),
        d4_probabilities=d4.astype(np.float32),
    )
    return {
        "row_count": len(uids),
        "ce_sha256": _sha256(ce_path),
        "view_consistency_sha256": _sha256(consistency_path),
        "output_sha256": _sha256(output_path),
        "maximum_probability_sum_error": float(np.abs(d4.sum(axis=1) - 1.0).max()),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--ce-bundle-dir", type=Path, required=True)
    value.add_argument("--consistency-bundle-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    destination = args.output_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    folds: dict[str, Any] = {}
    for fold in (0, 1, 2):
        name = f"fold_{fold}_aircraft_bundle.npz"
        folds[str(fold)] = build_fold(
            args.ce_bundle_dir / name,
            args.consistency_bundle_dir / name,
            destination / name,
        )
    audit = {
        "status": "pass",
        "method": "fixed_equal_probability_ensemble",
        "ce_weight": 0.5,
        "view_consistency_weight": 0.5,
        "weight_was_not_tuned": True,
        "folds": folds,
    }
    (destination / "ensemble_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
