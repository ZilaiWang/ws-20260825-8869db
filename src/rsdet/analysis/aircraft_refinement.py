"""R1-1 aircraft-only proposal-domain refinement helpers.

The module intentionally keeps training policy outside the detector.  Ship and
vehicle proposals are structural bypasses; only detector-routed aircraft may be
reclassified inside the 20-class aircraft taxonomy.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsdet.analysis.proposal_reranking import (
    ProposalRecord,
    RerankVariant,
    load_logits,
)

AIRCRAFT_CLASS_IDS = tuple(range(4, 24))
R11_CONTRACT_VERSION = "r1_aircraft_refinement_v1"


def load_aircraft_training_rows(path: str | Path) -> list[dict[str, str]]:
    """Load proposal crops that are valid for the deployed aircraft router.

    Both the ground-truth label and detector route must be aircraft.  This
    excludes cross-coarse oracle examples that can never reach this head at
    deployment time.
    """

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "proposal_uid",
            "source_relative_path",
            "fold",
            "class_id",
            "detector_category_id",
            "crop_xyxy",
            "view",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"proposal crop manifest 缺列: {sorted(missing)}")
        rows = [
            dict(row)
            for row in reader
            if int(row["class_id"]) in AIRCRAFT_CLASS_IDS
            and int(row["detector_category_id"]) in AIRCRAFT_CLASS_IDS
            and row["view"] in {"deployable_positive", "oracle_positive"}
        ]
    if not rows:
        raise ValueError("aircraft proposal 训练集为空")
    seen: set[str] = set()
    for row in rows:
        uid = row["proposal_uid"].strip()
        fold = int(row["fold"])
        if not uid or uid in seen:
            raise ValueError(f"proposal_uid 为空或重复: {uid!r}")
        if fold not in {0, 1, 2}:
            raise ValueError(f"proposal {uid} fold 非法: {fold}")
        seen.add(uid)
    return rows


def audit_aircraft_training_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    fold_counts = Counter(int(row["fold"]) for row in rows)
    class_counts = Counter(int(row["class_id"]) for row in rows)
    view_counts = Counter(str(row["view"]) for row in rows)
    if set(fold_counts) != {0, 1, 2}:
        raise ValueError(f"训练 rows 未覆盖三折: {dict(fold_counts)}")
    if set(class_counts) != set(AIRCRAFT_CLASS_IDS):
        missing = set(AIRCRAFT_CLASS_IDS) - set(class_counts)
        raise ValueError(f"训练 rows 未覆盖全部 20 飞机类: {sorted(missing)}")
    return {
        "contract_version": R11_CONTRACT_VERSION,
        "status": "pass",
        "row_count": len(rows),
        "fold_counts": dict(sorted(fold_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "view_counts": dict(sorted(view_counts.items())),
        "minimum_class_count": min(class_counts.values()),
        "maximum_class_count": max(class_counts.values()),
    }


def build_aircraft_variants(config: Mapping[str, Any]) -> list[RerankVariant]:
    """Build a compact aircraft-conditional gate/fusion grid."""

    variants = [
        RerankVariant("D0_detector", relabel=False, target_coarse="aircraft"),
        RerankVariant("A1_hard", relabel=True, target_coarse="aircraft"),
    ]
    for top in config["min_top_probabilities"]:
        for margin in config["min_margins"]:
            variants.append(
                RerankVariant(
                    f"A2_gate_p{float(top):.2f}_m{float(margin):.2f}",
                    relabel=True,
                    min_top_probability=float(top),
                    min_margin=float(margin),
                    target_coarse="aircraft",
                )
            )
    for alpha in config["fusion_alphas"]:
        variants.append(
            RerankVariant(
                f"A3_fuse_a{float(alpha):.2f}",
                relabel=False,
                fusion_alpha=float(alpha),
                target_coarse="aircraft",
            )
        )
    alpha = float(config["gated_fusion_alpha"])
    for top in config["min_top_probabilities"]:
        for margin in config["min_margins"]:
            variants.append(
                RerankVariant(
                    f"A4_gate_fuse_p{float(top):.2f}_m{float(margin):.2f}_a{alpha:.2f}",
                    relabel=True,
                    min_top_probability=float(top),
                    min_margin=float(margin),
                    fusion_alpha=alpha,
                    target_coarse="aircraft",
                )
            )
    if len({item.name for item in variants}) != len(variants):
        raise ValueError("R1-1 variant 名称重复")
    return variants


def aircraft_conditional_probabilities(
    logits: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Convert 25-way logits to a 25-vector carrying p(class|aircraft)."""

    result: dict[str, np.ndarray] = {}
    indices = np.asarray(AIRCRAFT_CLASS_IDS, dtype=np.int64)
    for uid, vector in logits.items():
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (25,) or not np.isfinite(values).all():
            raise ValueError(f"{uid} logits 非法: {values.shape}")
        aircraft = values[indices]
        shifted = aircraft - float(aircraft.max())
        probability = np.exp(shifted)
        probability /= float(probability.sum())
        full = np.zeros(25, dtype=np.float64)
        full[indices] = probability
        result[uid] = full
    return result


def load_aircraft_bundle(
    directory: str | Path,
    records: Sequence[ProposalRecord],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load exact OOF identity logits and D4 aircraft probabilities."""

    root = Path(directory).expanduser().resolve()
    expected = {
        item.proposal_uid for item in records if item.category_id in AIRCRAFT_CLASS_IDS
    }
    identity: dict[str, np.ndarray] = {}
    d4: dict[str, np.ndarray] = {}
    for fold in (0, 1, 2):
        path = root / f"fold_{fold}_aircraft_bundle.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            uids = np.asarray(payload["proposal_uids"])
            identity_values = np.asarray(payload["identity_logits"], dtype=np.float32)
            d4_values = np.asarray(payload["d4_probabilities"], dtype=np.float32)
        if identity_values.shape != (len(uids), 25):
            raise ValueError(f"{path} identity shape 非法")
        if d4_values.shape != (len(uids), 20):
            raise ValueError(f"{path} D4 shape 非法")
        if not np.isfinite(identity_values).all() or not np.isfinite(d4_values).all():
            raise ValueError(f"{path} 包含 NaN/Inf")
        if not np.allclose(d4_values.sum(axis=1), 1.0, atol=2e-5):
            raise ValueError(f"{path} D4 probability 未归一化")
        for uid, identity_row, d4_row in zip(
            uids.tolist(), identity_values, d4_values, strict=True
        ):
            key = str(uid)
            if key in identity:
                raise ValueError(f"bundle UID 重复: {key}")
            identity[key] = identity_row
            full = np.zeros(25, dtype=np.float64)
            full[list(AIRCRAFT_CLASS_IDS)] = d4_row
            d4[key] = full
    if set(identity) != expected or set(d4) != expected:
        raise ValueError(
            f"bundle coverage 不完整: missing={len(expected-set(identity))}, "
            f"extra={len(set(identity)-expected)}"
        )
    return identity, d4


def condition_probabilities(
    records: Sequence[ProposalRecord],
    base_logits_dir: str | Path,
    bundle_dir: str | Path | None,
    *,
    view: str,
) -> dict[str, np.ndarray]:
    """Build a complete probability mapping while preserving bypass records."""

    base_logits = load_logits(base_logits_dir, records)
    if bundle_dir is None:
        if view != "identity":
            raise ValueError("无 bundle 时只允许 identity")
        return aircraft_conditional_probabilities(base_logits)
    identity, d4 = load_aircraft_bundle(bundle_dir, records)
    merged_logits = dict(base_logits)
    if view == "identity":
        merged_logits.update(identity)
        return aircraft_conditional_probabilities(merged_logits)
    if view != "d4":
        raise ValueError(f"未知 view={view}")
    probabilities = aircraft_conditional_probabilities(base_logits)
    probabilities.update(d4)
    return probabilities


def adaptive_d4_probabilities(
    records: Sequence[ProposalRecord],
    base_logits_dir: str | Path,
    bundle_dir: str | Path,
    *,
    maximum_identity_probability: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Use D4 only when the identity prediction is not sufficiently confident."""

    if not 0.0 < maximum_identity_probability < 1.0:
        raise ValueError("adaptive D4 probability threshold 必须位于 (0, 1)")
    base_logits = load_logits(base_logits_dir, records)
    identity_logits, d4 = load_aircraft_bundle(bundle_dir, records)
    probabilities = aircraft_conditional_probabilities(base_logits)
    identity = aircraft_conditional_probabilities(identity_logits)
    fold_by_uid = {
        item.proposal_uid: item.fold
        for item in records
        if item.category_id in AIRCRAFT_CLASS_IDS
    }
    per_fold = {fold: {"aircraft": 0, "d4": 0} for fold in (0, 1, 2)}
    for uid, identity_probability in identity.items():
        fold = fold_by_uid[uid]
        use_d4 = (
            float(identity_probability[list(AIRCRAFT_CLASS_IDS)].max())
            < maximum_identity_probability
        )
        probabilities[uid] = d4[uid] if use_d4 else identity_probability
        per_fold[fold]["aircraft"] += 1
        per_fold[fold]["d4"] += int(use_d4)
    d4_count = sum(item["d4"] for item in per_fold.values())
    aircraft_count = sum(item["aircraft"] for item in per_fold.values())
    full_view_equivalents = aircraft_count + 7 * d4_count
    audit = {
        "maximum_identity_probability": maximum_identity_probability,
        "aircraft_proposal_count": aircraft_count,
        "d4_proposal_count": d4_count,
        "d4_fraction": d4_count / aircraft_count,
        "view_equivalents": full_view_equivalents,
        "view_compute_ratio_vs_full_d4": full_view_equivalents / (8 * aircraft_count),
        "per_fold": per_fold,
    }
    return probabilities, audit


__all__ = [
    "AIRCRAFT_CLASS_IDS",
    "R11_CONTRACT_VERSION",
    "aircraft_conditional_probabilities",
    "adaptive_d4_probabilities",
    "audit_aircraft_training_rows",
    "build_aircraft_variants",
    "condition_probabilities",
    "load_aircraft_bundle",
    "load_aircraft_training_rows",
]
