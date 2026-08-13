"""Auditable class-level physical attributes for aircraft refinement.

The taxonomy is semantic metadata, not an image source.  It deliberately keeps
only attributes that remain meaningful in a normalized overhead crop.  The
helpers in this module are torch-free so the mapping can be audited locally
before any GPU experiment is admitted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from rsdet.data.xh_dataset import FINE_NAMES

AIRCRAFT_ATTRIBUTE_CONTRACT = "aircraft_physical_attributes_v1"
AIRCRAFT_FINE_NAMES = tuple(FINE_NAMES[4:24])


def load_aircraft_attribute_taxonomy(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the frozen aircraft attribute dictionary."""

    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("aircraft attribute taxonomy 必须是 YAML mapping")
    if payload.get("contract_version") != AIRCRAFT_ATTRIBUTE_CONTRACT:
        raise ValueError("aircraft attribute taxonomy contract_version 非法")
    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, dict) or not dimensions:
        raise ValueError("aircraft attribute taxonomy 缺少 dimensions")

    expected_names = set(AIRCRAFT_FINE_NAMES)
    normalized: dict[str, Any] = dict(payload)
    normalized_dimensions: dict[str, Any] = {}
    for dimension, specification in dimensions.items():
        if not isinstance(dimension, str) or not dimension.strip():
            raise ValueError("attribute dimension 名称非法")
        if not isinstance(specification, dict):
            raise ValueError(f"attribute dimension {dimension} 配置非法")
        values = specification.get("values")
        assignments = specification.get("class_values")
        if (
            not isinstance(values, list)
            or len(values) < 2
            or len(values) != len(set(values))
            or not all(isinstance(item, str) and item for item in values)
        ):
            raise ValueError(f"attribute dimension {dimension} values 非法")
        if not isinstance(assignments, dict) or set(assignments) != expected_names:
            missing = sorted(expected_names - set(assignments or {}))
            extra = sorted(set(assignments or {}) - expected_names)
            raise ValueError(
                f"attribute dimension {dimension} 类覆盖非法: missing={missing}, extra={extra}"
            )
        unknown = sorted(set(assignments.values()) - set(values))
        if unknown:
            raise ValueError(f"attribute dimension {dimension} 含未知状态: {unknown}")
        normalized_dimensions[dimension] = {
            "description": str(specification.get("description", "")),
            "values": list(values),
            "class_values": {name: str(assignments[name]) for name in AIRCRAFT_FINE_NAMES},
        }
    normalized["dimensions"] = normalized_dimensions
    return normalized


def attribute_target_table(taxonomy: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return one categorical target per fine class and attribute dimension."""

    result: dict[str, tuple[int, ...]] = {}
    for dimension, specification in taxonomy["dimensions"].items():
        index = {value: position for position, value in enumerate(specification["values"])}
        result[str(dimension)] = tuple(
            index[specification["class_values"][name]] for name in AIRCRAFT_FINE_NAMES
        )
    return result


def audit_aircraft_attribute_taxonomy(taxonomy: Mapping[str, Any]) -> dict[str, Any]:
    """Quantify coverage, sharing, collisions, and pairwise discrimination."""

    targets = attribute_target_table(taxonomy)
    signatures = {
        name: tuple(targets[dimension][class_index] for dimension in targets)
        for class_index, name in enumerate(AIRCRAFT_FINE_NAMES)
    }
    signature_groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for name, signature in signatures.items():
        signature_groups[signature].append(name)

    total_pairs = 0
    separated_pairs = 0
    separation_histogram: Counter[int] = Counter()
    for left, right in combinations(AIRCRAFT_FINE_NAMES, 2):
        separation = sum(a != b for a, b in zip(signatures[left], signatures[right], strict=True))
        separation_histogram[separation] += 1
        total_pairs += 1
        separated_pairs += int(separation > 0)

    dimensions: dict[str, Any] = {}
    singleton_states = 0
    for dimension, specification in taxonomy["dimensions"].items():
        counts = Counter(specification["class_values"].values())
        singleton_states += sum(value == 1 for value in counts.values())
        dimensions[dimension] = {
            "state_count": len(specification["values"]),
            "class_count_by_state": {
                state: counts[state] for state in specification["values"]
            },
            "minimum_classes_per_state": min(counts.values()),
            "maximum_classes_per_state": max(counts.values()),
        }

    collision_groups = [
        names for names in signature_groups.values() if len(names) > 1
    ]
    return {
        "contract_version": AIRCRAFT_ATTRIBUTE_CONTRACT,
        "status": "pass",
        "aircraft_class_count": len(AIRCRAFT_FINE_NAMES),
        "dimension_count": len(targets),
        "dimensions": dimensions,
        "unique_signature_count": len(signature_groups),
        "collision_groups": collision_groups,
        "collision_class_count": sum(map(len, collision_groups)),
        "singleton_state_count": singleton_states,
        "pair_count": total_pairs,
        "separated_pair_count": separated_pairs,
        "separated_pair_fraction": separated_pairs / total_pairs,
        "separation_dimension_histogram": dict(sorted(separation_histogram.items())),
        "class_signatures": {
            name: {
                dimension: taxonomy["dimensions"][dimension]["class_values"][name]
                for dimension in targets
            }
            for name in AIRCRAFT_FINE_NAMES
        },
    }


def labels20_to_attribute_targets(
    labels20: Sequence[int], taxonomy: Mapping[str, Any]
) -> dict[str, list[int]]:
    """Map 0--19 aircraft labels to categorical attribute targets."""

    table = attribute_target_table(taxonomy)
    values = [int(label) for label in labels20]
    if any(label < 0 or label >= len(AIRCRAFT_FINE_NAMES) for label in values):
        raise ValueError("aircraft labels20 必须位于 [0, 19]")
    return {
        dimension: [targets[label] for label in values]
        for dimension, targets in table.items()
    }
