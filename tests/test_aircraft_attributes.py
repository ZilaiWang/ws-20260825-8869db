from __future__ import annotations

from pathlib import Path

import pytest

from rsdet.analysis.aircraft_attributes import (
    AIRCRAFT_FINE_NAMES,
    attribute_target_table,
    audit_aircraft_attribute_taxonomy,
    labels20_to_attribute_targets,
    load_aircraft_attribute_taxonomy,
)

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "configs/metadata/aircraft_physical_attributes_v1.yaml"


def test_formal_attribute_taxonomy_is_complete_and_shared() -> None:
    taxonomy = load_aircraft_attribute_taxonomy(TAXONOMY)
    audit = audit_aircraft_attribute_taxonomy(taxonomy)
    assert audit["status"] == "pass"
    assert audit["aircraft_class_count"] == 20
    assert audit["dimension_count"] == 5
    assert audit["separated_pair_fraction"] > 0.90
    assert audit["unique_signature_count"] < 20
    # Rare but physically diagnostic states (one/three/eight engines and two
    # distinctive planforms) are retained explicitly instead of being merged.
    assert audit["singleton_state_count"] == 6
    table = attribute_target_table(taxonomy)
    assert all(len(targets) == len(AIRCRAFT_FINE_NAMES) for targets in table.values())


def test_label_mapping_rejects_out_of_range() -> None:
    taxonomy = load_aircraft_attribute_taxonomy(TAXONOMY)
    mapped = labels20_to_attribute_targets([0, 19], taxonomy)
    assert all(len(values) == 2 for values in mapped.values())
    with pytest.raises(ValueError, match="aircraft labels20"):
        labels20_to_attribute_targets([20], taxonomy)
