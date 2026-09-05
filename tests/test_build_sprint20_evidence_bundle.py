from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/build_sprint20_evidence_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_sprint20_evidence_bundle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fold_groups_are_sorted_and_source_disjoint() -> None:
    split = {
        "samples": [
            {"fold": 1, "group_id": "b"},
            {"fold": 0, "group_id": "c"},
            {"fold": 0, "group_id": "a"},
            {"fold": 2, "group_id": "d"},
            {"fold": 0, "group_id": "a"},
        ]
    }

    assert MODULE._fold_groups(split) == {0: ["a", "c"], 1: ["b"], 2: ["d"]}


def test_fold_groups_reject_source_overlap() -> None:
    split = {
        "samples": [
            {"fold": 0, "group_id": "same"},
            {"fold": 1, "group_id": "same"},
            {"fold": 2, "group_id": "other"},
        ]
    }

    with pytest.raises(ValueError, match="more than one fold"):
        MODULE._fold_groups(split)


def test_atomic_write_replaces_complete_file(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_text("old", encoding="utf-8")

    MODULE._atomic_write(target, "new\n")

    assert target.read_text(encoding="utf-8") == "new\n"
    assert not list(tmp_path.glob(".*.tmp"))
