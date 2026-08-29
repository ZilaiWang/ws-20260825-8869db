from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_pseudo_dino_open_set_heads.py"
SPEC = importlib.util.spec_from_file_location("train_pseudo_dino_open_set_heads", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(*, foreground: bool, fine: int = 0) -> dict[str, str]:
    return {
        "is_foreground": "1" if foreground else "0",
        "support_gt_category_id": str(fine) if foreground else "",
    }


def test_target_class() -> None:
    assert MODULE.target_class(_row(foreground=True, fine=7)) == 7
    assert MODULE.target_class(_row(foreground=False)) == 25


def test_balanced_batches_are_deterministic_and_balanced() -> None:
    rows = [_row(foreground=False) for _ in range(12)]
    rows += [_row(foreground=True, fine=value) for value in (0, 1, 4, 5, 24)]
    first = MODULE.balanced_batches(rows, batch_size=8, batches_per_epoch=3, seed=9)
    second = MODULE.balanced_batches(rows, batch_size=8, batches_per_epoch=3, seed=9)
    assert first == second
    assert all(sum(MODULE.target_class(rows[index]) == 25 for index in batch) == 4 for batch in first)
    assert all(len(batch) == 8 for batch in first)
