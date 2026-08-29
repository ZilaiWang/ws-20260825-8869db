from __future__ import annotations

from pathlib import Path


def test_coarse_checkpoint_naming_contract() -> None:
    root = Path("/tmp/checkpoints")
    names = {
        root / f"coarse_{coarse}_fold{fold}.pt"
        for fold in (0, 1, 2)
        for coarse in ("ship", "aircraft", "vehicle")
    }
    assert len(names) == 9
    assert root / "coarse_vehicle_fold2.pt" in names
