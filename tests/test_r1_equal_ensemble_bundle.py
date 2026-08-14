from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.build_r1_equal_ensemble_bundle import build_fold


def _bundle(path: Path, uids: list[str], d4: np.ndarray, identity: np.ndarray) -> None:
    np.savez_compressed(
        path,
        proposal_uids=np.asarray(uids),
        identity_logits=identity.astype(np.float32),
        d4_probabilities=d4.astype(np.float32),
    )


def test_build_fold_aligns_uid_order_and_averages(tmp_path: Path) -> None:
    ce = tmp_path / "ce.npz"
    vc = tmp_path / "vc.npz"
    output = tmp_path / "out.npz"
    ce_d4 = np.zeros((2, 20), dtype=np.float32)
    ce_d4[:, 0] = 1.0
    vc_d4 = np.zeros((2, 20), dtype=np.float32)
    vc_d4[:, 1] = 1.0
    ce_identity = np.arange(50, dtype=np.float32).reshape(2, 25)
    vc_identity = ce_identity[::-1]
    _bundle(ce, ["a", "b"], ce_d4, ce_identity)
    _bundle(vc, ["b", "a"], vc_d4, vc_identity)
    audit = build_fold(ce, vc, output)
    assert audit["row_count"] == 2
    with np.load(output, allow_pickle=False) as payload:
        assert payload["proposal_uids"].tolist() == ["a", "b"]
        assert np.allclose(payload["d4_probabilities"][:, :2], 0.5)
        assert np.allclose(payload["d4_probabilities"].sum(axis=1), 1.0)
