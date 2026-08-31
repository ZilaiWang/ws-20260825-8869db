from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


def test_selective_augmentation_keeps_nonroute_rows_zero(tmp_path: Path) -> None:
    cache = tmp_path / "cache.npz"
    scores = tmp_path / "scores.npz"
    output = tmp_path / "output.npz"
    summary = tmp_path / "summary.json"
    np.savez_compressed(
        cache,
        features=np.ones((3, 2), dtype=np.float16),
        fold=np.asarray([0, 1, 2]),
        image_id=np.asarray([10, 11, 12]),
        category_id=np.asarray([0, 24, 1]),
        bbox_xyxy=np.asarray([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]], dtype=np.float32),
    )
    np.savez_compressed(
        scores,
        candidate_index=np.asarray([1]),
        probabilities=np.asarray([[0.7, 0.2, 0.1]], dtype=np.float32),
        fold=np.asarray([1]),
        image_id=np.asarray([11]),
        category_id=np.asarray([24]),
        bbox_xyxy=np.asarray([[1, 1, 2, 2]], dtype=np.float32),
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "augment_omq_with_selective_open_set.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--cache",
            str(cache),
            "--open-set-scores",
            str(scores),
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
        check=True,
    )
    with np.load(output, allow_pickle=False) as payload:
        features = payload["features"]
    assert features.shape == (3, 7)
    assert np.all(features[[0, 2], 2:] == 0)
    assert features[1, 2] == 1
    assert np.allclose(features[1, 3:6], [0.7, 0.2, 0.1], atol=1e-3)
    assert np.isclose(features[1, 6], 0.5, atol=1e-3)
