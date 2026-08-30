import csv
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_merge_hera_pav_oof_orders_and_covers_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("candidate_id",))
        writer.writeheader()
        writer.writerows({"candidate_id": value} for value in range(6))

    inputs = []
    for fold, candidate_ids in enumerate(([4, 1], [5, 0], [3, 2])):
        path = tmp_path / f"fold{fold}.npz"
        ids = np.asarray(candidate_ids, dtype=np.int64)
        np.savez_compressed(
            path,
            candidate_id=ids,
            foreground_logit=ids.astype(np.float32) + 0.25,
        )
        inputs.append(path)

    output = tmp_path / "merged.npz"
    summary = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/merge_hera_pav_oof.py",
            "--inputs",
            *(str(path) for path in inputs),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
        check=True,
    )
    merged = np.load(output, allow_pickle=False)
    np.testing.assert_array_equal(merged["candidate_id"], np.arange(6))
    np.testing.assert_allclose(merged["foreground_logit"], np.arange(6) + 0.25)
