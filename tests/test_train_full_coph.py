import argparse
from pathlib import Path

from scripts.train_full_coph import build_train_args


def test_full_coph_contract_matches_admitted_continuation(tmp_path: Path) -> None:
    args = argparse.Namespace(epochs=40, imgsz=1024, batch=12, workers=8, seed=42)
    result = build_train_args(
        dataset=tmp_path / "full.yaml", output_dir=tmp_path / "out", args=args
    )
    assert result["epochs"] == 40
    assert result["lr0"] == 0.0005
    assert result["close_mosaic"] == 20
    assert result["val"] is False
    assert result["patience"] == 0
    assert result["exist_ok"] is False
