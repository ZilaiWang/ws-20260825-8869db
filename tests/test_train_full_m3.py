from argparse import Namespace
from pathlib import Path

from scripts.train_full_m3 import build_train_args


def test_full_m3_train_contract_matches_formal_cv3() -> None:
    args = Namespace(epochs=120, imgsz=1024, batch=4, workers=8, seed=42)
    result = build_train_args(
        dataset=Path("/tmp/dataset.yaml"), output_dir=Path("/tmp/m3"), args=args
    )
    assert result["epochs"] == 120
    assert result["imgsz"] == 1024
    assert result["batch"] == 4
    assert result["lr0"] == 0.0002
    assert result["weight_decay"] == 0.0001
    assert result["val"] is False
    assert result["patience"] == 0
    assert result["exist_ok"] is False
