import importlib.util
from pathlib import Path

import pytest


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/resume_progressive_resolution_ddp.py"
    spec = importlib.util.spec_from_file_location("progressive_ddp_resume", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_payload():
    return {
        "epoch": 1, "optimizer": {"state": "present"}, "ema": "present",
        "train_results": {"epoch": [1, 2]},
        "train_args": {
            "epochs": 40, "imgsz": 1280, "batch": 8, "seed": 42,
            "optimizer": "AdamW", "lr0": 0.0002, "lrf": 0.1,
            "mosaic": 0.0, "close_mosaic": 40, "cos_lr": True,
            "nbs": 64, "val": False, "deterministic": True,
        },
    }


def test_accept_exact_epoch2_checkpoint():
    load_script().validate_checkpoint_payload(valid_payload())


@pytest.mark.parametrize("field,value", [("batch", 24), ("epochs", 160), ("lr0", 0.002)])
def test_reject_wrong_source_recipe(field, value):
    payload = valid_payload()
    payload["train_args"][field] = value
    with pytest.raises(ValueError, match="source recipe mismatch"):
        load_script().validate_checkpoint_payload(payload)


def test_reject_incomplete_optimizer_or_epoch_history():
    payload = valid_payload()
    payload["optimizer"] = None
    with pytest.raises(ValueError, match="optimizer or EMA"):
        load_script().validate_checkpoint_payload(payload)
    payload = valid_payload()
    payload["train_results"]["epoch"] = [1]
    with pytest.raises(ValueError, match="history"):
        load_script().validate_checkpoint_payload(payload)
