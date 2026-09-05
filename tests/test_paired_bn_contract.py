import copy
from pathlib import Path

import pytest

from rsdet.experiments.paired_trend import OFFICIAL_WEIGHT_SHA, read, sha
from scripts import run_paired_bn_recalibration as bn
from scripts import run_paired_trend as pipeline

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data/splits/paired_trend_v1"


def test_training_selection_never_includes_heldouts():
    samples = read(BUNDLE / "manifest.json")["samples"]
    rows = bn.ordered_training(samples, 8, 42)
    assert len(rows) == 3136 and {r["split"] for r in rows} == {"train"}
    assert rows == bn.ordered_training(list(reversed(samples)), 8, 42)
    assert len({r["group_id"] for r in rows}) == 190
    with pytest.raises(ValueError, match="duplicate image"):
        bn.ordered_training(samples + samples[:1], 8, 42)
    with pytest.raises(ValueError, match="equal batches"):
        bn.ordered_training(rows[:-1], 8, 42)


def test_unimplemented_recipe_change_is_rejected(tmp_path, monkeypatch):
    from rsdet.experiments.paired_trend import write

    bad = read(bn.CONFIG)
    bad["passes"] = 2
    p = tmp_path / "wrong_recipe.json"
    write(p, bad)
    monkeypatch.setattr(bn, "CONFIG", p)
    with pytest.raises(ValueError, match="unsupported change"):
        bn.build_plan(BUNDLE, tmp_path)


def parent_lineage(bundle, checkpoint):
    ids = sorted(s["image_id"] for s in read(bundle / "manifest.json")["samples"] if s["split"] == "train")
    return {
        "bundle_sha256": sha(bundle / "contract.json"), "checkpoint_sha256": sha(checkpoint),
        "train_image_ids": ids, "training_recipe": {}, "external_training_data": False,
        "ancestors": [{"kind": "audited_official_initialization", "checkpoint_sha256": OFFICIAL_WEIGHT_SHA},
                      {"kind": "trained_on_frozen_train", "checkpoint_sha256": sha(checkpoint), "train_image_ids": ids}],
    }


@pytest.mark.parametrize("mutation", ["valid", "parent", "heldout", "state", "repeat", "sha"])
def test_bn_provenance_requires_single_explicit_train_only_transformation(tmp_path, mutation):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"test parent")
    cp = tmp_path / "bn.pt"
    cp.write_bytes(b"test candidate")
    lin = bn.derive_lineage(parent_lineage(BUNDLE, parent), checkpoint_sha=sha(cp), plan_sha="a" * 64, receipt_sha="b" * 64)
    row = lin["ancestors"][-1]
    if mutation == "valid":
        pipeline.validate_lineage(lin, BUNDLE, cp)
        return
    if mutation == "parent":
        row["parent_checkpoint_sha256"] = "wrong"
    elif mutation == "heldout":
        row["train_image_ids"] = row["train_image_ids"] + [-1]
    elif mutation == "state":
        row["non_bn_state_bitwise_equal"] = False
    elif mutation == "sha":
        row["receipt_sha256"] = "x" * 64
    else:
        lin["ancestors"].append(copy.deepcopy(row))
    with pytest.raises(ValueError):
        pipeline.validate_lineage(lin, BUNDLE, cp)


def test_bn_evaluation_cannot_bypass_actual_receipts(tmp_path, monkeypatch):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    cp = tmp_path / "candidate.pt"
    cp.write_bytes(b"candidate")
    lineage = bn.derive_lineage(parent_lineage(BUNDLE, parent), checkpoint_sha=sha(cp), plan_sha="a" * 64, receipt_sha="b" * 64)
    path = tmp_path / "lineage.json"
    from rsdet.experiments.paired_trend import write
    write(path, lineage)
    monkeypatch.setattr(pipeline, "validate_bundle", lambda *_: {})
    with pytest.raises(ValueError, match="requires its original baseline"):
        pipeline.evaluate(BUNDLE, tmp_path, cp, path, tmp_path / "eval", "cpu")
    with pytest.raises(ValueError, match="not the audited output"):
        pipeline.evaluate(BUNDLE, tmp_path, cp, path, tmp_path / "eval", "cpu", tmp_path / "base/evaluation")
