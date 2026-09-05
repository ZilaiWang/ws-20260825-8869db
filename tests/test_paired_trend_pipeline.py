"""GPU-free end-to-end contract tests; never reported as model accuracy."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsdet.experiments.paired_trend import (
    OFFICIAL_WEIGHT_SHA,
    PARTS,
    VERSION,
    group_assignment,
    read,
    sha,
    validate_bundle,
    write,
)
from scripts import run_paired_deployment_regression as deployment
from scripts import run_paired_trend as pipeline


def fixture_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    samples = []
    for part_index, part in enumerate(PARTS):
        annotations, images = [], []
        for label in range(25):
            i = 100 * part_index + label + 1
            samples.append(
                {
                    "image_id": i,
                    "relative_path": f"images/{i}.jpg",
                    "group_id": f"g{i}",
                    "split": part,
                }
            )
            images.append({"id": i, "file_name": f"images/{i}.jpg"})
            annotations.append({"image_id": i, "category_id": label, "bbox": [0, 0, 10, 10]})
        images.append({"id": 100 * part_index + 99, "file_name": "negative.jpg"})
        write(bundle / f"{part}_gt.json", {"images": images, "annotations": annotations})
    write(bundle / "manifest.json", {"samples": samples})
    write(
        bundle / "contract.json",
        {
            "version": VERSION,
            "project_config_sha256": sha(pipeline.PROJECT),
            "files": {p.name: sha(p) for p in bundle.iterdir()},
            "threshold_grid": {"start": 0.001, "stop": 1.0, "step": 0.005},
            "inference": {},
            "deadband": 0.5,
        },
    )
    return bundle


def lineage(bundle, checkpoint):
    ids = [
        s["image_id"] for s in read(bundle / "manifest.json")["samples"] if s["split"] == "train"
    ]
    return {
        "bundle_sha256": sha(bundle / "contract.json"),
        "checkpoint_sha256": sha(checkpoint),
        "train_image_ids": ids,
        "training_recipe": {"fixture": True},
        "external_training_data": False,
        "ancestors": [
            {"checkpoint_sha256": OFFICIAL_WEIGHT_SHA, "kind": "audited_official_initialization"},
            {
                "checkpoint_sha256": sha(checkpoint),
                "kind": "trained_on_frozen_train",
                "train_image_ids": list(ids),
            },
        ],
    }


def test_group_allocation_is_data_only_disjoint_and_all_classes_present():
    samples, images, annotations = [], [], []
    for group in range(12):
        for label in range(25):
            i = group * 25 + label + 1
            path = f"images/{i}.jpg"
            samples.append({"image_id": i, "relative_path": path, "group_id": f"g{group}"})
            images.append({"id": i, "file_name": path})
            annotations.append({"image_id": i, "category_id": label})
    gt = {"images": images, "annotations": annotations}
    a, audit = group_assignment(samples, gt)
    b, _ = group_assignment(list(reversed(samples)), gt)
    assert a == b
    assert set(a.values()) == set(PARTS)
    assert len(a) == 12 and audit["train_minimum_fraction_per_fine"] == 0.5
    bad = copy.deepcopy(gt)
    bad["annotations"] = [
        r for r in bad["annotations"] if r["category_id"] != 9 or r["image_id"] <= 50
    ]
    with pytest.raises(ValueError, match="fewer than three"):
        group_assignment(samples, bad)


def test_lineage_rejects_full_or_unverified_init(tmp_path):
    bundle = fixture_bundle(tmp_path)
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"fixture, not model")
    good = lineage(bundle, checkpoint)
    pipeline.validate_lineage(good, bundle, checkpoint)
    bad = copy.deepcopy(good)
    bad["train_image_ids"].append(101)
    with pytest.raises(ValueError, match="training universe"):
        pipeline.validate_lineage(bad, bundle, checkpoint)
    bad = copy.deepcopy(good)
    bad["ancestors"][0]["checkpoint_sha256"] = "full"
    with pytest.raises(ValueError, match="initialization"):
        pipeline.validate_lineage(bad, bundle, checkpoint)
    bad = copy.deepcopy(good)
    bad["ancestors"][-1]["train_image_ids"].append(101)
    with pytest.raises(ValueError, match="heldout"):
        pipeline.validate_lineage(bad, bundle, checkpoint)


def test_frozen_gt_tampering_detected(tmp_path):
    bundle = fixture_bundle(tmp_path)
    validate_bundle(bundle, pipeline.PROJECT)
    (bundle / "confirmation_gt.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_bundle(bundle, pipeline.PROJECT)


def test_end_to_end_selection_confirmation_and_cache(tmp_path, monkeypatch):
    bundle = fixture_bundle(tmp_path)
    # Only expensive image/checkpoint execution is replaced. Real scoring and
    # control flow run over a complete 25-fine + negative-image fixture.
    original_validate = pipeline.validate_bundle
    monkeypatch.setattr(pipeline, "validate_bundle", lambda b, p, d=None: original_validate(b, p))
    calls = []

    def infer(checkpoint, gt, root, output, config, device):
        calls.append((checkpoint.stem, gt.stem))
        a = read(gt)["annotations"]
        n = 20 if checkpoint.stem == "baseline" else (25 if checkpoint.stem == "better" else 10)
        records = [{**x, "score": 0.8} for x in a[:n]]
        # Rejectable negative at low score participates in threshold selection.
        records.append(
            {
                "image_id": read(gt)["images"][-1]["id"],
                "category_id": 24,
                "bbox": [0, 0, 10, 10],
                "score": 0.1,
            }
        )
        p = output / "predictions_low.json"
        write(p, records)
        return p

    monkeypatch.setattr(pipeline, "infer", infer)
    for name in ("baseline", "better", "worse"):
        checkpoint = tmp_path / f"{name}.pt"
        checkpoint.write_bytes(name.encode())
        lp = tmp_path / f"{name}_lineage.json"
        write(lp, lineage(bundle, checkpoint))
        pipeline.evaluate(
            bundle,
            tmp_path,
            checkpoint,
            lp,
            tmp_path / name,
            "cpu",
            None if name == "baseline" else tmp_path / "baseline",
        )
    base = read(tmp_path / "baseline/review.json")
    better = read(tmp_path / "better/review.json")
    worse = read(tmp_path / "worse/review.json")
    assert base["next_action"] == "baseline_cached"
    assert better["next_action"] == "deployment_regression"
    assert better["development"]["delta_total_score"] is None
    assert not worse["confirmation_evaluated"]
    assert ("worse", "confirmation_gt") not in calls
    assert 0.1 < better["threshold"] <= 0.8
    assert read(tmp_path / "better/confirmation_metrics.json")["threshold"] == better["threshold"]
    assert not better["automatic_full_or_submission_admission"]
    assert not (tmp_path / "better/confirmation/threshold.json").exists()
    with pytest.raises(FileExistsError):
        pipeline.evaluate(
            bundle,
            tmp_path,
            tmp_path / "better.pt",
            tmp_path / "better_lineage.json",
            tmp_path / "better",
            "cpu",
            tmp_path / "baseline",
        )


@pytest.mark.parametrize(
    "rows",
    [
        [{"image_id": 9999, "category_id": 24, "bbox": [0, 0, 1, 1], "score": 0.5}],
        [{"image_id": 101, "category_id": 99, "bbox": [0, 0, 1, 1], "score": 0.5}],
        [{"image_id": 101, "category_id": 24, "bbox": [0, 0, 0, 1], "score": 0.5}],
    ],
)
def test_invalid_predictions_not_silently_dropped(tmp_path, rows):
    bundle = fixture_bundle(tmp_path)
    p = tmp_path / "pred.json"
    write(p, rows)
    with pytest.raises(ValueError):
        pipeline.metrics(bundle / "development_gt.json", p, 0.5)


def test_epoch_completion_validation(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text("epoch,train/box_loss\n1,.2\n2,.1\n")
    pipeline.check_epochs(p, 2)
    p.write_text("epoch,train/box_loss\n1,.2\n3,.1\n")
    with pytest.raises(ValueError):
        pipeline.check_epochs(p, 2)
    p.write_text("epoch,train/box_loss\n1,.2\n2,nan\n")
    with pytest.raises(ValueError, match="nonfinite"):
        pipeline.check_epochs(p, 2)


def test_anchor_audit_uses_declared_pair_and_never_fits_confirmation(tmp_path, monkeypatch):
    from scripts import run_paired_anchor_audit as anchor

    bundle = fixture_bundle(tmp_path)
    base = tmp_path / "base"
    output = tmp_path / "anchor"
    plan = {"bundle_sha256": sha(bundle / "contract.json"),
            "recipe": {"foundation": {"epochs": 1}, "adaptation": {"epochs": 1}}}
    write(base / "plan.json", plan)
    official = Path(__file__).resolve().parents[1] / "reports/experiments/formal_attempt2_20260903/analysis_summary.json"
    declaration = anchor.prepare(base, bundle, official, output)
    assert declaration["foundation_inference"]["imgsz"] == 1024
    assert declaration["official"]["delta_total_score"] == pytest.approx(4.4679559)
    previous = OFFICIAL_WEIGHT_SHA
    for stage in ("foundation", "adaptation"):
        cp = base / stage / "run/weights/last.pt"
        cp.parent.mkdir(parents=True)
        cp.write_bytes(stage.encode())
        csv_path = base / stage / "run/results.csv"
        csv_path.write_text("epoch,train/box_loss\n1,0.1\n")
        write(base / stage / "complete.json", {
            "checkpoint_sha256": sha(cp), "initial_sha256": previous,
            "plan_sha256": sha(base / "plan.json"), "results_sha256": sha(csv_path),
        })
        previous = sha(cp)
    parent = base / "foundation/run/weights/last.pt"
    final = base / "adaptation/run/weights/last.pt"
    lin = lineage(bundle, final)
    lin["ancestors"].insert(1, {**lin["ancestors"][-1], "checkpoint_sha256": sha(parent)})
    write(base / "lineage.json", lin)
    with pytest.raises(ValueError, match="before foundation completes"):
        anchor.prepare(base, bundle, official, tmp_path / "too_late")
    original_validate = pipeline.validate_bundle
    monkeypatch.setattr(pipeline, "validate_bundle", lambda b, p, d=None: original_validate(b, p))
    calls = []

    def fake_infer(cp, gt, data, out, config, device):
        calls.append((cp.parent.parent.parent.name, gt.stem, config))
        rows = read(gt)["annotations"]
        if cp == parent:
            rows = rows[:20]
        pred = out / "predictions_low.json"
        write(pred, [{**a, "score": 0.8} for a in rows])
        write(out / "inference.json", {"request": {"config": config, "checkpoint_sha256": sha(cp)}})
        return pred

    monkeypatch.setattr(pipeline, "infer", fake_infer)
    pipeline.evaluate(bundle, tmp_path, final, base / "lineage.json", base / "evaluation", "cpu")
    original_select = pipeline.select_threshold
    selects = []

    def select(gt, pred, grid):
        selects.append(gt.name)
        return original_select(gt, pred, grid)

    monkeypatch.setattr(pipeline, "select_threshold", select)
    result = anchor.execute(bundle, tmp_path, output, "cpu")
    assert selects == ["development_gt.json"]
    assert result["prospective_comparisons"] == 0 and result["hit_rate_estimate"] is None
    assert result["local_total_delta"] is None
    assert result["comparisons"]["confirmation"]["direction"] == "positive"
    assert result["direction_checks"]["confirmation"]["quality_direction_matches_known_official_pair"]
    assert not result["automatic_full_or_submission_admission"]
    assert calls[-1][2]["imgsz"] == 1024
    with pytest.raises(FileExistsError):
        anchor.execute(bundle, tmp_path, output, "cpu")


def test_official_anchor_rejects_score_drift():
    from scripts.run_paired_anchor_audit import official_anchor

    source = Path(__file__).resolve().parents[1] / "reports/experiments/formal_attempt2_20260903/analysis_summary.json"
    raw = read(source)
    raw["submissions"]["v2.0"]["metrics"]["score_details"]["total_score"] += 1
    with pytest.raises(ValueError, match="displayed precision"):
        official_anchor(raw)


def test_real_split_has_all_classes_and_preserves_tu160_train_support():
    root = Path(__file__).resolve().parents[1] / "data/splits/paired_trend_v1"
    validate_bundle(root, pipeline.PROJECT)
    samples = read(root / "manifest.json")["samples"]
    assert len(samples) == 4481
    assert all(
        len({s["split"] for s in samples if s["group_id"] == g}) == 1
        for g in {s["group_id"] for s in samples}
    )
    support = read(root / "support.json")
    assert support["train"]["fine_gt"]["9"] == 352
    assert support["development"]["fine_gt"]["9"] + support["confirmation"]["fine_gt"]["9"] == 9
    assert all(len(support[p]["fine_gt"]) == 25 for p in PARTS)


def test_baseline_training_orchestration_no_resume_no_holdout(tmp_path, monkeypatch):
    out = tmp_path / "train"
    out.mkdir()
    weights = tmp_path / "official.pt"
    weights.write_bytes(b"fixture official weight")
    recipe = read(
        Path(__file__).resolve().parents[1] / "data/splits/paired_trend_v1/contract.json"
    )["baseline"]
    plan = {"bundle_sha256": "fixture", "recipe": recipe, "train_image_ids": [1, 2]}
    write(out / "plan.json", plan)
    (out / "dataset.yaml").write_text("train: train_only.txt\nval: train_only.txt\n")
    monkeypatch.setattr(pipeline, "materialize", lambda *a: plan)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    )
    calls = []

    class FakeYOLO:
        def __init__(self, path):
            self.path = path
            self.names = dict(enumerate(pipeline.FINE_NAMES))

        def train(self, **kwargs):
            calls.append(kwargs)
            stage = Path(kwargs["project"]) / "run"
            (stage / "weights").mkdir(parents=True)
            (stage / "weights/last.pt").write_bytes(f"fixture {self.path}".encode())
            (stage / "results.csv").write_text(
                "epoch,train/box_loss\n"
                + "".join(f"{i},0.1\n" for i in range(1, kwargs["epochs"] + 1))
            )

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr("rsdet.innovation.trainers.rotate90_augmentations", lambda **k: [])
    pipeline.train_baseline(tmp_path, tmp_path, weights, out, "cuda:0")
    assert [c["epochs"] for c in calls] == [160, 40]
    assert [c["imgsz"] for c in calls] == [1024, 1280]
    assert all(c["val"] is False and c["patience"] == 0 and "resume" not in c for c in calls)
    assert calls[1]["mosaic"] == 0
    pipeline.train_baseline(tmp_path, tmp_path, weights, out, "cuda:0")
    assert len(calls) == 2  # Completed SHA-checked stages reused, never retrained.
    (out / "adaptation/run/weights/last.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cache mismatch"):
        pipeline.train_baseline(tmp_path, tmp_path, weights, out, "cuda:0")


def test_b_regression_exercises_real_output_schema_with_fake_gpu(tmp_path, monkeypatch):
    import json

    from PIL import Image

    from rsdet.contracts import Prediction
    from rsdet.pipeline import large_image
    from rsdet.submission import competition

    background = tmp_path / "bg"
    background.mkdir()
    image_path = background / "small.png"
    Image.new("RGB", (16, 16)).save(image_path)
    row = {
        "image_id": 0,
        "file_name": "small.png",
        "width": 16,
        "height": 16,
        "sha256": sha(image_path),
    }
    manifest = background / "background_100mp_manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    bundle = tmp_path / "bundle"
    write(bundle / "contract.json", {"fixture": True})
    review = tmp_path / "review"
    write(review / "review.json", {"bundle_sha256": sha(bundle / "contract.json"), "artifacts": {}})
    write(review / "lineage.json", {})
    write(review / "threshold.json", {"threshold": 0.5})
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not a model")
    config = tmp_path / "config.json"
    write(
        config,
        {
            "bundle_sha256": sha(bundle / "contract.json"),
            "background_manifest_sha256": sha(manifest),
            "submission_config": {"model": {}},
            "translations": [[0, 0], [0, 0]],
            "placements": [
                {
                    "x": 0,
                    "y": 0,
                    "source": {"file_name": "small.png"},
                    "annotations": [{"category_id": 0, "bbox": [1, 1, 5, 5]}],
                }
            ],
        },
    )
    monkeypatch.setattr(deployment, "validate_bundle", lambda *a: {"inference": {}})
    monkeypatch.setattr(deployment, "validate_lineage", lambda *a: None)
    monkeypatch.setattr(deployment, "canvas", lambda *a: Image.new("RGB", (16, 16)))

    def fake_infer(*args):
        output = args[3] / "predictions_low.json"
        write(output, [])
        return output

    monkeypatch.setattr(deployment, "infer", fake_infer)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="fixture",
            cuda=SimpleNamespace(
                synchronize=lambda *a: None,
                get_device_properties=lambda *a: SimpleNamespace(name="fixture", uuid="fixture"),
            ),
        ),
    )
    monkeypatch.setattr(
        competition,
        "CompetitionDetector",
        lambda *a: SimpleNamespace(detector=None, pipeline_config=None, predict=lambda *a: []),
    )
    monkeypatch.setattr(
        large_image, "run_pipeline", lambda *a, **k: (Prediction(0, [], [], []), None)
    )
    result = deployment.execute(
        bundle, config, background, background, checkpoint, review, tmp_path / "output", "cuda:0"
    )
    assert result["status"] == "pass" and result["engineering_only"]
    assert not result["container_gpu_tested"]
    assert result["background_fp"] == 0
    assert all(c["fn"] == 1 for c in result["entrypoint_parity"])
    assert result["output_counts"] == {"images": 2, "objects": 0}
