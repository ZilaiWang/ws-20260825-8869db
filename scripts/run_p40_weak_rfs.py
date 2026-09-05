#!/usr/bin/env python3
"""Single-GPU frozen P40 sampling ablation; no full training or deployment."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.experiments.fixed_proxy import review_quality_delta
from rsdet.utils.config import load_config
from scripts.analyze_p40_vehicle_zoom import metrics, read, sha

CONFIG = ROOT / "configs/experiments/p40_weak_rfs_v1.json"
BASE = Path("/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-CV3-V1")
BASE_EVAL = Path("/root/autodl-tmp/results/SCALEROUTE-PLAN15-PROGRESSIVE40-FIXED-BENCHMARKS-V1")
SOURCES = [Path("/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1/s1024")] + [
    Path(f"/root/autodl-tmp/results/YOLO-SCALE-CV3-CONFIRM-V1/s1024/fold_{f}") for f in (1, 2)]
GTS = [Path("/root/autodl-tmp/capscale-assets/instances_val.json")] + [
    Path(f"/root/autodl-tmp/capscale-cv3-assets/fold_{f}/instances_val.json") for f in (1, 2)]
PSEUDO = {"hard": Path("/root/autodl-tmp/pseudo10k-trial-mix-local"),
          "sentinel": Path("/root/autodl-tmp/pseudo10k-trial-mix-sentinel-v1")}
LEGACY_BASELINE_SUMMARY_SHA256 = {
    "hard": "b93ade16a1b487467e13384e37761232d8eae3e5775d494f736b84021f06e5d0",
    "sentinel": "6ab668a68c2beb3c39de0bc6a36b629ce7fc8d5ac8e9a0c1eaf71581cb12e1a9",
}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def call(out, stage, argv):
    (out / "status.txt").write_text(stage+"\n")
    env = os.environ.copy()
    local_paths = [str(ROOT / "src"), str(ROOT)]
    if env.get("PYTHONPATH"):
        local_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(local_paths)
    with (out / f"{stage}.log").open("w") as log:
        subprocess.run(
            [sys.executable, "-u", *map(str, argv)],
            cwd=ROOT,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )


def validate_inference_geometry(summary, *, allow_legacy_imgsz=False):
    """Validate geometry while accepting only the SHA-pinned legacy baseline schema."""
    if summary.get("imgsz") is None:
        if not allow_legacy_imgsz:
            raise ValueError("inference receipt has no imgsz")
    elif summary["imgsz"] != 1280:
        raise ValueError("inference image size differs")
    pipeline = summary.get("pipeline", {})
    if any(pipeline.get(k) != v for k, v in (("tile_size", 1024), ("overlap", 256))):
        raise ValueError("inference tiling geometry differs")


def audit_inputs(config_path=CONFIG):
    config = read(config_path)
    from rsdet.innovation.weak_rfs import weak_image_weights

    records, heldout_sets = [], []
    whole = {str((Path("/root/autodl-tmp/data") / r["file_name"]).resolve())
             for r in read(BASE / "aggregate/ground_truth.json")["images"]}
    parent_contract = read(ROOT / "configs/experiments/p40_vehicle_zoom_rescue_v1.json")
    if sha(BASE / "aggregate/crossfit_frontier.json") != parent_contract["frontier_sha256"]:
        raise ValueError("baseline calibration changed")
    source_sets = []
    for condition, folder in PSEUDO.items():
        if sha(folder / "ground_truth.json") != parent_contract["ground_truth_sha256"][condition]:
            raise ValueError("fixed stress GT changed")
        source_sets.append({Path(s).name for r in read(folder / "ground_truth.json")["images"] for s in r["source_images"]})
    if any(len(s) != 600 for s in source_sets) or source_sets[0] & source_sets[1]:
        raise ValueError("stress sources overlap/incomplete")
    for fold, source in enumerate(SOURCES):
        weight = source / "runs/foundation/weights/last.pt"
        if sha(weight) != config["source_weight_sha256"][fold]:
            raise ValueError(f"wrong source weight fold {fold}")
        old = read(BASE / f"fold_{fold}/adaptation/training_contract.json")
        for key in ("epochs", "imgsz", "batch", "workers", "seed", "lr0", "lrf", "optimizer", "mosaic", "rotate90_p"):
            if old[key] != config[key]:
                raise ValueError(f"unmatched baseline hyperparameter: {key}")
        if old["initial_checkpoint_sha256"] != sha(weight):
            raise ValueError("baseline used another initialization")
        data = yaml.safe_load((source / "dataset.yaml").read_text())
        if sha(source / "dataset.yaml") != old["data_sha256"]:
            raise ValueError("baseline data YAML changed")
        if data["nc"] != 25 or list(data["names"]) != [str(i) for i in range(25)]:
            raise ValueError("wrong taxonomy")
        train = [str(Path(p).resolve()) for p in Path(data["train"]).read_text().splitlines() if p]
        val = [str(Path(p).resolve()) for p in Path(data["val"]).read_text().splitlines() if p]
        if len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val) or len(train)+len(val) != 4481:
            raise ValueError("invalid source partition")
        expected = {str((Path(data["path"]) / r["file_name"]).resolve()) for r in read(GTS[fold])["images"]}
        if set(val) != expected:
            raise ValueError("val partition differs from frozen GT")
        if set(train) != whole - expected:
            raise ValueError("train is not the exact complement of heldout")
        labels, label_shas = [], {}
        for image in train:
            path = Path(image)
            if not path.is_file():
                raise FileNotFoundError(path)
            label = Path(str(path).replace("/images/", "/labels/")).with_suffix(".txt")
            classes = set()
            for line in label.read_text().splitlines():
                values = list(map(float, line.split()))
                if len(values) != 5 or not all(math.isfinite(v) for v in values) or values[0] != int(values[0]) or not all(0 <= v <= 1 for v in values[1:]):
                    raise ValueError(f"invalid training label {label}")
                classes.add(int(values[0]))
            labels.append(classes)
            label_shas[str(label)] = sha(label)
        _, sampling = weak_image_weights(labels)
        records.append({"fold": fold, "weight_sha256": sha(weight), "data_sha256": sha(source / "dataset.yaml"),
            "train_list_sha256": sha(Path(data["train"])), "val_list_sha256": sha(Path(data["val"])),
            "train_images": len(train), "val_images": len(val), "sampling": sampling,
            "training_label_shas": label_shas})
        heldout_sets.append(set(val))
    if len(set.union(*heldout_sets)) != 4481 or sum(map(len, heldout_sets)) != 4481:
        raise ValueError("CV3 heldout sets overlap or have missing images")
    contract_paths = [p for directory in ("scripts", "src", "configs") for p in (ROOT / directory).rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".yaml", ".sh")]
    return json.loads(json.dumps({"config": config, "config_sha256": sha(config_path), "folds": records,
            "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in contract_paths}}, allow_nan=False))


def evaluate(out, condition, config_path=CONFIG):
    gt_path = PSEUDO[condition] / "ground_truth.json"
    if sha(gt_path) != read(ROOT / "configs/experiments/p40_vehicle_zoom_rescue_v1.json")["ground_truth_sha256"][condition]:
        raise ValueError("wrong fixed stress GT")
    gt = load_coco_ground_truth(gt_path)
    folds = {r["id"]: r["fold"] for r in read(gt_path)["images"]}
    protocol = parse_evaluation_protocol(load_config(ROOT / "configs/project.yaml"))
    result = {}
    for name, predictions, frontier in (
        ("baseline", BASE_EVAL / condition / "progressive_i1280/predictions.json", BASE / "aggregate/crossfit_frontier.json"),
        ("candidate", out / condition / "predictions.json", out / "aggregate/crossfit_frontier.json"),
    ):
        summary_path = predictions.parent / "run_summary.json"
        summary = read(summary_path)
        if sha(predictions) != summary["predictions_sha256"] or summary["selected_folds"] != [0, 1, 2]:
            raise ValueError("bad prediction receipt")
        parent_contract = read(ROOT / "configs/experiments/p40_vehicle_zoom_rescue_v1.json")
        expected_weights = parent_contract["weights_sha256"] if name == "baseline" else [
            sha(out / f"fold_{f}/adaptation/runs/resolution_adaptation/weights/last.pt") for f in range(3)]
        if [r["weight_sha256"] for r in summary["folds"]] != expected_weights:
            raise ValueError("wrong inference weights")
        if name == "baseline" and "imgsz" not in summary:
            # This benchmark was generated before run_summary recorded imgsz.  Its
            # immutable receipt is the only legacy schema accepted here; candidate
            # receipts must always state their image size explicitly.
            if sha(summary_path) != LEGACY_BASELINE_SUMMARY_SHA256[condition]:
                raise ValueError("unrecognized legacy baseline receipt")
            validate_inference_geometry(summary, allow_legacy_imgsz=True)
        else:
            validate_inference_geometry(summary)
        if any(r["source_fold"] != folds[r["image_id"]] for r in read(predictions)):
            raise ValueError("wrong heldout prediction fold")
        raw = load_coco_predictions(predictions)
        thresholds = {int(k): v for k, v in read(frontier)["frontiers"]["0.150"]["crossfit_thresholds"].items()}
        selected = {i: [r for r in raw.get(i, []) if r["score"] >= thresholds[folds[i]]] for i in folds}
        result[name] = metrics(gt, selected, protocol)
        result[f"{name}_thresholds"] = thresholds
        result[f"{name}_pred_sha256"] = sha(predictions)
    result["review"] = review_quality_delta(result["baseline"]["platform"], result["candidate"]["platform"],
        stage=condition, minimum=read(config_path)[f"{condition}_quality_delta_min_exclusive"])
    result["status"] = "complete"
    result["no_posthoc_class_routing"] = True
    result["not_official_score_prediction"] = True
    write(out / f"{condition}_comparison.json", result)
    return result["review"]["direction_pass"]


def recover_evaluation_keyerror(out):
    """Continue only the evaluation stage after the legacy-summary KeyError."""
    if (out / "status.txt").read_text().strip() != "failed_KeyError":
        raise ValueError("recovery is limited to the recorded legacy-summary KeyError")
    frozen = read(out / "preflight.json")
    if frozen["config_sha256"] != sha(CONFIG) or frozen["config"] != read(CONFIG):
        raise ValueError("frozen experiment configuration changed")
    current_self = sha(Path(__file__))
    old_self = frozen["code_sha256"][str(Path(__file__).resolve().relative_to(ROOT))]
    for relative, expected in frozen["code_sha256"].items():
        if relative == str(Path(__file__).resolve().relative_to(ROOT)):
            continue
        if sha(ROOT / relative) != expected:
            raise ValueError(f"non-recovery code changed: {relative}")
    weight_receipts = []
    for fold in range(3):
        run = out / f"fold_{fold}/adaptation/runs/resolution_adaptation"
        rows = list(csv.DictReader((run / "results.csv").open()))
        if [int(r["epoch"]) for r in rows] != list(range(1, 41)) or not all(
            math.isfinite(float(v)) for r in rows for k, v in r.items() if "loss" in k
        ):
            raise ValueError("completed training evidence is invalid")
        weight = run / "weights/last.pt"
        training = read(out / f"fold_{fold}/adaptation/training_result.json")
        sampling = read(run / "weak_rfs_audit.json")
        if sha(weight) != training["checkpoint_sha256"]:
            raise ValueError("training weight receipt mismatch")
        for key in ("images", "image_counts_by_fine", "weak_factors", "samples_per_epoch"):
            if sampling[key] != frozen["folds"][fold]["sampling"][key]:
                raise ValueError(f"actual dataloader differs from preflight: {key}")
        weight_receipts.append({"fold": fold, "last_pt_sha256": sha(weight)})
    hard_summary = read(out / "hard/run_summary.json")
    if sha(out / "hard/predictions.json") != hard_summary["predictions_sha256"]:
        raise ValueError("completed Hard inference receipt is invalid")
    write(out / "recovery_legacy_summary_keyerror.json", {
        "status": "evaluation_only_recovery",
        "reason": "baseline run_summary predates the imgsz field; immutable summary SHA is now required",
        "frozen_driver_sha256": old_self,
        "recovery_driver_sha256": current_self,
        "training_not_restarted": True,
        "hard_inference_not_restarted": True,
        "weights": weight_receipts,
    })
    (out / "status.txt").write_text("recover_evaluate_hard\n")
    if not evaluate(out, "hard"):
        (out / "status.txt").write_text("complete_stopped_hard\n")
        return
    weights = [out / f"fold_{f}/adaptation/runs/resolution_adaptation/weights/last.pt" for f in range(3)]
    call(out, "infer_sentinel", ["scripts/run_multifamily_cv3_pseudo_eval.py", "--pseudo-root", PSEUDO["sentinel"],
        "--family", "yolo", "--weights", *weights, "--output-dir", out / "sentinel", "--score-floor", .001,
        "--batch-size", 4, "--device", "cuda:0", "--imgsz", 1280, "--tile-size", 1024, "--overlap", 256])
    if not evaluate(out, "sentinel"):
        (out / "status.txt").write_text("complete_stopped_sentinel\n")
        return
    (out / "status.txt").write_text("complete_positive_review_required_no_full\n")


def run(out, frozen):
    if audit_inputs() != frozen:
        raise ValueError("frozen assets/code changed")
    weights = []
    for f, source in enumerate(SOURCES):
        foldout = out / f"fold_{f}"
        call(out, f"train_fold_{f}", ["scripts/train_progressive_resolution_adaptation.py", "--weak-rfs",
            "--weights", source / "runs/foundation/weights/last.pt", "--data", source / "dataset.yaml",
            "--output", foldout / "adaptation", "--epochs", 40, "--imgsz", 1280, "--batch", 8,
            "--workers", 4, "--device", "cuda:0", "--seed", 42, "--lr0", .0002, "--lrf", .10, "--rotate90-p", 1.])
        weights.append(foldout / "adaptation/runs/resolution_adaptation/weights/last.pt")
        rows = list(csv.DictReader((foldout / "adaptation/runs/resolution_adaptation/results.csv").open()))
        if [int(r["epoch"]) for r in rows] != list(range(1, 41)) or not all(
            math.isfinite(float(v)) for r in rows for k, v in r.items() if "loss" in k):
            raise ValueError("training epoch count/nonfinite loss")
        training = read(foldout / "adaptation/training_result.json")
        if sha(weights[-1]) != training["checkpoint_sha256"]:
            raise ValueError("training weight receipt mismatch")
        sampling = read(foldout / "adaptation/runs/resolution_adaptation/weak_rfs_audit.json")
        for key in ("images", "image_counts_by_fine", "weak_factors", "samples_per_epoch"):
            if sampling[key] != frozen["folds"][f]["sampling"][key]:
                raise ValueError(f"actual dataloader differs from preflight: {key}")
        call(out, f"materialize_fold_{f}", ["scripts/materialize_yolo_cross_resolution_infer_config.py",
            "--source", source / "resolved_infer.yaml", "--output", foldout / "resolved_infer.yaml",
            "--predictions", foldout / "predictions_low.json", "--imgsz", 1280, "--checkpoint", weights[-1]])
        call(out, f"infer_fold_{f}", ["scripts/infer_cv3_oof.py", "--config", foldout / "resolved_infer.yaml"])
    call(out, "aggregate", ["scripts/merge_cv3_coco_ledgers.py", "--gt", *GTS, "--pred",
        *[out / f"fold_{f}/predictions_low.json" for f in range(3)], "--output-gt", out / "aggregate/ground_truth.json",
        "--output-pred", out / "aggregate/predictions_low.json"])
    call(out, "calibrate_normal", ["scripts/analyze_cv3_oof_pseudo_frontier.py", "--gt", out / "aggregate/ground_truth.json",
        "--pred", out / "aggregate/predictions_low.json", "--output", out / "aggregate/crossfit_frontier.json", "--threshold-step", .005])
    for condition in ("hard", "sentinel"):
        call(out, f"infer_{condition}", ["scripts/run_multifamily_cv3_pseudo_eval.py", "--pseudo-root", PSEUDO[condition],
            "--family", "yolo", "--weights", *weights, "--output-dir", out / condition, "--score-floor", .001,
            "--batch-size", 4, "--device", "cuda:0", "--imgsz", 1280, "--tile-size", 1024, "--overlap", 256])
        if not evaluate(out, condition):
            (out / "status.txt").write_text(f"complete_stopped_{condition}\n")
            return
    (out / "status.txt").write_text("complete_positive_review_required_no_full\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "run", "smoke", "recover-evaluation-keyerror"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.action == "smoke":
        audit_inputs()
        a.output.mkdir(parents=True, exist_ok=False)
        data = yaml.safe_load((SOURCES[0] / "dataset.yaml").read_text())
        train = Path(data["train"]).read_text().splitlines()
        subset = [train[min(len(train)-1, i*len(train)//32)] for i in range(32)]
        (a.output / "train.txt").write_text("\n".join(subset)+"\n")
        (a.output / "val.txt").write_text("\n".join(subset[:8])+"\n")
        data.update(train=str(a.output / "train.txt"), val=str(a.output / "val.txt"))
        (a.output / "dataset.yaml").write_text(yaml.safe_dump(data))
        call(a.output, "smoke_one_epoch_not_scientific", ["scripts/train_progressive_resolution_adaptation.py", "--weak-rfs",
            "--weights", SOURCES[0] / "runs/foundation/weights/last.pt", "--data", a.output / "dataset.yaml",
            "--output", a.output / "adaptation", "--epochs", 1, "--imgsz", 1280, "--batch", 8,
            "--workers", 0, "--device", "cuda:0", "--seed", 42, "--lr0", .0002, "--lrf", .10, "--rotate90-p", 1.])
        (a.output / "status.txt").write_text("smoke_pass_not_a_candidate\n")
    elif a.action == "prepare":
        audit = audit_inputs()
        a.output.mkdir(parents=True, exist_ok=False)
        write(a.output / "preflight.json", audit)
        (a.output / "status.txt").write_text("prepared_not_started\n")
        print(json.dumps({"folds": [{k: r[k] for k in ("fold", "train_images", "val_images", "sampling")} for r in audit["folds"]]}, indent=2))
    elif a.action == "run":
        import fcntl

        with (a.output / "execution.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (a.output / "status.txt").read_text().strip() != "prepared_not_started":
                raise ValueError("refusing duplicate/restart of non-prepared experiment")
            try:
                run(a.output, read(a.output / "preflight.json"))
            except BaseException as exc:
                (a.output / "status.txt").write_text(f"failed_{type(exc).__name__}\n")
                raise
    else:
        recover_evaluation_keyerror(a.output)


if __name__ == "__main__":
    main()
