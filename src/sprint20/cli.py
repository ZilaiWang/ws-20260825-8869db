"""Local audit/probe/replay tools. Every output is explicit about evidence role."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import BASE_COMMIT
from .policy import prediction_fingerprint

SOURCES = {
    "src/rsdet/contracts.py": "0104397437f9a0504db2b15dfaf2eee65061c681",
    "src/rsdet/submission/competition.py": "cd6c48d39b31117580065a6c2245af31428e5100",
    "src/rsdet/submission/aircraft_d4.py": "e8063f435e95fe76be4f1443723a463741457c25",
    "src/rsdet/models/ultralytics_adapter.py": "1fafe58a25841e859a48aa73c4548c10e4a63c87",
    "src/rsdet/pipeline/large_image.py": "f6b7f034efb23ad7f4de93eecbcbe3a1d2ca6917",
    "src/rsdet/evaluation/absolute_score.py": "1228ae6e16f078e07725dc9dd6aa0762e7d8f048",
    "src/rsdet/evaluation/official_metric.py": "bcdc99c05ba2ecde308de422b12ba231fba7591a",
    "src/rsdet/data/xh_dataset.py": "c3797c4275f3355610b93fd873f597e0ff88fb3f",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def require_base_ancestor(repo: Path, base_commit: str = BASE_COMMIT) -> str:
    """Require the audited base in history without forbidding additive commits."""
    current = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", base_commit, current],
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode != 0:
        raise RuntimeError(
            f"Audited base {base_commit} is not an ancestor of HEAD {current}: "
            f"{ancestry.stderr.strip()}"
        )
    return current


def audit(args):
    from ultralytics import YOLO

    from .heads import inspect_head, require_version

    repo = Path(args.repo).resolve()
    current = require_base_ancestor(repo)
    mismatches = {}
    for path, expected in SOURCES.items():
        raw = (repo / path).read_bytes()
        actual = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if actual != expected:
            mismatches[path] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(
            f"Audited source drift from base={BASE_COMMIT}: HEAD={current}, changed={mismatches}"
        )
    config = read(args.config)
    weight = Path(config["model"]["weight_path"])
    actual_sha = sha(weight)
    if actual_sha != config["model"]["expected_sha256"]:
        raise RuntimeError("Weight SHA mismatch")
    import importlib

    imported_paths = {}
    for module_name in (
        "rsdet.submission.competition",
        "rsdet.models.ultralytics_adapter",
        "rsdet.evaluation.official_metric",
        "rsdet.evaluation.absolute_score",
    ):
        module = importlib.import_module(module_name)
        actual_path = Path(module.__file__).resolve()
        expected_path = (repo / "src" / (module_name.replace(".", "/") + ".py")).resolve()
        if actual_path != expected_path:
            raise RuntimeError(f"Mixed source roots: {module_name} resolved to {actual_path}")
        imported_paths[module_name] = str(actual_path)
    info = inspect_head(YOLO(str(weight)))  # NO inference/fuse before inspection.
    output = {
        "base_commit": BASE_COMMIT,
        "head_commit": current,
        "base_is_ancestor": True,
        "source_blob_checks": "pass",
        "ultralytics": require_version(),
        "weight_sha256": actual_sha,
        "head": info,
        "imported_paths": imported_paths,
        "inference_executed": False,
        "semantic_warning": "FSC class24 is 发射车; not generic external vehicles",
        "formal_admission": False,
    }
    save(args.out, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def make_detector(config, head):
    from rsdet.models.ultralytics_adapter import UltralyticsDetector

    from .heads import SharedHeadCapture, assert_available, require_version, select_native_otm

    require_version()
    c = config["model"]
    if c.get("inference_adapter") != "shared_offline":
        raise ValueError("Probe requires current shared_offline P40 config")
    weight = Path(c["weight_path"])
    if sha(weight) != c["expected_sha256"]:
        raise ValueError("Weight SHA mismatch")
    detector = UltralyticsDetector(
        family="yolo",
        imgsz=int(c["imgsz"]),
        confidence=float(c["confidence"]),
        iou=float(c["iou"]),
        max_detections=int(c["max_detections"]),
        half=bool(c.get("half", True)),
        agnostic_nms=False,
    )
    detector.load(str(weight))
    assert_available(detector._model, "shared" if head == "shared" else head)
    if head == "otm":
        select_native_otm(detector)
    detector.to(str(config["device"]))
    detector.eval()
    return SharedHeadCapture(detector) if head == "shared" else detector


def probe(args):
    from PIL import Image

    from rsdet.pipeline.large_image import run_pipeline
    from rsdet.submission.competition import _pipeline_config_from_mapping, load_submission_config

    from .heads import SharedHeadPipeline

    config = load_submission_config(args.config)
    coco = read(args.coco)
    ids = [int(row["id"]) for row in coco["images"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate COCO image IDs")
    detector = make_detector(config, args.head)
    pipeline = _pipeline_config_from_mapping(config["pipeline"], "pipeline")
    shared = None
    if args.head == "shared":
        shared = SharedHeadPipeline(
            detector, pipeline, otm_labels=[0, 1, 2, 3, 24], primary_threshold=0, otm_threshold=0
        )
    output = {
        "schema": "sprint20_fused_head_cache_v1",
        "head": args.head,
        "role": args.role,
        "base_commit": BASE_COMMIT,
        "config_sha256": sha(args.config),
        "coco_sha256": sha(args.coco),
        "weight_sha256": config["model"]["expected_sha256"],
        "pipeline": asdict(pipeline),
        "inference_model": config["model"],
        "aircraft_d4_applied": False,
        "threshold_stage": "low_floor_after_production_fusion",
        "formal_admission": False,
        "images": [],
    }
    # Explicit cold-first/warm-rest timings; no claiming an official latency.
    for row in coco["images"]:
        path = Path(args.image_root) / row["file_name"]
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB")).copy()
        if (rgb.shape[1], rgb.shape[0]) != (int(row["width"]), int(row["height"])):
            raise ValueError(f"COCO/image dimensions differ: {path}")
        sync()
        started = time.perf_counter()
        if shared:
            _, timing = shared.predict_image(rgb, int(row["id"]))
            pred = shared.last_primary
        else:
            pred, timing = run_pipeline(
                rgb, detector, config=pipeline, parent_image_id=int(row["id"])
            )
            timing = timing.to_dict()
        sync()
        wall = time.perf_counter() - started
        item = {
            "image_id": int(row["id"]),
            "file_name": row["file_name"],
            "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "prediction": asdict(pred),
            "wall_seconds_excluding_read": wall,
            "timing": timing,
        }
        if shared:
            item["otm_prediction"] = asdict(shared.last_otm)
        for key in ("group_id", "source_group", "fold"):
            if key in row:
                item[key] = row[key]
        output["images"].append(item)
        print(
            f"{args.head} image={row['id']} detections={len(pred.labels)} seconds={wall:.3f}",
            flush=True,
        )
    save(args.out, output)
    if shared:
        detector.close()


def parity(args):
    from rsdet.contracts import Prediction

    native, shared = read(args.native), read(args.shared)
    a = {int(x["image_id"]): x for x in native["images"]}
    b = {int(x["image_id"]): x for x in shared["images"]}
    if set(a) != set(b) or not a:
        raise ValueError("Parity must cover exactly the same nonempty image set")
    field = "otm_prediction" if args.head == "otm" else "prediction"
    differences = []
    for i in a:
        if a[i]["pixel_sha256"] != b[i]["pixel_sha256"]:
            raise ValueError("Image pixels changed")
        pa, pb = Prediction(**a[i]["prediction"]), Prediction(**b[i][field])
        ca, cb = prediction_fingerprint(pa), prediction_fingerprint(pb)
        if ca != cb:
            differences.append(
                {
                    "image_id": i,
                    "native_only": sum((ca - cb).values()),
                    "shared_only": sum((cb - ca).values()),
                }
            )
    save(
        args.out,
        {
            "head": args.head,
            "exact_multiset_equal": not differences,
            "images": len(a),
            "differences": differences,
            "native_cache_sha256": sha(args.native),
            "shared_cache_sha256": sha(args.shared),
        },
    )
    if differences:
        raise SystemExit("PARITY FAIL: do not deploy shared mode; retain native diagnostic results")
    print("EXACT_HEAD_PARITY_PASS")


def replay(args):
    from .evaluation import (
        cache_predictions,
        evaluate,
        group_counts,
        gt_from_coco,
        paired_bootstrap,
        route_dicts,
        validate_frozen_policy,
    )

    policy = read(args.policy)
    group_declaration = validate_frozen_policy(policy)
    coco, bc, ac = read(args.coco), read(args.baseline), read(args.candidate)
    if bc.get("head") != "oto" or ac.get("head") != "otm":
        raise ValueError(
            "Scientific replay requires native OTO and native OTM caches; shared is for parity"
        )
    for key in ("weight_sha256", "coco_sha256", "pipeline", "inference_model", "role"):
        if bc.get(key) != ac.get(key):
            raise ValueError(f"Head-only comparison changed {key}")
    if bc.get("coco_sha256") != sha(args.coco):
        raise ValueError("Evaluation COCO differs from inference cache provenance")
    gt = gt_from_coco(coco)
    bp = cache_predictions(bc, policy["primary_threshold"])
    ap = cache_predictions(ac, policy["alternative_threshold"])
    if set(bp) != set(gt) or set(ap) != set(gt):
        raise ValueError("Prediction image set must include ALL GT images and negative images")
    for left, right in ((bc, ac),):
        li = {int(x["image_id"]): x["pixel_sha256"] for x in left["images"]}
        ri = {int(x["image_id"]): x["pixel_sha256"] for x in right["images"]}
        if li != ri:
            raise ValueError("Baseline/candidate pixels changed")
    combined = route_dicts(bp, ap, policy["alternative_labels"])
    baseline = evaluate(gt, bp, float(policy["baseline_latency_seconds"]))
    candidate = evaluate(gt, combined, float(policy["candidate_latency_seconds"]))
    output = {
        "policy_sha256": sha(args.policy),
        "policy": policy,
        "baseline_cache_sha256": sha(args.baseline),
        "candidate_cache_sha256": sha(args.candidate),
        "baseline_role": bc["role"],
        "candidate_role": ac["role"],
        "declared_group_separation": group_declaration,
        "actual_group_separation_checked": False,
        "baseline": baseline,
        "candidate": candidate,
        "delta_score": candidate["score"]["total_score"] - baseline["score"]["total_score"],
        "formal_admission": False,
        "warning": "Local evidence only. Full-seen/pseudo-seam/short-lineage caches cannot certify final generalization.",
    }
    if args.groups:
        groups_payload = read(args.groups)
        samples = groups_payload["samples"]
        image_to_group = {}
        for sample in samples:
            image_id = int(sample["image_id"])
            group = sample.get(args.group_key)
            if group is None or not str(group).strip():
                raise ValueError("Missing/empty source group")
            if image_id in image_to_group and image_to_group[image_id] != str(group):
                raise ValueError("One image assigned to different source groups")
            image_to_group[image_id] = str(group)
        actual_evaluation_groups = {image_to_group[i] for i in gt}
        if actual_evaluation_groups & set(group_declaration["selection_groups"]):
            raise ValueError("Actual evaluation images overlap policy selection source groups")
        if group_declaration["evaluation_groups"] and actual_evaluation_groups != set(
            group_declaration["evaluation_groups"]
        ):
            raise ValueError("Declared evaluation groups do not match actual images")
        output["actual_group_separation_checked"] = bool(group_declaration["selection_groups"])
        bg, bcounts = group_counts(gt, bp, image_to_group)
        ag, acounts = group_counts(gt, combined, image_to_group)
        if bg != ag:
            raise ValueError("Group mismatch")
        output["bootstrap"] = paired_bootstrap(
            bcounts,
            acounts,
            policy["baseline_latency_seconds"],
            policy["candidate_latency_seconds"],
            repetitions=args.bootstrap,
        )
    save(args.out, output)
    print(f"LOCAL_ONLY delta_score={output['delta_score']:.6f}; no automatic admission")


def benchmark_d4(args):
    from PIL import Image

    from rsdet.contracts import Prediction
    from rsdet.submission.aircraft_d4 import AircraftD4ClassifierRuntime, filter_prediction_by_score

    from .bounded_d4 import BoundedD4Runtime

    config, cache = read(args.config), read(args.cache)
    c = config["aircraft_classifier_model"]
    if sha(c["weight_path"]) != c["expected_sha256"]:
        raise ValueError("Classifier SHA mismatch")
    baseline = AircraftD4ClassifierRuntime(c, config["device"])
    bounded = BoundedD4Runtime(baseline)
    rows = []
    for item in cache["images"]:
        with Image.open(Path(args.image_root) / item["file_name"]) as im:
            rgb = np.asarray(im.convert("RGB")).copy()
        if hashlib.sha256(rgb.tobytes()).hexdigest() != item["pixel_sha256"]:
            raise ValueError("Cached image pixels changed")
        prediction = filter_prediction_by_score(
            Prediction(**item["prediction"]), float(config["post_fusion_score_threshold"])
        )
        # Warm both execution shapes on this fixed image; not part of timing.
        reference = baseline.refine(rgb, prediction)
        trial = bounded.refine(rgb, prediction)
        equal = prediction_fingerprint(reference) == prediction_fingerprint(trial)
        times = {"baseline": [], "bounded": []}
        for _ in range(args.repeats):
            for name, runtime in (
                ("baseline", baseline),
                ("bounded", bounded),
                ("bounded", bounded),
                ("baseline", baseline),
            ):
                sync()
                start = time.perf_counter()
                result = runtime.refine(rgb, prediction)
                sync()
                times[name].append(time.perf_counter() - start)
                equal &= prediction_fingerprint(reference) == prediction_fingerprint(result)
        rows.append(
            {
                "image_id": item["image_id"],
                "exact_output_equal": bool(equal),
                "seconds": times,
                "bounded_audit": dict(bounded.last_audit),
            }
        )
        print(
            f"D4 image={item['image_id']} exact={equal} "
            f"base={np.median(times['baseline']):.3f}s bounded={np.median(times['bounded']):.3f}s"
        )
    output = {
        "scope": "D4 refinement only; not end-to-end/official latency",
        "images": rows,
        "all_outputs_exact": bool(rows) and all(x["exact_output_equal"] for x in rows),
        "formal_admission": False,
    }
    save(args.out, output)
    if not output["all_outputs_exact"]:
        raise SystemExit("D4 parity failed. Leave bounded_d4=false.")


def config_edit(args):
    payload = read(args.base)
    options = {"mode": args.mode, "bounded_d4": args.bounded_d4, "allow_experimental": True}
    if args.mode == "shared":
        if args.otm_threshold is None or not args.otm_labels:
            raise ValueError("Shared mode needs frozen OTM labels/threshold")
        options.update(otm_threshold=args.otm_threshold, otm_labels=args.otm_labels)
    elif args.mode == "otm" and args.otm_threshold is not None:
        payload["post_fusion_score_threshold"] = args.otm_threshold
    payload["sprint20"] = options
    payload["deployment_role"] = "EXPERIMENTAL_NOT_ADMITTED"
    save(args.out, payload)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="command", required=True)
    a = sp.add_parser("audit")
    a.add_argument("--repo", required=True)
    a.add_argument("--config", required=True)
    a.add_argument("--out", required=True)
    a.set_defaults(func=audit)
    a = sp.add_parser("probe")
    for key in ("config", "coco", "image-root", "out"):
        a.add_argument("--" + key, required=True)
    a.add_argument("--head", choices=["oto", "otm", "shared"], required=True)
    a.add_argument(
        "--role",
        choices=["outer_oof_mature", "outer_oof_short", "full_seen", "engineering_only"],
        required=True,
    )
    a.set_defaults(func=probe)
    a = sp.add_parser("parity")
    for key in ("native", "shared", "out"):
        a.add_argument("--" + key, required=True)
    a.add_argument("--head", choices=["oto", "otm"], required=True)
    a.set_defaults(func=parity)
    a = sp.add_parser("replay")
    for key in ("coco", "baseline", "candidate", "policy", "out"):
        a.add_argument("--" + key, required=True)
    a.add_argument("--groups")
    a.add_argument("--group-key", default="group_id")
    a.add_argument("--bootstrap", type=int, default=3000)
    a.set_defaults(func=replay)
    a = sp.add_parser("benchmark-d4")
    for key in ("config", "cache", "image-root", "out"):
        a.add_argument("--" + key, required=True)
    a.add_argument("--repeats", type=int, default=2)
    a.set_defaults(func=benchmark_d4)
    a = sp.add_parser("config")
    for key in ("base", "out"):
        a.add_argument("--" + key, required=True)
    a.add_argument("--mode", choices=["oto", "otm", "shared"], default="oto")
    a.add_argument("--bounded-d4", action="store_true")
    a.add_argument("--otm-labels", type=int, nargs="+")
    a.add_argument("--otm-threshold", type=float)
    a.set_defaults(func=config_edit)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
