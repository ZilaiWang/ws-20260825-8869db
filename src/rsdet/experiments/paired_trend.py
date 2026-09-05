"""Frozen, group-disjoint single-model trend benchmark (no new matcher)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PARTS = ("train", "development", "confirmation")
VERSION = "paired_trend_review_v1"
OFFICIAL_WEIGHT_SHA = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: Any) -> None:
    """Immutable artifact, including for failed/partial runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def safe_path(root: Path, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe path: {relative}")
    target = (root / p).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes root: {relative}")
    return target


def label_relative(relative: str) -> str:
    parts = list(Path(relative).parts)
    parts[parts.index("images")] = "labels"
    return str(Path(*parts).with_suffix(".txt"))


def group_assignment(samples: list[dict], gt: dict, *, seed: int = 20260903) -> tuple[dict, dict]:
    """Data-only MILP. No model/prediction/score is accepted by this function.

    Preserve groups, all fine classes in each part, >=50% of each class for
    training. Target 70/15/15 images and fine counts. A bounded deterministic
    solver is sufficient: this is allocation, not a scientific optimality claim.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    groups = sorted({s["group_id"] for s in samples})
    gi = {g: i for i, g in enumerate(groups)}
    by_id = {s["image_id"]: s for s in samples}
    if len(by_id) != len(samples) or len({s["relative_path"] for s in samples}) != len(samples):
        raise ValueError("duplicate sample identity/path")
    images = {i["id"]: i for i in gt["images"]}
    if len(images) != len(gt["images"]) or set(images) != set(by_id):
        raise ValueError("GT/sample universe mismatch")
    for i, s in by_id.items():
        if s["relative_path"] != images[i]["file_name"]:
            raise ValueError("GT/sample filename mismatch")
    counts = np.zeros((len(groups), 25), dtype=int)
    for a in gt["annotations"]:
        if a["image_id"] not in by_id or a["category_id"] not in range(25):
            raise ValueError("unknown annotation identity/category")
        counts[gi[by_id[a["image_id"]]["group_id"]], a["category_id"]] += 1
    support = (counts > 0).sum(axis=0)
    if np.any(support < 3):
        raise ValueError("infeasible: a fine class has fewer than three source groups")
    image_counts = np.array([sum(s["group_id"] == g for s in samples) for g in groups])
    features = np.column_stack([image_counts, counts])
    totals = features.sum(axis=0)
    n, k = len(groups), features.shape[1]
    # Binary assignments plus positive/negative normalized-target deviations.
    nvar = 3 * n + 2 * 3 * k
    c = np.zeros(nvar)
    for i, g in enumerate(groups):
        for p in range(3):
            token = hashlib.sha256(f"{seed}:{g}:{p}".encode()).digest()
            c[3 * i + p] = int.from_bytes(token[:4], "big") / 2**32 * 1e-7
    for p in range(3):
        for f in range(k):
            weight = (8.0 if f == 0 else 1.0) / totals[f]
            c[3 * n + 2 * (p * k + f) : 3 * n + 2 * (p * k + f) + 2] = weight
    rows, lo, hi = [], [], []

    def add(indices, values, lower, upper):
        rows.append((indices, values))
        lo.append(lower)
        hi.append(upper)

    for i in range(n):
        add([3 * i + p for p in range(3)], [1] * 3, 1, 1)
    ratios = (0.70, 0.15, 0.15)
    for p in range(3):
        idx = [3 * i + p for i in range(n)]
        for f in range(k):
            d = 3 * n + 2 * (p * k + f)
            target = ratios[p] * totals[f]
            add(idx + [d, d + 1], list(features[:, f]) + [-1, 1], target, target)
        # Images may deviate by 5 percentage points to respect indivisible groups.
        add(
            idx,
            list(image_counts),
            (ratios[p] - 0.05) * len(samples),
            (ratios[p] + 0.05) * len(samples),
        )
        for label in range(25):
            minimum = int(np.ceil(0.5 * counts[:, label].sum())) if p == 0 else 1
            add(idx, list(counts[:, label]), minimum, np.inf)
            # Two sources where feasible; TU160's three groups necessarily use one each.
            source_min = 1 if support[label] < 6 else 2
            add(idx, list((counts[:, label] > 0).astype(int)), source_min, np.inf)
    matrix = lil_matrix((len(rows), nvar))
    for i, (indices, values) in enumerate(rows):
        matrix[i, indices] = values
    upper = np.full(nvar, np.inf)
    upper[: 3 * n] = 1
    integrality = np.zeros(nvar)
    integrality[: 3 * n] = 1
    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(np.zeros(nvar), upper),
        constraints=LinearConstraint(matrix.tocsr(), lo, hi),
        options={"time_limit": 30, "mip_rel_gap": 0.01},
    )
    if result.x is None:
        raise ValueError(f"no feasible group split: {result.message}")
    values = result.x[: 3 * n].reshape(n, 3)
    if not np.allclose(values, np.round(values), atol=1e-5):
        raise ValueError("nonintegral allocation")
    ax = matrix @ result.x
    if np.any(ax < np.array(lo) - 1e-5) or np.any(ax > np.array(hi) + 1e-5):
        raise ValueError("allocation constraints not satisfied")
    assignment = {g: PARTS[int(values[i].argmax())] for i, g in enumerate(groups)}
    audit = {
        "selection_inputs": "source groups and GT labels only; no predictions",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "mip_gap": float(result.mip_gap),
        "seed": seed,
        "train_minimum_fraction_per_fine": 0.5,
        "group_count": n,
    }
    return assignment, audit


def freeze(
    source: Path, gt_path: Path, data_root: Path, project: Path, output: Path, background: Path
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    samples = sorted(read(source)["samples"], key=lambda s: s["image_id"])
    gt = read(gt_path)
    assignment, solver = group_assignment(samples, gt)
    boxes = defaultdict(list)
    for a in gt["annotations"]:
        boxes[a["image_id"]].append(a)
    images = {i["id"]: i for i in gt["images"]}
    inventory, duplicate_groups = [], defaultdict(set)
    # Check all local labels against COCO used in allocation; no alternate label set.
    for s in samples:
        rel = s["relative_path"]
        ip, lp = safe_path(data_root, rel), safe_path(data_root, label_relative(rel))
        image = images[s["image_id"]]
        actual = []
        for line in lp.read_text().splitlines():
            label, cx, cy, w, h = map(float, line.split())
            actual.append(
                (
                    int(label),
                    np.array([cx - w / 2, cy - h / 2, w, h])
                    * [image["width"], image["height"], image["width"], image["height"]],
                )
            )
        expected = sorted(boxes[s["image_id"]], key=lambda a: (a["category_id"], *a["bbox"]))
        actual.sort(key=lambda a: (a[0], *a[1]))
        if len(actual) != len(expected) or any(
            a[0] != b["category_id"] or not np.allclose(a[1], b["bbox"], atol=0.02, rtol=0)
            for a, b in zip(actual, expected)
        ):
            raise ValueError(f"local YOLO labels != frozen COCO: {rel}")
        digest = sha(ip)
        duplicate_groups[digest].add(assignment[s["group_id"]])
        inventory.append(
            {
                "image_id": s["image_id"],
                "relative_path": rel,
                "group_id": s["group_id"],
                "split": assignment[s["group_id"]],
                "image_sha256": digest,
                "label_sha256": sha(lp),
            }
        )
    if any(len(parts) > 1 for parts in duplicate_groups.values()):
        raise ValueError("identical images cross split boundaries")
    bg_manifest = background / "background_100mp_manifest.jsonl"
    bg_rows = [json.loads(line) for line in bg_manifest.read_text().splitlines() if line]
    for row in bg_rows:
        if sha(safe_path(background, row["file_name"])) != row["sha256"]:
            raise ValueError("background image SHA mismatch")
    if sha(bg_manifest) != read(background / "freeze_decision.json")["manifest_sha256"]:
        raise ValueError("background manifest does not match existing freeze")
    write(output / "manifest.json", {"version": VERSION, "samples": inventory})
    stats = {}
    for part in PARTS:
        selected = [s for s in inventory if s["split"] == part]
        ids = {s["image_id"] for s in selected}
        annotations = [a for a in gt["annotations"] if a["image_id"] in ids]
        counter = Counter(a["category_id"] for a in annotations)
        fine_groups = {
            str(label): len(
                {
                    s["group_id"]
                    for s in selected
                    if any(a["category_id"] == label for a in boxes[s["image_id"]])
                }
            )
            for label in range(25)
        }
        stats[part] = {
            "images": len(ids),
            "groups": len({s["group_id"] for s in selected}),
            "annotations": len(annotations),
            "fine_gt": dict(sorted(counter.items())),
            "fine_groups": fine_groups,
            "empty_gt_images": sum(not boxes[i] for i in ids),
        }
        if len(counter) != 25:
            raise ValueError(f"incomplete taxonomy in {part}")
        write(
            output / f"{part}_gt.json",
            {
                "images": [images[i] for i in sorted(ids)],
                "annotations": annotations,
                "categories": gt["categories"],
            },
        )
    write(output / "support.json", stats)
    files = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    contract = {
        "version": VERSION,
        "metric_protocol": "platform_observed_20260831",
        "status": "data_frozen_baseline_training_pending",
        "files": files,
        "project_config_sha256": sha(project),
        "source_group_sha256": sha(source),
        "source_gt_sha256": sha(gt_path),
        "split_audit": solver,
        "new_unseen_blind_dataset": False,
        "limitation": "historically inspected official sources; fresh model holdouts, not new data",
        "threshold_grid": {"start": 0.001, "stop": 1.0, "step": 0.005},
        "deadband": 0.5,
        "negative_images_in_score_test": 0,
        "deployment_regression": {
            "background_manifest_sha256": sha(bg_manifest),
            "background_images": len(bg_rows),
            "background_megapixels": sum(r["width"] * r["height"] for r in bg_rows) / 1e6,
            "background_is_known_stress_test": True,
        },
        "baseline": {
            "official_initial_weight_sha256": OFFICIAL_WEIGHT_SHA,
            "model": "YOLO26s",
            "seed": 42,
            "rotate90_p": 1.0,
            "foundation": {
                "epochs": 160,
                "imgsz": 1024,
                "batch": 12,
                "lr0": 0.002,
                "lrf": 0.01,
                "warmup_epochs": 3,
                "close_mosaic": 20,
            },
            "adaptation": {
                "epochs": 40,
                "imgsz": 1280,
                "batch": 8,
                "lr0": 0.0002,
                "lrf": 0.1,
                "warmup_epochs": 1,
                "close_mosaic": 40,
                "mosaic": 0.0,
            },
            "batch_note": "single-GPU stable reference; same maturity, not bitwise historical DDP",
        },
        "inference": {
            "imgsz": 1280,
            "score_floor": 0.001,
            "iou": 0.7,
            "max_det": 500,
            "batch": 4,
            "half": True,
        },
    }
    write(output / "contract.json", contract)
    return contract


def validate_bundle(bundle: Path, project: Path, data_root: Path | None = None) -> dict:
    contract = read(bundle / "contract.json")
    if contract["version"] != VERSION or sha(project) != contract["project_config_sha256"]:
        raise ValueError("protocol changed; create a new version, never silently reuse")
    for relative, expected in contract["files"].items():
        if sha(safe_path(bundle, relative)) != expected:
            raise ValueError(f"frozen artifact changed: {relative}")
    if data_root is not None:
        for s in read(bundle / "manifest.json")["samples"]:
            for relative, expected in (
                (s["relative_path"], s["image_sha256"]),
                (label_relative(s["relative_path"]), s["label_sha256"]),
            ):
                if sha(safe_path(data_root, relative)) != expected:
                    raise ValueError(f"dataset changed: {relative}")
    return contract
