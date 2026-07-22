#!/usr/bin/env python3
"""Mine positive-enriched MAR20 review candidates from multi-route retrieval + SIFT."""

from __future__ import annotations

import argparse
import csv
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageOps

from rsdet.grouping.cache import PlaceFeatureCache
from rsdet.grouping.contracts import (
    MASKED_PATCH_PROTOCOL_VERSION,
    atomic_write_json,
    canonical_pair_uid,
    parse_node_uid,
    sha256_file,
)
from rsdet.grouping.masks import build_protocol_foreground_mask
from rsdet.grouping.registry import load_annotations, load_registry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine enriched MAR20 local-overlap candidates")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--mar20-root", required=True)
    parser.add_argument("--routes-json", required=True)
    parser.add_argument("--existing-pairs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k-per-route", type=int, default=12)
    parser.add_argument("--row-search-multiplier", type=int, default=8)
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pregeometry-limit", type=int, default=1600)
    parser.add_argument("--phash-top-k", type=int, default=5)
    parser.add_argument("--sift-max-dimension", type=int, default=1024)
    parser.add_argument("--sift-features", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=202625)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _phash_bits(image: Image.Image) -> np.ndarray:
    import cv2

    gray = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32
    )
    transformed = cv2.dct(gray)
    low = transformed[:8, :8].copy()
    threshold = float(np.median(low.reshape(-1)[1:]))
    return (low.reshape(-1) > threshold).astype(np.uint8)


def _l2(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def _cache_rows(
    cache_dir: str, feature: str, view_type: str | None
) -> tuple[list[str], np.ndarray]:
    cache = PlaceFeatureCache(cache_dir)
    if feature not in cache.feature_names:
        raise ValueError(f"{feature} not in {cache_dir}")
    payload = cache.load_all()
    keep = np.ones(len(payload["row__node_uid"]), dtype=bool)
    if view_type:
        keep &= payload["row__view_type"].astype(str) == view_type
    nodes = payload["row__node_uid"].astype(str)[keep].tolist()
    features = _l2(payload[f"feature__{feature}"][keep])
    if not nodes:
        raise ValueError(f"route {feature}/{view_type} has no rows")
    return nodes, features


def _route_candidates(
    nodes: list[str],
    features: np.ndarray,
    *,
    target_nodes: set[str],
    top_k: int,
    row_multiplier: int,
    batch_size: int,
    device: str,
) -> dict[str, tuple[float, int]]:
    node_to_rows: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        node_to_rows.setdefault(node, []).append(index)
    query_nodes = sorted(target_nodes & set(node_to_rows), key=parse_node_uid)
    result: dict[str, tuple[float, int]] = {}
    row_k = min(features.shape[0], max(top_k * row_multiplier, top_k + 1))
    query_indices = [index for node in query_nodes for index in node_to_rows[node]]
    scores_by_query_node: dict[str, dict[str, float]] = {node: {} for node in query_nodes}
    try:
        import torch
    except ImportError:
        torch = None
    use_torch = bool(
        torch is not None
        and (device == "cpu" or (device.startswith("cuda") and torch.cuda.is_available()))
    )
    if use_torch:
        target = torch.device(device)
        database = torch.from_numpy(features).to(target)
    for offset in range(0, len(query_indices), batch_size):
        batch_indices = query_indices[offset : offset + batch_size]
        if use_torch:
            query = database[batch_indices]
            score_matrix = query @ database.T
            values, indices = torch.topk(score_matrix, k=row_k, dim=1, largest=True, sorted=True)
            batch_scores = values.float().cpu().numpy()
            batch_neighbors = indices.cpu().numpy()
        else:
            score_matrix = features[batch_indices] @ features.T
            batch_neighbors = np.argpartition(score_matrix, -row_k, axis=1)[:, -row_k:]
            selected = np.take_along_axis(score_matrix, batch_neighbors, axis=1)
            order = np.argsort(selected, axis=1)[:, ::-1]
            batch_neighbors = np.take_along_axis(batch_neighbors, order, axis=1)
            batch_scores = np.take_along_axis(score_matrix, batch_neighbors, axis=1)
        for local, query_index in enumerate(batch_indices):
            node = nodes[query_index]
            scores_by_node = scores_by_query_node[node]
            for index, score in zip(batch_neighbors[local], batch_scores[local], strict=True):
                neighbor = nodes[int(index)]
                if neighbor == node:
                    continue
                score_value = float(score)
                if score_value > scores_by_node.get(neighbor, -math.inf):
                    scores_by_node[neighbor] = score_value
    for node in query_nodes:
        scores_by_node = scores_by_query_node[node]
        ordered = sorted(
            scores_by_node.items(), key=lambda item: (-item[1], parse_node_uid(item[0]))
        )[:top_k]
        for rank, (neighbor, score) in enumerate(ordered, 1):
            pair_uid = canonical_pair_uid(node, neighbor)
            previous = result.get(pair_uid)
            if (
                previous is None
                or rank < previous[1]
                or (rank == previous[1] and score > previous[0])
            ):
                result[pair_uid] = (score, rank)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    registry_path = Path(args.registry).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()
    registry = load_registry(registry_path)
    annotations = load_annotations(annotations_path)
    rows = {row["node_uid"]: row for row in registry}
    target_nodes = {uid for uid, row in rows.items() if row["is_target"] == "1"}
    mar20_root = Path(args.mar20_root).expanduser().resolve()
    routes_path = Path(args.routes_json).expanduser().resolve()
    route_config = json.loads(routes_path.read_text(encoding="utf-8"))
    routes = route_config.get("routes", [])
    if not routes:
        raise ValueError("routes-json has no routes")
    existing: set[str] = set()
    if args.existing_pairs:
        existing = {
            row["pair_uid"] for row in _read_csv(Path(args.existing_pairs).expanduser().resolve())
        }
    candidates: dict[str, dict[str, Any]] = {}
    route_candidate_counts: dict[str, int] = {}

    def add(
        pair_uid: str,
        route: str,
        *,
        score: float | None = None,
        rank: int | None = None,
        phash_distance: int | None = None,
    ) -> None:
        if pair_uid in existing:
            return
        left, right = pair_uid.split("--")
        entry = candidates.setdefault(
            pair_uid,
            {
                "pair_uid": pair_uid,
                "node_u": left,
                "node_v": right,
                "routes": set(),
                "best_similarity": None,
                "best_rank": None,
                "phash_distance": None,
            },
        )
        entry["routes"].add(route)
        if score is not None and (
            entry["best_similarity"] is None or score > entry["best_similarity"]
        ):
            entry["best_similarity"] = score
        if rank is not None and (entry["best_rank"] is None or rank < entry["best_rank"]):
            entry["best_rank"] = rank
        if phash_distance is not None and (
            entry["phash_distance"] is None or phash_distance < entry["phash_distance"]
        ):
            entry["phash_distance"] = phash_distance

    for route in routes:
        name = str(route["name"])
        nodes, features = _cache_rows(
            str(route["cache_dir"]), str(route["feature"]), route.get("view_type")
        )
        mined = _route_candidates(
            nodes,
            features,
            target_nodes=target_nodes,
            top_k=args.top_k_per_route,
            row_multiplier=args.row_search_multiplier,
            batch_size=args.retrieval_batch_size,
            device=args.device,
        )
        route_candidate_counts[name] = len(mined)
        for pair_uid, (score, rank) in mined.items():
            add(pair_uid, name, score=score, rank=rank)

    ordered_uids = sorted(rows, key=parse_node_uid)

    @lru_cache(maxsize=None)
    def load_rgb(uid: str) -> Image.Image:
        path = (mar20_root / rows[uid]["original_relative_path"]).resolve()
        path.relative_to(mar20_root)
        with Image.open(path) as image:
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")

    phashes = np.stack([_phash_bits(load_rgb(uid)) for uid in ordered_uids])
    uid_index = {uid: index for index, uid in enumerate(ordered_uids)}
    for uid in sorted(target_nodes, key=parse_node_uid):
        index = uid_index[uid]
        distances = np.count_nonzero(phashes != phashes[index], axis=1)
        order = np.argsort(distances, kind="stable")
        emitted = 0
        for neighbor_index in order:
            neighbor = ordered_uids[int(neighbor_index)]
            if neighbor == uid:
                continue
            add(
                canonical_pair_uid(uid, neighbor),
                "phash64",
                phash_distance=int(distances[int(neighbor_index)]),
                rank=emitted + 1,
            )
            emitted += 1
            if emitted == args.phash_top_k:
                break
    by_pixel: dict[str, list[str]] = {}
    for uid, row in rows.items():
        by_pixel.setdefault(row["original_pixel_sha256"], []).append(uid)
    for group in by_pixel.values():
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                if left in target_nodes or right in target_nodes:
                    add(canonical_pair_uid(left, right), "exact_pixel", score=1.0, rank=0)

    route_union_counts = {
        route: sum(route in item["routes"] for item in candidates.values())
        for route in sorted({name for item in candidates.values() for name in item["routes"]})
    }

    pregeometry = sorted(
        candidates.values(),
        key=lambda item: (
            0 if "exact_pixel" in item["routes"] else 1,
            item["best_rank"] if item["best_rank"] is not None else 999999,
            -(item["best_similarity"] if item["best_similarity"] is not None else -1.0),
            item["phash_distance"] if item["phash_distance"] is not None else 999,
            item["pair_uid"],
        ),
    )[: args.pregeometry_limit]

    import cv2

    sift = cv2.SIFT_create(nfeatures=args.sift_features)

    @lru_cache(maxsize=None)
    def local_features(uid: str) -> tuple[list[Any], np.ndarray | None, tuple[int, int]]:
        image = load_rgb(uid)
        scale = min(1.0, args.sift_max_dimension / max(image.size))
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        gray = np.asarray(image.convert("L").resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)
        mask = build_protocol_foreground_mask(
            image.size,
            [box["xyxy"] for box in annotations[uid]["boxes"]],
            dilation_ratio=0.15,
        ).resize(size, Image.Resampling.NEAREST)
        background = 255 - np.asarray(mask, dtype=np.uint8)
        keypoints, descriptors = sift.detectAndCompute(gray, background)
        return keypoints, descriptors, size

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    for item in pregeometry:
        key_u, desc_u, size_u = local_features(item["node_u"])
        key_v, desc_v, size_v = local_features(item["node_v"])
        good = []
        if desc_u is not None and desc_v is not None and len(desc_u) >= 2 and len(desc_v) >= 2:
            for matches in matcher.knnMatch(desc_u, desc_v, k=2):
                if len(matches) == 2 and matches[0].distance < 0.75 * matches[1].distance:
                    good.append(matches[0])
        inliers = 0
        inlier_ratio = coverage_u = coverage_v = 0.0
        median_error = None
        if len(good) >= 8:
            points_u = np.float32([key_u[match.queryIdx].pt for match in good])
            points_v = np.float32([key_v[match.trainIdx].pt for match in good])
            homography, status = cv2.findHomography(points_u, points_v, cv2.RANSAC, 4.0)
            if homography is not None and status is not None:
                selected = status.reshape(-1).astype(bool)
                inliers = int(selected.sum())
                inlier_ratio = inliers / len(good)
                if inliers:
                    src = points_u[selected]
                    dst = points_v[selected]
                    projected = cv2.perspectiveTransform(src[:, None, :], homography)[:, 0, :]
                    median_error = float(np.median(np.linalg.norm(projected - dst, axis=1)))

                    def coverage(points: np.ndarray, size: tuple[int, int]) -> float:
                        span = np.ptp(points, axis=0)
                        return float((span[0] * span[1]) / max(size[0] * size[1], 1))

                    coverage_u = coverage(src, size_u)
                    coverage_v = coverage(dst, size_v)
        item.update(
            {
                "sift_good_matches": len(good),
                "sift_inliers": inliers,
                "sift_inlier_ratio": inlier_ratio,
                "sift_coverage_u": coverage_u,
                "sift_coverage_v": coverage_v,
                "sift_median_reprojection_error": median_error,
            }
        )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "enriched_candidate_pairs.csv"
    fields = [
        "pair_uid",
        "node_u",
        "node_v",
        "target_relation",
        "cross_official_side",
        "routes",
        "route_count",
        "best_similarity",
        "best_rank",
        "phash_distance",
        "sift_good_matches",
        "sift_inliers",
        "sift_inlier_ratio",
        "sift_coverage_u",
        "sift_coverage_v",
        "sift_median_reprojection_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in pregeometry:
            left_target = item["node_u"] in target_nodes
            right_target = item["node_v"] in target_nodes
            writer.writerow(
                {
                    **{key: item.get(key) for key in fields},
                    "target_relation": "target_target"
                    if left_target and right_target
                    else "target_bridge",
                    "cross_official_side": int(
                        rows[item["node_u"]]["official_side"]
                        != rows[item["node_v"]]["official_side"]
                    ),
                    "routes": "+".join(sorted(item["routes"])),
                    "route_count": len(item["routes"]),
                }
            )
    summary = {
        "status": "pass",
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "registry_sha256": sha256_file(registry_path),
        "annotations_sha256": sha256_file(annotations_path),
        "routes_json_sha256": sha256_file(routes_path),
        "route_count": len(routes),
        "route_candidate_counts_before_union": dict(sorted(route_candidate_counts.items())),
        "route_union_counts": route_union_counts,
        "existing_pair_count": len(existing),
        "union_candidate_count": len(candidates),
        "geometry_scored_count": len(pregeometry),
        "geometry_scored_target_target_count": sum(
            item["node_u"] in target_nodes and item["node_v"] in target_nodes
            for item in pregeometry
        ),
        "geometry_scored_route_counts": {
            route: sum(route in item["routes"] for item in pregeometry)
            for route in route_union_counts
        },
        "geometry_supported_count": sum(
            int(item.get("sift_inliers", 0)) >= 12 for item in pregeometry
        ),
        "geometry_strong_supported_count": sum(
            int(item.get("sift_inliers", 0)) >= 12
            and float(item.get("sift_inlier_ratio", 0.0)) >= 0.25
            and min(
                float(item.get("sift_coverage_u", 0.0)),
                float(item.get("sift_coverage_v", 0.0)),
            )
            >= 0.01
            and item.get("sift_median_reprojection_error") is not None
            and float(item["sift_median_reprojection_error"]) <= 4.0
            for item in pregeometry
        ),
        "output_sha256": sha256_file(path),
        "formal_edge_admission": False,
    }
    atomic_write_json(output / "candidate_mining_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
