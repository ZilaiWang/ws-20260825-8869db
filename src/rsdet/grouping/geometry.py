"""Local background evidence used by the conservative MAR20 grouping pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SiftFeatures:
    points: np.ndarray
    descriptors: np.ndarray
    image_size: tuple[int, int]


def phash64(image: Image.Image) -> np.ndarray:
    """Return the deterministic 64-bit perceptual hash used only as evidence."""

    import cv2

    gray = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32
    )
    low = cv2.dct(gray)[:8, :8].reshape(-1)
    threshold = float(np.median(low[1:]))
    return (low > threshold).astype(np.uint8)


def pool_masked_patch_tokens(
    tokens: np.ndarray,
    valid: np.ndarray,
    *,
    input_grid: int = 37,
    output_grid: int = 19,
    minimum_valid_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask-aware adaptive pooling for one DINO patch-token grid."""

    values = np.asarray(tokens, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or values.shape[0] != input_grid * input_grid:
        raise ValueError("tokens shape does not match input grid")
    if mask.shape != (input_grid * input_grid,):
        raise ValueError("valid mask shape does not match input grid")
    if not 0 < output_grid <= input_grid or not 0 <= minimum_valid_fraction <= 1:
        raise ValueError("invalid pooling configuration")
    values = values.reshape(input_grid, input_grid, -1)
    mask = mask.reshape(input_grid, input_grid)
    pooled = np.zeros((output_grid, output_grid, values.shape[-1]), dtype=np.float32)
    pooled_valid = np.zeros((output_grid, output_grid), dtype=bool)
    for row in range(output_grid):
        y0 = math.floor(row * input_grid / output_grid)
        y1 = math.ceil((row + 1) * input_grid / output_grid)
        for column in range(output_grid):
            x0 = math.floor(column * input_grid / output_grid)
            x1 = math.ceil((column + 1) * input_grid / output_grid)
            local_mask = mask[y0:y1, x0:x1]
            fraction = float(local_mask.mean())
            if fraction + 1e-12 < minimum_valid_fraction or not local_mask.any():
                continue
            local_values = values[y0:y1, x0:x1][local_mask]
            vector = local_values.mean(axis=0)
            norm = float(np.linalg.norm(vector))
            if norm > 1e-12 and np.isfinite(vector).all():
                pooled[row, column] = vector / norm
                pooled_valid[row, column] = True
    return pooled.reshape(-1, values.shape[-1]), pooled_valid.reshape(-1)


def extract_sift_features(
    gray: np.ndarray,
    background_mask: np.ndarray,
    *,
    nfeatures: int = 2500,
) -> SiftFeatures:
    import cv2

    image = np.asarray(gray, dtype=np.uint8)
    mask = np.asarray(background_mask, dtype=np.uint8)
    if image.ndim != 2 or mask.shape != image.shape:
        raise ValueError("gray/background mask shape mismatch")
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = sift.detectAndCompute(image, mask)
    points = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)
    if not len(points):
        points = np.empty((0, 2), dtype=np.float32)
        descriptors = np.empty((0, 128), dtype=np.float32)
    else:
        descriptors = np.asarray(descriptors, dtype=np.float32)
    return SiftFeatures(points=points, descriptors=descriptors, image_size=(image.shape[1], image.shape[0]))


def _mutual_ratio_matches(
    descriptors_u: np.ndarray, descriptors_v: np.ndarray, ratio: float
) -> list[tuple[int, int, float]]:
    import cv2

    if len(descriptors_u) < 2 or len(descriptors_v) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def accepted(left: np.ndarray, right: np.ndarray) -> dict[int, tuple[int, float]]:
        result = {}
        for matches in matcher.knnMatch(left, right, k=2):
            if len(matches) == 2 and matches[0].distance < ratio * matches[1].distance:
                result[int(matches[0].queryIdx)] = (
                    int(matches[0].trainIdx),
                    float(matches[0].distance),
                )
        return result

    forward = accepted(descriptors_u, descriptors_v)
    reverse = accepted(descriptors_v, descriptors_u)
    return [
        (query, target, distance)
        for query, (target, distance) in forward.items()
        if reverse.get(target, (-1, math.inf))[0] == query
    ]


def _grid_occupancy(points: np.ndarray, size: tuple[int, int], grid: int = 4) -> int:
    if not len(points):
        return 0
    width, height = size
    x = np.clip((points[:, 0] / max(width, 1) * grid).astype(int), 0, grid - 1)
    y = np.clip((points[:, 1] / max(height, 1) * grid).astype(int), 0, grid - 1)
    return int(len(set(zip(x.tolist(), y.tolist()))))


def _hull_fraction(points: np.ndarray, size: tuple[int, int]) -> float:
    import cv2

    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull) / max(size[0] * size[1], 1))


def _direction_metrics(points_u: np.ndarray, points_v: np.ndarray) -> tuple[float, float]:
    if not len(points_u):
        return 0.0, 0.0
    delta = points_v - points_u
    angles = (np.arctan2(delta[:, 1], delta[:, 0]) + 2 * np.pi) % (2 * np.pi)
    histogram, _ = np.histogram(angles, bins=12, range=(0, 2 * np.pi))
    probabilities = histogram[histogram > 0] / max(histogram.sum(), 1)
    entropy = float(-(probabilities * np.log(probabilities)).sum() / math.log(12))
    return float(histogram.max() / max(histogram.sum(), 1)), entropy


def _transform_metrics(
    points_u: np.ndarray,
    points_v: np.ndarray,
    *,
    model: str,
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    import cv2

    minimum = 4 if model == "homography" else 3
    prefix = model
    empty = {
        f"{prefix}_inliers": 0,
        f"{prefix}_inlier_ratio": 0.0,
        f"{prefix}_median_error": None,
        f"{prefix}_p95_error": None,
        f"{prefix}_matrix": "",
        f"{prefix}_inlier_mask": np.zeros(len(points_u), dtype=bool),
    }
    if len(points_u) < minimum:
        return empty
    cv2.setRNGSeed(int(seed) & 0x7FFFFFFF)
    if model == "similarity":
        matrix, status = cv2.estimateAffinePartial2D(
            points_u,
            points_v,
            method=cv2.RANSAC,
            ransacReprojThreshold=threshold,
            maxIters=3000,
            confidence=0.995,
            refineIters=10,
        )
    elif model == "affine":
        matrix, status = cv2.estimateAffine2D(
            points_u,
            points_v,
            method=cv2.RANSAC,
            ransacReprojThreshold=threshold,
            maxIters=3000,
            confidence=0.995,
            refineIters=10,
        )
    elif model == "homography":
        matrix, status = cv2.findHomography(
            points_u, points_v, cv2.RANSAC, threshold, maxIters=3000, confidence=0.995
        )
    else:
        raise ValueError(f"unknown model={model}")
    if matrix is None or status is None:
        return empty
    if not np.isfinite(matrix).all():
        return empty
    selected = status.reshape(-1).astype(bool)
    if model == "homography":
        projected = cv2.perspectiveTransform(points_u[:, None, :], matrix)[:, 0, :]
    else:
        projected = cv2.transform(points_u[:, None, :], matrix)[:, 0, :]
    errors = np.linalg.norm(projected - points_v, axis=1)
    if not np.isfinite(projected).all() or not np.isfinite(errors).all():
        return empty
    inlier_errors = errors[selected]
    if not np.isfinite(inlier_errors).all():
        return empty
    return {
        f"{prefix}_inliers": int(selected.sum()),
        f"{prefix}_inlier_ratio": float(selected.mean()),
        f"{prefix}_median_error": float(np.median(inlier_errors)) if len(inlier_errors) else None,
        f"{prefix}_p95_error": float(np.quantile(inlier_errors, 0.95))
        if len(inlier_errors)
        else None,
        f"{prefix}_matrix": ";".join(f"{value:.9g}" for value in matrix.reshape(-1)),
        f"{prefix}_inlier_mask": selected,
    }


def sift_pair_evidence(
    features_u: SiftFeatures,
    features_v: SiftFeatures,
    *,
    ratio: float = 0.75,
    ransac_fraction_diagonal: float = 0.005,
    repeat_count: int = 20,
    seed: int = 202625,
) -> dict[str, Any]:
    """Compute mutual SIFT and distribution-aware geometric evidence."""

    matches = _mutual_ratio_matches(features_u.descriptors, features_v.descriptors, ratio)
    points_u = np.asarray([features_u.points[left] for left, _, _ in matches], dtype=np.float32)
    points_v = np.asarray([features_v.points[right] for _, right, _ in matches], dtype=np.float32)
    if not len(matches):
        points_u = np.empty((0, 2), dtype=np.float32)
        points_v = np.empty((0, 2), dtype=np.float32)
    diagonal = max(
        math.hypot(*features_u.image_size), math.hypot(*features_v.image_size), 1.0
    )
    threshold = max(3.0, ransac_fraction_diagonal * diagonal)
    output: dict[str, Any] = {
        "sift_keypoints_u": len(features_u.points),
        "sift_keypoints_v": len(features_v.points),
        "sift_mutual_ratio_matches": len(matches),
        "sift_ransac_threshold": threshold,
    }
    for model in ("similarity", "affine", "homography"):
        output.update(
            _transform_metrics(
                points_u, points_v, model=model, threshold=threshold, seed=seed
            )
        )
    selected = output["similarity_inlier_mask"]
    if int(selected.sum()) < 3:
        selected = output["affine_inlier_mask"]
    inlier_u = points_u[selected]
    inlier_v = points_v[selected]
    dominant, entropy = _direction_metrics(inlier_u, inlier_v)
    output.update(
        {
            "sift_unique_spatial_inliers": int(len(inlier_u)),
            "sift_grid_occupancy_u": _grid_occupancy(inlier_u, features_u.image_size),
            "sift_grid_occupancy_v": _grid_occupancy(inlier_v, features_v.image_size),
            "sift_hull_fraction_u": _hull_fraction(inlier_u, features_u.image_size),
            "sift_hull_fraction_v": _hull_fraction(inlier_v, features_v.image_size),
            "sift_dominant_direction_ratio": dominant,
            "sift_direction_entropy": entropy,
        }
    )
    passes = []
    for repeat in range(max(repeat_count, 0)):
        repeated = _transform_metrics(
            points_u,
            points_v,
            model="similarity",
            threshold=threshold,
            seed=seed + repeat + 1,
        )
        passes.append(
            repeated["similarity_inliers"] >= 8
            and repeated["similarity_inlier_ratio"] >= 0.2
        )
    output["sift_ransac_repeat_count"] = len(passes)
    output["sift_ransac_repeat_pass_rate"] = float(np.mean(passes)) if passes else None
    for key in ("similarity_inlier_mask", "affine_inlier_mask", "homography_inlier_mask"):
        output.pop(key, None)
    return output


def patch_pair_evidence(
    tokens_u: np.ndarray,
    valid_u: np.ndarray,
    tokens_v: np.ndarray,
    valid_v: np.ndarray,
    *,
    grid: int = 19,
) -> dict[str, Any]:
    """Compute mutual nearest-neighbour overlap evidence on pooled DINO tokens."""

    left = np.asarray(tokens_u, dtype=np.float32)
    right = np.asarray(tokens_v, dtype=np.float32)
    mask_u = np.asarray(valid_u, dtype=bool)
    mask_v = np.asarray(valid_v, dtype=bool)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != grid * grid:
        raise ValueError("patch token shapes must match [grid*grid,D]")
    if mask_u.shape != (grid * grid,) or mask_v.shape != (grid * grid,):
        raise ValueError("patch validity shapes mismatch")
    index_u, index_v = np.flatnonzero(mask_u), np.flatnonzero(mask_v)
    output: dict[str, Any] = {
        "patch_valid_u": int(len(index_u)),
        "patch_valid_v": int(len(index_v)),
        "patch_mutual_count": 0,
        "patch_similarity_mean": None,
        "patch_similarity_p10": None,
        "patch_similarity_median": None,
        "patch_matches_ge_0p7": 0,
        "patch_matches_ge_0p8": 0,
        "patch_matches_ge_0p9": 0,
        "patch_grid_occupancy_u": 0,
        "patch_grid_occupancy_v": 0,
    }
    if not len(index_u) or not len(index_v):
        return output
    scores = left[index_u] @ right[index_v].T
    forward = scores.argmax(axis=1)
    reverse = scores.argmax(axis=0)
    pairs = [(local_u, local_v) for local_u, local_v in enumerate(forward) if reverse[local_v] == local_u]
    if not pairs:
        return output
    local_u = np.asarray([pair[0] for pair in pairs], dtype=int)
    local_v = np.asarray([pair[1] for pair in pairs], dtype=int)
    similarities = scores[local_u, local_v]
    full_u, full_v = index_u[local_u], index_v[local_v]
    points_u = np.column_stack((full_u % grid, full_u // grid)).astype(np.float32)
    points_v = np.column_stack((full_v % grid, full_v // grid)).astype(np.float32)
    output.update(
        {
            "patch_mutual_count": int(len(pairs)),
            "patch_similarity_mean": float(similarities.mean()),
            "patch_similarity_p10": float(np.quantile(similarities, 0.10)),
            "patch_similarity_median": float(np.median(similarities)),
            "patch_matches_ge_0p7": int((similarities >= 0.7).sum()),
            "patch_matches_ge_0p8": int((similarities >= 0.8).sum()),
            "patch_matches_ge_0p9": int((similarities >= 0.9).sum()),
            "patch_grid_occupancy_u": _grid_occupancy(points_u, (grid, grid)),
            "patch_grid_occupancy_v": _grid_occupancy(points_v, (grid, grid)),
        }
    )
    return output
