"""Deterministic local-PCA/VLAD utilities for MAR20 place retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def l2_normalize(values: np.ndarray, *, axis: int = -1, epsilon: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.maximum(norms, epsilon)


@dataclass(frozen=True)
class LocalPcaVladCodebook:
    layer: int
    local_input_dimension: int
    local_dimension: int
    cluster_count: int
    pca_mean: np.ndarray
    pca_components: np.ndarray
    centers: np.ndarray

    def validate(self) -> None:
        if self.layer < 0 or self.local_dimension <= 0 or self.cluster_count <= 1:
            raise ValueError("VLAD codebook metadata invalid")
        if self.pca_mean.shape != (self.local_input_dimension,):
            raise ValueError("pca_mean shape invalid")
        if self.pca_components.shape != (self.local_dimension, self.local_input_dimension):
            raise ValueError("pca_components shape invalid")
        if self.centers.shape != (self.cluster_count, self.local_dimension):
            raise ValueError("centers shape invalid")
        for value in (self.pca_mean, self.pca_components, self.centers):
            if not np.isfinite(value).all():
                raise ValueError("VLAD codebook contains NaN/Inf")

    def project_local(self, tokens: np.ndarray) -> np.ndarray:
        self.validate()
        values = np.asarray(tokens, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.local_input_dimension:
            raise ValueError("local token shape incompatible with codebook")
        projected = (values - self.pca_mean) @ self.pca_components.T
        return l2_normalize(projected)

    def encode(self, tokens: np.ndarray) -> np.ndarray:
        projected = self.project_local(tokens)
        # Preserve the Euclidean assignment learned by MiniBatchKMeans.
        # Normalising centres and switching to cosine assignment would silently
        # alter both the Voronoi cells and the origin of every VLAD residual.
        squared_distance = (
            np.square(projected).sum(axis=1, keepdims=True)
            + np.square(self.centers).sum(axis=1)[None, :]
            - 2.0 * (projected @ self.centers.T)
        )
        assignment = squared_distance.argmin(axis=1)
        residuals = np.zeros((self.cluster_count, self.local_dimension), dtype=np.float32)
        for cluster in range(self.cluster_count):
            selected = projected[assignment == cluster]
            if selected.size:
                residuals[cluster] = (selected - self.centers[cluster]).sum(axis=0)
        residuals = l2_normalize(residuals, axis=1)
        return l2_normalize(residuals.reshape(1, -1))[0]

    def to_payload(self) -> dict[str, np.ndarray]:
        self.validate()
        return {
            "layer": np.asarray([self.layer], dtype=np.int64),
            "local_input_dimension": np.asarray([self.local_input_dimension], dtype=np.int64),
            "local_dimension": np.asarray([self.local_dimension], dtype=np.int64),
            "cluster_count": np.asarray([self.cluster_count], dtype=np.int64),
            "pca_mean": self.pca_mean.astype(np.float32),
            "pca_components": self.pca_components.astype(np.float32),
            "centers": self.centers.astype(np.float32),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "LocalPcaVladCodebook":
        value = cls(
            layer=int(np.asarray(payload["layer"]).reshape(-1)[0]),
            local_input_dimension=int(np.asarray(payload["local_input_dimension"]).reshape(-1)[0]),
            local_dimension=int(np.asarray(payload["local_dimension"]).reshape(-1)[0]),
            cluster_count=int(np.asarray(payload["cluster_count"]).reshape(-1)[0]),
            pca_mean=np.asarray(payload["pca_mean"], dtype=np.float32),
            pca_components=np.asarray(payload["pca_components"], dtype=np.float32),
            centers=np.asarray(payload["centers"], dtype=np.float32),
        )
        value.validate()
        return value


def fit_local_pca_vlad(
    tokens: np.ndarray,
    *,
    layer: int,
    local_dimension: int,
    cluster_count: int,
    seed: int,
) -> LocalPcaVladCodebook:
    """Fit deterministic PCA and MiniBatchKMeans on image-balanced token samples."""

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    values = np.asarray(tokens, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("tokens must be a finite [N,D] matrix")
    if not 1 <= local_dimension <= min(values.shape):
        raise ValueError("local_dimension is not feasible")
    if not 2 <= cluster_count < values.shape[0]:
        raise ValueError("cluster_count is not feasible")
    pca = PCA(
        n_components=local_dimension,
        svd_solver="randomized",
        whiten=False,
        random_state=seed,
    )
    projected = l2_normalize(pca.fit_transform(values).astype(np.float32))
    kmeans = MiniBatchKMeans(
        n_clusters=cluster_count,
        batch_size=min(4096, max(cluster_count * 16, 256)),
        n_init=3,
        max_iter=200,
        reassignment_ratio=0.0,
        random_state=seed,
    )
    kmeans.fit(projected)
    codebook = LocalPcaVladCodebook(
        layer=int(layer),
        local_input_dimension=int(values.shape[1]),
        local_dimension=int(local_dimension),
        cluster_count=int(cluster_count),
        pca_mean=np.asarray(pca.mean_, dtype=np.float32),
        pca_components=np.asarray(pca.components_, dtype=np.float32),
        centers=np.asarray(kmeans.cluster_centers_, dtype=np.float32),
    )
    codebook.validate()
    return codebook


def fit_global_pca(
    descriptors: np.ndarray, *, output_dimension: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("descriptors must be a finite [N,D] matrix")
    if not 1 <= output_dimension <= min(values.shape):
        raise ValueError("output_dimension is not feasible")
    pca = PCA(
        n_components=output_dimension,
        svd_solver="randomized",
        whiten=True,
        random_state=seed,
    )
    pca.fit(values)
    scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-12)).astype(np.float32)
    components = np.asarray(pca.components_, dtype=np.float32) / scale[:, None]
    return np.asarray(pca.mean_, dtype=np.float32), components


def apply_global_pca(
    descriptors: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    values = np.asarray(descriptors, dtype=np.float32)
    center = np.asarray(mean, dtype=np.float32)
    projection = np.asarray(components, dtype=np.float32)
    if values.ndim != 2 or center.shape != (values.shape[1],):
        raise ValueError("global PCA mean shape invalid")
    if projection.ndim != 2 or projection.shape[1] != values.shape[1]:
        raise ValueError("global PCA components shape invalid")
    return l2_normalize((values - center) @ projection.T)
