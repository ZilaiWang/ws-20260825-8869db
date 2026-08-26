"""Two-view PAV dataset, metadata normalization, and balanced sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsdet.hera_guard.manifest import PAV_METADATA_COLUMNS


@dataclass(frozen=True)
class MetadataStandardizer:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, object]]) -> "MetadataStandardizer":
        import numpy as np

        if not rows:
            raise ValueError("cannot fit metadata normalization on empty rows")
        values = np.asarray(
            [[float(row[column]) for column in PAV_METADATA_COLUMNS] for row in rows],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("metadata contains non-finite values")
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(tuple(mean.tolist()), tuple(std.tolist()))

    def transform(self, row: Mapping[str, object]) -> list[float]:
        return [
            (float(row[column]) - self.mean[index]) / self.std[index]
            for index, column in enumerate(PAV_METADATA_COLUMNS)
        ]


class PAVManifestDataset:
    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        data_root: str | Path,
        metadata_standardizer: MetadataStandardizer,
        resolution: int = 224,
        augment_d4: bool = False,
    ) -> None:
        if not rows or resolution <= 0:
            raise ValueError("PAV dataset rows/resolution are invalid")
        self.rows = list(rows)
        self.data_root = Path(data_root).expanduser().resolve()
        self.standardizer = metadata_standardizer
        self.resolution = int(resolution)
        self.augment_d4 = bool(augment_d4)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image

        from rsdet.data.crop_classification import render_crop

        row = self.rows[index]
        path = self.data_root / str(row["source_relative_path"])
        with Image.open(path) as source:
            source = source.convert("RGB")
            tight = render_crop(
                source,
                tuple(float(row[f"tight_{axis}"]) for axis in ("x0", "y0", "x1", "y1")),
                self.resolution,
            )
            context = render_crop(
                source,
                tuple(float(row[f"context_{axis}"]) for axis in ("x0", "y0", "x1", "y1")),
                self.resolution,
            )

        def tensor(image: Any) -> Any:
            array = np.asarray(image, dtype=np.float32).copy().transpose(2, 0, 1)
            return torch.from_numpy(array).div_(255.0)

        tight_tensor, context_tensor = tensor(tight), tensor(context)
        if self.augment_d4:
            code = int(torch.randint(0, 8, ()).item())
            rotation = code % 4
            if rotation:
                tight_tensor = torch.rot90(tight_tensor, rotation, dims=(1, 2))
                context_tensor = torch.rot90(context_tensor, rotation, dims=(1, 2))
            if code >= 4:
                tight_tensor = torch.flip(tight_tensor, dims=(2,))
                context_tensor = torch.flip(context_tensor, dims=(2,))
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
        return {
            "tight": (tight_tensor - mean) / std,
            "context": (context_tensor - mean) / std,
            "metadata": torch.tensor(self.standardizer.transform(row), dtype=torch.float32),
            "foreground": torch.tensor(float(row["target_foreground"])),
            "coarse": torch.tensor(int(row["target_coarse"]), dtype=torch.long),
            "fine": torch.tensor(max(int(row["target_fine"]), 0), dtype=torch.long),
            "quality": torch.tensor(float(row["target_quality"])),
            "protect": torch.tensor(float(row["target_protect"])),
            "active_fp": torch.tensor(
                float(
                    row.get(
                        "target_active_fp",
                        int(str(row.get("workpoint_role", "")) == "active_fp"),
                    )
                )
            ),
            "candidate_id": torch.tensor(int(row["candidate_id"]), dtype=torch.long),
        }


def balanced_sampling_weights(rows: Sequence[Mapping[str, object]]) -> list[float]:
    """Prioritize active FP, protected TP, and rare foreground classes."""

    from collections import Counter

    fine_counts = Counter(
        int(row["target_fine"]) for row in rows if int(row["target_foreground"]) == 1
    )
    maximum = max(fine_counts.values(), default=1)
    weights: list[float] = []
    for row in rows:
        role = str(row["workpoint_role"])
        is_foreground = int(row["target_foreground"]) == 1
        if role == "active_fp":
            weight = 4.0
        elif role == "protected_tp":
            weight = 2.0
        elif is_foreground:
            count = fine_counts[int(row["target_fine"])]
            weight = min(math.sqrt(maximum / count), 10.0)
        else:
            weight = 0.25
        weights.append(float(weight))
    return weights


__all__ = [
    "MetadataStandardizer",
    "PAVManifestDataset",
    "balanced_sampling_weights",
]
