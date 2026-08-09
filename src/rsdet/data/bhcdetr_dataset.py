"""PyTorch input pipeline for the paper-aligned BHC-DETR detector.

The competition annotations are YOLO horizontal boxes.  This module keeps the
source data immutable, applies the two-view augmentation described by the
paper for training, and emits DETR targets as normalized ``cxcywh`` boxes.

``BHCDetrDataset[index]`` returns a list of view dictionaries.  Training
samples contain ``views_per_image`` independently seeded views (two by
default), while evaluation samples contain one deterministic view.  Use
:func:`bhcdetr_collate` to flatten those views into one model batch.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset

    _TORCH_AVAILABLE = True
except (
    ImportError,
    OSError,
):  # pragma: no cover - exercised in torch-free environments
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

    class Dataset:  # type: ignore[no-redef]
        pass


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_MAX_TORCH_IMAGE_ID = 2**63 - 1


@dataclass(frozen=True)
class BHCDetrSample:
    """One source image and its YOLO annotation file."""

    image_id: int
    relative_path: str
    image_path: Path
    label_path: Path


def _safe_relative_path(value: Any, *, field: str, row_index: int) -> str:
    """Normalize a manifest path while rejecting traversal and drive paths."""

    if value is None:
        raise ValueError(f"manifest row {row_index} has no {field}")
    text = str(value).strip().replace("\\", "/")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        not text
        or "\x00" in text
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"manifest row {row_index} contains unsafe {field}={value!r}")
    return posix.as_posix()


def _resolve_under_root(root: Path, relative_path: str, *, field: str) -> Path:
    parts = PurePosixPath(relative_path).parts
    resolved = root.joinpath(*parts).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{field} escapes data_root: {relative_path!r}")
    return resolved


def _row_path(row: Mapping[str, Any], *, row_index: int) -> str:
    for field in (
        "relative_path",
        "image_relative_path",
        "source_relative_path",
        "image_path",
        "image",
        "path",
    ):
        if field in row and str(row[field]).strip():
            return _safe_relative_path(row[field], field=field, row_index=row_index)
    raise ValueError(f"manifest row {row_index} has no image relative path")


def _row_label_path(row: Mapping[str, Any], *, row_index: int) -> str | None:
    for field in ("label_relative_path", "label_path", "annotation_path"):
        if field in row and str(row[field]).strip():
            return _safe_relative_path(row[field], field=field, row_index=row_index)
    return None


def _derived_label_path(image_relative_path: str) -> str:
    image_path = PurePosixPath(image_relative_path)
    parts = list(image_path.parts)
    try:
        image_prefix = parts.index("images")
    except ValueError:
        parts.insert(0, "labels")
    else:
        parts[image_prefix] = "labels"
    return PurePosixPath(*parts).with_suffix(".txt").as_posix()


def _parse_image_id(value: Any, *, row_index: int) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"manifest row {row_index} has invalid image_id={value!r}")
    try:
        parsed = int(str(value).strip())
    except ValueError as error:
        raise ValueError(
            f"manifest row {row_index} has invalid image_id={value!r}"
        ) from error
    if str(parsed) != str(value).strip() or not 0 < parsed <= _MAX_TORCH_IMAGE_ID:
        raise ValueError(f"manifest row {row_index} has invalid image_id={value!r}")
    return parsed


def _read_manifest(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            rows = payload.get("samples")
        else:
            rows = payload
        if not isinstance(rows, list):
            raise TypeError(
                "JSON manifest must be a list or an object with a samples list"
            )
        if any(not isinstance(row, Mapping) for row in rows):
            raise TypeError("every JSON manifest sample must be an object")
        return list(rows)
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"CSV manifest is empty: {path}")
        return rows
    raise ValueError("manifest must be a .json or .csv file")


def _select_manifest_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    held_out_fold: int | None,
    csv_view: bool,
) -> list[tuple[int, Mapping[str, Any], str]]:
    prepared: list[tuple[int, Mapping[str, Any], str]] = []
    has_split: list[bool] = []
    has_fold: list[bool] = []
    for row_index, row in enumerate(rows):
        relative_path = _row_path(row, row_index=row_index)
        prepared.append((row_index, row, relative_path))
        has_split.append("split" in row and bool(str(row.get("split", "")).strip()))
        has_fold.append("fold" in row and bool(str(row.get("fold", "")).strip()))

    if any(has_split) and not all(has_split):
        raise ValueError("manifest mixes samples with and without split")
    if any(has_fold) and not all(has_fold):
        raise ValueError("manifest mixes samples with and without fold")

    if all(has_split):
        selected = [
            item
            for item in prepared
            if str(item[1]["split"]).strip().lower() == split.lower()
        ]
    elif all(has_fold):
        if type(held_out_fold) is not int or held_out_fold < 0:
            raise ValueError("fold manifest requires a non-negative held_out_fold")
        if split.lower() not in {"train", "val"}:
            raise ValueError("fold manifest split must be 'train' or 'val'")
        selected = []
        for item in prepared:
            try:
                fold = int(str(item[1]["fold"]).strip())
            except ValueError as error:
                raise ValueError(
                    f"manifest row {item[0]} has invalid fold={item[1]['fold']!r}"
                ) from error
            if fold < 0 or str(fold) != str(item[1]["fold"]).strip():
                raise ValueError(
                    f"manifest row {item[0]} has invalid fold={item[1]['fold']!r}"
                )
            is_validation = fold == held_out_fold
            if (split.lower() == "val") == is_validation:
                selected.append(item)
    elif csv_view:
        # A CSV file may itself be an already-materialized split view.
        selected = prepared
    else:
        raise ValueError("JSON samples require either split or fold fields")

    if not selected:
        raise ValueError(f"manifest selected no samples for split={split!r}")
    return selected


def _manifest_samples(
    data_root: Path,
    manifest: Path,
    *,
    split: str,
    held_out_fold: int | None,
    require_labels: bool,
) -> tuple[BHCDetrSample, ...]:
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    rows = _read_manifest(manifest)
    selected = _select_manifest_rows(
        rows,
        split=split,
        held_out_fold=held_out_fold,
        csv_view=manifest.suffix.lower() == ".csv",
    )
    parsed: list[tuple[int | None, str, Path, Path]] = []
    explicit_ids: list[bool] = []
    seen_paths: set[str] = set()
    for row_index, row, relative_path in selected:
        key = relative_path.casefold()
        if key in seen_paths:
            raise ValueError(
                f"manifest contains duplicate image path={relative_path!r}"
            )
        seen_paths.add(key)
        image_path = _resolve_under_root(data_root, relative_path, field="image path")
        if image_path.suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
            raise FileNotFoundError(image_path)
        explicit_label = _row_label_path(row, row_index=row_index)
        label_relative = explicit_label or _derived_label_path(relative_path)
        label_path = _resolve_under_root(data_root, label_relative, field="label path")
        if require_labels and not label_path.is_file():
            raise FileNotFoundError(label_path)
        image_id = _parse_image_id(row.get("image_id"), row_index=row_index)
        explicit_ids.append(image_id is not None)
        parsed.append((image_id, relative_path, image_path, label_path))

    if any(explicit_ids) and not all(explicit_ids):
        raise ValueError("manifest mixes samples with and without image_id")
    if all(explicit_ids):
        seen_ids: set[int] = set()
        for image_id, _relative, _image, _label in parsed:
            assert image_id is not None
            if image_id in seen_ids:
                raise ValueError(f"manifest contains duplicate image_id={image_id}")
            seen_ids.add(image_id)
        ordered = sorted(parsed, key=lambda item: (int(item[0]), item[1].casefold()))
    else:
        ordered = sorted(parsed, key=lambda item: item[1].casefold())

    samples: list[BHCDetrSample] = []
    for position, (image_id, relative_path, image_path, label_path) in enumerate(
        ordered, start=1
    ):
        samples.append(
            BHCDetrSample(
                image_id=image_id if image_id is not None else position,
                relative_path=relative_path,
                image_path=image_path,
                label_path=label_path,
            )
        )
    return tuple(samples)


def _directory_samples(
    data_root: Path,
    *,
    split: str,
    require_labels: bool,
) -> tuple[BHCDetrSample, ...]:
    image_dir = (data_root / "images" / split).resolve()
    label_dir = (data_root / "labels" / split).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    if require_labels and not label_dir.is_dir():
        raise FileNotFoundError(label_dir)
    images = sorted(
        (
            path.resolve()
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(data_root).as_posix().casefold(),
    )
    samples: list[BHCDetrSample] = []
    for image_id, image_path in enumerate(images, start=1):
        image_relative = image_path.relative_to(data_root).as_posix()
        within_split = image_path.relative_to(image_dir).with_suffix(".txt")
        label_path = (label_dir / within_split).resolve()
        if not label_path.is_relative_to(data_root):
            raise ValueError(f"label path escapes data_root: {label_path}")
        if require_labels and not label_path.is_file():
            raise FileNotFoundError(label_path)
        samples.append(
            BHCDetrSample(
                image_id=image_id,
                relative_path=image_relative,
                image_path=image_path,
                label_path=label_path,
            )
        )
    if not samples:
        raise ValueError(f"no supported images found in {image_dir}")
    return tuple(samples)


def _read_yolo_boxes(
    path: Path,
    *,
    image_width: int,
    image_height: int,
    num_classes: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)

    boxes: list[list[float]] = []
    labels: list[int] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: YOLO row must contain five values")
        try:
            label = int(fields[0])
            cx, cy, width, height = (float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid YOLO value") from error
        if label < 0 or (num_classes is not None and label >= num_classes):
            raise ValueError(f"{path}:{line_number}: invalid class id {label}")
        values = np.asarray((cx, cy, width, height), dtype=np.float64)
        if not np.isfinite(values).all() or width <= 0.0 or height <= 0.0:
            raise ValueError(f"{path}:{line_number}: invalid YOLO box")
        x0 = (cx - width / 2.0) * image_width
        y0 = (cy - height / 2.0) * image_height
        x1 = (cx + width / 2.0) * image_width
        y1 = (cy + height / 2.0) * image_height
        clipped = [
            min(max(x0, 0.0), float(image_width)),
            min(max(y0, 0.0), float(image_height)),
            min(max(x1, 0.0), float(image_width)),
            min(max(y1, 0.0), float(image_height)),
        ]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            raise ValueError(f"{path}:{line_number}: YOLO box is outside the image")
        boxes.append(clipped)
        labels.append(label)
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(labels, dtype=np.int64),
    )


def _shift_image(
    image: np.ndarray,
    valid_pixels: np.ndarray,
    *,
    dx: int,
    dy: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = valid_pixels.shape
    shifted = np.zeros_like(image)
    shifted_valid = np.zeros_like(valid_pixels)
    source_x0 = max(0, -dx)
    source_y0 = max(0, -dy)
    source_x1 = min(width, width - dx)
    source_y1 = min(height, height - dy)
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        return shifted, shifted_valid
    target_x0 = source_x0 + dx
    target_y0 = source_y0 + dy
    target_x1 = source_x1 + dx
    target_y1 = source_y1 + dy
    shifted[target_y0:target_y1, target_x0:target_x1] = image[
        source_y0:source_y1, source_x0:source_x1
    ]
    shifted_valid[target_y0:target_y1, target_x0:target_x1] = valid_pixels[
        source_y0:source_y1, source_x0:source_x1
    ]
    return shifted, shifted_valid


def _augment(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
    flip_probability: float,
    max_shift_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = image.shape[:2]
    output = image
    output_boxes = boxes.copy()
    valid_pixels = np.ones((height, width), dtype=np.bool_)
    horizontal = bool(rng.random() < flip_probability)
    vertical = bool(rng.random() < flip_probability)
    if horizontal:
        output = np.ascontiguousarray(output[:, ::-1])
        valid_pixels = np.ascontiguousarray(valid_pixels[:, ::-1])
        if len(output_boxes):
            old_x0 = output_boxes[:, 0].copy()
            output_boxes[:, 0] = width - output_boxes[:, 2]
            output_boxes[:, 2] = width - old_x0
    if vertical:
        output = np.ascontiguousarray(output[::-1, :])
        valid_pixels = np.ascontiguousarray(valid_pixels[::-1, :])
        if len(output_boxes):
            old_y0 = output_boxes[:, 1].copy()
            output_boxes[:, 1] = height - output_boxes[:, 3]
            output_boxes[:, 3] = height - old_y0

    max_dx = int(math.floor(width * max_shift_ratio))
    max_dy = int(math.floor(height * max_shift_ratio))
    dx = int(rng.integers(-max_dx, max_dx + 1)) if max_dx else 0
    dy = int(rng.integers(-max_dy, max_dy + 1)) if max_dy else 0
    if dx or dy:
        output, valid_pixels = _shift_image(output, valid_pixels, dx=dx, dy=dy)
        output_boxes[:, (0, 2)] += dx
        output_boxes[:, (1, 3)] += dy
        output_boxes[:, (0, 2)] = np.clip(output_boxes[:, (0, 2)], 0.0, float(width))
        output_boxes[:, (1, 3)] = np.clip(output_boxes[:, (1, 3)], 0.0, float(height))

    keep = (output_boxes[:, 2] > output_boxes[:, 0]) & (
        output_boxes[:, 3] > output_boxes[:, 1]
    )
    return (
        np.ascontiguousarray(output),
        output_boxes[keep],
        labels[keep],
        valid_pixels,
        {
            "horizontal_flip": horizontal,
            "vertical_flip": vertical,
            "shift_x": dx,
            "shift_y": dy,
        },
    )


def _letterbox(
    image: np.ndarray,
    boxes: np.ndarray,
    valid_pixels: np.ndarray,
    *,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = image.shape[:2]
    nominal_scale = min(image_size / width, image_size / height)
    resized_width = min(image_size, max(1, int(round(width * nominal_scale))))
    resized_height = min(image_size, max(1, int(round(height * nominal_scale))))
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2

    resized = np.asarray(
        Image.fromarray(image, mode="RGB").resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )
    resized_valid = (
        np.asarray(
            Image.fromarray(valid_pixels.astype(np.uint8) * 255, mode="L").resize(
                (resized_width, resized_height), Image.Resampling.NEAREST
            )
        )
        > 0
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    valid_canvas = np.zeros((image_size, image_size), dtype=np.bool_)
    valid_canvas[top : top + resized_height, left : left + resized_width] = (
        resized_valid
    )

    transformed = boxes.copy()
    if len(transformed):
        scale_x = resized_width / width
        scale_y = resized_height / height
        transformed[:, (0, 2)] = transformed[:, (0, 2)] * scale_x + left
        transformed[:, (1, 3)] = transformed[:, (1, 3)] * scale_y + top
        transformed[:, (0, 2)] = np.clip(transformed[:, (0, 2)], 0.0, float(image_size))
        transformed[:, (1, 3)] = np.clip(transformed[:, (1, 3)], 0.0, float(image_size))
    return (
        canvas,
        ~valid_canvas,
        transformed,
        {
            "scale_x": resized_width / width,
            "scale_y": resized_height / height,
            "left": left,
            "top": top,
            "resized_width": resized_width,
            "resized_height": resized_height,
        },
    )


def _normalized_cxcywh(boxes: np.ndarray, image_size: int) -> np.ndarray:
    result = np.empty_like(boxes, dtype=np.float32)
    if not len(boxes):
        return result.reshape(0, 4)
    result[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5 / image_size
    result[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5 / image_size
    result[:, 2] = (boxes[:, 2] - boxes[:, 0]) / image_size
    result[:, 3] = (boxes[:, 3] - boxes[:, 1]) / image_size
    return np.clip(result, 0.0, 1.0)


class BHCDetrDataset(Dataset):
    """YOLO dataset producing paper-style multi-view BHC-DETR inputs.

    Args:
        data_root: Root containing ``images`` and ``labels``.
        split: Runtime split name, normally ``train`` or ``val``.
        manifest: Optional JSON samples manifest or materialized CSV split view.
        held_out_fold: For samples with ``fold``, that fold is validation and
            all remaining folds are training.
        training: Defaults to ``split == 'train'``.  Evaluation always emits a
            single unaugmented view.
        max_shift_ratio: Maximum integer translation independently along each
            image axis, expressed as a fraction of that axis.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        *,
        manifest: str | Path | None = None,
        held_out_fold: int | None = None,
        image_size: int = 1024,
        training: bool | None = None,
        views_per_image: int = 2,
        flip_probability: float = 0.5,
        max_shift_ratio: float = 0.05,
        seed: int = 0,
        require_labels: bool = True,
        num_classes: int | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("BHCDetrDataset requires PyTorch")
        if not isinstance(split, str) or not split.strip():
            raise ValueError("split must be a non-empty string")
        if (
            isinstance(image_size, bool)
            or not isinstance(image_size, int)
            or image_size <= 0
        ):
            raise ValueError("image_size must be a positive integer")
        if (
            isinstance(views_per_image, bool)
            or not isinstance(views_per_image, int)
            or views_per_image <= 0
        ):
            raise ValueError("views_per_image must be a positive integer")
        if (
            not math.isfinite(float(flip_probability))
            or not 0.0 <= flip_probability <= 1.0
        ):
            raise ValueError("flip_probability must be in [0, 1]")
        if (
            not math.isfinite(float(max_shift_ratio))
            or not 0.0 <= max_shift_ratio < 1.0
        ):
            raise ValueError("max_shift_ratio must be in [0, 1)")
        if training is not None and type(training) is not bool:
            raise ValueError("training must be bool or None")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if num_classes is not None and (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes <= 0
        ):
            raise ValueError("num_classes must be a positive integer or None")

        self.data_root = Path(data_root).expanduser().resolve()
        if not self.data_root.is_dir():
            raise FileNotFoundError(self.data_root)
        self.split = split.strip().lower()
        self.image_size = image_size
        self.training = self.split == "train" if training is None else bool(training)
        self.views_per_image = views_per_image
        self.flip_probability = float(flip_probability)
        self.max_shift_ratio = float(max_shift_ratio)
        self.seed = seed
        self.epoch = 0
        self.require_labels = bool(require_labels)
        self.num_classes = num_classes
        if manifest is None:
            self.samples = _directory_samples(
                self.data_root,
                split=self.split,
                require_labels=self.require_labels,
            )
            self.manifest_path: Path | None = None
        else:
            self.manifest_path = Path(manifest).expanduser().resolve()
            self.samples = _manifest_samples(
                self.data_root,
                self.manifest_path,
                split=self.split,
                held_out_fold=held_out_fold,
                require_labels=self.require_labels,
            )

        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        self._normalization_mean = mean
        self._normalization_std = std

    def set_epoch(self, epoch: int) -> None:
        """Select deterministic but different augmentation streams per epoch."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def _rng(self, image_id: int, view_index: int) -> np.random.Generator:
        sequence = np.random.SeedSequence([self.seed, self.epoch, image_id, view_index])
        return np.random.default_rng(sequence)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        height, width = image.shape[:2]
        boxes, labels = _read_yolo_boxes(
            sample.label_path,
            image_width=width,
            image_height=height,
            num_classes=self.num_classes,
        )
        view_count = self.views_per_image if self.training else 1
        views: list[dict[str, Any]] = []
        for view_index in range(view_count):
            if self.training:
                (
                    view_image,
                    view_boxes,
                    view_labels,
                    valid_pixels,
                    augmentation,
                ) = _augment(
                    image,
                    boxes,
                    labels,
                    rng=self._rng(sample.image_id, view_index),
                    flip_probability=self.flip_probability,
                    max_shift_ratio=self.max_shift_ratio,
                )
            else:
                view_image = image.copy()
                view_boxes = boxes.copy()
                view_labels = labels.copy()
                valid_pixels = np.ones((height, width), dtype=np.bool_)
                augmentation = {
                    "horizontal_flip": False,
                    "vertical_flip": False,
                    "shift_x": 0,
                    "shift_y": 0,
                }
            letterboxed, padding_mask, pixel_boxes, letterbox = _letterbox(
                view_image,
                view_boxes,
                valid_pixels,
                image_size=self.image_size,
            )
            normalized_boxes = _normalized_cxcywh(pixel_boxes, self.image_size)
            chw = np.ascontiguousarray(letterboxed.transpose(2, 0, 1))
            image_tensor = torch.from_numpy(chw).to(dtype=torch.float32).div_(255.0)
            image_tensor = (
                image_tensor - self._normalization_mean
            ) / self._normalization_std
            boxes_tensor = torch.from_numpy(normalized_boxes.copy()).to(torch.float32)
            labels_tensor = torch.from_numpy(view_labels.copy()).to(torch.int64)
            target = {
                "boxes": boxes_tensor,
                "labels": labels_tensor,
                "area": boxes_tensor[:, 2] * boxes_tensor[:, 3],
                "iscrowd": torch.zeros(len(labels_tensor), dtype=torch.int64),
                "image_id": torch.tensor(sample.image_id, dtype=torch.int64),
                "source_image_id": torch.tensor(sample.image_id, dtype=torch.int64),
                "view_id": torch.tensor(view_index, dtype=torch.int64),
                "orig_size": torch.tensor((height, width), dtype=torch.int64),
                "size": torch.tensor(
                    (self.image_size, self.image_size), dtype=torch.int64
                ),
                "relative_path": sample.relative_path,
                "box_format": "cxcywh_normalized",
                "augmentation": augmentation,
                "letterbox": letterbox,
            }
            views.append(
                {
                    "image": image_tensor,
                    "padding_mask": torch.from_numpy(padding_mask.copy()).to(
                        torch.bool
                    ),
                    "target": target,
                }
            )
        return views


def bhcdetr_collate(batch: Sequence[Any]) -> dict[str, Any]:
    """Flatten per-source views and stack tensors for :class:`BHCDetr`."""

    flattened: list[Mapping[str, Any]] = []
    for item in batch:
        if isinstance(item, Mapping):
            flattened.append(item)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if any(not isinstance(view, Mapping) for view in item):
                raise TypeError("every BHC-DETR view must be a mapping")
            flattened.extend(item)
        else:
            raise TypeError("BHC-DETR batch items must be views or sequences of views")
    if not flattened:
        raise ValueError("cannot collate an empty BHC-DETR batch")
    images = torch.stack([view["image"] for view in flattened], dim=0)
    padding_masks = torch.stack([view["padding_mask"] for view in flattened], dim=0)
    targets = [view["target"] for view in flattened]
    return {
        "images": images,
        "padding_masks": padding_masks,
        "targets": targets,
    }


# Compatibility spellings for callers that use an all-caps acronym or ``_fn``.
BHCDETRDataset = BHCDetrDataset
bhcdetr_collate_fn = bhcdetr_collate


__all__ = [
    "BHCDETRDataset",
    "BHCDetrDataset",
    "BHCDetrSample",
    "IMAGE_SUFFIXES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "bhcdetr_collate",
    "bhcdetr_collate_fn",
]
