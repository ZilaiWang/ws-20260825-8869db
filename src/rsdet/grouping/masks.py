"""MAR20 飞机背景隔离、稳定视图和背景 tile 生成。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from rsdet.grouping.contracts import canonical_pixel_sha256, stable_json_sha256

ROTATIONS = (0, 90, 180, 270)
VIEW_TYPES = ("original", "masked_inpaint", "background_tiles")


@dataclass(frozen=True)
class RenderedPlaceInput:
    image: Image.Image
    node_uid: str
    view_type: str
    rotation: int
    item_index: int
    input_sha256: str
    source_box: tuple[int, int, int, int] | None
    valid_background_fraction: float


@dataclass(frozen=True)
class MaskedPatchInput:
    """DINO 原图输入及与 patch grid 一一对应的有效背景掩码。"""

    image: Image.Image
    valid_patch_mask: np.ndarray
    node_uid: str
    rotation: int
    input_sha256: str
    patch_mask_sha256: str
    foreground_fraction: float
    valid_patch_fraction: float
    valid_patch_count: int
    patch_count: int


def _validated_boxes(
    boxes: Iterable[Sequence[float]], width: int, height: int
) -> list[tuple[float, float, float, float]]:
    result: list[tuple[float, float, float, float]] = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError("xyxy 框必须为 4 个值")
        x1, y1, x2, y2 = map(float, box)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("xyxy 框必须为有限数")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"xyxy 越界或为空: {(x1, y1, x2, y2)}")
        result.append((x1, y1, x2, y2))
    return result


def build_foreground_mask(
    image_size: tuple[int, int],
    boxes: Iterable[Sequence[float]],
    *,
    dilation_ratio: float,
) -> Image.Image:
    """按每个框宽高外扩后生成 0/255 union mask。"""

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size 必须为正")
    if not 0 <= dilation_ratio <= 1:
        raise ValueError("dilation_ratio 必须在 [0,1]")
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in _validated_boxes(boxes, width, height):
        dx = (x2 - x1) * dilation_ratio
        dy = (y2 - y1) * dilation_ratio
        left = max(0, math.floor(x1 - dx))
        top = max(0, math.floor(y1 - dy))
        right = min(width, math.ceil(x2 + dx))
        bottom = min(height, math.ceil(y2 + dy))
        mask[top:bottom, left:right] = 255
    return Image.fromarray(mask, mode="L")


def build_protocol_foreground_mask(
    image_size: tuple[int, int],
    boxes: Iterable[Sequence[float]],
    *,
    dilation_ratio: float,
    minimum_dilation_pixels: float = 8.0,
    maximum_dilation_pixels: float = 40.0,
) -> Image.Image:
    """Build the v1.2 isotropic aircraft mask: clip(r*max(w,h), 8, 40)."""

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size 必须为正")
    if not 0 <= dilation_ratio <= 1:
        raise ValueError("dilation_ratio 必须在 [0,1]")
    if not 0 <= minimum_dilation_pixels <= maximum_dilation_pixels:
        raise ValueError("掩码外扩像素范围非法")
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in _validated_boxes(boxes, width, height):
        padding = float(
            np.clip(
                dilation_ratio * max(x2 - x1, y2 - y1),
                minimum_dilation_pixels,
                maximum_dilation_pixels,
            )
        )
        left = max(0, math.floor(x1 - padding))
        top = max(0, math.floor(y1 - padding))
        right = min(width, math.ceil(x2 + padding))
        bottom = min(height, math.ceil(y2 + padding))
        mask[top:bottom, left:right] = 255
    return Image.fromarray(mask, mode="L")


def apply_mask_fill(
    image: Image.Image,
    mask: Image.Image,
    *,
    method: str,
    blur_radius: float = 11.0,
    telea_radius: float = 5.0,
) -> Image.Image:
    """只替换 mask 内像素；方法和参数必须写入缓存指纹。"""

    rgb = ImageOps.exif_transpose(image).convert("RGB")
    binary = mask.convert("L")
    if binary.size != rgb.size:
        raise ValueError("image/mask 尺寸不一致")
    mask_array = np.asarray(binary, dtype=np.uint8) > 0
    if not mask_array.any():
        return rgb.copy()
    if mask_array.all():
        raise ValueError("mask 覆盖整图，无法构造背景视图")
    if method == "blur":
        fill = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        output = rgb.copy()
        output.paste(fill, mask=binary)
        return output
    array = np.asarray(rgb, dtype=np.uint8)
    if method == "local_mean":
        mean = np.rint(array[~mask_array].mean(axis=0)).clip(0, 255).astype(np.uint8)
        output_array = array.copy()
        output_array[mask_array] = mean
        return Image.fromarray(output_array, mode="RGB")
    if method == "telea":
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("Telea inpaint 需要 opencv-python-headless") from error
        output_array = cv2.inpaint(
            array,
            mask_array.astype(np.uint8) * 255,
            float(telea_radius),
            cv2.INPAINT_TELEA,
        )
        return Image.fromarray(output_array, mode="RGB")
    raise ValueError(f"未知 mask fill method={method!r}")


def _tile_candidates(
    mask: Image.Image,
    *,
    tile_size: int,
    stride: int,
    min_valid_fraction: float,
) -> list[tuple[float, int, int, int, int]]:
    width, height = mask.size
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size/stride 必须大于 0")
    if tile_size > width or tile_size > height:
        return []
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction 必须在 (0,1]")
    foreground = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    xs = list(range(0, width - tile_size + 1, stride))
    ys = list(range(0, height - tile_size + 1, stride))
    if xs[-1] != width - tile_size:
        xs.append(width - tile_size)
    if ys[-1] != height - tile_size:
        ys.append(height - tile_size)
    result: list[tuple[float, int, int, int, int]] = []
    for y in ys:
        for x in xs:
            valid = 1.0 - float(foreground[y : y + tile_size, x : x + tile_size].mean())
            if valid + 1e-12 >= min_valid_fraction:
                result.append((valid, x, y, x + tile_size, y + tile_size))
    return result


def select_background_tiles(
    image: Image.Image,
    mask: Image.Image,
    *,
    tile_size: int = 224,
    stride: int = 112,
    min_valid_fraction: float = 0.95,
    max_tiles: int = 8,
) -> list[tuple[Image.Image, tuple[int, int, int, int], float]]:
    """按有效背景和空间覆盖确定性选择最多 max_tiles 个 tile。"""

    if max_tiles <= 0:
        raise ValueError("max_tiles 必须大于 0")
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    if rgb.size != mask.size:
        raise ValueError("image/mask 尺寸不一致")
    candidates = _tile_candidates(
        mask,
        tile_size=tile_size,
        stride=stride,
        min_valid_fraction=min_valid_fraction,
    )
    if not candidates:
        return []
    # 第一块优先有效背景最多；后续最大化与已选中心的最小距离。
    candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
    selected = [candidates.pop(0)]
    diagonal = max(math.hypot(*rgb.size), 1.0)
    while candidates and len(selected) < max_tiles:

        def score(item: tuple[float, int, int, int, int]) -> tuple[float, float, int, int]:
            _, x1, y1, x2, y2 = item
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            minimum = min(
                math.hypot(
                    center[0] - (other[1] + other[3]) / 2.0,
                    center[1] - (other[2] + other[4]) / 2.0,
                )
                / diagonal
                for other in selected
            )
            return minimum, item[0], -y1, -x1

        best = max(candidates, key=score)
        candidates.remove(best)
        selected.append(best)
    result = []
    for valid, x1, y1, x2, y2 in selected:
        result.append((rgb.crop((x1, y1, x2, y2)), (x1, y1, x2, y2), valid))
    return result


def rotate_image(image: Image.Image, rotation: int) -> Image.Image:
    if rotation not in ROTATIONS:
        raise ValueError(f"rotation 必须属于 {ROTATIONS}")
    transpose = {
        0: None,
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }[rotation]
    return image.copy() if transpose is None else image.transpose(transpose)


def resize_for_dinov2(image: Image.Image, size: int = 518) -> Image.Image:
    if size <= 0 or size % 14 != 0:
        raise ValueError("DINOv2 输入尺寸必须为 14 的正整数倍")
    return image.convert("RGB").resize((size, size), resample=Image.Resampling.BICUBIC)


def patch_validity_from_mask(
    mask: Image.Image,
    *,
    input_size: int = 518,
    patch_size: int = 14,
    maximum_foreground_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate valid patch mask and per-patch foreground fraction.

    A patch remains available for place aggregation only when the fraction covered
    by the dilated aircraft mask is no greater than the frozen threshold.  Nearest
    resize preserves the binary mask contract; the returned layout follows the
    row-major DINO patch-token order.
    """

    if input_size <= 0 or patch_size <= 0 or input_size % patch_size:
        raise ValueError("input_size 必须是 patch_size 的正整数倍")
    if not 0 <= maximum_foreground_fraction <= 1:
        raise ValueError("maximum_foreground_fraction 必须在 [0,1]")
    resized = mask.convert("L").resize((input_size, input_size), resample=Image.Resampling.NEAREST)
    foreground = np.asarray(resized, dtype=np.uint8) > 0
    grid = input_size // patch_size
    fractions = foreground.reshape(grid, patch_size, grid, patch_size).mean(axis=(1, 3))
    valid = fractions <= maximum_foreground_fraction + 1e-12
    return valid.reshape(-1), fractions.astype(np.float32, copy=False).reshape(-1)


def render_masked_patch_inputs(
    *,
    node_uid: str,
    image: Image.Image,
    boxes: Iterable[Sequence[float]],
    rotations: Sequence[int] = ROTATIONS,
    input_size: int = 518,
    patch_size: int = 14,
    dilation_ratio: float = 0.15,
    maximum_patch_foreground_fraction: float = 0.20,
) -> list[MaskedPatchInput]:
    """Render original-image DINO inputs with synchronized feature-level masks."""

    rotations = tuple(rotations)
    if not rotations or len(rotations) != len(set(rotations)):
        raise ValueError("rotations 必须非空且无重复")
    if any(value not in ROTATIONS for value in rotations):
        raise ValueError(f"rotations 必须来自 {ROTATIONS}")
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    mask = build_protocol_foreground_mask(
        rgb.size,
        boxes,
        dilation_ratio=dilation_ratio,
    )
    canonical_image = resize_for_dinov2(rgb, input_size)
    canonical_mask = mask.resize((input_size, input_size), resample=Image.Resampling.NEAREST)
    patch_count = (input_size // patch_size) ** 2
    outputs: list[MaskedPatchInput] = []
    for rotation in rotations:
        rotated_image = rotate_image(canonical_image, rotation)
        rotated_mask = rotate_image(canonical_mask, rotation)
        valid, fractions = patch_validity_from_mask(
            rotated_mask,
            input_size=input_size,
            patch_size=patch_size,
            maximum_foreground_fraction=maximum_patch_foreground_fraction,
        )
        valid_count = int(valid.sum())
        if valid_count <= 0:
            raise ValueError(f"{node_uid} rotation={rotation}: 无有效背景 patch")
        outputs.append(
            MaskedPatchInput(
                image=rotated_image,
                valid_patch_mask=valid.astype(bool, copy=False),
                node_uid=node_uid,
                rotation=rotation,
                input_sha256=canonical_pixel_sha256(rotated_image),
                patch_mask_sha256=stable_json_sha256(
                    {
                        "valid": valid.astype(np.uint8).tolist(),
                        "maximum_foreground_fraction": maximum_patch_foreground_fraction,
                    }
                ),
                foreground_fraction=float(fractions.mean()),
                valid_patch_fraction=float(valid.mean()),
                valid_patch_count=valid_count,
                patch_count=patch_count,
            )
        )
    return outputs


def render_place_inputs(
    *,
    node_uid: str,
    image: Image.Image,
    boxes: Iterable[Sequence[float]],
    view_types: Sequence[str],
    rotations: Sequence[int] = ROTATIONS,
    input_size: int = 518,
    dilation_ratio: float = 0.15,
    fill_method: str = "telea",
    tile_size: int = 224,
    tile_stride: int = 112,
    tile_valid_fraction: float = 0.95,
    max_tiles: int = 8,
) -> list[RenderedPlaceInput]:
    """按冻结合同渲染 DINO 输入；background tile 保留独立 item。"""

    unknown = set(view_types) - set(VIEW_TYPES)
    if unknown:
        raise ValueError(f"未知 view_types: {sorted(unknown)}")
    if not view_types or len(view_types) != len(set(view_types)):
        raise ValueError("view_types 必须非空且无重复")
    rotations = tuple(rotations)
    if not rotations or any(value not in ROTATIONS for value in rotations):
        raise ValueError(f"rotations 必须来自 {ROTATIONS}")
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    boxes = list(boxes)
    mask = build_foreground_mask(rgb.size, boxes, dilation_ratio=dilation_ratio)
    mask_fraction = float((np.asarray(mask, dtype=np.uint8) > 0).mean())
    base_items: dict[str, list[tuple[Image.Image, tuple[int, int, int, int] | None, float]]] = {}
    if "original" in view_types:
        base_items["original"] = [(rgb, None, 1.0 - mask_fraction)]
    if "masked_inpaint" in view_types:
        base_items["masked_inpaint"] = [
            (apply_mask_fill(rgb, mask, method=fill_method), None, 1.0 - mask_fraction)
        ]
    if "background_tiles" in view_types:
        base_items["background_tiles"] = select_background_tiles(
            rgb,
            mask,
            tile_size=tile_size,
            stride=tile_stride,
            min_valid_fraction=tile_valid_fraction,
            max_tiles=max_tiles,
        )
    result: list[RenderedPlaceInput] = []
    for view_type in view_types:
        for item_index, (base, source_box, valid_fraction) in enumerate(base_items[view_type]):
            canonical = resize_for_dinov2(base, input_size)
            for rotation in rotations:
                rendered = rotate_image(canonical, rotation)
                result.append(
                    RenderedPlaceInput(
                        image=rendered,
                        node_uid=node_uid,
                        view_type=view_type,
                        rotation=rotation,
                        item_index=item_index,
                        input_sha256=canonical_pixel_sha256(rendered),
                        source_box=source_box,
                        valid_background_fraction=valid_fraction,
                    )
                )
    return result
