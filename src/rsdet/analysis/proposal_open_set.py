"""Pure-Python label contract for V5 proposal-domain open-set evidence."""

from __future__ import annotations

OPEN_FOREGROUND = 0
OPEN_STRUCTURED_BACKGROUND = 1
OPEN_ORDINARY_BACKGROUND = 2
OPEN_IGNORE = -1
OPEN_LABEL_NAMES = (
    "foreground",
    "structured_background",
    "ordinary_background",
)
STRUCTURED_CONFUSER_CLASSES = frozenset({2, 3, 23, 24})


def proposal_open_set_label(
    *,
    is_valid: bool,
    fine_correct: bool,
    crop_top1_class: int,
    structured_classes: frozenset[int] = STRUCTURED_CONFUSER_CLASSES,
) -> int:
    """Map an OOF proposal to the registered three-way target."""

    if not 0 <= int(crop_top1_class) <= 24:
        raise ValueError("crop_top1_class must be within 0..24")
    if is_valid:
        return OPEN_FOREGROUND
    if fine_correct:
        return OPEN_IGNORE
    return (
        OPEN_STRUCTURED_BACKGROUND
        if int(crop_top1_class) in structured_classes
        else OPEN_ORDINARY_BACKGROUND
    )


__all__ = [
    "OPEN_FOREGROUND",
    "OPEN_IGNORE",
    "OPEN_LABEL_NAMES",
    "OPEN_ORDINARY_BACKGROUND",
    "OPEN_STRUCTURED_BACKGROUND",
    "STRUCTURED_CONFUSER_CLASSES",
    "proposal_open_set_label",
]
