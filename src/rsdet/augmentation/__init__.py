"""Target-conditioned augmentation utilities for HERA-Guard APEX."""

from .background_retrieval import BackgroundDescriptor, describe_image, retrieve_backgrounds
from .jitter_hard_negative import Box, JitterPolicy, iou, sample_hard_negative_boxes
from .object_scale_refinement import ScaleCropPolicy, build_scale_crop
from .prototype_memory import PrototypeMemoryBank

__all__ = [
    "BackgroundDescriptor",
    "Box",
    "JitterPolicy",
    "PrototypeMemoryBank",
    "ScaleCropPolicy",
    "build_scale_crop",
    "describe_image",
    "iou",
    "retrieve_backgrounds",
    "sample_hard_negative_boxes",
]
