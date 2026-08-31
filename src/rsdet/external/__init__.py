"""Audited external remote-sensing dataset adapters."""

from rsdet.external.dior import DIOR_TO_COARSE, import_dior
from rsdet.external.dota import COARSE_CATEGORIES, DOTA_TO_COARSE, import_dota

__all__ = [
    "COARSE_CATEGORIES",
    "DIOR_TO_COARSE",
    "DOTA_TO_COARSE",
    "import_dior",
    "import_dota",
]
