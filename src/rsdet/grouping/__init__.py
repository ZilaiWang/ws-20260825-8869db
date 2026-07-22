"""MAR20 来源重叠防护分组工具。"""

from rsdet.grouping.contracts import (
    PROTOCOL_VERSION,
    EvidenceLabel,
    Scope,
    canonical_pair_uid,
)

__all__ = ["EvidenceLabel", "PROTOCOL_VERSION", "Scope", "canonical_pair_uid"]
