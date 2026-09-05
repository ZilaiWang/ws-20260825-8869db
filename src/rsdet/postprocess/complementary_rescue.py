"""Extend the same incumbent-preserving rule independently to declared classes."""
from copy import deepcopy

from rsdet.postprocess.vehicle_rescue import append_vehicle_rescue


def append_class_rescue(incumbent, auxiliary, *, category_iou):
    # Validate the complete universe/records before category filtering.
    append_vehicle_rescue(incumbent, auxiliary)
    result = deepcopy(incumbent)
    counts = {"auxiliary_candidates": 0, "suppressed_overlap": 0, "added": 0}
    for category, iou in sorted(category_iou.items()):
        if int(category) != category or category not in range(25):
            raise ValueError("unknown rescue category")
        base = {i: [dict(deepcopy(r), category_id=24) for r in rows
                    if r["category_id"] == category] for i, rows in incumbent.items()}
        aux = {i: [dict(deepcopy(r), category_id=24) for r in rows
                   if r["category_id"] == category] for i, rows in auxiliary.items()}
        merged, stats = append_vehicle_rescue(base, aux, dedup_iou=iou)
        for i in result:
            result[i].extend(dict(r, category_id=category) for r in merged[i][len(base[i]):])
        counts["auxiliary_candidates"] += stats["auxiliary_vehicle"]
        counts["suppressed_overlap"] += stats["suppressed_overlap"]
        counts["added"] += stats["added_vehicle"]
    return result, counts
