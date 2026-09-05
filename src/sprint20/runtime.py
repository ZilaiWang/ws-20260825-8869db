"""Additive official-contract entrypoint; no edits to the incumbent's sources."""

from __future__ import annotations

from .policy import WEAK_LABELS, validate_rate


def detector_factory(config):
    from rsdet.models.ultralytics_adapter import UltralyticsDetector
    from rsdet.submission.competition import CompetitionDetector

    from .bounded_d4 import BoundedD4Runtime
    from .heads import SharedHeadCapture, SharedHeadPipeline, select_native_otm

    options = dict(config.get("sprint20") or {})
    mode = options.get("mode", "oto")
    bounded = options.get("bounded_d4", False)
    if not isinstance(bounded, bool) or mode not in {"oto", "otm", "shared"}:
        raise ValueError("invalid sprint20 options")
    if (mode != "oto" or bounded) and options.get("allow_experimental") is not True:
        raise ValueError("Experimental mode requires explicit allow_experimental=true")
    if mode != "oto" and (config.get("agreement_model") or config.get("resolution_expert_model")):
        raise ValueError("Cannot mix sprint head probe with historical experts/rescue")
    detector = CompetitionDetector(config)
    if mode != "oto":
        if not isinstance(detector.detector, UltralyticsDetector):
            raise ValueError("Readout probe requires inference_adapter=shared_offline")
        if mode == "otm":
            select_native_otm(detector.detector)
        else:
            labels = options.get("otm_labels")
            if not isinstance(labels, list) or not labels or len(set(labels)) != len(labels):
                raise ValueError("shared mode requires explicit unique otm_labels")
            if any(
                isinstance(x, bool) or not isinstance(x, int) or x not in WEAK_LABELS
                for x in labels
            ):
                raise ValueError("OTM may own only Ship/FSC labels")
            t0 = validate_rate(config.get("post_fusion_score_threshold"), "P40 threshold")
            t1 = validate_rate(options.get("otm_threshold"), "OTM threshold")
            if t1 < float(config["pipeline"]["score_threshold"]):
                raise ValueError("OTM threshold is below the cached candidate floor")
            capture = SharedHeadCapture(detector.detector)
            detector.resolution_runtime = SharedHeadPipeline(
                capture,
                detector.pipeline_config,
                otm_labels=labels,
                primary_threshold=t0,
                otm_threshold=t1,
            )
            # Both owners were filtered separately AFTER independent fusion.
            # A second uniform P40 threshold would wrongly delete OTM outputs.
            detector.config["post_fusion_score_threshold"] = None
    if bounded:
        if detector.aircraft_classifier is None:
            raise ValueError("bounded_d4 requires the frozen aircraft classifier")
        detector.aircraft_classifier = BoundedD4Runtime(
            detector.aircraft_classifier, margin=float(options.get("bounded_margin", 2e-4))
        )
    return detector
