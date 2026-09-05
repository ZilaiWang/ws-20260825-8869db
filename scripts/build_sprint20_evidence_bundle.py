#!/usr/bin/env python3
"""Build the read-only Sprint20 evidence-boundary bundle.

The bundle records historical assets, data/training lineage, selection history,
and declared test coverage.  It deliberately makes no accuracy or deployment
decision.  Historical hashes are frozen constants instead of values learned
from the files at execution time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "reports/audits/HERA_SPRINT20_EVIDENCE_20260905"

FROZEN_HASHES = {
    "v2_build_manifest": "1627addf575858ba39c879fc7f565d10d2cde98f548fdeecb7edb1cd6793312d",
    "v2_config_dist": "50156c2d3143f930cb6f07f0a72e76b69ad64897f363e2fdea05718c80a52e22",
    "v2_config_source": "50156c2d3143f930cb6f07f0a72e76b69ad64897f363e2fdea05718c80a52e22",
    "v2_weight": "b0df7981f6ad58fe8eca65fb0deef54feed55caf300c2c219ac1ccb3500c8012",
    "v2_adapter": "e65f4afb237fa0206fdadf1b11f66f660966073051d8fc6519d0b42547f43097",
    "v2_competition_source": "68a8a59d1de6cad54946dfe01ed04ff5fc6d36c67138d0ac617c78b0282acb2a",
    "v2_push_log": "9982debe652dff88ec38c86e5d322d27e91e895e7961d38d3c41d1768cfe48b2",
    "v2_image_static": "216b15e44c3ccaf2ddb9084afc1d5c4199919fa61bb0b6640fcf0fb895196a4f",
    "v2_parity_result": "53b29b7aa28b00e015c3aa6a4c86bbb26d6db7e9809d49548c07b0c12e17fbdd",
    "v2_parity_summary": "844a21c2e821a4c52f818776de280278387e1469f2b22e2a46635e08c123d4b9",
    "split": "27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331",
    "split_groups": "5b87536cd49eb1ebcb79a0d1bc539a623d07c19557e07a082954cdb793ce2033",
    "sprint20_asset_audit": "a08eea9990e8c48f3fadd653980eae95774a91721d5219a5d98ef487bb425165",
    "sprint20_shared_oto_parity": "e6c33f43392186c2f580a0b0bba365179a5308052b2f3b431edcf2961c7a9c63",
    "sprint20_shared_otm_parity": "e6a7a387b501bf7fba34a11d84218755a39e2061b5df17c39cafdb70f0ce3b16",
    "sprint20_oof_manifest": "5a06773d18977a8d8330d4b6d2561eac236564bb526a1e8d4132964b7b754c60",
    "sprint20_oof_v7": "ede8175c93a24ba302f062d90d6e836a48a5aa54e825eac7e526c18bc876150f",
    "fold0_contract": "b3c2c5d3ef1c4be3cc3a4f7720881e2b4f91504829716d8c4d56497e1b367530",
    "fold0_result": "d7a41e8808ffac859c082c3da01a5a53a5370efe32165bc0f37380c8e084d08e",
    "fold0_args": "6d979c6ca55da8010c8f0250246b705acc18cb4d720e47b5dd04b9e1c865db99",
    "fold1_contract": "7e107ee64968a32459271fbea69d379b4c61974d2d837ab92db15f9ec37927c2",
    "fold1_result": "d42a1746fa9a94a3277d16170140ae2239146b25a72bf018aa01d3dc82b7466a",
    "fold1_args": "cb19fb04efd39f1b400444f2e19fe69611e3ede8fe9867b9dae1998a6e95cb3e",
    "fold2_contract": "82480945def99e578e86455ed2f215f56b425e8c1b7a40d634a9083d542b579f",
    "fold2_result": "4bafd10e21740966bd36601ad80b7fb155c99be15ae8749cd1b9d3e9c6c9d179",
    "fold2_args": "c1fd6f588d00a6dac0dc1eda8851b5ea2d3ae88bf1f11aea61dd05e8882ef769",
}

ARTIFACT_PATHS = {
    "v2_build_manifest": "dist/p40-full-s1280-frozen0536-final/BUILD_MANIFEST.json",
    "v2_config_dist": "dist/p40-full-s1280-frozen0536-final/app/config.json",
    "v2_config_source": "submission/docker/configs/progressive40_full_s1280_frozen0536_v1.json",
    "v2_weight": "dist/p40-full-s1280-frozen0536-final/models/model.pt",
    "v2_adapter": "dist/p40-full-s1280-frozen0536-final/app/rsdet/models/ultralytics_adapter.py",
    "v2_competition_source": "dist/p40-full-s1280-frozen0536-final/app/rsdet/submission/competition.py",
    "v2_push_log": "outputs/P40-DEPLOYMENT-PREFLIGHT-20260903/push-v2.0-20260903-161501.log",
    "v2_image_static": "outputs/P40-DEPLOYMENT-PREFLIGHT-20260903/image_static.log",
    "v2_parity_result": "outputs/P40-DEPLOYMENT-PREFLIGHT-20260903/parity/result.json",
    "v2_parity_summary": "outputs/P40-DEPLOYMENT-PREFLIGHT-20260903/parity/parity_summary.json",
    "split": "data/splits/cv3_airport_proxy_k60_v2.json",
    "split_groups": "data/splits/cv3_airport_proxy_k60_v2_groups.json",
    "sprint20_asset_audit": "outputs/HERA-SPRINT20-20260905/asset_audit.json",
    "sprint20_shared_oto_parity": "outputs/HERA-SPRINT20-20260905/full_seen/parity_oto.json",
    "sprint20_shared_otm_parity": "outputs/HERA-SPRINT20-20260905/full_seen/parity_otm.json",
    "sprint20_oof_manifest": "outputs/HERA-SPRINT20-20260905/p40_short_oof/input_manifest.json",
    "sprint20_oof_v7": "outputs/HERA-SPRINT20-20260905/p40_short_oof/crossfit_routing_v7.json",
}
for _fold in range(3):
    _base = f"outputs/HERA-SPRINT20-20260905/p40_short_oof/lineage/fold_{_fold}"
    ARTIFACT_PATHS[f"fold{_fold}_contract"] = f"{_base}/training_contract.json"
    ARTIFACT_PATHS[f"fold{_fold}_result"] = f"{_base}/training_result.json"
    ARTIFACT_PATHS[f"fold{_fold}_args"] = f"{_base}/p40_args.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_fingerprint(repo: Path, paths: list[str]) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = []
    for relative in sorted(paths):
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        file_hash = sha256_file(path)
        files.append({"path": relative, "sha256": file_hash, "bytes": path.stat().st_size})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash))
    return {"aggregate_sha256": digest.hexdigest(), "files": files}


def _fold_groups(split: dict[str, Any]) -> dict[int, list[str]]:
    result: dict[int, set[str]] = {0: set(), 1: set(), 2: set()}
    for sample in split["samples"]:
        fold = int(sample["fold"])
        if fold not in result:
            raise ValueError(f"Unexpected fold {fold}")
        result[fold].add(str(sample["group_id"]))
    groups = {fold: sorted(values) for fold, values in result.items()}
    flattened = [group for fold in range(3) for group in groups[fold]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("A source group appears in more than one fold")
    return groups


def _asset_rows(repo: Path) -> list[dict[str, Any]]:
    rows = []
    for artifact_id, relative in ARTIFACT_PATHS.items():
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        actual = sha256_file(path)
        expected = FROZEN_HASHES[artifact_id]
        if actual != expected:
            raise ValueError(
                f"Frozen artifact changed: {relative}: expected {expected}, got {actual}"
            )
        rows.append(
            {
                "id": artifact_id,
                "path": relative,
                "expected_sha256": expected,
            }
        )
    return rows


def _reference_assets(repo: Path) -> dict[str, Any]:
    manifest = _json(repo / ARTIFACT_PATHS["v2_build_manifest"])
    sprint20_code = [
        str(path.relative_to(repo))
        for path in (repo / "src/sprint20").glob("*.py")
    ]
    sprint20_code.extend(
        [
            "configs/experiments/hera_sprint20_p40_d4_otm_ship23_candidate_v1.json",
            "scripts/analyze_sprint20_oof_routing.py",
        ]
    )
    sprint20_fingerprint = _tree_fingerprint(repo, sprint20_code)
    return {
        "schema_version": 1,
        "audit_date": "2026-09-05",
        "historical_submission": {
            "tag": "v2.0",
            "submission_id": 3953,
            "platform_score": 76.6010,
            "materialized_directory": "dist/p40-full-s1280-frozen0536-final",
            "materialized_manifest_sha256": FROZEN_HASHES["v2_build_manifest"],
            "manifest_source_commit": manifest["source_commit"],
            "manifest_source_tree_dirty": manifest["source_tree_dirty"],
            "local_image_id": "sha256:db2a0eaacc0608eecd80193f2cefb83995214288da0250d405b8f016e8ae1303",
            "registry_push_digest": "sha256:db2a0eaacc0608eecd80193f2cefb83995214288da0250d405b8f016e8ae1303",
            "platform_pulled_digest": "UNKNOWN",
            "identity_status": "strong_local_chain_not_platform_digest_attestation",
            "head_path_status": "strongly_supported_by_exact_assets_and_pinned_source_not_platform_attested",
            "notes": [
                "The exact materialized source is authoritative because the manifest records a dirty source tree.",
                "The platform response does not expose the digest it pulled.",
                "The historical adapter/config use the pinned Ultralytics 8.4.103 shared_offline path, but no platform field explicitly names OTO or OTM.",
            ],
        },
        "sprint20_native_control": {
            "code_base": "ab51106949ad8369a5cccd862fcc19e1739cdeb2 plus the exact Sprint20 files fingerprinted below",
            "code_fingerprint": sprint20_fingerprint,
            "weight_sha256": FROZEN_HASHES["v2_weight"],
            "environment": {
                "ultralytics": "8.4.103",
                "torch": "2.5.1+cu121",
                "gpu": "RTX 3090",
            },
            "outputs": [
                ARTIFACT_PATHS["sprint20_asset_audit"],
                ARTIFACT_PATHS["sprint20_oof_manifest"],
                ARTIFACT_PATHS["sprint20_oof_v7"],
            ],
            "publication_parity": "NOT_DIRECTLY_PROVEN_BY_SPRINT20_NATIVE_COMPARISON",
        },
        "sprint20_shared_implementation": {
            "code_base": "same exact Sprint20 file set as native control",
            "code_aggregate_sha256": sprint20_fingerprint["aggregate_sha256"],
            "shared_oto_vs_native_oto": "PASS_4481_OF_4481_EXACT",
            "shared_otm_vs_native_otm": "FAIL_61_IMAGES_COORDINATE_DIFFERENCES",
            "outputs": [
                ARTIFACT_PATHS["sprint20_shared_oto_parity"],
                ARTIFACT_PATHS["sprint20_shared_otm_parity"],
            ],
            "deployment_admission": False,
        },
        "published_parity_scope": {
            "entrypoint_vs_frozen_ledger": "PASS_12_IMAGES_3913_BOXES_EXACT",
            "result": ARTIFACT_PATHS["v2_parity_result"],
            "summary": ARTIFACT_PATHS["v2_parity_summary"],
            "limitation": "This comparison proves the supplied historical entrypoint/ledger pair only; it does not independently attest the platform-pulled image digest or name the detector head.",
        },
        "artifact_inventory": [
            {
                "id": row["id"],
                "path": row["path"],
                "sha256": row["expected_sha256"],
                "bytes": (repo / row["path"]).stat().st_size,
            }
            for row in _asset_rows(repo)
        ],
    }


def _lineage_manifest(repo: Path) -> dict[str, Any]:
    split = _json(repo / ARTIFACT_PATHS["split"])
    folds = _fold_groups(split)
    all_groups = sorted({group for values in folds.values() for group in values})
    datasets = []
    for fold in range(3):
        datasets.extend(
            [
                {
                    "id": f"train_fold{fold}",
                    "groups": sorted(
                        group
                        for other, values in folds.items()
                        if other != fold
                        for group in values
                    ),
                    "role": "training",
                    "disclosure": "inspected",
                },
                {
                    "id": f"eval_fold{fold}",
                    "groups": folds[fold],
                    "role": "diagnostic",
                    "disclosure": "inspected",
                },
            ]
        )
    datasets.extend(
        [
            {
                "id": "oof_all",
                "groups": all_groups,
                "role": "development",
                "disclosure": "inspected",
            },
            {
                "id": "full_seen",
                "groups": all_groups,
                "role": "training",
                "disclosure": "inspected",
            },
            {
                "id": "initializer_sources",
                "groups": ["UNKNOWN_OFFICIAL_PRETRAINING_SOURCE"],
                "role": "training",
                "disclosure": "unknown",
            },
        ]
    )
    nodes = [
        {
            "id": "official_initializer",
            "kind": "fit",
            "parents": [],
            "exposure_datasets": ["initializer_sources"],
            "lineage_complete": False,
        }
    ]
    for fold in range(3):
        nodes.extend(
            [
                {
                    "id": f"s1024_fold{fold}_40e",
                    "kind": "fit",
                    "parents": ["official_initializer"],
                    "exposure_datasets": [f"train_fold{fold}"],
                    "lineage_complete": True,
                },
                {
                    "id": f"p40_fold{fold}_40e",
                    "kind": "fit",
                    "parents": [f"s1024_fold{fold}_40e"],
                    "exposure_datasets": [f"train_fold{fold}"],
                    "lineage_complete": True,
                },
            ]
        )
    for fold in range(3):
        other = [candidate for candidate in range(3) if candidate != fold]
        nodes.extend(
            [
                {
                    "id": f"select_threshold_for_fold{fold}",
                    "kind": "select",
                    "parents": [f"p40_fold{value}_40e" for value in other],
                    "exposure_datasets": [f"eval_fold{value}" for value in other],
                    "lineage_complete": True,
                },
                {
                    "id": f"fixed_qhs_ms_policy_fold{fold}",
                    "kind": "select",
                    "parents": [
                        f"p40_fold{fold}_40e",
                        f"select_threshold_for_fold{fold}",
                    ],
                    "exposure_datasets": [],
                    "lineage_complete": True,
                },
            ]
        )
    nodes.extend(
        [
            {
                "id": "p40_full_mature",
                "kind": "fit",
                "parents": ["official_initializer"],
                "exposure_datasets": ["full_seen"],
                "lineage_complete": True,
            },
            {
                "id": "posthoc_scope_selection_qhs_ms",
                "kind": "select",
                "parents": [
                    "fixed_qhs_ms_policy_fold0",
                    "fixed_qhs_ms_policy_fold1",
                    "fixed_qhs_ms_policy_fold2",
                    "p40_full_mature",
                ],
                "exposure_datasets": ["oof_all", "full_seen"],
                "lineage_complete": True,
            },
            {
                "id": "sprint20_qhs_ms_development_result",
                "kind": "inspect",
                "parents": ["posthoc_scope_selection_qhs_ms"],
                "exposure_datasets": ["oof_all"],
                "lineage_complete": True,
            },
        ]
    )
    claims = [
        {
            "id": f"threshold_policy_fold{fold}_strict_independence",
            "candidate_node": f"fixed_qhs_ms_policy_fold{fold}",
            "evaluation_dataset": f"eval_fold{fold}",
            "claimed_role": "independent_confirmation",
        }
        for fold in range(3)
    ]
    claims.append(
        {
            "id": "qhs_ms_overall_development_evidence",
            "candidate_node": "sprint20_qhs_ms_development_result",
            "evaluation_dataset": "oof_all",
            "claimed_role": "development",
        }
    )
    return {
        "schema_version": 1,
        "notes": (
            "Read-only Sprint20 lineage declaration. The audit intentionally records "
            "post-hoc scope selection and indirect cross-fit exposure. It is not an accuracy "
            "or deployment approval."
        ),
        "datasets": datasets,
        "nodes": nodes,
        "claims": claims,
        "artifacts": _asset_rows(repo),
        "comparisons": [
            {
                "id": "historical_dist_vs_source_config",
                "left": "v2_config_dist",
                "right": "v2_config_source",
                "mode": "json_exact",
            }
        ],
        "signature_checks": [
            {
                "id": "short_oof_vs_mature_full_training",
                "required_fields": [
                    "stages",
                    "stage_epochs",
                    "data_scope",
                    "initialization_lineage",
                    "effective_batch",
                    "resume_or_migration",
                    "selection_rule",
                ],
                "left": {
                    "stages": ["S1024", "P40_1280"],
                    "stage_epochs": [40, 40],
                    "data_scope": "two_source_disjoint_folds_per_model",
                    "initialization_lineage": "official_initializer_then_fold_s1024",
                    "effective_batch": "S1024 not fully reconstructed here; P40 batch8 per single-GPU fold",
                    "resume_or_migration": "fresh_stage2_from_fold_s1024_last",
                    "selection_rule": "fixed_last",
                },
                "right": {
                    "stages": ["S1024", "P40_1280"],
                    "stage_epochs": [160, 40],
                    "data_scope": "all_4481_official_training_images",
                    "initialization_lineage": "official_initializer_then_full_s1024",
                    "effective_batch": "S1024 not reconstructed here; P40 nominal64 for epochs1-2 then nominal72 after three-GPU migration",
                    "resume_or_migration": "full_s1024_then_stage2_with_recorded_ddp_recovery",
                    "selection_rule": "fixed_last",
                },
            }
        ],
        "test_coverage": [
            {
                "id": "shared_head_control_flow",
                "required_cases": [
                    "native_oto_real_weight",
                    "native_otm_real_weight",
                    "shared_oto_full_dataset",
                    "shared_otm_full_dataset",
                    "empty_batch",
                    "batch_tail",
                    "missing_or_fused_head",
                    "invalid_class_count",
                    "pixel_hash_collision_guard",
                    "actual_gpu_container_entrypoint",
                ],
                "covered_cases": [
                    "native_oto_real_weight",
                    "native_otm_real_weight",
                    "shared_oto_full_dataset",
                    "shared_otm_full_dataset",
                    "missing_or_fused_head",
                    "invalid_class_count",
                    "pixel_hash_collision_guard",
                ],
            },
            {
                "id": "bounded_d4_control_flow",
                "required_cases": [
                    "certified_early_keep",
                    "full_fallback",
                    "mixed_batch",
                    "no_aircraft",
                    "probability_boundary",
                    "invalid_probability",
                    "invalid_label",
                    "batch_tail",
                    "hard100_output_parity",
                    "end_to_end_container_timing",
                ],
                "covered_cases": [
                    "certified_early_keep",
                    "full_fallback",
                    "mixed_batch",
                    "no_aircraft",
                    "probability_boundary",
                    "invalid_probability",
                    "invalid_label",
                    "hard100_output_parity",
                ],
            },
            {
                "id": "historical_v2_delivery",
                "required_cases": [
                    "frozen_config_hash",
                    "frozen_weight_hash",
                    "historical_source_hash",
                    "12_image_entrypoint_parity",
                    "cpu_no_network_smoke",
                    "linux_direct_gpu_entrypoint",
                    "gpu_container_entrypoint",
                    "platform_pulled_digest_attestation",
                ],
                "covered_cases": [
                    "frozen_config_hash",
                    "frozen_weight_hash",
                    "historical_source_hash",
                    "12_image_entrypoint_parity",
                    "cpu_no_network_smoke",
                    "linux_direct_gpu_entrypoint",
                ],
            },
        ],
    }


def _test_coverage() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "declared Sprint20 and historical v2 branches",
        "principle": "Test count is not used as a substitute for branch coverage.",
        "suites": [
            {
                "id": "sprint20_unit_and_repository",
                "status": "passed_for_executed_cases",
                "covered": [
                    "audit ancestor and unrelated-commit fail-closed paths",
                    "official matcher/scorer regression and coarse/fine taxonomy",
                    "empty rare-class handling and negative images",
                    "inclusive score-threshold boundary",
                    "selection/evaluation group-overlap rejection",
                    "native head pre-fusion audit and missing-head failure",
                    "instance-local head switch and reused-model rejection",
                    "cache pixel-hash guard and deep copy",
                    "class ownership and duplicate-preserving multiset comparison",
                    "D4 certification boundary, adversarial completions, relabel and NMS",
                    "invalid probabilities, labels and view counts",
                ],
                "not_covered": [
                    "actual official platform image digest attestation",
                    "actual GPU Docker execution of the historical v2 image",
                    "all empty/tail batches on the shared-head real-weight path",
                ],
            },
            {
                "id": "sprint20_real_gpu_integration",
                "status": "mixed",
                "covered": [
                    "native/shared decode integration with pinned Ultralytics",
                    "26-image exact parity for both heads",
                    "4481-image exact parity for shared OTO",
                    "100-hard-case D4 fallback/early-keep mixture and exact output",
                ],
                "failed_or_rejected": [
                    "shared OTM exact parity failed on 61 of 4481 images",
                    "bounded D4 increased matched hard100 local total time by 3.93 percent",
                ],
                "not_covered": [
                    "official GPU container end-to-end latency",
                    "unseen-platform input behavior",
                ],
            },
            {
                "id": "evidence_audit_tool",
                "status": "46_passed_zero_skipped",
                "covered": [
                    "normal manifests",
                    "path traversal and escaping symlinks",
                    "duplicate JSON keys and non-finite values",
                    "missing files and SHA mismatch",
                    "dependency cycles and indirect overlap",
                    "unknown lineage, signature fields and missing coverage",
                    "refusal to overwrite an existing report",
                ],
                "limitation": "Synthetic tests validate the metadata tool, not Sprint20 model accuracy or deployment.",
            },
        ],
        "timing_boundary": {
            "native_and_shared_head_times": "local component measurements, not official end-to-end latency",
            "d4_times": "same-process local classifier/runtime measurements; no percentage is extrapolated to the full platform",
        },
        "duplicate_command_display": {
            "conclusion": "NO_EVIDENCE_OF_DUPLICATE_CONCURRENT_WRITER",
            "basis": "The orchestration trace contained one running execution/session while the UI rendered the same command twice; completed artifacts used non-overwriting paths. Historical OS process state can no longer be independently reconstructed.",
        },
    }


SELECTION_HISTORY = """# Sprint20 selection history

Date: 2026-09-05. This ledger preserves every material variant inspected in order. It is not a release decision.

|Order|Variant or question|Data inspected|Observed result|Decision at the time|Evidence role after audit|
|---:|---|---|---|---|---|
|1|Native OTO vs native OTM wiring|26 continuous images, 216 GT|Ship TP unchanged; OTM added 2 FP; Aircraft unchanged|Continue only as a mechanism probe|Integration sanity|
|2|Native OTM replaces all output|4481 full-seen images|Recall rose but FDR also rose|Do not replace the whole detector|Training-seen diagnostic|
|3|OTM owns all Ship classes 0–3|4481 full-seen|Score delta -0.1756|Reject broad Ship ownership|Training-seen selection input|
|4|OTM owns QHS/MS classes 2–3|4481 full-seen|Score delta +0.8811; +114 TP/+23 FP|Retain for OOF development|Training-seen selection input|
|5|OTM owns FSC class 24|4481 full-seen|Score delta +1.6423|Retain temporarily for OOF check|Training-seen selection input|
|6|OTM owns Ship and FSC|4481 full-seen|Score delta +1.4668|Do not prefer over narrower options|Training-seen selection input|
|7|All OTM policy|short three-fold OOF|Fold deltas -2.9933/+2.5309/+1.8621 at target 0.10|Reject as unstable|Development-selected OOF|
|8|OTM owns all Ship|short three-fold OOF|Fold deltas +0.4161/+0.6166/+1.9716|Reject after considering rare-class/full-seen risk|Development-selected OOF|
|9|OTM owns QHS/MS only|short three-fold OOF|Fold deltas +0.2985/+0.2155/+0.2625; group bootstrap positive probability 98.32%|Choose as narrow research candidate|Post-hoc development evidence, not independent confirmation|
|10|OTM owns FSC only|short three-fold OOF|Fold deltas -3.5072/+1.4532/-0.7300|Reject|Development-selected OOF|
|11|Exact fixed-primary QHS/MS policy at same Ship risk|short three-fold OOF, exact v7 replay|Fold deltas +0.2847/+0.1666/+0.2601; merged +0.2401; bootstrap positive probability 98.32%|Keep as mechanism evidence|Post-hoc development evidence|
|12|QHS/MS at arbitrary fixed macro-FDR targets|same short OOF|Deltas at 0.10/0.12/0.15/0.20 were -1.6044/-1.6660/-1.6691/-1.1726|Do not describe arbitrary-target results as positive|Development diagnostic|
|13|Shared OTO implementation|26 images then 4481 images|Exact parity on 4481/4481|Implementation parity passed|Implementation evidence only|
|14|Shared OTM implementation|26 images then 4481 images|61 images had coordinate differences|Reject shared-head deployment|Failed implementation parity|
|15|Bounded D4 early exit|26 easy images, historical CE hard100, current-consistency hard100|Exact outputs; matched hard100 local time +3.93%|Reject bounded D4 speed path|Engineering diagnostic|

The QHS/MS subset was chosen after several scopes and all three folds/full-seen diagnostics were inspected. Cross-fitted thresholds do not restore independence for the higher-level scope choice. The three fold fits also overlap in training sources, so their positive deltas are correlated, not three independent trials.
"""


EVIDENCE_LIMITATIONS = """# Sprint20 evidence limitations

1. The historical v2.0 reference has a strong local identity chain: exact materialized directory and manifest, frozen configuration/weight/source hashes, local image ID, matching registry push digest, successful platform record, and a 12-image entrypoint parity result. The platform did not return the digest it actually pulled, so this is not platform-side digest attestation.
2. The historical build manifest records `source_tree_dirty=true`. Its exact materialized source files and manifest are authoritative; the recorded Git commit alone is insufficient to reproduce the image.
3. The exact configuration and pinned historical source strongly support the baseline execution path, but no platform result field explicitly identifies the detector head as OTO or OTM. The report must not infer the actual submitted head only from a current framework default.
4. The three P40 fold models are source-disjoint for their direct held-out predictions, but their lineage is `S1024/40e -> P40/40e`, not the mature full model's `S1024/160e -> P40/40e`. Their role is `outer_oof_short` directional development evidence.
5. QHS/MS was selected after whole-head, broad class-scope, narrow class-scope, fold and full-seen results were inspected. Its positive fold deltas and source bootstrap therefore describe a development-selected procedure, not an untouched confirmation.
6. Selecting a threshold for one fold from predictions produced by the other fold models is not a strict nested evaluation: those other models were trained on the held-out fold's source groups. The dependency audit records these indirect paths explicitly.
7. K-fold training sets overlap. Three positive fold deltas are correlated and cannot be interpreted as three independent successes. Source-group bootstrap quantifies conditional variation in the already-inspected data; it does not remove selection bias or create a new domain.
8. Full-seen results are mechanism diagnostics and selection inputs because the full model trained on all 4,481 images. They are not generalization evidence.
9. Shared OTO parity passed on all 4,481 images, while shared OTM parity failed on 61 images. No tolerance was introduced after observing this failure. The shared OTM path is not deployment-admitted.
10. Bounded D4 preserved outputs in the tested samples, but the current hard100 path was slower. Component timings are not official end-to-end timings and are not extrapolated to a platform score.
11. Test counts establish only that executed cases passed. Missing empty/tail real-weight paths, actual GPU Docker execution, and platform digest attestation remain missing regardless of the total number of tests.
12. The external evidence-audit utility checks declared metadata, hashes, dependency paths and self-reported coverage. It does not inspect images, deserialize weights, measure performance, prove grouping correctness, or approve a release.
13. The duplicate SSH command displayed in the orchestration transcript is not proof of two writers. One execution/session was observed, but historical process state cannot now be independently reconstructed.
14. No second server was used for this audit because all remaining work was deterministic, read-only metadata validation. Repeating the same selection on another server would not create independent evidence.
15. Several large or operational artifacts under `dist/` and `outputs/` are intentionally not tracked by Git. The lineage manifest is directly runnable in this audited local workspace; a clean clone retains paths and frozen hashes but cannot re-hash absent local artifacts until they are restored from the project archive.
"""


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_bundle(repo: Path, output: Path, *, replace: bool) -> None:
    files = {
        "actual_reference_assets.json": json.dumps(
            _reference_assets(repo), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        "training_and_selection_lineage.json": json.dumps(
            _lineage_manifest(repo), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        "selection_history.md": SELECTION_HISTORY,
        "test_case_coverage.json": json.dumps(
            _test_coverage(), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        "evidence_limitations.md": EVIDENCE_LIMITATIONS,
    }
    output.mkdir(parents=True, exist_ok=True)
    existing = [output / name for name in files if (output / name).exists()]
    if existing and not replace:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite evidence outputs: {joined}")
    for name, content in files.items():
        _atomic_write(output / name, content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _write_bundle(args.repo.resolve(), args.output.resolve(), replace=args.replace)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
