"""Audit declared provenance and immutable artifacts; never certify accuracy.

Only standard-library imports are used. No checkpoint is deserialized and no
model, deployment, threshold search, or data-label modification is performed.
The result is conditional on the truth and completeness of the input metadata.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


class AuditInputError(ValueError):
    """Invalid or ambiguous input metadata."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditInputError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _bad_constant(value: str) -> Any:
    raise AuditInputError(f"Non-finite JSON constant: {value}")


def _validate_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AuditInputError("Non-finite numeric value")
    if isinstance(value, dict):
        for item in value.values():
            _validate_finite(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite(item)


def load_json(path: str | Path) -> Any:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_bad_constant,
    )
    _validate_finite(value)
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise AuditInputError("Artifact path must be a nonempty relative string")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise AuditInputError("Artifact paths must be relative to --root")
    root = root.resolve()
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise AuditInputError(f"Artifact escapes root: {relative!r}")
    if not resolved.is_file():
        raise AuditInputError(f"Artifact is not a regular file: {relative!r}")
    return resolved


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditInputError(f"{name} must be a JSON object")
    return value


def _rows(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
        raise AuditInputError(f"{name} must be a list of objects")
    return value


def _strings(value: Any, name: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(x, str) or not x.strip() for x in value
    ):
        raise AuditInputError(f"{name} must be a list of nonblank strings")
    if nonempty and not value:
        raise AuditInputError(f"{name} must not be empty")
    if len(set(value)) != len(value):
        raise AuditInputError(f"{name} contains duplicate identifiers")
    return value


def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("id")
        if not isinstance(key, str) or not key.strip() or key in out:
            raise AuditInputError(f"{name}: missing, blank, or duplicate id {key!r}")
        out[key] = row
    return out


def _escape(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def json_differences(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    """Exact structural differences; arrays remain ordered; types are strict.

    No numerical tolerance, rounding, sorting, or implicit field exclusion is
    applied. Missing values and JSON null are explicitly distinguished.
    """
    if type(left) is not type(right):
        return [{"path": path, "left": left, "right": right, "reason": "type"}]
    if isinstance(left, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            sub = path + "/" + _escape(key)
            if key not in left:
                result.append({"path": sub, "left_missing": True, "right": right[key]})
            elif key not in right:
                result.append({"path": sub, "left": left[key], "right_missing": True})
            else:
                result.extend(json_differences(left[key], right[key], sub))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return [{"path": path, "reason": "array_length", "left_length": len(left),
                     "right_length": len(right)}]
        result = []
        for index, (first, second) in enumerate(zip(left, right)):
            result.extend(json_differences(first, second, path + f"/{index}"))
        return result
    return [] if left == right else [{"path": path, "left": left, "right": right}]


def audit_manifest(manifest: dict[str, Any], root: str | Path) -> dict[str, Any]:
    manifest = _mapping(manifest, "manifest")
    _validate_finite(manifest)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise AuditInputError("schema_version must be integer 1")
    allowed = {"schema_version", "datasets", "nodes", "claims", "artifacts", "comparisons",
               "signature_checks", "test_coverage", "notes"}
    unknown_fields = set(manifest) - allowed
    if unknown_fields:
        raise AuditInputError(f"Unknown top-level fields: {sorted(unknown_fields)}")
    root = Path(root)
    findings: list[dict[str, Any]] = []

    def add(level: str, code: str, detail: Any) -> None:
        findings.append({"level": level, "code": code, "detail": detail})

    datasets = _index(_rows(manifest.get("datasets", []), "datasets"), "datasets")
    nodes = _index(_rows(manifest.get("nodes", []), "nodes"), "nodes")
    artifacts = _index(_rows(manifest.get("artifacts", []), "artifacts"), "artifacts")
    claims = _index(_rows(manifest.get("claims", []), "claims"), "claims")
    groups: dict[str, set[str]] = {}
    for key, row in datasets.items():
        groups[key] = set(_strings(row.get("groups"), f"dataset {key}.groups", nonempty=True))
        if row.get("role") not in {"training", "development", "confirmation", "diagnostic"}:
            raise AuditInputError(f"dataset {key}: invalid role")
        if row.get("disclosure") not in {"untouched", "inspected", "unknown"}:
            raise AuditInputError(f"dataset {key}: invalid disclosure")

    for key, row in nodes.items():
        if row.get("kind") not in {"fit", "predict", "select", "inspect", "package"}:
            raise AuditInputError(f"node {key}: invalid kind")
        _strings(row.get("parents"), f"node {key}.parents")
        _strings(row.get("exposure_datasets"), f"node {key}.exposure_datasets")
        if type(row.get("lineage_complete")) is not bool:
            raise AuditInputError(f"node {key}.lineage_complete must be boolean")
        if set(row["parents"]) - set(nodes):
            raise AuditInputError(f"node {key}: unknown parent")
        if set(row["exposure_datasets"]) - set(datasets):
            raise AuditInputError(f"node {key}: unknown exposure dataset")

    # Validate the whole declared graph, including nodes not currently claimed.
    active: set[str] = set()
    complete: set[str] = set()

    def visit(key: str) -> None:
        if key in active:
            raise AuditInputError(f"Dependency graph contains a cycle at {key}")
        if key in complete:
            return
        active.add(key)
        for parent in nodes[key]["parents"]:
            visit(parent)
        active.remove(key)
        complete.add(key)

    for key in nodes:
        visit(key)

    def ancestry(key: str) -> dict[str, list[str]]:
        paths = {key: [key]}
        stack = [key]
        while stack:
            current = stack.pop()
            for parent in nodes[current]["parents"]:
                if parent not in paths:
                    paths[parent] = paths[current] + [parent]
                    stack.append(parent)
        return paths

    claim_results: list[dict[str, Any]] = []
    for claim_id, claim in claims.items():
        target = claim.get("candidate_node")
        evaluation = claim.get("evaluation_dataset")
        if target not in nodes or evaluation not in datasets:
            raise AuditInputError(f"claim {claim_id}: unknown candidate or evaluation dataset")
        role = claim.get("claimed_role")
        if role not in {"independent_confirmation", "development", "diagnostic"}:
            raise AuditInputError(f"claim {claim_id}: invalid claimed_role")
        paths = ancestry(target)
        overlaps = []
        unknown = []
        for node_id, path in paths.items():
            node = nodes[node_id]
            if not node["lineage_complete"]:
                unknown.append(node_id)
            for data_id in node["exposure_datasets"]:
                common = sorted(groups[data_id] & groups[evaluation])
                if common:
                    overlaps.append({"node": node_id, "kind": node["kind"],
                                     "exposure_dataset": data_id, "groups": common,
                                     "dependency_path": path})
        novelty = datasets[evaluation]["disclosure"]
        dataset_role = datasets[evaluation]["role"]
        if overlaps:
            status = "CONTRADICTED"
        elif unknown:
            status = "UNKNOWN"
        elif novelty == "unknown":
            status = "UNKNOWN"
        elif novelty == "inspected" or dataset_role != "confirmation":
            status = "NOT_AN_UNTOUCHED_CONFIRMATION_SET"
        else:
            status = "CONSISTENT_WITH_DECLARED_METADATA_ONLY"
        result = {"id": claim_id, "claimed_role": role, "status": status,
                  "overlapping_exposures": overlaps, "incomplete_lineage_nodes": unknown,
                  "disclosure": novelty, "evaluation_dataset_role": dataset_role}
        claim_results.append(result)
        if role == "independent_confirmation" and status != "CONSISTENT_WITH_DECLARED_METADATA_ONLY":
            add("error", "INDEPENDENCE_CLAIM_UNSUPPORTED", result)
        elif status != "CONSISTENT_WITH_DECLARED_METADATA_ONLY":
            add("warning", "EVIDENCE_ROLE_LIMITATION", result)

    artifact_results: dict[str, dict[str, Any]] = {}
    artifact_paths: dict[str, Path] = {}
    for key, row in artifacts.items():
        path = contained_file(root, row.get("path"))
        artifact_paths[key] = path
        actual = sha256_file(path)
        expected = row.get("expected_sha256")
        if expected is not None and (
            not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)
        ):
            raise AuditInputError(f"artifact {key}: SHA must be lowercase hexadecimal")
        status = "HASH_MATCH" if actual == expected else (
            "HASH_MISMATCH" if expected is not None else "NO_PREVIOUS_HASH_ATTESTATION")
        artifact_results[key] = {"path": row["path"], "actual_sha256": actual,
                                 "expected_sha256": expected, "status": status,
                                 "bytes": path.stat().st_size}
        if status == "HASH_MISMATCH":
            add("error", "ARTIFACT_HASH_MISMATCH", key)
        elif expected is None:
            add("warning", "ARTIFACT_PREVIOUS_HASH_MISSING", key)

    comparison_results = []
    comparisons = _index(_rows(manifest.get("comparisons", []), "comparisons"), "comparisons")
    for key, row in comparisons.items():
        left_id, right_id = row.get("left"), row.get("right")
        if left_id not in artifact_paths or right_id not in artifact_paths:
            raise AuditInputError(f"comparison {key}: unknown artifact")
        mode = row.get("mode")
        if mode not in {"bytes", "json_exact"}:
            raise AuditInputError(f"comparison {key}: mode must be bytes or json_exact")
        same_bytes = artifact_results[left_id]["actual_sha256"] == artifact_results[right_id]["actual_sha256"]
        diffs = [] if mode == "bytes" else json_differences(
            load_json(artifact_paths[left_id]), load_json(artifact_paths[right_id]))
        equal = same_bytes if mode == "bytes" else not diffs
        result = {"id": key, "mode": mode, "equal": equal, "byte_equal": same_bytes,
                  "difference_count": len(diffs), "differences": diffs,
                  "scope": "these_artifacts_only_not_unseen_inputs"}
        comparison_results.append(result)
        if not equal:
            add("error", "DECLARED_ARTIFACT_EQUALITY_FAILED", result)

    signature_results = []
    signatures = _index(_rows(manifest.get("signature_checks", []), "signature_checks"), "signature_checks")
    for key, row in signatures.items():
        fields = _strings(row.get("required_fields"), f"signature {key}.required_fields", nonempty=True)
        left = _mapping(row.get("left"), f"signature {key}.left")
        right = _mapping(row.get("right"), f"signature {key}.right")
        missing = [field for field in fields if field not in left or field not in right]
        diffs = json_differences({k: left[k] for k in fields if k in left},
                                 {k: right[k] for k in fields if k in right})
        status = "UNKNOWN" if missing else ("DIFFERENT" if diffs else "DECLARED_FIELDS_EQUAL")
        result = {"id": key, "status": status, "missing_fields": missing, "differences": diffs}
        signature_results.append(result)
        if status != "DECLARED_FIELDS_EQUAL":
            add("warning", "TRAINING_SIGNATURE_NOT_EQUIVALENT", result)

    coverage_results = []
    coverage = _index(_rows(manifest.get("test_coverage", []), "test_coverage"), "test_coverage")
    for key, row in coverage.items():
        required = set(_strings(row.get("required_cases"), f"coverage {key}.required_cases", nonempty=True))
        covered = set(_strings(row.get("covered_cases"), f"coverage {key}.covered_cases"))
        missing = sorted(required - covered)
        result = {"id": key, "missing_cases": missing,
                  "status": "INCOMPLETE" if missing else "DECLARED_CASES_COVERED",
                  "coverage_is_self_reported": True}
        coverage_results.append(result)
        if missing:
            add("warning", "REQUIRED_TEST_CASES_NOT_COVERED", result)

    if not any((claims, artifacts, comparisons, signatures, coverage)):
        add("warning", "NO_CHECKS_DECLARED", "No claim, artifact, signature, or case-coverage checks were declared")

    return {
        "schema_version": 1,
        "tool_version": "1.0.0",
        "limitations": [
            "Input metadata and group identifiers are self-reported, not independently authenticated.",
            "An empty overlap cannot prove an incomplete training history is uncontaminated.",
            "This tool does not execute models, optimize metrics, or choose a release.",
            "Artifact equality applies only to supplied files, not all possible inputs.",
            "No statistical confidence or out-of-distribution guarantee is produced.",
        ],
        "accuracy_or_deployment_approved": False,
        "claim_checks": claim_results,
        "artifact_checks": artifact_results,
        "comparisons": comparison_results,
        "signature_checks": signature_results,
        "test_coverage": coverage_results,
        "findings": findings,
        "errors": sum(x["level"] == "error" for x in findings),
        "warnings": sum(x["level"] == "warning" for x in findings),
    }
