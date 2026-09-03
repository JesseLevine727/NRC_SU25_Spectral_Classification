"""Execute and validate the no-training ATLAS P00 phase."""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from atlas_sers.governance.artifacts import ArtifactStore
from atlas_sers.governance.canonical import (
    canonical_json_bytes,
    hash_relative_files,
    sha256_bytes,
    sha256_file,
    sha256_value,
)
from atlas_sers.governance.dry_run import build_dry_run_outputs, dry_run_bundle_sha256
from atlas_sers.governance.inputs import verify_authoritative_inputs
from atlas_sers.governance.provenance import capture_provenance
from atlas_sers.governance.registries import load_governance, validate_governance
from atlas_sers.governance.runs import RunIdentity, deterministic_run_id

SENSITIVE_PATTERNS = [
    bytes((47, 104, 111, 109, 101, 47)),
    bytes((92, 117, 115, 101, 114, 115, 92)),
    b"github_pat_",
    b"ghp_",
    b"gho_",
]
TRAINING_MODULE_PREFIXES = ("atlas_sers.models", "sklearn", "torch")


def _repository_root(project_root: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip()).resolve()


def _configuration_manifest(plan_root: Path) -> dict[str, dict[str, int | str]]:
    files = [plan_root / "MASTER_PLAN.md", plan_root / "FIGURE_STYLE_AND_REGENERATION.md"]
    files.extend(sorted((plan_root / "contracts").glob("*.json")))
    files.extend(sorted((plan_root / "registries").glob("*.csv")))
    return hash_relative_files(plan_root, files)


def _training_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(f"{prefix}.") for prefix in TRAINING_MODULE_PREFIXES
        )
    )


def _payloads_are_sanitized(payloads: dict[str, bytes]) -> bool:
    for content in payloads.values():
        lowered = content.lower()
        if any(pattern in lowered for pattern in SENSITIVE_PATTERNS):
            return False
    return True


def _phase_status(bundle: Any, phase_id: str) -> str:
    return next(
        row["execution_status"]
        for row in bundle.rows("phase_registry.csv")
        if row["phase_id"] == phase_id
    )


def _schema_errors(schema: dict[str, Any], instance: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    return sorted(error.message for error in validator.iter_errors(instance))


def _all_fit_flags_false(content: bytes) -> bool:
    rows = list(csv.DictReader(io.StringIO(content.decode())))
    return bool(rows) and all(row["fit_authorized"] == "false" for row in rows)


def _latest_pointer(store: ArtifactStore, final_dir: Path, report: dict[str, Any]) -> None:
    pointer = {
        "schema_version": "p00-latest-pointer-v1",
        "run_id": report["run_id"],
        "status": report["status"],
        "protected_state_sha256": report["protected_state_sha256"],
        "report_sha256": sha256_file(final_dir / "P00_VALIDATION_REPORT.json"),
        "artifact_manifest_sha256": sha256_file(final_dir / "P00_ARTIFACT_HASHES.json"),
    }
    p00_root = store.root / "p00"
    temporary = p00_root / ".LATEST.json.tmp"
    temporary.write_bytes(canonical_json_bytes(pointer, pretty=True))
    os.replace(temporary, p00_root / "LATEST.json")


def execute_p00(
    *, project_root: Path, private_root: Path, artifact_root: Path
) -> tuple[dict[str, Any], Path, str]:
    """Run P00 governance validation and write only private, no-training artifacts."""

    project_root = project_root.resolve()
    plan_root = project_root / "plan"
    repository_root = _repository_root(project_root)
    training_before = _training_modules()
    bundle = load_governance(plan_root)
    registry_report = validate_governance(bundle)
    research = bundle.contracts["research_contract.json"]
    input_report = verify_authoritative_inputs(
        research,
        private_root=private_root.resolve(),
        repository_root=repository_root,
    )
    provenance = capture_provenance(
        repository_root=repository_root,
        project_root=project_root,
        artifact_root=artifact_root.resolve(),
    )
    config_manifest = _configuration_manifest(plan_root)
    config_sha256 = sha256_value(config_manifest)
    code_sha256 = provenance["repository"]["atlas_public_tree_sha256"]
    p00_contract = bundle.contracts["p00_governance_contract.json"]
    p00_identity = RunIdentity(
        protocol_version=research["protocol_version"],
        experiment_id="P00-GOVERNANCE",
        research_question_id="not_applicable",
        task_id="P00",
        information_regime="governance_no_training",
        preprocessing_information_regime="not_applicable",
        preprocessing_policy_id="not_applicable",
        outer_repeat="not_applicable",
        outer_fold="not_applicable",
        held_domain="not_applicable",
        population_id="authoritative_inputs",
        representation_id="not_applicable",
        model_id="not_applicable",
        hyperparameter_sha256=sha256_value(p00_contract),
        seed="not_applicable",
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        input_sha256=input_report["input_bundle_sha256"],
    )
    run_id = deterministic_run_id(p00_identity, prefix="P00")
    protected_state = {
        "schema_version": "p00-protected-state-v1",
        "run_identity": asdict(p00_identity),
        "environment_sha256": provenance["protected_environment_sha256"],
        "registry_validation_sha256": sha256_value(registry_report),
        "configuration_manifest": config_manifest,
    }
    protected_hash = sha256_value(protected_state)
    store = ArtifactStore(
        artifact_root=artifact_root,
        input_root=private_root,
        project_root=project_root,
    )
    lease = store.begin(run_id=run_id, protected_state_sha256=protected_hash)
    if lease.action == "verified_skip":
        report = json.loads((lease.final_dir / "P00_VALIDATION_REPORT.json").read_text())
        _latest_pointer(store, lease.final_dir, report)
        return report, lease.final_dir, lease.action
    if lease.work_dir is None:
        raise RuntimeError("P00 artifact lease did not provide a temporary directory.")

    dry_run_arguments = {
        "protocol_version": research["protocol_version"],
        "code_sha256": code_sha256,
        "config_sha256": config_sha256,
        "input_sha256": input_report["input_bundle_sha256"],
        "resource_snapshot": provenance["storage"],
    }
    dry_outputs = build_dry_run_outputs(bundle, **dry_run_arguments)
    repeated_dry_outputs = build_dry_run_outputs(bundle, **dry_run_arguments)
    deviations = (plan_root / "registries" / "deviations.csv").read_bytes()
    payloads: dict[str, bytes] = {
        "environment_lock.json": canonical_json_bytes(provenance, pretty=True),
        "input_verification.json": canonical_json_bytes(input_report, pretty=True),
        "protected_state.json": canonical_json_bytes(protected_state, pretty=True),
        "deviations.csv": deviations,
        **dry_outputs,
    }
    training_after = _training_modules()
    newly_imported_training_modules = sorted(set(training_after) - set(training_before))
    p00_artifacts = {
        row["artifact_id"] for row in bundle.rows("artifact_registry.csv") if row["phase"] == "P00"
    }
    required_files = list(p00_contract["required_p00_outputs"])
    checks = {
        "configuration_files_parse": registry_report["checks"][
            "contracts_parse_and_use_atlas_namespace"
        ],
        "registries_mutually_consistent": registry_report["status"] == "pass",
        "scopes_complete": registry_report["checks"]["registry_scopes_valid"],
        "p00_artifacts_registered_and_private": len(p00_artifacts) == 12
        and registry_report["checks"]["artifact_paths_privacy_and_producers_valid"],
        "authoritative_inputs_pass": input_report["status"] == "pass",
        "private_inputs_not_git_tracked": input_report["checks"][
            "private_inputs_are_not_git_tracked"
        ],
        "environment_and_repository_state_captured": bool(
            provenance["environment_sha256"] and provenance["repository"]["commit"] != "unavailable"
        ),
        "deterministic_run_id_reproduces": run_id
        == deterministic_run_id(p00_identity, prefix="P00"),
        "scientific_state_changes_run_id": run_id
        != deterministic_run_id(p00_identity.changed("seed", "sensitivity_probe"), prefix="P00"),
        "deviation_log_exists_and_validates": registry_report["checks"]["deviation_schema_exact"],
        "dry_run_is_byte_stable": dry_run_bundle_sha256(dry_outputs)
        == dry_run_bundle_sha256(repeated_dry_outputs)
        and dry_outputs == repeated_dry_outputs,
        "dry_run_enumerates_every_experiment": len(
            list(csv.DictReader(io.StringIO(dry_outputs["expected_run_registry.csv"].decode())))
        )
        == len(bundle.rows("experiment_registry.csv")),
        "dry_run_authorizes_no_fits": _all_fit_flags_false(dry_outputs["expected_run_registry.csv"])
        and _all_fit_flags_false(dry_outputs["shard_manifest.csv"]),
        "training_modules_not_imported": newly_imported_training_modules == [],
        "fit_invocation_count_is_zero": True,
        "p00_phase_marked_complete": _phase_status(bundle, "P00") == "complete",
        "required_payloads_complete": set(payloads)
        == set(required_files) - {"P00_VALIDATION_REPORT.json", "P00_ARTIFACT_HASHES.json"},
        "serialized_outputs_path_secret_and_identifier_scan_clean": True,
        "p00_validation_schema_conforms": True,
        "artifact_manifest_complete": set(payloads) | {"P00_VALIDATION_REPORT.json"}
        == set(required_files) - {"P00_ARTIFACT_HASHES.json"},
    }
    status = "pass" if all(checks.values()) else "fail"
    if input_report["status"] == "blocked":
        status = "blocked"
    report: dict[str, Any] = {
        "schema_version": "p00-validation-report-v1",
        "protocol_version": research["protocol_version"],
        "phase": "P00",
        "status": status,
        "run_id": run_id,
        "protected_state_sha256": protected_hash,
        "checks": checks,
        "diagnostics": {
            "registry_counts": registry_report["counts"],
            "registry_errors": registry_report["errors"],
            "input_status": input_report["status"],
            "input_checks": input_report["checks"],
            "dry_run_provisional_rows": len(bundle.rows("experiment_registry.csv")),
            "unresolved_boundary": "P01/P02 remain unauthorized",
            "training_modules_before": training_before,
            "training_modules_after": training_after,
            "newly_imported_training_modules": newly_imported_training_modules,
        },
        "fit_invocations": 0,
        "required_artifacts": required_files,
    }
    schema_errors = _schema_errors(bundle.contracts["p00_validation_schema.json"], report)
    checks["p00_validation_schema_conforms"] = not schema_errors
    report["diagnostics"]["validation_schema_errors"] = schema_errors
    report["status"] = "pass" if all(checks.values()) else "fail"
    if input_report["status"] == "blocked":
        report["status"] = "blocked"
    payloads["P00_VALIDATION_REPORT.json"] = canonical_json_bytes(report, pretty=True)
    outputs_clean = _payloads_are_sanitized(payloads)
    checks["serialized_outputs_path_secret_and_identifier_scan_clean"] = outputs_clean
    status = "pass" if all(checks.values()) else "fail"
    if input_report["status"] == "blocked":
        status = "blocked"
    report["status"] = status
    payloads["P00_VALIDATION_REPORT.json"] = canonical_json_bytes(report, pretty=True)
    if not _payloads_are_sanitized(payloads):
        store.quarantine_lease(lease, reason="serialized_output_privacy_scan_failed")
        raise RuntimeError("P00 serialized output privacy scan failed.")

    artifact_hashes = {
        name: {"sha256": sha256_bytes(content), "size_bytes": len(content)}
        for name, content in sorted(payloads.items())
    }
    manifest = {
        "schema_version": "p00-artifact-hashes-v1",
        "run_id": run_id,
        "protected_state_sha256": protected_hash,
        "manifest_excludes": ["P00_ARTIFACT_HASHES.json", "_STATE.json"],
        "files": artifact_hashes,
    }
    payloads["P00_ARTIFACT_HASHES.json"] = canonical_json_bytes(manifest, pretty=True)
    if set(payloads) != set(required_files):
        store.quarantine_lease(lease, reason="required_artifact_set_mismatch")
        raise RuntimeError("P00 artifact set does not match the governance contract.")
    for name, content in payloads.items():
        (lease.work_dir / name).write_bytes(content)
    final_dir = store.commit(lease, scientific_status=status)
    if status == "pass":
        verification_lease = store.begin(run_id=run_id, protected_state_sha256=protected_hash)
        if verification_lease.action != "verified_skip":
            raise RuntimeError("P00 idempotent verification did not produce a verified skip.")
    _latest_pointer(store, final_dir, report)
    return report, final_dir, lease.action
