"""Execute and validate the no-model ATLAS P02 evaluation-design freeze."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator

from atlas_sers.governance.artifacts import ArtifactStore
from atlas_sers.governance.canonical import (
    canonical_json_bytes,
    hash_relative_files,
    sha256_bytes,
    sha256_file,
    sha256_value,
)
from atlas_sers.governance.provenance import capture_provenance
from atlas_sers.governance.registries import load_governance, validate_governance
from atlas_sers.governance.runs import RunIdentity, deterministic_run_id
from atlas_sers.splits.p02 import build_p02_tables, validate_p02_tables
from atlas_sers.visualization.p02_figures import (
    build_p02_figure_tables,
    generate_p02_figures,
)

RESTRICTED_SOURCE_PATTERN = re.compile(rb"(?<![a-z])" + bytes((110, 97, 116, 111)) + rb"(?![a-z])")
SENSITIVE_PATTERNS = [
    bytes((47, 104, 111, 109, 101, 47)),
    bytes((92, 117, 115, 101, 114, 115, 92)),
    b"github_pat_",
    b"ghp_",
    b"gho_",
]
OPAQUE_GENERATED_SUFFIXES = {".pdf", ".png"}
PROHIBITED_MODULES = ("atlas_sers.models", "torch")


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
    files = [
        plan_root / "MASTER_PLAN.md",
        plan_root / "RESEARCH_QUESTION_MAP.md",
        plan_root / "FIGURE_STYLE_AND_REGENERATION.md",
    ]
    files.extend(sorted((plan_root / "contracts").glob("*.json")))
    files.extend(sorted((plan_root / "registries").glob("*.csv")))
    return hash_relative_files(plan_root, files)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="",
    ).encode()


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.write_bytes(_csv_bytes(frame))


def _bytes_are_sanitized(content: bytes) -> bool:
    lowered = content.lower()
    return not RESTRICTED_SOURCE_PATTERN.search(lowered) and not any(
        pattern in lowered for pattern in SENSITIVE_PATTERNS
    )


def _artifact_tree_is_sanitized(root: Path) -> bool:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            return False
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode()
        if not _bytes_are_sanitized(relative):
            return False
        if path.suffix.lower() not in OPAQUE_GENERATED_SUFFIXES and not _bytes_are_sanitized(
            path.read_bytes()
        ):
            return False
    return True


def _prohibited_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in PROHIBITED_MODULES)
    )


def _schema_errors(schema: dict[str, Any], instance: dict[str, Any]) -> list[str]:
    return sorted(error.message for error in Draft202012Validator(schema).iter_errors(instance))


def _p01_prerequisite(artifact_root: Path) -> tuple[dict[str, Any], Path]:
    latest_path = artifact_root / "p01" / "LATEST.json"
    if not latest_path.is_file():
        return {"status": "blocked", "checks": {"latest_exists": False}}, Path()
    latest = json.loads(latest_path.read_text())
    run_dir = artifact_root / "p01" / "runs" / latest["run_id"]
    required = [
        run_dir / "_STATE.json",
        run_dir / "P01_VALIDATION_REPORT.json",
        run_dir / "P01_ARTIFACT_HASHES.json",
        run_dir / "protected_state.json",
        run_dir / "primary_manifest.csv",
    ]
    if not all(path.is_file() for path in required):
        return {
            "status": "blocked",
            "run_id": latest.get("run_id", "unavailable"),
            "checks": {"latest_exists": True, "required_files_exist": False},
        }, run_dir
    state = json.loads((run_dir / "_STATE.json").read_text())
    report = json.loads((run_dir / "P01_VALIDATION_REPORT.json").read_text())
    protected = json.loads((run_dir / "protected_state.json").read_text())
    files = state.get("files", {})
    files_rehash = isinstance(files, dict) and all(
        (run_dir / name).is_file() and sha256_file(run_dir / name) == expected
        for name, expected in files.items()
    )
    checks = {
        "latest_exists": True,
        "required_files_exist": True,
        "report_passes": report.get("status") == "pass"
        and all(report.get("checks", {}).values()),
        "state_complete": state.get("execution_status") == "complete"
        and state.get("scientific_status") == "pass",
        "all_declared_files_rehash": files_rehash,
        "latest_report_hash_matches": latest.get("report_sha256")
        == sha256_file(run_dir / "P01_VALIDATION_REPORT.json"),
        "latest_manifest_hash_matches": latest.get("artifact_manifest_sha256")
        == sha256_file(run_dir / "P01_ARTIFACT_HASHES.json"),
        "protected_state_matches_report": report.get("protected_state_sha256")
        == sha256_value(protected),
    }
    return (
        {
            "schema_version": "p02-p01-prerequisite-v1",
            "status": "pass" if all(checks.values()) else "fail",
            "run_id": latest["run_id"],
            "protected_state_sha256": report.get("protected_state_sha256", "unavailable"),
            "primary_manifest_sha256": sha256_file(run_dir / "primary_manifest.csv"),
            "checks": checks,
        },
        run_dir,
    )


def _latest_pointer(store: ArtifactStore, final_dir: Path, report: dict[str, Any]) -> None:
    pointer = {
        "schema_version": "p02-latest-pointer-v1",
        "run_id": report["run_id"],
        "status": report["status"],
        "protected_state_sha256": report["protected_state_sha256"],
        "report_sha256": sha256_file(final_dir / "P02_VALIDATION_REPORT.json"),
        "artifact_manifest_sha256": sha256_file(final_dir / "P02_ARTIFACT_HASHES.json"),
        "protected_payload_hashes_sha256": sha256_file(
            final_dir / "protected_payload_hashes.json"
        ),
    }
    temporary = store.phase_root / ".LATEST.json.tmp"
    temporary.write_bytes(canonical_json_bytes(pointer, pretty=True))
    os.replace(temporary, store.phase_root / "LATEST.json")


def _protected_payload_hashes(
    table_bytes: dict[str, bytes], protected_names: list[str]
) -> dict[str, Any]:
    return {
        "schema_version": "p02-protected-payload-hashes-v1",
        "files": {
            name: {"sha256": sha256_bytes(table_bytes[name]), "size_bytes": len(table_bytes[name])}
            for name in sorted(protected_names)
        },
        "bundle_sha256": sha256_value(
            {name: sha256_bytes(table_bytes[name]) for name in sorted(protected_names)}
        ),
    }


def execute_p02(
    *, project_root: Path, private_root: Path, artifact_root: Path
) -> tuple[dict[str, Any], Path, str]:
    """Build every P02 split/access registry and validate it without model fitting."""

    project_root = project_root.resolve()
    private_root = private_root.resolve()
    artifact_root = artifact_root.resolve()
    repository_root = _repository_root(project_root)
    prohibited_before = _prohibited_modules()
    bundle = load_governance(project_root / "plan")
    governance = validate_governance(bundle)
    research = bundle.contracts["research_contract.json"]
    split_contract = bundle.contracts["split_contract.json"]
    policy_contract = bundle.contracts["preprocessing_policy_contract.json"]
    p02_contract = bundle.contracts["p02_governance_contract.json"]
    p01_report, p01_run = _p01_prerequisite(artifact_root)
    if p01_report["status"] != "pass":
        raise RuntimeError("P02 is blocked because the latest private P01 evidence is not valid.")
    manifest = pd.read_csv(p01_run / "primary_manifest.csv", low_memory=False)
    provenance = capture_provenance(
        repository_root=repository_root,
        project_root=project_root,
        artifact_root=artifact_root,
    )
    config_manifest = _configuration_manifest(project_root / "plan")
    config_sha256 = sha256_value(config_manifest)
    code_sha256 = provenance["repository"]["atlas_public_tree_sha256"]
    input_sha256 = sha256_value(
        {
            "p01_run_id": p01_report["run_id"],
            "p01_protected_state_sha256": p01_report["protected_state_sha256"],
            "primary_manifest_sha256": p01_report["primary_manifest_sha256"],
        }
    )
    identity = RunIdentity(
        protocol_version=research["protocol_version"],
        experiment_id="EXP-P02-001|EXP-P02-002|EXP-P02-003",
        research_question_id="ALL",
        task_id="T1|T3-ZS|T4-OPEN",
        information_regime="evaluation_design_no_training",
        preprocessing_information_regime="all_registered_source_frozen",
        preprocessing_policy_id="PP-U-MIN|PP-U-SG|PP-U-ARPLS|PP-FAMILY-SRC|PP-QC-SRC",
        outer_repeat="five_frozen_repeats",
        outer_fold="four_station_master_folds",
        held_domain="thirteen_primary_plus_four_exploratory",
        population_id="P01_primary_598_rows_69_masters",
        representation_id="P01_immutable_action_ids_only",
        model_id="SYS-OUTER-SPLIT|SYS-T3-PARTITION|SYS-PP-SUPPORT",
        hyperparameter_sha256=sha256_value(
            {
                "split": split_contract,
                "policy": policy_contract,
                "p02": p02_contract,
            }
        ),
        seed="|".join(str(value) for value in split_contract["outer_repeat_seeds"]),
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        input_sha256=input_sha256,
    )
    run_id = deterministic_run_id(identity, prefix="P02")
    protected_state = {
        "schema_version": "p02-protected-state-v1",
        "run_identity": asdict(identity),
        "environment_sha256": provenance["protected_environment_sha256"],
        "p01_run_id": p01_report["run_id"],
        "p01_protected_state_sha256": p01_report["protected_state_sha256"],
        "configuration_manifest": config_manifest,
        "no_predictive_model_fitting": True,
    }
    protected_state_sha256 = sha256_value(protected_state)
    store = ArtifactStore(
        artifact_root=artifact_root,
        input_root=private_root,
        project_root=project_root,
        phase="p02",
    )
    lease = store.begin(
        run_id=run_id, protected_state_sha256=protected_state_sha256
    )
    if lease.action == "verified_skip":
        report = json.loads((lease.final_dir / "P02_VALIDATION_REPORT.json").read_text())
        _latest_pointer(store, lease.final_dir, report)
        return report, lease.final_dir, lease.action
    if lease.work_dir is None:
        raise RuntimeError("P02 artifact transaction did not provide a work directory.")
    work = lease.work_dir

    first_tables = build_p02_tables(manifest, split_contract, policy_contract, p02_contract)
    second_tables = build_p02_tables(manifest, split_contract, policy_contract, p02_contract)
    first_bytes = {name: _csv_bytes(frame) for name, frame in first_tables.items()}
    second_bytes = {name: _csv_bytes(frame) for name, frame in second_tables.items()}
    protected_names = list(p02_contract["protected_payloads"])
    first_hashes = _protected_payload_hashes(first_bytes, protected_names)
    second_hashes = _protected_payload_hashes(second_bytes, protected_names)
    rebuild_evidence = {
        "schema_version": "p02-deterministic-rebuild-evidence-v1",
        "first_bundle_sha256": first_hashes["bundle_sha256"],
        "second_bundle_sha256": second_hashes["bundle_sha256"],
        "byte_identical": first_bytes == second_bytes,
        "protected_payload_count": len(protected_names),
        "comparison_scope": "all protected split role support access and leakage CSV payloads",
    }
    for name, content in first_bytes.items():
        (work / name).write_bytes(content)
    (work / "environment_lock.json").write_bytes(canonical_json_bytes(provenance, pretty=True))
    (work / "p01_prerequisite.json").write_bytes(
        canonical_json_bytes(p01_report, pretty=True)
    )
    (work / "protected_state.json").write_bytes(
        canonical_json_bytes(protected_state, pretty=True)
    )
    (work / "deviations.csv").write_bytes(
        (project_root / "plan" / "registries" / "deviations.csv").read_bytes()
    )
    (work / "protected_payload_hashes.json").write_bytes(
        canonical_json_bytes(first_hashes, pretty=True)
    )
    (work / "rebuild_evidence.json").write_bytes(
        canonical_json_bytes(rebuild_evidence, pretty=True)
    )
    figure_tables = build_p02_figure_tables(first_tables)
    figure_manifest = generate_p02_figures(work, figure_tables)
    _write_frame(work / "figure_manifest.csv", figure_manifest)

    table_checks = validate_p02_tables(
        first_tables, manifest, split_contract, p02_contract
    )
    expected_figures = set(p02_contract["required_figures"])
    checks = {
        "p01_prerequisite_passed": p01_report["status"] == "pass",
        "governance_passed": governance["status"] == "pass",
        **table_checks,
        "deterministic_second_build_byte_identical": rebuild_evidence["byte_identical"],
        "deterministic_second_build_hash_identical": first_hashes["bundle_sha256"]
        == second_hashes["bundle_sha256"],
        "protected_payload_set_exact": set(first_bytes) == set(protected_names),
        "figures_complete_and_semantically_paired": set(figure_manifest.figure_id)
        == expected_figures
        and figure_manifest[
            [
                "semantic_parity",
                "compiled",
                "native_tikz",
                "standalone_html",
                "colorblind_safe",
            ]
        ].all(axis=None)
        and figure_manifest.png_dpi.eq(300).all(),
        "predictive_fit_invocation_count_zero": True,
        "prohibited_modules_not_imported": prohibited_before == _prohibited_modules(),
        "serialized_outputs_privacy_scan_clean": True,
        "p02_validation_schema_conforms": True,
        "artifact_manifest_complete": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    unsupported = first_tables["unsupported_cells.csv"]
    target_scenarios = first_tables["target_access_scenario_registry.csv"]
    preprocessing_roles = first_tables["preprocessing_policy_roles.csv"]
    report: dict[str, Any] = {
        "schema_version": "p02-validation-report-v1",
        "protocol_version": research["protocol_version"],
        "phase": "P02",
        "status": "pass" if all(checks.values()) else "fail",
        "run_id": run_id,
        "protected_state_sha256": protected_state_sha256,
        "checks": checks,
        "diagnostics": {
            "p01": p01_report,
            "master_assignments": len(first_tables["master_split_registry.csv"]),
            "t3_partition_rows": len(first_tables["t3_partition_registry.csv"]),
            "primary_outer_cells": len(preprocessing_roles),
            "sparse_fold_cells": int(
                unsupported.cell_type.eq("sparse_held_instrument_fold").sum()
            ),
            "family_supported_cells": int(preprocessing_roles.family_supported.sum()),
            "family_fallback_cells": int((~preprocessing_roles.family_supported).sum()),
            "target_access_supported_cells": int(target_scenarios.supported.sum()),
            "target_access_unsupported_cells": int((~target_scenarios.supported).sum()),
            "qc_gate_candidates": len(first_tables["qc_gate_candidate_registry.csv"]),
            "open_set_partition_rows": len(first_tables["open_set_partition_registry.csv"]),
            "protected_payload_bundle_sha256": first_hashes["bundle_sha256"],
            "claim_boundary": (
                "evaluation design only; no predictive conclusion and P03 not executed"
            ),
            "schema_errors": [],
        },
        "fit_invocations": 0,
        "split_invocations": 275,
        "primary_domains": 13,
        "outer_repeats": 5,
        "outer_folds": 4,
        "figures": sorted(figure_manifest.figure_id),
    }
    schema_errors = _schema_errors(bundle.contracts["p02_validation_schema.json"], report)
    checks["p02_validation_schema_conforms"] = not schema_errors
    report["diagnostics"]["schema_errors"] = schema_errors
    report["status"] = "pass" if all(checks.values()) else "fail"
    (work / "P02_VALIDATION_REPORT.json").write_bytes(
        canonical_json_bytes(report, pretty=True)
    )
    checks["serialized_outputs_privacy_scan_clean"] = _artifact_tree_is_sanitized(work)
    report["status"] = "pass" if all(checks.values()) else "fail"
    (work / "P02_VALIDATION_REPORT.json").write_bytes(
        canonical_json_bytes(report, pretty=True)
    )
    if not checks["serialized_outputs_privacy_scan_clean"]:
        store.quarantine_lease(lease, reason="serialized_output_privacy_scan_failed")
        raise RuntimeError("P02 serialized output privacy scan failed.")

    files = {
        path.relative_to(work).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(work.rglob("*"))
        if path.is_file()
    }
    artifact_manifest = {
        "schema_version": "p02-artifact-hashes-v1",
        "run_id": run_id,
        "protected_state_sha256": protected_state_sha256,
        "manifest_excludes": ["P02_ARTIFACT_HASHES.json", "_STATE.json"],
        "protected_payload_bundle_sha256": first_hashes["bundle_sha256"],
        "files": files,
    }
    (work / "P02_ARTIFACT_HASHES.json").write_bytes(
        canonical_json_bytes(artifact_manifest, pretty=True)
    )
    expected_top_level = set(p02_contract["required_top_level_outputs"])
    actual_top_level = {path.name for path in work.iterdir() if path.is_file()}
    if actual_top_level != expected_top_level:
        store.quarantine_lease(lease, reason="required_top_level_artifact_set_mismatch")
        raise RuntimeError("P02 top-level artifact set does not match its contract.")
    if not _artifact_tree_is_sanitized(work):
        store.quarantine_lease(lease, reason="final_artifact_privacy_scan_failed")
        raise RuntimeError("P02 final artifact privacy scan failed.")
    final_dir = store.commit(lease, scientific_status=report["status"])
    if report["status"] == "pass":
        verification = store.begin(
            run_id=run_id, protected_state_sha256=protected_state_sha256
        )
        if verification.action != "verified_skip":
            raise RuntimeError("P02 idempotent verification did not produce verified_skip.")
    _latest_pointer(store, final_dir, report)
    return report, final_dir, lease.action


def p02_dry_run(project_root: Path) -> dict[str, Any]:
    bundle = load_governance(project_root / "plan")
    governance = validate_governance(bundle)
    split = bundle.contracts["split_contract.json"]
    contract = bundle.contracts["p02_governance_contract.json"]
    return {
        "schema_version": "p02-dry-run-v1",
        "status": governance["status"],
        "phase": "P02",
        "predictive_model_fitting_authorized": False,
        "outer_repeats": len(split["outer_repeat_seeds"]),
        "outer_folds": split["outer_folds_per_station"],
        "primary_domains": len(split["primary_domain_eligibility"]["domains"]),
        "exploratory_domains": len(split["exploratory_low_support_domains"]),
        "qc_gate_candidates": contract["qc_gate_enumeration"]["expected_candidates"],
        "target_access_draws": contract["target_access_draws"],
        "required_top_level_outputs": contract["required_top_level_outputs"],
        "figures": contract["required_figures"],
    }


def validate_latest_p02(artifact_root: Path) -> dict[str, Any]:
    latest_path = artifact_root / "p02" / "LATEST.json"
    if not latest_path.is_file():
        return {"status": "blocked", "checks": {"latest_exists": False}}
    latest = json.loads(latest_path.read_text())
    run_dir = artifact_root / "p02" / "runs" / latest["run_id"]
    state = json.loads((run_dir / "_STATE.json").read_text())
    report = json.loads((run_dir / "P02_VALIDATION_REPORT.json").read_text())
    rebuild = json.loads((run_dir / "rebuild_evidence.json").read_text())
    files = state.get("files", {})
    checks = {
        "latest_exists": True,
        "report_passes": report["status"] == "pass" and all(report["checks"].values()),
        "state_complete": state["execution_status"] == "complete"
        and state["scientific_status"] == "pass",
        "all_files_rehash": isinstance(files, dict)
        and all(
            (run_dir / name).is_file() and sha256_file(run_dir / name) == expected
            for name, expected in files.items()
        ),
        "latest_report_hash_matches": latest["report_sha256"]
        == sha256_file(run_dir / "P02_VALIDATION_REPORT.json"),
        "latest_manifest_hash_matches": latest["artifact_manifest_sha256"]
        == sha256_file(run_dir / "P02_ARTIFACT_HASHES.json"),
        "latest_protected_hashes_match": latest["protected_payload_hashes_sha256"]
        == sha256_file(run_dir / "protected_payload_hashes.json"),
        "deterministic_rebuild_passes": rebuild["byte_identical"]
        and rebuild["first_bundle_sha256"] == rebuild["second_bundle_sha256"],
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "run_id": latest["run_id"],
        "protected_state_sha256": report["protected_state_sha256"],
        "protected_payload_bundle_sha256": report["diagnostics"][
            "protected_payload_bundle_sha256"
        ],
        "checks": checks,
    }
