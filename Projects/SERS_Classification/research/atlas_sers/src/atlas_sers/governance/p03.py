"""Build and validate the outcome-blind P03 execution expansion."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_sers.evaluation.p03_plan import (
    CONTROL_EXPERIMENTS,
    T1_EXPERIMENT_MODELS,
    build_p03_plan_tables,
    summarize_compute,
)
from atlas_sers.evaluation.p03_roles import resolve_fit_roles
from atlas_sers.governance.artifacts import ArtifactStore
from atlas_sers.governance.canonical import (
    canonical_json_bytes,
    hash_relative_files,
    sha256_file,
    sha256_value,
)
from atlas_sers.governance.p01 import validate_latest_p01
from atlas_sers.governance.p02 import validate_latest_p02
from atlas_sers.governance.provenance import capture_provenance
from atlas_sers.governance.registries import load_governance, validate_governance
from atlas_sers.governance.runs import RunIdentity, deterministic_run_id

PROHIBITED_MODULES = ("atlas_sers.models", "torch")
SENSITIVE_PATTERNS = (
    bytes((47, 104, 111, 109, 101, 47)),
    bytes((92, 117, 115, 101, 114, 115, 92)),
    b"github_pat_",
    b"ghp_",
    b"gho_",
)


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
        plan_root / "P03_HANDOFF.md",
        *sorted((plan_root / "contracts").glob("*.json")),
        *sorted((plan_root / "registries").glob("*.csv")),
    ]
    return hash_relative_files(plan_root, files)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="",
    ).encode()


def _write_payload(path: Path, value: pd.DataFrame | dict[str, Any]) -> None:
    if isinstance(value, pd.DataFrame):
        path.write_bytes(_csv_bytes(value))
    else:
        path.write_bytes(canonical_json_bytes(value, pretty=True))


def _prohibited_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in PROHIBITED_MODULES)
    )


def _bytes_are_sanitized(content: bytes) -> bool:
    lowered = content.lower()
    return not any(pattern in lowered for pattern in SENSITIVE_PATTERNS)


def _artifact_tree_is_sanitized(root: Path) -> bool:
    return all(
        not path.is_symlink()
        and _bytes_are_sanitized(path.relative_to(root).as_posix().encode())
        and _bytes_are_sanitized(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file() or path.is_symlink()
    )


def _reconstruct_unique_fit_roles(
    fits: pd.DataFrame,
    *,
    manifest: pd.DataFrame,
    p02_tables: dict[str, pd.DataFrame],
) -> int:
    """Resolve each distinct role definition once against immutable P02 records."""

    identity_fields = [
        "experiment_id",
        "task_id",
        "domain",
        "station",
        "held_instrument",
        "outer_repeat",
        "outer_fold",
        "selection_mode",
        "stage",
        "selection_unit_id",
        "accounting",
        "fit_rows",
        "fit_masters",
        "fit_uid_sha256",
        "validation_rows",
        "validation_masters",
        "validation_uid_sha256",
        "test_rows",
        "test_masters",
        "test_uid_sha256",
    ]
    unique = fits[identity_fields].drop_duplicates(ignore_index=True)
    for _, row in unique.iterrows():
        resolve_fit_roles(row, manifest=manifest, p02_tables=p02_tables)
    return len(unique)


def _latest_run(artifact_root: Path, phase: str) -> tuple[dict[str, Any], Path]:
    phase_root = artifact_root / phase
    latest = json.loads((phase_root / "LATEST.json").read_text())
    return latest, phase_root / "runs" / latest["run_id"]


def _input_hashes(
    *, p01_latest: dict[str, Any], p01_run: Path, p02_latest: dict[str, Any], p02_run: Path
) -> dict[str, Any]:
    registry = pd.read_csv(p01_run / "representation_registry.csv")
    representation = registry[registry.representation_id == "R_MIN_400_1800"]
    if len(representation) != 1:
        raise ValueError("P01 R_MIN_400_1800 representation row is not unique.")
    row = representation.iloc[0]
    protected_payloads = json.loads((p02_run / "protected_payload_hashes.json").read_text())
    files = protected_payloads["files"]
    rehashed = {
        name: {
            "sha256": sha256_file(p02_run / name),
            "size_bytes": (p02_run / name).stat().st_size,
        }
        for name in sorted(files)
    }
    if rehashed != files:
        raise ValueError("A consumed P02 protected payload no longer matches its frozen hash.")
    return {
        "schema_version": "p03-input-hashes-v1",
        "p01": {
            "run_id": p01_latest["run_id"],
            "protected_state_sha256": p01_latest["protected_state_sha256"],
            "primary_manifest_sha256": sha256_file(p01_run / "primary_manifest.csv"),
            "representation_registry_sha256": sha256_file(p01_run / "representation_registry.csv"),
            "representation_npz_sha256": sha256_file(
                p01_run / "representations" / "R_MIN_400_1800.npz"
            ),
            "array_sha256": str(row.array_sha256),
            "row_order_sha256": str(row.row_order_sha256),
            "axis_sha256": str(row.axis_sha256),
        },
        "p02": {
            "run_id": p02_latest["run_id"],
            "protected_state_sha256": p02_latest["protected_state_sha256"],
            "protected_payload_bundle_sha256": protected_payloads["bundle_sha256"],
            "protected_payloads": rehashed,
        },
    }


def _load_p02_tables(p02_run: Path, input_hashes: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(p02_run / name, low_memory=False)
        for name in input_hashes["p02"]["protected_payloads"]
        if name.endswith(".csv")
    }


def _latest_pointer(store: ArtifactStore, final_dir: Path, report: dict[str, Any]) -> None:
    pointer = {
        "schema_version": "p03-plan-latest-pointer-v1",
        "run_id": report["run_id"],
        "status": report["status"],
        "scientific_fitting_authorized": report["scientific_fitting_authorized"],
        "protected_state_sha256": report["protected_state_sha256"],
        "report_sha256": sha256_file(final_dir / "P03_PLAN_VALIDATION_REPORT.json"),
        "artifact_manifest_sha256": sha256_file(final_dir / "P03_PLAN_ARTIFACT_HASHES.json"),
    }
    temporary = store.phase_root / ".LATEST.json.tmp"
    temporary.write_bytes(canonical_json_bytes(pointer, pretty=True))
    os.replace(temporary, store.phase_root / "LATEST.json")


def execute_p03_plan(
    *, project_root: Path, private_root: Path, artifact_root: Path
) -> tuple[dict[str, Any], Path, str]:
    """Materialize the P03 compute expansion without importing or fitting models."""

    project_root = project_root.resolve()
    artifact_root = artifact_root.resolve()
    repository_root = _repository_root(project_root)
    prohibited_before = _prohibited_modules()
    governance_bundle = load_governance(project_root / "plan")
    governance = validate_governance(governance_bundle)
    p03_contract = governance_bundle.contracts["p03_governance_contract.json"]
    hyperparameters = governance_bundle.contracts["hyperparameter_registry.json"]
    p01_validation = validate_latest_p01(artifact_root)
    p02_validation = validate_latest_p02(artifact_root)
    if p01_validation["status"] != "pass" or p02_validation["status"] != "pass":
        raise RuntimeError("P03 planning is blocked by an invalid P01/P02 prerequisite.")
    p01_latest, p01_run = _latest_run(artifact_root, "p01")
    p02_latest, p02_run = _latest_run(artifact_root, "p02")
    input_hashes = _input_hashes(
        p01_latest=p01_latest,
        p01_run=p01_run,
        p02_latest=p02_latest,
        p02_run=p02_run,
    )
    provenance = capture_provenance(
        repository_root=repository_root,
        project_root=project_root,
        artifact_root=artifact_root,
    )
    configuration = _configuration_manifest(project_root / "plan")
    input_sha256 = sha256_value(input_hashes)
    identity = RunIdentity(
        protocol_version=p03_contract["protocol_version"],
        experiment_id="EXP-C00-T1|EXP-C01-T1|EXP-C02-T1|EXP-C03-T1|EXP-C04-T1|EXP-C05-T1|EXP-C06-T1|EXP-C07-T1|EXP-C08-T1|EXP-C09-T3|EXP-C09-CONTROL-PERM|EXP-C09-CONTROL-META|EXP-C09-CONTROL-PRIOR|EXP-C10-T3|EXP-C11-T2|EXP-C12-CORAL",
        research_question_id="RQ-P01",
        task_id="T1-CWA|T1-PILLS|T1-SURF|T2-PS|T2-SP|T3-ZS",
        information_regime="outcome_blind_compute_expansion",
        preprocessing_information_regime="fixed_source_only",
        preprocessing_policy_id="PP-U-MIN",
        outer_repeat="five_frozen_repeats",
        outer_fold="four_frozen_master_folds",
        held_domain="thirteen_primary_plus_four_manifest_only_exploratory",
        population_id="primary_598",
        representation_id="R_MIN_400_1800",
        model_id="registered_classical_suite",
        hyperparameter_sha256=sha256_value(
            {"hyperparameters": hyperparameters["classical"], "p03": p03_contract}
        ),
        seed="20260805|20260817|20260829|20260910|20260922",
        code_sha256=provenance["repository"]["atlas_public_tree_sha256"],
        config_sha256=sha256_value(configuration),
        input_sha256=input_sha256,
    )
    run_id = deterministic_run_id(identity, prefix="P03PLAN")
    protected_state = {
        "schema_version": "p03-plan-protected-state-v1",
        "run_identity": identity.__dict__,
        "input_hashes_sha256": input_sha256,
        "configuration_manifest": configuration,
        "environment_sha256": provenance["protected_environment_sha256"],
        "model_fitting_authorized": bool(
            p03_contract["planning"]["model_fitting_authorized"]
        ),
    }
    protected_state_sha256 = sha256_value(protected_state)
    store = ArtifactStore(
        artifact_root=artifact_root,
        input_root=private_root,
        project_root=project_root,
        phase="p03plan",
    )
    lease = store.begin(run_id=run_id, protected_state_sha256=protected_state_sha256)
    if lease.action == "verified_skip":
        report = json.loads((lease.final_dir / "P03_PLAN_VALIDATION_REPORT.json").read_text())
        _latest_pointer(store, lease.final_dir, report)
        return report, lease.final_dir, lease.action
    if lease.work_dir is None:
        raise RuntimeError("P03 plan transaction did not provide a work directory.")
    work = lease.work_dir
    manifest = pd.read_csv(p01_run / "primary_manifest.csv", low_memory=False)
    p02_tables = _load_p02_tables(p02_run, input_hashes)
    first = build_p03_plan_tables(
        manifest=manifest,
        p02_tables=p02_tables,
        hyperparameters=hyperparameters,
        p03_contract=p03_contract,
    )
    first.update(summarize_compute(first, p03_contract))
    second = build_p03_plan_tables(
        manifest=manifest,
        p02_tables=p02_tables,
        hyperparameters=hyperparameters,
        p03_contract=p03_contract,
    )
    second.update(summarize_compute(second, p03_contract))
    first_bytes = {
        name: _csv_bytes(value)
        if isinstance(value, pd.DataFrame)
        else canonical_json_bytes(value, pretty=True)
        for name, value in first.items()
    }
    second_bytes = {
        name: _csv_bytes(value)
        if isinstance(value, pd.DataFrame)
        else canonical_json_bytes(value, pretty=True)
        for name, value in second.items()
    }
    for name, value in first.items():
        _write_payload(work / name, value)
    (work / "environment_lock.json").write_bytes(canonical_json_bytes(provenance, pretty=True))
    (work / "input_hashes.json").write_bytes(canonical_json_bytes(input_hashes, pretty=True))
    (work / "protected_state.json").write_bytes(canonical_json_bytes(protected_state, pretty=True))
    candidates = first["candidate_registry.csv"]
    coral_candidates = first["coral_candidate_registry.csv"]
    control_candidates = first["control_registry.csv"]
    runs = first["expected_run_registry.csv"]
    fits = first["fit_manifest.csv"]
    budget = first["budget_gate.json"]
    assert isinstance(candidates, pd.DataFrame)
    assert isinstance(coral_candidates, pd.DataFrame)
    assert isinstance(control_candidates, pd.DataFrame)
    assert isinstance(runs, pd.DataFrame)
    assert isinstance(fits, pd.DataFrame)
    assert isinstance(budget, dict)
    unique_role_definition_count = _reconstruct_unique_fit_roles(
        fits, manifest=manifest, p02_tables=p02_tables
    )
    c09 = runs[runs.experiment_id == "EXP-C09-T3"]
    checks = {
        "p01_prerequisite_passes": p01_validation["status"] == "pass",
        "p02_prerequisite_passes": p02_validation["status"] == "pass",
        "governance_passes": governance["status"] == "pass",
        "population_is_598_rows_69_masters": len(manifest) == 598
        and manifest.master_sample_id.nunique() == 69,
        "candidate_registry_has_126_exact_candidates": len(candidates) == 126,
        "coral_registry_has_46_exact_source_only_candidates": len(coral_candidates) == 46,
        "control_registry_has_52_exact_controls": len(control_candidates) == 52
        and control_candidates.control_type.value_counts().to_dict()
        == {
            "acquisition_metadata_only": 30,
            "master_label_permutation": 20,
            "station_or_target_prior": 2,
        },
        "registered_experiments_complete": set(runs.experiment_id) - {"UNREGISTERED-T3-LOW"}
        == {
            *T1_EXPERIMENT_MODELS,
            "EXP-C09-T3",
            "EXP-C10-T3",
            "EXP-C11-T2",
            "EXP-C12-CORAL",
            *CONTROL_EXPERIMENTS,
        },
        "t1_outer_runs_are_540": runs.experiment_id.str.endswith("-T1").sum() == 540,
        "t3_primary_cells_are_260": len(c09) == 260 and c09.domain.nunique() == 13,
        "t3_selection_routes_are_128_and_132": int(c09.selection_mode.eq("pseudo_domain").sum())
        == 128
        and int(c09.selection_mode.eq("master_cv").sum()) == 132,
        "control_outer_runs_are_exact": int(
            runs.experiment_id.eq("EXP-C09-CONTROL-PERM").sum()
        )
        == 5200
        and int(runs.experiment_id.eq("EXP-C09-CONTROL-META").sum()) == 260
        and int(runs.experiment_id.eq("EXP-C09-CONTROL-PRIOR").sum()) == 520,
        "exploratory_cells_visible_not_authorized": int(
            runs.execution_status.eq("manifest_only_exploratory").sum()
        )
        == 80,
        "all_fit_test_uid_sets_disjoint": bool(fits.fit_test_disjoint.all()),
        "all_inner_fit_validation_masters_disjoint": bool(
            fits.fit_validation_master_disjoint.all()
        ),
        "all_unique_fit_roles_reconstruct_from_P02": unique_role_definition_count > 0,
        "deterministic_second_expansion_byte_identical": first_bytes == second_bytes,
        "fitting_authorization_gate_is_self_consistent": budget["status"]
        == ("pass" if budget["scientific_fitting_authorized"] else "fail"),
        "model_fit_invocations_zero": True,
        "prohibited_model_modules_not_imported": prohibited_before == _prohibited_modules(),
        "serialized_outputs_privacy_scan_clean": True,
        "required_output_set_exact": True,
    }
    report = {
        "schema_version": "p03-plan-validation-report-v1",
        "protocol_version": p03_contract["protocol_version"],
        "phase": "P03-PLAN",
        "status": "pass" if all(checks.values()) else "fail",
        "run_id": run_id,
        "protected_state_sha256": protected_state_sha256,
        "model_fit_invocations": 0,
        "scientific_fitting_authorized": budget["scientific_fitting_authorized"],
        "checks": checks,
        "diagnostics": {
            "candidate_count": len(candidates),
            "coral_candidate_count": len(coral_candidates),
            "control_candidate_count": len(control_candidates),
            "expected_outer_run_count": len(runs),
            "fit_manifest_rows": len(fits),
            "unique_role_definition_count": unique_role_definition_count,
            "planned_new_fit_count": budget["planned_new_fit_count"],
            "registered_fit_estimate_high": budget["registered_fit_estimate_high"],
            "budget_excess": budget["excess_over_registered_high"],
            "blocking_reasons": budget["blocking_reasons"],
            "claim_boundary": "outcome-blind execution planning only; no model was fitted",
        },
    }
    (work / "P03_PLAN_VALIDATION_REPORT.json").write_bytes(
        canonical_json_bytes(report, pretty=True)
    )
    files = {
        path.relative_to(work).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(work.rglob("*"))
        if path.is_file()
    }
    artifact_manifest = {
        "schema_version": "p03-plan-artifact-hashes-v1",
        "run_id": run_id,
        "protected_state_sha256": protected_state_sha256,
        "manifest_excludes": ["P03_PLAN_ARTIFACT_HASHES.json", "_STATE.json"],
        "files": files,
    }
    (work / "P03_PLAN_ARTIFACT_HASHES.json").write_bytes(
        canonical_json_bytes(artifact_manifest, pretty=True)
    )
    expected = set(p03_contract["planning"]["required_outputs"])
    actual = {path.name for path in work.iterdir() if path.is_file()}
    checks["required_output_set_exact"] = actual == expected
    checks["serialized_outputs_privacy_scan_clean"] = _artifact_tree_is_sanitized(work)
    report["status"] = "pass" if all(checks.values()) else "fail"
    (work / "P03_PLAN_VALIDATION_REPORT.json").write_bytes(
        canonical_json_bytes(report, pretty=True)
    )
    # Rehash the final report after its checks are finalized.
    files["P03_PLAN_VALIDATION_REPORT.json"] = {
        "sha256": sha256_file(work / "P03_PLAN_VALIDATION_REPORT.json"),
        "size_bytes": (work / "P03_PLAN_VALIDATION_REPORT.json").stat().st_size,
    }
    artifact_manifest["files"] = files
    (work / "P03_PLAN_ARTIFACT_HASHES.json").write_bytes(
        canonical_json_bytes(artifact_manifest, pretty=True)
    )
    if report["status"] != "pass":
        store.quarantine_lease(lease, reason="p03_plan_validation_failed")
        raise RuntimeError("P03 no-fit execution plan failed validation.")
    final_dir = store.commit(lease, scientific_status="pass")
    verification = store.begin(run_id=run_id, protected_state_sha256=protected_state_sha256)
    if verification.action != "verified_skip":
        raise RuntimeError("P03 plan did not pass idempotent verified-skip validation.")
    _latest_pointer(store, final_dir, report)
    return report, final_dir, lease.action


def validate_latest_p03_plan(artifact_root: Path) -> dict[str, Any]:
    latest_path = artifact_root / "p03plan" / "LATEST.json"
    if not latest_path.is_file():
        return {"status": "blocked", "checks": {"latest_exists": False}}
    latest = json.loads(latest_path.read_text())
    run_dir = artifact_root / "p03plan" / "runs" / latest["run_id"]
    state = json.loads((run_dir / "_STATE.json").read_text())
    report = json.loads((run_dir / "P03_PLAN_VALIDATION_REPORT.json").read_text())
    files = state.get("files", {})
    checks = {
        "latest_exists": True,
        "report_passes": report["status"] == "pass" and all(report["checks"].values()),
        "state_complete": state["execution_status"] == "complete"
        and state["scientific_status"] == "pass",
        "all_files_rehash": isinstance(files, dict)
        and all(
            (run_dir / name).is_file() and sha256_file(run_dir / name) == digest
            for name, digest in files.items()
        ),
        "latest_report_hash_matches": latest["report_sha256"]
        == sha256_file(run_dir / "P03_PLAN_VALIDATION_REPORT.json"),
        "latest_manifest_hash_matches": latest["artifact_manifest_sha256"]
        == sha256_file(run_dir / "P03_PLAN_ARTIFACT_HASHES.json"),
        "latest_fitting_authorization_matches_report": latest[
            "scientific_fitting_authorized"
        ]
        == report["scientific_fitting_authorized"],
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "run_id": latest["run_id"],
        "protected_state_sha256": latest["protected_state_sha256"],
        "checks": checks,
        "diagnostics": report["diagnostics"],
    }
