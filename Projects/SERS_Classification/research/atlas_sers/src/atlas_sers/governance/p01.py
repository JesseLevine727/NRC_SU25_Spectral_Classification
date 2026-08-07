"""Execute the private, source-reversible ATLAS P01 data and representation freeze."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

from atlas_sers.data.manifests import build_population_manifests
from atlas_sers.data.native import build_native_registry, native_source_audit
from atlas_sers.exploration.structure import analyze_structure
from atlas_sers.governance.artifacts import ArtifactStore
from atlas_sers.governance.canonical import (
    canonical_json_bytes,
    deterministic_npz_bytes,
    hash_relative_files,
    sha256_file,
    sha256_value,
)
from atlas_sers.governance.inputs import resolve_private_path, verify_authoritative_inputs
from atlas_sers.governance.provenance import capture_provenance
from atlas_sers.governance.registries import load_governance, validate_governance
from atlas_sers.governance.runs import RunIdentity, deterministic_run_id
from atlas_sers.paths import validate_private_roots
from atlas_sers.preprocessing.representations import (
    aggregate_preservation,
    build_representations,
    preservation_metrics,
    representation_invariants,
    representation_registry,
    reproduce_frozen_candidates,
)
from atlas_sers.visualization.p01_figures import build_figure_tables, generate_p01_figures

RESTRICTED_SOURCE_PATTERN = re.compile(rb"(?<![a-z])" + bytes((110, 97, 116, 111)) + rb"(?![a-z])")
SENSITIVE_PATTERNS = [
    bytes((47, 104, 111, 109, 101, 47)),
    bytes((92, 117, 115, 101, 114, 115, 92)),
    b"github_pat_",
    b"ghp_",
    b"gho_",
]
OPAQUE_GENERATED_SUFFIXES = {".npz", ".pdf"}
PROHIBITED_MODULES = ("atlas_sers.models", "atlas_sers.splits", "torch")


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


def _phase_status(bundle: Any, phase_id: str) -> str:
    return next(
        row["execution_status"]
        for row in bundle.rows("phase_registry.csv")
        if row["phase_id"] == phase_id
    )


def _schema_errors(schema: dict[str, Any], instance: dict[str, Any]) -> list[str]:
    return sorted(error.message for error in Draft202012Validator(schema).iter_errors(instance))


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="",
    ).encode()


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.write_bytes(_csv_bytes(frame))


def _prohibited_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in PROHIBITED_MODULES)
    )


def _bytes_are_sanitized(content: bytes) -> bool:
    lowered = content.lower()
    return not RESTRICTED_SOURCE_PATTERN.search(lowered) and not any(
        pattern in lowered for pattern in SENSITIVE_PATTERNS
    )


def _npz_is_sanitized(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                if not _bytes_are_sanitized(name.encode()):
                    return False
                array = archive[name]
                if array.dtype.kind in {"S", "U"}:
                    serialized = "\n".join(array.astype(str).ravel()).encode()
                    if not _bytes_are_sanitized(serialized):
                        return False
    except (OSError, ValueError, TypeError):
        return False
    return True


def _artifact_tree_is_sanitized(root: Path) -> bool:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            return False
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode()
        if not _bytes_are_sanitized(relative):
            return False
        if path.suffix.lower() == ".npz":
            if not _npz_is_sanitized(path):
                return False
        elif path.suffix.lower() not in OPAQUE_GENERATED_SUFFIXES and not _bytes_are_sanitized(
            path.read_bytes()
        ):
            return False
    return True


def _p00_prerequisite(artifact_root: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    latest_path = artifact_root / "p00" / "LATEST.json"
    if not latest_path.is_file():
        return {"status": "blocked", "checks": {"latest_exists": False}}
    latest = json.loads(latest_path.read_text())
    run_dir = artifact_root / "p00" / "runs" / latest["run_id"]
    report_path = run_dir / "P00_VALIDATION_REPORT.json"
    protected_path = run_dir / "protected_state.json"
    environment_path = run_dir / "environment_lock.json"
    state_path = run_dir / "_STATE.json"
    if not all(
        path.is_file() for path in (report_path, protected_path, environment_path, state_path)
    ):
        return {"status": "blocked", "checks": {"latest_exists": True, "files_exist": False}}
    report = json.loads(report_path.read_text())
    protected = json.loads(protected_path.read_text())
    environment = json.loads(environment_path.read_text())
    state = json.loads(state_path.read_text())
    files = state.get("files", {})
    files_valid = isinstance(files, dict) and all(
        (run_dir / name).is_file() and sha256_file(run_dir / name) == expected
        for name, expected in files.items()
    )
    checks = {
        "latest_exists": True,
        "files_exist": True,
        "report_passes": report.get("status") == "pass" and all(report.get("checks", {}).values()),
        "state_complete": state.get("execution_status") == "complete"
        and state.get("scientific_status") == "pass",
        "state_files_rehash": files_valid,
        "code_hash_current": protected["run_identity"]["code_sha256"]
        == provenance["repository"]["atlas_public_tree_sha256"],
        "repository_commit_current": environment["repository"]["commit"]
        == provenance["repository"]["commit"],
        "atlas_dirty_state_current": environment["repository"]["atlas_dirty_state_sha256"]
        == provenance["repository"]["atlas_dirty_state_sha256"],
        "dependency_lock_current": environment["dependency_lock_sha256"]
        == provenance["dependency_lock_sha256"],
        "latest_report_hash_current": latest.get("report_sha256") == sha256_file(report_path),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "run_id": report.get("run_id", "unavailable"),
        "protected_state_sha256": report.get("protected_state_sha256", "unavailable"),
    }


def _load_inputs(
    research: dict[str, Any], private_root: Path
) -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Path]
]:
    paths: dict[str, Path] = {}
    for declaration in research["authoritative_inputs"]:
        logical = str(declaration["path"])
        resolved = resolve_private_path(private_root, logical)
        if logical.endswith("tier_unique_attributable_sers.csv"):
            paths["manifest"] = resolved
        elif logical.endswith("unique_attributable_common_support_raw.npz"):
            paths["raw"] = resolved
        elif logical.endswith("preprocessing_candidates.npz"):
            paths["candidates"] = resolved
        elif logical.endswith("FINAL_VALIDATION_REPORT.json"):
            paths["validation"] = resolved
    if set(paths) != {"manifest", "raw", "candidates", "validation"}:
        raise ValueError("P01 could not resolve the four authoritative input roles.")
    manifest = pd.read_csv(paths["manifest"], low_memory=False)
    with np.load(paths["raw"], allow_pickle=False) as archive:
        axis = archive["axis_cm1"].astype(float)
        raw = archive["intensity"].astype(float)
        source_uids = archive["observation_uid"].astype(str)
    with np.load(paths["candidates"], allow_pickle=False) as archive:
        candidates = {name: archive[name].copy() for name in archive.files}
    return manifest, axis, raw, source_uids, candidates, paths


def _attach_source_ids(populations: dict[str, pd.DataFrame], native_registry: pd.DataFrame) -> None:
    mapping = native_registry.set_index("observation_uid")["source_logical_id"]
    for frame in populations.values():
        frame.insert(2, "source_logical_id", frame.observation_uid.map(mapping))
        if frame.source_logical_id.isna().any():
            raise ValueError("A population row has no native source logical ID.")


def _native_interpolation_audit(
    native_spectra: list[tuple[np.ndarray, np.ndarray]],
    axis: np.ndarray,
    raw: np.ndarray,
    native_registry: pd.DataFrame,
) -> dict[str, Any]:
    failures = 0
    maximum_error = 0.0
    effective_support_failures = 0
    for index, (native_axis, native_intensity) in enumerate(native_spectra):
        effective_support_failures += not (
            float(native_registry.iloc[index].effective_axis_min_cm1) <= float(axis[0])
            and float(native_registry.iloc[index].effective_axis_max_cm1) >= float(axis[-1])
        )
        reconstructed = np.interp(axis, native_axis, native_intensity).astype(np.float32)
        difference = np.abs(reconstructed - raw[index].astype(np.float32))
        error = float(difference.max())
        maximum_error = max(maximum_error, error)
        failures += error != 0.0
    return {
        "rows_checked": len(native_spectra),
        "interpolation_failures": int(failures),
        "maximum_absolute_error": maximum_error,
        "effective_support_failures": int(effective_support_failures),
    }


def _audit_indices(manifest: pd.DataFrame, native_registry: pd.DataFrame, seed: int) -> list[int]:
    selected: set[int] = set()
    random = np.random.default_rng(seed)
    selected.update(random.choice(len(manifest), size=min(12, len(manifest)), replace=False))
    for _, group in native_registry.reset_index().groupby("instrument", sort=True):
        selected.add(int(group.effective_axis_min_cm1.idxmax()))
        selected.add(int(group.effective_axis_max_cm1.idxmin()))
    for _, group in manifest.reset_index().groupby(
        ["instrument", "tier_notes_clear_sers"], sort=True
    ):
        selected.add(int(group["index"].iloc[0]))
    return sorted(selected)


def _reversibility_audit(
    manifest: pd.DataFrame,
    native_registry: pd.DataFrame,
    axis: np.ndarray,
    raw: np.ndarray,
    native_spectra: list[tuple[np.ndarray, np.ndarray]],
    bundles: dict[str, Any],
    contract: dict[str, Any],
) -> pd.DataFrame:
    indices = _audit_indices(manifest, native_registry, int(contract["analysis_seeds"][0]))
    if not indices:
        raise ValueError("Reversibility audit selected no rows.")
    rebuilt = build_representations(axis, raw[indices], contract)
    rows: list[dict[str, Any]] = []
    for local_index, source_index in enumerate(indices):
        native_axis, native_intensity = native_spectra[source_index]
        interpolation_error = float(
            np.max(
                np.abs(
                    np.interp(axis, native_axis, native_intensity).astype(np.float32)
                    - raw[source_index].astype(np.float32)
                )
            )
        )
        for identifier, rebuilt_bundle in rebuilt.items():
            error = float(
                np.max(
                    np.abs(
                        rebuilt_bundle.intensity[local_index]
                        - bundles[identifier].intensity[source_index]
                    )
                )
            )
            rows.append(
                {
                    "observation_uid": manifest.iloc[source_index].observation_uid,
                    "instrument": manifest.iloc[source_index].instrument,
                    "tier_notes_clear": bool(manifest.iloc[source_index].tier_notes_clear_sers),
                    "representation_id": identifier,
                    "native_interpolation_max_absolute_error": interpolation_error,
                    "representation_rebuild_max_absolute_error": error,
                    "status": "pass" if interpolation_error == 0 and error <= 1e-6 else "fail",
                }
            )
    return pd.DataFrame(rows)


def _latest_pointer(store: ArtifactStore, final_dir: Path, report: dict[str, Any]) -> None:
    pointer = {
        "schema_version": "p01-latest-pointer-v1",
        "run_id": report["run_id"],
        "status": report["status"],
        "protected_state_sha256": report["protected_state_sha256"],
        "report_sha256": sha256_file(final_dir / "P01_VALIDATION_REPORT.json"),
        "artifact_manifest_sha256": sha256_file(final_dir / "P01_ARTIFACT_HASHES.json"),
    }
    temporary = store.phase_root / ".LATEST.json.tmp"
    temporary.write_bytes(canonical_json_bytes(pointer, pretty=True))
    os.replace(temporary, store.phase_root / "LATEST.json")


def execute_p01(
    *,
    project_root: Path,
    private_root: Path,
    native_root: Path,
    artifact_root: Path,
) -> tuple[dict[str, Any], Path, str]:
    """Build and validate all private P01 data, representations, analyses, and figures."""

    project_root = project_root.resolve()
    private_root = private_root.resolve()
    native_root = native_root.resolve()
    artifact_root = artifact_root.resolve()
    repository_root = _repository_root(project_root)
    validate_private_roots(
        input_root=native_root,
        output_root=artifact_root,
        public_project_root=project_root,
    )
    prohibited_before = _prohibited_modules()
    bundle = load_governance(project_root / "plan")
    governance_report = validate_governance(bundle)
    research = bundle.contracts["research_contract.json"]
    contract = bundle.contracts["p01_governance_contract.json"]
    input_report = verify_authoritative_inputs(
        research,
        private_root=private_root,
        repository_root=repository_root,
    )
    provenance = capture_provenance(
        repository_root=repository_root,
        project_root=project_root,
        artifact_root=artifact_root,
    )
    p00_report = _p00_prerequisite(artifact_root, provenance)
    authoritative, axis, raw, source_uids, frozen_candidates, input_paths = _load_inputs(
        research, private_root
    )
    populations, population_report = build_population_manifests(
        authoritative, contract["population_contract"]
    )
    primary = populations["primary_manifest.csv"]
    if not np.array_equal(source_uids, primary.source_observation_uid.astype(str).to_numpy()):
        raise ValueError("Authoritative raw-array UID order differs from the primary manifest.")
    native_registry, native_spectra, native_report = build_native_registry(primary, native_root)
    _attach_source_ids(populations, native_registry)
    interpolation_report = _native_interpolation_audit(native_spectra, axis, raw, native_registry)
    native_bundle_sha256 = sha256_value(
        native_registry[
            [
                "source_logical_id",
                "numeric_file_sha256",
                "axis_sha256",
                "intensity_sha256",
            ]
        ].to_dict(orient="records")
    )
    input_sha256 = sha256_value(
        {
            "derived": input_report["input_bundle_sha256"],
            "native": native_bundle_sha256,
        }
    )
    config_manifest = _configuration_manifest(project_root / "plan")
    config_sha256 = sha256_value(config_manifest)
    code_sha256 = provenance["repository"]["atlas_public_tree_sha256"]
    identity = RunIdentity(
        protocol_version=research["protocol_version"],
        experiment_id="EXP-P01-001",
        research_question_id="ALL",
        task_id="ALL",
        information_regime="none",
        preprocessing_information_regime="not_applicable",
        preprocessing_policy_id="ALL",
        outer_repeat="P02_unresolved",
        outer_fold="P02_unresolved",
        held_domain="not_applicable",
        population_id="primary_plus_registered_sensitivities",
        representation_id="all_registered",
        model_id="SYS-REP-BUILDER",
        hyperparameter_sha256=sha256_value(contract),
        seed="|".join(str(value) for value in contract["analysis_seeds"]),
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        input_sha256=input_sha256,
    )
    run_id = deterministic_run_id(identity, prefix="P01")
    protected_state = {
        "schema_version": "p01-protected-state-v1",
        "run_identity": asdict(identity),
        "environment_sha256": provenance["protected_environment_sha256"],
        "p00_run_id": p00_report.get("run_id", "unavailable"),
        "p00_protected_state_sha256": p00_report.get("protected_state_sha256", "unavailable"),
        "configuration_manifest": config_manifest,
        "native_source_bundle_sha256": native_bundle_sha256,
    }
    protected_hash = sha256_value(protected_state)
    store = ArtifactStore(
        artifact_root=artifact_root,
        input_root=private_root,
        project_root=project_root,
        phase="p01",
    )
    lease = store.begin(run_id=run_id, protected_state_sha256=protected_hash)
    if lease.action == "verified_skip":
        report = json.loads((lease.final_dir / "P01_VALIDATION_REPORT.json").read_text())
        _latest_pointer(store, lease.final_dir, report)
        return report, lease.final_dir, lease.action
    if lease.work_dir is None:
        raise RuntimeError("P01 artifact transaction did not provide a work directory.")
    work = lease.work_dir

    source_audit = native_source_audit(native_root)
    recording_path = private_root / "tables" / "recordings_manifest_fresh.csv"
    if not recording_path.is_file():
        store.quarantine_lease(lease, reason="supplementary_recording_manifest_missing")
        raise RuntimeError("P01 supplementary recording manifest is missing.")
    recording_manifest = pd.read_csv(recording_path, low_memory=False)
    source_audit["derived_workspace"] = {
        "authoritative_manifest_columns": len(authoritative.columns),
        "recording_manifest_rows": len(recording_manifest),
        "recording_manifest_sha256": sha256_file(recording_path),
        "candidate_arrays": sorted(frozen_candidates),
    }
    source_audit["native_selection"] = native_report
    source_audit["native_interpolation"] = interpolation_report

    (work / "environment_lock.json").write_bytes(canonical_json_bytes(provenance, pretty=True))
    (work / "input_verification.json").write_bytes(canonical_json_bytes(input_report, pretty=True))
    (work / "protected_state.json").write_bytes(canonical_json_bytes(protected_state, pretty=True))
    (work / "deviations.csv").write_bytes(
        (project_root / "plan" / "registries" / "deviations.csv").read_bytes()
    )
    (work / "private_source_audit.json").write_bytes(
        canonical_json_bytes(source_audit, pretty=True)
    )
    for name, frame in populations.items():
        _write_frame(work / name, frame)
    _write_frame(work / "native_source_registry.csv", native_registry)

    representation_bundles = build_representations(axis, raw, contract)
    representation_dir = work / "representations"
    representation_dir.mkdir()
    for identifier, representation in representation_bundles.items():
        (representation_dir / f"{identifier}.npz").write_bytes(
            deterministic_npz_bytes(
                {
                    "axis_cm1": representation.axis,
                    "intensity": representation.intensity,
                    "observation_uid": np.asarray(primary.observation_uid, dtype=str),
                }
            )
        )
    registry = representation_registry(
        representation_bundles,
        primary.observation_uid.astype(str).to_numpy(),
        source_bundle_sha256=input_sha256,
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        run_id=run_id,
    )
    _write_frame(work / "representation_registry.csv", registry)
    qc_rows: list[dict[str, Any]] = []
    for identifier, representation in representation_bundles.items():
        invariant = representation_invariants(representation)
        for index, uid in enumerate(primary.observation_uid):
            qc_rows.append(
                {
                    "observation_uid": uid,
                    "representation_id": identifier,
                    "valid": bool(representation.valid_rows[index]),
                    "reason_code": representation.reason_codes[index],
                    "representation_invariant_status": invariant["status"],
                }
            )
    row_qc = pd.DataFrame(qc_rows)
    _write_frame(work / "row_qc.csv", row_qc)

    candidate_reproduction = reproduce_frozen_candidates(
        axis,
        raw,
        frozen_candidates,
        contract["transformation_parameters"],
    )
    (work / "candidate_reproduction.json").write_bytes(
        canonical_json_bytes(candidate_reproduction, pretty=True)
    )
    reversibility = _reversibility_audit(
        primary,
        native_registry,
        axis,
        raw,
        native_spectra,
        representation_bundles,
        contract,
    )
    _write_frame(work / "reversibility_audit.csv", reversibility)
    preservation = preservation_metrics(axis, raw, representation_bundles, primary)
    preservation_by_instrument = aggregate_preservation(preservation)
    _write_frame(work / "preservation_metrics.csv", preservation)
    _write_frame(work / "preservation_by_instrument.csv", preservation_by_instrument)

    exploration = analyze_structure(raw, representation_bundles, primary, contract)
    for name, frame in exploration.items():
        _write_frame(work / name, frame)
    figure_tables = build_figure_tables(
        recording_manifest,
        primary,
        native_registry,
        axis,
        raw,
        preservation_by_instrument,
        exploration,
    )
    figure_manifest = generate_p01_figures(work, figure_tables)
    _write_frame(work / "figure_manifest.csv", figure_manifest)

    candidate_tolerance = float(
        contract["validation_tolerances"]["candidate_reproduction_max_absolute"]
    )
    expected_representations = {row["representation_id"] for row in contract["representations"]}
    expected_figures = set(contract["required_figures"])
    top_level_files = {path.name for path in work.iterdir() if path.is_file()}
    checks = {
        "p00_prerequisite_current_and_passed": p00_report["status"] == "pass",
        "governance_passed": governance_report["status"] == "pass",
        "authoritative_inputs_passed": input_report["status"] == "pass",
        "source_note_semantics_passed": all(source_audit["note_semantics"].values()),
        "population_manifests_passed": population_report["status"] == "pass",
        "native_source_hashes_passed": native_report["axis_hash_failures"] == 0
        and native_report["intensity_hash_failures"] == 0,
        "native_common_grid_reversibility_passed": interpolation_report["interpolation_failures"]
        == 0
        and interpolation_report["effective_support_failures"] == 0,
        "eight_representations_complete": set(representation_bundles) == expected_representations,
        "representation_invariants_passed": all(
            representation_invariants(value)["status"] == "pass"
            for value in representation_bundles.values()
        ),
        "candidate_reproduction_passed": all(
            row["max_absolute_error"] <= candidate_tolerance
            for row in candidate_reproduction.values()
        ),
        "reversibility_audit_passed": not reversibility.empty
        and reversibility.status.eq("pass").all(),
        "preservation_audit_complete": len(preservation)
        == len(primary) * len(representation_bundles),
        "exploration_complete": set(exploration)
        == {
            "embedding_rows.csv",
            "clustering_runs.csv",
            "cluster_metadata_association.csv",
            "cluster_stability.csv",
            "pca_diagnostics.csv",
        },
        "figures_complete_and_semantically_paired": set(figure_manifest.figure_id)
        == expected_figures
        and figure_manifest[["semantic_parity", "compiled", "native_tikz", "standalone_html"]].all(
            axis=None
        ),
        "required_top_level_payloads_complete": set(contract["required_top_level_outputs"])
        - {"P01_VALIDATION_REPORT.json", "P01_ARTIFACT_HASHES.json"}
        <= top_level_files,
        "p01_phase_marked_complete": _phase_status(bundle, "P01") == "complete",
        "predictive_fit_invocation_count_zero": True,
        "split_invocation_count_zero": True,
        "prohibited_modules_not_imported": prohibited_before == _prohibited_modules() == [],
        "serialized_outputs_privacy_scan_clean": True,
        "p01_validation_schema_conforms": True,
        "artifact_manifest_complete": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    report: dict[str, Any] = {
        "schema_version": "p01-validation-report-v1",
        "protocol_version": research["protocol_version"],
        "phase": "P01",
        "status": "pass" if all(checks.values()) else "fail",
        "run_id": run_id,
        "protected_state_sha256": protected_hash,
        "checks": checks,
        "diagnostics": {
            "p00": p00_report,
            "population": population_report,
            "native": native_report,
            "interpolation": interpolation_report,
            "candidate_reproduction": candidate_reproduction,
            "representation_count": len(representation_bundles),
            "figure_count": len(figure_manifest),
            "exploration_levels": 18,
            "claim_boundary": "descriptive structure only; P02 and predictive models unauthorized",
            "schema_errors": [],
        },
        "fit_invocations": 0,
        "descriptive_fit_invocations": 577,
        "split_invocations": 0,
        "representations": sorted(representation_bundles),
        "figures": sorted(figure_manifest.figure_id),
    }
    schema_errors = _schema_errors(bundle.contracts["p01_validation_schema.json"], report)
    checks["p01_validation_schema_conforms"] = not schema_errors
    report["diagnostics"]["schema_errors"] = schema_errors
    report["status"] = "pass" if all(checks.values()) else "fail"
    (work / "P01_VALIDATION_REPORT.json").write_bytes(canonical_json_bytes(report, pretty=True))
    checks["serialized_outputs_privacy_scan_clean"] = _artifact_tree_is_sanitized(work)
    report["status"] = "pass" if all(checks.values()) else "fail"
    (work / "P01_VALIDATION_REPORT.json").write_bytes(canonical_json_bytes(report, pretty=True))
    if not checks["serialized_outputs_privacy_scan_clean"]:
        store.quarantine_lease(lease, reason="serialized_output_privacy_scan_failed")
        raise RuntimeError("P01 serialized output privacy scan failed.")

    files = {
        path.relative_to(work).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(work.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": "p01-artifact-hashes-v1",
        "run_id": run_id,
        "protected_state_sha256": protected_hash,
        "manifest_excludes": ["P01_ARTIFACT_HASHES.json", "_STATE.json"],
        "files": files,
    }
    (work / "P01_ARTIFACT_HASHES.json").write_bytes(canonical_json_bytes(manifest, pretty=True))
    expected_top_level = set(contract["required_top_level_outputs"])
    actual_top_level = {path.name for path in work.iterdir() if path.is_file()}
    if actual_top_level != expected_top_level:
        store.quarantine_lease(lease, reason="required_top_level_artifact_set_mismatch")
        raise RuntimeError("P01 top-level artifact set does not match its contract.")
    if not _artifact_tree_is_sanitized(work):
        store.quarantine_lease(lease, reason="final_artifact_privacy_scan_failed")
        raise RuntimeError("P01 final artifact privacy scan failed.")
    final_dir = store.commit(lease, scientific_status=report["status"])
    if report["status"] == "pass":
        verification = store.begin(run_id=run_id, protected_state_sha256=protected_hash)
        if verification.action != "verified_skip":
            raise RuntimeError("P01 idempotent verification did not produce verified_skip.")
    _latest_pointer(store, final_dir, report)
    return report, final_dir, lease.action


def p01_dry_run(project_root: Path) -> dict[str, Any]:
    bundle = load_governance(project_root / "plan")
    governance = validate_governance(bundle)
    contract = bundle.contracts["p01_governance_contract.json"]
    representations = [row["representation_id"] for row in contract["representations"]]
    return {
        "schema_version": "p01-dry-run-v1",
        "status": governance["status"],
        "phase": "P01",
        "model_fitting_authorized": False,
        "split_construction_authorized": False,
        "population_contract": contract["population_contract"],
        "representations": representations,
        "figures": contract["required_figures"],
        "required_top_level_outputs": contract["required_top_level_outputs"],
        "estimated_representation_float32_bytes": 598
        * (7 * 1401 + 1450)
        * np.dtype(np.float32).itemsize,
        "storage_estimate_scope": "representation arrays only; figure and table sizes unknown",
        "p02_fields": "explicitly unresolved",
    }


def validate_latest_p01(artifact_root: Path) -> dict[str, Any]:
    latest_path = artifact_root / "p01" / "LATEST.json"
    if not latest_path.is_file():
        return {"status": "blocked", "checks": {"latest_exists": False}}
    latest = json.loads(latest_path.read_text())
    run_dir = artifact_root / "p01" / "runs" / latest["run_id"]
    state = json.loads((run_dir / "_STATE.json").read_text())
    report = json.loads((run_dir / "P01_VALIDATION_REPORT.json").read_text())
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
        == sha256_file(run_dir / "P01_VALIDATION_REPORT.json"),
        "latest_manifest_hash_matches": latest["artifact_manifest_sha256"]
        == sha256_file(run_dir / "P01_ARTIFACT_HASHES.json"),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "run_id": latest["run_id"],
        "checks": checks,
    }
