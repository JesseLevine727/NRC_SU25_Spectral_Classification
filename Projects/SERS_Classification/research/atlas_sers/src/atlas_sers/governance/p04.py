"""Build and validate the locked outcome-blind P04 execution plan."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation.p04_plan import build_p04_plan
from atlas_sers.governance.artifacts import ArtifactStore
from atlas_sers.governance.canonical import canonical_json_bytes, sha256_file, sha256_value

FROZEN_MANIFEST_SHA256 = (
    "db1f298a76aeb9962db004776a9f41d6c9afe5b76c39aa9277a24848108d5f90"
)
FROZEN_P02_SPLIT_SHA256 = (
    "d92da67742dd74693da518395c06dd1f33c16145d5d786a557166c8a6cb05558"
)


def _latest_run(artifact_root: Path, phase: str) -> tuple[dict[str, Any], Path]:
    latest = json.loads((artifact_root / phase / "LATEST.json").read_text())
    run = artifact_root / phase / "runs" / str(latest["run_id"])
    if latest.get("status") != "pass" or not run.is_dir():
        raise RuntimeError(f"The latest {phase.upper()} prerequisite is invalid.")
    return latest, run


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="",
    ).encode()


def _write_latest(store: ArtifactStore, final_dir: Path, report: dict[str, Any]) -> None:
    pointer = {
        "schema_version": "nato-sers-p04-plan-latest-v1",
        "run_id": report["run_id"],
        "status": report["status"],
        "scientific_fitting_authorized": report["scientific_fitting_authorized"],
        "protected_state_sha256": report["protected_state_sha256"],
        "report_sha256": sha256_file(final_dir / "P04_PLAN_VALIDATION_REPORT.json"),
    }
    temporary = store.phase_root / ".LATEST.json.tmp"
    temporary.write_bytes(canonical_json_bytes(pointer, pretty=True))
    os.replace(temporary, store.phase_root / "LATEST.json")


def execute_p04_plan(
    *, project_root: Path, private_root: Path, artifact_root: Path
) -> tuple[dict[str, Any], Path, str]:
    project_root = project_root.resolve()
    artifact_root = artifact_root.resolve()
    p01_latest, p01_run = _latest_run(artifact_root, "p01")
    p02_latest, p02_run = _latest_run(artifact_root, "p02")
    inputs = {
        "manifest": p01_run / "primary_manifest.csv",
        "representation": p01_run / "representations/R_MIN_400_1800.npz",
        "master_splits": p02_run / "master_split_registry.csv",
        "t3_partitions": p02_run / "t3_partition_registry.csv",
        "inner_selection": p02_run / "inner_selection_registry.csv",
        "inner_fallback": p02_run / "inner_master_split_registry.csv",
        "contract": project_root / "plan/contracts/p04_execution_contract.json",
        "hyperparameters": project_root / "plan/contracts/hyperparameter_registry.json",
        "split_contract": project_root / "plan/contracts/split_contract.json",
        "p03_handoff": project_root / "plan/P03_HANDOFF.md",
        "p13_completion": project_root / "plan/P13_COMPLETION_AUDIT.md",
    }
    missing = [name for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P04 prerequisites are missing: {missing}")
    if sha256_file(inputs["manifest"]) != FROZEN_MANIFEST_SHA256:
        raise ValueError("P04 manifest differs from the frozen primary population.")
    if sha256_file(inputs["master_splits"]) != FROZEN_P02_SPLIT_SHA256:
        raise ValueError("P04 split registry differs from the frozen P02 split.")
    contract = json.loads(inputs["contract"].read_text())
    manifest = pd.read_csv(inputs["manifest"], low_memory=False)
    with np.load(inputs["representation"], allow_pickle=False) as payload:
        representation_checks = {
            "keys_exact": sorted(payload.files)
            == ["axis_cm1", "intensity", "observation_uid"],
            "shape_exact": payload["intensity"].shape == (598, 1401),
            "uid_order_exact": np.array_equal(
                payload["observation_uid"].astype(str),
                manifest.observation_uid.astype(str).to_numpy(),
            ),
            "axis_exact": np.array_equal(
                payload["axis_cm1"].astype(float),
                np.arange(400, 1801, dtype=float),
            ),
            "finite": bool(np.isfinite(payload["intensity"]).all()),
            "row_minmax": bool(
                np.allclose(payload["intensity"].min(axis=1), 0, atol=1e-6)
                and np.allclose(payload["intensity"].max(axis=1), 1, atol=1e-6)
            ),
        }
    if not all(representation_checks.values()):
        raise ValueError("P04 R_MIN_400_1800 representation audit failed.")
    from atlas_sers.models.deep import architecture_audit

    architecture = architecture_audit(class_count=3)
    architecture_checks = {
        "input_exact": architecture.input_shape == (2, 1, 1401),
        "stem_exact": architecture.stem_shape == (2, 24, 1401),
        "transition_1_exact": architecture.transition_1_shape == (2, 48, 701),
        "transition_2_exact": architecture.transition_2_shape == (2, 64, 351),
        "pool_exact": architecture.pooled_shape == (2, 64, 16),
        "embedding_64": architecture.embedding_shape == (2, 64),
        "station_classes_3": architecture.logits_shape == (2, 3),
        "below_parameter_ceiling": architecture.trainable_parameters
        < int(contract["architecture"]["maximum_trainable_parameters"]),
        "no_batch_normalization": architecture.batch_normalization_modules == 0,
    }
    if not all(architecture_checks.values()):
        raise ValueError("P04 architecture audit failed closed.")
    tables = build_p04_plan(
        manifest=manifest,
        master_splits=pd.read_csv(inputs["master_splits"], low_memory=False),
        t3_partitions=pd.read_csv(inputs["t3_partitions"], low_memory=False),
        inner_selection=pd.read_csv(inputs["inner_selection"], low_memory=False),
        inner_fallback=pd.read_csv(inputs["inner_fallback"], low_memory=False),
        contract=contract,
    )
    if tables.validation_report["status"] != "pass":
        raise RuntimeError("P04 outcome-blind expansion failed.")
    input_hashes = {
        "schema_version": "nato-sers-p04-input-hashes-v1",
        "p01_run_id": p01_latest["run_id"],
        "p02_run_id": p02_latest["run_id"],
        "files": {name: sha256_file(path) for name, path in inputs.items()},
        "outcome_boundary": {
            "p03_and_p13_outcomes_read_for_selection": False,
            "p03_handoff_and_p13_completion_used_only_as_hashed_governance_prerequisites": True,
        },
    }
    code_hashes = {
        name: sha256_file(project_root / relative)
        for name, relative in {
            "model": "src/atlas_sers/models/deep.py",
            "plan": "src/atlas_sers/evaluation/p04_plan.py",
            "governance": "src/atlas_sers/governance/p04.py",
            "contract": "plan/contracts/p04_execution_contract.json",
        }.items()
    }
    protected = {
        "schema_version": "nato-sers-p04-plan-protected-state-v1",
        "protocol_version": contract["protocol_version"],
        "input_hashes_sha256": sha256_value(input_hashes),
        "code_hashes": code_hashes,
        "architecture": architecture.__dict__,
        "counts": tables.validation_report["counts"],
        "scientific_fitting_authorized": True,
    }
    protected_hash = sha256_value(protected)
    run_id = f"P04PLAN-{protected_hash[:24]}"
    store = ArtifactStore(
        artifact_root=artifact_root,
        input_root=private_root,
        project_root=project_root,
        phase="p04plan",
    )
    lease = store.begin(run_id=run_id, protected_state_sha256=protected_hash)
    if lease.action == "verified_skip":
        report = json.loads((lease.final_dir / "P04_PLAN_VALIDATION_REPORT.json").read_text())
        _write_latest(store, lease.final_dir, report)
        return report, lease.final_dir, lease.action
    if lease.work_dir is None:
        raise RuntimeError("P04 plan transaction has no working directory.")
    payloads: dict[str, pd.DataFrame | dict[str, Any]] = {
        "candidate_registry.csv": tables.candidate_registry,
        "context_registry.csv": tables.context_registry,
        "role_registry.csv": tables.role_registry,
        "fit_manifest.csv": tables.fit_manifest,
        "expected_endpoint_registry.csv": tables.expected_endpoint_registry,
        "shard_manifest.csv": tables.shard_manifest,
        "input_hashes.json": input_hashes,
        "protected_state.json": protected,
    }
    for name, payload in payloads.items():
        (lease.work_dir / name).write_bytes(
            _csv_bytes(payload)
            if isinstance(payload, pd.DataFrame)
            else canonical_json_bytes(payload, pretty=True)
        )
    report = {
        **tables.validation_report,
        "phase": "P04-PLAN",
        "run_id": run_id,
        "protected_state_sha256": protected_hash,
        "model_fit_invocations": 0,
        "scientific_fitting_authorized": True,
        "representation_checks": representation_checks,
        "architecture_checks": architecture_checks,
        "architecture": architecture.__dict__,
        "claim_boundary": (
            "outcome-blind P04 expansion only; no P04 model fitted and no P03/P13 "
            "held outcome read"
        ),
    }
    (lease.work_dir / "P04_PLAN_VALIDATION_REPORT.json").write_bytes(
        canonical_json_bytes(report, pretty=True)
    )
    final = store.commit(lease, scientific_status="pass")
    verification = store.begin(run_id=run_id, protected_state_sha256=protected_hash)
    if verification.action != "verified_skip":
        raise RuntimeError("P04 plan failed verified-skip reconciliation.")
    _write_latest(store, final, report)
    return report, final, lease.action


def validate_latest_p04_plan(artifact_root: Path) -> dict[str, Any]:
    latest_path = artifact_root / "p04plan/LATEST.json"
    if not latest_path.is_file():
        return {"status": "blocked", "checks": {"latest_exists": False}}
    latest = json.loads(latest_path.read_text())
    run = artifact_root / "p04plan/runs" / str(latest["run_id"])
    state = json.loads((run / "_STATE.json").read_text())
    report = json.loads((run / "P04_PLAN_VALIDATION_REPORT.json").read_text())
    files = state.get("files", {})
    checks = {
        "latest_exists": True,
        "report_passes": report["status"] == "pass" and all(report["checks"].values()),
        "representation_passes": all(report["representation_checks"].values()),
        "architecture_passes": all(report["architecture_checks"].values()),
        "zero_model_fits": report["model_fit_invocations"] == 0,
        "fitting_authorized": report["scientific_fitting_authorized"] is True,
        "state_complete": state["execution_status"] == "complete"
        and state["scientific_status"] == "pass",
        "all_files_rehash": isinstance(files, dict)
        and all(
            (run / name).is_file() and sha256_file(run / name) == digest
            for name, digest in files.items()
        ),
        "report_hash_matches": latest["report_sha256"]
        == sha256_file(run / "P04_PLAN_VALIDATION_REPORT.json"),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "run_id": latest["run_id"],
        "protected_state_sha256": latest["protected_state_sha256"],
        "checks": checks,
        "counts": report["counts"],
        "architecture": report["architecture"],
    }
