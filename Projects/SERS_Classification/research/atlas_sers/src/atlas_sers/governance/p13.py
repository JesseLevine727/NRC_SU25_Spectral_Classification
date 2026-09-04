"""Build and validate the locked P13 outcome-blind execution plan."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation.p13_plan import POLICY_REPRESENTATIONS, build_p13_plan
from atlas_sers.governance.artifacts import ArtifactStore
from atlas_sers.governance.canonical import canonical_json_bytes, sha256_file, sha256_value

FROZEN_MANIFEST_SHA256 = (
    "db1f298a76aeb9962db004776a9f41d6c9afe5b76c39aa9277a24848108d5f90"
)
FROZEN_P02_SPLIT_SHA256 = (
    "d92da67742dd74693da518395c06dd1f33c16145d5d786a557166c8a6cb05558"
)
P03_EXECUTION_RUN_ID = "P03-513a0f9686c37cbc0d682645"


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="",
    ).encode()


def _latest_run(artifact_root: Path, phase: str) -> tuple[dict[str, Any], Path]:
    latest = json.loads((artifact_root / phase / "LATEST.json").read_text())
    run = artifact_root / phase / "runs" / str(latest["run_id"])
    if latest.get("status") != "pass" or not run.is_dir():
        raise RuntimeError(f"The latest {phase.upper()} prerequisite is not valid.")
    return latest, run


def _representation_audit(p01_run: Path, manifest: pd.DataFrame) -> dict[str, Any]:
    expected_uids = manifest.observation_uid.astype(str).to_numpy()
    records: dict[str, Any] = {}
    for policy_id, representation_id in POLICY_REPRESENTATIONS.items():
        path = p01_run / "representations" / f"{representation_id}.npz"
        with np.load(path, allow_pickle=False) as payload:
            keys = sorted(payload.files)
            uids = payload["observation_uid"].astype(str)
            axis = np.asarray(payload["axis_cm1"])
            intensity = np.asarray(payload["intensity"])
            checks = {
                "keys_exact": keys == ["axis_cm1", "intensity", "observation_uid"],
                "shape_exact": intensity.shape == (598, 1401),
                "axis_shape_exact": axis.shape == (1401,),
                "uid_order_exact": np.array_equal(uids, expected_uids),
                "finite": bool(np.isfinite(axis).all() and np.isfinite(intensity).all()),
            }
        if not all(checks.values()):
            raise ValueError(f"Frozen representation parity failed for {policy_id}.")
        records[policy_id] = {
            "representation_id": representation_id,
            "file_sha256": sha256_file(path),
            "observation_uid_sha256": sha256_value(uids.tolist()),
            "axis_sha256": sha256_value(axis.tolist()),
            "shape": list(intensity.shape),
            "checks": checks,
        }
    return records


def _plan_inputs(project_root: Path, artifact_root: Path) -> dict[str, Any]:
    p01_latest, p01_run = _latest_run(artifact_root, "p01")
    p02_latest, p02_run = _latest_run(artifact_root, "p02")
    p03plan_latest, p03plan_run = _latest_run(artifact_root, "p03plan")
    p03_run = artifact_root / "p03" / "runs" / P03_EXECUTION_RUN_ID
    if not p03_run.is_dir():
        raise RuntimeError("The frozen completed P03 execution run is unavailable.")
    paths = {
        "manifest": p01_run / "primary_manifest.csv",
        "master_splits": p02_run / "master_split_registry.csv",
        "inner_splits": p02_run / "inner_master_split_registry.csv",
        "candidate_registry": p03plan_run / "candidate_registry.csv",
        "selected_specs": p03_run
        / "final_aggregation/shards/shard-000000/selected_model_specs.csv",
        "selection_trace": p03_run
        / "selection_aggregation/shards/shard-000000/selection_trace.parquet",
        "domain_support": project_root
        / "plan/registries/p13_domain_support_registry.csv",
        "crossover_support": project_root
        / "plan/registries/p13_crossover_support_registry.csv",
        "decision_registry": project_root / "plan/registries/p13_decision_registry.csv",
        "execution_contract": project_root / "plan/contracts/p13_execution_contract.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P13 prerequisite files are missing: {missing}")
    if sha256_file(paths["manifest"]) != FROZEN_MANIFEST_SHA256:
        raise ValueError("P13 manifest differs from the locked 598-spectrum population.")
    if sha256_file(paths["master_splits"]) != FROZEN_P02_SPLIT_SHA256:
        raise ValueError("P13 master splits differ from the locked P02 registry.")
    return {
        "p01_latest": p01_latest,
        "p01_run": p01_run,
        "p02_latest": p02_latest,
        "p02_run": p02_run,
        "p03plan_latest": p03plan_latest,
        "p03plan_run": p03plan_run,
        "p03_run": p03_run,
        "paths": paths,
    }


def _latest_pointer(store: ArtifactStore, final_dir: Path, report: dict[str, Any]) -> None:
    pointer = {
        "schema_version": "nato-sers-p13-plan-latest-v1",
        "run_id": report["run_id"],
        "status": report["status"],
        "scientific_fitting_authorized": report["scientific_fitting_authorized"],
        "protected_state_sha256": report["protected_state_sha256"],
        "report_sha256": sha256_file(final_dir / "P13_PLAN_VALIDATION_REPORT.json"),
    }
    temporary = store.phase_root / ".LATEST.json.tmp"
    temporary.write_bytes(canonical_json_bytes(pointer, pretty=True))
    os.replace(temporary, store.phase_root / "LATEST.json")


def execute_p13_plan(
    *, project_root: Path, private_root: Path, artifact_root: Path
) -> tuple[dict[str, Any], Path, str]:
    """Materialize and validate P13 roles and fits before any P13 model fitting."""

    project_root = project_root.resolve()
    artifact_root = artifact_root.resolve()
    inputs = _plan_inputs(project_root, artifact_root)
    paths: dict[str, Path] = inputs["paths"]
    manifest = pd.read_csv(paths["manifest"], low_memory=False)
    representations = _representation_audit(inputs["p01_run"], manifest)
    tables = build_p13_plan(
        manifest=manifest,
        master_splits=pd.read_csv(paths["master_splits"], low_memory=False),
        inner_splits=pd.read_csv(paths["inner_splits"], low_memory=False),
        domain_support=pd.read_csv(paths["domain_support"], low_memory=False),
        candidate_registry=pd.read_csv(paths["candidate_registry"], low_memory=False),
        selected_specs=pd.read_csv(paths["selected_specs"], low_memory=False),
        selection_trace=pd.read_parquet(paths["selection_trace"]),
        input_paths=paths,
    )
    if tables.validation_report["status"] != "pass":
        raise RuntimeError("P13 no-fit expansion failed validation.")
    input_hashes = {
        "schema_version": "nato-sers-p13-input-hashes-v1",
        "p01_run_id": inputs["p01_latest"]["run_id"],
        "p02_run_id": inputs["p02_latest"]["run_id"],
        "p03_plan_run_id": inputs["p03plan_latest"]["run_id"],
        "p03_execution_run_id": P03_EXECUTION_RUN_ID,
        "files": tables.input_hashes,
        "representations": representations,
    }
    code_hashes = {
        name: sha256_file(project_root / relative)
        for name, relative in {
            "plan": "src/atlas_sers/evaluation/p13_plan.py",
            "governance": "src/atlas_sers/governance/p13.py",
            "contract": "plan/contracts/p13_execution_contract.json",
        }.items()
    }
    protected_state = {
        "schema_version": "nato-sers-p13-plan-protected-state-v1",
        "protocol_version": "nato-sers-p13-v1-locked",
        "input_hashes_sha256": sha256_value(input_hashes),
        "code_hashes": code_hashes,
        "counts": tables.validation_report["counts"],
        "scientific_fitting_authorized": True,
    }
    protected_state_sha256 = sha256_value(protected_state)
    run_id = f"P13PLAN-{protected_state_sha256[:24]}"
    store = ArtifactStore(
        artifact_root=artifact_root,
        input_root=private_root,
        project_root=project_root,
        phase="p13plan",
    )
    lease = store.begin(run_id=run_id, protected_state_sha256=protected_state_sha256)
    if lease.action == "verified_skip":
        report = json.loads(
            (lease.final_dir / "P13_PLAN_VALIDATION_REPORT.json").read_text()
        )
        _latest_pointer(store, lease.final_dir, report)
        return report, lease.final_dir, lease.action
    if lease.work_dir is None:
        raise RuntimeError("P13 plan transaction has no work directory.")
    work = lease.work_dir
    payloads: dict[str, pd.DataFrame | dict[str, Any]] = {
        "context_registry.csv": tables.context_registry,
        "role_registry.csv": tables.role_registry,
        "procedure_registry.csv": tables.procedure_registry,
        "fit_manifest.csv": tables.fit_manifest,
        "expected_endpoint_registry.csv": tables.expected_endpoint_registry,
        "shard_manifest.csv": tables.shard_manifest,
        "input_hashes.json": input_hashes,
        "protected_state.json": protected_state,
    }
    for name, payload in payloads.items():
        content = (
            _csv_bytes(payload)
            if isinstance(payload, pd.DataFrame)
            else canonical_json_bytes(payload, pretty=True)
        )
        (work / name).write_bytes(content)
    report = {
        **tables.validation_report,
        "phase": "P13-PLAN",
        "run_id": run_id,
        "protected_state_sha256": protected_state_sha256,
        "model_fit_invocations": 0,
        "scientific_fitting_authorized": True,
        "claim_boundary": "outcome-blind execution planning only; no P13 model was fitted",
        "compute": {
            "planned_fit_invocations": len(tables.fit_manifest),
            "planned_outer_endpoints": len(tables.expected_endpoint_registry),
            "resumable_shards": len(tables.shard_manifest),
            "maximum_fits_in_one_shard": int(tables.shard_manifest.fit_count.max()),
        },
    }
    (work / "P13_PLAN_VALIDATION_REPORT.json").write_bytes(
        canonical_json_bytes(report, pretty=True)
    )
    if report["status"] != "pass" or not all(report["checks"].values()):
        store.quarantine_lease(lease, reason="p13_plan_validation_failed")
        raise RuntimeError("P13 no-fit execution plan failed closed.")
    final_dir = store.commit(lease, scientific_status="pass")
    verification = store.begin(
        run_id=run_id, protected_state_sha256=protected_state_sha256
    )
    if verification.action != "verified_skip":
        raise RuntimeError("P13 plan did not pass verified-skip reconciliation.")
    _latest_pointer(store, final_dir, report)
    return report, final_dir, lease.action


def validate_latest_p13_plan(artifact_root: Path) -> dict[str, Any]:
    latest_path = artifact_root / "p13plan" / "LATEST.json"
    if not latest_path.is_file():
        return {"status": "blocked", "checks": {"latest_exists": False}}
    latest = json.loads(latest_path.read_text())
    run = artifact_root / "p13plan" / "runs" / str(latest["run_id"])
    state = json.loads((run / "_STATE.json").read_text())
    report = json.loads((run / "P13_PLAN_VALIDATION_REPORT.json").read_text())
    files = state.get("files", {})
    checks = {
        "latest_exists": True,
        "report_passes": report["status"] == "pass"
        and all(report["checks"].values()),
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
        == sha256_file(run / "P13_PLAN_VALIDATION_REPORT.json"),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "run_id": latest["run_id"],
        "protected_state_sha256": latest["protected_state_sha256"],
        "checks": checks,
        "compute": report["compute"],
    }
