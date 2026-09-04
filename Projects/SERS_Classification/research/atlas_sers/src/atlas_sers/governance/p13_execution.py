"""Governed, resumable execution boundary for locked P13 shards."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from atlas_sers.evaluation.p03_runtime import P03Dataset
from atlas_sers.evaluation.p13_runtime import execute_p13_shard_rows
from atlas_sers.governance.canonical import canonical_json_bytes, sha256_file, sha256_value
from atlas_sers.governance.p03_store import P03ShardStore
from atlas_sers.governance.p13 import validate_latest_p13_plan


@dataclass(frozen=True)
class P13ExecutionContext:
    run_id: str
    protected_state_sha256: str
    plan_run: Path
    p01_run: Path
    execution_root: Path


def _latest_run(artifact_root: Path, phase: str) -> Path:
    latest = json.loads((artifact_root / phase / "LATEST.json").read_text())
    return artifact_root / phase / "runs" / str(latest["run_id"])


def execution_context(*, artifact_root: Path, project_root: Path) -> P13ExecutionContext:
    validation = validate_latest_p13_plan(artifact_root)
    if validation["status"] != "pass":
        raise PermissionError("The P13 no-fit plan has not authorized execution.")
    plan_run = _latest_run(artifact_root, "p13plan")
    p01_run = _latest_run(artifact_root, "p01")
    protected = {
        "schema_version": "nato-sers-p13-execution-state-v1",
        "plan_protected_state_sha256": validation["protected_state_sha256"],
        "runtime_sha256": sha256_file(
            project_root / "src/atlas_sers/evaluation/p13_runtime.py"
        ),
        "execution_boundary_sha256": sha256_file(
            project_root / "src/atlas_sers/governance/p13_execution.py"
        ),
    }
    digest = sha256_value(protected)
    run_id = f"P13-{digest[:24]}"
    root = artifact_root / "p13" / "runs" / run_id
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "protected_state.json"
    content = canonical_json_bytes(
        {**protected, "protected_state_sha256": digest, "run_id": run_id},
        pretty=True,
    )
    try:
        descriptor = os.open(state_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        for _ in range(100):
            try:
                observed = state_path.read_bytes()
            except OSError:
                observed = b""
            if observed == content:
                break
            if observed:
                raise RuntimeError(
                    "P13 execution protected state conflicts with existing state."
                ) from error
            time.sleep(0.01)
        else:
            raise RuntimeError("P13 execution state creation did not complete.") from error
    else:
        try:
            os.write(descriptor, content)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return P13ExecutionContext(run_id, digest, plan_run, p01_run, root)


def _load_tables(context: P13ExecutionContext) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(context.plan_run / f"{name}.csv", low_memory=False)
        for name in (
            "context_registry",
            "role_registry",
            "procedure_registry",
            "fit_manifest",
            "shard_manifest",
        )
    }


def _load_dataset(context: P13ExecutionContext, representation_id: str) -> P03Dataset:
    path = context.p01_run / "representations" / f"{representation_id}.npz"
    with np.load(path, allow_pickle=False) as payload:
        intensity = payload["intensity"].copy()
        uids = payload["observation_uid"].astype(str)
    manifest = pd.read_csv(context.p01_run / "primary_manifest.csv", low_memory=False)
    return P03Dataset.from_frozen_representation(
        intensity=intensity,
        representation_uids=uids,
        metadata=manifest,
    )


def execute_shard(
    *, artifact_root: Path, project_root: Path, shard_index: int
) -> tuple[Path, str, str]:
    context = execution_context(artifact_root=artifact_root, project_root=project_root)
    tables = _load_tables(context)
    shards = tables["shard_manifest"].reset_index(drop=True)
    if shard_index < 0 or shard_index >= len(shards):
        raise IndexError("P13 shard index is outside the frozen manifest.")
    shard = shards.iloc[shard_index]
    contexts = tables["context_registry"]
    selected_contexts = contexts[
        contexts.domain_id.astype(str).eq(str(shard.domain_id))
        & contexts.policy_id.astype(str).eq(str(shard.policy_id))
        & contexts.outer_repeat.eq(int(shard.outer_repeat))
    ]
    representation_ids = selected_contexts.representation_id.astype(str).unique()
    if len(representation_ids) != 1:
        raise ValueError("P13 shard does not resolve to one representation.")
    dataset = _load_dataset(context, representation_ids[0])
    role_ids = set(selected_contexts.role_context_id.astype(str))
    roles = tables["role_registry"][
        tables["role_registry"].role_context_id.astype(str).isin(role_ids)
    ]
    context_ids = set(selected_contexts.context_id.astype(str))
    procedures = tables["procedure_registry"][
        tables["procedure_registry"].context_id.astype(str).isin(context_ids)
    ]
    fits = tables["fit_manifest"][
        tables["fit_manifest"].context_id.astype(str).isin(context_ids)
    ]
    store = P03ShardStore(run_root=context.execution_root / "prediction")
    lease = store.begin(
        shard_id=shard_index,
        protected_state_sha256=context.protected_state_sha256,
    )
    if lease.action == "verified_skip":
        return lease.final_dir, lease.action, context.run_id
    if lease.temporary_dir is None:
        raise RuntimeError("P13 shard lease has no temporary directory.")
    try:
        with threadpool_limits(limits=1):
            result = execute_p13_shard_rows(
                dataset=dataset,
                contexts=selected_contexts,
                role_registry=roles,
                procedure_registry=procedures,
                fit_manifest=fits,
                scientific_fitting_authorized=True,
            )
        result.fit_status.to_csv(
            lease.temporary_dir / "fit_status.csv", index=False
        )
        result.calibration_status.to_csv(
            lease.temporary_dir / "calibration_status.csv", index=False
        )
        result.fold_endpoint_status.to_csv(
            lease.temporary_dir / "fold_endpoint_status.csv", index=False
        )
        result.predictions.to_parquet(
            lease.temporary_dir / "predictions.parquet",
            index=False,
            compression="zstd",
        )
        descriptor = {
            "schema_version": "nato-sers-p13-prediction-shard-v1",
            "shard_index": shard_index,
            "shard_id": str(shard.shard_id),
            "domain_id": str(shard.domain_id),
            "policy_id": str(shard.policy_id),
            "outer_repeat": int(shard.outer_repeat),
            "fit_rows": len(result.fit_status),
            "endpoint_rows": len(result.fold_endpoint_status),
            "prediction_rows": len(result.predictions),
            "terminal_fit_statuses": sorted(set(result.fit_status.status.astype(str))),
        }
        (lease.temporary_dir / "shard_descriptor.json").write_bytes(
            canonical_json_bytes(descriptor, pretty=True)
        )
        return store.commit(lease), lease.action, context.run_id
    except Exception:
        store.abort(lease, reason="p13_prediction_shard_execution_failed")
        raise


def execute_batch(
    *,
    artifact_root: Path,
    project_root: Path,
    worker_index: int,
    worker_count: int,
    start_index: int = 0,
    stop_index: int | None = None,
) -> dict[str, Any]:
    if worker_count < 1 or worker_index < 0 or worker_index >= worker_count:
        raise ValueError("Invalid P13 worker partition.")
    context = execution_context(artifact_root=artifact_root, project_root=project_root)
    total = len(pd.read_csv(context.plan_run / "shard_manifest.csv"))
    stop = total if stop_index is None else min(stop_index, total)
    indices = [
        index
        for index in range(max(0, start_index), stop)
        if index % worker_count == worker_index
    ]
    actions: dict[str, int] = {}
    for index in indices:
        _, action, _ = execute_shard(
            artifact_root=artifact_root,
            project_root=project_root,
            shard_index=index,
        )
        actions[action] = actions.get(action, 0) + 1
    return {
        "status": "pass",
        "run_id": context.run_id,
        "worker_index": worker_index,
        "worker_count": worker_count,
        "processed": len(indices),
        "actions": actions,
    }
