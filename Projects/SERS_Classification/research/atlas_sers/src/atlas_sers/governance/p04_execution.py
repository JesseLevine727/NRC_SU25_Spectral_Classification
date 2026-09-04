"""Resumable, fail-closed execution boundary for P04 compact D0 contexts."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from atlas_sers.evaluation.p04_runtime import (
    P04Dataset,
    select_candidate,
    status_record,
    train_fixed_epochs,
    train_with_validation,
)
from atlas_sers.governance.canonical import canonical_json_bytes, sha256_file, sha256_value
from atlas_sers.governance.p03_store import P03ShardStore
from atlas_sers.governance.p04 import validate_latest_p04_plan


@dataclass(frozen=True)
class P04ExecutionContext:
    run_id: str
    protected_state_sha256: str
    plan_run: Path
    p01_run: Path
    execution_root: Path
    contract: dict[str, Any]


def _latest_run(artifact_root: Path, phase: str) -> Path:
    latest = json.loads((artifact_root / phase / "LATEST.json").read_text())
    return artifact_root / phase / "runs" / str(latest["run_id"])


def execution_context(*, artifact_root: Path, project_root: Path) -> P04ExecutionContext:
    validation = validate_latest_p04_plan(artifact_root)
    if validation["status"] != "pass":
        raise PermissionError("The P04 no-fit plan has not authorized execution.")
    plan_run = _latest_run(artifact_root, "p04plan")
    p01_run = _latest_run(artifact_root, "p01")
    contract_path = project_root / "plan/contracts/p04_execution_contract.json"
    protected = {
        "schema_version": "nato-sers-p04-execution-state-v1",
        "plan_protected_state_sha256": validation["protected_state_sha256"],
        "runtime_sha256": sha256_file(
            project_root / "src/atlas_sers/evaluation/p04_runtime.py"
        ),
        "execution_boundary_sha256": sha256_file(
            project_root / "src/atlas_sers/governance/p04_execution.py"
        ),
        "model_sha256": sha256_file(project_root / "src/atlas_sers/models/deep.py"),
        "contract_sha256": sha256_file(contract_path),
        "torch_version": torch.__version__,
    }
    digest = sha256_value(protected)
    run_id = f"P04-{digest[:24]}"
    root = artifact_root / "p04/runs" / run_id
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "protected_state.json"
    content = canonical_json_bytes(
        {**protected, "protected_state_sha256": digest, "run_id": run_id}, pretty=True
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
                raise RuntimeError("P04 execution state conflicts with existing state.") from error
            time.sleep(0.01)
        else:
            raise RuntimeError("P04 execution state creation did not complete.") from error
    else:
        try:
            os.write(descriptor, content)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return P04ExecutionContext(
        run_id,
        digest,
        plan_run,
        p01_run,
        root,
        json.loads(contract_path.read_text()),
    )


def _tables(context: P04ExecutionContext) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(context.plan_run / f"{name}.csv", low_memory=False)
        for name in (
            "candidate_registry",
            "context_registry",
            "role_registry",
            "fit_manifest",
            "expected_endpoint_registry",
            "shard_manifest",
        )
    }


def _dataset(context: P04ExecutionContext) -> P04Dataset:
    path = context.p01_run / "representations/R_MIN_400_1800.npz"
    with np.load(path, allow_pickle=False) as payload:
        intensity = payload["intensity"].copy()
        uids = payload["observation_uid"].astype(str)
    manifest = pd.read_csv(context.p01_run / "primary_manifest.csv", low_memory=False)
    return P04Dataset.from_frozen_representation(
        intensity=intensity, representation_uids=uids, metadata=manifest
    )


def _role(
    dataset: P04Dataset, roles: pd.DataFrame, role_id: str
) -> tuple[np.ndarray, pd.DataFrame]:
    selected = roles[roles.role_id.astype(str).eq(str(role_id))]
    if selected.empty:
        raise ValueError(f"P04 role {role_id} is empty or absent.")
    return dataset.subset(selected.observation_uid.astype(str).tolist())


def _configure_torch(device: torch.device) -> None:
    configured = str(os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""))
    if configured not in {"", ":4096:8"}:
        raise RuntimeError("P04 requires CUBLAS_WORKSPACE_CONFIG=:4096:8.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _development_freeze_path(context: P04ExecutionContext) -> Path:
    return context.execution_root / "D0_DEVELOPMENT_FREEZE.json"


def _validate_development_freeze(context: P04ExecutionContext) -> dict[str, Any]:
    path = _development_freeze_path(context)
    if not path.is_file():
        raise PermissionError("EXP-N00-T3 is blocked until EXP-N00-DEV is frozen.")
    freeze = json.loads(path.read_text())
    if freeze.get("protected_state_sha256") != context.protected_state_sha256:
        raise PermissionError("The P04 development freeze belongs to different code/state.")
    if freeze.get("g2_status") != "pass":
        raise PermissionError("EXP-N00-T3 is blocked because G2 did not pass.")
    if freeze.get("file_sha256") != sha256_value(
        {key: value for key, value in freeze.items() if key != "file_sha256"}
    ):
        raise PermissionError("The P04 development freeze hash is invalid.")
    return freeze


def _execute_context(
    *,
    context: P04ExecutionContext,
    tables: dict[str, pd.DataFrame],
    dataset: P04Dataset,
    shard_index: int,
    device: torch.device,
) -> tuple[Path, str]:
    contexts = tables["context_registry"]
    selected = contexts[contexts.shard_index.eq(shard_index)]
    if len(selected) != 1:
        raise IndexError("P04 shard index does not identify one frozen context.")
    context_row = selected.iloc[0]
    if str(context_row.phase_gate) == "held_evaluation":
        _validate_development_freeze(context)
    context_id = str(context_row.context_id)
    roles = tables["role_registry"][
        tables["role_registry"].context_id.astype(str).eq(context_id)
    ]
    fits = tables["fit_manifest"][
        tables["fit_manifest"].context_id.astype(str).eq(context_id)
    ]
    candidates = tables["candidate_registry"]
    store = P03ShardStore(run_root=context.execution_root / "contexts")
    lease = store.begin(
        shard_id=shard_index, protected_state_sha256=context.protected_state_sha256
    )
    if lease.action == "verified_skip":
        return lease.final_dir, lease.action
    if lease.temporary_dir is None:
        raise RuntimeError("P04 context lease has no temporary directory.")
    optimization = context.contract["optimization"]
    status_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    try:
        selection_fits = fits[fits.stage.eq("inner_selection")]
        for fit in selection_fits.itertuples(index=False):
            train_values, train_metadata = _role(dataset, roles, str(fit.fit_role_id))
            validation_values, validation_metadata = _role(
                dataset, roles, str(fit.validation_role_id)
            )
            result = train_with_validation(
                fit_id=str(fit.fit_id),
                train_values=train_values,
                train_metadata=train_metadata,
                validation_values=validation_values,
                validation_metadata=validation_metadata,
                learning_rate=float(fit.learning_rate),
                weight_decay=float(fit.weight_decay),
                batch_size=int(fit.batch_size),
                seed=int(fit.seed),
                maximum_epochs=int(optimization["maximum_epochs"]),
                minimum_epochs=int(optimization["minimum_epochs"]),
                patience=int(optimization["patience"]),
                gradient_clip_norm=float(optimization["gradient_clip_norm"]),
                device=device,
            )
            record = status_record(pd.Series(fit._asdict()), result)
            record.update(
                {
                    "learning_rate": float(fit.learning_rate),
                    "weight_decay": float(fit.weight_decay),
                    "hyperparameter_sha256": str(fit.hyperparameter_sha256),
                }
            )
            status_rows.append(record)
            if not result.history.empty:
                history = result.history.assign(
                    context_id=context_id,
                    experiment_id=str(context_row.experiment_id),
                    stage="inner_selection",
                    candidate_id=str(fit.candidate_id),
                    selection_unit_id=str(fit.selection_unit_id),
                    seed=int(fit.seed),
                )
                history_frames.append(history)
            if not result.predictions.empty:
                prediction = result.predictions.assign(
                    context_id=context_id,
                    experiment_id=str(context_row.experiment_id),
                    candidate_id=str(fit.candidate_id),
                    selection_unit_id=str(fit.selection_unit_id),
                    seed=int(fit.seed),
                )
                validation_frames.append(prediction)
            del result
        fit_status = pd.DataFrame(status_rows)
        winner, selection_trace = select_candidate(
            fit_status,
            candidates,
            int(context_row.selection_unit_count),
        )
        selection_trace.insert(0, "context_id", context_id)
        selection_trace.insert(1, "experiment_id", str(context_row.experiment_id))
        if winner is not None:
            selected_candidate = str(winner.candidate_id)
            outer_values, outer_metadata = _role(
                dataset,
                roles,
                str(fits[fits.stage.eq("final_selected_refit")].iloc[0].fit_role_id),
            )
            test_values, test_metadata = _role(
                dataset,
                roles,
                str(fits[fits.stage.eq("final_selected_refit")].iloc[0].test_role_id),
            )
            validation_predictions = pd.concat(validation_frames, ignore_index=True)
            selected_validation = validation_predictions[
                validation_predictions.candidate_id.eq(selected_candidate)
            ]
            for fit in fits[fits.stage.eq("final_selected_refit")].itertuples(index=False):
                seed_status = fit_status[
                    fit_status.candidate_id.eq(selected_candidate)
                    & fit_status.seed.eq(int(fit.seed))
                    & fit_status.status.eq("complete")
                ]
                seed_predictions = selected_validation[
                    selected_validation.seed.eq(int(fit.seed))
                ]
                if len(seed_status) != int(context_row.selection_unit_count):
                    continue
                selected_epoch = int(
                    np.clip(
                        np.rint(seed_status.best_epoch.astype(float).median()),
                        int(optimization["minimum_epochs"]),
                        int(optimization["maximum_epochs"]),
                    )
                )
                result, checkpoint, calibration = train_fixed_epochs(
                    fit_id=str(fit.fit_id),
                    train_values=outer_values,
                    train_metadata=outer_metadata,
                    test_values=test_values,
                    test_metadata=test_metadata,
                    learning_rate=float(winner.learning_rate),
                    weight_decay=float(winner.weight_decay),
                    batch_size=int(fit.batch_size),
                    seed=int(fit.seed),
                    epochs=selected_epoch,
                    gradient_clip_norm=float(optimization["gradient_clip_norm"]),
                    device=device,
                    calibration_predictions=seed_predictions,
                )
                record = status_record(
                    pd.Series(fit._asdict()),
                    result,
                    selected_candidate_id=selected_candidate,
                )
                record.update(
                    {
                        "learning_rate": float(winner.learning_rate),
                        "weight_decay": float(winner.weight_decay),
                        "hyperparameter_sha256": str(winner.hyperparameter_sha256),
                    }
                )
                status_rows.append(record)
                if not result.history.empty:
                    history_frames.append(
                        result.history.assign(
                            context_id=context_id,
                            experiment_id=str(context_row.experiment_id),
                            stage="final_selected_refit",
                            candidate_id=selected_candidate,
                            selection_unit_id="outer_fit",
                            seed=int(fit.seed),
                        )
                    )
                if not result.predictions.empty:
                    test_frames.append(
                        result.predictions.assign(
                            context_id=context_id,
                            experiment_id=str(context_row.experiment_id),
                            candidate_id=selected_candidate,
                            seed=int(fit.seed),
                            domain=str(context_row.domain),
                            held_instrument=str(context_row.held_instrument),
                            outer_repeat=int(context_row.outer_repeat),
                            outer_fold=int(context_row.outer_fold),
                        )
                    )
                if checkpoint is not None:
                    checkpoint_dir = lease.temporary_dir / "checkpoints"
                    checkpoint_dir.mkdir(exist_ok=True)
                    (checkpoint_dir / f"{fit.fit_id}.pt").write_bytes(checkpoint)
                if calibration is not None:
                    calibration_rows.append(
                        {
                            "fit_id": str(fit.fit_id),
                            "context_id": context_id,
                            "seed": int(fit.seed),
                            "temperature": calibration.temperature,
                            "calibration_observations": calibration.observations,
                            "calibration_masters": calibration.masters,
                            "calibration_master_sha256": calibration.fit_master_uid_sha256,
                            "optimizer_success": calibration.optimizer_success,
                            "optimizer_objective": calibration.optimizer_objective,
                            "calibration_state_sha256": calibration.state_sha256,
                        }
                    )
        all_status = pd.DataFrame(status_rows)
        all_history = (
            pd.concat(history_frames, ignore_index=True)
            if history_frames
            else pd.DataFrame()
        )
        all_validation = (
            pd.concat(validation_frames, ignore_index=True)
            if validation_frames
            else pd.DataFrame()
        )
        all_test = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
        all_status.to_csv(lease.temporary_dir / "fit_status.csv", index=False)
        selection_trace.to_csv(lease.temporary_dir / "selection_trace.csv", index=False)
        pd.DataFrame(calibration_rows).to_csv(
            lease.temporary_dir / "calibration_status.csv", index=False
        )
        all_history.to_parquet(
            lease.temporary_dir / "training_histories.parquet",
            index=False,
            compression="zstd",
        )
        all_validation.to_parquet(
            lease.temporary_dir / "source_validation_predictions.parquet",
            index=False,
            compression="zstd",
        )
        all_test.to_parquet(
            lease.temporary_dir / "test_predictions.parquet",
            index=False,
            compression="zstd",
        )
        descriptor = {
            "schema_version": "nato-sers-p04-context-shard-v1",
            "shard_index": shard_index,
            "context_id": context_id,
            "experiment_id": str(context_row.experiment_id),
            "protected_state_sha256": context.protected_state_sha256,
            "device": str(device),
            "planned_fits": len(fits),
            "executed_fits": len(all_status),
            "terminal_statuses": all_status.status.value_counts().sort_index().to_dict(),
            "selected_candidate_id": (
                None if winner is None else str(winner.candidate_id)
            ),
            "source_validation_prediction_rows": len(all_validation),
            "test_prediction_rows": len(all_test),
            "checkpoint_count": len(list(lease.temporary_dir.glob("checkpoints/*.pt"))),
        }
        (lease.temporary_dir / "shard_descriptor.json").write_bytes(
            canonical_json_bytes(descriptor, pretty=True)
        )
        return store.commit(lease), lease.action
    except Exception:
        store.abort(lease, reason="p04_context_execution_failed")
        raise


def execute_shard(
    *, artifact_root: Path, project_root: Path, shard_index: int
) -> tuple[Path, str, str]:
    context = execution_context(artifact_root=artifact_root, project_root=project_root)
    tables = _tables(context)
    dataset = _dataset(context)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_torch(device)
    path, action = _execute_context(
        context=context,
        tables=tables,
        dataset=dataset,
        shard_index=shard_index,
        device=device,
    )
    return path, action, context.run_id


def execute_batch(
    *,
    artifact_root: Path,
    project_root: Path,
    phase: str,
    start_index: int = 0,
    stop_index: int | None = None,
) -> dict[str, Any]:
    if phase not in {"development", "held_evaluation", "all"}:
        raise ValueError("P04 phase must be development, held_evaluation, or all.")
    context = execution_context(artifact_root=artifact_root, project_root=project_root)
    tables = _tables(context)
    dataset = _dataset(context)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_torch(device)
    shards = tables["shard_manifest"].sort_values("shard_index")
    if phase != "all":
        shards = shards[shards.phase_gate.eq(phase)]
    maximum = int(shards.shard_index.max()) + 1 if len(shards) else 0
    stop = maximum if stop_index is None else stop_index
    indices = [
        int(value)
        for value in shards.shard_index
        if int(value) >= start_index and int(value) < stop
    ]
    actions: dict[str, int] = {}
    started = time.perf_counter()
    for position, index in enumerate(indices, start=1):
        _, action = _execute_context(
            context=context,
            tables=tables,
            dataset=dataset,
            shard_index=index,
            device=device,
        )
        actions[action] = actions.get(action, 0) + 1
        if position % 10 == 0 or position == len(indices):
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "completed": position,
                        "requested": len(indices),
                        "last_shard_index": index,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        "status": "pass",
        "run_id": context.run_id,
        "phase": phase,
        "device": str(device),
        "processed": len(indices),
        "actions": actions,
        "elapsed_seconds": time.perf_counter() - started,
    }


def freeze_development(*, artifact_root: Path, project_root: Path) -> dict[str, Any]:
    context = execution_context(artifact_root=artifact_root, project_root=project_root)
    tables = _tables(context)
    expected = tables["context_registry"][
        tables["context_registry"].phase_gate.eq("development")
    ].sort_values("shard_index")
    store = P03ShardStore(run_root=context.execution_root / "contexts")
    validation = store.validation_table(
        {
            int(row.shard_index): context.protected_state_sha256
            for row in expected.itertuples(index=False)
        }
    )
    if not all(row["valid"] for row in validation):
        missing = [row["shard_id"] for row in validation if not row["valid"]]
        raise RuntimeError(f"P04 development is incomplete for shards: {missing}")
    statuses = []
    traces = []
    for row in expected.itertuples(index=False):
        shard = context.execution_root / "contexts/shards" / f"shard-{row.shard_index:06d}"
        statuses.append(pd.read_csv(shard / "fit_status.csv", low_memory=False))
        traces.append(pd.read_csv(shard / "selection_trace.csv", low_memory=False))
    fit_status = pd.concat(statuses, ignore_index=True)
    selection = fit_status[fit_status.stage.eq("inner_selection")]
    finite_fraction = float(selection.status.eq("complete").mean())
    threshold = float(context.contract["gates"]["g2_finite_fit_fraction_minimum"])
    selected = pd.concat(traces, ignore_index=True)
    checks = {
        "all_60_development_contexts_complete": len(expected) == 60,
        "all_planned_selection_fits_recorded": len(selection)
        == int(
            tables["fit_manifest"]
            .query("experiment_id == 'EXP-N00-DEV' and stage == 'inner_selection'")
            .shape[0]
        ),
        "finite_fit_fraction_at_least_95_percent": finite_fraction >= threshold,
        "every_context_has_one_selection": bool(
            selected.groupby("context_id").selected.sum().eq(1).all()
        ),
        "no_held_evaluation_shard_required_for_gate": True,
        "no_outer_test_metric_used_for_gate": True,
        "no_p03_or_p13_outcome_used_for_gate": True,
    }
    payload = {
        "schema_version": "nato-sers-p04-development-freeze-v1",
        "run_id": context.run_id,
        "protected_state_sha256": context.protected_state_sha256,
        "g2_status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "selection_fit_count": len(selection),
        "finite_selection_fit_count": int(selection.status.eq("complete").sum()),
        "finite_selection_fit_fraction": finite_fraction,
        "diagnostic_counts": selection.diagnostic.fillna("failure").value_counts().to_dict(),
        "selected_candidate_counts": (
            selected[selected.selected].candidate_id.value_counts().sort_index().to_dict()
        ),
        "selection_evidence_sha256": sha256_value(
            selected[
                [
                    "context_id",
                    "candidate_id",
                    "mean_balanced_accuracy",
                    "worst_balanced_accuracy",
                    "mean_macro_f1",
                    "selected",
                ]
            ].to_dict("records")
        ),
        "claim_boundary": (
            "G2 reflects only source inner optimization validity; development outer-test "
            "performance and P03/P13 held outcomes were not consulted"
        ),
    }
    payload["file_sha256"] = sha256_value(payload)
    path = _development_freeze_path(context)
    content = canonical_json_bytes(payload, pretty=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError("A conflicting P04 development freeze already exists.")
    path.write_bytes(content)
    if payload["g2_status"] != "pass":
        raise RuntimeError("P04 development failed G2; held evaluation remains blocked.")
    return payload
