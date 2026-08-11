"""Protected-input loader and execution boundary for P03 selection shards."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from atlas_sers.evaluation.p03_analysis import build_p03_diagnostic_tables
from atlas_sers.evaluation.p03_collect import (
    build_all_selection_results,
    collect_selection_evidence,
    load_selection_predictions,
)
from atlas_sers.evaluation.p03_controls import metadata_control_candidate_registry
from atlas_sers.evaluation.p03_finalize import (
    execute_permutation_outer_refit,
    execute_prior_control_outer_refit,
)
from atlas_sers.evaluation.p03_metrics import (
    IDENTITY_COLUMNS,
    build_expected_endpoint_registry,
    build_p03_metric_tables,
)
from atlas_sers.evaluation.p03_outer import (
    OuterExecutionResult,
    execute_fixed_prior_t1_outer,
    execute_selected_procedure_outer,
)
from atlas_sers.evaluation.p03_plan import assign_selection_shards
from atlas_sers.evaluation.p03_reconcile import collect_outer_evidence
from atlas_sers.evaluation.p03_reporting import (
    build_p04_comparator_freeze,
    build_selected_model_specs,
    comparator_json_bytes,
    render_p03_report,
    render_p04_handoff,
)
from atlas_sers.evaluation.p03_results import (
    P03ResultIdentity,
    normalize_p03_predictions,
    validate_p03_prediction_schema,
)
from atlas_sers.evaluation.p03_runtime import TERMINAL_STATUSES, P03Dataset
from atlas_sers.evaluation.p03_worker import execute_selection_shard
from atlas_sers.governance.canonical import (
    canonical_json_bytes,
    hash_relative_files,
    sha256_file,
    sha256_value,
)
from atlas_sers.governance.p01 import validate_latest_p01
from atlas_sers.governance.p02 import validate_latest_p02
from atlas_sers.governance.p03 import validate_latest_p03_plan
from atlas_sers.governance.p03_store import P03ShardStore
from atlas_sers.governance.provenance import capture_provenance
from atlas_sers.visualization.p03_figures import (
    FIGURE_SLUGS,
    build_p03_figure_definitions,
    generate_p03_figures,
)

EXPECTED_EXECUTABLE_OUTER_RUNS = 8082
EXPECTED_FINAL_ENDPOINTS = 8142
EXPECTED_FIT_MANIFEST_ROWS = 260356


def _native_thread_limit(context: P03ExecutionContext) -> int:
    limit = int(context.p03_contract["resource_estimation"]["native_math_threads_per_worker"])
    if limit != 1:
        raise RuntimeError("P03 requires exactly one native math thread per worker.")
    return limit


@dataclass(frozen=True)
class P03ExecutionContext:
    execution_run_id: str
    execution_protected_state_sha256: str
    plan_run_id: str
    plan_run_dir: Path
    code_sha256: str
    config_sha256: str
    input_sha256: str
    dataset: P03Dataset
    manifest: pd.DataFrame
    p02_tables: dict[str, pd.DataFrame]
    fit_manifest: pd.DataFrame
    candidate_registry: pd.DataFrame
    coral_candidate_registry: pd.DataFrame
    control_registry: pd.DataFrame
    selection_shard_manifest: pd.DataFrame
    p03_contract: dict[str, Any]
    scientific_fitting_authorized: bool


def _repository_root(project_root: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip()).resolve()


def _latest_run(artifact_root: Path, phase: str) -> tuple[dict[str, Any], Path]:
    latest = json.loads((artifact_root / phase / "LATEST.json").read_text())
    return latest, artifact_root / phase / "runs" / str(latest["run_id"])


def _verify_current_state(
    *, project_root: Path, artifact_root: Path, protected_state: dict[str, Any]
) -> None:
    configuration = protected_state["configuration_manifest"]
    plan_root = project_root / "plan"
    current_configuration = hash_relative_files(
        plan_root, [plan_root / relative for relative in configuration]
    )
    if current_configuration != configuration:
        raise RuntimeError("Current P03 configuration differs from the no-fit plan.")
    provenance = capture_provenance(
        repository_root=_repository_root(project_root),
        project_root=project_root,
        artifact_root=artifact_root,
    )
    identity = protected_state["run_identity"]
    if provenance["repository"]["atlas_public_tree_sha256"] != identity["code_sha256"]:
        raise RuntimeError("Current ATLAS source tree differs from the no-fit plan.")
    if sha256_value(configuration) != identity["config_sha256"]:
        raise RuntimeError("No-fit plan configuration identity is internally inconsistent.")


def _verify_class_vocabularies(
    manifest: pd.DataFrame, p03_contract: dict[str, Any]
) -> None:
    vocabularies = p03_contract["class_vocabulary"]
    for station in ("cwa", "pills", "surfaces"):
        observed = set(
            manifest.loc[manifest.station == station, "target_analyte"].astype(str)
        )
        declared = {str(value) for value in vocabularies[station]}
        if observed != declared:
            raise RuntimeError(f"Frozen P03 class vocabulary differs for station {station}.")
    shared = {"4_ANPP", "benzyl_fentanyl"}
    for task in ("T2-PS", "T2-SP"):
        if set(str(value) for value in vocabularies[task]) != shared:
            raise RuntimeError(f"Frozen P03 class vocabulary differs for task {task}.")


def load_latest_p03_execution_context(
    *,
    project_root: Path,
    artifact_root: Path,
    require_authorized: bool = True,
) -> P03ExecutionContext:
    """Load only a current, hash-matching P03 plan and its immutable inputs."""

    project_root = project_root.resolve()
    artifact_root = artifact_root.resolve()
    p01_validation = validate_latest_p01(artifact_root)
    p02_validation = validate_latest_p02(artifact_root)
    p03_validation = validate_latest_p03_plan(artifact_root)
    if any(
        report["status"] != "pass"
        for report in (p01_validation, p02_validation, p03_validation)
    ):
        raise RuntimeError("P03 execution prerequisite validation failed.")
    p01_latest, p01_run = _latest_run(artifact_root, "p01")
    p02_latest, p02_run = _latest_run(artifact_root, "p02")
    p03_latest, plan_run = _latest_run(artifact_root, "p03plan")
    input_hashes = json.loads((plan_run / "input_hashes.json").read_text())
    protected_state = json.loads((plan_run / "protected_state.json").read_text())
    report = json.loads((plan_run / "P03_PLAN_VALIDATION_REPORT.json").read_text())
    budget = json.loads((plan_run / "budget_gate.json").read_text())
    p03_contract = json.loads(
        (project_root / "plan" / "contracts" / "p03_governance_contract.json").read_text()
    )
    if input_hashes["p01"]["run_id"] != p01_latest["run_id"]:
        raise RuntimeError("P03 plan no longer references the latest immutable P01 run.")
    if input_hashes["p02"]["run_id"] != p02_latest["run_id"]:
        raise RuntimeError("P03 plan no longer references the latest immutable P02 run.")
    _verify_current_state(
        project_root=project_root,
        artifact_root=artifact_root,
        protected_state=protected_state,
    )
    authorized = bool(
        report["scientific_fitting_authorized"]
        and budget["scientific_fitting_authorized"]
        and p03_latest["scientific_fitting_authorized"]
        and p03_contract["planning"]["model_fitting_authorized"]
    )
    if require_authorized and not authorized:
        reasons = ",".join(str(value) for value in budget["blocking_reasons"])
        raise PermissionError(f"P03 scientific fitting remains unauthorized: {reasons}")
    representation_path = p01_run / "representations" / "R_MIN_400_1800.npz"
    if sha256_file(representation_path) != input_hashes["p01"]["representation_npz_sha256"]:
        raise RuntimeError("Frozen P01 representation file differs from the P03 plan.")
    manifest_path = p01_run / "primary_manifest.csv"
    if sha256_file(manifest_path) != input_hashes["p01"]["primary_manifest_sha256"]:
        raise RuntimeError("Frozen P01 manifest differs from the P03 plan.")
    manifest = pd.read_csv(manifest_path, low_memory=False)
    with np.load(representation_path, allow_pickle=False) as archive:
        dataset = P03Dataset.from_frozen_representation(
            intensity=archive["intensity"],
            representation_uids=archive["observation_uid"],
            metadata=manifest,
        )
    p02_tables = {
        name: pd.read_csv(p02_run / name, low_memory=False)
        for name in input_hashes["p02"]["protected_payloads"]
        if name.endswith(".csv")
    }
    _verify_class_vocabularies(manifest, p03_contract)
    fit_manifest = pd.read_csv(plan_run / "fit_manifest.csv", low_memory=False)
    candidates = pd.read_csv(plan_run / "candidate_registry.csv", low_memory=False)
    coral_candidates = pd.read_csv(
        plan_run / "coral_candidate_registry.csv", low_memory=False
    )
    control_candidates = pd.read_csv(plan_run / "control_registry.csv", low_memory=False)
    selection_shards = pd.read_csv(plan_run / "selection_shard_manifest.csv")
    execution_state = {
        "schema_version": "p03-execution-protected-state-v1",
        "plan_run_id": p03_latest["run_id"],
        "plan_protected_state_sha256": p03_latest["protected_state_sha256"],
        "p01_protected_state_sha256": p01_latest["protected_state_sha256"],
        "p02_protected_state_sha256": p02_latest["protected_state_sha256"],
        "fit_manifest_sha256": sha256_file(plan_run / "fit_manifest.csv"),
        "candidate_registry_sha256": sha256_file(plan_run / "candidate_registry.csv"),
        "coral_candidate_registry_sha256": sha256_file(
            plan_run / "coral_candidate_registry.csv"
        ),
        "control_registry_sha256": sha256_file(plan_run / "control_registry.csv"),
        "selection_shard_manifest_sha256": sha256_file(
            plan_run / "selection_shard_manifest.csv"
        ),
        "scientific_fitting_authorized": authorized,
    }
    execution_hash = sha256_value(execution_state)
    plan_identity = protected_state["run_identity"]
    return P03ExecutionContext(
        execution_run_id=f"P03-{execution_hash[:24]}",
        execution_protected_state_sha256=execution_hash,
        plan_run_id=str(p03_latest["run_id"]),
        plan_run_dir=plan_run,
        code_sha256=str(plan_identity["code_sha256"]),
        config_sha256=str(plan_identity["config_sha256"]),
        input_sha256=str(plan_identity["input_sha256"]),
        dataset=dataset,
        manifest=manifest,
        p02_tables=p02_tables,
        fit_manifest=fit_manifest,
        candidate_registry=candidates,
        coral_candidate_registry=coral_candidates,
        control_registry=control_candidates,
        selection_shard_manifest=selection_shards,
        p03_contract=p03_contract,
        scientific_fitting_authorized=authorized,
    )


def _execute_selection_shard_with_context(
    *,
    context: P03ExecutionContext,
    artifact_root: Path,
    shard_id: int,
    assigned_selection_rows: pd.DataFrame | None = None,
) -> tuple[Path, str, str]:
    """Resolve one planned selection shard, execute it, and commit atomically."""

    planned = context.selection_shard_manifest[
        context.selection_shard_manifest.selection_shard_id == shard_id
    ]
    if len(planned) != 1:
        raise ValueError(f"Selection shard {shard_id} is not in the frozen P03 plan.")
    row = planned.iloc[0]
    target = int(context.p03_contract["resource_estimation"]["shard_target_fits"])
    if assigned_selection_rows is None:
        direct = context.fit_manifest[context.fit_manifest.accounting.eq("new_fit")]
        assigned = assign_selection_shards(direct, target=target)
    else:
        assigned = assigned_selection_rows
    shard_rows = assigned[assigned.selection_shard_id == shard_id].drop(
        columns=["selection_shard_id", "selection_kind"]
    )
    if shard_rows.empty:
        raise ValueError(f"Selection shard {shard_id} is not in the frozen P03 plan.")
    if (
        len(shard_rows) != int(row.fit_count)
        or str(shard_rows.fit_id.iloc[0]) != str(row.first_fit_id)
        or str(shard_rows.fit_id.iloc[-1]) != str(row.last_fit_id)
        or sha256_value(sorted(shard_rows.fit_id.astype(str))) != str(row.fit_id_sha256)
    ):
        raise RuntimeError("Reconstructed selection shard differs from the no-fit plan.")
    selection_kind = str(row.selection_kind)
    if selection_kind == "standard":
        candidate_registry = context.candidate_registry
        allowed_stages = {str(shard_rows.stage.iloc[0])}
    elif selection_kind == "source_covariance":
        if context.p03_contract["coral"]["status"] != (
            "resolved_source_to_source_covariance_augmentation_v1"
        ):
            raise PermissionError("P03 C12 source-only covariance method remains unresolved.")
        candidate_registry = context.coral_candidate_registry
        allowed_stages = {"inner_source_coral_selection"}
    elif selection_kind == "metadata_control":
        if context.p03_contract["negative_controls"]["status"] != (
            "resolved_p03_negative_controls_v1"
        ):
            raise PermissionError("P03 negative-control scope remains unresolved.")
        candidate_registry = metadata_control_candidate_registry(
            context.control_registry
        )
        allowed_stages = {"metadata_inner_selection"}
    else:
        raise RuntimeError(f"Unknown P03 selection kind: {selection_kind}")
    store = P03ShardStore(
        run_root=artifact_root / "p03" / "runs" / context.execution_run_id / "selection"
    )
    path, action = execute_selection_shard(
        store=store,
        shard_id=shard_id,
        protected_state_sha256=context.execution_protected_state_sha256,
        dataset=context.dataset,
        fit_rows=shard_rows,
        candidate_registry=candidate_registry,
        manifest=context.manifest,
        p02_tables=context.p02_tables,
        p03_contract=context.p03_contract,
        scientific_fitting_authorized=context.scientific_fitting_authorized,
        allowed_stages=allowed_stages,
        native_thread_limit=_native_thread_limit(context),
    )
    return path, action, context.execution_run_id


def execute_latest_selection_shard(
    *, project_root: Path, artifact_root: Path, shard_id: int
) -> tuple[Path, str, str]:
    """Load the frozen context once and execute one selection shard."""

    context = load_latest_p03_execution_context(
        project_root=project_root,
        artifact_root=artifact_root,
        require_authorized=True,
    )
    return _execute_selection_shard_with_context(
        context=context,
        artifact_root=artifact_root,
        shard_id=shard_id,
    )


def aggregate_latest_selection(
    *, project_root: Path, artifact_root: Path
) -> tuple[Path, str, str]:
    """Validate every selection shard and atomically freeze all selections."""

    context = load_latest_p03_execution_context(
        project_root=project_root,
        artifact_root=artifact_root,
        require_authorized=True,
    )
    run_root = artifact_root / "p03" / "runs" / context.execution_run_id
    target = int(context.p03_contract["resource_estimation"]["shard_target_fits"])
    evidence = collect_selection_evidence(
        selection_run_root=run_root / "selection",
        fit_manifest=context.fit_manifest,
        selection_shard_manifest=context.selection_shard_manifest,
        protected_state_sha256=context.execution_protected_state_sha256,
        shard_target_fits=target,
    )
    expected_runs = pd.read_csv(
        context.plan_run_dir / "expected_run_registry.csv", low_memory=False
    )
    results = build_all_selection_results(
        fit_manifest=context.fit_manifest,
        selection_unit_metrics=evidence.selection_unit_metrics,
        expected_run_registry=expected_runs,
        candidate_registry=context.candidate_registry,
        coral_candidate_registry=context.coral_candidate_registry,
        control_registry=context.control_registry,
    )
    store = P03ShardStore(run_root=run_root / "selection_aggregation")
    lease = store.begin(
        shard_id=0,
        protected_state_sha256=context.execution_protected_state_sha256,
    )
    if lease.action == "verified_skip":
        return lease.final_dir, lease.action, context.execution_run_id
    if lease.temporary_dir is None:
        raise RuntimeError("Selection aggregation lease has no temporary directory.")
    try:
        results.selections.to_csv(
            lease.temporary_dir / "selection_summary.csv", index=False
        )
        results.traces.to_parquet(
            lease.temporary_dir / "selection_trace.parquet",
            index=False,
            compression="zstd",
        )
        evidence.shard_validation.to_csv(
            lease.temporary_dir / "selection_shard_validation.csv", index=False
        )
        failure_counts = (
            evidence.fit_status.status.astype(str).value_counts().sort_index().to_dict()
        )
        descriptor = {
            "schema_version": "p03-selection-aggregation-v1",
            "execution_run_id": context.execution_run_id,
            "protected_state_sha256": context.execution_protected_state_sha256,
            "planned_selection_fit_count": len(evidence.fit_status),
            "selection_fit_id_sha256": sha256_value(
                sorted(evidence.fit_status.fit_id.astype(str))
            ),
            "selection_outer_run_count": len(results.selections),
            "selection_outer_run_id_sha256": sha256_value(
                sorted(results.selections.outer_run_id.astype(str))
            ),
            "selection_trace_rows": len(results.traces),
            "terminal_status_counts": failure_counts,
        }
        (lease.temporary_dir / "selection_aggregation_descriptor.json").write_bytes(
            canonical_json_bytes(descriptor, pretty=True)
        )
        return store.commit(lease), lease.action, context.execution_run_id
    except Exception:
        store.abort(lease, reason="selection_aggregation_failed")
        raise


def _selection_freeze(
    context: P03ExecutionContext, *, artifact_root: Path
) -> pd.DataFrame:
    root = (
        artifact_root
        / "p03"
        / "runs"
        / context.execution_run_id
        / "selection_aggregation"
    )
    store = P03ShardStore(run_root=root)
    validation = store.validation_table(
        {0: context.execution_protected_state_sha256}
    )
    if len(validation) != 1 or not bool(validation[0]["valid"]):
        raise RuntimeError("P03 outer execution requires a valid selection freeze.")
    summary = pd.read_csv(
        store.shards / store._name(0) / "selection_summary.csv",  # noqa: SLF001
        low_memory=False,
    )
    if len(summary) != 2302 or not summary.outer_run_id.astype(str).is_unique:
        raise RuntimeError("P03 selection freeze has invalid outer-run coverage.")
    return summary


def _selected_source_manifest(
    *,
    context: P03ExecutionContext,
    selection: pd.Series,
) -> pd.DataFrame:
    experiment = str(selection.experiment_id)
    source_outer = (
        str(selection.source_c09_outer_run_id)
        if experiment == "EXP-C10-T3"
        else str(selection.outer_run_id)
    )
    stage = {
        "EXP-C11-T2": "training_station_inner_selection",
        "EXP-C12-CORAL": "inner_source_coral_selection",
        "EXP-C09-CONTROL-META": "metadata_inner_selection",
    }.get(experiment, "inner_selection")
    manifest = context.fit_manifest[
        context.fit_manifest.outer_run_id.astype(str).eq(source_outer)
        & context.fit_manifest.stage.astype(str).eq(stage)
        & context.fit_manifest.accounting.astype(str).eq("new_fit")
    ]
    if manifest.empty:
        raise RuntimeError("Selected outer run has no source selection manifest.")
    return manifest


def _selected_registry(
    context: P03ExecutionContext, *, experiment_id: str
) -> pd.DataFrame:
    if experiment_id == "EXP-C12-CORAL":
        return context.coral_candidate_registry
    if experiment_id == "EXP-C09-CONTROL-META":
        return metadata_control_candidate_registry(context.control_registry)
    return context.candidate_registry


def _execute_outer_dependency(
    *,
    context: P03ExecutionContext,
    run_row: pd.Series,
    outer_fit_rows: pd.DataFrame,
    selection_summary: pd.DataFrame,
    artifact_root: Path,
) -> OuterExecutionResult:
    experiment = str(run_row.experiment_id)
    if experiment == "EXP-C00-T1":
        return execute_fixed_prior_t1_outer(
            dataset=context.dataset,
            final_fit_rows=outer_fit_rows,
            candidate_registry=context.candidate_registry,
            manifest=context.manifest,
            p02_tables=context.p02_tables,
            p03_contract=context.p03_contract,
            scientific_fitting_authorized=True,
        )
    if experiment == "EXP-C09-CONTROL-PERM":
        real = selection_summary[
            selection_summary.outer_run_id.astype(str).eq(
                str(run_row.source_outer_run_id)
            )
        ]
        if len(real) != 1:
            raise RuntimeError("Permutation control lacks its frozen real C09 selection.")
        if str(real.status.iloc[0]) != "complete":
            statuses = pd.DataFrame(
                [
                    {
                        "fit_id": str(row.fit_id),
                        "outer_run_id": str(row.outer_run_id),
                        "status": "excluded_by_protocol",
                        "reason_code": "real_C09_selection_dependency_not_complete",
                        "model_id": str(row.model_id),
                        "candidate_id": str(row.candidate_id),
                        "seed": row.seed,
                        "fit_uid_sha256": str(row.fit_uid_sha256),
                    }
                    for row in outer_fit_rows.itertuples(index=False)
                ]
            )
            return OuterExecutionResult(
                statuses, None, pd.DataFrame(), pd.DataFrame()
            )
        outcome = execute_permutation_outer_refit(
            dataset=context.dataset,
            permutation_fit_rows=outer_fit_rows,
            real_selection=real.iloc[0],
            candidate_registry=context.candidate_registry,
            control_registry=context.control_registry,
            manifest=context.manifest,
            p02_tables=context.p02_tables,
            p03_contract=context.p03_contract,
            scientific_fitting_authorized=True,
        )
        return OuterExecutionResult(
            outcome.fit_status,
            None,
            pd.DataFrame(),
            outcome.final_predictions,
        )
    if experiment == "EXP-C09-CONTROL-PRIOR":
        if len(outer_fit_rows) != 1:
            raise RuntimeError("Prior control outer run does not have exactly one fit row.")
        outcome = execute_prior_control_outer_refit(
            dataset=context.dataset,
            prior_fit_row=outer_fit_rows.iloc[0],
            control_registry=context.control_registry,
            manifest=context.manifest,
            p02_tables=context.p02_tables,
            p03_contract=context.p03_contract,
            scientific_fitting_authorized=True,
        )
        return OuterExecutionResult(
            outcome.fit_status,
            None,
            pd.DataFrame(),
            outcome.final_predictions,
        )
    selected = selection_summary[
        selection_summary.outer_run_id.astype(str).eq(str(run_row.outer_run_id))
    ]
    if len(selected) != 1:
        raise RuntimeError("Selected P03 outer run lacks exactly one selection row.")
    selection = selected.iloc[0]
    registry = _selected_registry(context, experiment_id=experiment)
    selection_manifest = _selected_source_manifest(
        context=context, selection=selection
    )
    calibration_rows = outer_fit_rows[
        outer_fit_rows.stage.astype(str).str.contains("calibration")
    ]
    needs_cache = calibration_rows.empty or calibration_rows.accounting.astype(str).eq(
        "cache_reuse"
    ).all()
    cached_predictions = pd.DataFrame()
    if needs_cache and str(selection.status) == "complete":
        selected_id = str(selection.selected_candidate_id)
        fit_ids = set(
            selection_manifest.loc[
                selection_manifest.candidate_id.astype(str).eq(selected_id),
                "fit_id",
            ].astype(str)
        )
        cached_predictions = load_selection_predictions(
            selection_run_root=(
                artifact_root
                / "p03"
                / "runs"
                / context.execution_run_id
                / "selection"
            ),
            fit_manifest=context.fit_manifest,
            fit_ids=fit_ids,
            shard_target_fits=int(
                context.p03_contract["resource_estimation"]["shard_target_fits"]
            ),
        )
    return execute_selected_procedure_outer(
        dataset=context.dataset,
        outer_fit_rows=outer_fit_rows,
        selection=selection,
        candidate_registry=registry,
        selection_fit_manifest=selection_manifest,
        cached_selection_predictions=cached_predictions,
        manifest=context.manifest,
        p02_tables=context.p02_tables,
        p03_contract=context.p03_contract,
        scientific_fitting_authorized=True,
        t2_first_repeat_only=experiment == "EXP-C11-T2",
    )


def _execute_outer_index_with_context(
    *,
    context: P03ExecutionContext,
    artifact_root: Path,
    outer_index: int,
    executable_runs: pd.DataFrame | None = None,
    selection_summary: pd.DataFrame | None = None,
) -> tuple[Path, str, str]:
    """Execute one dependency-resolved outer task and commit it atomically."""

    if executable_runs is None:
        expected_runs = pd.read_csv(
            context.plan_run_dir / "expected_run_registry.csv", low_memory=False
        )
        executable = expected_runs[
            ~expected_runs.execution_status.astype(str).eq(
                "manifest_only_exploratory"
            )
        ].reset_index(drop=True)
    else:
        executable = executable_runs
    if outer_index < 0 or outer_index >= len(executable):
        raise ValueError(
            f"Outer index {outer_index} is outside the frozen range 0..{len(executable) - 1}."
        )
    run_row = executable.iloc[outer_index]
    outer_fit_rows = context.fit_manifest[
        context.fit_manifest.outer_run_id.astype(str).eq(str(run_row.outer_run_id))
    ].copy()
    if outer_fit_rows.empty:
        raise RuntimeError("Executable outer run has no fit-manifest rows.")
    run_root = artifact_root / "p03" / "runs" / context.execution_run_id
    store = P03ShardStore(run_root=run_root / "outer")
    lease = store.begin(
        shard_id=outer_index,
        protected_state_sha256=context.execution_protected_state_sha256,
    )
    if lease.action == "verified_skip":
        return lease.final_dir, lease.action, context.execution_run_id
    if lease.temporary_dir is None:
        raise RuntimeError("Outer execution lease has no temporary directory.")
    try:
        selections = (
            selection_summary
            if selection_summary is not None
            else _selection_freeze(context, artifact_root=artifact_root)
        )
        native_thread_limit = _native_thread_limit(context)
        with threadpool_limits(limits=native_thread_limit):
            result = _execute_outer_dependency(
                context=context,
                run_row=run_row,
                outer_fit_rows=outer_fit_rows,
                selection_summary=selections,
                artifact_root=artifact_root,
            )
        if result.fit_status.empty or not result.fit_status.fit_id.astype(str).is_unique:
            raise RuntimeError("Outer execution lacks unique terminal fit-status records.")
        selection_stages = {
            "inner_selection",
            "training_station_inner_selection",
            "inner_source_coral_selection",
            "metadata_inner_selection",
        }
        planned_terminal_ids = set(
            outer_fit_rows.loc[
                ~outer_fit_rows.stage.astype(str).isin(selection_stages), "fit_id"
            ].astype(str)
        )
        observed_terminal_ids = set(result.fit_status.fit_id.astype(str))
        if planned_terminal_ids != observed_terminal_ids:
            raise RuntimeError(
                "Outer terminal status coverage differs: "
                f"missing={len(planned_terminal_ids - observed_terminal_ids)} "
                f"extra={len(observed_terminal_ids - planned_terminal_ids)}"
            )
        if not result.fit_status.status.astype(str).isin(TERMINAL_STATUSES).all():
            raise RuntimeError("Outer execution contains a nonterminal fit status.")
        result.fit_status.to_csv(lease.temporary_dir / "fit_status.csv", index=False)
        if result.calibration is not None:
            (lease.temporary_dir / "calibration.json").write_bytes(
                canonical_json_bytes(result.calibration, pretty=True)
            )
        if not result.calibration_predictions.empty:
            result.calibration_predictions.to_parquet(
                lease.temporary_dir / "calibration_predictions.parquet",
                index=False,
                compression="zstd",
            )
        if not result.final_predictions.empty:
            result.final_predictions.to_parquet(
                lease.temporary_dir / "final_predictions.parquet",
                index=False,
                compression="zstd",
            )
        descriptor = {
            "schema_version": "p03-outer-execution-v1",
            "outer_index": outer_index,
            "outer_run_id": str(run_row.outer_run_id),
            "experiment_id": str(run_row.experiment_id),
            "protected_state_sha256": context.execution_protected_state_sha256,
            "terminal_fit_count": len(result.fit_status),
            "terminal_fit_id_sha256": sha256_value(
                sorted(result.fit_status.fit_id.astype(str))
            ),
            "terminal_status_counts": result.fit_status.status.astype(
                str
            ).value_counts().sort_index().to_dict(),
            "calibration_prediction_rows": len(result.calibration_predictions),
            "final_prediction_rows": len(result.final_predictions),
            "native_math_threads_per_worker": native_thread_limit,
        }
        (lease.temporary_dir / "outer_descriptor.json").write_bytes(
            canonical_json_bytes(descriptor, pretty=True)
        )
        return store.commit(lease), lease.action, context.execution_run_id
    except Exception:
        store.abort(lease, reason="outer_execution_failed")
        raise


def execute_latest_outer_index(
    *, project_root: Path, artifact_root: Path, outer_index: int
) -> tuple[Path, str, str]:
    """Load the frozen context once and execute one outer index."""

    context = load_latest_p03_execution_context(
        project_root=project_root,
        artifact_root=artifact_root,
        require_authorized=True,
    )
    return _execute_outer_index_with_context(
        context=context,
        artifact_root=artifact_root,
        outer_index=outer_index,
    )


def _partition_batch_ids(
    ids: list[int],
    *,
    worker_index: int,
    worker_count: int,
    start_index: int,
    stop_index: int | None,
    max_tasks: int | None,
) -> list[int]:
    if worker_count < 1 or worker_index < 0 or worker_index >= worker_count:
        raise ValueError("P03 batch worker index/count is invalid.")
    if start_index < 0 or (stop_index is not None and stop_index < start_index):
        raise ValueError("P03 batch start/stop range is invalid.")
    if max_tasks is not None and max_tasks < 1:
        raise ValueError("P03 batch max-tasks must be positive when supplied.")
    selected = [
        identifier
        for identifier in ids
        if identifier >= start_index
        and (stop_index is None or identifier < stop_index)
        and identifier % worker_count == worker_index
    ]
    return selected[:max_tasks] if max_tasks is not None else selected


def execute_latest_selection_batch(
    *,
    project_root: Path,
    artifact_root: Path,
    worker_index: int = 0,
    worker_count: int = 1,
    start_index: int = 0,
    stop_index: int | None = None,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Execute one deterministic partition of selection shards with one context load."""

    context = load_latest_p03_execution_context(
        project_root=project_root,
        artifact_root=artifact_root,
        require_authorized=True,
    )
    shard_ids = sorted(context.selection_shard_manifest.selection_shard_id.astype(int))
    assigned = _partition_batch_ids(
        shard_ids,
        worker_index=worker_index,
        worker_count=worker_count,
        start_index=start_index,
        stop_index=stop_index,
        max_tasks=max_tasks,
    )
    direct = context.fit_manifest[context.fit_manifest.accounting.eq("new_fit")]
    selection_rows = assign_selection_shards(
        direct,
        target=int(context.p03_contract["resource_estimation"]["shard_target_fits"]),
    )
    actions: list[str] = []
    for shard_id in assigned:
        _, action, _ = _execute_selection_shard_with_context(
            context=context,
            artifact_root=artifact_root,
            shard_id=shard_id,
            assigned_selection_rows=selection_rows,
        )
        actions.append(action)
    return {
        "status": "pass",
        "execution_run_id": context.execution_run_id,
        "batch_kind": "selection",
        "worker_index": worker_index,
        "worker_count": worker_count,
        "assigned_task_count": len(assigned),
        "first_task_id": assigned[0] if assigned else None,
        "last_task_id": assigned[-1] if assigned else None,
        "action_counts": pd.Series(actions, dtype=str).value_counts().sort_index().to_dict(),
    }


def execute_latest_outer_batch(
    *,
    project_root: Path,
    artifact_root: Path,
    worker_index: int = 0,
    worker_count: int = 1,
    start_index: int = 0,
    stop_index: int | None = None,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Execute one deterministic partition of outer indices with frozen dependencies cached."""

    context = load_latest_p03_execution_context(
        project_root=project_root,
        artifact_root=artifact_root,
        require_authorized=True,
    )
    expected_runs = pd.read_csv(
        context.plan_run_dir / "expected_run_registry.csv", low_memory=False
    )
    executable = expected_runs[
        ~expected_runs.execution_status.astype(str).eq(
            "manifest_only_exploratory"
        )
    ].reset_index(drop=True)
    assigned = _partition_batch_ids(
        list(range(len(executable))),
        worker_index=worker_index,
        worker_count=worker_count,
        start_index=start_index,
        stop_index=stop_index,
        max_tasks=max_tasks,
    )
    selections = _selection_freeze(context, artifact_root=artifact_root)
    actions: list[str] = []
    for outer_index in assigned:
        _, action, _ = _execute_outer_index_with_context(
            context=context,
            artifact_root=artifact_root,
            outer_index=outer_index,
            executable_runs=executable,
            selection_summary=selections,
        )
        actions.append(action)
    return {
        "status": "pass",
        "execution_run_id": context.execution_run_id,
        "batch_kind": "outer",
        "worker_index": worker_index,
        "worker_count": worker_count,
        "assigned_task_count": len(assigned),
        "first_task_id": assigned[0] if assigned else None,
        "last_task_id": assigned[-1] if assigned else None,
        "action_counts": pd.Series(actions, dtype=str).value_counts().sort_index().to_dict(),
    }


def aggregate_latest_final(
    *, project_root: Path, artifact_root: Path
) -> tuple[Path, str, str]:
    """Reconcile every terminal fit and atomically reconstruct final P03 metrics."""

    context = load_latest_p03_execution_context(
        project_root=project_root,
        artifact_root=artifact_root,
        require_authorized=True,
    )
    expected_runs = pd.read_csv(
        context.plan_run_dir / "expected_run_registry.csv", low_memory=False
    )
    executable = expected_runs[
        ~expected_runs.execution_status.astype(str).eq(
            "manifest_only_exploratory"
        )
    ]
    if len(context.fit_manifest) != EXPECTED_FIT_MANIFEST_ROWS:
        raise RuntimeError(
            "P03 final aggregation received an amended fit-manifest size without "
            "a corresponding versioned execution update."
        )
    if len(executable) != EXPECTED_EXECUTABLE_OUTER_RUNS:
        raise RuntimeError(
            "P03 final aggregation received an amended executable-run count without "
            "a corresponding versioned execution update."
        )
    run_root = artifact_root / "p03" / "runs" / context.execution_run_id
    target = int(context.p03_contract["resource_estimation"]["shard_target_fits"])
    evidence = collect_outer_evidence(
        p03_run_root=run_root,
        fit_manifest=context.fit_manifest,
        expected_run_registry=expected_runs,
        selection_shard_manifest=context.selection_shard_manifest,
        protected_state_sha256=context.execution_protected_state_sha256,
        shard_target_fits=target,
    )
    expected_endpoints = build_expected_endpoint_registry(
        expected_run_registry=expected_runs,
        fit_manifest=context.fit_manifest,
        candidate_registry=context.candidate_registry,
        class_vocabulary=context.p03_contract["class_vocabulary"],
    )
    if len(expected_endpoints) != EXPECTED_FINAL_ENDPOINTS:
        raise RuntimeError(
            "P03 final endpoint count differs from the frozen 8,142-endpoint design."
        )
    experiment_registry = pd.read_csv(
        project_root / "plan" / "registries" / "experiment_registry.csv",
        low_memory=False,
    )
    normalized_predictions = normalize_p03_predictions(
        evidence.final_predictions,
        primary_manifest=context.manifest,
        experiment_registry=experiment_registry,
        identity=P03ResultIdentity(
            run_id=context.execution_run_id,
            code_sha256=context.code_sha256,
            config_sha256=context.config_sha256,
            input_sha256=context.input_sha256,
            preprocessing_policy_sha256=sha256_file(
                project_root
                / "plan"
                / "contracts"
                / "preprocessing_policy_contract.json"
            ),
        ),
    )
    result_schema = json.loads(
        (project_root / "plan" / "contracts" / "result_schema.json").read_text()
    )
    prediction_schema_report = validate_p03_prediction_schema(
        normalized_predictions,
        result_schema=result_schema,
    )
    metric_tables = build_p03_metric_tables(
        normalized_predictions,
        expected_endpoints=expected_endpoints,
    )
    if len(metric_tables.pooled_metrics) != (
        2 * expected_endpoints[list(IDENTITY_COLUMNS)].drop_duplicates().shape[0]
    ):
        raise RuntimeError("P03 pooled metric endpoint coverage is incomplete.")
    failure_ledger = evidence.fit_status[
        ~evidence.fit_status.status.astype(str).eq("complete")
    ].copy()
    selections = _selection_freeze(context, artifact_root=artifact_root)
    selection_aggregation_store = P03ShardStore(
        run_root=run_root / "selection_aggregation"
    )
    selection_trace = pd.read_parquet(
        selection_aggregation_store.shards
        / selection_aggregation_store._name(0)  # noqa: SLF001
        / "selection_trace.parquet"
    )
    metadata_candidates = metadata_control_candidate_registry(
        context.control_registry
    )
    selected_model_specs = build_selected_model_specs(
        selections=selections,
        candidate_registry=context.candidate_registry,
        coral_candidate_registry=context.coral_candidate_registry,
        metadata_candidate_registry=metadata_candidates,
    )
    diagnostics = build_p03_diagnostic_tables(
        selections=selections,
        selection_traces=selection_trace,
        pooled_metrics=metric_tables.pooled_metrics,
        final_predictions=normalized_predictions,
        master_predictions=metric_tables.master_predictions,
        terminal_fit_ledger=evidence.fit_status,
        fit_manifest=context.fit_manifest,
    )
    figure_definitions = build_p03_figure_definitions(
        pooled_metrics=metric_tables.pooled_metrics,
        diagnostics=diagnostics,
    )
    store = P03ShardStore(run_root=run_root / "final_aggregation")
    lease = store.begin(
        shard_id=0,
        protected_state_sha256=context.execution_protected_state_sha256,
    )
    if lease.action == "verified_skip":
        return lease.final_dir, lease.action, context.execution_run_id
    if lease.temporary_dir is None:
        raise RuntimeError("Final aggregation lease has no temporary directory.")
    try:
        evidence.fit_status.to_parquet(
            lease.temporary_dir / "terminal_fit_ledger.parquet",
            index=False,
            compression="zstd",
        )
        failure_ledger.to_parquet(
            lease.temporary_dir / "failure_ledger.parquet",
            index=False,
            compression="zstd",
        )
        normalized_predictions.to_parquet(
            lease.temporary_dir / "final_predictions.parquet",
            index=False,
            compression="zstd",
        )
        metric_tables.master_predictions.to_parquet(
            lease.temporary_dir / "master_predictions.parquet",
            index=False,
            compression="zstd",
        )
        expected_endpoints.to_parquet(
            lease.temporary_dir / "expected_endpoint_registry.parquet",
            index=False,
            compression="zstd",
        )
        metric_tables.pooled_metrics.to_csv(
            lease.temporary_dir / "pooled_metrics.csv", index=False
        )
        metric_tables.domain_summaries.to_csv(
            lease.temporary_dir / "domain_summaries.csv", index=False
        )
        selected_model_specs.to_csv(
            lease.temporary_dir / "selected_model_specs.csv", index=False
        )
        diagnostic_tables = {
            "selection_frequency.csv": diagnostics.selection_frequency,
            "selection_stability.csv": diagnostics.selection_stability,
            "selection_margins.csv": diagnostics.selection_margins,
            "endpoint_coverage.csv": diagnostics.endpoint_coverage,
            "t1_t3_comparison.csv": diagnostics.t1_t3_comparison,
            "spectrum_master_comparison.csv": diagnostics.spectrum_master_comparison,
            "confusion_summary.csv": diagnostics.confusion,
            "reliability_summary.csv": diagnostics.reliability,
            "control_summary.csv": diagnostics.control_summary,
            "cost_summary.csv": diagnostics.cost_summary,
        }
        for filename, table in diagnostic_tables.items():
            table.to_csv(lease.temporary_dir / filename, index=False)
        figure_manifest = generate_p03_figures(
            lease.temporary_dir, figure_definitions
        )
        figure_manifest.to_csv(
            lease.temporary_dir / "figure_manifest.csv", index=False
        )
        comparator = build_p04_comparator_freeze(
            execution_run_id=context.execution_run_id,
            protected_state_sha256=context.execution_protected_state_sha256,
            selected_model_specs=selected_model_specs,
            expected_endpoints=expected_endpoints,
            pooled_metrics=metric_tables.pooled_metrics,
        )
        (lease.temporary_dir / "p04_comparator_freeze.json").write_bytes(
            comparator_json_bytes(comparator)
        )
        (lease.temporary_dir / "P04_HANDOFF.md").write_text(
            render_p04_handoff(comparator)
        )
        (lease.temporary_dir / "P03_REPORT.md").write_text(
            render_p03_report(
                execution_run_id=context.execution_run_id,
                plan_run_id=context.plan_run_id,
                protected_state_sha256=context.execution_protected_state_sha256,
                fit_manifest_rows=len(context.fit_manifest),
                terminal_fit_ledger=evidence.fit_status,
                pooled_metrics=metric_tables.pooled_metrics,
                domain_summaries=metric_tables.domain_summaries,
                selection_frequency=diagnostics.selection_frequency,
                selection_stability=diagnostics.selection_stability,
                endpoint_coverage=diagnostics.endpoint_coverage,
                control_summary=diagnostics.control_summary,
                cost_summary=diagnostics.cost_summary,
                figure_manifest=figure_manifest,
            )
        )
        evidence.outer_validation.to_csv(
            lease.temporary_dir / "outer_shard_validation.csv", index=False
        )
        evidence.calibration_records.to_json(
            lease.temporary_dir / "calibration_records.jsonl",
            orient="records",
            lines=True,
        )
        (lease.temporary_dir / "prediction_schema_validation.json").write_bytes(
            canonical_json_bytes(prediction_schema_report, pretty=True)
        )
        pooled = metric_tables.pooled_metrics
        descriptor = {
            "schema_version": "p03-final-aggregation-v1",
            "execution_run_id": context.execution_run_id,
            "protected_state_sha256": context.execution_protected_state_sha256,
            "planned_fit_count": len(context.fit_manifest),
            "terminal_fit_count": len(evidence.fit_status),
            "terminal_fit_id_sha256": sha256_value(
                sorted(evidence.fit_status.fit_id.astype(str))
            ),
            "terminal_status_counts": evidence.fit_status.status.astype(
                str
            ).value_counts().sort_index().to_dict(),
            "failure_ledger_rows": len(failure_ledger),
            "executable_outer_run_count": len(executable),
            "valid_outer_shard_count": int(evidence.outer_validation.valid.sum()),
            "expected_final_endpoint_count": len(expected_endpoints),
            "expected_final_endpoint_sha256": sha256_value(
                sorted(
                    expected_endpoints.outer_run_id.astype(str)
                    + "|"
                    + expected_endpoints.procedure_id.astype(str)
                )
            ),
            "final_prediction_rows": len(normalized_predictions),
            "prediction_schema_status": prediction_schema_report["status"],
            "master_prediction_rows": len(metric_tables.master_predictions),
            "pooled_metric_rows": len(pooled),
            "complete_metric_rows": int(pooled.endpoint_status.eq("complete").sum()),
            "unavailable_metric_rows": int(
                pooled.endpoint_status.eq("unavailable").sum()
            ),
            "domain_summary_rows": len(metric_tables.domain_summaries),
            "calibration_record_rows": len(evidence.calibration_records),
            "diagnostic_table_rows": {
                filename: len(table)
                for filename, table in sorted(diagnostic_tables.items())
            },
            "figure_count": len(figure_manifest),
            "figure_ids": figure_manifest.figure_id.astype(str).tolist(),
            "figure_data_sha256": sha256_value(
                figure_manifest[["figure_id", "data_sha256"]].to_dict(
                    orient="records"
                )
            ),
            "selected_model_spec_rows": len(selected_model_specs),
            "p04_comparator_selection_mapping_sha256": comparator[
                "selection_mapping_sha256"
            ],
            "p04_comparator_outer_test_endpoint_sha256": comparator[
                "outer_test_endpoint_sha256"
            ],
        }
        (lease.temporary_dir / "final_aggregation_descriptor.json").write_bytes(
            canonical_json_bytes(descriptor, pretty=True)
        )
        return store.commit(lease), lease.action, context.execution_run_id
    except Exception:
        store.abort(lease, reason="final_aggregation_failed")
        raise


def _validate_final_aggregation_directory(
    path: Path, *, context: P03ExecutionContext
) -> dict[str, Any]:
    descriptor = json.loads((path / "final_aggregation_descriptor.json").read_text())
    terminal = pd.read_parquet(path / "terminal_fit_ledger.parquet")
    expected_endpoints = pd.read_parquet(path / "expected_endpoint_registry.parquet")
    pooled = pd.read_csv(path / "pooled_metrics.csv", low_memory=False)
    figures = pd.read_csv(path / "figure_manifest.csv", low_memory=False)
    schema_report = json.loads((path / "prediction_schema_validation.json").read_text())
    comparator = json.loads((path / "p04_comparator_freeze.json").read_text())
    expected_metric_rows = 2 * expected_endpoints[
        list(IDENTITY_COLUMNS)
    ].drop_duplicates().shape[0]
    figure_flag_columns = [
        "semantic_parity",
        "compiled",
        "native_tikz",
        "standalone_html",
        "colorblind_safe",
        "vector_only",
    ]
    figure_flags = figures[figure_flag_columns].apply(
        lambda values: (
            values
            if pd.api.types.is_bool_dtype(values)
            else values.astype(str).str.lower().eq("true")
        )
    )
    checks = {
        "descriptor_execution_run_matches": descriptor.get("execution_run_id")
        == context.execution_run_id,
        "descriptor_protected_state_matches": descriptor.get(
            "protected_state_sha256"
        )
        == context.execution_protected_state_sha256,
        "fit_manifest_count_exact": len(context.fit_manifest)
        == EXPECTED_FIT_MANIFEST_ROWS,
        "terminal_fit_count_exact": len(terminal) == len(context.fit_manifest),
        "terminal_fit_ids_unique": terminal.fit_id.astype(str).is_unique,
        "terminal_fit_ids_exact": set(terminal.fit_id.astype(str))
        == set(context.fit_manifest.fit_id.astype(str)),
        "terminal_statuses_valid": terminal.status.astype(str).isin(
            TERMINAL_STATUSES
        ).all(),
        "terminal_fit_hash_matches_descriptor": sha256_value(
            sorted(terminal.fit_id.astype(str))
        )
        == descriptor.get("terminal_fit_id_sha256"),
        "expected_endpoint_count_exact": len(expected_endpoints)
        == EXPECTED_FINAL_ENDPOINTS,
        "pooled_metric_row_count_exact": len(pooled) == expected_metric_rows,
        "pooled_metric_statuses_terminal": pooled.endpoint_status.astype(str).isin(
            {"complete", "unavailable"}
        ).all(),
        "prediction_schema_passes": schema_report.get("status") == "pass",
        "prediction_schema_validated_all_rows": int(
            schema_report.get("validated_rows", -1)
        )
        == int(descriptor.get("final_prediction_rows", -2)),
        "figure_ids_exact": set(figures.figure_id.astype(str))
        == set(FIGURE_SLUGS),
        "figure_forms_pass": figure_flags.all(axis=None),
        "figure_png_dpi_exact": figures.png_dpi.astype(int).eq(300).all(),
        "p04_comparator_source_matches": comparator.get("source_execution_run_id")
        == context.execution_run_id,
        "p04_comparator_outer_count_exact": int(
            comparator.get("outer_cell_count", -1)
        )
        == 260,
        "reports_present": (path / "P03_REPORT.md").is_file()
        and (path / "P04_HANDOFF.md").is_file(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "schema_version": "p03-execution-validation-v1",
        "execution_run_id": context.execution_run_id,
        "protected_state_sha256": context.execution_protected_state_sha256,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "diagnostics": {
            "fit_manifest_rows": len(context.fit_manifest),
            "terminal_fit_rows": len(terminal),
            "expected_endpoint_rows": len(expected_endpoints),
            "pooled_metric_rows": len(pooled),
            "final_prediction_rows": int(descriptor.get("final_prediction_rows", -1)),
            "figure_count": len(figures),
            "terminal_status_counts": terminal.status.astype(
                str
            ).value_counts().sort_index().to_dict(),
        },
    }


def validate_latest_p03_execution(
    *, project_root: Path, artifact_root: Path
) -> tuple[dict[str, Any], Path, str]:
    """Independently rehash and validate the complete P03 execution bundle."""

    context = load_latest_p03_execution_context(
        project_root=project_root,
        artifact_root=artifact_root,
        require_authorized=True,
    )
    run_root = artifact_root / "p03" / "runs" / context.execution_run_id
    final_store = P03ShardStore(run_root=run_root / "final_aggregation")
    validation = final_store.validation_table(
        {0: context.execution_protected_state_sha256}
    )
    if len(validation) != 1 or not bool(validation[0]["valid"]):
        raise RuntimeError("P03 final aggregation is absent, corrupt, or conflicting.")
    final_path = final_store.shards / final_store._name(0)  # noqa: SLF001
    report = _validate_final_aggregation_directory(final_path, context=context)
    if report["status"] != "pass":
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise RuntimeError(f"P03 execution validation failed: {failed}")
    store = P03ShardStore(run_root=run_root / "execution_validation")
    lease = store.begin(
        shard_id=0,
        protected_state_sha256=context.execution_protected_state_sha256,
    )
    if lease.action == "verified_skip":
        saved = json.loads(
            (lease.final_dir / "P03_EXECUTION_VALIDATION_REPORT.json").read_text()
        )
        if saved != report:
            raise RuntimeError("Saved P03 validation report differs from reconstruction.")
        return report, lease.final_dir, lease.action
    if lease.temporary_dir is None:
        raise RuntimeError("P03 execution validation lease has no temporary directory.")
    try:
        (lease.temporary_dir / "P03_EXECUTION_VALIDATION_REPORT.json").write_bytes(
            canonical_json_bytes(report, pretty=True)
        )
        path = store.commit(lease)
        return report, path, lease.action
    except Exception:
        store.abort(lease, reason="execution_validation_failed")
        raise
