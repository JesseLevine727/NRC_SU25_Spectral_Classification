"""Validate, collect, and aggregate completed P03 selection shards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from atlas_sers.evaluation.p03_controls import metadata_control_candidate_registry
from atlas_sers.evaluation.p03_plan import (
    FIXED_SUITE,
    assign_selection_shards,
)
from atlas_sers.evaluation.p03_selection import (
    OuterSelectionResult,
    derive_fixed_family_selections,
    select_outer_candidates,
)
from atlas_sers.governance.canonical import sha256_value
from atlas_sers.governance.p03_store import P03ShardStore


@dataclass(frozen=True)
class SelectionEvidence:
    fit_status: pd.DataFrame
    selection_unit_metrics: pd.DataFrame
    shard_validation: pd.DataFrame


def _selection_assignments(
    fit_manifest: pd.DataFrame, *, shard_target_fits: int
) -> pd.DataFrame:
    direct = fit_manifest[fit_manifest.accounting.astype(str).eq("new_fit")]
    return assign_selection_shards(direct, target=shard_target_fits)


def collect_selection_evidence(
    *,
    selection_run_root: Path,
    fit_manifest: pd.DataFrame,
    selection_shard_manifest: pd.DataFrame,
    protected_state_sha256: str,
    shard_target_fits: int,
) -> SelectionEvidence:
    """Rehash every planned shard and prove one terminal record per planned fit."""

    expected_shards = {
        int(value): protected_state_sha256
        for value in selection_shard_manifest.selection_shard_id
    }
    store = P03ShardStore(run_root=selection_run_root)
    validation = pd.DataFrame(store.validation_table(expected_shards))
    if len(validation) != len(expected_shards) or not validation.valid.all():
        invalid = validation.loc[~validation.valid, "shard_id"].astype(int).tolist()
        raise RuntimeError(f"P03 selection shards are incomplete or corrupt: {invalid}")
    assignments = _selection_assignments(
        fit_manifest, shard_target_fits=shard_target_fits
    )
    expected_fit_ids = set(assignments.fit_id.astype(str))
    status_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    for planned in selection_shard_manifest.sort_values(
        "selection_shard_id", kind="stable"
    ).itertuples(index=False):
        shard_id = int(planned.selection_shard_id)
        shard = store.shards / store._name(shard_id)  # noqa: SLF001 - validated store path
        descriptor = json.loads((shard / "shard_descriptor.json").read_text())
        assigned = assignments[assignments.selection_shard_id == shard_id]
        expected_hash = sha256_value(sorted(assigned.fit_id.astype(str)))
        if (
            len(assigned) != int(planned.fit_count)
            or descriptor["fit_count"] != int(planned.fit_count)
            or descriptor["fit_id_sha256"] != expected_hash
            or expected_hash != str(planned.fit_id_sha256)
        ):
            raise RuntimeError(f"Selection shard {shard_id} differs from the frozen plan.")
        statuses = pd.read_csv(shard / "fit_status.csv", low_memory=False)
        metrics = pd.read_csv(shard / "selection_unit_metrics.csv", low_memory=False)
        if set(statuses.fit_id.astype(str)) != set(assigned.fit_id.astype(str)):
            raise RuntimeError(f"Selection shard {shard_id} status coverage differs.")
        if set(metrics.fit_id.astype(str)) != set(assigned.fit_id.astype(str)):
            raise RuntimeError(f"Selection shard {shard_id} metric coverage differs.")
        status_frames.append(statuses)
        metric_frames.append(metrics)
    statuses = pd.concat(status_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    if not statuses.fit_id.astype(str).is_unique or not metrics.fit_id.astype(str).is_unique:
        raise RuntimeError("Collected selection evidence contains duplicate fit IDs.")
    if set(statuses.fit_id.astype(str)) != expected_fit_ids:
        raise RuntimeError("Collected selection status coverage is not exact.")
    if set(metrics.fit_id.astype(str)) != expected_fit_ids:
        raise RuntimeError("Collected selection metric coverage is not exact.")
    if not statuses.status.astype(str).isin(
        {
            "complete",
            "unsupported_candidate",
            "rank_failure",
            "convergence_failure",
            "numerical_failure",
            "resource_failure",
            "data_failure",
            "fit_failure",
        }
    ).all():
        raise RuntimeError("Collected selection evidence contains a nonterminal status.")
    merged = statuses[["fit_id", "status"]].merge(
        metrics[["fit_id", "status"]],
        on="fit_id",
        validate="one_to_one",
        suffixes=("_fit", "_metric"),
    )
    if not merged.status_fit.astype(str).eq(merged.status_metric.astype(str)).all():
        raise RuntimeError("Fit and metric terminal statuses disagree.")
    return SelectionEvidence(statuses, metrics, validation)


def build_all_selection_results(
    *,
    fit_manifest: pd.DataFrame,
    selection_unit_metrics: pd.DataFrame,
    expected_run_registry: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    coral_candidate_registry: pd.DataFrame,
    control_registry: pd.DataFrame,
) -> OuterSelectionResult:
    """Resolve every standard, fixed-family, C12, and metadata selection."""

    summaries: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []

    def select(
        stage: str,
        registry: pd.DataFrame,
        kind: str,
    ) -> OuterSelectionResult:
        manifest = fit_manifest[
            fit_manifest.stage.astype(str).eq(stage)
            & fit_manifest.accounting.astype(str).eq("new_fit")
        ]
        fit_ids = set(manifest.fit_id.astype(str))
        metrics = selection_unit_metrics[
            selection_unit_metrics.fit_id.astype(str).isin(fit_ids)
        ]
        result = select_outer_candidates(
            selection_fit_manifest=manifest,
            selection_unit_metrics=metrics,
            candidate_registry=registry,
        )
        result.selections["selection_kind"] = kind
        result.traces["selection_kind"] = kind
        return result

    standard_t1_t3 = select("inner_selection", candidate_registry, "standard")
    summaries.append(standard_t1_t3.selections)
    traces.append(standard_t1_t3.traces)
    t2 = select("training_station_inner_selection", candidate_registry, "standard")
    summaries.append(t2.selections)
    traces.append(t2.traces)
    coral = select(
        "inner_source_coral_selection",
        coral_candidate_registry,
        "source_covariance",
    )
    summaries.append(coral.selections)
    traces.append(coral.traces)
    metadata_registry = metadata_control_candidate_registry(control_registry)
    metadata = select(
        "metadata_inner_selection",
        metadata_registry,
        "metadata_control",
    )
    summaries.append(metadata.selections)
    traces.append(metadata.traces)
    c09_manifest = fit_manifest[
        fit_manifest.experiment_id.astype(str).eq("EXP-C09-T3")
        & fit_manifest.stage.astype(str).eq("inner_selection")
    ]
    c09_fit_ids = set(c09_manifest.fit_id.astype(str))
    c10 = derive_fixed_family_selections(
        c09_selection_fit_manifest=c09_manifest,
        c09_selection_unit_metrics=selection_unit_metrics[
            selection_unit_metrics.fit_id.astype(str).isin(c09_fit_ids)
        ],
        candidate_registry=candidate_registry,
        expected_run_registry=expected_run_registry,
        fixed_models=FIXED_SUITE,
    )
    c10.selections["selection_kind"] = "fixed_family_reuse"
    c10.traces["selection_kind"] = "fixed_family_reuse"
    summaries.append(c10.selections)
    traces.append(c10.traces)
    selections = pd.concat(summaries, ignore_index=True).sort_values(
        ["experiment_id", "outer_run_id"], kind="stable"
    ).reset_index(drop=True)
    trace = pd.concat(traces, ignore_index=True).sort_values(
        ["experiment_id", "outer_run_id", "declared_candidate_order"],
        kind="stable",
    ).reset_index(drop=True)
    expected_counts = {
        "EXP-C01-T1": 60,
        "EXP-C02-T1": 60,
        "EXP-C03-T1": 60,
        "EXP-C04-T1": 60,
        "EXP-C05-T1": 60,
        "EXP-C06-T1": 60,
        "EXP-C07-T1": 60,
        "EXP-C08-T1": 60,
        "EXP-C09-T3": 260,
        "EXP-C09-CONTROL-META": 260,
        "EXP-C10-T3": 1040,
        "EXP-C11-T2": 2,
        "EXP-C12-CORAL": 260,
    }
    observed_counts = selections.experiment_id.value_counts().to_dict()
    if observed_counts != expected_counts or len(selections) != 2302:
        raise RuntimeError(
            f"P03 selection coverage differs: observed={observed_counts}"
        )
    if not selections.outer_run_id.astype(str).is_unique:
        raise RuntimeError("P03 selection summary contains duplicate outer-run IDs.")
    return OuterSelectionResult(selections, trace)


def load_selection_predictions(
    *,
    selection_run_root: Path,
    fit_manifest: pd.DataFrame,
    fit_ids: set[str],
    shard_target_fits: int,
) -> pd.DataFrame:
    """Load only selected cross-fit prediction rows from their validated shards."""

    if not fit_ids:
        raise ValueError("Selection prediction request contains no fit IDs.")
    assignments = _selection_assignments(
        fit_manifest, shard_target_fits=shard_target_fits
    )
    requested = assignments[assignments.fit_id.astype(str).isin(fit_ids)]
    if set(requested.fit_id.astype(str)) != {str(value) for value in fit_ids}:
        raise ValueError("Requested selection prediction fit IDs are not all planned.")
    store = P03ShardStore(run_root=selection_run_root)
    frames: list[pd.DataFrame] = []
    for shard_id, rows in requested.groupby("selection_shard_id", sort=True):
        local_ids = sorted(rows.fit_id.astype(str))
        path = store.shards / store._name(int(shard_id))  # noqa: SLF001
        frame = pd.read_parquet(
            path / "validation_predictions.parquet",
            filters=[("fit_id", "in", local_ids)],
        )
        frame = frame[frame.fit_id.astype(str).isin(local_ids)]
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    if set(predictions.fit_id.astype(str)) != {str(value) for value in fit_ids}:
        raise RuntimeError("Selected cross-fit prediction coverage is incomplete.")
    return predictions
