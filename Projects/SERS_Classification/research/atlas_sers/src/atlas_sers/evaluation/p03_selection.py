"""Deterministic, source-only aggregation of P03 inner candidate evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from atlas_sers.evaluation.classical import select_lexicographic_candidate
from atlas_sers.governance.canonical import sha256_value

OUTER_CONTEXT = (
    "experiment_id",
    "task_id",
    "outer_run_id",
    "domain",
    "station",
    "held_instrument",
    "outer_repeat",
    "outer_fold",
    "selection_mode",
)


@dataclass(frozen=True)
class OuterSelectionResult:
    selections: pd.DataFrame
    traces: pd.DataFrame


def derive_fixed_family_selections(
    *,
    c09_selection_fit_manifest: pd.DataFrame,
    c09_selection_unit_metrics: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    expected_run_registry: pd.DataFrame,
    fixed_models: tuple[str, ...],
) -> OuterSelectionResult:
    """Reuse C09 source evidence to select hyperparameters within each C10 family."""

    c09_runs = expected_run_registry[expected_run_registry.experiment_id == "EXP-C09-T3"]
    c10_runs = expected_run_registry[expected_run_registry.experiment_id == "EXP-C10-T3"]
    match_keys = ["domain", "station", "held_instrument", "outer_repeat", "outer_fold"]
    if c09_runs.duplicated(match_keys).any():
        raise ValueError("C09 expected-run keys are not unique.")
    summaries: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []
    for model_id in fixed_models:
        family_manifest = c09_selection_fit_manifest[
            c09_selection_fit_manifest.model_id == model_id
        ]
        family_fit_ids = set(family_manifest.fit_id.astype(str))
        family_metrics = c09_selection_unit_metrics[
            c09_selection_unit_metrics.fit_id.astype(str).isin(family_fit_ids)
        ]
        selected = select_outer_candidates(
            selection_fit_manifest=family_manifest,
            selection_unit_metrics=family_metrics,
            candidate_registry=candidate_registry,
        )
        mapping = c09_runs[["outer_run_id", *match_keys]].rename(
            columns={"outer_run_id": "source_c09_outer_run_id"}
        ).merge(
            c10_runs.loc[c10_runs.model_id == model_id, ["outer_run_id", *match_keys]],
            on=match_keys,
            how="inner",
            validate="one_to_one",
        )
        if len(mapping) != len(c09_runs):
            raise ValueError(f"C10 family {model_id} does not map one-to-one onto C09 cells.")
        summary = selected.selections.rename(
            columns={"outer_run_id": "source_c09_outer_run_id"}
        ).merge(
            mapping[["source_c09_outer_run_id", "outer_run_id"]],
            on="source_c09_outer_run_id",
            how="inner",
            validate="one_to_one",
        )
        summary["experiment_id"] = "EXP-C10-T3"
        summary["fixed_family_model_id"] = model_id
        summaries.append(summary)
        trace = selected.traces.rename(
            columns={"outer_run_id": "source_c09_outer_run_id"}
        ).merge(
            mapping[["source_c09_outer_run_id", "outer_run_id"]],
            on="source_c09_outer_run_id",
            how="inner",
            validate="many_to_one",
        )
        trace["experiment_id"] = "EXP-C10-T3"
        trace["fixed_family_model_id"] = model_id
        traces.append(trace)
    return OuterSelectionResult(
        pd.concat(summaries, ignore_index=True).sort_values(
            ["fixed_family_model_id", "outer_run_id"], kind="stable"
        ),
        pd.concat(traces, ignore_index=True).sort_values(
            ["fixed_family_model_id", "outer_run_id", "declared_candidate_order"],
            kind="stable",
        ),
    )


def _normalize(value: Any) -> Any:
    return None if pd.isna(value) else value


def _selection_hash(metrics: pd.DataFrame, winner: str | None) -> str:
    evidence_fields = [
        "fit_id",
        "candidate_id",
        "selection_unit_id",
        "seed",
        "status",
        "balanced_accuracy",
        "macro_f1",
    ]
    evidence = [
        {field: _normalize(row[field]) for field in evidence_fields}
        for _, row in metrics.sort_values("fit_id", kind="stable").iterrows()
    ]
    return sha256_value({"source_only_evidence": evidence, "selected_candidate_id": winner})


def select_outer_candidates(
    *,
    selection_fit_manifest: pd.DataFrame,
    selection_unit_metrics: pd.DataFrame,
    candidate_registry: pd.DataFrame,
) -> OuterSelectionResult:
    """Select one candidate per outer run after proving complete fit accounting."""

    if selection_fit_manifest.empty:
        raise ValueError("Selection fit manifest is empty.")
    if not selection_fit_manifest.fit_id.astype(str).is_unique:
        raise ValueError("Selection fit manifest contains duplicate fit IDs.")
    if not selection_unit_metrics.fit_id.astype(str).is_unique:
        raise ValueError("Selection evidence contains duplicate fit IDs.")
    expected = set(selection_fit_manifest.fit_id.astype(str))
    observed = set(selection_unit_metrics.fit_id.astype(str))
    if expected != observed:
        raise ValueError(
            "Selection evidence fit-ID coverage differs from the no-fit manifest: "
            f"missing={len(expected - observed)} extra={len(observed - expected)}"
        )
    joined = selection_fit_manifest[
        ["fit_id", "outer_run_id", "hyperparameter_sha256"]
    ].merge(
        selection_unit_metrics,
        on=["fit_id", "outer_run_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_manifest", ""),
    )
    summaries: list[dict[str, Any]] = []
    traces: list[pd.DataFrame] = []
    for outer_run_id, expected_cell in selection_fit_manifest.groupby(
        "outer_run_id", sort=True
    ):
        context = {field: expected_cell.iloc[0][field] for field in OUTER_CONTEXT}
        if any(expected_cell[field].nunique(dropna=False) != 1 for field in OUTER_CONTEXT):
            raise ValueError(f"Outer selection context is inconsistent for {outer_run_id}.")
        metrics = joined[joined.outer_run_id == outer_run_id].copy()
        expected_units = set(expected_cell.selection_unit_id.astype(str))
        if set(metrics.selection_unit_id.astype(str)) != expected_units:
            raise ValueError(f"Selection-unit coverage differs for {outer_run_id}.")
        candidate_ids = set(expected_cell.candidate_id.astype(str))
        registry = candidate_registry[
            candidate_registry.candidate_id.astype(str).isin(candidate_ids)
        ].copy()
        if set(registry.candidate_id.astype(str)) != candidate_ids:
            raise ValueError(f"Candidate registry coverage differs for {outer_run_id}.")
        selected_candidate: str | None = None
        selected_model: str | None = None
        selected_hyperparameters: str | None = None
        reason: str | None = None
        try:
            winner, trace = select_lexicographic_candidate(metrics, registry)
            selected_candidate = str(winner.candidate_id)
            selected_model = str(winner.model_id)
            selected_hyperparameters = str(
                registry.loc[
                    registry.candidate_id.astype(str) == selected_candidate,
                    "hyperparameter_sha256",
                ].iloc[0]
            )
            status = "complete"
        except ValueError as error:
            if str(error) != "No candidate has complete selection-unit support.":
                raise
            # Re-run aggregation with a synthetic ineligible placeholder is not
            # acceptable; preserve every raw terminal record instead.
            trace = registry.copy()
            trace["complete_support"] = False
            trace["selected"] = False
            status = "selection_failure"
            reason = "no_candidate_has_complete_selection_unit_and_seed_support"
        trace = trace.copy()
        for field, value in context.items():
            trace[field] = value
        traces.append(trace)
        summaries.append(
            {
                **context,
                "status": status,
                "reason_code": reason,
                "selected_candidate_id": selected_candidate,
                "selected_model_id": selected_model,
                "selected_hyperparameter_sha256": selected_hyperparameters,
                "selection_unit_count": len(expected_units),
                "expected_fit_count": len(expected_cell),
                "terminal_fit_count": len(metrics),
                "selection_evidence_fit_id_sha256": sha256_value(
                    sorted(metrics.fit_id.astype(str))
                ),
                "selection_state_sha256": _selection_hash(metrics, selected_candidate),
            }
        )
    return OuterSelectionResult(
        pd.DataFrame(summaries).sort_values("outer_run_id", kind="stable").reset_index(drop=True),
        pd.concat(traces, ignore_index=True).sort_values(
            ["outer_run_id", "declared_candidate_order"], kind="stable"
        ),
    )
