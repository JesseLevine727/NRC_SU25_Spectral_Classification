"""Outcome-blind definitions for the complete P03 diagnostic result layer."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.governance.canonical import sha256_value

FIXED_T1_EXPERIMENTS = {
    "C-PCA-LDA": "EXP-C03-T1",
    "C-LOGREG-EN": "EXP-C05-T1",
    "C-RBF-SVM": "EXP-C06-T1",
    "C-EXTRA-TREES": "EXP-C08-T1",
}

SPECTRUM_MASTER_COLUMNS = [
    "experiment_id",
    "task_id",
    "domain",
    "station",
    "procedure_id",
    "outer_repeat",
    "spectrum_status",
    "master_status",
    "spectrum_balanced_accuracy",
    "master_balanced_accuracy",
    "master_minus_spectrum",
    "pair_status",
]
CONFUSION_COLUMNS = [
    "station",
    "aggregation_level",
    "true_index",
    "predicted_index",
    "true_label",
    "predicted_label",
    "mean_row_fraction",
    "minimum_row_fraction",
    "maximum_row_fraction",
    "mean_count_per_repeat",
    "mean_true_support_per_repeat",
    "repeat_count",
    "minimum_domain_count",
]
RELIABILITY_COLUMNS = [
    "aggregation_level",
    "station",
    "bin_index",
    "bin_lower",
    "bin_upper",
    "mean_confidence",
    "minimum_confidence",
    "maximum_confidence",
    "mean_empirical_accuracy",
    "minimum_empirical_accuracy",
    "maximum_empirical_accuracy",
    "mean_observations_per_repeat",
    "repeat_count",
]


@dataclass(frozen=True)
class P03DiagnosticTables:
    selection_frequency: pd.DataFrame
    selection_stability: pd.DataFrame
    selection_margins: pd.DataFrame
    endpoint_coverage: pd.DataFrame
    t1_t3_comparison: pd.DataFrame
    spectrum_master_comparison: pd.DataFrame
    confusion: pd.DataFrame
    reliability: pd.DataFrame
    control_summary: pd.DataFrame
    cost_summary: pd.DataFrame


def _require(frame: pd.DataFrame, fields: set[str], *, name: str) -> None:
    missing = fields - set(frame)
    if missing:
        raise ValueError(f"{name} misses fields: {sorted(missing)}")


def _as_bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1"})


def _normalized_entropy(counts: pd.Series) -> float:
    probabilities = counts.to_numpy(dtype=float) / float(counts.sum())
    if len(probabilities) <= 1:
        return 0.0
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(len(probabilities)))


def build_selection_diagnostics(
    selections: pd.DataFrame, traces: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize outcome frequency, repeat agreement, and winner margins."""

    _require(
        selections,
        {
            "experiment_id",
            "task_id",
            "outer_run_id",
            "station",
            "domain",
            "held_instrument",
            "outer_repeat",
            "outer_fold",
            "selection_kind",
            "status",
            "selected_model_id",
            "selected_candidate_id",
        },
        name="P03 selection summary",
    )
    if selections.empty or not selections.outer_run_id.astype(str).is_unique:
        raise ValueError("P03 selection summary must contain unique outer runs.")
    work = selections.copy()
    complete = work.status.astype(str).eq("complete")
    work["selection_outcome_model"] = work.selected_model_id.where(
        complete, "<selection_failure>"
    ).fillna("<selection_failure>")
    work["selection_outcome_candidate"] = work.selected_candidate_id.where(
        complete, "<selection_failure>"
    ).fillna("<selection_failure>")
    frequency_keys = [
        "experiment_id",
        "task_id",
        "station",
        "selection_kind",
        "selection_outcome_model",
        "selection_outcome_candidate",
    ]
    frequency = (
        work.groupby(frequency_keys, sort=True, dropna=False)
        .size()
        .rename("selection_count")
        .reset_index()
    )
    denominator_keys = frequency_keys[:4]
    denominators = (
        work.groupby(denominator_keys, sort=True, dropna=False)
        .size()
        .rename("selection_denominator")
        .reset_index()
    )
    frequency = frequency.merge(
        denominators, on=denominator_keys, validate="many_to_one"
    )
    frequency["selection_fraction"] = (
        frequency.selection_count / frequency.selection_denominator
    )

    stability_keys = [
        "experiment_id",
        "task_id",
        "station",
        "domain",
        "held_instrument",
        "outer_fold",
        "selection_kind",
    ]
    if "fixed_family_model_id" in work:
        stability_keys.append("fixed_family_model_id")
    stability_rows: list[dict[str, Any]] = []
    for identity, group in work.groupby(stability_keys, sort=True, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        outcomes = group.selection_outcome_candidate.astype(str)
        counts = outcomes.value_counts(sort=True)
        modal = sorted(counts[counts.eq(counts.max())].index.astype(str))[0]
        stability_rows.append(
            {
                **dict(zip(stability_keys, identity, strict=True)),
                "repeat_count": int(group.outer_repeat.nunique()),
                "selection_count": len(group),
                "complete_selection_count": int(group.status.astype(str).eq("complete").sum()),
                "distinct_outcome_count": len(counts),
                "modal_candidate_id": modal,
                "modal_count": int(counts.loc[modal]),
                "modal_fraction": float(counts.loc[modal] / len(group)),
                "normalized_entropy": _normalized_entropy(counts),
                "selection_state_set_sha256": sha256_value(
                    sorted(group.selection_state_sha256.astype(str))
                )
                if "selection_state_sha256" in group
                else None,
            }
        )
    stability = pd.DataFrame(stability_rows)

    margin_rows: list[dict[str, Any]] = []
    if not traces.empty:
        _require(
            traces,
            {
                "outer_run_id",
                "candidate_id",
                "complete_support",
                "selected",
                "mean_balanced_accuracy",
                "worst_balanced_accuracy",
                "mean_macro_f1",
            },
            name="P03 selection trace",
        )
        context_fields = [
            field
            for field in (
                "experiment_id",
                "task_id",
                "domain",
                "station",
                "held_instrument",
                "outer_repeat",
                "outer_fold",
                "selection_kind",
                "fixed_family_model_id",
            )
            if field in traces
        ]
        for outer_run_id, group in traces.groupby("outer_run_id", sort=True):
            supported = group[_as_bool(group.complete_support)].copy()
            selected = supported[_as_bool(supported.selected)]
            if len(selected) != 1:
                margin_rows.append(
                    {
                        "outer_run_id": str(outer_run_id),
                        **{
                            field: group.iloc[0][field]
                            for field in context_fields
                        },
                        "margin_status": "unavailable",
                        "selected_candidate_id": None,
                        "supported_candidate_count": len(supported),
                        "mean_ba_margin": None,
                        "worst_ba_margin": None,
                        "macro_f1_margin": None,
                    }
                )
                continue
            winner = selected.iloc[0]
            competitors = supported[
                ~supported.candidate_id.astype(str).eq(str(winner.candidate_id))
            ]
            runner = (
                competitors.sort_values(
                    [
                        "mean_balanced_accuracy",
                        "worst_balanced_accuracy",
                        "mean_macro_f1",
                        "complexity_rank",
                        "declared_candidate_order",
                    ],
                    ascending=[False, False, False, True, True],
                    kind="stable",
                ).iloc[0]
                if not competitors.empty
                else None
            )
            margin_rows.append(
                {
                    "outer_run_id": str(outer_run_id),
                    **{field: winner[field] for field in context_fields},
                    "margin_status": "complete",
                    "selected_candidate_id": str(winner.candidate_id),
                    "runner_up_candidate_id": (
                        str(runner.candidate_id) if runner is not None else None
                    ),
                    "supported_candidate_count": len(supported),
                    "mean_ba_margin": (
                        float(winner.mean_balanced_accuracy - runner.mean_balanced_accuracy)
                        if runner is not None
                        else None
                    ),
                    "worst_ba_margin": (
                        float(winner.worst_balanced_accuracy - runner.worst_balanced_accuracy)
                        if runner is not None
                        else None
                    ),
                    "macro_f1_margin": (
                        float(winner.mean_macro_f1 - runner.mean_macro_f1)
                        if runner is not None
                        else None
                    ),
                }
            )
    margins = pd.DataFrame(margin_rows)
    return frequency, stability, margins


def build_endpoint_coverage(pooled_metrics: pd.DataFrame) -> pd.DataFrame:
    """Retain planned, complete, and unavailable pooled endpoints in denominators."""

    _require(
        pooled_metrics,
        {
            "experiment_id",
            "task_id",
            "station",
            "procedure_id",
            "aggregation_level",
            "endpoint_status",
            "expected_observations",
            "expected_masters",
        },
        name="P03 pooled metrics",
    )
    keys = [
        "experiment_id",
        "task_id",
        "station",
        "procedure_id",
        "aggregation_level",
    ]
    rows: list[dict[str, Any]] = []
    for identity, group in pooled_metrics.groupby(keys, sort=True, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        complete = group.endpoint_status.astype(str).eq("complete")
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "planned_endpoint_count": len(group),
                "complete_endpoint_count": int(complete.sum()),
                "unavailable_endpoint_count": int((~complete).sum()),
                "completion_fraction": float(complete.mean()),
                "expected_observation_memberships": int(
                    pd.to_numeric(group.expected_observations).fillna(0).sum()
                ),
                "expected_master_memberships": int(
                    pd.to_numeric(group.expected_masters).fillna(0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_t1_t3_comparison(pooled_metrics: pd.DataFrame) -> pd.DataFrame:
    """Match fixed classical families across within-station and zero-shot regimes."""

    complete = pooled_metrics[
        pooled_metrics.endpoint_status.astype(str).eq("complete")
    ].copy()
    keys = ["station", "procedure_id", "outer_repeat", "aggregation_level"]
    t1_parts: list[pd.DataFrame] = []
    for model_id, experiment_id in FIXED_T1_EXPERIMENTS.items():
        part = complete[
            complete.experiment_id.astype(str).eq(experiment_id)
            & complete.procedure_id.astype(str).eq(model_id)
        ]
        if not part.empty:
            t1_parts.append(part)
    t1 = pd.concat(t1_parts, ignore_index=True) if t1_parts else pd.DataFrame()
    t3 = complete[
        complete.experiment_id.astype(str).eq("EXP-C10-T3")
        & complete.procedure_id.astype(str).isin(FIXED_T1_EXPERIMENTS)
    ]
    columns = [
        *keys,
        "within_station_balanced_accuracy",
        "zero_shot_balanced_accuracy",
        "zero_shot_minus_within_station",
        "within_station_endpoint_count",
        "zero_shot_domain_count",
    ]
    if t1.empty or t3.empty:
        return pd.DataFrame(columns=columns)
    t1_summary = (
        t1.groupby(keys, sort=True, as_index=False)
        .agg(
            within_station_balanced_accuracy=("balanced_accuracy", "mean"),
            within_station_endpoint_count=("domain", "nunique"),
        )
    )
    t3_summary = (
        t3.groupby(keys, sort=True, as_index=False)
        .agg(
            zero_shot_balanced_accuracy=("balanced_accuracy", "mean"),
            zero_shot_domain_count=("domain", "nunique"),
        )
    )
    result = t1_summary.merge(t3_summary, on=keys, validate="one_to_one")
    result["zero_shot_minus_within_station"] = (
        result.zero_shot_balanced_accuracy
        - result.within_station_balanced_accuracy
    )
    return result[columns]


def build_spectrum_master_comparison(pooled_metrics: pd.DataFrame) -> pd.DataFrame:
    """Pair the two registered aggregation levels within every primary C09 endpoint."""

    primary = pooled_metrics[
        pooled_metrics.experiment_id.astype(str).eq("EXP-C09-T3")
    ].copy()
    keys = [
        "experiment_id",
        "task_id",
        "domain",
        "station",
        "procedure_id",
        "outer_repeat",
    ]
    status = primary.pivot(index=keys, columns="aggregation_level", values="endpoint_status")
    score = primary.pivot(index=keys, columns="aggregation_level", values="balanced_accuracy")
    required_levels = {"spectrum", "instrument_balanced_master"}
    if not required_levels <= set(status) or not required_levels <= set(score):
        return pd.DataFrame(columns=SPECTRUM_MASTER_COLUMNS)
    result = status.reset_index()[keys].copy()
    result["spectrum_status"] = status["spectrum"].to_numpy()
    result["master_status"] = status["instrument_balanced_master"].to_numpy()
    result["spectrum_balanced_accuracy"] = score["spectrum"].to_numpy()
    result["master_balanced_accuracy"] = score[
        "instrument_balanced_master"
    ].to_numpy()
    result["master_minus_spectrum"] = (
        result.master_balanced_accuracy - result.spectrum_balanced_accuracy
    )
    result["pair_status"] = np.where(
        result.spectrum_status.eq("complete") & result.master_status.eq("complete"),
        "complete",
        "unavailable",
    )
    return result[SPECTRUM_MASTER_COLUMNS]


def _vocabulary(value: Any) -> tuple[str, ...]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list) or len(parsed) < 2:
        raise ValueError("A P03 class vocabulary is invalid.")
    return tuple(str(label) for label in parsed)


def build_confusion_summary(pooled_metrics: pd.DataFrame) -> pd.DataFrame:
    """Average row-normalized C09 confusion over technical split repeats."""

    primary = pooled_metrics[
        pooled_metrics.experiment_id.astype(str).eq("EXP-C09-T3")
        & pooled_metrics.endpoint_status.astype(str).eq("complete")
    ]
    if primary.empty:
        return pd.DataFrame(columns=CONFUSION_COLUMNS)
    _require(
        primary,
        {
            "station",
            "aggregation_level",
            "outer_repeat",
            "domain",
            "class_vocabulary_json",
            "confusion_matrix_json",
        },
        name="P03 complete primary metrics",
    )
    repeat_rows: list[dict[str, Any]] = []
    group_keys = ["station", "aggregation_level", "outer_repeat"]
    for identity, group in primary.groupby(group_keys, sort=True, dropna=False):
        vocabularies = group.class_vocabulary_json.astype(str).unique()
        if len(vocabularies) != 1:
            raise ValueError("Confusion aggregation mixes class vocabularies.")
        vocabulary = _vocabulary(vocabularies[0])
        matrices = [
            np.asarray(json.loads(str(value)), dtype=float)
            for value in group.confusion_matrix_json
        ]
        if any(matrix.shape != (len(vocabulary), len(vocabulary)) for matrix in matrices):
            raise ValueError("A confusion matrix disagrees with its class vocabulary.")
        matrix = np.sum(matrices, axis=0)
        support = matrix.sum(axis=1)
        fractions = np.divide(
            matrix,
            support[:, None],
            out=np.full_like(matrix, np.nan),
            where=support[:, None] > 0,
        )
        station, aggregation_level, outer_repeat = identity
        for true_index, true_label in enumerate(vocabulary):
            for predicted_index, predicted_label in enumerate(vocabulary):
                repeat_rows.append(
                    {
                        "station": station,
                        "aggregation_level": aggregation_level,
                        "outer_repeat": outer_repeat,
                        "true_index": true_index,
                        "predicted_index": predicted_index,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(matrix[true_index, predicted_index]),
                        "true_support": int(support[true_index]),
                        "row_fraction": fractions[true_index, predicted_index],
                        "domain_count": int(group.domain.nunique()),
                    }
                )
    repeat_frame = pd.DataFrame(repeat_rows)
    cell_keys = [
        "station",
        "aggregation_level",
        "true_index",
        "predicted_index",
        "true_label",
        "predicted_label",
    ]
    return (
        repeat_frame.groupby(cell_keys, sort=True, as_index=False)
        .agg(
            mean_row_fraction=("row_fraction", "mean"),
            minimum_row_fraction=("row_fraction", "min"),
            maximum_row_fraction=("row_fraction", "max"),
            mean_count_per_repeat=("count", "mean"),
            mean_true_support_per_repeat=("true_support", "mean"),
            repeat_count=("outer_repeat", "nunique"),
            minimum_domain_count=("domain_count", "min"),
        )
    )[CONFUSION_COLUMNS]


def _probability_matrix(values: pd.Series) -> np.ndarray:
    rows: list[np.ndarray] = []
    for value in values:
        parsed = json.loads(value) if isinstance(value, str) else value
        rows.append(np.asarray(parsed, dtype=float))
    matrix = np.vstack(rows)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Reliability probabilities are invalid.")
    return matrix / matrix.sum(axis=1, keepdims=True)


def build_reliability_summary(
    final_predictions: pd.DataFrame, master_predictions: pd.DataFrame
) -> pd.DataFrame:
    """Build ten fixed-width reliability bins and average curves over repeats."""

    sources = (
        ("spectrum", final_predictions),
        ("instrument_balanced_master", master_predictions),
    )
    per_repeat: list[dict[str, Any]] = []
    for aggregation_level, source in sources:
        if source.empty:
            continue
        _require(
            source,
            {
                "experiment_id",
                "station",
                "outer_repeat",
                "true_label",
                "predicted_label",
                "probabilities",
                "probability_status",
            },
            name=f"P03 {aggregation_level} predictions",
        )
        frame = source[
            source.experiment_id.astype(str).eq("EXP-C09-T3")
            & source.probability_status.astype(str).eq("cross_fitted_temperature")
        ].copy()
        if frame.empty:
            continue
        probabilities = _probability_matrix(frame.probabilities)
        frame["confidence"] = probabilities.max(axis=1)
        frame["correct"] = frame.true_label.astype(str).eq(
            frame.predicted_label.astype(str)
        )
        frame["bin_index"] = np.minimum(
            np.floor(frame.confidence.to_numpy(dtype=float) * 10).astype(int), 9
        )
        for identity, group in frame.groupby(
            ["station", "outer_repeat", "bin_index"], sort=True
        ):
            station, outer_repeat, bin_index = identity
            per_repeat.append(
                {
                    "aggregation_level": aggregation_level,
                    "station": station,
                    "outer_repeat": outer_repeat,
                    "bin_index": int(bin_index),
                    "bin_lower": int(bin_index) / 10,
                    "bin_upper": (int(bin_index) + 1) / 10,
                    "observations": len(group),
                    "mean_confidence": float(group.confidence.mean()),
                    "empirical_accuracy": float(group.correct.mean()),
                }
            )
    if not per_repeat:
        return pd.DataFrame(columns=RELIABILITY_COLUMNS)
    per_repeat_frame = pd.DataFrame(per_repeat)
    keys = ["aggregation_level", "station", "bin_index", "bin_lower", "bin_upper"]
    return (
        per_repeat_frame.groupby(keys, sort=True, as_index=False)
        .agg(
            mean_confidence=("mean_confidence", "mean"),
            minimum_confidence=("mean_confidence", "min"),
            maximum_confidence=("mean_confidence", "max"),
            mean_empirical_accuracy=("empirical_accuracy", "mean"),
            minimum_empirical_accuracy=("empirical_accuracy", "min"),
            maximum_empirical_accuracy=("empirical_accuracy", "max"),
            mean_observations_per_repeat=("observations", "mean"),
            repeat_count=("outer_repeat", "nunique"),
        )
    )[RELIABILITY_COLUMNS]


def build_control_summary(pooled_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize real, permuted-label, metadata-only, and prior C09 procedures."""

    experiments = {
        "EXP-C09-T3": "real spectra",
        "EXP-C09-CONTROL-PERM": "permuted master labels",
        "EXP-C09-CONTROL-META": "acquisition metadata only",
        "EXP-C09-CONTROL-PRIOR": "source prior",
    }
    frame = pooled_metrics[
        pooled_metrics.experiment_id.astype(str).isin(experiments)
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["control_kind"] = frame.experiment_id.astype(str).map(experiments)
    keys = [
        "experiment_id",
        "control_kind",
        "procedure_id",
        "aggregation_level",
        "outer_repeat",
    ]
    repeat_rows: list[dict[str, Any]] = []
    for identity, group in frame.groupby(keys, sort=True, dropna=False):
        complete = group[group.endpoint_status.astype(str).eq("complete")]
        repeat_rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "planned_domain_count": int(group.domain.nunique()),
                "complete_domain_count": int(complete.domain.nunique()),
                "mean_domain_balanced_accuracy": complete.balanced_accuracy.mean(),
                "worst_domain_balanced_accuracy": complete.balanced_accuracy.min(),
            }
        )
    repeat_frame = pd.DataFrame(repeat_rows)
    summary_keys = keys[:-1]
    result = (
        repeat_frame.groupby(summary_keys, sort=True, as_index=False)
        .agg(
            mean_domain_balanced_accuracy=("mean_domain_balanced_accuracy", "mean"),
            minimum_repeat_balanced_accuracy=("mean_domain_balanced_accuracy", "min"),
            maximum_repeat_balanced_accuracy=("mean_domain_balanced_accuracy", "max"),
            worst_domain_balanced_accuracy=("worst_domain_balanced_accuracy", "min"),
            repeat_count=("outer_repeat", "nunique"),
            minimum_complete_domain_count=("complete_domain_count", "min"),
            expected_domain_count=("planned_domain_count", "max"),
        )
    )
    result["summary_status"] = np.where(
        result.minimum_complete_domain_count.eq(result.expected_domain_count),
        "complete",
        "incomplete_terminal_cells",
    )
    return result


def _stage_group(stage: str) -> str:
    value = str(stage)
    if "selection" in value:
        return "selection"
    if "calibration" in value:
        return "calibration"
    if value.startswith("final") or "final_refit" in value or "permutation" in value:
        return "final_refit_and_prediction"
    return "bookkeeping_or_control"


def build_cost_summary(
    terminal_fit_ledger: pd.DataFrame, fit_manifest: pd.DataFrame
) -> pd.DataFrame:
    """Reconstruct M23–M25 timing, latency, and size with exact denominators."""

    _require(
        terminal_fit_ledger,
        {"fit_id", "status"},
        name="P03 terminal fit ledger",
    )
    _require(
        fit_manifest,
        {
            "fit_id",
            "experiment_id",
            "task_id",
            "model_id",
            "stage",
            "accounting",
            "validation_rows",
            "test_rows",
        },
        name="P03 fit manifest",
    )
    if not terminal_fit_ledger.fit_id.astype(str).is_unique:
        raise ValueError("P03 terminal fit ledger contains duplicate fit IDs.")
    manifest_fields = [
        "fit_id",
        "experiment_id",
        "task_id",
        "model_id",
        "stage",
        "accounting",
        "validation_rows",
        "test_rows",
    ]
    planned = fit_manifest[manifest_fields].rename(
        columns={"model_id": "model_id_planned"}
    )
    observed_fields = [
        "fit_id",
        "status",
        "model_id",
        "elapsed_seconds",
        "inference_seconds",
        "prediction_rows",
        "serialized_model_bytes",
    ]
    # Selection status rows also carry planned identity fields such as stage,
    # experiment_id, and task_id. Keep the manifest authoritative for those
    # fields so the mixed selection/outer ledger cannot create merge suffixes.
    observed = terminal_fit_ledger.reindex(columns=observed_fields).rename(
        columns={"model_id": "model_id_observed"}
    )
    joined = planned.merge(
        observed,
        on="fit_id",
        how="left",
        validate="one_to_one",
    )
    if joined.status.isna().any():
        raise ValueError("P03 cost accounting misses a terminal fit record.")
    observed_model = joined.get(
        "model_id_observed", pd.Series(index=joined.index, dtype=object)
    )
    joined["effective_model_id"] = observed_model.fillna(joined.model_id_planned)
    joined["stage_group"] = joined.stage.map(_stage_group)
    joined["elapsed_seconds"] = pd.to_numeric(
        joined.get("elapsed_seconds"), errors="coerce"
    )
    joined["inference_seconds"] = pd.to_numeric(
        joined.get("inference_seconds"), errors="coerce"
    )
    joined["prediction_rows"] = pd.to_numeric(
        joined.get("prediction_rows"), errors="coerce"
    )
    selection_rows = joined.validation_rows.where(
        joined.stage_group.isin({"selection", "calibration"}), 0
    )
    joined["inference_rows"] = joined.prediction_rows.where(
        joined.prediction_rows.gt(0), selection_rows
    )
    joined["milliseconds_per_prediction"] = np.where(
        joined.inference_seconds.notna() & joined.inference_rows.gt(0),
        1000 * joined.inference_seconds / joined.inference_rows,
        np.nan,
    )
    joined["serialized_model_bytes"] = pd.to_numeric(
        joined.get("serialized_model_bytes"), errors="coerce"
    )
    keys = [
        "experiment_id",
        "task_id",
        "effective_model_id",
        "stage_group",
    ]
    rows: list[dict[str, Any]] = []
    for identity, group in joined.groupby(keys, sort=True, dropna=False):
        timed = group.elapsed_seconds.dropna()
        latency = group.milliseconds_per_prediction.dropna()
        sizes = group.serialized_model_bytes.dropna()
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "planned_fit_records": len(group),
                "complete_fit_records": int(group.status.astype(str).eq("complete").sum()),
                "terminal_failure_records": int(
                    (~group.status.astype(str).eq("complete")).sum()
                ),
                "timed_fit_records": len(timed),
                "total_training_seconds": timed.sum() if len(timed) else None,
                "median_training_seconds": timed.median() if len(timed) else None,
                "p95_training_seconds": timed.quantile(0.95) if len(timed) else None,
                "timed_inference_records": len(latency),
                "inference_prediction_rows": int(group.inference_rows.fillna(0).sum()),
                "median_milliseconds_per_prediction": (
                    latency.median() if len(latency) else None
                ),
                "p95_milliseconds_per_prediction": (
                    latency.quantile(0.95) if len(latency) else None
                ),
                "sized_model_records": len(sizes),
                "median_serialized_model_bytes": sizes.median() if len(sizes) else None,
                "maximum_serialized_model_bytes": sizes.max() if len(sizes) else None,
            }
        )
    return pd.DataFrame(rows)


def build_p03_diagnostic_tables(
    *,
    selections: pd.DataFrame,
    selection_traces: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    final_predictions: pd.DataFrame,
    master_predictions: pd.DataFrame,
    terminal_fit_ledger: pd.DataFrame,
    fit_manifest: pd.DataFrame,
) -> P03DiagnosticTables:
    """Build every registered post-fit diagnostic from frozen private evidence."""

    frequency, stability, margins = build_selection_diagnostics(
        selections, selection_traces
    )
    return P03DiagnosticTables(
        selection_frequency=frequency,
        selection_stability=stability,
        selection_margins=margins,
        endpoint_coverage=build_endpoint_coverage(pooled_metrics),
        t1_t3_comparison=build_t1_t3_comparison(pooled_metrics),
        spectrum_master_comparison=build_spectrum_master_comparison(pooled_metrics),
        confusion=build_confusion_summary(pooled_metrics),
        reliability=build_reliability_summary(final_predictions, master_predictions),
        control_summary=build_control_summary(pooled_metrics),
        cost_summary=build_cost_summary(terminal_fit_ledger, fit_manifest),
    )
