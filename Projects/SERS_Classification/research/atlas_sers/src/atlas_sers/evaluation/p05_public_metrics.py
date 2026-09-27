"""Strict, anonymous public P05 metric tables for downstream rendering.

This module is pure and in-memory: it never reads a file, touches torch, fits,
infers or selects. Callers authenticate the frozen P05 aggregation and
comparison tables before handing them in. Only allowlisted public columns are
emitted; private identifiers such as ``observation_uid``,
``master_sample_id``, ``context_id``, ``test_uid`` hashes, ``true_label``,
predictions, class vocabularies and source paths are never copied. Context
identity is replaced by an anonymous 1-based ``point_index`` derived from the
sorted private context IDs.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from atlas_sers.evaluation import p05_comparison

__all__ = [
    "P05PublicMetricsError",
    "build_public_metrics",
    "P05_PUBLIC_TABLE_NAMES",
    "P05_PUBLIC_MODELS",
    "P05_PUBLIC_STATIONS",
    "P05_PUBLIC_AGGREGATIONS",
    "P05_PUBLIC_METRICS",
    "P05_PUBLIC_PAIRED_METRICS",
    "P05_PUBLIC_PHASES",
]

P05_PUBLIC_MODELS = ("D0-M", "P05-SELECTED", "D3")
P05_PUBLIC_AGGREGATIONS = ("M01", "M06")
P05_PUBLIC_STATIONS = ("cwa", "pills", "surfaces")
P05_PUBLIC_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "negative_log_likelihood",
    "brier_score",
    "ece",
)
P05_PUBLIC_PAIRED_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "negative_log_likelihood",
    "ece",
)
P05_PUBLIC_PHASES = ("development", "held_evaluation")
PHASE_BY_EXPERIMENT = {"P05-CORE-DEV": "development", "P05-CORE-T3": "held_evaluation"}
_UNIT_INTERVAL = ("balanced_accuracy", "macro_f1", "ece")
_NON_NEGATIVE = ("negative_log_likelihood", "brier_score")
_COUNTS = ("observations", "physical_masters")

P05_PUBLIC_TABLE_NAMES = (
    "strategy_contexts",
    "strategy_domains",
    "strategy_summary",
    "paired_contexts",
    "paired_domains",
    "comparison_summary",
    "coverage",
)

_STRATEGY_SOURCE_COLUMNS = (
    "context_id",
    "experiment_id",
    "station",
    "domain",
    "held_instrument",
    "model_id",
    "observed_class_count",
    *_COUNTS,
    *P05_PUBLIC_METRICS,
)
_STRATEGY_CONTEXT_COLUMNS = (
    "point_index",
    "station",
    "phase",
    "domain",
    "held_instrument",
    "model_id",
    "aggregation_id",
    *P05_PUBLIC_METRICS,
    *_COUNTS,
    "observed_class_count",
)
_STRATEGY_GROUP = ("station", "phase", "domain", "held_instrument", "model_id", "aggregation_id")
_STRATEGY_SUMMARY_GROUP = ("station", "phase", "model_id", "aggregation_id")
_PAIRED_SOURCE_COLUMNS = (
    "context_id",
    "station",
    "domain",
    "held_instrument",
    "model_id",
    "reference_model_id",
    "aggregation_id",
    "model_complete",
    "reference_complete",
    "common_complete",
    *(f"model_{metric}" for metric in P05_PUBLIC_PAIRED_METRICS),
    *(f"reference_{metric}" for metric in P05_PUBLIC_PAIRED_METRICS),
    *(f"delta_{metric}" for metric in P05_PUBLIC_PAIRED_METRICS),
)
_PAIRED_CONTEXT_COLUMNS = (
    "point_index",
    "station",
    "domain",
    "held_instrument",
    "model_id",
    "reference_model_id",
    "aggregation_id",
    "model_complete",
    "reference_complete",
    "common_complete",
    *(f"model_{metric}" for metric in P05_PUBLIC_PAIRED_METRICS),
    *(f"reference_{metric}" for metric in P05_PUBLIC_PAIRED_METRICS),
    *(f"delta_{metric}" for metric in P05_PUBLIC_PAIRED_METRICS),
)
_PAIRED_GROUP = (
    "station",
    "domain",
    "held_instrument",
    "model_id",
    "reference_model_id",
    "aggregation_id",
)
_COMPARISON_SUMMARY_COLUMNS = (
    "station",
    "model_id",
    "reference_model_id",
    "aggregation_id",
    "planned_contexts",
    "common_complete_contexts",
    "common_coverage",
    "planned_domains",
    "contributing_common_domains",
    "model_mean_balanced_accuracy_equal_contexts",
    "reference_mean_balanced_accuracy_equal_contexts",
    "model_mean_balanced_accuracy_equal_domains",
    "reference_mean_balanced_accuracy_equal_domains",
    "mean_delta_balanced_accuracy",
    "mean_delta_macro_f1",
    "mean_delta_negative_log_likelihood",
    "mean_delta_ece",
    "model_worst_domain_balanced_accuracy_common",
    "reference_worst_domain_balanced_accuracy_common",
    "model_failure_sensitive_mean_balanced_accuracy_missing_as_zero",
    "reference_failure_sensitive_mean_balanced_accuracy_missing_as_zero",
    "failure_sensitive_missing_policy",
)
_COMPARISON_SUMMARY_KEY = ("station", "model_id", "reference_model_id", "aggregation_id")
_COMPARISON_SUMMARY_TEXT = ("failure_sensitive_missing_policy",)
_COMPARISON_SUMMARY_INT = (
    "planned_contexts",
    "common_complete_contexts",
    "planned_domains",
    "contributing_common_domains",
)
_COMPARISON_SUMMARY_FLOAT = tuple(
    column
    for column in _COMPARISON_SUMMARY_COLUMNS
    if column not in _COMPARISON_SUMMARY_KEY
    and column not in _COMPARISON_SUMMARY_TEXT
    and column not in _COMPARISON_SUMMARY_INT
)
_COVERAGE_COLUMNS = (
    "station",
    "model_id",
    "expected_contexts",
    "complete_contexts",
    "incomplete_contexts",
    "complete_coverage",
)


class P05PublicMetricsError(ValueError):
    """Raised when public P05 metric inputs are missing or inconsistent."""


def _mapping(value, code):
    if not isinstance(value, Mapping):
        raise P05PublicMetricsError(code)
    return value


def _frame(tables, key, code):
    if key not in tables:
        raise P05PublicMetricsError(code)
    frame = tables[key]
    if not isinstance(frame, pd.DataFrame):
        raise P05PublicMetricsError(code)
    return frame.copy()


def _require(frame, columns, code):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise P05PublicMetricsError(f"{code}:{','.join(missing)}")


def _text(value, code):
    if not isinstance(value, str) or not value or value != value.strip():
        raise P05PublicMetricsError(code)
    return value


def _instrument(value, code):
    if not isinstance(value, str) or value != value.strip():
        raise P05PublicMetricsError(code)
    return value


def _text_column(frame, column, code, *, allow_blank=False):
    validator = _instrument if allow_blank else _text
    for value in frame[column]:
        validator(value, code)


def _station_column(frame, column, code):
    for value in frame[column]:
        if not isinstance(value, str) or value not in P05_PUBLIC_STATIONS:
            raise P05PublicMetricsError(code)


def _integer(value, code):
    if isinstance(value, (bool, np.bool_)) or value is None or value is pd.NA:
        raise P05PublicMetricsError(code)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return int(number)
        raise P05PublicMetricsError(code)
    if isinstance(value, str):
        if value and value == value.strip():
            try:
                number = int(value)
            except ValueError:
                raise P05PublicMetricsError(code) from None
            if str(number) == value:
                return number
    raise P05PublicMetricsError(code)


def _strict_bool(value, code):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise P05PublicMetricsError(code)


def _missing(value):
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(float(value)))
    return False


def _float_array(frame, column, code):
    values = []
    for value in frame[column].tolist():
        if isinstance(value, (bool, np.bool_)):
            raise P05PublicMetricsError(code)
        if _missing(value):
            raise P05PublicMetricsError(code)
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise P05PublicMetricsError(code) from None
        if not np.isfinite(number):
            raise P05PublicMetricsError(code)
        if column in _UNIT_INTERVAL and not 0.0 <= number <= 1.0:
            raise P05PublicMetricsError(code)
        if column in _NON_NEGATIVE and number < 0.0:
            raise P05PublicMetricsError(code)
        values.append(number)
    return np.asarray(values, dtype=float)


def _check_score(value, complete, metric, code):
    if isinstance(value, (bool, np.bool_)):
        raise P05PublicMetricsError(code)
    if complete:
        if _missing(value):
            raise P05PublicMetricsError(code)
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise P05PublicMetricsError(code) from None
        if not np.isfinite(number):
            raise P05PublicMetricsError(code)
        if metric in _UNIT_INTERVAL and not 0.0 <= number <= 1.0:
            raise P05PublicMetricsError(code)
        if metric in _NON_NEGATIVE and number < 0.0:
            raise P05PublicMetricsError(code)
    elif not _missing(value):
        raise P05PublicMetricsError(code)


def _context_index(spectrum, master):
    identifiers = None
    for frame in (spectrum, master):
        if "context_id" not in frame.columns:
            raise P05PublicMetricsError("context_id_missing")
        current = {str(value) for value in frame["context_id"].tolist()}
        if identifiers is None:
            identifiers = current
        elif current != identifiers:
            raise P05PublicMetricsError("strategy_context_mismatch")
    if not identifiers:
        raise P05PublicMetricsError("contexts_empty")
    return {identifier: index for index, identifier in enumerate(sorted(identifiers), start=1)}


def _strategy_contexts(spectrum, master, index):
    frames = []
    for aggregation_id, frame in (("M01", spectrum), ("M06", master)):
        _require(frame, _STRATEGY_SOURCE_COLUMNS, "strategy_columns_missing")
        if "aggregation_id" in frame.columns:
            if not frame["aggregation_id"].astype(str).eq(aggregation_id).all():
                raise P05PublicMetricsError("strategy_aggregation_mismatch")
        prepared = frame.copy()
        prepared["aggregation_id"] = aggregation_id
        frames.append(prepared)
    combined = pd.concat(frames, ignore_index=True)
    for column in ("context_id", "domain", "model_id"):
        _text_column(combined, column, f"strategy_{column}_invalid")
    _station_column(combined, "station", "strategy_station_unknown")
    _text_column(combined, "held_instrument", "strategy_held_instrument_invalid", allow_blank=True)
    if not combined["model_id"].isin(P05_PUBLIC_MODELS).all():
        raise P05PublicMetricsError("strategy_unknown_model")
    phase = combined["experiment_id"].map(PHASE_BY_EXPERIMENT)
    if phase.isna().any():
        raise P05PublicMetricsError("strategy_unknown_phase")
    combined["phase"] = phase
    for column in (*_COUNTS, "observed_class_count"):
        combined[column] = [
            _integer(value, f"strategy_{column}_invalid") for value in combined[column]
        ]
    if (combined["observations"] <= 0).any() or (combined["physical_masters"] <= 0).any():
        raise P05PublicMetricsError("strategy_count_nonpositive")
    if (combined["physical_masters"] > combined["observations"]).any():
        raise P05PublicMetricsError("strategy_master_exceeds_observations")
    m06 = combined["aggregation_id"].eq("M06")
    if (
        combined.loc[m06, "observations"].to_numpy()
        != combined.loc[m06, "physical_masters"].to_numpy()
    ).any():
        raise P05PublicMetricsError("strategy_m06_count_mismatch")
    if not combined["observed_class_count"].between(1, 3).all():
        raise P05PublicMetricsError("strategy_observed_class_count_invalid")
    for column in P05_PUBLIC_METRICS:
        combined[column] = _float_array(combined, column, f"strategy_{column}_invalid")
    if combined.duplicated(["context_id", "model_id", "aggregation_id"]).any():
        raise P05PublicMetricsError("strategy_duplicate_key")
    expected = {
        (model, aggregation)
        for model in P05_PUBLIC_MODELS
        for aggregation in P05_PUBLIC_AGGREGATIONS
    }
    for _, cell in combined.groupby("context_id", sort=False):
        if set(zip(cell["model_id"], cell["aggregation_id"], strict=True)) != expected:
            raise P05PublicMetricsError("strategy_context_incomplete")
        for column in ("station", "phase", "domain", "held_instrument"):
            if cell[column].nunique(dropna=False) != 1:
                raise P05PublicMetricsError("strategy_context_metadata_conflict")
        for _, aggregation_cell in cell.groupby("aggregation_id", sort=False):
            if aggregation_cell["observed_class_count"].nunique() != 1:
                raise P05PublicMetricsError("strategy_class_count_conflict")
            if (
                aggregation_cell["observations"].nunique() != 1
                or aggregation_cell["physical_masters"].nunique() != 1
            ):
                raise P05PublicMetricsError("strategy_count_conflict")
    held_ids = {
        str(row.context_id)
        for row in combined.itertuples(index=False)
        if row.phase == "held_evaluation"
    }
    if not held_ids:
        raise P05PublicMetricsError("strategy_held_contexts_empty")
    output = combined.copy()
    output["point_index"] = output["context_id"].map(index)
    if output["point_index"].isna().any():
        raise P05PublicMetricsError("strategy_context_unindexed")
    output["point_index"] = output["point_index"].astype(int)
    output = output[list(_STRATEGY_CONTEXT_COLUMNS)]
    output = output.sort_values(
        ["point_index", "model_id", "aggregation_id"], kind="stable"
    ).reset_index(drop=True)
    return output, held_ids


def _strategy_domains(contexts):
    rows = []
    for keys, cell in contexts.groupby(list(_STRATEGY_GROUP), sort=True, dropna=False):
        record = dict(zip(_STRATEGY_GROUP, keys, strict=True))
        record["completed_contexts"] = int(cell["point_index"].nunique())
        counts = cell["observed_class_count"].value_counts()
        record["contexts_with_1_observed_class"] = int(counts.get(1, 0))
        record["contexts_with_2_observed_classes"] = int(counts.get(2, 0))
        record["contexts_with_3_observed_classes"] = int(counts.get(3, 0))
        record["min_test_masters"] = int(cell["physical_masters"].min())
        record["max_test_masters"] = int(cell["physical_masters"].max())
        for metric in P05_PUBLIC_METRICS:
            record[f"mean_{metric}_equal_contexts"] = float(cell[metric].mean())
        rows.append(record)
    columns = [
        *_STRATEGY_GROUP,
        "completed_contexts",
        "contexts_with_1_observed_class",
        "contexts_with_2_observed_classes",
        "contexts_with_3_observed_classes",
        "min_test_masters",
        "max_test_masters",
        *(f"mean_{metric}_equal_contexts" for metric in P05_PUBLIC_METRICS),
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(list(_STRATEGY_GROUP), kind="stable").reset_index(drop=True)


def _strategy_summary(contexts):
    rows = []
    for keys, cell in contexts.groupby(list(_STRATEGY_SUMMARY_GROUP), sort=True, dropna=False):
        record = dict(zip(_STRATEGY_SUMMARY_GROUP, keys, strict=True))
        record["contexts"] = int(cell["point_index"].nunique())
        record["domains"] = int(cell["domain"].nunique())
        for metric in P05_PUBLIC_METRICS:
            record[f"mean_{metric}_equal_contexts"] = float(cell[metric].mean())
            record[f"mean_{metric}_equal_domains"] = float(
                cell.groupby("domain")[metric].mean().mean()
            )
        record["worst_domain_balanced_accuracy"] = float(
            cell.groupby("domain")["balanced_accuracy"].mean().min()
        )
        rows.append(record)
    columns = [
        *_STRATEGY_SUMMARY_GROUP,
        "contexts",
        "domains",
        *(f"mean_{metric}_equal_contexts" for metric in P05_PUBLIC_METRICS),
        *(f"mean_{metric}_equal_domains" for metric in P05_PUBLIC_METRICS),
        "worst_domain_balanced_accuracy",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(list(_STRATEGY_SUMMARY_GROUP), kind="stable").reset_index(drop=True)


def _paired_contexts(paired, index, contexts, held_ids):
    _require(paired, _PAIRED_SOURCE_COLUMNS, "paired_columns_missing")
    frame = paired.copy()
    for column in ("context_id", "domain", "model_id", "reference_model_id"):
        _text_column(frame, column, f"paired_{column}_invalid")
    _station_column(frame, "station", "paired_station_unknown")
    _text_column(frame, "held_instrument", "paired_held_instrument_invalid", allow_blank=True)
    if not frame["model_id"].isin(P05_PUBLIC_MODELS).all():
        raise P05PublicMetricsError("paired_unknown_model")
    if not frame["reference_model_id"].isin(p05_comparison.ALL_MODELS).all():
        raise P05PublicMetricsError("paired_unknown_reference")
    if not frame["aggregation_id"].isin(P05_PUBLIC_AGGREGATIONS).all():
        raise P05PublicMetricsError("paired_unknown_aggregation")
    if not set(zip(frame["model_id"], frame["reference_model_id"], strict=True)) <= set(
        p05_comparison.PAIRS
    ):
        raise P05PublicMetricsError("paired_unknown_pair")
    if frame.duplicated(["context_id", "model_id", "reference_model_id", "aggregation_id"]).any():
        raise P05PublicMetricsError("paired_duplicate_key")
    if {str(value) for value in frame["context_id"].tolist()} != held_ids:
        raise P05PublicMetricsError("paired_context_set_mismatch")
    expected_pairs = {
        (model, reference, aggregation)
        for model, reference in p05_comparison.PAIRS
        for aggregation in P05_PUBLIC_AGGREGATIONS
    }
    meta_lookup = contexts.drop_duplicates("point_index").set_index("point_index")
    score_lookup = {}
    for row in contexts.itertuples(index=False):
        score_lookup[(int(row.point_index), row.model_id, row.aggregation_id)] = {
            metric: float(getattr(row, metric)) for metric in P05_PUBLIC_PAIRED_METRICS
        }
    for context_id, cell in frame.groupby("context_id", sort=False):
        if (
            set(
                zip(
                    cell["model_id"],
                    cell["reference_model_id"],
                    cell["aggregation_id"],
                    strict=True,
                )
            )
            != expected_pairs
        ):
            raise P05PublicMetricsError("paired_context_incomplete")
        for column in ("station", "domain", "held_instrument"):
            if cell[column].nunique(dropna=False) != 1:
                raise P05PublicMetricsError("paired_context_metadata_conflict")
        expected_meta = meta_lookup.loc[index[str(context_id)]]
        for column in ("station", "domain", "held_instrument"):
            if str(cell[column].iloc[0]) != str(expected_meta[column]):
                raise P05PublicMetricsError("paired_strategy_metadata_mismatch")
    for row in frame.itertuples(index=False):
        model_complete = _strict_bool(row.model_complete, "paired_model_complete_invalid")
        if not model_complete:
            raise P05PublicMetricsError("paired_complete_strategy_missing")
        reference_complete = _strict_bool(
            row.reference_complete, "paired_reference_complete_invalid"
        )
        common_complete = _strict_bool(row.common_complete, "paired_common_complete_invalid")
        if common_complete != (model_complete and reference_complete):
            raise P05PublicMetricsError("paired_completeness_inconsistent")
        for metric in P05_PUBLIC_PAIRED_METRICS:
            model_value = getattr(row, f"model_{metric}")
            reference_value = getattr(row, f"reference_{metric}")
            delta_value = getattr(row, f"delta_{metric}")
            _check_score(model_value, model_complete, metric, f"paired_model_{metric}_invalid")
            _check_score(
                reference_value, reference_complete, metric, f"paired_reference_{metric}_invalid"
            )
            if common_complete:
                if isinstance(delta_value, (bool, np.bool_)) or _missing(delta_value):
                    raise P05PublicMetricsError(f"paired_delta_{metric}_invalid")
                try:
                    delta_number = float(delta_value)
                except (TypeError, ValueError):
                    raise P05PublicMetricsError(f"paired_delta_{metric}_invalid") from None
                if not np.isfinite(delta_number) or not np.isclose(
                    delta_number,
                    float(model_value) - float(reference_value),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise P05PublicMetricsError(f"paired_delta_{metric}_mismatch")
            elif not _missing(delta_value):
                raise P05PublicMetricsError(f"paired_delta_{metric}_unexpected")
        for side, model in (("model", row.model_id), ("reference", row.reference_model_id)):
            if model not in P05_PUBLIC_MODELS:
                continue
            if side == "reference" and not reference_complete:
                raise P05PublicMetricsError("paired_complete_strategy_missing")
            expected_score = score_lookup.get(
                (index[str(row.context_id)], model, row.aggregation_id)
            )
            if expected_score is None:
                raise P05PublicMetricsError("paired_strategy_score_missing")
            for metric in P05_PUBLIC_PAIRED_METRICS:
                if not np.isclose(
                    float(getattr(row, f"{side}_{metric}")),
                    expected_score[metric],
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise P05PublicMetricsError("paired_strategy_score_mismatch")
    output = frame.copy()
    for column in ("model_complete", "reference_complete", "common_complete"):
        output[column] = output[column].map(bool)
    for metric in P05_PUBLIC_PAIRED_METRICS:
        for side in ("model", "reference", "delta"):
            output[f"{side}_{metric}"] = output[f"{side}_{metric}"].astype(float)
    output["point_index"] = output["context_id"].map(index)
    if output["point_index"].isna().any():
        raise P05PublicMetricsError("paired_context_unindexed")
    output["point_index"] = output["point_index"].astype(int)
    output = output[list(_PAIRED_CONTEXT_COLUMNS)]
    return output.sort_values(
        ["point_index", "model_id", "reference_model_id", "aggregation_id"], kind="stable"
    ).reset_index(drop=True)


def _paired_domains(paired_contexts):
    rows = []
    for keys, cell in paired_contexts.groupby(list(_PAIRED_GROUP), sort=True, dropna=False):
        record = dict(zip(_PAIRED_GROUP, keys, strict=True))
        record["planned_contexts"] = int(len(cell))
        common = cell[cell["common_complete"]]
        record["common_contexts"] = int(len(common))
        record["common_coverage"] = float(len(common) / len(cell)) if len(cell) else np.nan
        for metric in P05_PUBLIC_PAIRED_METRICS:
            if common.empty:
                record[f"mean_model_{metric}"] = np.nan
                record[f"mean_reference_{metric}"] = np.nan
                record[f"mean_delta_{metric}"] = np.nan
            else:
                record[f"mean_model_{metric}"] = float(common[f"model_{metric}"].mean())
                record[f"mean_reference_{metric}"] = float(common[f"reference_{metric}"].mean())
                record[f"mean_delta_{metric}"] = float(common[f"delta_{metric}"].mean())
        rows.append(record)
    columns = [
        *_PAIRED_GROUP,
        "planned_contexts",
        "common_contexts",
        "common_coverage",
        *(f"mean_model_{metric}" for metric in P05_PUBLIC_PAIRED_METRICS),
        *(f"mean_reference_{metric}" for metric in P05_PUBLIC_PAIRED_METRICS),
        *(f"mean_delta_{metric}" for metric in P05_PUBLIC_PAIRED_METRICS),
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values(list(_PAIRED_GROUP), kind="stable").reset_index(drop=True)


def _comparison_summary(summary, paired_contexts):
    _require(summary, _COMPARISON_SUMMARY_COLUMNS, "comparison_summary_columns_missing")
    frame = summary.copy()
    _station_column(frame, "station", "comparison_summary_station_unknown")
    _text_column(frame, "reference_model_id", "comparison_summary_reference_invalid")
    if not frame["model_id"].isin(P05_PUBLIC_MODELS).all():
        raise P05PublicMetricsError("comparison_summary_unknown_model")
    if not frame["reference_model_id"].isin(p05_comparison.ALL_MODELS).all():
        raise P05PublicMetricsError("comparison_summary_unknown_reference")
    if not frame["aggregation_id"].isin(P05_PUBLIC_AGGREGATIONS).all():
        raise P05PublicMetricsError("comparison_summary_unknown_aggregation")
    if not set(zip(frame["model_id"], frame["reference_model_id"], strict=True)) <= set(
        p05_comparison.PAIRS
    ):
        raise P05PublicMetricsError("comparison_summary_unknown_pair")
    if frame.duplicated(list(_COMPARISON_SUMMARY_KEY)).any():
        raise P05PublicMetricsError("comparison_summary_duplicate_key")
    provided = (
        frame[list(_COMPARISON_SUMMARY_COLUMNS)]
        .sort_values(list(_COMPARISON_SUMMARY_KEY), kind="stable")
        .reset_index(drop=True)
    )
    expected = p05_comparison._summary(paired_contexts)
    expected = (
        expected[list(_COMPARISON_SUMMARY_COLUMNS)]
        .sort_values(list(_COMPARISON_SUMMARY_KEY), kind="stable")
        .reset_index(drop=True)
    )
    if len(provided) != len(expected):
        raise P05PublicMetricsError("comparison_summary_row_mismatch")
    for column in _COMPARISON_SUMMARY_KEY:
        if provided[column].tolist() != expected[column].tolist():
            raise P05PublicMetricsError("comparison_summary_key_mismatch")
    for column in _COMPARISON_SUMMARY_TEXT:
        if provided[column].tolist() != expected[column].tolist():
            raise P05PublicMetricsError("comparison_summary_policy_mismatch")
    for column in _COMPARISON_SUMMARY_INT:
        provided[column] = [
            _integer(value, "comparison_summary_count_invalid") for value in provided[column]
        ]
        if provided[column].tolist() != expected[column].tolist():
            raise P05PublicMetricsError("comparison_summary_count_mismatch")
    for column in _COMPARISON_SUMMARY_FLOAT:
        if any(isinstance(value, (bool, np.bool_)) for value in provided[column]):
            raise P05PublicMetricsError("comparison_summary_numeric_invalid")
        try:
            left = provided[column].to_numpy(dtype=float)
            right = expected[column].to_numpy(dtype=float)
        except (TypeError, ValueError):
            raise P05PublicMetricsError("comparison_summary_numeric_invalid") from None
        if not np.allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=True):
            raise P05PublicMetricsError("comparison_summary_numeric_mismatch")
        provided[column] = left
    return provided


def _coverage(coverage, contexts, index, held_ids, paired):
    _require(
        coverage,
        ("context_id", "station", "domain", "held_instrument", "model_id", "complete"),
        "comparison_coverage_columns_missing",
    )
    frame = coverage.copy()
    _text_column(frame, "context_id", "coverage_context_invalid")
    _station_column(frame, "station", "coverage_station_unknown")
    _text_column(frame, "domain", "coverage_domain_invalid")
    _text_column(frame, "held_instrument", "coverage_held_instrument_invalid", allow_blank=True)
    _text_column(frame, "model_id", "coverage_model_invalid")
    if not frame["model_id"].isin(p05_comparison.ALL_MODELS).all():
        raise P05PublicMetricsError("coverage_unknown_model")
    for value in frame["complete"]:
        _strict_bool(value, "coverage_complete_invalid")
    if frame.duplicated(["context_id", "model_id"]).any():
        raise P05PublicMetricsError("coverage_duplicate_key")
    if {str(value) for value in frame["context_id"].tolist()} != held_ids:
        raise P05PublicMetricsError("coverage_context_set_mismatch")
    expected_models = set(p05_comparison.ALL_MODELS)
    meta_lookup = contexts.drop_duplicates("point_index").set_index("point_index")
    for context_id, cell in frame.groupby("context_id", sort=False):
        if set(cell["model_id"]) != expected_models:
            raise P05PublicMetricsError("coverage_context_incomplete")
        expected = meta_lookup.loc[index[str(context_id)]]
        for column in ("station", "domain", "held_instrument"):
            if not cell[column].eq(expected[column]).all():
                raise P05PublicMetricsError("coverage_strategy_metadata_mismatch")
    coverage_lookup = frame.set_index(["context_id", "model_id"])["complete"]
    for row in paired.itertuples(index=False):
        for side, model in (("model", row.model_id), ("reference", row.reference_model_id)):
            if bool(coverage_lookup.loc[(row.context_id, model)]) != bool(
                getattr(row, f"{side}_complete")
            ):
                raise P05PublicMetricsError("coverage_paired_completeness_mismatch")
    rows = []
    for keys, cell in frame.groupby(["station", "model_id"], sort=True, dropna=False):
        station, model_id = keys
        expected = int(len(cell))
        complete = int(sum(bool(value) for value in cell["complete"]))
        rows.append(
            {
                "station": station,
                "model_id": model_id,
                "expected_contexts": expected,
                "complete_contexts": complete,
                "incomplete_contexts": expected - complete,
                "complete_coverage": float(complete / expected) if expected else np.nan,
            }
        )
    frame_out = pd.DataFrame(rows, columns=list(_COVERAGE_COLUMNS))
    return frame_out.sort_values(["station", "model_id"], kind="stable").reset_index(drop=True)


def build_public_metrics(*, aggregation_tables, comparison_tables) -> dict[str, pd.DataFrame]:
    """Build strict, anonymous public P05 metric tables from authenticated inputs."""
    aggregation = _mapping(aggregation_tables, "aggregation_tables_malformed")
    comparison = _mapping(comparison_tables, "comparison_tables_malformed")
    spectrum = _frame(aggregation, "spectrum_metrics", "aggregation_spectrum_missing")
    master = _frame(aggregation, "master_metrics", "aggregation_master_missing")
    paired = _frame(comparison, "paired_metrics", "comparison_paired_missing")
    coverage = _frame(comparison, "coverage", "comparison_coverage_missing")
    summary = _frame(comparison, "summary", "comparison_summary_missing")
    index = _context_index(spectrum, master)
    contexts, held_ids = _strategy_contexts(spectrum, master, index)
    paired_contexts = _paired_contexts(paired, index, contexts, held_ids)
    return {
        "strategy_contexts": contexts,
        "strategy_domains": _strategy_domains(contexts),
        "strategy_summary": _strategy_summary(contexts),
        "paired_contexts": paired_contexts,
        "paired_domains": _paired_domains(paired_contexts),
        "comparison_summary": _comparison_summary(summary, paired_contexts),
        "coverage": _coverage(coverage, contexts, index, held_ids, paired),
    }
