"""Private release metric-table adapter for T360.

The adapter copies five existing aggregate tables and derives the descriptive
class-recall summary from stored counts. Primary scores are never recomputed.
No context/master/observation/unit identifier is exported.
"""

from __future__ import annotations

import math
import numbers

import numpy as np
import pandas as pd

from atlas_sers.evaluation import p08_universal_analysis as _universal

ESTIMANDS = _universal.ESTIMANDS
POLICIES = _universal.POLICIES
ENDPOINTS = _universal.ENDPOINTS
MODELS = _universal.MODELS

__all__ = ["prepare_metric_tables"]

_METRIC_COLUMNS = (
    "balanced_accuracy",
    "macro_f1",
    "negative_log_likelihood",
    "brier_score",
    "ece",
)

_LEADING_LABELS = ("estimand", "policy_id", "endpoint")

_DIRECT_TABLES = (
    "model_summary",
    "domain_metrics",
    "confusion",
    "reliability_bins",
    "equal_context_reliability",
)

_EXPECTED_COLUMNS = {
    "model_summary": [
        "model_id",
        "contexts",
        "domains",
        "unit_appearances",
        "physical_masters",
        "distinct_units",
        *_METRIC_COLUMNS,
    ],
    "domain_metrics": [
        "model_id",
        "domain",
        "station",
        "instrument",
        "contexts",
        "unit_appearances",
        "physical_masters",
        "distinct_units",
        *_METRIC_COLUMNS,
    ],
    "confusion": [
        "scope",
        "station",
        "model_id",
        "true_label",
        "predicted_label",
        "count",
        "true_appearances",
        "row_fraction",
    ],
    "reliability_bins": [
        "scope",
        "station",
        "model_id",
        "bin_index",
        "bin_lower",
        "bin_upper",
        "count",
        "sum_confidence",
        "sum_correct",
        "mean_confidence",
        "accuracy",
    ],
    "equal_context_reliability": [
        "scope",
        "model_id",
        "contexts",
        "equal_context_ece",
    ],
}

_CLASS_RECALL_COLUMNS = [
    "model_id",
    "context_id",
    "domain",
    "station",
    "class_label",
    "correct",
    "support",
    "recall",
]

_LABEL_COLUMNS = {
    "model_summary": ("model_id",),
    "domain_metrics": ("model_id", "domain", "station", "instrument"),
    "confusion": ("scope", "station", "model_id", "true_label", "predicted_label"),
    "reliability_bins": ("scope", "station", "model_id"),
    "equal_context_reliability": ("scope", "model_id"),
    "class_sensitivity": ("model_id", "domain", "station", "class_label"),
}

_CLASS_RECALL_LABELS = ("model_id", "context_id", "domain", "station", "class_label")

_SORT_KEYS = {
    "model_summary": ("estimand", "policy_id", "endpoint", "model_id"),
    "domain_metrics": ("estimand", "policy_id", "endpoint", "model_id", "domain"),
    "confusion": (
        "estimand",
        "policy_id",
        "endpoint",
        "scope",
        "station",
        "model_id",
        "true_label",
        "predicted_label",
    ),
    "reliability_bins": (
        "estimand",
        "policy_id",
        "endpoint",
        "scope",
        "station",
        "model_id",
        "bin_index",
    ),
    "equal_context_reliability": (
        "estimand",
        "policy_id",
        "endpoint",
        "scope",
        "model_id",
    ),
    "class_sensitivity": (
        "estimand",
        "policy_id",
        "endpoint",
        "model_id",
        "domain",
        "station",
        "class_label",
    ),
}

_CLASS_SENSITIVITY_COLUMNS = [
    "estimand",
    "policy_id",
    "endpoint",
    "model_id",
    "domain",
    "station",
    "class_label",
    "planned_contexts",
    "supported_contexts",
    "sum_correct",
    "sum_support",
    "pooled_repeated_appearance_recall",
    "mean_supported_context_recall",
]


_MISSING_COLUMNS = {
    "class_recall": {"recall"},
    "confusion": {"row_fraction"},
    "reliability_bins": {"mean_confidence", "accuracy"},
}
_COUNT_COLUMNS = {
    "contexts",
    "domains",
    "unit_appearances",
    "physical_masters",
    "distinct_units",
    "count",
    "true_appearances",
    "bin_index",
    "sum_correct",
    "correct",
    "support",
}


def _check_numeric_value(table: str, column: str, value: object) -> None:
    missing_allowed = column in _MISSING_COLUMNS.get(table, set())
    if value is None or value is pd.NA:
        if missing_allowed:
            return
        raise ValueError(f"{table}: missing required number in {column}")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{table}: {column} must contain real numeric scalars")
    number = float(value)
    if math.isnan(number) and missing_allowed:
        return
    if not math.isfinite(number):
        raise ValueError(f"{table}: nonfinite required number in {column}")
    if column in _COUNT_COLUMNS and (number < 0 or not number.is_integer()):
        raise ValueError(f"{table}: invalid count in {column}")


def _check_numeric_column(table: str, column: str, series: pd.Series) -> None:
    for value in series:
        _check_numeric_value(table, column, value)


def _require_columns(
    table: str,
    frame: pd.DataFrame,
    expected: list[str],
    labels: tuple[str, ...],
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{table}: expected a DataFrame, got {type(frame)!r}")
    if frame.columns.duplicated().any():
        raise ValueError(f"{table}: duplicate columns {list(frame.columns)!r}")
    present = set(frame.columns)
    if present != set(expected):
        missing = sorted(set(expected) - present)
        unexpected = sorted(present - set(expected))
        raise ValueError(
            f"{table}: expected columns {list(expected)!r}; "
            f"missing {missing!r}; unexpected {unexpected!r}"
        )
    for column in expected:
        series = frame[column]
        if column in labels:
            for value in series:
                if not isinstance(value, str):
                    raise TypeError(f"{table}: label column {column!r} holds non-text {value!r}")
        else:
            _check_numeric_column(table, column, series)
    if "model_id" in expected:
        if set(frame["model_id"]) != set(MODELS):
            raise ValueError(f"{table}: incomplete or unexpected model identities")


def _class_sensitivity(
    recall: pd.DataFrame,
    estimand: str,
    policy_id: str,
    endpoint: str,
) -> pd.DataFrame:
    _require_columns("class_recall", recall, _CLASS_RECALL_COLUMNS, _CLASS_RECALL_LABELS)
    if recall.duplicated(["model_id", "context_id", "class_label"]).any():
        raise ValueError("class_recall: duplicate context/class row")
    if (recall["correct"] > recall["support"]).any():
        raise ValueError("class_recall: correct exceeds support")
    if not recall.loc[recall.support.eq(0), "recall"].isna().all():
        raise ValueError("class_recall: absent class must have undefined recall")
    if recall.loc[recall.support.gt(0), "recall"].isna().any():
        raise ValueError("class_recall: supported class must have defined recall")
    frame = recall.copy()
    frame["estimand"] = estimand
    frame["policy_id"] = policy_id
    frame["endpoint"] = endpoint
    keys = [
        "estimand",
        "policy_id",
        "endpoint",
        "model_id",
        "domain",
        "station",
        "class_label",
    ]
    grouped = frame.groupby(keys, as_index=False, sort=False).agg(
        planned_contexts=("recall", "size"),
        supported_contexts=("support", lambda series: int(series.gt(0).sum())),
        sum_correct=("correct", "sum"),
        sum_support=("support", "sum"),
        mean_supported_context_recall=("recall", "mean"),
    )
    grouped["sum_correct"] = grouped["sum_correct"].astype(int)
    grouped["sum_support"] = grouped["sum_support"].astype(int)
    denominator = grouped["sum_support"].where(grouped["sum_support"].gt(0))
    grouped["pooled_repeated_appearance_recall"] = grouped["sum_correct"] / denominator
    return grouped[_CLASS_SENSITIVITY_COLUMNS]


def prepare_metric_tables(analysis: dict) -> dict[str, pd.DataFrame]:
    """Return the six released metric tables for one ``analyze_panel`` result."""

    metrics = analysis["metrics"]
    if set(metrics) != set(ESTIMANDS):
        raise ValueError("metric estimand scope changed")
    collected: dict[str, list[pd.DataFrame]] = {table: [] for table in _DIRECT_TABLES}
    collected["class_sensitivity"] = []

    for estimand in ESTIMANDS:
        if set(metrics[estimand]) != set(POLICIES):
            raise ValueError("metric policy scope changed")
        for policy_id in POLICIES:
            if set(metrics[estimand][policy_id]) != set(ENDPOINTS):
                raise ValueError("metric endpoint scope changed")
            for endpoint in ENDPOINTS:
                panel = metrics[estimand][policy_id][endpoint]
                for table in _DIRECT_TABLES:
                    base = panel[table]
                    expected = _EXPECTED_COLUMNS[table]
                    _require_columns(table, base, expected, _LABEL_COLUMNS[table])
                    labelled = base.copy()
                    labelled["estimand"] = estimand
                    labelled["policy_id"] = policy_id
                    labelled["endpoint"] = endpoint
                    collected[table].append(labelled[[*_LEADING_LABELS, *expected]])
                collected["class_sensitivity"].append(
                    _class_sensitivity(panel["class_recall"], estimand, policy_id, endpoint)
                )

    tables: dict[str, pd.DataFrame] = {}
    for table, frames in collected.items():
        combined = pd.concat(frames, ignore_index=True)
        tables[table] = combined.sort_values(list(_SORT_KEYS[table]), kind="stable").reset_index(
            drop=True
        )
    return tables
