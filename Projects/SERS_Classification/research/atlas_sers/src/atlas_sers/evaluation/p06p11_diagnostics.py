"""Descriptive P06/P11 domain-stability diagnostics and an honest G4 checklist.

This module is deliberately pure and in memory: it never reads a file, fits,
selects, imputes or samples anything. ``domain_diagnostics`` turns an
authenticated ``paired_metrics`` table (as produced by the frozen P05
comparison) into public aggregates that never expose context, master or
observation identifiers. ``g4_checklist`` maps a primary summary row plus a
standardised interval table onto six explicit gate criteria and a descriptive
decision, without hard coding study results.

Everything here is descriptive. Nothing in this module supports a
confirmatory claim, and the symmetry sign-flip table is explicitly labelled as
a sensitivity device that assumes shared masters are not independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas_sers.evaluation.p05_comparison import (
    AGGREGATIONS,
    CLASSICAL_MODELS,
    PAIRS,
)

__all__ = [
    "ALL_ENDPOINTS",
    "EXPLORATORY_ENDPOINTS",
    "P06P11ChecklistError",
    "P06P11DiagnosticsError",
    "PRIMARY_ENDPOINT",
    "domain_diagnostics",
    "g4_checklist",
]

PRIMARY_MODEL = "P05-SELECTED"
PRIMARY_REFERENCE = CLASSICAL_MODELS[0]
PRIMARY_AGGREGATION = "M01"
PRIMARY_ENDPOINT = (PRIMARY_MODEL, PRIMARY_REFERENCE, PRIMARY_AGGREGATION)
PRIMARY_DOMAIN_COUNT = 13

ALL_ENDPOINTS = tuple(
    (model, reference, aggregation)
    for model, reference in PAIRS
    for aggregation in AGGREGATIONS
)
EXPLORATORY_ENDPOINTS = tuple(
    endpoint for endpoint in ALL_ENDPOINTS if endpoint != PRIMARY_ENDPOINT
)
# The exploratory family is fixed by study design, never by which fixture rows
# happen to be present.
FAMILY_SIZE = len(EXPLORATORY_ENDPOINTS)
MAX_SIGN_GROUPS = 20
SYMMETRY_ASSUMPTION = "symmetry_sensitivity_shared_masters_not_independent"
INTERVAL_OK_REASONS = ("ok", "degenerate_distribution")

_KEY = ("model_id", "reference_model_id", "aggregation_id", "context_id")
_SUMMARY_IDENTITY = ("model_id", "reference_model_id", "aggregation_id")
_DOMAIN_IDENTITY = (
    "model_id",
    "reference_model_id",
    "aggregation_id",
    "domain",
    "station",
    "held_instrument",
)
_STRING_COLUMNS = (
    "model_id",
    "reference_model_id",
    "aggregation_id",
    "context_id",
    "domain",
    "station",
    "held_instrument",
)
_REQUIRED_PAIRED_COLUMNS = (
    *_KEY,
    "domain",
    "station",
    "held_instrument",
    "common_complete",
    "model_balanced_accuracy",
    "reference_balanced_accuracy",
    "delta_balanced_accuracy",
)

DOMAIN_COLUMNS = (
    *_DOMAIN_IDENTITY,
    "planned_contexts",
    "complete_contexts",
    "model_ba",
    "reference_ba",
    "delta",
    "reason_code",
)
SUMMARY_COLUMNS = (
    *_SUMMARY_IDENTITY,
    "supported_domains",
    "planned_domains",
    "complete_contexts",
    "planned_contexts",
    "model_mean_balanced_accuracy",
    "reference_mean_balanced_accuracy",
    "mean_delta",
    "median_delta",
    "q25_delta",
    "q75_delta",
    "min_delta",
    "max_delta",
    "positive_domains",
    "negative_domains",
    "tied_domains",
    "model_worst_domain_balanced_accuracy",
    "reference_worst_domain_balanced_accuracy",
    "worst_difference",
    "reason_code",
)
LEAVE_ONE_OUT_COLUMNS = (
    *_SUMMARY_IDENTITY,
    "exclusion_type",
    "exclusion_id",
    "removed_domains",
    "remaining_domains",
    "delta_after_exclusion",
    "model_mean_after_exclusion",
    "reference_mean_after_exclusion",
    "reason_code",
)
SIGN_FLIP_COLUMNS = (
    *_SUMMARY_IDENTITY,
    "kind",
    "number_groups",
    "assignments",
    "observed_delta",
    "p_descriptive",
    "p_holm",
    "family",
    "assumption_label",
    "reason_code",
)


class P06P11DiagnosticsError(ValueError):
    """Raised when the P06/P11 diagnostic input is missing or inconsistent."""


class P06P11ChecklistError(ValueError):
    """Raised when the G4 checklist inputs are missing or inconsistent."""


def _quartiles(values: np.ndarray) -> tuple[float, float]:
    q25, q75 = np.percentile(values, [25.0, 75.0], method="linear")
    return float(q25), float(q75)


def _require_columns(frame: pd.DataFrame, columns, code: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise P06P11DiagnosticsError(f"{code}:{','.join(missing)}")


def _validate_strings(frame: pd.DataFrame) -> None:
    for column in _STRING_COLUMNS:
        for value in frame[column]:
            if not isinstance(value, str) or not value or value != value.strip():
                raise P06P11DiagnosticsError(f"string_invalid:{column}")


def _validate_booleans(frame: pd.DataFrame, column: str) -> None:
    for value in frame[column]:
        if not isinstance(value, (bool, np.bool_)):
            raise P06P11DiagnosticsError(f"boolean_invalid:{column}")


def _is_real_number(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _unit_interval(frame: pd.DataFrame, column: str, complete) -> np.ndarray:
    selected = frame.loc[complete, column]
    for value in selected:
        if not _is_real_number(value):
            raise P06P11DiagnosticsError(f"{column}_invalid")
    values = selected.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise P06P11DiagnosticsError(f"{column}_invalid")
    return values


def _validate_paired_metrics(paired_metrics: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(paired_metrics, pd.DataFrame) or paired_metrics.empty:
        raise P06P11DiagnosticsError("paired_metrics_empty")
    if paired_metrics.columns.duplicated().any():
        raise P06P11DiagnosticsError("paired_metrics_duplicate_columns")
    frame = paired_metrics.copy()
    _require_columns(frame, _REQUIRED_PAIRED_COLUMNS, "paired_metrics_columns_missing")
    _validate_strings(frame)
    _validate_booleans(frame, "common_complete")
    frame["common_complete"] = frame["common_complete"].astype(bool)

    if frame.duplicated(list(_KEY)).any():
        raise P06P11DiagnosticsError("paired_metrics_duplicate_key")
    known_pairs = set(PAIRS)
    for model, reference in (
        frame[["model_id", "reference_model_id"]].drop_duplicates().itertuples(index=False)
    ):
        if (model, reference) not in known_pairs:
            raise P06P11DiagnosticsError("paired_metrics_unknown_pair")
    if not set(frame.aggregation_id.unique()).issubset(set(AGGREGATIONS)):
        raise P06P11DiagnosticsError("paired_metrics_unknown_aggregation")

    for column in ("domain", "station", "held_instrument"):
        if frame.groupby("context_id")[column].nunique().gt(1).any():
            raise P06P11DiagnosticsError("context_metadata_inconsistent")
    for column in ("station", "held_instrument"):
        if frame.groupby("domain")[column].nunique().gt(1).any():
            raise P06P11DiagnosticsError("domain_metadata_inconsistent")

    for _, cell in frame.groupby(["model_id", "reference_model_id"], sort=True):
        common_sets = {
            str(aggregation): frozenset(
                sub.loc[sub.common_complete, "context_id"].astype(str)
            )
            for aggregation, sub in cell.groupby("aggregation_id", sort=True)
        }
        if len(common_sets) == len(AGGREGATIONS) and len(set(common_sets.values())) > 1:
            raise P06P11DiagnosticsError("endpoint_common_context_mismatch")

    complete = frame.common_complete
    model_ba = _unit_interval(frame, "model_balanced_accuracy", complete)
    reference_ba = _unit_interval(frame, "reference_balanced_accuracy", complete)
    delta_values = frame.loc[complete, "delta_balanced_accuracy"]
    for value in delta_values:
        if not _is_real_number(value):
            raise P06P11DiagnosticsError("delta_balanced_accuracy_invalid")
    delta = delta_values.to_numpy(dtype=float)
    if not np.isfinite(delta).all():
        raise P06P11DiagnosticsError("delta_balanced_accuracy_invalid")
    if not np.allclose(delta, model_ba - reference_ba, atol=1e-12, rtol=0.0):
        raise P06P11DiagnosticsError("delta_balanced_accuracy_inconsistent")
    return frame


def _domain_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, cell in frame.groupby(list(_DOMAIN_IDENTITY), sort=True, dropna=False):
        record = dict(zip(_DOMAIN_IDENTITY, keys, strict=True))
        planned = int(len(cell))
        complete_cell = cell[cell.common_complete]
        complete = int(len(complete_cell))
        if complete:
            model_ba = float(complete_cell.model_balanced_accuracy.astype(float).mean())
            reference_ba = float(complete_cell.reference_balanced_accuracy.astype(float).mean())
            record.update(
                planned_contexts=planned,
                complete_contexts=complete,
                model_ba=model_ba,
                reference_ba=reference_ba,
                delta=model_ba - reference_ba,
                reason_code="ok",
            )
        else:
            record.update(
                planned_contexts=planned,
                complete_contexts=0,
                model_ba=np.nan,
                reference_ba=np.nan,
                delta=np.nan,
                reason_code="no_common_complete_contexts",
            )
        rows.append(record)
    return pd.DataFrame(rows, columns=list(DOMAIN_COLUMNS))


def _summary_table(domain: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, cell in domain.groupby(list(_SUMMARY_IDENTITY), sort=True, dropna=False):
        supported = cell[cell.complete_contexts > 0]
        record = dict(zip(_SUMMARY_IDENTITY, keys, strict=True))
        record.update(
            supported_domains=int(len(supported)),
            planned_domains=int(len(cell)),
            complete_contexts=int(cell.complete_contexts.sum()),
            planned_contexts=int(cell.planned_contexts.sum()),
        )
        if supported.empty:
            record.update(
                model_mean_balanced_accuracy=np.nan,
                reference_mean_balanced_accuracy=np.nan,
                mean_delta=np.nan,
                median_delta=np.nan,
                q25_delta=np.nan,
                q75_delta=np.nan,
                min_delta=np.nan,
                max_delta=np.nan,
                positive_domains=0,
                negative_domains=0,
                tied_domains=0,
                model_worst_domain_balanced_accuracy=np.nan,
                reference_worst_domain_balanced_accuracy=np.nan,
                worst_difference=np.nan,
                reason_code="no_supported_domains",
            )
        else:
            deltas = supported.delta.to_numpy(dtype=float)
            model_values = supported.model_ba.to_numpy(dtype=float)
            reference_values = supported.reference_ba.to_numpy(dtype=float)
            q25, q75 = _quartiles(deltas)
            model_worst = float(model_values.min())
            reference_worst = float(reference_values.min())
            record.update(
                model_mean_balanced_accuracy=float(model_values.mean()),
                reference_mean_balanced_accuracy=float(reference_values.mean()),
                mean_delta=float(deltas.mean()),
                median_delta=float(np.median(deltas)),
                q25_delta=q25,
                q75_delta=q75,
                min_delta=float(deltas.min()),
                max_delta=float(deltas.max()),
                # Numerical zero: raw deltas are floats, so exact ties can
                # surface as tiny residues (e.g. -2.22e-16). Use the same
                # absolute tolerance already used for score reproduction
                # (1e-12); raw deltas/point estimates/G4 margins and the
                # signflip/bootstrap arrays are left untouched.
                positive_domains=int((deltas > 1e-12).sum()),
                negative_domains=int((deltas < -1e-12).sum()),
                tied_domains=int((abs(deltas) <= 1e-12).sum()),
                model_worst_domain_balanced_accuracy=model_worst,
                reference_worst_domain_balanced_accuracy=reference_worst,
                worst_difference=model_worst - reference_worst,
                reason_code="ok",
            )
        rows.append(record)
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def _exclusion_record(keys, exclusion_type, exclusion_id, supported, remaining) -> dict:
    model, reference, aggregation = keys
    record = {
        "model_id": model,
        "reference_model_id": reference,
        "aggregation_id": aggregation,
        "exclusion_type": exclusion_type,
        "exclusion_id": exclusion_id,
        "removed_domains": int(len(supported) - len(remaining)),
        "remaining_domains": int(len(remaining)),
    }
    if remaining.empty:
        record.update(
            delta_after_exclusion=np.nan,
            model_mean_after_exclusion=np.nan,
            reference_mean_after_exclusion=np.nan,
            reason_code="no_residual_domains",
        )
    else:
        record.update(
            delta_after_exclusion=float(remaining.delta.astype(float).mean()),
            model_mean_after_exclusion=float(remaining.model_ba.astype(float).mean()),
            reference_mean_after_exclusion=float(remaining.reference_ba.astype(float).mean()),
            reason_code="ok",
        )
    return record


def _leave_one_out(domain: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, cell in domain.groupby(list(_SUMMARY_IDENTITY), sort=True, dropna=False):
        supported = cell[cell.complete_contexts > 0]
        if supported.empty:
            continue
        for domain_id in supported.domain:
            domain_id = str(domain_id)
            remaining = supported[supported.domain.astype(str) != domain_id]
            rows.append(
                _exclusion_record(keys, "domain", domain_id, supported, remaining)
            )
        for instrument in sorted(str(value) for value in supported.held_instrument.unique()):
            remaining = supported[supported.held_instrument.astype(str) != instrument]
            rows.append(
                _exclusion_record(keys, "instrument", instrument, supported, remaining)
            )
    return pd.DataFrame(rows, columns=list(LEAVE_ONE_OUT_COLUMNS))


def _sign_unavailable(model, reference, aggregation, kind, reason_code) -> dict:
    return {
        "model_id": model,
        "reference_model_id": reference,
        "aggregation_id": aggregation,
        "kind": kind,
        "number_groups": 0,
        "assignments": None,
        "observed_delta": None,
        "p_descriptive": None,
        "p_holm": None,
        "family": None,
        "assumption_label": SYMMETRY_ASSUMPTION,
        "reason_code": reason_code,
    }


def _sign_record(
    model, reference, aggregation, kind, groups, deltas, group_index, observed
) -> dict:
    record = {
        "model_id": model,
        "reference_model_id": reference,
        "aggregation_id": aggregation,
        "kind": kind,
        "number_groups": int(groups),
        "assignments": None,
        "observed_delta": observed,
        "p_descriptive": None,
        "p_holm": None,
        "family": None,
        "assumption_label": SYMMETRY_ASSUMPTION,
        "reason_code": "ok",
    }
    if groups == 0:
        record["reason_code"] = "no_supported_domains"
        return record
    if groups > MAX_SIGN_GROUPS:
        record["reason_code"] = "too_many_groups"
        return record
    count = 1 << groups
    index = np.arange(count, dtype=np.int64)[:, None]
    bits = np.bitwise_and(
        np.right_shift(index, np.arange(groups, dtype=np.int64)), 1
    ).astype(np.int8)
    signs = (1 - 2 * bits).astype(np.float64)
    domain_signs = signs[:, group_index]
    nulls = (domain_signs @ deltas) / len(deltas)
    hits = int(np.count_nonzero(np.abs(nulls) >= abs(observed) - 1e-12))
    record["assignments"] = int(count)
    record["p_descriptive"] = float(hits / count)
    return record


def _sign_flip(domain: pd.DataFrame) -> pd.DataFrame:
    supported_by_endpoint = {
        (str(keys[0]), str(keys[1]), str(keys[2])): cell[cell.complete_contexts > 0]
        for keys, cell in domain.groupby(list(_SUMMARY_IDENTITY), sort=True, dropna=False)
    }
    rows = []
    for model, reference, aggregation in ALL_ENDPOINTS:
        supported = supported_by_endpoint.get((model, reference, aggregation))
        if supported is None or supported.empty:
            rows.append(
                _sign_unavailable(
                    model, reference, aggregation, "domain", "no_supported_domains"
                )
            )
            rows.append(
                _sign_unavailable(
                    model, reference, aggregation, "instrument", "no_supported_domains"
                )
            )
            continue
        deltas = supported.delta.to_numpy(dtype=float)
        observed = float(deltas.mean())
        instruments = [str(value) for value in supported.held_instrument]
        lookup = {name: position for position, name in enumerate(sorted(set(instruments)))}
        instrument_index = np.array([lookup[name] for name in instruments], dtype=np.int64)
        rows.append(
            _sign_record(
                model,
                reference,
                aggregation,
                "domain",
                len(deltas),
                deltas,
                np.arange(len(deltas), dtype=np.int64),
                observed,
            )
        )
        rows.append(
            _sign_record(
                model,
                reference,
                aggregation,
                "instrument",
                len(lookup),
                deltas,
                instrument_index,
                observed,
            )
        )
    return pd.DataFrame(rows, columns=list(SIGN_FLIP_COLUMNS))


def _holm_adjusted(sign: pd.DataFrame) -> pd.DataFrame:
    frame = sign.copy()
    frame["p_holm"] = None
    frame["family"] = None
    for kind in ("domain", "instrument"):
        subset = frame[frame.kind.astype(str).eq(kind)]
        index_by_endpoint = {
            (str(row.model_id), str(row.reference_model_id), str(row.aggregation_id)): index
            for index, row in subset.iterrows()
        }
        primary_index = index_by_endpoint.get(PRIMARY_ENDPOINT)
        if primary_index is not None:
            frame.at[primary_index, "family"] = "primary_unadjusted_descriptive"

        entries = []
        for endpoint in EXPLORATORY_ENDPOINTS:
            index = index_by_endpoint.get(endpoint)
            if index is None:
                entries.append((None, None))
                continue
            value = frame.at[index, "p_descriptive"]
            if value is None or not np.isfinite(float(value)):
                entries.append((None, index))
            else:
                entries.append((float(value), index))

        order = sorted(
            range(len(entries)),
            key=lambda position: (
                entries[position][0] if entries[position][0] is not None else 1.0,
                position,
            ),
        )
        previous = 0.0
        for rank, position in enumerate(order):
            base = entries[position][0] if entries[position][0] is not None else 1.0
            previous = min(1.0, max(previous, (FAMILY_SIZE - rank) * base))
            index = entries[position][1]
            if index is None:
                continue
            frame.at[index, "family"] = "exploratory_holm"
            if entries[position][0] is not None:
                frame.at[index, "p_holm"] = previous
    return frame


def domain_diagnostics(paired_metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build descriptive domain-stability tables from frozen comparison metrics.

    The returned mapping holds ``domain_metrics`` (identity plus per-domain
    planned/complete context counts and equal-context means), ``summary``
    (per pair/endpoint equal-domain aggregates), ``leave_one_out`` (domain and
    instrument exclusions) and ``sign_flip`` (symmetry sensitivity statistics
    with Holm-adjusted exploratory p-values and one unadjusted primary row).
    """

    frame = _validate_paired_metrics(paired_metrics)
    domain = _domain_table(frame)
    summary = _summary_table(domain)
    leave_one_out = _leave_one_out(domain)
    sign_flip = _holm_adjusted(_sign_flip(domain))
    return {
        "domain_metrics": domain,
        "summary": summary,
        "leave_one_out": leave_one_out,
        "sign_flip": sign_flip,
    }


def _checklist_bool(value, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise P06P11ChecklistError(f"boolean_input_invalid:{name}")
    return bool(value)


def _checklist_t1(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise P06P11ChecklistError("t1_difference_invalid")
    number = float(value)
    if not np.isfinite(number):
        raise P06P11ChecklistError("t1_difference_invalid")
    return number


def _finite_number(value):
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _primary_summary_row(summary) -> pd.Series:
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        raise P06P11ChecklistError("summary_empty")
    if summary.columns.duplicated().any():
        raise P06P11ChecklistError("summary_duplicate_columns")
    required = list(_SUMMARY_IDENTITY)
    missing = [column for column in required if column not in summary.columns]
    if missing:
        raise P06P11ChecklistError(f"summary_columns_missing:{','.join(missing)}")
    mask = (
        summary.model_id.astype(str).eq(PRIMARY_MODEL)
        & summary.reference_model_id.astype(str).eq(PRIMARY_REFERENCE)
        & summary.aggregation_id.astype(str).eq(PRIMARY_AGGREGATION)
    )
    selected = summary[mask]
    if len(selected) != 1:
        raise P06P11ChecklistError("primary_summary_row_not_unique")
    return selected.iloc[0]


def _criterion(name, observed, threshold, operator, status, reason_code, source_scope) -> dict:
    return {
        "criterion": name,
        "observed": observed,
        "threshold": threshold,
        "operator": operator,
        "status": status,
        "reason_code": reason_code,
        "source_scope": source_scope,
    }


def _domain_support_status(row):
    supported = _finite_number(row.get("supported_domains"))
    planned = _finite_number(row.get("planned_domains"))
    if supported is None or planned is None:
        return "unassessable", "domain_support_missing"
    if not float(supported).is_integer() or not float(planned).is_integer():
        return "unassessable", "domain_support_not_integral"
    if supported != planned:
        return "unassessable", "incomplete_domain_support"
    if supported != PRIMARY_DOMAIN_COUNT:
        return "unassessable", "domain_support_not_primary_count"
    return "assessable", "ok"


def _support_criterion(name, row, column, threshold, operator, source_scope) -> dict:
    support, support_reason = _domain_support_status(row)
    if support != "assessable":
        return _criterion(
            name, None, threshold, operator, "unassessable", support_reason, source_scope
        )
    value = _finite_number(row.get(column))
    if value is None:
        return _criterion(
            name, None, threshold, operator, "unassessable", "metric_missing", source_scope
        )
    if column == "positive_domains":
        supported = _finite_number(row.get("supported_domains"))
        invalid_count = (
            supported is None
            or not float(value).is_integer()
            or value < 0.0
            or value > supported
        )
        if invalid_count:
            return _criterion(
                name,
                None,
                threshold,
                operator,
                "unassessable",
                "positive_domains_invalid",
                source_scope,
            )
    passes = value >= threshold if operator == ">=" else value > threshold
    status = "supported" if passes else "failed"
    reason = "ok" if passes else "below_threshold"
    return _criterion(name, value, threshold, operator, status, reason, source_scope)


def _interval_state(intervals):
    if not isinstance(intervals, pd.DataFrame) or intervals.empty:
        return "unassessable", None, "hierarchical_interval_missing"
    if intervals.columns.duplicated().any():
        raise P06P11ChecklistError("interval_duplicate_columns")
    key_columns = ("model_id", "reference_model_id", "aggregation_id", "method")
    missing = [column for column in key_columns if column not in intervals.columns]
    if missing:
        return "unassessable", None, "interval_columns_missing"
    if intervals.duplicated(list(key_columns)).any():
        raise P06P11ChecklistError("interval_duplicate_key")
    if not all(
        column in intervals.columns for column in ("lower", "upper", "reason_code")
    ):
        return "unassessable", None, "interval_columns_missing"
    mask = (
        intervals.model_id.astype(str).eq(PRIMARY_MODEL)
        & intervals.reference_model_id.astype(str).eq(PRIMARY_REFERENCE)
        & intervals.aggregation_id.astype(str).eq(PRIMARY_AGGREGATION)
        & intervals.method.astype(str).eq("hierarchical")
    )
    selected = intervals[mask]
    if selected.empty:
        return "unassessable", None, "hierarchical_interval_missing"
    if len(selected) > 1:
        raise P06P11ChecklistError("hierarchical_interval_not_unique")
    record = selected.iloc[0]
    reason = record.get("reason_code")
    if not isinstance(reason, str) or reason not in INTERVAL_OK_REASONS:
        return "unassessable", None, "hierarchical_interval_unavailable"
    lower = _finite_number(record.get("lower"))
    upper = _finite_number(record.get("upper"))
    if lower is None or upper is None:
        return "unassessable", None, "hierarchical_interval_bounds_invalid"
    if lower < -1.0 or lower > 1.0 or upper < -1.0 or upper > 1.0:
        return "unassessable", None, "hierarchical_interval_bounds_out_of_range"
    if lower > upper:
        return "unassessable", None, "hierarchical_interval_bounds_inconsistent"
    return "assessable", lower, "ok"


def _interval_criterion(intervals) -> dict:
    state, lower, reason = _interval_state(intervals)
    name = "hierarchical_interval_lower_gt_0"
    scope = "hierarchical_interval"
    if state != "assessable":
        return _criterion(name, None, 0.0, ">", "unassessable", reason, scope)
    passes = lower > 0.0
    return _criterion(
        name,
        lower,
        0.0,
        ">",
        "supported" if passes else "failed",
        "ok" if passes else "not_above_zero",
        scope,
    )


def _t1_criterion(t1_value, provenance_verified: bool) -> dict:
    name = "t1_difference_at_least_neg_0.02"
    scope = "t1_provenance"
    if not provenance_verified:
        return _criterion(
            name, None, -0.02, ">=", "unassessable", "t1_provenance_unverified", scope
        )
    if t1_value is None:
        return _criterion(
            name, None, -0.02, ">=", "unassessable", "t1_difference_missing", scope
        )
    passes = t1_value >= -0.02
    return _criterion(
        name,
        t1_value,
        -0.02,
        ">=",
        "supported" if passes else "failed",
        "ok" if passes else "below_threshold",
        scope,
    )


def _preservation_criterion(preserved: bool) -> dict:
    return _criterion(
        "input_preservation_verified",
        preserved,
        True,
        "is",
        "supported" if preserved else "failed",
        "ok" if preserved else "input_preservation_failed",
        "input_preservation",
    )


def g4_checklist(
    summary: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    input_preservation_verified: bool,
    t1_difference: float | None = None,
    t1_provenance_verified: bool = False,
) -> dict:
    """Evaluate six descriptive G4 gate criteria for the primary contrast.

    Only the unique ``P05-SELECTED`` vs the selected classical model ``M01``
    summary row is used, together with the original ``hierarchical`` interval;
    a ``crossed_weight`` interval can never substitute for it. Returns a mapping
    with a ``criteria`` frame and a ``decision`` dictionary. ``status`` is
    ``supported`` only when every criterion is supported, ``failed`` when any
    criterion fails, otherwise ``unassessable``.
    """

    preserved = _checklist_bool(input_preservation_verified, "input_preservation_verified")
    provenance = _checklist_bool(t1_provenance_verified, "t1_provenance_verified")
    t1_value = _checklist_t1(t1_difference)

    row = _primary_summary_row(summary)
    rows = [
        _support_criterion(
            "mean_delta_at_least_0.03", row, "mean_delta", 0.03, ">=", "primary_m01_summary"
        ),
        _interval_criterion(intervals),
        _support_criterion(
            "positive_domains_at_least_8",
            row,
            "positive_domains",
            8,
            ">=",
            "primary_m01_summary",
        ),
        _support_criterion(
            "worst_domain_difference_at_least_neg_0.03",
            row,
            "worst_difference",
            -0.03,
            ">=",
            "primary_m01_summary",
        ),
        _t1_criterion(t1_value, provenance),
        _preservation_criterion(preserved),
    ]
    criteria = pd.DataFrame(
        rows,
        columns=[
            "criterion",
            "observed",
            "threshold",
            "operator",
            "status",
            "reason_code",
            "source_scope",
        ],
    )
    supported = int(criteria.status.eq("supported").sum())
    failed = int(criteria.status.eq("failed").sum())
    unassessable = int(criteria.status.eq("unassessable").sum())
    if supported == len(rows):
        status = "supported"
    elif failed > 0:
        status = "failed"
    else:
        status = "unassessable"
    decision = {
        "promote": supported == len(rows),
        "status": status,
        "supported_criteria": supported,
        "failed_criteria": failed,
        "unassessable_criteria": unassessable,
    }
    return {"criteria": criteria, "decision": decision}
