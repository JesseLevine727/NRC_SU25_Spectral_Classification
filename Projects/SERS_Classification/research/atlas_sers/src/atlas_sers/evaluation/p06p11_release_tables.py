"""Deterministic assembly of the P06/P11 release tables.

``prepare_tables`` consumes frozen, already-authenticated tables plus the
M01/M06 panel frames and produces a self-contained release bundle.  The
frozen tables are never mutated: every inferred table returned is a fresh
copy, with the sole exception of the three count fields in ``summary``,
which are re-derived from ``domain_metrics`` for secondary comparisons.
"""

from __future__ import annotations

import math
import numbers

import pandas as pd

from atlas_sers.evaluation.p06p11_metrics import build_metrics

__all__ = ["prepare_tables"]

TABLE_KEYS = (
    "domain_metrics",
    "summary",
    "intervals",
    "feasibility",
    "leave_one_out",
    "sign_flip",
    "g4_criteria",
)

SUMMARY_KEY = ("model_id", "reference_model_id", "aggregation_id")
COUNT_COLUMNS = ("positive_domains", "negative_domains", "tied_domains")

TIE_CORRECTION_COLUMNS = (
    "model_id",
    "reference_model_id",
    "aggregation_id",
    "before_positive",
    "before_negative",
    "before_tied",
    "after_positive",
    "after_negative",
    "after_tied",
)

PRIMARY_MODEL = "P05-SELECTED"
REFERENCE_MODEL = "C-SELECTED"
ENDPOINTS = ("M01", "M06")
PRIMARY_SCOPE = "primary_common"
SECONDARY_SCOPE = "full_support"

METRICS_KEY = ("scope", "aggregation_id", "model_id", "domain")

TIE_TOLERANCE = 1e-12
DELTA_ATOL = 1e-12
CROSSCHECK_ATOL = 1e-12
CROSSCHECK_RTOL = 0.0


def _require_columns(frame, columns, label):
    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"{label} has duplicate column headers: {duplicated}")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _scalar(frame, row, column):
    return frame.iat[row, frame.columns.get_loc(column)]


def _as_real(value, label):
    if isinstance(value, bool) or isinstance(value, str):
        raise ValueError(f"{label} must be a real number, got {value!r}")
    if not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _as_count(value, label):
    result = _as_real(value, label)
    if result != math.floor(result):
        raise ValueError(f"{label} must be a whole number, got {value!r}")
    return int(result)


def _is_primary(model_id, reference_id):
    return model_id == PRIMARY_MODEL and reference_id == REFERENCE_MODEL


def _recount(deltas):
    positive = negative = tied = 0
    for delta in deltas:
        if delta > TIE_TOLERANCE:
            positive += 1
        elif delta < -TIE_TOLERANCE:
            negative += 1
        else:
            tied += 1
    return positive, negative, tied


def _metrics_lookup(metrics):
    if "domain_metrics" not in metrics:
        raise ValueError("build_metrics result is missing 'domain_metrics'")
    metric_domain = metrics["domain_metrics"]
    _require_columns(
        metric_domain,
        METRICS_KEY + ("balanced_accuracy",),
        "metrics.domain_metrics",
    )
    lookup = {}
    for row in range(len(metric_domain)):
        key = tuple(_scalar(metric_domain, row, column) for column in METRICS_KEY)
        if key in lookup:
            raise ValueError(f"duplicate metrics lookup key: {key!r}")
        lookup[key] = _as_real(
            _scalar(metric_domain, row, "balanced_accuracy"),
            "metrics.balanced_accuracy",
        )
    return lookup


def _crosscheck(frozen_domain_metrics, metrics):
    lookup = _metrics_lookup(metrics)
    count = 0
    max_error = 0.0
    for row in range(len(frozen_domain_metrics)):
        model_id = _scalar(frozen_domain_metrics, row, "model_id")
        reference_id = _scalar(frozen_domain_metrics, row, "reference_model_id")
        # These partial-support contrasts have no corresponding descriptive scope.
        if reference_id == REFERENCE_MODEL and model_id != PRIMARY_MODEL:
            continue
        scope = PRIMARY_SCOPE if _is_primary(model_id, reference_id) else SECONDARY_SCOPE
        aggregation_id = _scalar(frozen_domain_metrics, row, "aggregation_id")
        domain = _scalar(frozen_domain_metrics, row, "domain")
        comparisons = (
            (model_id, _as_real(_scalar(frozen_domain_metrics, row, "model_ba"), "model_ba")),
            (
                reference_id,
                _as_real(_scalar(frozen_domain_metrics, row, "reference_ba"), "reference_ba"),
            ),
        )
        for lookup_model, observed in comparisons:
            key = (scope, aggregation_id, lookup_model, domain)
            if key not in lookup:
                raise ValueError(f"missing crosscheck match for {key!r}")
            expected = lookup[key]
            if not math.isclose(
                observed, expected, rel_tol=CROSSCHECK_RTOL, abs_tol=CROSSCHECK_ATOL
            ):
                raise ValueError(
                    f"crosscheck mismatch for {key!r}: observed={observed!r} expected={expected!r}"
                )
            count += 1
            max_error = max(max_error, abs(observed - expected))
    return count, max_error


def prepare_tables(tables, panels):
    missing_tables = [key for key in TABLE_KEYS if key not in tables]
    if missing_tables:
        raise KeyError(f"missing table(s): {missing_tables}")

    summary = tables["summary"]
    domain_metrics = tables["domain_metrics"]

    _require_columns(summary, SUMMARY_KEY + COUNT_COLUMNS + ("supported_domains",), "summary")
    _require_columns(
        domain_metrics,
        SUMMARY_KEY + ("domain", "complete_contexts", "model_ba", "reference_ba", "delta"),
        "domain_metrics",
    )

    summary_keys = []
    seen_summary = set()
    for row in range(len(summary)):
        key = tuple(_scalar(summary, row, column) for column in SUMMARY_KEY)
        if key[2] not in ENDPOINTS:
            raise ValueError(f"unsupported endpoint: {key[2]!r}")
        if key in seen_summary:
            raise ValueError(f"duplicate summary key: {key!r}")
        seen_summary.add(key)
        summary_keys.append(key)

    summary_key_set = set(summary_keys)
    for endpoint in ENDPOINTS:
        if (PRIMARY_MODEL, REFERENCE_MODEL, endpoint) not in summary_key_set:
            raise ValueError(
                f"missing primary comparison ({PRIMARY_MODEL!r}, "
                f"{REFERENCE_MODEL!r}) at endpoint {endpoint!r}"
            )

    domain_groups = {}
    seen_domain = set()
    for row in range(len(domain_metrics)):
        key = tuple(_scalar(domain_metrics, row, column) for column in SUMMARY_KEY)
        if key[2] not in ENDPOINTS:
            raise ValueError(f"unsupported endpoint: {key[2]!r}")
        domain = _scalar(domain_metrics, row, "domain")
        if (key, domain) in seen_domain:
            raise ValueError(f"duplicate domain_metrics key: {key!r}/{domain!r}")
        seen_domain.add((key, domain))
        model_ba = _as_real(_scalar(domain_metrics, row, "model_ba"), "model_ba")
        reference_ba = _as_real(_scalar(domain_metrics, row, "reference_ba"), "reference_ba")
        delta = _as_real(_scalar(domain_metrics, row, "delta"), "delta")
        if not math.isclose(delta, model_ba - reference_ba, rel_tol=0.0, abs_tol=DELTA_ATOL):
            raise ValueError(
                f"delta != model_ba - reference_ba for {key!r}/{domain!r}: "
                f"delta={delta!r} model_ba={model_ba!r} reference_ba={reference_ba!r}"
            )
        domain_groups.setdefault(key, []).append(delta)

    if summary_key_set != set(domain_groups):
        missing_keys = summary_key_set - set(domain_groups)
        extra_keys = set(domain_groups) - summary_key_set
        raise ValueError(
            "summary/domain_metrics key sets differ: "
            f"missing={sorted(repr(k) for k in missing_keys)} "
            f"extra={sorted(repr(k) for k in extra_keys)}"
        )

    inference_tables = {key: tables[key].copy(deep=True) for key in TABLE_KEYS}
    corrected_summary = inference_tables["summary"]
    count_positions = [corrected_summary.columns.get_loc(column) for column in COUNT_COLUMNS]

    tie_rows = []
    for index, key in enumerate(summary_keys):
        group = domain_groups[key]
        supported_domains = _as_count(
            _scalar(summary, index, "supported_domains"), "supported_domains"
        )
        if supported_domains != len(group):
            raise ValueError(
                f"supported_domains={supported_domains} but found "
                f"{len(group)} domain rows for {key!r}"
            )
        after = _recount(group)
        before = tuple(
            _as_count(_scalar(summary, index, column), column) for column in COUNT_COLUMNS
        )
        if _is_primary(key[0], key[1]):
            if before != after:
                raise ValueError(
                    f"primary endpoint counts changed for {key!r}: before={before} after={after}"
                )
            continue
        if before == after:
            continue
        for position, value in zip(count_positions, after, strict=True):
            corrected_summary.iat[index, position] = value
        tie_rows.append(
            {
                "model_id": key[0],
                "reference_model_id": key[1],
                "aggregation_id": key[2],
                "before_positive": before[0],
                "before_negative": before[1],
                "before_tied": before[2],
                "after_positive": after[0],
                "after_negative": after[1],
                "after_tied": after[2],
            }
        )

    tie_corrections = pd.DataFrame(tie_rows, columns=list(TIE_CORRECTION_COLUMNS))

    metrics = build_metrics(panels)
    crosscheck_count, crosscheck_max_error = _crosscheck(domain_metrics, metrics)

    return {
        "inference_tables": inference_tables,
        "metrics": metrics,
        "tie_corrections": tie_corrections,
        "crosscheck_count": crosscheck_count,
        "crosscheck_max_error": crosscheck_max_error,
    }
