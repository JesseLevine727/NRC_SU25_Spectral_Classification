"""Deletion-stability semantic tables for the P06/P11 figures."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

_FIGURE_ID = "F_P06_deletion_stability"
_TITLE = "Does the mean difference survive removing one domain or instrument?"
_CAPTION = (
    "RQ-P01 (S), source-only selection; 69 physical masters and 557 held spectra. "
    "Primary selected CNN minus selected classical; PP-U-MIN, 400-1800 cm^-1. "
    "Each coloured point removes one of 13 station-instrument domains or one of ten "
    "instrument identities (including every domain using that instrument). The black "
    "cross retains all 252/260 paired contexts. Axes show M01 and M06 mean differences; "
    "M06 combines predictions, not spectra. No refitting. These are descriptive "
    "deletions, not independent experiments or confidence intervals."
)
_MODELS = ("M01", "M06")
_AGGREGATION = "P05-SELECTED"
_REFERENCE = "C-SELECTED"
_SERIES = {
    "domain": ("removed_domain", "#0072B2", "circle"),
    "instrument": ("removed_instrument", "#E69F00", "diamond"),
}
_COLUMNS = [
    "figure_id",
    "panel",
    "label",
    "series",
    "x",
    "y",
    "lower",
    "upper",
    "domain",
    "instrument",
    "complete_contexts",
    "remaining_domains",
    "color",
    "marker",
]
_LOO_COLUMNS = [
    "model_id",
    "reference_model_id",
    "aggregation_id",
    "exclusion_type",
    "exclusion_id",
    "removed_domains",
    "remaining_domains",
    "delta_after_exclusion",
    "model_mean_after_exclusion",
    "reference_mean_after_exclusion",
    "reason_code",
]
_SUMMARY_COLUMNS = [
    "model_id",
    "reference_model_id",
    "aggregation_id",
    "mean_delta",
    "supported_domains",
    "reason_code",
]


def _text(value, what):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{what} must not have leading or trailing whitespace")
    return value


def _reason(value):
    if _text(value, "reason_code") != "ok":
        raise ValueError("reason_code must be ok")


def _real(value, what):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{what} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be a finite real number")
    return number


def _positive_int(value, what):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{what} must be a positive integer")
    if isinstance(value, (int, np.integer)):
        number = int(value)
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        number = int(value)
    else:
        raise ValueError(f"{what} must be a positive integer")
    if number <= 0:
        raise ValueError(f"{what} must be a positive integer")
    return number


def _validate_table(frame, required, what):
    if frame.columns.duplicated().any():
        raise ValueError(f"{what} has duplicate columns")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{what} is missing required columns: {missing}")


def _is_primary(row):
    return (
        row.get("model_id") == _AGGREGATION
        and row.get("reference_model_id") == _REFERENCE
        and row.get("aggregation_id") in _MODELS
    )


def _read_summary(summary):
    found = {}
    for row in summary.to_dict("records"):
        if not _is_primary(row):
            continue
        endpoint = _text(row.get("aggregation_id"), "aggregation_id")
        if endpoint in found:
            raise ValueError("duplicate primary summary row")
        _reason(row.get("reason_code"))
        found[endpoint] = (
            _real(row.get("mean_delta"), "mean_delta"),
            _positive_int(row.get("supported_domains"), "supported_domains"),
        )
    for endpoint in _MODELS:
        if endpoint not in found:
            raise ValueError("missing primary summary row")
    if found["M01"][1] != found["M06"][1]:
        raise ValueError("supported_domains mismatch across endpoints")
    return found


def build_deletion_semantics(tables):
    """Build the deletion-stability semantic table from validated inputs."""
    summary = tables["summary"]
    leave_one_out = tables["leave_one_out"]
    _validate_table(summary, _SUMMARY_COLUMNS, "summary")
    _validate_table(leave_one_out, _LOO_COLUMNS, "leave_one_out")
    summaries = _read_summary(summary)
    supported = summaries["M01"][1]
    groups, seen = {}, set()
    for row in leave_one_out.to_dict("records"):
        if not _is_primary(row):
            continue
        endpoint = _text(row.get("aggregation_id"), "aggregation_id")
        exclusion_type = row.get("exclusion_type")
        if not isinstance(exclusion_type, str) or exclusion_type not in _SERIES:
            raise ValueError("unknown exclusion_type")
        series, color, marker = _SERIES[exclusion_type]
        exclusion_id = _text(row.get("exclusion_id"), "exclusion_id")
        _reason(row.get("reason_code"))
        identity = (endpoint, series, exclusion_id)
        if identity in seen:
            raise ValueError("duplicate deletion key")
        seen.add(identity)
        delta = _real(row.get("delta_after_exclusion"), "delta_after_exclusion")
        model_mean = _real(row.get("model_mean_after_exclusion"), "model_mean_after_exclusion")
        ref_mean = _real(
            row.get("reference_mean_after_exclusion"), "reference_mean_after_exclusion"
        )
        if abs(delta - (model_mean - ref_mean)) > 1e-12:
            raise ValueError("delta inconsistent with means")
        if not (0.0 <= model_mean <= 1.0 and 0.0 <= ref_mean <= 1.0):
            raise ValueError("means must lie in [0, 1]")
        groups.setdefault((series, exclusion_id), {})[endpoint] = (
            series,
            color,
            marker,
            delta,
            _positive_int(row.get("removed_domains"), "removed_domains"),
            _positive_int(row.get("remaining_domains"), "remaining_domains"),
        )

    if {key[0] for key in groups} != {"removed_domain", "removed_instrument"}:
        raise ValueError("deletions must include both domain and instrument types")

    records = []
    for series, exclusion_id in sorted(groups):
        endpoints = groups[(series, exclusion_id)]
        if any(endpoint not in endpoints for endpoint in _MODELS):
            raise ValueError("unpaired deletion key")
        first, second = endpoints["M01"], endpoints["M06"]
        if first[4:6] != second[4:6]:
            raise ValueError("deletion counts mismatch across endpoints")
        if first[4] + first[5] != supported:
            raise ValueError("removed + remaining must equal supported_domains")
        is_domain = series == "removed_domain"
        records.append(
            {
                "figure_id": _FIGURE_ID,
                "panel": "joint",
                "label": exclusion_id,
                "series": series,
                "x": 100.0 * first[3],
                "y": 100.0 * second[3],
                "lower": np.nan,
                "upper": np.nan,
                "domain": exclusion_id if is_domain else "",
                "instrument": "" if is_domain else exclusion_id,
                "complete_contexts": np.nan,
                "remaining_domains": first[5],
                "color": first[1],
                "marker": first[2],
            }
        )
    records.append(
        {
            "figure_id": _FIGURE_ID,
            "panel": "joint",
            "label": "Full paired support",
            "series": "full_reference",
            "x": 100.0 * summaries["M01"][0],
            "y": 100.0 * summaries["M06"][0],
            "lower": np.nan,
            "upper": np.nan,
            "domain": "",
            "instrument": "",
            "complete_contexts": np.nan,
            "remaining_domains": supported,
            "color": "#000000",
            "marker": "cross",
        }
    )
    semantic = pd.DataFrame(records, columns=_COLUMNS)
    return {
        _FIGURE_ID: {"semantic": semantic, "title": _TITLE, "caption": _CAPTION, "kind": "scatter"}
    }
