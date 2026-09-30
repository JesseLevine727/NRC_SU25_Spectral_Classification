"""Interval figure-table semantics for the P06 deliverable."""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd

__all__ = ["build_interval_semantics"]

_EFFECT_ID = "F_P06_effect_intervals"
_WEIGHT_ID = "F_P06_weight_sensitivity"

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

_ENDPOINTS = ("M01", "M06")

# (model_id, reference_model_id, label, series, color, marker)
_EFFECT_ROWS = (
    (
        "P05-SELECTED",
        "C-SELECTED",
        "Selected CNN - selected classical",
        "primary",
        "#0072B2",
        "circle",
    ),
    ("D3", "D0-M", "D3 - matched CNN", "secondary", "#777777", "diamond"),
    ("P05-SELECTED", "D0-M", "Selected CNN - matched CNN", "secondary", "#777777", "diamond"),
    (
        "P05-SELECTED",
        "C-RANDOM-FOREST",
        "Selected CNN - Random Forest",
        "secondary",
        "#777777",
        "diamond",
    ),
    (
        "P05-SELECTED",
        "C-EXTRA-TREES",
        "Selected CNN - Extra Trees",
        "secondary",
        "#777777",
        "diamond",
    ),
    ("D3", "C-RANDOM-FOREST", "D3 - Random Forest", "secondary", "#777777", "diamond"),
)

# (method, label, color, marker)
_WEIGHT_ROWS = (
    ("crossed_weight", "Shared sample + instrument", "#0072B2", "circle"),
    ("master_weight", "Shared sample only", "#009E73", "square"),
    ("instrument_weight", "Instrument only", "#D55E00", "diamond"),
)

_INPUT_COLUMNS = (
    "model_id",
    "reference_model_id",
    "aggregation_id",
    "method",
    "point_estimate",
    "lower",
    "upper",
    "reason_code",
    "planned_draws",
    "defined_draws",
    "undefined_draws",
)

_EFFECT_CAPTION = (
    "(P: M01 selected pair; S: other contrasts). Source-only selection; "
    "69 physical masters, 557 held spectra, 13 station-instrument domains, ten instruments. "
    "RQ-P01 primary/secondary contrasts; PP-U-MIN, 400-1800 cm^-1. Points are "
    "fixed balanced-accuracy differences; lines are 95% percentile intervals "
    "from 10,000 shared master/instrument weighted draws. Conditional on saved "
    "fits and observed support; secondary intervals are not multiplicity-adjusted. "
    "M01 uses individual spectra; M06 combines predictions, not spectra. "
    "Each contrast uses its own paired context support."
)
_WEIGHT_CAPTION = (
    "RQ-P01 (S); PP-U-MIN, 400-1800 cm^-1, source-only selection. "
    "69 physical masters, 557 held spectra, 13 domains and ten instruments. "
    "Same fixed primary difference and 252/260 common contexts. Lines are 95% "
    "percentile intervals from 10,000 positive-weight draws, conditional on saved "
    "fits and observed support. The original hierarchical interval is unavailable: "
    "all 10,000 draws had an originally represented empty chemical group. This "
    "sensitivity does not replace the original advancement criterion."
)


def _real(value, where):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{where} must be a real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{where} must be finite")
    return number


def _validate(row, key):
    where = "/".join(str(part) for part in key)
    if row["reason_code"] != "ok":
        raise ValueError(f"reason_code must be 'ok' for {where}")
    point = _real(row["point_estimate"], f"{where}.point_estimate")
    lower = _real(row["lower"], f"{where}.lower")
    upper = _real(row["upper"], f"{where}.upper")
    planned = _real(row["planned_draws"], f"{where}.planned_draws")
    defined = _real(row["defined_draws"], f"{where}.defined_draws")
    undefined = _real(row["undefined_draws"], f"{where}.undefined_draws")
    if lower > upper:
        raise ValueError(f"lower exceeds upper for {where}")
    if (planned, defined, undefined) != (10000.0, 10000.0, 0.0):
        raise ValueError(f"draw counts must be 10000/10000/0 for {where}")
    return point, lower, upper


def _selected(table, requested):
    if not isinstance(table, pd.DataFrame):
        raise TypeError("tables['intervals'] must be a pandas DataFrame")
    if table.columns.duplicated().any():
        raise ValueError("duplicate interval columns")
    absent = [column for column in _INPUT_COLUMNS if column not in table.columns]
    if absent:
        raise ValueError(f"intervals table missing columns: {absent}")
    selected = {}
    for row in table.to_dict("records"):
        key = (row["model_id"], row["reference_model_id"], row["aggregation_id"], row["method"])
        if key not in requested:
            continue
        if key in selected:
            raise ValueError(f"duplicate interval row for {key}")
        selected[key] = _validate(row, key)
    missing = requested - set(selected)
    if missing:
        raise ValueError(f"missing interval rows: {sorted(missing)}")
    return selected


def _semantic_row(figure_id, panel, label, series, point, lower, upper, y, color, marker):
    return {
        "figure_id": figure_id,
        "panel": panel,
        "label": label,
        "series": series,
        "x": 100.0 * point,
        "y": y,
        "lower": 100.0 * lower,
        "upper": 100.0 * upper,
        "domain": "",
        "instrument": "",
        "complete_contexts": np.nan,
        "remaining_domains": np.nan,
        "color": color,
        "marker": marker,
    }


def _requested():
    effect = {
        (model, reference, endpoint, "crossed_weight")
        for endpoint in _ENDPOINTS
        for model, reference, *_ in _EFFECT_ROWS
    }
    weight = {
        ("P05-SELECTED", "C-SELECTED", endpoint, method)
        for endpoint in _ENDPOINTS
        for method, *_ in _WEIGHT_ROWS
    }
    return effect, weight


def build_interval_semantics(tables):
    if not isinstance(tables, dict) or "intervals" not in tables:
        raise ValueError("tables must provide an 'intervals' frame")
    effect_requested, weight_requested = _requested()
    effect = _selected(tables["intervals"], effect_requested)
    weight = _selected(tables["intervals"], weight_requested)

    effect_rows = []
    for endpoint in _ENDPOINTS:
        for index, (model, reference, label, series, color, marker) in enumerate(_EFFECT_ROWS):
            point, lower, upper = effect[(model, reference, endpoint, "crossed_weight")]
            effect_rows.append(
                _semantic_row(
                    _EFFECT_ID,
                    endpoint,
                    label,
                    series,
                    point,
                    lower,
                    upper,
                    len(_EFFECT_ROWS) - index,
                    color,
                    marker,
                )
            )

    weight_rows = []
    for endpoint in _ENDPOINTS:
        for index, (method, label, color, marker) in enumerate(_WEIGHT_ROWS):
            point, lower, upper = weight[("P05-SELECTED", "C-SELECTED", endpoint, method)]
            weight_rows.append(
                _semantic_row(
                    _WEIGHT_ID,
                    endpoint,
                    label,
                    method,
                    point,
                    lower,
                    upper,
                    len(_WEIGHT_ROWS) - index,
                    color,
                    marker,
                )
            )

    return {
        _EFFECT_ID: {
            "semantic": pd.DataFrame(effect_rows, columns=_COLUMNS),
            "title": "Paired differences and conditional intervals",
            "caption": _EFFECT_CAPTION,
            "kind": "interval",
        },
        _WEIGHT_ID: {
            "semantic": pd.DataFrame(weight_rows, columns=_COLUMNS),
            "title": "Which observed dependencies widen the interval?",
            "caption": _WEIGHT_CAPTION,
            "kind": "interval",
        },
    }
