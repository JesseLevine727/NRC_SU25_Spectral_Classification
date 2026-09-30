"""Semantic table for the P06 primary scatter (T038)."""

from __future__ import annotations

import numpy as np
import pandas as pd

FIGURE_ID = "F_P06_primary_scatter"
KIND = "scatter"
TITLE = "Selected CNN versus selected classical."
CAPTION = (
    "RQ-P01 (P/S). PP-U-MIN, 400\u20131800 cm^-1; source-only selection. "
    "69 physical masters, 557 held spectra, 13 station-instrument domains and ten instruments; "
    "252/260 common contexts. Each dot is a domain mean, not a sample. "
    "M01 uses individual spectra; M06 combines predictions, not raw spectra. "
    "The diagonal denotes equality. Repeated contexts are not independent."
)
_ENDPOINTS, _MODEL_ID, _REFERENCE_ID = ("M01", "M06"), "P05-SELECTED", "C-SELECTED"
_COLUMNS = (
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
)
_REQUIRED = (
    "model_id",
    "reference_model_id",
    "aggregation_id",
    "domain",
    "station",
    "held_instrument",
    "planned_contexts",
    "complete_contexts",
    "model_ba",
    "reference_ba",
    "delta",
)
_METADATA = _REQUIRED[:6]
_STYLE = {
    "cwa": ("#0072B2", "circle"),
    "pills": ("#D55E00", "square"),
    "surfaces": ("#009E73", "triangle-up"),
}


def _finite(frame, column):
    values = frame[column]
    if (
        pd.api.types.is_bool_dtype(values)
        or values.map(lambda value: isinstance(value, (bool, np.bool_))).any()
    ):
        raise ValueError(f"{column!r} must be numeric, not boolean")
    numeric = pd.to_numeric(values, errors="coerce")
    if not np.isfinite(numeric).all():
        raise ValueError(f"{column!r} must be finite")
    return numeric


def _integer(frame, column):
    numeric = _finite(frame, column)
    if (numeric < 1).any() or (numeric != np.floor(numeric)).any():
        raise ValueError(f"{column!r} must be a positive integer")
    return numeric.astype("int64")


def _text(frame, column):
    ok = frame[column].map(lambda value: isinstance(value, str) and value.strip() != "")
    if not ok.all():
        raise ValueError(f"{column!r} must be a nonempty string")


def build_semantics(tables):
    frame = tables["domain_metrics"]
    headers = list(frame.columns)
    if len(headers) != len(set(headers)):
        raise ValueError("duplicate column headers")
    missing = [column for column in _REQUIRED if column not in headers]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    for column in _METADATA:
        _text(frame, column)
    if not frame["station"].isin(_STYLE).all():
        raise ValueError("unknown station label")

    selected = frame[
        (frame["model_id"] == _MODEL_ID) & (frame["reference_model_id"] == _REFERENCE_ID)
    ].copy()
    if not set(_ENDPOINTS).issubset(set(selected["aggregation_id"])):
        raise ValueError("both endpoints must be present")
    if selected.duplicated(subset=["aggregation_id", "domain"]).any():
        raise ValueError("duplicate (endpoint, domain)")
    domain_sets = selected.groupby("aggregation_id")["domain"].agg(frozenset)
    if len(set(domain_sets)) != 1:
        raise ValueError("domain sets differ between endpoints")
    for _, group in selected.groupby("domain"):
        if group["station"].nunique() != 1 or group["held_instrument"].nunique() != 1:
            raise ValueError("domain station/instrument inconsistent")

    model_ba = _finite(selected, "model_ba")
    reference_ba = _finite(selected, "reference_ba")
    if (model_ba.clip(0, 1) != model_ba).any() or (reference_ba.clip(0, 1) != reference_ba).any():
        raise ValueError("BA outside [0, 1]")
    delta = _finite(selected, "delta")
    if not np.allclose(delta, model_ba - reference_ba, rtol=0.0, atol=1e-12):
        raise ValueError("delta does not equal model_ba - reference_ba")
    planned = _integer(selected, "planned_contexts")
    complete = _integer(selected, "complete_contexts")
    if (complete > planned).any():
        raise ValueError("complete_contexts exceeds planned_contexts")

    semantic = pd.DataFrame(
        {
            "figure_id": FIGURE_ID,
            "panel": selected["aggregation_id"].to_numpy(),
            "label": selected["domain"].to_numpy(),
            "series": selected["station"].to_numpy(),
            "x": 100.0 * reference_ba.to_numpy(),
            "y": 100.0 * model_ba.to_numpy(),
            "lower": np.nan,
            "upper": np.nan,
            "domain": selected["domain"].to_numpy(),
            "instrument": selected["held_instrument"].to_numpy(),
            "complete_contexts": complete.to_numpy(),
            "remaining_domains": np.nan,
            "color": selected["station"].map(lambda station: _STYLE[station][0]).to_numpy(),
            "marker": selected["station"].map(lambda station: _STYLE[station][1]).to_numpy(),
        }
    )
    semantic = (
        semantic[list(_COLUMNS)]
        .sort_values(["panel", "series", "label"], kind="mergesort")
        .reset_index(drop=True)
    )
    return {FIGURE_ID: {"semantic": semantic, "title": TITLE, "caption": CAPTION, "kind": KIND}}
