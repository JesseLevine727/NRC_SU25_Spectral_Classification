# ruff: noqa: E501
"""P05 diagnostic figures from authenticated anonymous aggregate tables.

This module is pure rendering. It never reads scientific inputs, trains,
predicts, selects, calibrates, bootstraps or inspects private reference frames.
Callers authenticate the anonymous public metric, source-diagnostic and
reliability tables produced by the corresponding P05 modules and hand them in
unchanged.

Every plotted point is first projected into one explicit allowlisted canonical
semantic table that contains only generic figure coordinates and public
metadata (figure, station, panel, row, column, series, phase, aggregation,
recipe, selection mode, domain, held instrument, anonymous point index, x, y,
count and denominator). Private identifiers, source paths, labels, predictions
and arbitrary blocking reasons are never copied. All native pgfplots/TikZ,
offline Plotly HTML, vector PDF and 300 dpi PNG previews are rendered from that
single projected table and share its SHA-256 digest.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from atlas_sers.evaluation.p05_comparison import PAIRS
from atlas_sers.evaluation.p05_public_metrics import (
    P05_PUBLIC_AGGREGATIONS,
    P05_PUBLIC_MODELS,
    P05_PUBLIC_PHASES,
    P05_PUBLIC_STATIONS,
)
from atlas_sers.evaluation.p05_selection import (
    MASTER_ONLY_MODES,
    MAXIMUM_EPOCH,
    MINIMUM_EPOCH,
    PSEUDO_DOMAIN_MODE,
    RECIPE_IDS,
    SELECTION_MODES,
    SLOT_KINDS,
)
from atlas_sers.governance.canonical import sha256_file
from atlas_sers.visualization.p04_figures import _write_html
from atlas_sers.visualization.p05_benchmark_figures import (
    BenchmarkFigureError,
    _as_float,
    _as_int,
    _html_literal,
    _require_text,
    _require_unit,
    _tex_literal,
)
from atlas_sers.visualization.p05_figure_runtime import _compile
from atlas_sers.visualization.p05_smoke_figures import (
    _configure_deterministic_pdf,
    _guard_output_root,
)

__all__ = ["DiagnosticFigureError", "generate_diagnostic_figures"]

FAMILY_PAIRED = "paired_domains"
FAMILY_SOURCE = "source_vs_held"
FAMILY_EPOCH = "best_epoch_distribution"
FAMILY_RELIABILITY = "reliability_bins"

STATION_ORDER = tuple(P05_PUBLIC_STATIONS)
STATION_LABELS = {"cwa": "CWA", "pills": "Pills", "surfaces": "Surfaces"}
MODEL_ORDER = tuple(P05_PUBLIC_MODELS)
AGGREGATION_ORDER = tuple(P05_PUBLIC_AGGREGATIONS)
PHASE_ORDER = tuple(P05_PUBLIC_PHASES)
RECIPE_ORDER = tuple(sorted(RECIPE_IDS))
SLOT_KIND_ORDER = tuple(sorted(SLOT_KINDS))
REGISTERED_REFERENCES = tuple(sorted({str(reference) for _, reference in PAIRS}))
PAIR_SET = frozenset((str(model), str(reference)) for model, reference in PAIRS)

SOURCE_ROW_LABELS = {0: "pseudo-domain", 1: "master-CV"}

MODEL_LABELS = {
    "D0-M": "Matched CNN (D0-M)",
    "P05-SELECTED": "Source-selected CNN",
    "D3": "Combined-loss CNN (D3)",
}
MODEL_TEX_COLOR = {"D0-M": "p05Blue", "P05-SELECTED": "p05Orange", "D3": "p05Purple"}
MODEL_TEX_MARK = {"D0-M": "*", "P05-SELECTED": "square", "D3": "triangle"}
MODEL_TEX_OPEN = {"D0-M": False, "P05-SELECTED": True, "D3": True}
MODEL_HTML_COLOR = {"D0-M": "#0072B2", "P05-SELECTED": "#E69F00", "D3": "#CC79A7"}
MODEL_HTML_SYMBOL = {"D0-M": "circle", "P05-SELECTED": "square-open", "D3": "triangle-up-open"}

RECIPE_TEX_COLORS = ("p05Blue", "p05Orange", "p05Green", "p05Purple", "p05Red", "p05Sky")
RECIPE_TEX_MARKS = ("*", "square", "triangle", "diamond", "*", "square")
RECIPE_TEX_OPEN = (False, True, True, True, True, True)
RECIPE_HTML_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")
RECIPE_HTML_SYMBOLS = (
    "circle",
    "square-open",
    "triangle-up-open",
    "diamond-open",
    "circle-open",
    "square",
)

SEMANTIC_COLUMNS = (
    "figure",
    "station",
    "panel",
    "panel_row",
    "panel_col",
    "series",
    "phase",
    "aggregation_id",
    "recipe",
    "selection_mode",
    "held_instrument",
    "domain",
    "x",
    "y",
    "count",
    "denominator",
    "point_index",
    "bin_index",
)

PAIRED_DOMAIN_COLUMNS = (
    "station",
    "domain",
    "held_instrument",
    "model_id",
    "reference_model_id",
    "aggregation_id",
    "planned_contexts",
    "common_contexts",
    "common_coverage",
    "mean_model_balanced_accuracy",
    "mean_reference_balanced_accuracy",
    "mean_delta_balanced_accuracy",
)
SOURCE_VS_HELD_COLUMNS = (
    "point_index",
    "station",
    "phase",
    "domain",
    "held_instrument",
    "model_id",
    "aggregation_id",
    "selection_mode",
    "recipe",
    "source_mean_seed_unit_balanced_accuracy",
    "source_mean_seed_unit_negative_log_likelihood",
    "source_unit_count",
    "guard_unit_count",
    "held_balanced_accuracy",
    "source_evidence_policy",
    "source_transfer_validation_available",
    "source_vs_held_policy",
)
BEST_EPOCH_COLUMNS = (
    "station",
    "phase",
    "selection_mode",
    "recipe",
    "slot_kind",
    "best_epoch",
    "fit_count",
)
RELIABILITY_COLUMNS = (
    "station",
    "phase",
    "model_id",
    "aggregation_id",
    "bin_index",
    "count",
    "mean_confidence",
    "observed_accuracy",
    "signed_gap",
    "bin_weight",
)

MANIFEST_NAME = "P05_diagnostic_manifest.json"
SEMANTIC_NAME = "semantic_data.csv"
ALLOWED_SUFFIXES = {".csv", ".tex", ".pdf", ".png", ".html"}
COVERAGE_TOLERANCE = 1e-12
DELTA_TOLERANCE = 1e-12
RELIABILITY_TOLERANCE = 1e-12

PAIRED_CAPTION = (
    "Each point is a station-instrument domain mean balanced accuracy over common complete outer contexts.",
    "Domains without a common complete outer context contribute no plotted point; missing values are not zero and are never imputed.",
    "The vertical dotted line marks zero change; repeated outer splits are not independent chemical samples.",
    "M01 scores individual spectra after three-seed probability averaging.",
    "M06 averages probabilities within each instrument, then equally across instruments for a sample; raw spectra are not averaged.",
)
SOURCE_CAPTION = (
    "The source coordinate averages per-spectrum best-validation balanced accuracy across seeds within each inherited selection unit, then equally across units; guards are excluded.",
    "The outer coordinate is the registered three-seed probability-averaged ensemble metric.",
    "These are different estimators: the source score was used for selection and is optimistic; their difference is descriptive, not an unbiased transfer gap.",
    "Pseudo-domain validation and master-only validation are shown separately; availability does not establish successful generalization.",
    "M01 scores individual spectra after three-seed probability averaging.",
    "M06 averages probabilities within each instrument, then equally across instruments for a sample; raw spectra are not averaged.",
)
EPOCH_CAPTION = (
    "Each point is the fraction of source fits selecting that best epoch for a station, phase, slot kind and recipe, aggregated across selection modes.",
    "A best epoch can be as early as epoch 1 although the run must train for at least 30 epochs; this does not authorize a new schedule and does not imply a terminal epoch.",
    "No interpolation, smoothing or confidence interval is shown.",
)
RELIABILITY_CAPTION = (
    "Each panel pools context appearances (M01 spectra or M06 instrument-balanced masters) into up to ten equal-mass confidence bins.",
    "Repeated appearances are not independent samples; no confidence interval or significance test is shown.",
    "The pooled reliability curve is descriptive and differs from the registered mean per-context ECE; no extra calibration or temperature is fitted.",
    "M01 scores individual spectra after three-seed probability averaging.",
    "M06 averages probabilities within each instrument, then equally across instruments for a sample; raw spectra are not averaged.",
)


class DiagnosticFigureError(ValueError):
    """Raised when P05 diagnostic aggregate inputs are missing or inconsistent."""


# --------------------------------------------------------------------------- #
# Validation helpers (reuse the benchmark/public modules where appropriate)
# --------------------------------------------------------------------------- #


def _text(value: object, code: str) -> str:
    try:
        return _require_text(value, code)
    except BenchmarkFigureError:
        raise DiagnosticFigureError(code) from None


def _optional_text(value: object, code: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise DiagnosticFigureError(code)
    return value


def _integer(value: object, code: str) -> int:
    try:
        return _as_int(value, code)
    except BenchmarkFigureError:
        raise DiagnosticFigureError(code) from None


def _number(value: object, code: str) -> float:
    try:
        return _as_float(value, code)
    except BenchmarkFigureError:
        raise DiagnosticFigureError(code) from None


def _unit(value: object, code: str) -> float:
    try:
        return _require_unit(value, code)
    except BenchmarkFigureError:
        raise DiagnosticFigureError(code) from None


def _require_missing(value: object, code: str) -> None:
    if not pd.isna(value):
        raise DiagnosticFigureError(code)


def _enum(value: object, allowed: object, code: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise DiagnosticFigureError(code)
    return value


def _require_mapping(value: object, code: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise DiagnosticFigureError(code)
    return value


def _require_frame(tables: Mapping, key: str, code: str) -> pd.DataFrame:
    if key not in tables:
        raise DiagnosticFigureError(code)
    frame = tables[key]
    if not isinstance(frame, pd.DataFrame):
        raise DiagnosticFigureError(code)
    return frame


def _require_columns(frame: pd.DataFrame, columns: object, code: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DiagnosticFigureError(f"{code}:{','.join(missing)}")


# --------------------------------------------------------------------------- #
# Canonical semantic projection
# --------------------------------------------------------------------------- #


def _figure_id(family: str, *parts: object) -> str:
    return "|".join([family, *(str(part) for part in parts)])


def _make_row(
    *,
    figure: str,
    station: str,
    panel: str,
    panel_row: int,
    panel_col: int,
    series: str,
    x: float,
    y: float,
    count: int,
    denominator: int,
    phase: str = "",
    aggregation_id: str = "",
    recipe: str = "",
    selection_mode: str = "",
    held_instrument: str = "",
    domain: str = "",
    point_index: object = None,
    bin_index: object = None,
) -> dict:
    return {
        "figure": figure,
        "station": station,
        "panel": panel,
        "panel_row": int(panel_row),
        "panel_col": int(panel_col),
        "series": series,
        "phase": phase,
        "aggregation_id": aggregation_id,
        "recipe": recipe,
        "selection_mode": selection_mode,
        "held_instrument": held_instrument,
        "domain": domain,
        "x": float(x),
        "y": float(y),
        "count": int(count),
        "denominator": int(denominator),
        "point_index": point_index if point_index is not None else pd.NA,
        "bin_index": bin_index if bin_index is not None else pd.NA,
    }


def _project_paired_domains(table: pd.DataFrame) -> list[dict]:
    _require_columns(table, PAIRED_DOMAIN_COLUMNS, "paired_domains_columns_missing")
    if table.empty:
        return []
    validated = []
    seen: set = set()
    domain_map: dict = {}
    for record in table[list(PAIRED_DOMAIN_COLUMNS)].to_dict("records"):
        station = _enum(record["station"], STATION_ORDER, "paired_domains_station_unknown")
        domain = _text(record["domain"], "paired_domains_domain_invalid")
        instrument = _optional_text(record["held_instrument"], "paired_domains_instrument_invalid")
        model = _enum(record["model_id"], MODEL_ORDER, "paired_domains_model_unknown")
        reference = _enum(
            record["reference_model_id"],
            REGISTERED_REFERENCES,
            "paired_domains_reference_unknown",
        )
        if (model, reference) not in PAIR_SET:
            raise DiagnosticFigureError("paired_domains_pair_unregistered")
        aggregation = _enum(
            record["aggregation_id"], AGGREGATION_ORDER, "paired_domains_aggregation_unknown"
        )
        planned = _integer(record["planned_contexts"], "paired_domains_planned_invalid")
        if planned <= 0:
            raise DiagnosticFigureError("paired_domains_planned_invalid")
        common = _integer(record["common_contexts"], "paired_domains_common_invalid")
        if common < 0 or common > planned:
            raise DiagnosticFigureError("paired_domains_common_invalid")
        coverage = _unit(record["common_coverage"], "paired_domains_coverage_invalid")
        if abs(coverage - common / planned) > COVERAGE_TOLERANCE:
            raise DiagnosticFigureError("paired_domains_coverage_inconsistent")
        if common > 0:
            model_ba = _unit(
                record["mean_model_balanced_accuracy"], "paired_domains_model_ba_invalid"
            )
            reference_ba = _unit(
                record["mean_reference_balanced_accuracy"],
                "paired_domains_reference_ba_invalid",
            )
            delta = _number(record["mean_delta_balanced_accuracy"], "paired_domains_delta_invalid")
            if abs(delta - (model_ba - reference_ba)) > DELTA_TOLERANCE:
                raise DiagnosticFigureError("paired_domains_delta_inconsistent")
            if not -1.0 <= delta <= 1.0:
                raise DiagnosticFigureError("paired_domains_delta_out_of_range")
        else:
            _require_missing(
                record["mean_model_balanced_accuracy"],
                "paired_domains_model_ba_expected_missing",
            )
            _require_missing(
                record["mean_reference_balanced_accuracy"],
                "paired_domains_reference_ba_expected_missing",
            )
            _require_missing(
                record["mean_delta_balanced_accuracy"],
                "paired_domains_delta_expected_missing",
            )
            delta = reference_ba = np.nan
        key = (domain, model, reference, aggregation)
        if key in seen:
            raise DiagnosticFigureError("paired_domains_duplicate_key")
        seen.add(key)
        observed = (station, instrument)
        if domain in domain_map and domain_map[domain] != observed:
            raise DiagnosticFigureError("paired_domains_domain_conflict")
        domain_map[domain] = observed
        validated.append(
            (
                station,
                domain,
                instrument,
                model,
                reference,
                aggregation,
                planned,
                common,
                delta,
                reference_ba,
            )
        )
    rows = []
    for (
        station,
        domain,
        instrument,
        model,
        reference,
        aggregation,
        planned,
        common,
        delta,
        reference_ba,
    ) in validated:
        rows.append(
            _make_row(
                figure=_figure_id(FAMILY_PAIRED, reference, aggregation),
                station=station,
                panel=STATION_LABELS[station],
                panel_row=0,
                panel_col=STATION_ORDER.index(station),
                series=model,
                aggregation_id=aggregation,
                held_instrument=instrument,
                domain=domain,
                x=delta,
                y=reference_ba,
                count=common,
                denominator=planned,
            )
        )
    return rows


def _project_source_vs_held(table: pd.DataFrame) -> list[dict]:
    _require_columns(table, SOURCE_VS_HELD_COLUMNS, "source_vs_held_columns_missing")
    if table.empty:
        return []
    rows = []
    entries: list = []
    seen: set = set()
    for record in table[list(SOURCE_VS_HELD_COLUMNS)].to_dict("records"):
        point_index = _integer(record["point_index"], "source_vs_held_point_index_invalid")
        if point_index <= 0:
            raise DiagnosticFigureError("source_vs_held_point_index_invalid")
        station = _enum(record["station"], STATION_ORDER, "source_vs_held_station_unknown")
        phase = _enum(record["phase"], PHASE_ORDER, "source_vs_held_phase_unknown")
        domain = _text(record["domain"], "source_vs_held_domain_invalid")
        instrument = _optional_text(record["held_instrument"], "source_vs_held_instrument_invalid")
        model = _enum(record["model_id"], MODEL_ORDER, "source_vs_held_model_unknown")
        aggregation = _enum(
            record["aggregation_id"], AGGREGATION_ORDER, "source_vs_held_aggregation_unknown"
        )
        mode = _enum(record["selection_mode"], SELECTION_MODES, "source_vs_held_mode_unknown")
        recipe = _enum(record["recipe"], RECIPE_ORDER, "source_vs_held_recipe_unknown")
        if model in ("D0-M", "D3") and recipe != model:
            raise DiagnosticFigureError("source_vs_held_model_recipe_mismatch")
        if mode in MASTER_ONLY_MODES and model == "P05-SELECTED" and recipe != "D0-M":
            raise DiagnosticFigureError("source_vs_held_master_recipe_mismatch")
        source_ba = _unit(
            record["source_mean_seed_unit_balanced_accuracy"],
            "source_vs_held_source_ba_invalid",
        )
        held_ba = _unit(record["held_balanced_accuracy"], "source_vs_held_held_ba_invalid")
        unit_count = _integer(record["source_unit_count"], "source_vs_held_unit_count_invalid")
        nll = _number(
            record["source_mean_seed_unit_negative_log_likelihood"],
            "source_vs_held_nll_invalid",
        )
        if not np.isfinite(nll) or nll < 0.0:
            raise DiagnosticFigureError("source_vs_held_nll_invalid")
        guard_count = _integer(record["guard_unit_count"], "source_vs_held_guard_count_invalid")
        if unit_count <= 0:
            raise DiagnosticFigureError("source_vs_held_unit_count_invalid")
        if guard_count < 0:
            raise DiagnosticFigureError("source_vs_held_guard_count_invalid")
        available = record["source_transfer_validation_available"]
        if isinstance(available, (bool, np.bool_)):
            available_flag = bool(available)
        else:
            raise DiagnosticFigureError("source_vs_held_policy_flag_invalid")
        if mode == PSEUDO_DOMAIN_MODE:
            row_index = 0
            if guard_count != 3:
                raise DiagnosticFigureError("source_vs_held_pseudo_guard_invalid")
        elif mode in MASTER_ONLY_MODES:
            row_index = 1
            if guard_count != 0:
                raise DiagnosticFigureError("source_vs_held_master_guard_invalid")
        else:
            raise DiagnosticFigureError("source_vs_held_mode_unmapped")
        if available_flag != (mode == PSEUDO_DOMAIN_MODE):
            raise DiagnosticFigureError("source_vs_held_policy_flag_inconsistent")
        key = (point_index, model, aggregation)
        if key in seen:
            raise DiagnosticFigureError("source_vs_held_duplicate_key")
        seen.add(key)
        rows.append(
            _make_row(
                figure=_figure_id(FAMILY_SOURCE, phase, aggregation),
                station=station,
                panel=f"{STATION_LABELS[station]}:{SOURCE_ROW_LABELS[row_index]}",
                panel_row=row_index,
                panel_col=STATION_ORDER.index(station),
                series=model,
                phase=phase,
                aggregation_id=aggregation,
                recipe=recipe,
                selection_mode=mode,
                held_instrument=instrument,
                domain=domain,
                x=source_ba,
                y=held_ba,
                count=unit_count,
                denominator=guard_count,
                point_index=point_index,
            )
        )
        entries.append(
            (
                point_index,
                station,
                phase,
                domain,
                instrument,
                model,
                aggregation,
                mode,
                unit_count,
            )
        )
    by_point: dict = {}
    for entry in entries:
        by_point.setdefault(entry[0], []).append(entry)
    full_endpoints = {
        (model, aggregation) for model in MODEL_ORDER for aggregation in AGGREGATION_ORDER
    }
    for group_entries in by_point.values():
        if {(entry[5], entry[6]) for entry in group_entries} != full_endpoints:
            raise DiagnosticFigureError("source_vs_held_endpoints_incomplete")
        base = group_entries[0]
        for entry in group_entries[1:]:
            if (entry[1], entry[2], entry[3], entry[4], entry[7], entry[8]) != (
                base[1],
                base[2],
                base[3],
                base[4],
                base[7],
                base[8],
            ):
                raise DiagnosticFigureError("source_vs_held_point_inconsistent")
    return rows


def _project_best_epoch(table: pd.DataFrame) -> list[dict]:
    _require_columns(table, BEST_EPOCH_COLUMNS, "best_epoch_columns_missing")
    if table.empty:
        return []
    validated = []
    seen: set = set()
    for record in table[list(BEST_EPOCH_COLUMNS)].to_dict("records"):
        station = _enum(record["station"], STATION_ORDER, "best_epoch_station_unknown")
        phase = _enum(record["phase"], PHASE_ORDER, "best_epoch_phase_unknown")
        mode = _enum(record["selection_mode"], SELECTION_MODES, "best_epoch_mode_unknown")
        recipe = _enum(record["recipe"], RECIPE_ORDER, "best_epoch_recipe_unknown")
        slot_kind = _enum(record["slot_kind"], SLOT_KIND_ORDER, "best_epoch_slot_kind_unknown")
        best_epoch = _integer(record["best_epoch"], "best_epoch_value_invalid")
        if not (MINIMUM_EPOCH <= best_epoch <= MAXIMUM_EPOCH):
            raise DiagnosticFigureError("best_epoch_value_out_of_range")
        fit_count = _integer(record["fit_count"], "best_epoch_fit_count_invalid")
        if fit_count <= 0:
            raise DiagnosticFigureError("best_epoch_fit_count_invalid")
        key = (station, phase, mode, recipe, slot_kind, best_epoch)
        if key in seen:
            raise DiagnosticFigureError("best_epoch_duplicate_key")
        seen.add(key)
        validated.append((station, phase, recipe, slot_kind, best_epoch, fit_count))
    totals: dict = {}
    aggregated: dict = {}
    for station, phase, recipe, slot_kind, best_epoch, fit_count in validated:
        group = (station, phase, slot_kind, recipe)
        totals[group] = totals.get(group, 0) + fit_count
        point = (station, phase, slot_kind, recipe, best_epoch)
        aggregated[point] = aggregated.get(point, 0) + fit_count
    rows = []
    for (station, phase, slot_kind, recipe, best_epoch), count in aggregated.items():
        denominator = totals[(station, phase, slot_kind, recipe)]
        fraction = count / denominator
        if not 0.0 <= fraction <= 1.0:
            raise DiagnosticFigureError("best_epoch_fraction_out_of_range")
        rows.append(
            _make_row(
                figure=_figure_id(FAMILY_EPOCH, phase, slot_kind),
                station=station,
                panel=STATION_LABELS[station],
                panel_row=0,
                panel_col=STATION_ORDER.index(station),
                series=recipe,
                phase=phase,
                recipe=recipe,
                x=best_epoch,
                y=fraction,
                count=count,
                denominator=denominator,
            )
        )
    return rows


def _project_reliability(table: pd.DataFrame) -> list[dict]:
    _require_columns(table, RELIABILITY_COLUMNS, "reliability_columns_missing")
    if table.empty:
        return []
    validated = []
    seen: set = set()
    for record in table[list(RELIABILITY_COLUMNS)].to_dict("records"):
        station = _enum(record["station"], STATION_ORDER, "reliability_station_unknown")
        phase = _enum(record["phase"], PHASE_ORDER, "reliability_phase_unknown")
        model = _enum(record["model_id"], MODEL_ORDER, "reliability_model_unknown")
        aggregation = _enum(
            record["aggregation_id"], AGGREGATION_ORDER, "reliability_aggregation_unknown"
        )
        bin_index = _integer(record["bin_index"], "reliability_bin_index_invalid")
        if bin_index <= 0:
            raise DiagnosticFigureError("reliability_bin_index_invalid")
        count = _integer(record["count"], "reliability_count_invalid")
        if count <= 0:
            raise DiagnosticFigureError("reliability_count_invalid")
        mean_confidence = _unit(record["mean_confidence"], "reliability_mean_confidence_invalid")
        observed_accuracy = _unit(
            record["observed_accuracy"], "reliability_observed_accuracy_invalid"
        )
        signed_gap = _number(record["signed_gap"], "reliability_signed_gap_invalid")
        if abs(signed_gap - (observed_accuracy - mean_confidence)) > RELIABILITY_TOLERANCE:
            raise DiagnosticFigureError("reliability_signed_gap_inconsistent")
        bin_weight = _number(record["bin_weight"], "reliability_bin_weight_invalid")
        key = (station, phase, model, aggregation, bin_index)
        if key in seen:
            raise DiagnosticFigureError("reliability_duplicate_key")
        seen.add(key)
        validated.append(
            (
                station,
                phase,
                model,
                aggregation,
                bin_index,
                count,
                mean_confidence,
                observed_accuracy,
                bin_weight,
            )
        )
    groups: dict = {}
    for entry in validated:
        groups.setdefault(entry[:4], []).append(entry)
    for entries in groups.values():
        total = sum(entry[5] for entry in entries)
        num_bins = min(10, total)
        quotient, remainder = divmod(total, num_bins)
        expected = [quotient + int(index < remainder) for index in range(num_bins)]
        ordered = sorted(entries, key=lambda entry: entry[4])
        if [entry[4] for entry in ordered] != list(range(1, num_bins + 1)):
            raise DiagnosticFigureError("reliability_bin_index_invalid")
        for entry in ordered:
            if entry[5] != expected[entry[4] - 1]:
                raise DiagnosticFigureError("reliability_bin_mass_inconsistent")
            if abs(entry[8] - entry[5] / total) > RELIABILITY_TOLERANCE:
                raise DiagnosticFigureError("reliability_bin_weight_inconsistent")
        confidences = [entry[6] for entry in ordered]
        for left, right in zip(confidences[:-1], confidences[1:], strict=True):
            if left > right + RELIABILITY_TOLERANCE:
                raise DiagnosticFigureError("reliability_confidence_order_invalid")
    rows = []
    for (
        station,
        phase,
        model,
        aggregation,
        bin_index,
        count,
        mean_confidence,
        observed_accuracy,
        _bin_weight,
    ) in validated:
        denominator = sum(entry[5] for entry in groups[(station, phase, model, aggregation)])
        rows.append(
            _make_row(
                figure=_figure_id(FAMILY_RELIABILITY, phase, aggregation),
                station=station,
                panel=STATION_LABELS[station],
                panel_row=0,
                panel_col=STATION_ORDER.index(station),
                series=model,
                phase=phase,
                aggregation_id=aggregation,
                bin_index=bin_index,
                x=mean_confidence,
                y=observed_accuracy,
                count=count,
                denominator=denominator,
            )
        )
    return rows


def _order_semantic(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.assign(
        _figure=frame.figure,
        _row=frame.panel_row.astype(int),
        _col=frame.panel_col.astype(int),
        _series=frame.series,
        _bin=frame.bin_index.fillna(0).astype(int),
        _point=frame.point_index.fillna(0).astype(int),
        _x=frame.x.astype(float),
        _domain=frame.domain,
    )
    ordered = ordered.sort_values(
        ["_figure", "_row", "_col", "_series", "_bin", "_point", "_x", "_domain"],
        kind="stable",
    )
    return ordered[list(SEMANTIC_COLUMNS)].reset_index(drop=True)


def _build_semantic(
    public_metrics: object, source_diagnostics: object, reliability: object
) -> pd.DataFrame:
    metrics = _require_mapping(public_metrics, "public_metrics_malformed")
    diagnostics = _require_mapping(source_diagnostics, "source_diagnostics_malformed")
    reliability_tables = _require_mapping(reliability, "reliability_malformed")
    rows: list[dict] = []
    rows.extend(
        _project_paired_domains(
            _require_frame(metrics, "paired_domains", "public_metrics_paired_domains_missing")
        )
    )
    rows.extend(
        _project_source_vs_held(
            _require_frame(
                diagnostics, "source_vs_held", "source_diagnostics_source_vs_held_missing"
            )
        )
    )
    rows.extend(
        _project_best_epoch(
            _require_frame(
                diagnostics,
                "best_epoch_distribution",
                "source_diagnostics_best_epoch_missing",
            )
        )
    )
    rows.extend(
        _project_reliability(
            _require_frame(reliability_tables, "reliability_bins", "reliability_bins_missing")
        )
    )
    if not rows:
        raise DiagnosticFigureError("no_diagnostic_points")
    frame = pd.DataFrame(rows, columns=list(SEMANTIC_COLUMNS))
    values = frame[["x", "y"]].to_numpy(dtype=float)
    nan_rows = np.isnan(values).any(axis=1)
    if nan_rows.any() and (frame.loc[nan_rows, "count"].to_numpy(dtype=int) != 0).any():
        raise DiagnosticFigureError("semantic_point_missing")
    if np.isinf(values).any():
        raise DiagnosticFigureError("semantic_point_nonfinite")
    return _order_semantic(frame)


# --------------------------------------------------------------------------- #
# Figure specification and styling
# --------------------------------------------------------------------------- #


def _figure_specs(frame: pd.DataFrame) -> list[dict]:
    specs = []
    for figure in sorted(frame["figure"].unique()):
        parts = str(figure).split("|")
        if len(parts) != 3:
            raise DiagnosticFigureError("malformed_figure_id")
        family, first, second = parts
        spec = {"figure": str(figure), "family": family}
        if family == FAMILY_PAIRED:
            spec["reference"] = first
            spec["aggregation"] = second
        elif family == FAMILY_SOURCE:
            spec["phase"] = first
            spec["aggregation"] = second
        elif family == FAMILY_EPOCH:
            spec["phase"] = first
            spec["slot_kind"] = second
        elif family == FAMILY_RELIABILITY:
            spec["phase"] = first
            spec["aggregation"] = second
        else:
            raise DiagnosticFigureError("unknown_figure_family")
        spec["stem"] = _figure_stem(spec)
        spec["title"] = _figure_title(spec)
        specs.append(spec)
    return specs


def _safe(value: object) -> str:
    return "".join(character if character.isalnum() else "_" for character in str(value))


def _figure_stem(spec: dict) -> str:
    family = spec["family"]
    if family == FAMILY_PAIRED:
        return f"P05D_paired_{_safe(spec['reference'])}_{spec['aggregation']}"
    if family == FAMILY_SOURCE:
        return f"P05D_source_vs_held_{spec['phase']}_{spec['aggregation']}"
    if family == FAMILY_EPOCH:
        return f"P05D_best_epoch_{spec['phase']}_{_safe(spec['slot_kind'])}"
    if family == FAMILY_RELIABILITY:
        return f"P05D_reliability_{spec['phase']}_{spec['aggregation']}"
    raise DiagnosticFigureError("unknown_figure_family")


def _figure_title(spec: dict) -> str:
    family = spec["family"]
    phase_label = {
        "development": "development",
        "held_evaluation": "held-instrument evaluation",
    }.get(spec.get("phase"), "")
    if family == FAMILY_PAIRED:
        return (
            "Paired balanced accuracy: new strategies vs "
            f"{spec['reference']} ({spec['aggregation']})"
        )
    if family == FAMILY_SOURCE:
        return f"Source versus outer balanced accuracy: {phase_label} ({spec['aggregation']})"
    if family == FAMILY_EPOCH:
        slot_label = {
            "inherited_selection_fit": "inherited source fits",
            "guard_selection_fit": "master-CV guard fits",
        }[spec["slot_kind"]]
        return f"Best-epoch distribution: {phase_label} ({slot_label})"
    if family == FAMILY_RELIABILITY:
        return f"Reliability: {phase_label} ({spec['aggregation']})"
    raise DiagnosticFigureError("unknown_figure_family")


def _caption_lines(spec: dict) -> tuple[str, ...]:
    family = spec["family"]
    if family == FAMILY_PAIRED:
        return PAIRED_CAPTION
    if family == FAMILY_SOURCE:
        return SOURCE_CAPTION
    if family == FAMILY_EPOCH:
        return EPOCH_CAPTION
    if family == FAMILY_RELIABILITY:
        return RELIABILITY_CAPTION
    raise DiagnosticFigureError("unknown_figure_family")


def _empty_panel_label(spec: dict) -> str:
    if spec["family"] == FAMILY_PAIRED:
        return "no common complete contexts"
    return "No plotted points"


def _panel_grid(spec: dict) -> tuple[int, int]:
    if spec["family"] == FAMILY_SOURCE:
        return (2, 3)
    return (1, 3)


def _panel_titles(spec: dict, rows: int, cols: int) -> dict:
    titles = {}
    for row in range(rows):
        for col in range(cols):
            letter = chr(65 + row * cols + col)
            station = STATION_ORDER[col]
            if spec["family"] == FAMILY_SOURCE:
                titles[(row, col)] = (
                    f"({letter}) {STATION_LABELS[station]}: {SOURCE_ROW_LABELS[row]}"
                )
            else:
                titles[(row, col)] = f"({letter}) {STATION_LABELS[station]}"
    return titles


def _panel_axes(spec: dict) -> dict:
    family = spec["family"]
    if family == FAMILY_PAIRED:
        return {
            "xlabel": "delta balanced accuracy (new - reference)",
            "ylabel": "reference balanced accuracy",
            "xmin": -1.0,
            "xmax": 1.0,
            "ymin": 0.0,
            "ymax": 1.0,
            "xtick": "-1,-0.5,0,0.5,1",
            "ytick": "0,0.5,1",
            "xmid": 0.0,
            "ymid": 0.5,
            "identity": "x0",
        }
    if family == FAMILY_SOURCE:
        return {
            "xlabel": "source validation balanced accuracy",
            "ylabel": "outer ensemble balanced accuracy",
            "xmin": 0.0,
            "xmax": 1.0,
            "ymin": 0.0,
            "ymax": 1.0,
            "xtick": "0,0.5,1",
            "ytick": "0,0.5,1",
            "xmid": 0.5,
            "ymid": 0.5,
            "identity": None,
        }
    if family == FAMILY_EPOCH:
        ticks = [int(value) for value in np.linspace(MINIMUM_EPOCH, MAXIMUM_EPOCH, 5)]
        return {
            "xlabel": "best epoch",
            "ylabel": "fraction of source fits",
            "xmin": float(MINIMUM_EPOCH),
            "xmax": float(MAXIMUM_EPOCH),
            "ymin": 0.0,
            "ymax": 1.0,
            "xtick": ",".join(str(value) for value in ticks),
            "ytick": "0,0.5,1",
            "xmid": 0.5 * (MINIMUM_EPOCH + MAXIMUM_EPOCH),
            "ymid": 0.5,
            "identity": None,
        }
    if family == FAMILY_RELIABILITY:
        return {
            "xlabel": "mean confidence",
            "ylabel": "observed accuracy",
            "xmin": 0.0,
            "xmax": 1.0,
            "ymin": 0.0,
            "ymax": 1.0,
            "xtick": "0,0.5,1",
            "ytick": "0,0.5,1",
            "xmid": 0.5,
            "ymid": 0.5,
            "identity": "xy",
        }
    raise DiagnosticFigureError("unknown_figure_family")


def _series_order(spec: dict) -> tuple:
    if spec["family"] == FAMILY_EPOCH:
        return RECIPE_ORDER
    return MODEL_ORDER


def _series_style(family: str, series: str) -> dict:
    if family == FAMILY_EPOCH:
        if series not in RECIPE_ORDER:
            raise DiagnosticFigureError("unknown_recipe_series")
        index = RECIPE_ORDER.index(series)
        if index >= len(RECIPE_TEX_COLORS):
            raise DiagnosticFigureError("recipe_palette_exhausted")
        return {
            "color": RECIPE_TEX_COLORS[index],
            "mark": RECIPE_TEX_MARKS[index],
            "open": RECIPE_TEX_OPEN[index],
            "size": "2.0pt",
            "html_color": RECIPE_HTML_COLORS[index],
            "html_symbol": RECIPE_HTML_SYMBOLS[index],
            "html_size": 9,
            "label": series,
        }
    if series not in MODEL_ORDER:
        raise DiagnosticFigureError("unknown_model_series")
    return {
        "color": MODEL_TEX_COLOR[series],
        "mark": MODEL_TEX_MARK[series],
        "open": MODEL_TEX_OPEN[series],
        "size": "1.6pt" if series == "D0-M" else "2.3pt",
        "html_color": MODEL_HTML_COLOR[series],
        "html_symbol": MODEL_HTML_SYMBOL[series],
        "html_size": 7 if series == "D0-M" else 10,
        "label": MODEL_LABELS[series],
    }


# --------------------------------------------------------------------------- #
# Native pgfplots/TikZ rendering
# --------------------------------------------------------------------------- #


def _tikz_preamble() -> list[str]:
    return [
        r"\documentclass[tikz,border=5pt]{standalone}",
        r"\ifdefined\pdfinfoomitdate\pdfinfoomitdate=1\fi",
        r"\ifdefined\pdfsuppressptexinfo\pdfsuppressptexinfo=-1\fi",
        r"\ifdefined\pdftrailerid\pdftrailerid{}\fi",
        r"\usepackage{pgfplots}",
        r"\usepgfplotslibrary{groupplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\definecolor{p05Blue}{HTML}{0072B2}",
        r"\definecolor{p05Orange}{HTML}{E69F00}",
        r"\definecolor{p05Green}{HTML}{009E73}",
        r"\definecolor{p05Purple}{HTML}{CC79A7}",
        r"\definecolor{p05Red}{HTML}{D55E00}",
        r"\definecolor{p05Sky}{HTML}{56B4E9}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
    ]


def _tikz_addplot(
    coordinates: str, *, color: str, mark: str, open_mark: bool, size: str, connect: bool
) -> str:
    fill = "none" if open_mark else color
    if connect:
        return (
            rf"\addplot[mark={mark}, mark size={size}, color={color}, line width=0.8pt, "
            rf"mark options={{fill={fill}, draw={color}}}] coordinates {{{coordinates}}};"
        )
    return (
        rf"\addplot[only marks, mark={mark}, mark size={size}, color={color}, "
        rf"mark options={{fill={fill}, draw={color}}}] coordinates {{{coordinates}}};"
    )


def _tikz_marker(color: str, mark: str, open_mark: bool) -> str:
    if mark == "*":
        return rf"\tikz[baseline=-0.6ex]\fill[{color}](0,0)circle(1.4pt);"
    if mark == "square":
        if open_mark:
            return rf"\tikz[baseline=-0.6ex]\draw[{color}](0,0)rectangle(2.8pt,2.8pt);"
        return rf"\tikz[baseline=-0.6ex]\fill[{color}](0,0)rectangle(2.8pt,2.8pt);"
    if mark == "triangle":
        if open_mark:
            return rf"\tikz[baseline=-0.6ex]\draw[{color}](0,0)--(1.6pt,2.8pt)--(3.2pt,0)--cycle;"
        return rf"\tikz[baseline=-0.6ex]\fill[{color}](0,0)--(1.6pt,2.8pt)--(3.2pt,0)--cycle;"
    if mark == "diamond":
        if open_mark:
            return rf"\tikz[baseline=-0.6ex]\draw[{color}](0,0)--(1.4pt,1.4pt)--(2.8pt,0)--(1.4pt,-1.4pt)--cycle;"
        return rf"\tikz[baseline=-0.6ex]\fill[{color}](0,0)--(1.4pt,1.4pt)--(2.8pt,0)--(1.4pt,-1.4pt)--cycle;"
    raise DiagnosticFigureError("unknown_marker")


def _legend_entries(spec: dict) -> list[dict]:
    entries = []
    for series in _series_order(spec):
        style = _series_style(spec["family"], series)
        entries.append(
            {
                "label": style["label"],
                "marker": _tikz_marker(style["color"], style["mark"], style["open"]),
            }
        )
    return entries


def _panel_tikz_lines(
    spec: dict, points: pd.DataFrame, row: int, col: int, axes: dict
) -> list[str]:
    lines = []
    if axes["identity"] == "x0":
        lines.append(r"\addplot[black, dotted, line width=0.8pt] coordinates {(0,0) (0,1)};")
    elif axes["identity"] == "xy":
        lines.append(r"\addplot[black, dotted, line width=0.8pt] coordinates {(0,0) (1,1)};")
    cell = points[(points.panel_row == row) & (points.panel_col == col)]
    cell = cell[cell["count"] > 0]
    connect = spec["family"] == FAMILY_RELIABILITY
    for series in _series_order(spec):
        sub = cell[cell.series.eq(series)]
        if sub.empty:
            continue
        if connect:
            sub = sub.sort_values("bin_index", kind="stable")
        coordinates = " ".join(
            f"({float(x):.17g},{float(y):.17g})" for x, y in zip(sub.x, sub.y, strict=True)
        )
        style = _series_style(spec["family"], series)
        lines.append(
            _tikz_addplot(
                coordinates,
                color=style["color"],
                mark=style["mark"],
                open_mark=style["open"],
                size=style["size"],
                connect=connect,
            )
        )
    if cell.empty:
        position = f"{axes['xmid']:.6g},{axes['ymid']:.6g}"
        lines.append(
            r"\node[font=\fontsize{8}{10}\selectfont,align=center] at (axis cs:"
            + position
            + r") {"
            + _empty_panel_label(spec)
            + r"};"
        )
    return lines


def _tikz_figure(spec: dict, points: pd.DataFrame, digest: str) -> str:
    axes = _panel_axes(spec)
    rows, cols = _panel_grid(spec)
    titles = _panel_titles(spec, rows, cols)
    lines = [
        f"% P05 diagnostic figure {spec['figure']}; allowlisted public aggregate points.",
        "% Each point is an aggregate mean or bin over repeated contexts, not an independent sample.",
        f"% data_sha256={digest}",
    ]
    lines.extend(_tikz_preamble())
    lines.append(r"\begin{groupplot}[")
    lines.append(
        rf"  group style={{group size={cols} by {rows}, horizontal sep=1.0cm, vertical sep=1.8cm}},"
    )
    lines.append(r"  width=4.8cm, height=4.8cm, scale only axis,")
    lines.append(
        rf"  xmin={axes['xmin']:.6g}, xmax={axes['xmax']:.6g}, ymin={axes['ymin']:.6g}, ymax={axes['ymax']:.6g},"
    )
    lines.append(rf"  xtick={{{axes['xtick']}}}, ytick={{{axes['ytick']}}},")
    lines.append(r"  tick label style={font=\fontsize{8}{10}\selectfont, color=black},")
    lines.append(r"  label style={font=\fontsize{8}{10}\selectfont, color=black},")
    lines.append(r"  title style={font=\fontsize{8}{10}\selectfont\bfseries, color=black},")
    lines.append(r"]")
    for row in range(rows):
        for col in range(cols):
            options = [f"title={{{titles[(row, col)]}}}", f"xlabel={{{axes['xlabel']}}}"]
            if col == 0:
                options.append(f"ylabel={{{axes['ylabel']}}}")
            lines.append(r"\nextgroupplot[" + ", ".join(options) + "]")
            lines.extend(_panel_tikz_lines(spec, points, row, col, axes))
    lines.append(r"\end{groupplot}")
    lines.append(
        r"\node[anchor=south, font=\fontsize{8}{10}\selectfont\bfseries, color=black, align=center, text width=16.6cm] at (current bounding box.north) {"
        + _tex_literal(spec["title"])
        + "};"
    )
    legend_body = r"\quad ".join(
        entry["marker"] + r"\ " + _tex_literal(entry["label"]) for entry in _legend_entries(spec)
    )
    lines.append(
        r"\node[anchor=north, font=\fontsize{8}{10}\selectfont, color=black, align=left, text width=16.6cm] (p05legend) at ([yshift=-2mm]current bounding box.south) {"
        + legend_body
        + "};"
    )
    caption = r"\\".join(_tex_literal(line) for line in _caption_lines(spec))
    lines.append(
        r"\node[anchor=north, font=\fontsize{8}{10}\selectfont, color=black, align=left, text width=16.6cm] at ([yshift=-1mm]p05legend.south) {"
        + caption
        + "};"
    )
    lines.extend([r"\end{tikzpicture}", r"\end{document}", ""])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Offline Plotly rendering
# --------------------------------------------------------------------------- #


def _html_caption(spec: dict) -> str:
    return "<br>".join(_html_literal(line) for line in _caption_lines(spec))


def _hover_customdata(spec: dict, sub: pd.DataFrame) -> list[list]:
    family = spec["family"]
    if family == FAMILY_PAIRED:
        return [
            [
                _html_literal(row.domain),
                _html_literal(row.held_instrument),
                int(row.count),
                int(row.denominator),
            ]
            for row in sub.itertuples(index=False)
        ]
    if family == FAMILY_SOURCE:
        return [
            [
                int(row.point_index),
                _html_literal(row.recipe),
                _html_literal(row.selection_mode),
                int(row.count),
                int(row.denominator),
            ]
            for row in sub.itertuples(index=False)
        ]
    return [[int(row.count), int(row.denominator)] for row in sub.itertuples(index=False)]


def _hover_template(spec: dict) -> str:
    family = spec["family"]
    if family == FAMILY_PAIRED:
        return (
            "%{customdata[0]} &middot; %{customdata[1]}"
            "<br>delta BA=%{x:.3f}<br>reference BA=%{y:.3f}"
            "<br>common complete contexts=%{customdata[2]}/%{customdata[3]}"
            "<extra></extra>"
        )
    if family == FAMILY_SOURCE:
        return (
            "point %{customdata[0]}<br>recipe=%{customdata[1]}"
            "<br>selection mode=%{customdata[2]}"
            "<br>source BA=%{x:.3f}<br>outer BA=%{y:.3f}"
            "<br>inherited units=%{customdata[3]}<br>guard units=%{customdata[4]}"
            "<extra></extra>"
        )
    if family == FAMILY_EPOCH:
        return (
            "best epoch=%{x}<br>fraction of fits=%{y:.3f}"
            "<br>fits=%{customdata[0]}/%{customdata[1]}<extra></extra>"
        )
    if family == FAMILY_RELIABILITY:
        return (
            "mean confidence=%{x:.3f}<br>observed accuracy=%{y:.3f}"
            "<br>bin appearances=%{customdata[0]}<br>total appearances=%{customdata[1]}"
            "<extra></extra>"
        )
    raise DiagnosticFigureError("unknown_figure_family")


def _plotly_figure(spec: dict, points: pd.DataFrame) -> go.Figure:
    axes = _panel_axes(spec)
    rows, cols = _panel_grid(spec)
    titles = _panel_titles(spec, rows, cols)
    figure = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[titles[(row, col)] for row in range(rows) for col in range(cols)],
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )
    connect = spec["family"] == FAMILY_RELIABILITY
    shown: set = set()
    for row in range(rows):
        for col in range(cols):
            panel_row, panel_col = row + 1, col + 1
            cell = points[(points.panel_row == row) & (points.panel_col == col)]
            cell = cell[cell["count"] > 0]
            for series in _series_order(spec):
                sub = cell[cell.series.eq(series)]
                if sub.empty:
                    continue
                if connect:
                    sub = sub.sort_values("bin_index", kind="stable")
                style = _series_style(spec["family"], series)
                trace = {
                    "x": [float(value) for value in sub.x],
                    "y": [float(value) for value in sub.y],
                    "mode": "lines+markers" if connect else "markers",
                    "name": style["label"],
                    "legendgroup": series,
                    "showlegend": series not in shown,
                    "marker": {
                        "symbol": style["html_symbol"],
                        "color": style["html_color"],
                        "size": style["html_size"],
                        "line": {"color": style["html_color"], "width": 1.2},
                    },
                    "customdata": _hover_customdata(spec, sub),
                    "hovertemplate": _hover_template(spec),
                }
                if connect:
                    trace["line"] = {"color": style["html_color"], "width": 1.4}
                figure.add_trace(go.Scatter(**trace), row=panel_row, col=panel_col)
                shown.add(series)
            if axes["identity"] == "x0":
                figure.add_trace(
                    go.Scatter(
                        x=[0.0, 0.0],
                        y=[0.0, 1.0],
                        mode="lines",
                        line={"color": "black", "dash": "dot", "width": 1},
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=panel_row,
                    col=panel_col,
                )
            elif axes["identity"] == "xy":
                figure.add_trace(
                    go.Scatter(
                        x=[0.0, 1.0],
                        y=[0.0, 1.0],
                        mode="lines",
                        line={"color": "black", "dash": "dot", "width": 1},
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=panel_row,
                    col=panel_col,
                )
            if cell.empty:
                empty_text = (
                    "no common<br>complete contexts"
                    if spec["family"] == FAMILY_PAIRED
                    else "No plotted<br>points"
                )
                figure.add_annotation(
                    x=axes["xmid"],
                    y=axes["ymid"],
                    text=empty_text,
                    showarrow=False,
                    row=panel_row,
                    col=panel_col,
                )
    for row in range(rows):
        for col in range(cols):
            figure.update_xaxes(
                title_text=axes["xlabel"],
                range=[axes["xmin"], axes["xmax"]],
                constrain="domain",
                row=row + 1,
                col=col + 1,
            )
            figure.update_yaxes(
                range=[axes["ymin"], axes["ymax"]], constrain="domain", row=row + 1, col=col + 1
            )
            if col == 0:
                figure.update_yaxes(title_text=axes["ylabel"], row=row + 1, col=col + 1)
    if spec["family"] in (FAMILY_SOURCE, FAMILY_RELIABILITY):
        for row in range(rows):
            for col in range(cols):
                index = row * cols + col + 1
                anchor = "x" if index == 1 else f"x{index}"
                figure.update_yaxes(scaleanchor=anchor, scaleratio=1.0, row=row + 1, col=col + 1)
    height = 740 if rows == 1 else 1080
    bottom = 230 if rows == 1 else 260
    legend_y = -0.20 if rows == 1 else -0.11
    caption_y = -0.30 if rows == 1 else -0.17
    figure.update_layout(
        title={"text": spec["title"], "x": 0.5},
        template="simple_white",
        font={"family": "Times New Roman, Times, serif", "color": "black", "size": 12},
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=1500,
        height=height,
        margin={"l": 70, "r": 40, "t": 110, "b": bottom},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": legend_y,
            "font": {"size": 11, "color": "black"},
        },
    )
    figure.add_annotation(
        x=0.5,
        y=caption_y,
        xref="paper",
        yref="paper",
        xanchor="center",
        yanchor="top",
        showarrow=False,
        align="left",
        font={"size": 10, "color": "black"},
        text=_html_caption(spec),
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
    figure.update_yaxes(showgrid=True, gridcolor="#e8e8e8")
    return figure


# --------------------------------------------------------------------------- #
# Manifest and entry point
# --------------------------------------------------------------------------- #


def _figure_counts(spec: dict, points: pd.DataFrame) -> dict:
    rows, cols = _panel_grid(spec)
    panel_counts = {}
    empty_panels = []
    for row in range(rows):
        for col in range(cols):
            cell = points[(points.panel_row == row) & (points.panel_col == col)]
            plotted = cell[cell["count"] > 0]
            label = f"r{row}c{col}"
            panel_counts[label] = int(len(plotted))
            if plotted.empty:
                empty_panels.append(label)
    return {
        "point_count": int((points["count"] > 0).sum()),
        "semantic_rows": int(len(points)),
        "panel_counts": panel_counts,
        "empty_panels": empty_panels,
    }


def _paired_context_contributions(frame: pd.DataFrame) -> dict:
    contributions: dict = {}
    paired = frame[frame.figure.str.startswith(FAMILY_PAIRED + "|")]
    for row in paired.itertuples(index=False):
        entry = contributions.setdefault(row.series, {"planned_contexts": 0, "common_contexts": 0})
        entry["planned_contexts"] += int(row.denominator)
        entry["common_contexts"] += int(row.count)
    return contributions


def _manifest_files(root: Path) -> list[dict]:
    files = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name == MANIFEST_NAME or path.is_symlink() or not path.is_file():
            continue
        if path.suffix not in ALLOWED_SUFFIXES:
            continue
        files.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    return files


def generate_diagnostic_figures(
    *,
    public_metrics: object,
    source_diagnostics: object,
    reliability: object,
    output_root: object,
    deadline: float | None = None,
) -> dict:
    """Project authenticated aggregate tables and render all P05 diagnostic figures.

    The three inputs are the anonymous tables returned by
    ``p05_public_metrics.build_public_metrics``,
    ``p05_source_diagnostics.build_source_diagnostics`` and
    ``p05_reliability.build_reliability``. Every plotted point is validated
    against known enum values, finite ranges, counts and duplicate keys, then
    projected into one canonical semantic table. All native TikZ, offline HTML,
    PDF and PNG previews are rendered from that single table and share its
    digest. The function never trains, predicts, selects, bootstraps or reads
    private reference frames.
    """

    frame = _build_semantic(public_metrics, source_diagnostics, reliability)
    specs = _figure_specs(frame)
    if not specs:
        raise DiagnosticFigureError("no_diagnostic_figures")
    root = _guard_output_root(output_root)
    _configure_deterministic_pdf()
    root.mkdir(parents=True)
    csv_path = root / SEMANTIC_NAME
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    digest = sha256_file(csv_path)
    figure_records = []
    for spec in specs:
        points = frame[frame.figure.eq(spec["figure"])].reset_index(drop=True)
        stem = spec["stem"]
        tex_path = root / f"{stem}.tex"
        html_path = root / f"{stem}.html"
        pdf_path = root / f"{stem}.pdf"
        png_path = root / f"{stem}.png"
        log_path = root / f"{stem}.log"
        _write_html(
            _plotly_figure(spec, points),
            html_path,
            digest=digest,
            description=spec["title"] + " " + _caption_lines(spec)[0],
        )
        tex_path.write_text(_tikz_figure(spec, points, digest), encoding="utf-8")
        if deadline is None:
            _compile(tex_path, pdf_path, png_path, log_path)
        else:
            _compile(tex_path, pdf_path, png_path, log_path, deadline=deadline)
        if digest not in tex_path.read_text(encoding="utf-8") or digest not in html_path.read_text(
            encoding="utf-8"
        ):
            raise DiagnosticFigureError("semantic_hash_parity_failed")
        figure_records.append(
            {
                "figure_id": spec["figure"],
                "family": spec["family"],
                "stem": stem,
                "semantic_sha256": digest,
                "tex_sha256": sha256_file(tex_path),
                "pdf_sha256": sha256_file(pdf_path),
                "png_sha256": sha256_file(png_path),
                "html_sha256": sha256_file(html_path),
                **_figure_counts(spec, points),
            }
        )
    manifest = {
        "data_sha256": digest,
        "semantic_path": SEMANTIC_NAME,
        "semantic_rows": int(len(frame)),
        "plotted_rows": int((frame["count"] > 0).sum()),
        "planned_common_model_context_contributions": _paired_context_contributions(frame),
        "figures": figure_records,
        "files": _manifest_files(root),
    }
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
