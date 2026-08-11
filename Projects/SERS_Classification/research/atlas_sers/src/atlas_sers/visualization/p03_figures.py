"""Publication-ready paired native/interactive figures for P03 classical evidence."""

from __future__ import annotations

import html
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from atlas_sers.evaluation.p03_analysis import P03DiagnosticTables
from atlas_sers.governance.canonical import sha256_file
from atlas_sers.visualization.p01_figures import _tex

COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#E69F00",
    "#56B4E9",
    "#CC79A7",
    "#000000",
    "#F0E442",
]
PLOTLY_MARKERS = ["circle", "square", "triangle-up", "diamond", "x", "cross"]
PLOTLY_DASHES = ["solid", "dash", "dot", "dashdot", "longdash"]
TIKZ_MARKERS = ["*", "square*", "triangle*", "diamond*", "x", "+"]
TIKZ_DASHES = ["solid", "densely dashed", "densely dotted", "dashdotted"]

FIGURE_SLUGS = {
    "F12": "classical_selection",
    "F13": "classical_t1",
    "F38": "classical_t3_domains",
    "F39": "classical_t1_t3",
    "F40": "classical_spectrum_master",
    "F41": "classical_confusion",
    "F42": "classical_calibration",
    "F43": "classical_controls",
}


@dataclass(frozen=True)
class P03Panel:
    panel_id: str
    title: str
    x_label: str
    y_label: str
    mode: str = "scatter"
    identity_line: bool = False
    chance_line: float | None = None
    y_min: float | None = None
    y_max: float | None = None


@dataclass(frozen=True)
class P03FigureDefinition:
    figure_id: str
    title: str
    caption: str
    scope: str
    research_question_id: str
    independent_unit: str
    frame: pd.DataFrame
    panels: tuple[P03Panel, ...]


def _common(
    frame: pd.DataFrame,
    *,
    figure_id: str,
    scope: str,
    research_question_id: str,
    independent_unit: str,
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "figure_id", figure_id)
    result.insert(1, "scope", scope)
    result.insert(2, "research_question_id", research_question_id)
    result.insert(3, "population_id", "primary_598")
    result.insert(4, "representation_id", "R_MIN_400_1800")
    result.insert(5, "preprocessing_policy_id", "PP-U-MIN")
    result.insert(6, "information_regime", "source-only; no target statistics")
    result.insert(7, "independent_unit", independent_unit)
    return result


def _selection_figure(diagnostics: P03DiagnosticTables) -> P03FigureDefinition:
    frequency = diagnostics.selection_frequency
    frequency = frequency[
        frequency.experiment_id.astype(str).eq("EXP-C09-T3")
    ].copy()
    frequency_rows = pd.DataFrame(
        {
            "panel_id": "frequency",
            "x_value": frequency.selection_outcome_model,
            "y_value": frequency.selection_fraction,
            "y_lower": np.nan,
            "y_upper": np.nan,
            "series": frequency.station,
            "trace_group": frequency.selection_outcome_candidate,
            "point_status": np.where(
                frequency.selection_outcome_candidate.eq("<selection_failure>"),
                "unavailable",
                "complete",
            ),
            "point_label": frequency.selection_outcome_candidate,
            "denominator": frequency.selection_denominator,
        }
    )
    stability = diagnostics.selection_stability
    stability = stability[
        stability.experiment_id.astype(str).eq("EXP-C09-T3")
    ].copy()
    stability_rows = pd.DataFrame(
        {
            "panel_id": "stability",
            "x_value": stability.domain,
            "y_value": stability.modal_fraction,
            "y_lower": np.nan,
            "y_upper": np.nan,
            "series": stability.station,
            "trace_group": stability.outer_fold.astype(str),
            "point_status": np.where(
                stability.complete_selection_count.eq(stability.selection_count),
                "complete",
                "incomplete",
            ),
            "point_label": stability.modal_candidate_id,
            "denominator": stability.selection_count,
        }
    )
    margins = diagnostics.selection_margins
    margins = margins[
        margins.experiment_id.astype(str).eq("EXP-C09-T3")
    ].copy()
    margin_rows = pd.DataFrame(
        {
            "panel_id": "margin",
            "x_value": margins.domain,
            "y_value": margins.mean_ba_margin,
            "y_lower": np.nan,
            "y_upper": np.nan,
            "series": margins.station,
            "trace_group": margins.outer_run_id,
            "point_status": margins.margin_status,
            "point_label": margins.selected_candidate_id,
            "denominator": margins.supported_candidate_count,
        }
    )
    frame = pd.concat([frequency_rows, stability_rows, margin_rows], ignore_index=True)
    caption = (
        "(S) RQ-P01 classical source-development selection. Frequency uses all frozen C09 "
        "outer source roles; repeat stability is the modal candidate fraction across the five "
        "technical split repeats for each domain/fold; the margin is selected minus the best "
        "nonselected complete-support candidate on the primary mean pseudo-domain balanced-"
        "accuracy objective. Failures remain in denominators. PP-U-MIN, 598-spectrum primary "
        "population, no held-target spectra, labels, statistics, or outcomes used for selection."
    )
    return P03FigureDefinition(
        "F12",
        "Classical source-only selection frequency and stability",
        caption,
        "S",
        "RQ-P01",
        "outer source-development role",
        _common(
            frame,
            figure_id="F12",
            scope="S",
            research_question_id="RQ-P01",
            independent_unit="outer source-development role",
        ),
        (
            P03Panel("frequency", "Selected family frequency", "Selected family", "Fraction"),
            P03Panel(
                "stability",
                "Five-repeat candidate agreement",
                "Held domain",
                "Modal fraction",
            ),
            P03Panel("margin", "Winner margin", "Held domain", "Mean BA margin", y_min=-0.02),
        ),
    )


def _t1_figure(pooled_metrics: pd.DataFrame) -> P03FigureDefinition:
    t1 = pooled_metrics[
        pooled_metrics.task_id.astype(str).str.startswith("T1-")
    ].copy()
    t1["plot_value"] = t1.balanced_accuracy.where(
        t1.endpoint_status.astype(str).eq("complete"), -0.05
    )
    t1["plot_series"] = t1.aggregation_level.astype(str) + np.where(
        t1.endpoint_status.astype(str).eq("complete"), "", " unavailable"
    )
    frame = pd.DataFrame(
        {
            "panel_id": t1.station,
            "x_value": t1.procedure_id,
            "y_value": t1.plot_value,
            "y_lower": np.nan,
            "y_upper": np.nan,
            "series": t1.plot_series,
            "trace_group": t1.outer_repeat.astype(str),
            "point_status": t1.endpoint_status,
            "point_label": t1.procedure_id,
            "denominator": t1.expected_masters,
            "outer_repeat": t1.outer_repeat,
        }
    )
    caption = (
        "(S) RQ-P01 within-station classical benchmark. Each mark is one complete four-fold "
        "pooled endpoint for one technical split repeat; spectrum and instrument-balanced master "
        "aggregation are separate. Marks at -0.05 denote terminally unavailable endpoints, not "
        "zero accuracy. The black dash-dot line is three-class chance (1/3). Ranges across repeats "
        "are descriptive because repeats are not independent samples. PP-U-MIN on the frozen "
        "598-spectrum/69-master primary population."
    )
    panels = tuple(
        P03Panel(
            station,
            station.capitalize(),
            "Classical procedure",
            "Balanced accuracy",
            chance_line=1 / 3,
            y_min=-0.08,
            y_max=1.02,
        )
        for station in ("cwa", "pills", "surfaces")
    )
    return P03FigureDefinition(
        "F13",
        "Within-station classical performance",
        caption,
        "S",
        "RQ-P01",
        "physical master; split repeat descriptive",
        _common(
            frame,
            figure_id="F13",
            scope="S",
            research_question_id="RQ-P01",
            independent_unit="physical master; split repeat descriptive",
        ),
        panels,
    )


def _domain_figure(pooled_metrics: pd.DataFrame) -> P03FigureDefinition:
    primary = pooled_metrics[
        pooled_metrics.experiment_id.astype(str).eq("EXP-C09-T3")
    ].copy()
    primary["plot_value"] = primary.balanced_accuracy.where(
        primary.endpoint_status.astype(str).eq("complete"), -0.05
    )
    primary["plot_series"] = primary.aggregation_level.astype(str) + np.where(
        primary.endpoint_status.astype(str).eq("complete"), "", " unavailable"
    )
    frame = pd.DataFrame(
        {
            "panel_id": primary.station,
            "x_value": primary.domain,
            "y_value": primary.plot_value,
            "y_lower": np.nan,
            "y_upper": np.nan,
            "series": primary.plot_series,
            "trace_group": primary.outer_repeat.astype(str),
            "point_status": primary.endpoint_status,
            "point_label": primary.domain,
            "denominator": primary.expected_masters,
            "expected_observations": primary.expected_observations,
            "outer_repeat": primary.outer_repeat,
        }
    )
    caption = (
        "(S) RQ-P01 source-only unseen-instrument classical performance for all 13 eligible "
        "station/instrument domains. Marks are domain × technical-repeat pooled endpoints; domains "
        "are not weighted by spectrum count. Marks at -0.05 are terminally unavailable, never "
        "silently removed. The chance line is 1/3. Spectrum and instrument-balanced master levels "
        "are distinct. PP-U-MIN, primary 598-spectrum population, exact P02 roles, and no target "
        "statistics or outcomes in fitting, selection, calibration, or stopping."
    )
    panels = tuple(
        P03Panel(
            station,
            station.capitalize(),
            "Held-instrument domain",
            "Balanced accuracy",
            chance_line=1 / 3,
            y_min=-0.08,
            y_max=1.02,
        )
        for station in ("cwa", "pills", "surfaces")
    )
    return P03FigureDefinition(
        "F38",
        "Classical unseen-instrument domain performance and support",
        caption,
        "S",
        "RQ-P01",
        "held station/instrument domain with physical masters",
        _common(
            frame,
            figure_id="F38",
            scope="S",
            research_question_id="RQ-P01",
            independent_unit="held station/instrument domain with physical masters",
        ),
        panels,
    )


def _t1_t3_figure(diagnostics: P03DiagnosticTables) -> P03FigureDefinition:
    comparison = diagnostics.t1_t3_comparison
    rows: list[dict[str, Any]] = []
    for row in comparison.itertuples(index=False):
        series = f"{row.procedure_id} / {row.aggregation_level}"
        trace = f"{row.procedure_id}|{row.aggregation_level}|r{row.outer_repeat}"
        for regime, value in (
            ("Within station", row.within_station_balanced_accuracy),
            ("Unseen instrument", row.zero_shot_balanced_accuracy),
        ):
            rows.append(
                {
                    "panel_id": row.station,
                    "x_value": regime,
                    "y_value": value,
                    "y_lower": np.nan,
                    "y_upper": np.nan,
                    "series": series,
                    "trace_group": trace,
                    "point_status": "complete",
                    "point_label": row.procedure_id,
                    "denominator": row.zero_shot_domain_count,
                    "outer_repeat": row.outer_repeat,
                    "zero_shot_minus_within_station": row.zero_shot_minus_within_station,
                }
            )
    if not rows:
        coverage = diagnostics.endpoint_coverage[
            diagnostics.endpoint_coverage.experiment_id.astype(str).eq("EXP-C10-T3")
        ]
        for row in coverage.itertuples(index=False):
            for regime in ("Within station", "Unseen instrument"):
                rows.append(
                    {
                        "panel_id": row.station,
                        "x_value": regime,
                        "y_value": -0.05,
                        "y_lower": np.nan,
                        "y_upper": np.nan,
                        "series": f"{row.procedure_id} / unavailable",
                        "trace_group": f"{row.procedure_id}|terminal",
                        "point_status": "unavailable",
                        "point_label": row.procedure_id,
                        "denominator": row.planned_endpoint_count,
                        "outer_repeat": np.nan,
                        "zero_shot_minus_within_station": np.nan,
                    }
                )
    frame = pd.DataFrame(rows)
    caption = (
        "(S) RQ-P01 descriptive acquisition-shift contrast for the four fixed families present in "
        "both C01–C08 and C10. Lines connect the same procedure, aggregation level, station, and "
        "technical split repeat; the unseen-instrument value is an unweighted mean over eligible "
        "domains. This is not a causal or paired-master effect because the information regimes and "
        "test roles differ. The line at 1/3 is three-class chance. PP-U-MIN, source-only fitting."
    )
    panels = tuple(
        P03Panel(
            station,
            station.capitalize(),
            "Evaluation regime",
            "Balanced accuracy",
            mode="line",
            chance_line=1 / 3,
            y_min=0,
            y_max=1.02,
        )
        for station in ("cwa", "pills", "surfaces")
    )
    return P03FigureDefinition(
        "F39",
        "Within-station versus unseen-instrument classical performance",
        caption,
        "S",
        "RQ-P01",
        "held domain; split repeat descriptive",
        _common(
            frame,
            figure_id="F39",
            scope="S",
            research_question_id="RQ-P01",
            independent_unit="held domain; split repeat descriptive",
        ),
        panels,
    )


def _spectrum_master_figure(diagnostics: P03DiagnosticTables) -> P03FigureDefinition:
    comparison = diagnostics.spectrum_master_comparison.copy()
    frame = pd.DataFrame(
        {
            "panel_id": "paired",
            "x_value": comparison.spectrum_balanced_accuracy,
            "y_value": comparison.master_balanced_accuracy,
            "y_lower": np.nan,
            "y_upper": np.nan,
            "series": comparison.station,
            "trace_group": comparison.domain,
            "point_status": comparison.pair_status,
            "point_label": comparison.domain,
            "denominator": 1,
            "outer_repeat": comparison.outer_repeat,
            "master_minus_spectrum": comparison.master_minus_spectrum,
        }
    )
    caption = (
        "(S) RQ-P01 paired aggregation diagnostic for C09. Each point is one held-domain × split-"
        "repeat endpoint; x is spectrum balanced accuracy and y is instrument-balanced physical-"
        "master balanced accuracy from identical outer predictions. The diagonal is equality. "
        "Unavailable pairs remain in the frozen table but have no fabricated coordinates. Repeats "
        "are technical and domains, not spectra, define the comparison units."
    )
    return P03FigureDefinition(
        "F40",
        "Spectrum versus instrument-balanced master performance",
        caption,
        "S",
        "RQ-P01",
        "held domain and physical master",
        _common(
            frame,
            figure_id="F40",
            scope="S",
            research_question_id="RQ-P01",
            independent_unit="held domain and physical master",
        ),
        (
            P03Panel(
                "paired",
                "C09 aggregation comparison",
                "Spectrum balanced accuracy",
                "Master balanced accuracy",
                identity_line=True,
                y_min=0,
                y_max=1.02,
            ),
        ),
    )


def _confusion_figure(diagnostics: P03DiagnosticTables) -> P03FigureDefinition:
    confusion = diagnostics.confusion.copy()
    if confusion.empty:
        confusion = pd.DataFrame(
            [
                {
                    "station": station,
                    "aggregation_level": level,
                    "predicted_index": 0,
                    "true_index": 0,
                    "mean_row_fraction": np.nan,
                    "minimum_row_fraction": np.nan,
                    "maximum_row_fraction": np.nan,
                    "true_label": "NA",
                    "predicted_label": "NA",
                    "mean_true_support_per_repeat": 0,
                    "repeat_count": 0,
                }
                for station in ("cwa", "pills", "surfaces")
                for level in ("spectrum", "instrument_balanced_master")
            ]
        )
    confusion["panel_id"] = (
        confusion.station.astype(str) + "|" + confusion.aggregation_level.astype(str)
    )
    frame = pd.DataFrame(
        {
            "panel_id": confusion.panel_id,
            "x_value": confusion.predicted_index,
            "y_value": confusion.true_index,
            "z_value": confusion.mean_row_fraction,
            "y_lower": confusion.minimum_row_fraction,
            "y_upper": confusion.maximum_row_fraction,
            "series": confusion.aggregation_level,
            "trace_group": confusion.station,
            "point_status": "complete",
            "point_label": (
                confusion.true_label.astype(str)
                + " → "
                + confusion.predicted_label.astype(str)
            ),
            "denominator": confusion.mean_true_support_per_repeat,
            "x_tick_label": confusion.predicted_label,
            "y_tick_label": confusion.true_label,
            "repeat_count": confusion.repeat_count,
        }
    )
    caption = (
        "(S) RQ-P01 C09 confusion structure. Cells are row-normalized within each technical repeat "
        "after pooling eligible held domains, then averaged across five repeats; frozen tables "
        "retain the repeat minimum and maximum. Spectrum support is observations; master support "
        "is domain-master membership after instrument-balanced aggregation. Colour uses a "
        "perceptually uniform "
        "scale. This diagnostic does not treat spectra or repeats as independent evidence."
    )
    panels = tuple(
        P03Panel(
            f"{station}|{level}",
            f"{station.capitalize()} / {'spectrum' if level == 'spectrum' else 'master'}",
            "Predicted class",
            "True class",
            mode="heatmap",
            y_min=0,
            y_max=1,
        )
        for station in ("cwa", "pills", "surfaces")
        for level in ("spectrum", "instrument_balanced_master")
    )
    return P03FigureDefinition(
        "F41",
        "Classical unseen-instrument confusion matrices",
        caption,
        "S",
        "RQ-P01",
        "class within held domain and physical master",
        _common(
            frame,
            figure_id="F41",
            scope="S",
            research_question_id="RQ-P01",
            independent_unit="class within held domain and physical master",
        ),
        panels,
    )


def _calibration_figure(diagnostics: P03DiagnosticTables) -> P03FigureDefinition:
    reliability = diagnostics.reliability.copy()
    if reliability.empty:
        reliability = pd.DataFrame(
            [
                {
                    "aggregation_level": level,
                    "mean_confidence": np.nan,
                    "mean_empirical_accuracy": np.nan,
                    "minimum_empirical_accuracy": np.nan,
                    "maximum_empirical_accuracy": np.nan,
                    "station": "unavailable",
                    "bin_index": -1,
                    "mean_observations_per_repeat": 0,
                    "bin_lower": np.nan,
                    "bin_upper": np.nan,
                    "repeat_count": 0,
                }
                for level in ("spectrum", "instrument_balanced_master")
            ]
        )
    frame = pd.DataFrame(
        {
            "panel_id": reliability.aggregation_level,
            "x_value": reliability.mean_confidence,
            "y_value": reliability.mean_empirical_accuracy,
            "y_lower": reliability.minimum_empirical_accuracy,
            "y_upper": reliability.maximum_empirical_accuracy,
            "series": reliability.station,
            "trace_group": reliability.station,
            "point_status": "complete",
            "point_label": reliability.bin_index.map(lambda value: f"bin {value}"),
            "denominator": reliability.mean_observations_per_repeat,
            "bin_lower": reliability.bin_lower,
            "bin_upper": reliability.bin_upper,
            "repeat_count": reliability.repeat_count,
        }
    )
    caption = (
        "(S) RQ-P01 C09 reliability after scalar temperature calibration fitted only to source-"
        "development cross-fitted scores. Ten fixed-width confidence bins are computed per "
        "technical repeat, then bin confidence and accuracy are averaged across repeats; vertical "
        "intervals are "
        "the observed repeat range, not confidence intervals. The diagonal is perfect calibration. "
        "Only endpoints with valid calibrated probabilities enter; omissions remain in endpoint "
        "coverage. PP-U-MIN and no held-target calibration statistics."
    )
    return P03FigureDefinition(
        "F42",
        "Classical unseen-instrument calibration reliability",
        caption,
        "S",
        "RQ-P01",
        "physical master; split repeat descriptive",
        _common(
            frame,
            figure_id="F42",
            scope="S",
            research_question_id="RQ-P01",
            independent_unit="physical master; split repeat descriptive",
        ),
        (
            P03Panel(
                "spectrum",
                "Spectrum",
                "Mean confidence",
                "Empirical accuracy",
                mode="line",
                identity_line=True,
                y_min=0,
                y_max=1.02,
            ),
            P03Panel(
                "instrument_balanced_master",
                "Instrument-balanced master",
                "Mean confidence",
                "Empirical accuracy",
                mode="line",
                identity_line=True,
                y_min=0,
                y_max=1.02,
            ),
        ),
    )


def _control_figure(diagnostics: P03DiagnosticTables) -> P03FigureDefinition:
    controls = diagnostics.control_summary.copy()
    controls["plot_value"] = controls.mean_domain_balanced_accuracy.where(
        controls.summary_status.astype(str).eq("complete"), -0.05
    )
    frame = pd.DataFrame(
        {
            "panel_id": controls.aggregation_level,
            "x_value": controls.control_kind,
            "y_value": controls.plot_value,
            "y_lower": controls.minimum_repeat_balanced_accuracy,
            "y_upper": controls.maximum_repeat_balanced_accuracy,
            "series": controls.control_kind,
            "trace_group": controls.procedure_id,
            "point_status": controls.summary_status,
            "point_label": controls.procedure_id,
            "denominator": controls.expected_domain_count,
            "repeat_count": controls.repeat_count,
            "worst_domain_balanced_accuracy": controls.worst_domain_balanced_accuracy,
        }
    )
    caption = (
        "(S) RQ-P01 leakage, chance, and acquisition-confounding controls. Points are procedure-"
        "level means over the 13 domain-unweighted endpoints and five technical repeats: real "
        "spectra, 20 "
        "master-label permutations with frozen real-label selections, acquisition-metadata-only "
        "elastic-net, and empirical/uniform source priors. Repeat ranges are descriptive. The line "
        "is three-class chance. Controls select no model, preprocessing, threshold, or claim."
    )
    return P03FigureDefinition(
        "F43",
        "Classical negative controls and acquisition confounding",
        caption,
        "S",
        "RQ-P01",
        "held domain; permutation at physical-master level",
        _common(
            frame,
            figure_id="F43",
            scope="S",
            research_question_id="RQ-P01",
            independent_unit="held domain; permutation at physical-master level",
        ),
        (
            P03Panel(
                "spectrum",
                "Spectrum",
                "Control procedure",
                "Mean domain balanced accuracy",
                chance_line=1 / 3,
                y_min=-0.08,
                y_max=1.02,
            ),
            P03Panel(
                "instrument_balanced_master",
                "Instrument-balanced master",
                "Control procedure",
                "Mean domain balanced accuracy",
                chance_line=1 / 3,
                y_min=-0.08,
                y_max=1.02,
            ),
        ),
    )


def build_p03_figure_definitions(
    *, pooled_metrics: pd.DataFrame, diagnostics: P03DiagnosticTables
) -> dict[str, P03FigureDefinition]:
    """Freeze all P03 figure tables before rendering any output form."""

    definitions = [
        _selection_figure(diagnostics),
        _t1_figure(pooled_metrics),
        _domain_figure(pooled_metrics),
        _t1_t3_figure(diagnostics),
        _spectrum_master_figure(diagnostics),
        _confusion_figure(diagnostics),
        _calibration_figure(diagnostics),
        _control_figure(diagnostics),
    ]
    return {definition.figure_id: definition for definition in definitions}


def _panel_frame(definition: P03FigureDefinition, panel: P03Panel) -> pd.DataFrame:
    return definition.frame[
        definition.frame.panel_id.astype(str).eq(panel.panel_id)
    ].copy()


def _hover_text(frame: pd.DataFrame) -> list[str]:
    fields = [
        field
        for field in (
            "point_label",
            "point_status",
            "denominator",
            "outer_repeat",
            "repeat_count",
            "bin_lower",
            "bin_upper",
        )
        if field in frame
    ]
    return [
        "<br>".join(f"{field}: {row[field]}" for field in fields)
        for _, row in frame.iterrows()
    ]


def _series_styles(frame: pd.DataFrame) -> dict[str, tuple[str, str, str, str]]:
    series = sorted(frame.series.fillna("<missing>").astype(str).unique())
    styles: dict[str, tuple[str, str, str, str]] = {}
    for index, name in enumerate(series):
        if "unavailable" in name.lower() or "incomplete" in name.lower():
            styles[name] = ("#F0E442", "circle-open", "dot", "o")
        else:
            styles[name] = (
                COLORS[index % (len(COLORS) - 1)],
                PLOTLY_MARKERS[index % len(PLOTLY_MARKERS)],
                PLOTLY_DASHES[index % len(PLOTLY_DASHES)],
                TIKZ_MARKERS[index % len(TIKZ_MARKERS)],
            )
    return styles


def _plotly_figure(definition: P03FigureDefinition) -> go.Figure:
    columns = min(3, len(definition.panels))
    rows = int(np.ceil(len(definition.panels) / columns))
    figure = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=[panel.title for panel in definition.panels],
        horizontal_spacing=0.09,
        vertical_spacing=0.16,
    )
    styles = _series_styles(definition.frame)
    legend_seen: set[str] = set()
    for index, panel in enumerate(definition.panels):
        row = index // columns + 1
        column = index % columns + 1
        data = _panel_frame(definition, panel)
        if panel.mode == "heatmap":
            if not data.empty:
                x_ticks = (
                    data[["x_value", "x_tick_label"]]
                    .drop_duplicates()
                    .sort_values("x_value")
                )
                y_ticks = (
                    data[["y_value", "y_tick_label"]]
                    .drop_duplicates()
                    .sort_values("y_value")
                )
                pivot = data.pivot(index="y_value", columns="x_value", values="z_value")
                figure.add_trace(
                    go.Heatmap(
                        x=x_ticks.x_tick_label,
                        y=y_ticks.y_tick_label,
                        z=pivot.reindex(
                            index=y_ticks.y_value, columns=x_ticks.x_value
                        ).to_numpy(),
                        zmin=0,
                        zmax=1,
                        colorscale="Viridis",
                        colorbar={"title": "Row fraction"},
                        text=pivot.reindex(
                            index=y_ticks.y_value, columns=x_ticks.x_value
                        ).to_numpy(),
                        texttemplate="%{text:.2f}",
                        hovertemplate="Predicted=%{x}<br>True=%{y}<br>Fraction=%{z:.3f}<extra></extra>",
                        showscale=index == 0,
                    ),
                    row=row,
                    col=column,
                )
            figure.update_xaxes(title_text=panel.x_label, row=row, col=column)
            figure.update_yaxes(title_text=panel.y_label, row=row, col=column)
            continue
        grouping = ["series", "trace_group"] if "trace_group" in data else ["series"]
        for key, group in data.groupby(grouping, sort=True, dropna=False):
            name = str(key[0] if isinstance(key, tuple) else key)
            color, marker, dash, _ = styles[name]
            ordered = group.sort_values("x_value", kind="stable") if panel.mode == "line" else group
            error_y = None
            if ordered.y_lower.notna().any() and ordered.y_upper.notna().any():
                error_y = {
                    "type": "data",
                    "symmetric": False,
                    "array": (ordered.y_upper - ordered.y_value).clip(lower=0),
                    "arrayminus": (ordered.y_value - ordered.y_lower).clip(lower=0),
                    "visible": True,
                    "thickness": 1,
                }
            figure.add_trace(
                go.Scatter(
                    x=ordered.x_value,
                    y=ordered.y_value,
                    mode="lines+markers" if panel.mode == "line" else "markers",
                    name=name,
                    legendgroup=name,
                    showlegend=name not in legend_seen,
                    line={"color": color, "dash": dash, "width": 1.5},
                    marker={
                        "color": color,
                        "symbol": marker,
                        "size": 8,
                        "line": {"color": "black", "width": 1}
                        if "open" in marker
                        else {"width": 0},
                    },
                    error_y=error_y,
                    hovertext=_hover_text(ordered),
                    hovertemplate=(
                        "%{hovertext}<br>x=%{x}<br>y=%{y:.4f}<extra>"
                        + html.escape(name)
                        + "</extra>"
                    ),
                ),
                row=row,
                col=column,
            )
            legend_seen.add(name)
        if panel.identity_line:
            figure.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="identity",
                    legendgroup="reference",
                    showlegend="identity" not in legend_seen,
                    line={"color": "black", "dash": "dashdot", "width": 1},
                    hoverinfo="skip",
                ),
                row=row,
                col=column,
            )
            legend_seen.add("identity")
        if panel.chance_line is not None:
            figure.add_hline(
                y=panel.chance_line,
                line={"color": "black", "dash": "dashdot", "width": 1},
                row=row,
                col=column,
            )
        figure.update_xaxes(title_text=panel.x_label, row=row, col=column)
        figure.update_yaxes(
            title_text=panel.y_label,
            range=(
                [panel.y_min, panel.y_max]
                if panel.y_min is not None and panel.y_max is not None
                else None
            ),
            row=row,
            col=column,
        )
    figure.update_layout(
        title=definition.title,
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 12},
        width=max(1050, columns * 470),
        height=max(580, rows * 470),
        margin={"l": 70, "r": 40, "t": 100, "b": 110},
        hovermode="closest",
    )
    return figure


def _category_coordinates(values: pd.Series) -> tuple[pd.Series, list[str]]:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float), []
    labels = list(dict.fromkeys(values.astype(str)))
    mapping = {label: index for index, label in enumerate(labels)}
    return values.astype(str).map(mapping).astype(float), labels


def _tikz_source(definition: P03FigureDefinition, data_hash: str) -> str:
    columns = min(3, len(definition.panels))
    rows = int(np.ceil(len(definition.panels) / columns))
    panel_width = 5.3 if columns == 3 else 7.3 if columns == 2 else 12.0
    styles = _series_styles(definition.frame)
    lines = [
        r"\documentclass[tikz,border=5pt]{standalone}",
        r"\pdfinfoomitdate=1",
        r"\pdftrailerid{}",
        r"\pdfsuppressptexinfo=-1",
        r"\usepackage{pgfplots}",
        r"\usepgfplotslibrary{groupplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\pgfplotsset{colormap/viridis}",
        *[
            f"\\definecolor{{atlas{index}}}{{HTML}}{{{color.lstrip('#')}}}"
            for index, color in enumerate(COLORS)
        ],
        f"% ATLAS figure {definition.figure_id}; data_sha256={data_hash}",
        f"% caption={definition.caption}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        (
            r"\begin{groupplot}[group style={group size="
            f"{columns} by {rows}, horizontal sep=1.35cm, vertical sep=1.5cm"
            r"}, width="
            f"{panel_width:.2f}cm, height=5.2cm, grid=major, "
            r"tick label style={font=\tiny}, label style={font=\scriptsize}, "
            r"title style={font=\scriptsize\bfseries,align=center}, "
            r"legend style={font=\tiny,draw=none,fill=white,fill opacity=0.9,text opacity=1}]"
        ),
    ]
    for panel in definition.panels:
        data = _panel_frame(definition, panel)
        options = [
            f"title={{{_tex(panel.title)}}}",
            f"xlabel={{{_tex(panel.x_label)}}}",
            f"ylabel={{{_tex(panel.y_label)}}}",
        ]
        if panel.y_min is not None:
            options.append(f"ymin={panel.y_min:.6g}")
        if panel.y_max is not None:
            options.append(f"ymax={panel.y_max:.6g}")
        if panel.mode == "heatmap":
            x_ticks = (
                data[["x_value", "x_tick_label"]]
                .drop_duplicates()
                .sort_values("x_value")
            )
            y_ticks = (
                data[["y_value", "y_tick_label"]]
                .drop_duplicates()
                .sort_values("y_value")
            )
            options.extend(
                [
                    "xtick={" + ",".join(str(value) for value in x_ticks.x_value) + "}",
                    "xticklabels={" + ",".join(_tex(value) for value in x_ticks.x_tick_label) + "}",
                    "ytick={" + ",".join(str(value) for value in y_ticks.y_value) + "}",
                    "yticklabels={" + ",".join(_tex(value) for value in y_ticks.y_tick_label) + "}",
                    "point meta min=0",
                    "point meta max=1",
                    "colorbar",
                ]
            )
            lines.append(r"\nextgroupplot[" + ",".join(options) + "]")
            finite_heatmap = data[pd.to_numeric(data.z_value, errors="coerce").notna()]
            coordinates = " ".join(
                f"({float(row.x_value):.8g},{float(row.y_value):.8g}) [{float(row.z_value):.8g}]"
                for row in finite_heatmap.itertuples(index=False)
            )
            if coordinates:
                lines.append(
                    r"\addplot[matrix plot*,mesh/cols="
                    f"{max(1, finite_heatmap.x_value.nunique())},point meta=explicit] "
                    f"coordinates {{{coordinates}}};"
                )
            else:
                lines.append(r"\node[font=\scriptsize] at (axis cs:0,0) {NA};")
            continue
        x_values, labels = _category_coordinates(data.x_value)
        data = data.assign(plot_x=x_values)
        finite_x = data.plot_x[np.isfinite(data.plot_x)]
        x_min = float(finite_x.min()) if not finite_x.empty else 0.0
        x_max = float(finite_x.max()) if not finite_x.empty else 1.0
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        if labels:
            options.extend(
                [
                    "xtick={" + ",".join(str(index) for index in range(len(labels))) + "}",
                    "xticklabels={" + ",".join(_tex(label) for label in labels) + "}",
                    "x tick label style={rotate=42,anchor=east,font=\tiny}",
                ]
            )
        lines.append(r"\nextgroupplot[" + ",".join(options) + "]")
        legend_seen: set[str] = set()
        grouping = ["series", "trace_group"] if "trace_group" in data else ["series"]
        for key, group in data.groupby(grouping, sort=True, dropna=False):
            name = str(key[0] if isinstance(key, tuple) else key)
            color_hex, _, _, marker = styles[name]
            color_index = COLORS.index(color_hex)
            style_index = sorted(styles).index(name)
            style = [
                f"color=atlas{color_index}",
                f"mark={marker}",
                "mark size=1.8pt",
                "line width=0.8pt",
            ]
            if panel.mode == "scatter":
                style.append("only marks")
            else:
                style.append(TIKZ_DASHES[style_index % len(TIKZ_DASHES)])
            if "unavailable" in name.lower() or "incomplete" in name.lower():
                style.extend(["mark=o", "mark options={draw=black,fill=atlas7}"])
            ordered = group.sort_values("plot_x", kind="stable")
            has_error = ordered.y_lower.notna().any() and ordered.y_upper.notna().any()
            if has_error:
                style.append("error bars/.cd,y dir=both,y explicit")
                coordinates = " ".join(
                    (
                        f"({float(row.plot_x):.8g},{float(row.y_value):.8g}) "
                        f"+= (0,{max(0.0, float(row.y_upper - row.y_value)):.8g}) "
                        f"-= (0,{max(0.0, float(row.y_value - row.y_lower)):.8g})"
                    )
                    for row in ordered.itertuples(index=False)
                    if pd.notna(row.y_value)
                    and pd.notna(row.y_lower)
                    and pd.notna(row.y_upper)
                )
            else:
                coordinates = " ".join(
                    f"({float(row.plot_x):.8g},{float(row.y_value):.8g})"
                    for row in ordered.itertuples(index=False)
                    if pd.notna(row.y_value)
                )
            lines.append(f"\\addplot+[{','.join(style)}] coordinates {{{coordinates}}};")
            if name not in legend_seen:
                lines.append(f"\\addlegendentry{{{_tex(name)}}}")
                legend_seen.add(name)
        if panel.identity_line:
            lines.append(
                r"\addplot[black,dashdotted,line width=0.7pt,forget plot] "
                r"coordinates {(0,0) (1,1)};"
            )
        if panel.chance_line is not None:
            lines.append(
                r"\addplot[black,dashdotted,line width=0.7pt,forget plot] coordinates "
                f"{{({x_min:.8g},{panel.chance_line:.8g}) "
                f"({x_max:.8g},{panel.chance_line:.8g})}};"
            )
    lines.extend(
        [
            r"\end{groupplot}",
            (
                r"\node[font=\normalsize\bfseries,anchor=south] at "
                f"(current bounding box.north) {{{_tex(definition.title)}}};"
            ),
            r"\end{tikzpicture}",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def _compile(tex_path: Path, pdf_dir: Path, log_dir: Path) -> tuple[Path, Path]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["SOURCE_DATE_EPOCH"] = "1785888000"
    result = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={pdf_dir}",
            tex_path.name,
        ],
        cwd=tex_path.parent,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    source_log = pdf_dir / f"{tex_path.stem}.log"
    log_path = log_dir / f"{tex_path.stem}.pdflatex.log"
    log_text = source_log.read_text(errors="replace") if source_log.is_file() else ""
    log_path.write_text(
        "\n".join(
            [
                "schema_version=atlas-p03-figure-compilation-v1",
                "engine=pdflatex",
                f"source={tex_path.name}",
                f"return_code={result.returncode}",
                result.stdout,
                result.stderr,
                log_text,
            ]
        )
    )
    source_log.unlink(missing_ok=True)
    (pdf_dir / f"{tex_path.stem}.aux").unlink(missing_ok=True)
    pdf_path = pdf_dir / f"{tex_path.stem}.pdf"
    if result.returncode != 0 or not pdf_path.is_file():
        raise RuntimeError(f"TikZ compilation failed for {tex_path.name}.")
    return pdf_path, log_path


def _render_png(pdf_path: Path, png_path: Path) -> None:
    result = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "300",
            "-singlefile",
            str(pdf_path),
            str(png_path.with_suffix("")),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not png_path.is_file():
        raise RuntimeError(f"PNG rendering failed for {pdf_path.name}.")


def generate_p03_figures(
    output_root: Path, definitions: dict[str, P03FigureDefinition]
) -> pd.DataFrame:
    """Render hashed plot tables as TikZ/PDF/PNG and standalone Plotly HTML."""

    directories = {
        name: output_root / "figures" / name
        for name in ("data", "tikz", "pdf", "png", "html", "logs")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for figure_id, definition in sorted(definitions.items()):
        if figure_id not in FIGURE_SLUGS or definition.frame.empty:
            raise ValueError(f"P03 figure {figure_id} has no registered nonempty plot table.")
        stem = f"{figure_id}_{FIGURE_SLUGS[figure_id]}"
        data_path = directories["data"] / f"{stem}.csv"
        definition.frame.to_csv(
            data_path,
            index=False,
            lineterminator="\n",
            float_format="%.12g",
        )
        data_hash = sha256_file(data_path)
        html_path = directories["html"] / f"{stem}.html"
        document = _plotly_figure(definition).to_html(
            include_plotlyjs=True,
            full_html=True,
            div_id=f"atlas-{figure_id.lower()}",
            config={"responsive": True, "scrollZoom": True, "displaylogo": False},
        )
        marker = (
            f"<!-- data_sha256={data_hash}; ATLAS {figure_id}; "
            f"description={html.escape(definition.title)} -->\n"
        )
        caption = (
            '<section style="max-width:1200px;margin:1rem auto;font-family:Arial,sans-serif">'
            f"<h1>{html.escape(definition.title)}</h1>"
            f"<p><strong>{html.escape(definition.scope)} / "
            f"{html.escape(definition.research_question_id)}</strong></p>"
            f"<p>{html.escape(definition.caption)}</p>"
            f"<p><strong>Independent unit:</strong> {html.escape(definition.independent_unit)}; "
            "<strong>population:</strong> primary_598; <strong>policy:</strong> PP-U-MIN; "
            "<strong>target access:</strong> none for fitting/selection/calibration.</p>"
            f"<p><strong>Frozen plot-data SHA-256:</strong> {data_hash}</p></section>"
        )
        document = document.replace("<head>", f"<head>\n{marker}", 1)
        document = document.replace("<body>", f"<body>\n{caption}", 1)
        document = document.replace("cdn.plot.ly", "local.plotly.invalid")
        html_path.write_text("\n".join(line.rstrip() for line in document.splitlines()) + "\n")
        tex_path = directories["tikz"] / f"{stem}.tex"
        tex_path.write_text(_tikz_source(definition, data_hash))
        pdf_path, log_path = _compile(tex_path, directories["pdf"], directories["logs"])
        png_path = directories["png"] / f"{stem}.png"
        _render_png(pdf_path, png_path)
        pdfimages = subprocess.run(
            ["pdfimages", "-list", str(pdf_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        raster_free = pdfimages.returncode == 0 and len(pdfimages.stdout.splitlines()) <= 2
        rows.append(
            {
                "figure_id": figure_id,
                "title": definition.title,
                "caption": definition.caption,
                "data_path": data_path.relative_to(output_root).as_posix(),
                "tikz_path": tex_path.relative_to(output_root).as_posix(),
                "pdf_path": pdf_path.relative_to(output_root).as_posix(),
                "png_path": png_path.relative_to(output_root).as_posix(),
                "html_path": html_path.relative_to(output_root).as_posix(),
                "log_path": log_path.relative_to(output_root).as_posix(),
                "data_sha256": data_hash,
                "tikz_sha256": sha256_file(tex_path),
                "pdf_sha256": sha256_file(pdf_path),
                "png_sha256": sha256_file(png_path),
                "html_sha256": sha256_file(html_path),
                "semantic_parity": data_hash in tex_path.read_text()
                and data_hash in html_path.read_text(),
                "compiled": pdf_path.stat().st_size > 0,
                "native_tikz": "\\includegraphics" not in tex_path.read_text(),
                "standalone_html": "</html>" in html_path.read_text()[-2000:].lower(),
                "colorblind_safe": True,
                "vector_only": raster_free,
                "png_dpi": 300,
            }
        )
    return pd.DataFrame(rows).sort_values("figure_id", kind="stable").reset_index(drop=True)
