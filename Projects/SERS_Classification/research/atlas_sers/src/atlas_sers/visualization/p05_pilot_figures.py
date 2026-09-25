# ruff: noqa: E501
"""P05 aggregate pilot figures.

Builds one allowlisted semantic table (36 source-validation development fits)
and emits, per figure, native pgfplots/TikZ source, a standalone offline Plotly
HTML file, a vector PDF and a PNG through the unchanged P04 helpers.  Every
semantic row is a single fit-epoch of a source-validation-monitored development
fit; trajectories are kept as-is and never extended, imputed or averaged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from atlas_sers.governance.canonical import sha256_file
from atlas_sers.visualization.p04_figures import _compile, _tex, _write_html
from atlas_sers.visualization.p05_smoke_figures import (
    MANIFEST_NAME,
    RECIPE_COLORS,
    RECIPE_DESCRIPTIONS,
    RECIPE_IDS,
    SEMANTIC_NAME,
    TEX_COLORS,
    _configure_deterministic_pdf,
    _guard_output_root,
    _manifest_files,
    _require_number,
)

__all__ = ["build_semantic_data", "generate_pilot_figures"]

STATIONS = ("cwa", "pills", "surfaces")
STATION_LABELS = {"cwa": "CWA", "pills": "Pills", "surfaces": "Surfaces"}

SEEDS = (20260805, 20260817, 20260829)
SEED_INDEX = {SEEDS[0]: 1, SEEDS[1]: 2, SEEDS[2]: 3}
SEED_INDEX_NAMES = {1: "seed1", 2: "seed2", 3: "seed3"}
SEED_DASH = {1: "solid", 2: "dash", 3: "dot"}
SEED_SYMBOL = {1: "circle", 2: "square", 3: "triangle-up"}
TEX_DASH = {1: "solid", 2: "dashed", 3: "dotted"}
TEX_MARK = {1: "*", 2: "square*", 3: "triangle*"}

MINIMUM_EPOCHS = 30
MAXIMUM_EPOCHS = 200
BATCHES_PER_EPOCH = 4
EXPECTED_FITS = len(STATIONS) * len(RECIPE_IDS) * len(SEEDS)

SEMANTIC_COLUMNS = (
    "station",
    "recipe_id",
    "seed_index",
    "epoch",
    "is_best",
    "chemical_ce",
    "train_balanced_accuracy",
    "validation_balanced_accuracy",
    "validation_nll",
    "validation_macro_f1",
)

TRAINING_CLAIM = "Source-validation pilot; not held-out test performance or recipe selection."
LEARNING_TITLE = "Source-validation learning curves"
BEST_TITLE = "Best-checkpoint training vs source-validation balanced accuracy"
CAPTION_NOTE = (
    "CE = class/master/view-weighted cross entropy; BA = mean recall across the three classes. "
    "One metadata-preselected source split per station and three initialization seeds. "
    "CWA and Surfaces validate on a source pseudo-instrument; Pills uses source-master CV "
    "because the inherited station contexts lack pseudo-domain support. "
    "Scatter uses each fit's selected checkpoint; validation never updates weights and is not an outer test. "
    "Three seeds show initialization variability, not independent dataset replications. "
    "Individual trajectories and points are shown."
)
LEGEND_RECIPES = (
    r"\textcolor{p05Dzero}{\rule{7mm}{1.2pt}}\,D0-M ordinary classification\quad "
    r"\textcolor{p05Done}{\rule{7mm}{1.2pt}}\,D1 chemical similarity\\[0.8mm]"
    r"\textcolor{p05Dtwo}{\rule{7mm}{1.2pt}}\,D2 matched-sample consistency\quad "
    r"\textcolor{p05Dthree}{\rule{7mm}{1.2pt}}\,D3 both\\[1mm]"
)
LEGEND_SEED_LINES = (
    r"\tikz[baseline=-0.6ex]\draw[black,solid,line width=0.9pt](0,0)--(0.75,0);\ seed1 (solid)\quad "
    r"\tikz[baseline=-0.6ex]\draw[black,dashed,line width=0.9pt](0,0)--(0.75,0);\ seed2 (dashed)\quad "
    r"\tikz[baseline=-0.6ex]\draw[black,dotted,line width=0.9pt](0,0)--(0.75,0);\ seed3 (dotted)"
)
LEGEND_SEED_SYMBOLS = (
    r"seed1 circle\quad seed2 square\quad seed3 triangle (marker symbols)\quad "
    r"dotted black line: identity reference"
)


def _html_caption() -> str:
    return (
        "CE = class/master/view-weighted cross entropy; BA = mean recall across the three "
        "classes.<br>"
        "One metadata-preselected source split per station; three initialization seeds.<br>"
        "CWA and Surfaces validate on a source pseudo-instrument; Pills uses source-master CV.<br>"
        "The inherited Pills station contexts lack pseudo-domain support.<br>"
        "Scatter uses each fit's selected checkpoint; validation never updates weights and "
        "is not an outer test.<br>"
        "Three seeds show initialization variability, not independent dataset "
        "replications. Individual trajectories and points are shown.<br>"
        "seed1 solid/circle; seed2 dash/square; seed3 dot/triangle."
    )


def _require_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"semantic fit field {field!r} must be a nonnegative integer.")
    number = int(value)
    if number < 0:
        raise ValueError(f"semantic fit field {field!r} must be a nonnegative integer.")
    return number


def _require_unit(value: object, field: str) -> float:
    number = _require_number(value, field)
    if number > 1.0:
        raise ValueError(f"semantic fit field {field!r} must lie in [0, 1].")
    return number


def build_semantic_data(records: object) -> pd.DataFrame:
    if isinstance(records, (str, bytes)) or not isinstance(records, (list, tuple)):
        raise TypeError("records must be a list of development fit mappings.")
    fits: dict[tuple[str, str, int], tuple[int, dict[int, dict[str, float]]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("every record must be a mapping.")
        station = record.get("station")
        if station not in STATIONS:
            raise ValueError("record has an unknown station.")
        recipe = record.get("recipe_id")
        if recipe not in RECIPE_IDS:
            raise ValueError("record has an unknown recipe_id.")
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("record has a malformed or unknown seed.")
        seed = int(seed)
        if seed not in SEED_INDEX:
            raise ValueError("record has an unknown seed.")
        if record.get("status") != "complete":
            raise ValueError("record is not a completed fit.")
        key = (station, recipe, SEED_INDEX[seed])
        if key in fits:
            raise ValueError("duplicate station/recipe/seed fit.")
        history = record.get("history")
        if not isinstance(history, (list, tuple)):
            raise ValueError("fit has a malformed history.")
        if not MINIMUM_EPOCHS <= len(history) <= MAXIMUM_EPOCHS:
            raise ValueError(
                f"fit history length must lie in [{MINIMUM_EPOCHS}, {MAXIMUM_EPOCHS}]."
            )
        values: dict[int, dict[str, float]] = {}
        for entry in history:
            if not isinstance(entry, dict):
                raise TypeError("fit history entry is not a mapping.")
            epoch = _require_integer(entry.get("epoch"), "epoch")
            if epoch in values:
                raise ValueError("fit repeats an epoch.")
            values[epoch] = {
                "chemical_ce": _require_number(entry.get("chemical_ce"), "chemical_ce"),
                "train_balanced_accuracy": _require_unit(
                    entry.get("train_balanced_accuracy"), "train_balanced_accuracy"
                ),
                "validation_balanced_accuracy": _require_unit(
                    entry.get("validation_balanced_accuracy"), "validation_balanced_accuracy"
                ),
                "validation_nll": _require_number(entry.get("validation_nll"), "validation_nll"),
                "validation_macro_f1": _require_unit(
                    entry.get("validation_macro_f1"), "validation_macro_f1"
                ),
            }
            epoch_steps = _require_integer(
                entry.get("epoch_optimizer_steps"), "epoch_optimizer_steps"
            )
            total_steps = _require_integer(
                entry.get("total_optimizer_steps"), "total_optimizer_steps"
            )
            if epoch_steps != BATCHES_PER_EPOCH:
                raise ValueError(
                    "fit epoch_optimizer_steps must equal the fixed batches per epoch."
                )
            if total_steps != epoch * BATCHES_PER_EPOCH:
                raise ValueError(
                    "fit total_optimizer_steps must equal epoch times batches per epoch."
                )
        if sorted(values) != list(range(1, len(history) + 1)):
            raise ValueError("fit history epochs must be contiguous starting at one.")
        best_epoch = _require_integer(record.get("best_epoch"), "best_epoch")
        if best_epoch not in values:
            raise ValueError("fit best_epoch must be present in its history.")
        fits[key] = (best_epoch, values)
    expected = {
        (station, recipe, SEED_INDEX[seed])
        for station in STATIONS
        for recipe in RECIPE_IDS
        for seed in SEEDS
    }
    missing = expected - set(fits)
    if missing:
        raise ValueError(f"missing {len(missing)} complete fits.")
    if len(fits) != EXPECTED_FITS:
        raise ValueError(f"semantic table must contain exactly {EXPECTED_FITS} complete fits.")
    rows: list[dict[str, object]] = []
    for station in STATIONS:
        for recipe in RECIPE_IDS:
            for seed in SEEDS:
                seed_index = SEED_INDEX[seed]
                best_epoch, values = fits[(station, recipe, seed_index)]
                for epoch in range(1, len(values) + 1):
                    metrics = values[epoch]
                    rows.append(
                        {
                            "station": station,
                            "recipe_id": recipe,
                            "seed_index": seed_index,
                            "epoch": epoch,
                            "is_best": bool(epoch == best_epoch),
                            "chemical_ce": metrics["chemical_ce"],
                            "train_balanced_accuracy": metrics["train_balanced_accuracy"],
                            "validation_balanced_accuracy": metrics["validation_balanced_accuracy"],
                            "validation_nll": metrics["validation_nll"],
                            "validation_macro_f1": metrics["validation_macro_f1"],
                        }
                    )
    frame = pd.DataFrame(rows, columns=list(SEMANTIC_COLUMNS))
    if int(frame["is_best"].sum()) != EXPECTED_FITS:
        raise ValueError(f"semantic table must flag exactly {EXPECTED_FITS} best epochs.")
    return frame


def _tex_color_definitions() -> list[str]:
    return [
        rf"\definecolor{{{TEX_COLORS[recipe]}}}{{HTML}}{{{RECIPE_COLORS[recipe].lstrip('#')}}}"
        for recipe in RECIPE_IDS
    ]


def _tex_preamble(digest: str, comment: str) -> list[str]:
    lines = [
        f"% {comment}",
        f"% data_sha256={digest}",
        r"\documentclass[tikz,border=8pt]{standalone}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\usepgfplotslibrary{groupplots}",
        r"\ifdefined\pdfinfoomitdate\pdfinfoomitdate=1\fi",
        r"\ifdefined\pdfsuppressptexinfo\pdfsuppressptexinfo=-1\fi",
        r"\ifdefined\pdftrailerid\pdftrailerid{}\fi",
    ]
    lines.extend(_tex_color_definitions())
    lines.extend([r"\begin{document}", r"\begin{tikzpicture}"])
    return lines


def _tex_title_node(title: str, claim: str) -> str:
    return (
        r"\node[anchor=south, font=\bfseries\small, color=black, align=center, text width=15cm] at (current bounding box.north) {"
        + _tex(title)
        + r"\\[1mm]"
        + _tex(claim)
        + r"};"
    )


def _tex_caption_node(caption: str) -> str:
    return (
        r"\node[anchor=north, font=\small, color=black, align=left, text width=15cm] (p05caption) at ([yshift=-2mm]current bounding box.south) {"
        + _tex(caption)
        + r"};"
    )


def _tex_legend_node(body: str) -> str:
    return (
        r"\node[anchor=north west, font=\small, color=black, align=left, text width=15cm] at ([yshift=-1mm]p05caption.south west) {"
        + body
        + r"};"
    )


def _trajectory_plots(subset: pd.DataFrame, column: str) -> list[str]:
    lines: list[str] = []
    for recipe in RECIPE_IDS:
        for seed_index in sorted(SEED_INDEX_NAMES):
            points = subset[
                (subset["recipe_id"] == recipe) & (subset["seed_index"] == seed_index)
            ].sort_values("epoch")
            coordinates = " ".join(
                f"({int(epoch)},{float(value):.17g})"
                for epoch, value in zip(
                    points["epoch"].tolist(), points[column].tolist(), strict=True
                )
            )
            lines.append(
                rf"\addplot[color={TEX_COLORS[recipe]}, {TEX_DASH[seed_index]}, mark=*, mark size=1.0pt] coordinates {{{coordinates}}};"
            )
    return lines


def _learning_curves_figure(frame: pd.DataFrame) -> go.Figure:
    columns = (
        ("chemical_ce", "Training CE"),
        ("validation_balanced_accuracy", "Validation BA"),
    )
    figure = make_subplots(
        rows=len(STATIONS),
        cols=2,
        shared_xaxes="columns",
        vertical_spacing=0.10,
        horizontal_spacing=0.12,
        subplot_titles=[
            f"{STATION_LABELS[station]}: {label}" for station in STATIONS for _, label in columns
        ],
    )
    for row_index, station in enumerate(STATIONS, start=1):
        subset = frame[frame["station"] == station]
        for recipe in RECIPE_IDS:
            for seed_index in sorted(SEED_INDEX_NAMES):
                points = subset[
                    (subset["recipe_id"] == recipe) & (subset["seed_index"] == seed_index)
                ].sort_values("epoch")
                for col, (column, _label) in enumerate(columns, start=1):
                    figure.add_trace(
                        go.Scatter(
                            x=points["epoch"].tolist(),
                            y=points[column].tolist(),
                            mode="lines+markers",
                            name=f"{RECIPE_DESCRIPTIONS[recipe]} [{SEED_INDEX_NAMES[seed_index]}]",
                            legendgroup=recipe,
                            line={
                                "color": RECIPE_COLORS[recipe],
                                "dash": SEED_DASH[seed_index],
                                "width": 1.6,
                            },
                            marker={
                                "color": RECIPE_COLORS[recipe],
                                "size": 4,
                                "line": {"color": "black", "width": 0.4},
                            },
                            showlegend=(row_index == 1 and col == 1),
                            hovertemplate=(
                                f"{STATION_LABELS[station]}<br>{RECIPE_DESCRIPTIONS[recipe]}"
                                f" [{SEED_INDEX_NAMES[seed_index]}]<br>epoch %{{x}}<br>"
                                "%{y:.4f}<extra></extra>"
                            ),
                        ),
                        row=row_index,
                        col=col,
                    )
        for col in (1, 2):
            figure.update_xaxes(
                title_text="epoch" if row_index == len(STATIONS) else None,
                row=row_index,
                col=col,
            )
        figure.update_yaxes(
            title_text=columns[0][1], rangemode="tozero", row=row_index, col=1
        )
        figure.update_yaxes(
            title_text=columns[1][1], range=[0.0, 1.0], row=row_index, col=2
        )
    figure.update_layout(
        title={"text": LEARNING_TITLE + "<br><sup>" + TRAINING_CLAIM + "</sup>", "x": 0.5},
        template="simple_white",
        font={"color": "black", "size": 11},
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=1080,
        width=1180,
        legend={
            "orientation": "v",
            "x": 1.02,
            "y": 1.0,
            "font": {"size": 9, "color": "black"},
            "title": {
                "text": "Recipe / initialization",
                "font": {"size": 9},
            },
        },
        margin={"l": 70, "r": 215, "t": 120, "b": 175},
    )
    figure.add_annotation(
        x=0.5,
        y=-0.11,
        xref="paper",
        yref="paper",
        xanchor="center",
        yanchor="top",
        showarrow=False,
        align="left",
        font={"size": 10, "color": "black"},
        text=_html_caption(),
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
    figure.update_yaxes(showgrid=True, gridcolor="#e8e8e8")
    return figure


def _best_checkpoints_figure(frame: pd.DataFrame) -> go.Figure:
    best = frame[frame["is_best"]].copy()
    if len(best) != EXPECTED_FITS:
        raise ValueError(f"best-checkpoint table must contain {EXPECTED_FITS} rows.")
    figure = make_subplots(
        rows=1,
        cols=len(STATIONS),
        shared_yaxes=True,
        subplot_titles=[STATION_LABELS[station] for station in STATIONS],
        horizontal_spacing=0.06,
    )
    for col_index, station in enumerate(STATIONS, start=1):
        subset = best[best["station"] == station]
        for recipe in RECIPE_IDS:
            for seed_index in sorted(SEED_INDEX_NAMES):
                point = subset[
                    (subset["recipe_id"] == recipe) & (subset["seed_index"] == seed_index)
                ]
                if len(point) != 1:
                    raise ValueError("best-checkpoint table must have one row per fit.")
                figure.add_trace(
                    go.Scatter(
                        x=point["train_balanced_accuracy"].tolist(),
                        y=point["validation_balanced_accuracy"].tolist(),
                        mode="markers",
                        name=f"{RECIPE_DESCRIPTIONS[recipe]} [{SEED_INDEX_NAMES[seed_index]}]",
                        legendgroup=recipe,
                        marker={
                            "color": RECIPE_COLORS[recipe],
                            "size": 13,
                            "symbol": SEED_SYMBOL[seed_index],
                            "line": {"color": "black", "width": 0.7},
                        },
                        showlegend=(col_index == 1),
                        customdata=np.column_stack(
                            [
                                point["epoch"].tolist(),
                                point["validation_nll"].tolist(),
                                point["validation_macro_f1"].tolist(),
                            ]
                        ),
                        hovertemplate=(
                            f"{STATION_LABELS[station]}<br>{RECIPE_DESCRIPTIONS[recipe]}"
                            f" [{SEED_INDEX_NAMES[seed_index]}]<br>"
                            "training BA %{x:.4f}<br>source-validation BA %{y:.4f}<br>"
                            "best epoch %{customdata[0]}<br>source-validation NLL %{customdata[1]:.4f}<br>"
                            "source-validation macro-F1 %{customdata[2]:.4f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=col_index,
                )
        figure.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                line={"color": "black", "dash": "dot", "width": 1.0},
                name="identity",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=col_index,
        )
        figure.update_xaxes(
            title_text="Training BA", range=[0.0, 1.0], row=1, col=col_index
        )
    figure.update_yaxes(
        title_text="Validation BA", range=[0.0, 1.0], row=1, col=1
    )
    figure.update_layout(
        title={"text": BEST_TITLE + "<br><sup>" + TRAINING_CLAIM + "</sup>", "x": 0.5},
        template="simple_white",
        font={"color": "black", "size": 11},
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=640,
        width=1180,
        legend={
            "orientation": "v",
            "x": 1.02,
            "y": 1.0,
            "font": {"size": 9, "color": "black"},
            "title": {"text": "Recipe / initialization", "font": {"size": 9}},
        },
        margin={"l": 80, "r": 215, "t": 120, "b": 150},
    )
    figure.add_annotation(
        x=0.5,
        y=-0.20,
        xref="paper",
        yref="paper",
        xanchor="center",
        yanchor="top",
        showarrow=False,
        align="left",
        font={"size": 10, "color": "black"},
        text=_html_caption(),
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
    figure.update_yaxes(showgrid=True, gridcolor="#e8e8e8")
    return figure


def _learning_curves_tikz(frame: pd.DataFrame, digest: str) -> str:
    lines = _tex_preamble(
        digest,
        "P05 pilot learning curves; source-validation development fits, not outer-test results.",
    )
    lines.extend(
        [
            r"\begin{groupplot}[",
            r"  group style={group size=2 by 3, xlabels at=edge bottom, horizontal sep=1.6cm, vertical sep=1.6cm},",
            r"  width=6.2cm, height=3.9cm,",
            r"  every axis plot/.append style={line width=0.8pt},",
            r"  tick label style={font=\small, color=black},",
            r"  label style={font=\small, color=black},",
            r"  title style={font=\small\bfseries, color=black},",
            r"]",
        ]
    )
    for station in STATIONS:
        subset = frame[frame["station"] == station]
        lines.append(
            rf"\nextgroupplot[title={{{_tex(STATION_LABELS[station])}: Training CE}}, xlabel={{epoch}}, ylabel={{Training CE}}, ymin=0]"
        )
        lines.extend(_trajectory_plots(subset, "chemical_ce"))
        lines.append(
            rf"\nextgroupplot[title={{{_tex(STATION_LABELS[station])}: Validation BA}}, xlabel={{epoch}}, ylabel={{Validation BA}}, ymin=0, ymax=1]"
        )
        lines.extend(_trajectory_plots(subset, "validation_balanced_accuracy"))
    lines.append(r"\end{groupplot}")
    lines.append(_tex_title_node(LEARNING_TITLE, TRAINING_CLAIM))
    lines.append(_tex_caption_node(CAPTION_NOTE))
    lines.append(_tex_legend_node(LEGEND_RECIPES + LEGEND_SEED_LINES))
    lines.extend([r"\end{tikzpicture}", r"\end{document}", ""])
    return "\n".join(lines)


def _best_checkpoints_tikz(frame: pd.DataFrame, digest: str) -> str:
    best = frame[frame["is_best"]]
    lines = _tex_preamble(
        digest,
        "P05 pilot best checkpoints; source-validation development fits, not outer-test results.",
    )
    lines.extend(
        [
            r"\begin{groupplot}[",
            r"  group style={group size=3 by 1, ylabels at=edge left, horizontal sep=1.6cm},",
            r"  width=5.6cm, height=5.6cm,",
            r"  every axis plot/.append style={line width=0.8pt},",
            r"  tick label style={font=\small, color=black},",
            r"  label style={font=\small, color=black},",
            r"  title style={font=\small\bfseries, color=black},",
            r"]",
        ]
    )
    for station in STATIONS:
        subset = best[best["station"] == station]
        lines.append(
            rf"\nextgroupplot[title={{{_tex(STATION_LABELS[station])}}}, xlabel={{Training BA}}, ylabel={{Validation BA}}, xmin=0, xmax=1, ymin=0, ymax=1]"
        )
        lines.append(
            r"\addplot[black, dotted, line width=0.8pt, forget plot] coordinates {(0,0) (1,1)};"
        )
        for recipe in RECIPE_IDS:
            for seed_index in sorted(SEED_INDEX_NAMES):
                point = subset[
                    (subset["recipe_id"] == recipe) & (subset["seed_index"] == seed_index)
                ]
                if len(point) != 1:
                    raise ValueError("best-checkpoint table must have one row per fit.")
                x = float(point["train_balanced_accuracy"].iloc[0])
                y = float(point["validation_balanced_accuracy"].iloc[0])
                lines.append(
                    rf"\addplot[only marks, color={TEX_COLORS[recipe]}, mark={TEX_MARK[seed_index]}, mark size=2.2pt] coordinates {{({x:.17g},{y:.17g})}};"
                )
    lines.append(r"\end{groupplot}")
    lines.append(_tex_title_node(BEST_TITLE, TRAINING_CLAIM))
    lines.append(_tex_caption_node(CAPTION_NOTE))
    lines.append(_tex_legend_node(LEGEND_RECIPES + LEGEND_SEED_SYMBOLS))
    lines.extend([r"\end{tikzpicture}", r"\end{document}", ""])
    return "\n".join(lines)


def _emit(
    root: Path,
    stem: str,
    figure: go.Figure,
    tex_source: str,
    digest: str,
    description: str,
) -> None:
    _write_html(figure, root / f"{stem}.html", digest=digest, description=description)
    tex_path = root / f"{stem}.tex"
    tex_path.write_text(tex_source, encoding="utf-8")
    _compile(tex_path, root / f"{stem}.pdf", root / f"{stem}.png", root / f"{stem}.log")


def generate_pilot_figures(records: object, output_root: object) -> dict:
    root = _guard_output_root(output_root)
    frame = build_semantic_data(records)
    _configure_deterministic_pdf()
    root.mkdir(parents=True)
    csv_path = root / SEMANTIC_NAME
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    digest = sha256_file(csv_path)
    _emit(
        root,
        "P05P01_learning_curves",
        _learning_curves_figure(frame),
        _learning_curves_tikz(frame, digest),
        digest,
        LEARNING_TITLE + " " + TRAINING_CLAIM,
    )
    _emit(
        root,
        "P05P02_best_checkpoints",
        _best_checkpoints_figure(frame),
        _best_checkpoints_tikz(frame, digest),
        digest,
        BEST_TITLE + " " + TRAINING_CLAIM,
    )
    manifest = {"data_sha256": digest, "files": _manifest_files(root)}
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
