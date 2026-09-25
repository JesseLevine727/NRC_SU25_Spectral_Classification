# ruff: noqa: E501
"""P05 training-only smoke figures.

Builds one allowlisted semantic table (256 fit-epoch rows) and emits, per figure,
native pgfplots/TikZ source, a standalone offline Plotly HTML file, a vector PDF
and a PNG through the unchanged P04 helpers. Every point is a single fit-epoch.
This is an eight-epoch numerical check on training fits; it is not a model
comparison on unseen data.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from atlas_sers.governance.canonical import sha256_file
from atlas_sers.visualization.p04_figures import _compile, _tex, _write_html

__all__ = ["build_semantic_data", "generate_smoke_figures"]

PRIMARY_KIND = "primary"
REPLAY_KIND = "replay"
KNOWN_EXECUTION_KINDS = (PRIMARY_KIND, REPLAY_KIND)

# Actual research source-role taxonomy. Only these four semantic codes are valid.
ROLE_LABELS = ("cwa_dense", "pills_dense", "surfaces_dense", "surfaces_sparse")
ROLE_SUPPORT = {
    "cwa_dense": 13,
    "pills_dense": 9,
    "surfaces_dense": 13,
    "surfaces_sparse": 4,
}
SPARSE_ROLE = "surfaces_sparse"
DISPLAY_LABELS = {
    "cwa_dense": "CWA",
    "pills_dense": "Pills",
    "surfaces_dense": "Surfaces",
    "surfaces_sparse": "Surfaces (sparse)",
}

RECIPE_IDS = ("D0-M", "D1", "D2", "D3")
RECIPE_DESCRIPTIONS = {
    "D0-M": "D0-M ordinary classification",
    "D1": "D1 chemical similarity",
    "D2": "D2 matched-sample consistency",
    "D3": "D3 both",
}
RECIPE_COLORS = {
    "D0-M": "#0072B2",
    "D1": "#E69F00",
    "D2": "#009E73",
    "D3": "#CC79A7",
}
TEX_COLORS = {
    "D0-M": "p05Dzero",
    "D1": "p05Done",
    "D2": "p05Dtwo",
    "D3": "p05Dthree",
}

SEEDS = (20260805, 20260817)
SEED_INDEX = {SEEDS[0]: 1, SEEDS[1]: 2}
SEED_INDEX_NAMES = {1: "seed1", 2: "seed2"}
SEED_DASH = {1: "solid", 2: "dash"}

EPOCHS = tuple(range(1, 9))
SEMANTIC_COLUMNS = (
    "role_label",
    "recipe_id",
    "seed_index",
    "epoch",
    "chemical_ce",
    "gradient_norm_mean",
)
EXPECTED_FITS = len(ROLE_LABELS) * len(RECIPE_IDS) * len(SEED_INDEX)
EXPECTED_ROWS = EXPECTED_FITS * len(EPOCHS)

MANIFEST_NAME = "P05_figure_manifest.json"
SEMANTIC_NAME = "semantic_data.csv"
ALLOWED_SUFFIXES = {".csv", ".tex", ".pdf", ".png", ".html"}

TRAINING_CLAIM = "Eight-epoch training-only numerical check; no unseen-data comparison."
SPARSE_DISCLAIMER = (
    "Sparse panel: cross-instrument measurements of the same sample are unavailable; "
    "D2/D0-M and D3/D1 overlap by construction; overlap is not evidence of learned invariance."
)
CAPTION_NOTE = (
    "Solid line: seed1; dashed line: seed2. D0-M ordinary classification; "
    "D1 chemical similarity; D2 matched-sample consistency; D3 both. "
) + SPARSE_DISCLAIMER

FIGURE_SPECS = (
    {
        "stem": "P05S01_training_ce",
        "column": "chemical_ce",
        "ylabel": "weighted chemical CE (training)",
        "title": "P05-S01 weighted chemical training CE over epochs 1-8",
    },
    {
        "stem": "P05S02_gradient_norm",
        "column": "gradient_norm_mean",
        "ylabel": "pre-clipping gradient norm",
        "title": "P05-S02 pre-clipping gradient norm over epochs 1-8",
    },
)


def _html_caption() -> str:
    return (
        "Solid line: seed1; dashed line: seed2.<br>"
        "D0-M ordinary classification; D1 chemical similarity; "
        "D2 matched-sample consistency; D3 both.<br>" + SPARSE_DISCLAIMER
    )


def _require_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"semantic fit field {field!r} is not numeric.")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"semantic fit field {field!r} must be finite and nonnegative.")
    return number


def _require_epoch(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("semantic fit has a malformed or out-of-range epoch.")
    epoch = int(value)
    if epoch not in EPOCHS:
        raise ValueError("semantic fit has a malformed or out-of-range epoch.")
    return epoch


def build_semantic_data(records: object) -> pd.DataFrame:
    if isinstance(records, (str, bytes)) or not isinstance(records, (list, tuple)):
        raise TypeError("records must be a list of execution summary mappings.")
    fits: dict[tuple[str, str, int], dict[int, tuple[float, float]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("every record must be a mapping.")
        kind = record.get("execution_kind")
        if kind not in KNOWN_EXECUTION_KINDS:
            raise ValueError("record has an unknown execution_kind.")
        if kind == REPLAY_KIND:
            continue
        if record.get("role_label") not in ROLE_LABELS:
            raise ValueError("primary record has an unknown role_label.")
        if record.get("recipe_id") not in RECIPE_IDS:
            raise ValueError("primary record has an unknown recipe_id.")
        seed = record.get("seed")
        if seed not in SEED_INDEX:
            raise ValueError("primary record has an unknown seed.")
        if record.get("status") != "complete":
            raise ValueError("primary record is not a completed fit.")
        key = (record["role_label"], record["recipe_id"], SEED_INDEX[seed])
        if key in fits:
            raise ValueError("duplicate primary fit.")
        history = record.get("history")
        if not isinstance(history, (list, tuple)) or len(history) != len(EPOCHS):
            raise ValueError("primary fit has a truncated or malformed history.")
        values: dict[int, tuple[float, float]] = {}
        for entry in history:
            if not isinstance(entry, dict):
                raise TypeError("primary fit history entry is not a mapping.")
            epoch = _require_epoch(entry.get("epoch"))
            if epoch in values:
                raise ValueError("primary fit repeats an epoch.")
            chemical = _require_number(entry.get("chemical_ce"), "chemical_ce")
            gradient = _require_number(entry.get("gradient_norm_mean"), "gradient_norm_mean")
            values[epoch] = (chemical, gradient)
        if set(values) != set(EPOCHS):
            raise ValueError("primary fit is missing one or more epochs 1-8.")
        fits[key] = values
    expected = {
        (role, recipe, seed_index)
        for role in ROLE_LABELS
        for recipe in RECIPE_IDS
        for seed_index in sorted(SEED_INDEX_NAMES)
    }
    missing = expected - set(fits)
    if missing:
        raise ValueError(f"missing {len(missing)} primary fits.")
    rows = []
    for role in ROLE_LABELS:
        for recipe in RECIPE_IDS:
            for seed_index in sorted(SEED_INDEX_NAMES):
                values = fits[(role, recipe, seed_index)]
                for epoch in EPOCHS:
                    chemical, gradient = values[epoch]
                    rows.append(
                        {
                            "role_label": role,
                            "recipe_id": recipe,
                            "seed_index": seed_index,
                            "epoch": epoch,
                            "chemical_ce": chemical,
                            "gradient_norm_mean": gradient,
                        }
                    )
    frame = pd.DataFrame(rows, columns=list(SEMANTIC_COLUMNS))
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"semantic table must contain {EXPECTED_ROWS} rows.")
    return frame


def _guard_output_root(output_root: object) -> Path:
    raw = Path(output_root)
    if not raw.is_absolute():
        raise ValueError("output_root must be an absolute path.")
    for candidate in (raw, *raw.parents):
        if candidate.is_symlink():
            raise ValueError("output_root must not traverse any symlinked component.")
    path = raw.resolve()
    if path.exists():
        raise FileExistsError("output_root already exists; refusing to overwrite.")
    return path


def _configure_deterministic_pdf() -> None:
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    os.environ.setdefault("FORCE_SOURCE_DATE", "1")


def _plotly_figure(frame: pd.DataFrame, spec: dict) -> go.Figure:
    roles = list(ROLE_LABELS)
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{DISPLAY_LABELS[role]} ({ROLE_SUPPORT[role]} masters)" for role in roles],
        shared_xaxes=True,
        horizontal_spacing=0.12,
        vertical_spacing=0.20,
    )
    column = spec["column"]
    for index, role in enumerate(roles):
        row = index // 2 + 1
        col = index % 2 + 1
        subset = frame[frame["role_label"] == role]
        for recipe in RECIPE_IDS:
            for seed_index in sorted(SEED_INDEX_NAMES):
                points = subset[
                    (subset["recipe_id"] == recipe) & (subset["seed_index"] == seed_index)
                ].sort_values("epoch")
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
                            "width": 2,
                        },
                        marker={
                            "color": RECIPE_COLORS[recipe],
                            "size": 6,
                            "line": {"color": "black", "width": 0.5},
                        },
                        showlegend=(index == 0),
                    ),
                    row=row,
                    col=col,
                )
        figure.update_xaxes(
            title_text="epoch (1-8)",
            tickmode="array",
            tickvals=list(EPOCHS),
            row=row,
            col=col,
        )
        figure.update_yaxes(title_text=spec["ylabel"], rangemode="tozero", row=row, col=col)
    figure.update_layout(
        title={
            "text": spec["title"] + "<br><sup>" + TRAINING_CLAIM + "</sup>",
            "x": 0.5,
        },
        template="simple_white",
        font={"color": "black", "size": 11},
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=880,
        width=1160,
        legend={
            "orientation": "v",
            "x": 1.02,
            "y": 1.0,
            "font": {"size": 9, "color": "black"},
            "title": {"text": "Recipe / seed (seed1 solid, seed2 dashed)", "font": {"size": 9}},
        },
        margin={"l": 70, "r": 240, "t": 120, "b": 150},
    )
    figure.add_annotation(
        x=0.5,
        y=-0.14,
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


def _tikz_source(frame: pd.DataFrame, spec: dict, digest: str) -> str:
    lines = [
        "% P05 training-only smoke figure; not a model comparison on unseen data.",
        f"% data_sha256={digest}",
        r"\documentclass[tikz,border=8pt]{standalone}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\usepgfplotslibrary{groupplots}",
        r"\ifdefined\pdfinfoomitdate\pdfinfoomitdate=1\fi",
        r"\ifdefined\pdfsuppressptexinfo\pdfsuppressptexinfo=-1\fi",
        r"\ifdefined\pdftrailerid\pdftrailerid{}\fi",
        r"\definecolor{p05Dzero}{HTML}{0072B2}",
        r"\definecolor{p05Done}{HTML}{E69F00}",
        r"\definecolor{p05Dtwo}{HTML}{009E73}",
        r"\definecolor{p05Dthree}{HTML}{CC79A7}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=2 by 2, xlabels at=edge bottom, ylabels at=edge left, horizontal sep=1.6cm, vertical sep=2.0cm},",
        r"  width=6.2cm, height=4.6cm,",
        r"  every axis plot/.append style={line width=0.9pt},",
        r"  tick label style={font=\small, color=black},",
        r"  label style={font=\small, color=black},",
        r"  title style={font=\small\bfseries, color=black},",
        r"]",
    ]
    for role in ROLE_LABELS:
        subset = frame[frame["role_label"] == role]
        lines.append(
            rf"\nextgroupplot[title={{{_tex(DISPLAY_LABELS[role])} ({ROLE_SUPPORT[role]} masters)}}, xlabel={{epoch (1--8)}}, ylabel={{{_tex(spec['ylabel'])}}}, xtick={{1,...,8}}, ymin=0]"
        )
        for recipe in RECIPE_IDS:
            for seed_index in sorted(SEED_INDEX_NAMES):
                points = subset[
                    (subset["recipe_id"] == recipe) & (subset["seed_index"] == seed_index)
                ].sort_values("epoch")
                coordinates = " ".join(
                    f"({int(epoch)},{float(value):.17g})"
                    for epoch, value in zip(
                        points["epoch"].tolist(),
                        points[spec["column"]].tolist(),
                        strict=True,
                    )
                )
                style = "solid" if seed_index == 1 else "dashed"
                lines.append(
                    rf"\addplot[color={TEX_COLORS[recipe]}, {style}, mark=*, mark size=1.2pt] coordinates {{{coordinates}}};"
                )
    lines.append(r"\end{groupplot}")
    lines.append(
        r"\node[anchor=south, font=\bfseries\small, color=black, align=center, text width=13cm] at (current bounding box.north) {"
        + _tex(spec["title"])
        + r"\\[1mm]"
        + _tex(TRAINING_CLAIM)
        + r"};"
    )
    lines.append(
        r"\node[anchor=north, font=\small, color=black, align=left, text width=13cm] (p05caption) at ([yshift=-2mm]current bounding box.south) {"
        + _tex(CAPTION_NOTE)
        + r"};"
    )
    lines.append(
        r"\node[anchor=north west, font=\small, color=black, align=left, text width=13cm] at ([yshift=-1mm]p05caption.south west) {"
        r"\textcolor{p05Dzero}{\rule{7mm}{1.2pt}}\,D0-M ordinary classification\quad "
        r"\textcolor{p05Done}{\rule{7mm}{1.2pt}}\,D1 chemical similarity\\[0.8mm]"
        r"\textcolor{p05Dtwo}{\rule{7mm}{1.2pt}}\,D2 matched-sample consistency\quad "
        r"\textcolor{p05Dthree}{\rule{7mm}{1.2pt}}\,D3 both\\[1mm]"
        r"\tikz[baseline=-0.6ex]\draw[black,solid,line width=0.9pt](0,0)--(0.75,0);\ seed1 (solid)\quad "
        r"\tikz[baseline=-0.6ex]\draw[black,dashed,line width=0.9pt](0,0)--(0.75,0);\ seed2 (dashed)"
        r"};"
    )
    lines.extend([r"\end{tikzpicture}", r"\end{document}", ""])
    return "\n".join(lines)


def _emit_figure(frame: pd.DataFrame, root: Path, spec: dict, digest: str) -> None:
    stem = spec["stem"]
    _write_html(
        _plotly_figure(frame, spec),
        root / f"{stem}.html",
        digest=digest,
        description=spec["title"] + " " + TRAINING_CLAIM,
    )
    tex_path = root / f"{stem}.tex"
    tex_path.write_text(_tikz_source(frame, spec, digest), encoding="utf-8")
    _compile(tex_path, root / f"{stem}.pdf", root / f"{stem}.png", root / f"{stem}.log")


def _manifest_files(root: Path) -> list[dict]:
    files = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name == MANIFEST_NAME or path.is_symlink() or not path.is_file():
            continue
        if path.suffix not in ALLOWED_SUFFIXES:
            continue
        files.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    return files


def generate_smoke_figures(records: object, output_root: object) -> dict:
    root = _guard_output_root(output_root)
    frame = build_semantic_data(records)
    _configure_deterministic_pdf()
    root.mkdir(parents=True)
    csv_path = root / SEMANTIC_NAME
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    digest = sha256_file(csv_path)
    for spec in FIGURE_SPECS:
        _emit_figure(frame, root, spec, digest)
    manifest = {"data_sha256": digest, "files": _manifest_files(root)}
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
