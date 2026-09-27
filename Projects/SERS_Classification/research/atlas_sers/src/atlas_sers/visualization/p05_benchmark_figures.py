# ruff: noqa: E501
"""P05 benchmark paired figures from public aggregate domain means.

Builds one allowlisted semantic table from caller-supplied station-instrument
domain aggregates and emits, per registered reference/aggregation pair, a
three-panel paired scatter (CWA/Pills/Surfaces) as native pgfplots/TikZ source,
a standalone offline Plotly HTML file, a vector PDF and a 300 dpi PNG through
the shared HTML writer and bounded P05 compiler. This module never reads scientific input files,
fits or imputes; every point is an aggregate mean over common complete
outer contexts, not an independent chemical sample.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from atlas_sers.evaluation.p05_comparison import AGGREGATIONS, P05_MODELS, PAIRS
from atlas_sers.governance.canonical import sha256_file
from atlas_sers.visualization.p04_figures import _write_html
from atlas_sers.visualization.p05_figure_runtime import _compile
from atlas_sers.visualization.p05_smoke_figures import (
    _configure_deterministic_pdf,
    _guard_output_root,
)

__all__ = ["BenchmarkFigureError", "generate_pair_figures"]

MODEL_ORDER = tuple(P05_MODELS)
REGISTERED_REFERENCES = tuple(dict.fromkeys(reference for _, reference in PAIRS))
PAIR_SET = frozenset((str(model), str(reference)) for model, reference in PAIRS)
STATION_ORDER = ("cwa", "pills", "surfaces")
STATION_LABELS = {"cwa": "CWA", "pills": "Pills", "surfaces": "Surfaces"}
AGGREGATION_ORDER = tuple(AGGREGATIONS)

INPUT_COLUMNS = (
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
SEMANTIC_COLUMNS = INPUT_COLUMNS

MODEL_LABELS = {
    "D0-M": "Matched CNN (D0-M)",
    "P05-SELECTED": "Source-selected CNN",
    "D3": "Combined-loss CNN (D3)",
}
MODEL_TEX_COLOR = {"D0-M": "p05Blue", "P05-SELECTED": "p05Orange", "D3": "p05Purple"}
MODEL_HTML_COLOR = {"D0-M": "#0072B2", "P05-SELECTED": "#E69F00", "D3": "#CC79A7"}
MODEL_TEX_MARK = {"D0-M": "*", "P05-SELECTED": "square", "D3": "triangle"}
MODEL_HTML_SYMBOL = {"D0-M": "circle", "P05-SELECTED": "square-open", "D3": "triangle-up-open"}

MANIFEST_NAME = "P05_benchmark_manifest.json"
SEMANTIC_NAME = "semantic_data.csv"
ALLOWED_SUFFIXES = {".csv", ".tex", ".pdf", ".png", ".html"}
COVERAGE_TOLERANCE = 1e-12
DELTA_TOLERANCE = 1e-12

CAPTION_LINES = (
    "Each point is a station-instrument domain mean balanced accuracy over common complete outer contexts.",
    "Only domains with at least one common complete outer context are plotted; absent points are not imputed.",
    "Repeated outer splits are not independent chemical samples.",
    "New neural strategies average three-seed calibrated probabilities before scoring.",
    "M01: individual spectra. M06: instrument-balanced mean prediction per physical sample, not averaging raw spectra.",
    "Sparse-class balanced accuracy covers observed classes only; no significance or confidence-interval claim is made.",
    "Empty panels indicate no common complete contexts for that station.",
)

TEX_MAP = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "%": r"\%",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


class BenchmarkFigureError(ValueError):
    """Raised when P05 benchmark aggregate inputs are missing or inconsistent."""


def _tex_literal(value: object) -> str:
    return "".join(TEX_MAP.get(character, character) for character in str(value))


def _html_literal(value: object) -> str:
    return html.escape(str(value), quote=True)


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BenchmarkFigureError(f"{field} must be a non-empty, trimmed string.")
    return value


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or value is None:
        raise BenchmarkFigureError(f"{field} must be an integer.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number) or number != int(number):
            raise BenchmarkFigureError(f"{field} must be an integer.")
        return int(number)
    raise BenchmarkFigureError(f"{field} must be an integer.")


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool) or value is None:
        raise BenchmarkFigureError(f"{field} must be numeric.")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise BenchmarkFigureError(f"{field} must be numeric.")
    number = float(value)
    if not np.isfinite(number):
        raise BenchmarkFigureError(f"{field} must be finite.")
    return number


def _require_unit(value: object, field: str) -> float:
    number = _as_float(value, field)
    if number < 0.0 or number > 1.0:
        raise BenchmarkFigureError(f"{field} must lie in [0, 1].")
    return number


def _require_missing(value: object, field: str) -> None:
    if not pd.isna(value):
        raise BenchmarkFigureError(f"{field} must be NaN when there are no common contexts.")


def _validated_row(record: dict) -> dict:
    station = _require_text(record["station"], "station")
    if station not in STATION_ORDER:
        raise BenchmarkFigureError(f"unsupported station: {station!r}")
    domain = _require_text(record["domain"], "domain")
    instrument = _require_text(record["held_instrument"], "held_instrument")
    model = _require_text(record["model_id"], "model_id")
    reference = _require_text(record["reference_model_id"], "reference_model_id")
    aggregation = _require_text(record["aggregation_id"], "aggregation_id")
    if model not in MODEL_ORDER:
        raise BenchmarkFigureError(f"unsupported model_id: {model!r}")
    if reference not in REGISTERED_REFERENCES:
        raise BenchmarkFigureError(f"unsupported reference_model_id: {reference!r}")
    if (model, reference) not in PAIR_SET:
        raise BenchmarkFigureError(f"unregistered model/reference pair: {model!r}/{reference!r}")
    if aggregation not in AGGREGATION_ORDER:
        raise BenchmarkFigureError(f"unsupported aggregation_id: {aggregation!r}")
    planned = _as_int(record["planned_contexts"], "planned_contexts")
    if planned <= 0:
        raise BenchmarkFigureError("planned_contexts must be positive.")
    common = _as_int(record["common_contexts"], "common_contexts")
    if common < 0 or common > planned:
        raise BenchmarkFigureError("common_contexts must satisfy 0 <= common <= planned.")
    coverage = _as_float(record["common_coverage"], "common_coverage")
    if abs(coverage - common / planned) > COVERAGE_TOLERANCE:
        raise BenchmarkFigureError("common_coverage must equal common_contexts / planned_contexts.")
    if common > 0:
        model_ba = _require_unit(
            record["mean_model_balanced_accuracy"], "mean_model_balanced_accuracy"
        )
        reference_ba = _require_unit(
            record["mean_reference_balanced_accuracy"], "mean_reference_balanced_accuracy"
        )
        delta_ba = _as_float(record["mean_delta_balanced_accuracy"], "mean_delta_balanced_accuracy")
        if abs(delta_ba - (model_ba - reference_ba)) > DELTA_TOLERANCE:
            raise BenchmarkFigureError(
                "mean_delta_balanced_accuracy must equal model minus reference."
            )
    else:
        _require_missing(record["mean_model_balanced_accuracy"], "mean_model_balanced_accuracy")
        _require_missing(
            record["mean_reference_balanced_accuracy"], "mean_reference_balanced_accuracy"
        )
        _require_missing(record["mean_delta_balanced_accuracy"], "mean_delta_balanced_accuracy")
        model_ba = reference_ba = delta_ba = float("nan")
    return {
        "station": station,
        "domain": domain,
        "held_instrument": instrument,
        "model_id": model,
        "reference_model_id": reference,
        "aggregation_id": aggregation,
        "planned_contexts": planned,
        "common_contexts": common,
        "common_coverage": coverage,
        "mean_model_balanced_accuracy": model_ba,
        "mean_reference_balanced_accuracy": reference_ba,
        "mean_delta_balanced_accuracy": delta_ba,
    }


def _order_semantic(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.assign(
        _station=frame.station.map({value: index for index, value in enumerate(STATION_ORDER)}),
        _domain=frame.domain,
        _model=frame.model_id.map({value: index for index, value in enumerate(MODEL_ORDER)}),
        _reference=frame.reference_model_id.map(
            {value: index for index, value in enumerate(REGISTERED_REFERENCES)}
        ),
        _aggregation=frame.aggregation_id.map(
            {value: index for index, value in enumerate(AGGREGATION_ORDER)}
        ),
    )
    ordered = ordered.sort_values(
        ["_station", "_domain", "_model", "_reference", "_aggregation"], kind="stable"
    )
    return ordered[list(SEMANTIC_COLUMNS)].reset_index(drop=True)


def _check_domain_consistency(frame: pd.DataFrame) -> None:
    for domain, cell in frame.groupby("domain", sort=False):
        if cell.station.nunique() != 1 or cell.held_instrument.nunique() != 1:
            raise BenchmarkFigureError(
                f"domain {domain!r} maps to multiple stations or instruments."
            )


def _check_unique(frame: pd.DataFrame) -> None:
    keys = ["domain", "model_id", "reference_model_id", "aggregation_id"]
    if frame.duplicated(keys).any():
        raise BenchmarkFigureError("duplicate domain/model/reference/aggregation keys.")


def _build_semantic(paired_domains: object) -> pd.DataFrame:
    if isinstance(paired_domains, (str, bytes)) or not isinstance(paired_domains, pd.DataFrame):
        raise BenchmarkFigureError(
            "paired_domains must be a pandas DataFrame of aggregate domain means."
        )
    if paired_domains.empty:
        raise BenchmarkFigureError("paired_domains must not be empty.")
    columns = list(paired_domains.columns)
    missing = [column for column in INPUT_COLUMNS if column not in columns]
    if missing:
        raise BenchmarkFigureError("paired_domains is missing columns: " + ",".join(missing))
    records = [
        _validated_row(record) for record in paired_domains[list(INPUT_COLUMNS)].to_dict("records")
    ]
    frame = pd.DataFrame(records, columns=list(SEMANTIC_COLUMNS))
    frame = _order_semantic(frame)
    _check_domain_consistency(frame)
    _check_unique(frame)
    return frame


def _figure_specs(frame: pd.DataFrame) -> list[dict]:
    specs = []
    for reference in REGISTERED_REFERENCES:
        for aggregation in AGGREGATION_ORDER:
            mask = frame.reference_model_id.eq(reference) & frame.aggregation_id.eq(aggregation)
            subset = frame[mask]
            if subset.empty:
                continue
            specs.append(
                {
                    "reference": reference,
                    "aggregation": aggregation,
                    "rows": subset.reset_index(drop=True),
                }
            )
    return specs


def _figure_stem(reference: str, aggregation: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in reference)
    return f"P05B_{safe}_{aggregation}"


def _figure_title(spec: dict) -> str:
    return f"P05 paired balanced accuracy: new strategies vs {spec['reference']} ({spec['aggregation']})"


def _figure_counts(rows: pd.DataFrame) -> dict:
    plotted = rows[rows.common_contexts.gt(0)]
    return {
        "semantic_rows": int(len(rows)),
        "plotted_rows": int(len(plotted)),
        "planned_model_context_contributions": int(rows.planned_contexts.sum()),
        "common_model_context_contributions": int(rows.common_contexts.sum()),
        "station_plotted": {
            station: int(plotted.station.eq(station).sum()) for station in STATION_ORDER
        },
    }


def _tikz_source(rows: pd.DataFrame, spec: dict, digest: str) -> str:
    lines = [
        "% P05 benchmark paired scatter; aggregate station-instrument domain means.",
        "% Each point is a domain mean over common complete outer contexts; not an independent sample.",
        f"% data_sha256={digest}",
        r"\documentclass[tikz,border=5pt]{standalone}",
        r"\ifdefined\pdfinfoomitdate\pdfinfoomitdate=1\fi",
        r"\ifdefined\pdfsuppressptexinfo\pdfsuppressptexinfo=-1\fi",
        r"\ifdefined\pdftrailerid\pdftrailerid{}\fi",
        r"\usepackage{pgfplots}",
        r"\usepgfplotslibrary{groupplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\definecolor{p05Blue}{HTML}{0072B2}",
        r"\definecolor{p05Orange}{HTML}{E69F00}",
        r"\definecolor{p05Purple}{HTML}{CC79A7}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=3 by 1, horizontal sep=1.1cm},",
        r"  width=4.8cm, height=4.8cm, scale only axis,",
        r"  xmin=0, xmax=1, ymin=0, ymax=1,",
        r"  xtick={0,0.5,1}, ytick={0,0.5,1},",
        r"  tick label style={font=\fontsize{8}{10}\selectfont, color=black},",
        r"  label style={font=\fontsize{8}{10}\selectfont, color=black},",
        r"  title style={font=\fontsize{8}{10}\selectfont\bfseries, color=black},",
        r"]",
    ]
    for index, station in enumerate(STATION_ORDER):
        options = [
            f"title={{({chr(65 + index)}) {STATION_LABELS[station]}}}",
            "xlabel={reference balanced accuracy}",
        ]
        if index == 0:
            options.append("ylabel={new-strategy balanced accuracy}")
        lines.append("\\nextgroupplot[" + ", ".join(options) + "]")
        panel = rows[rows.station.eq(station)]
        for model in MODEL_ORDER:
            points = panel[panel.model_id.eq(model) & panel.common_contexts.gt(0)]
            if points.empty:
                continue
            coordinates = " ".join(
                f"({float(row.mean_reference_balanced_accuracy):.17g},{float(row.mean_model_balanced_accuracy):.17g})"
                for row in points.itertuples(index=False)
            )
            color = MODEL_TEX_COLOR[model]
            mark = MODEL_TEX_MARK[model]
            fill = color if model == "D0-M" else "none"
            size = "1.6pt" if model == "D0-M" else "2.3pt"
            lines.append(
                rf"\addplot[only marks, mark={mark}, mark size={size}, color={color}, mark options={{fill={fill}, draw={color}}}] coordinates {{{coordinates}}};"
            )
        lines.append(r"\addplot[black, dotted, line width=0.8pt] coordinates {(0,0) (1,1)};")
        if not panel.common_contexts.gt(0).any():
            lines.append(
                r"\node[font=\fontsize{8}{10}\selectfont,align=center] at (axis cs:0.5,0.65) {No common\\complete contexts};"
            )
    lines.append(r"\end{groupplot}")
    lines.append(
        r"\node[anchor=south, font=\fontsize{8}{10}\selectfont\bfseries, color=black, align=center, text width=16.6cm] at (current bounding box.north) {"
        + _tex_literal(_figure_title(spec))
        + r"};"
    )
    lines.append(
        r"\node[anchor=north, font=\fontsize{8}{10}\selectfont, color=black, align=left, text width=16.6cm] (p05legend) at ([yshift=-2mm]current bounding box.south) {"
        r"\tikz[baseline=-0.6ex]\fill[p05Blue](0,0)circle(1.4pt);\ D0-M\quad "
        r"\tikz[baseline=-0.6ex]\draw[p05Orange](0,0)rectangle(2.8pt,2.8pt);\ Source-selected CNN\quad "
        r"\tikz[baseline=-0.6ex]\draw[p05Purple](0,0)--(1.6pt,2.8pt)--(3.2pt,0)--cycle;\ Combined-loss CNN (D3)\\[0.8mm]"
        r"\tikz[baseline=-0.6ex]\draw[black,dotted,line width=0.8pt](0,0)--(0.75,0);\ identity (equal balanced accuracy)"
        r"};"
    )
    caption = r"\\".join(_tex_literal(line) for line in CAPTION_LINES)
    lines.append(
        r"\node[anchor=north, font=\fontsize{8}{10}\selectfont, color=black, align=left, text width=16.6cm] at ([yshift=-1mm]p05legend.south) {"
        + caption
        + r"};"
    )
    lines.extend([r"\end{tikzpicture}", r"\end{document}", ""])
    return "\n".join(lines)


def _html_caption() -> str:
    return "<br>".join(CAPTION_LINES)


def _plotly_figure(rows: pd.DataFrame, spec: dict) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"({chr(65 + index)}) {STATION_LABELS[station]}"
            for index, station in enumerate(STATION_ORDER)
        ],
        horizontal_spacing=0.08,
    )
    shown = set()
    for column, station in enumerate(STATION_ORDER, start=1):
        panel = rows[rows.station.eq(station)]
        for model in MODEL_ORDER:
            points = panel[panel.model_id.eq(model) & panel.common_contexts.gt(0)]
            if points.empty:
                continue
            customdata = [
                [
                    _html_literal(row.domain),
                    _html_literal(row.held_instrument),
                    int(row.planned_contexts),
                    int(row.common_contexts),
                    float(row.mean_delta_balanced_accuracy),
                ]
                for row in points.itertuples(index=False)
            ]
            figure.add_trace(
                go.Scatter(
                    x=[float(value) for value in points.mean_reference_balanced_accuracy],
                    y=[float(value) for value in points.mean_model_balanced_accuracy],
                    mode="markers",
                    name=MODEL_LABELS[model],
                    legendgroup=model,
                    showlegend=model not in shown,
                    marker={
                        "symbol": MODEL_HTML_SYMBOL[model],
                        "color": MODEL_HTML_COLOR[model],
                        "size": 7 if model == "D0-M" else 10,
                        "line": {"color": MODEL_HTML_COLOR[model], "width": 1.2},
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "%{customdata[0]} &middot; %{customdata[1]}"
                        "<br>reference BA=%{x:.3f}<br>new-strategy BA=%{y:.3f}"
                        "<br>delta BA=%{customdata[4]:.3f}"
                        "<br>common complete contexts=%{customdata[3]}/%{customdata[2]}"
                        "<extra>" + MODEL_LABELS[model] + "</extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            shown.add(model)
        figure.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                line={"color": "black", "dash": "dot", "width": 1},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )
        if not panel.common_contexts.gt(0).any():
            figure.add_annotation(
                x=0.5,
                y=0.65,
                text="No common<br>complete contexts",
                showarrow=False,
                row=1,
                col=column,
            )
    for column in range(1, 4):
        figure.update_xaxes(
            title_text="reference balanced accuracy",
            range=[0, 1],
            constrain="domain",
            row=1,
            col=column,
        )
        figure.update_yaxes(
            range=[0, 1],
            scaleanchor="x" if column == 1 else f"x{column}",
            scaleratio=1,
            constrain="domain",
            row=1,
            col=column,
        )
    figure.update_yaxes(title_text="new-strategy balanced accuracy", row=1, col=1)
    figure.update_layout(
        title={"text": _figure_title(spec), "x": 0.5},
        template="simple_white",
        font={"family": "Times New Roman, Times, serif", "color": "black", "size": 12},
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=1500,
        height=740,
        margin={"l": 70, "r": 40, "t": 110, "b": 220},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.24,
            "font": {"size": 11, "color": "black"},
        },
    )
    figure.add_annotation(
        x=0.5,
        y=-0.32,
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


def _manifest_files(root: Path) -> list[dict]:
    files = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name == MANIFEST_NAME or path.is_symlink() or not path.is_file():
            continue
        if path.suffix not in ALLOWED_SUFFIXES:
            continue
        files.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    return files


def generate_pair_figures(
    paired_domains: object, output_root: object, *, deadline: float | None = None
) -> dict:
    frame = _build_semantic(paired_domains)
    specs = _figure_specs(frame)
    if not specs:
        raise BenchmarkFigureError("no registered reference/aggregation pairs are present.")
    root = _guard_output_root(output_root)
    _configure_deterministic_pdf()
    root.mkdir(parents=True)
    csv_path = root / SEMANTIC_NAME
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    digest = sha256_file(csv_path)
    figure_records = []
    for spec in specs:
        reference = spec["reference"]
        aggregation = spec["aggregation"]
        rows = spec["rows"]
        stem = _figure_stem(reference, aggregation)
        tex_path = root / f"{stem}.tex"
        html_path = root / f"{stem}.html"
        pdf_path = root / f"{stem}.pdf"
        png_path = root / f"{stem}.png"
        log_path = root / f"{stem}.log"
        _write_html(
            _plotly_figure(rows, spec),
            html_path,
            digest=digest,
            description=_figure_title(spec) + " " + CAPTION_LINES[0],
        )
        tex_path.write_text(_tikz_source(rows, spec, digest), encoding="utf-8")
        if deadline is None:
            _compile(tex_path, pdf_path, png_path, log_path)
        else:
            _compile(tex_path, pdf_path, png_path, log_path, deadline=deadline)
        if digest not in tex_path.read_text(encoding="utf-8") or digest not in html_path.read_text(
            encoding="utf-8"
        ):
            raise RuntimeError(f"{stem} semantic data hash parity failed.")
        figure_records.append(
            {
                "figure_id": f"P05B-{reference}-{aggregation}",
                "reference_model_id": reference,
                "aggregation_id": aggregation,
                "semantic_sha256": digest,
                "tex_sha256": sha256_file(tex_path),
                "pdf_sha256": sha256_file(pdf_path),
                "png_sha256": sha256_file(png_path),
                "html_sha256": sha256_file(html_path),
                **_figure_counts(rows),
            }
        )
    manifest = {
        "data_sha256": digest,
        "semantic_path": SEMANTIC_NAME,
        "figures": figure_records,
        "files": _manifest_files(root),
    }
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
