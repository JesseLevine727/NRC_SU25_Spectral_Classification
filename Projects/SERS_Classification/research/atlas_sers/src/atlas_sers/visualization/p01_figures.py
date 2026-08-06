"""Paired native TikZ/PDF and standalone HTML figures for P01."""

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

from atlas_sers.governance.canonical import sha256_file

COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#000000",
    "#6A3D9A",
    "#A6CEE3",
    "#B2DF8A",
]
PLOTLY_DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
PLOTLY_MARKERS = [
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "triangle-down",
    "cross",
    "x",
    "star",
]
TIKZ_LINE_STYLES = [
    "solid",
    "densely dashed",
    "densely dotted",
    "dashdotted",
    "loosely dashed",
    "loosely dotted",
]
TIKZ_MARKERS = ["*", "square*", "diamond*", "triangle*", "triangle", "+", "x", "pentagon*"]
REPRESENTATION_LABELS = {
    "R_NATIVE_COMMON_400_1849": "Native",
    "R_MIN_400_1800": "Min-max",
    "R_MIN_400_1849": "Min-max 1849",
    "R_SG_400_1800": "SG",
    "R_ARPLS_400_1800": "arPLS",
    "R_SNV_400_1800": "SNV",
    "R_VECTOR_400_1800": "L2",
    "R_AREA_400_1800": "Area",
    "R_D1_400_1800": "D1",
}
REPRESENTATION_ORDER = {identifier: index for index, identifier in enumerate(REPRESENTATION_LABELS)}


@dataclass(frozen=True)
class Panel:
    title: str
    x: str
    y: str
    series: str
    x_label: str
    y_label: str
    mode: str
    data_filter: tuple[str, str] | None = None
    trace_group: str | None = None
    y_tick_labels: str | None = None


FIGURE_SLUGS = {
    "F02": "observation_flow",
    "F03": "factor_support",
    "F04": "native_axis_coverage",
    "F05": "instrument_spectra",
    "F06": "preprocessing_preservation",
    "F07": "pca_structure",
    "F08": "nonlinear_embeddings",
    "F09": "cluster_association",
}


def _tex(value: Any) -> str:
    text = str(value)
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(source, target)
    return text


def _filtered(frame: pd.DataFrame, panel: Panel) -> pd.DataFrame:
    if panel.data_filter is None:
        return frame
    column, value = panel.data_filter
    return frame[frame[column].astype(str) == value]


def _category_map(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    if pd.api.types.is_numeric_dtype(values):
        return values.to_numpy(float), []
    labels = list(dict.fromkeys(values.astype(str)))
    mapping = {label: index for index, label in enumerate(labels)}
    return values.astype(str).map(mapping).to_numpy(float), labels


def _plotly_figure(title: str, frame: pd.DataFrame, panels: list[Panel]) -> go.Figure:
    figure = make_subplots(
        rows=1, cols=len(panels), subplot_titles=[panel.title for panel in panels]
    )
    legend_seen: set[str] = set()
    for column, panel in enumerate(panels, start=1):
        data = _filtered(frame, panel)
        series_order = sorted(data[panel.series].fillna("<missing>").astype(str).unique())
        grouping = [panel.series, panel.trace_group] if panel.trace_group else panel.series
        for key, group in data.groupby(grouping, sort=True, dropna=False):
            series = key[0] if isinstance(key, tuple) else key
            name = str(series)
            style_index = series_order.index(name)
            color = COLORS[style_index % len(COLORS)]
            common = {
                "x": group[panel.x],
                "y": group[panel.y],
                "name": name,
                "legendgroup": name,
                "showlegend": name not in legend_seen,
            }
            if panel.mode == "bar":
                trace = go.Bar(**common, marker={"color": color})
            else:
                mode = "lines" if panel.mode == "line" else "markers"
                trace = go.Scatter(
                    **common,
                    mode=mode,
                    line={"color": color, "dash": PLOTLY_DASHES[style_index % len(PLOTLY_DASHES)]},
                    marker={
                        "color": color,
                        "symbol": PLOTLY_MARKERS[style_index % len(PLOTLY_MARKERS)],
                    },
                )
            figure.add_trace(trace, row=1, col=column)
            legend_seen.add(name)
        figure.update_xaxes(title_text=panel.x_label, row=1, col=column)
        figure.update_yaxes(title_text=panel.y_label, row=1, col=column)
        if panel.y_tick_labels:
            ticks = (
                data[[panel.y, panel.y_tick_labels]]
                .drop_duplicates()
                .sort_values(panel.y)
            )
            figure.update_yaxes(
                tickmode="array",
                tickvals=ticks[panel.y],
                ticktext=ticks[panel.y_tick_labels],
                row=1,
                col=column,
            )
    figure.update_layout(
        title=title,
        template="plotly_white",
        colorway=COLORS,
        barmode="group",
        font={"family": "Arial, sans-serif", "size": 13},
        height=580,
        width=max(950, 650 * len(panels)),
        margin={"l": 70, "r": 30, "t": 90, "b": 100},
    )
    return figure


def _tikz_source(
    figure_id: str,
    title: str,
    data_sha256: str,
    frame: pd.DataFrame,
    panels: list[Panel],
) -> str:
    width = 0.92 / len(panels)
    lines = [
        r"\documentclass[tikz,border=4pt]{standalone}",
        r"\pdfinfoomitdate=1",
        r"\pdftrailerid{}",
        r"\pdfsuppressptexinfo=-1",
        r"\usepackage{pgfplots}",
        r"\usepgfplotslibrary{groupplots}",
        r"\pgfplotsset{compat=1.18}",
        *[
            f"\\definecolor{{atlas{index}}}{{HTML}}{{{color.lstrip('#')}}}"
            for index, color in enumerate(COLORS)
        ],
        f"% ATLAS figure {figure_id}; data_sha256={data_sha256}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        (
            r"\begin{groupplot}[group style={group size="
            f"{len(panels)} by 1, horizontal sep=1.3cm"
            r"}, width="
            f"{width:.3f}\\textwidth, height=0.52\\textwidth, grid=both, "
            r"tick label style={font=\scriptsize}, label style={font=\small}, "
            r"title style={font=\small\bfseries}, legend style={font=\scriptsize}]"
        ),
    ]
    for panel_index, panel in enumerate(panels):
        data = _filtered(frame, panel)
        x_values, labels = _category_map(data[panel.x])
        data = data.copy()
        data["_plot_x"] = x_values
        options = [
            f"title={{{_tex(panel.title)}}}",
            f"xlabel={{{_tex(panel.x_label)}}}",
            f"ylabel={{{_tex(panel.y_label)}}}",
        ]
        if labels and len(labels) <= 24:
            options.extend(
                [
                    "xtick={" + ",".join(str(index) for index in range(len(labels))) + "}",
                    "xticklabels={" + ",".join(_tex(label) for label in labels) + "}",
                    "x tick label style={rotate=55,anchor=east,font=\\tiny}",
                ]
            )
        if panel.y_tick_labels:
            ticks = (
                data[[panel.y, panel.y_tick_labels]]
                .drop_duplicates()
                .sort_values(panel.y)
            )
            options.extend(
                [
                    "ytick={" + ",".join(f"{value:.8g}" for value in ticks[panel.y]) + "}",
                    "yticklabels={"
                    + ",".join(_tex(value) for value in ticks[panel.y_tick_labels])
                    + "}",
                ]
            )
        lines.append(r"\nextgroupplot[" + ",".join(options) + "]")
        series_order = sorted(data[panel.series].fillna("<missing>").astype(str).unique())
        grouping = [panel.series, panel.trace_group] if panel.trace_group else panel.series
        legend_written: set[str] = set()
        for key, group in data.groupby(grouping, sort=True, dropna=False):
            series = key[0] if isinstance(key, tuple) else key
            series_name = str(series)
            series_index = series_order.index(series_name)
            color = f"atlas{series_index % len(COLORS)}"
            plot_options = f"color={color}"
            if panel.mode == "bar":
                plot_options += ",ybar,fill opacity=0.55"
            elif panel.mode == "scatter":
                marker = TIKZ_MARKERS[series_index % len(TIKZ_MARKERS)]
                plot_options += f",only marks,mark={marker},mark size=1.5pt"
            else:
                line_style = TIKZ_LINE_STYLES[series_index % len(TIKZ_LINE_STYLES)]
                plot_options += f",{line_style},mark=none,line width=0.8pt"
            coordinates = " ".join(
                f"({float(x_value):.8g},{float(y_value):.8g})"
                for x_value, y_value in zip(group["_plot_x"], group[panel.y], strict=True)
                if np.isfinite(float(x_value)) and np.isfinite(float(y_value))
            )
            lines.append(f"\\addplot+[{plot_options}] coordinates {{{coordinates}}};")
            if panel_index == 0 and series_name not in legend_written:
                lines.append(f"\\addlegendentry{{{_tex(series)}}}")
                legend_written.add(series_name)
    lines.extend(
        [
            r"\end{groupplot}",
            (
                "\\node[font=\\small\\bfseries,anchor=south] "
                f"at (current bounding box.north) {{{_tex(title)}}};"
            ),
            r"\end{tikzpicture}",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def _compile(tex_path: Path, pdf_dir: Path) -> tuple[Path, Path]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
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
    log_path = pdf_dir / f"{tex_path.stem}.log"
    if result.returncode != 0:
        raise RuntimeError(f"TikZ compilation failed for {tex_path.name}; inspect private log.")
    pdf_path = pdf_dir / f"{tex_path.stem}.pdf"
    if not pdf_path.is_file():
        raise RuntimeError(f"TikZ compilation did not create {pdf_path.name}.")
    (pdf_dir / f"{tex_path.stem}.aux").unlink(missing_ok=True)
    log_path.write_text(
        "\n".join(
            [
                "schema_version=atlas-p01-figure-compilation-v1",
                "engine=pdflatex",
                f"source={tex_path.name}",
                "return_code=0",
                f"pdf_sha256={sha256_file(pdf_path)}",
                "",
            ]
        )
    )
    return pdf_path, log_path


def build_figure_tables(
    recording_manifest: pd.DataFrame,
    primary: pd.DataFrame,
    native_registry: pd.DataFrame,
    raw_axis: np.ndarray,
    raw: np.ndarray,
    preservation_by_instrument: pd.DataFrame,
    exploration: dict[str, pd.DataFrame],
) -> dict[str, tuple[str, pd.DataFrame, list[Panel]]]:
    flow = pd.DataFrame(
        [
            ("recording observations", len(recording_manifest)),
            ("named sensor", int(recording_manifest.is_named_sers.fillna(False).sum())),
            ("parseable", int(recording_manifest.tier_all_parseable_sers.fillna(False).sum())),
            ("attributable", int(recording_manifest.tier_attributable_sers.fillna(False).sum())),
            ("unique attributable", len(primary)),
            ("notes clear", int(primary.tier_notes_clear_sers.astype(bool).sum())),
        ],
        columns=["stage", "count"],
    )
    flow["series"] = "observations"

    support = (
        primary.groupby(["station", "target_analyte", "instrument"], sort=True)
        .agg(spectra=("observation_uid", "size"), masters=("master_sample_id", "nunique"))
        .reset_index()
    )
    support["station_target"] = support.station + " | " + support.target_analyte

    coverage_rows: list[dict[str, Any]] = []
    for instrument, group in native_registry.groupby("instrument", sort=True):
        instrument_index = sorted(native_registry.instrument.unique()).index(instrument)
        for support_type, low, high in (
            ("native", group.axis_min_cm1.min(), group.axis_max_cm1.max()),
            (
                "effective",
                group.effective_axis_min_cm1.max(),
                group.effective_axis_max_cm1.min(),
            ),
        ):
            for endpoint, value in (("start", low), ("end", high)):
                coverage_rows.append(
                    {
                        "instrument": instrument,
                        "instrument_index": instrument_index,
                        "support_type": support_type,
                        "series": support_type,
                        "trace_group": instrument,
                        "endpoint": endpoint,
                        "axis_cm1": value,
                    }
                )
    coverage = pd.DataFrame(coverage_rows)

    spectral_rows: list[pd.DataFrame] = []
    for instrument, indices in primary.groupby("instrument", sort=True).indices.items():
        values = raw[np.asarray(indices)]
        ranges = np.ptp(values, axis=1, keepdims=True)
        scaled = (values - values.min(axis=1, keepdims=True)) / np.maximum(
            ranges, np.finfo(float).eps
        )
        for view, matrix in (("native intensity", values), ("row min-max", scaled)):
            spectral_rows.append(
                pd.DataFrame(
                    {
                        "axis_cm1": raw_axis,
                        "instrument": instrument,
                        "view": view,
                        "q10": np.quantile(matrix, 0.1, axis=0),
                        "median": np.median(matrix, axis=0),
                        "q90": np.quantile(matrix, 0.9, axis=0),
                    }
                )
            )
    spectra = pd.concat(spectral_rows, ignore_index=True)

    preservation = preservation_by_instrument.rename(
        columns={
            "median_shape_correlation": "shape_correlation",
            "median_top_peak_recall_pm5cm1": "peak_recall",
        }
    )
    preservation["representation_label"] = preservation.representation_id.map(
        REPRESENTATION_LABELS
    )
    preservation["representation_order"] = preservation.representation_id.map(
        REPRESENTATION_ORDER
    )
    preservation = preservation.sort_values(["representation_order", "instrument"])
    pca = exploration["pca_diagnostics.csv"].copy()
    pca["representation_label"] = pca.representation_id.map(REPRESENTATION_LABELS)
    pca["representation_order"] = pca.representation_id.map(REPRESENTATION_ORDER)
    pca = pca.sort_values(["representation_order", "level"])
    embeddings = exploration["embedding_rows.csv"]
    embedding = embeddings[
        (embeddings.representation_id == "R_MIN_400_1800") & (embeddings.level == "master")
    ].copy()
    association_source = exploration["cluster_metadata_association.csv"]
    association = association_source[
        (association_source.level == "spectrum")
        & association_source.metadata_field.isin(["target_analyte", "instrument"])
        & association_source.method.str.startswith("kmeans")
    ].copy()
    association = association.assign(
        row_type="association",
        metric=association.metadata_field,
        value=association.normalized_mutual_information,
    )[["representation_id", "row_type", "metric", "value"]]
    stability_source = exploration["cluster_stability.csv"]
    stability = stability_source[stability_source.level == "spectrum"].copy()
    stability = stability.assign(
        row_type="stability",
        metric=stability.method,
        value=stability.median_stability,
    )[["representation_id", "row_type", "metric", "value"]]
    association_and_stability = pd.concat([association, stability], ignore_index=True)
    association_and_stability["representation_label"] = (
        association_and_stability.representation_id.map(REPRESENTATION_LABELS)
    )
    association_and_stability["representation_order"] = (
        association_and_stability.representation_id.map(REPRESENTATION_ORDER)
    )
    association_and_stability = association_and_stability.sort_values(
        ["row_type", "representation_order", "metric"]
    )

    return {
        "F02": (
            "Archive observation and sensor-tier flow",
            flow,
            [Panel("Tier flow", "stage", "count", "series", "Tier", "Observations", "bar")],
        ),
        "F03": (
            "Target by station by instrument support",
            support,
            [
                Panel(
                    "Spectrum support",
                    "station_target",
                    "spectra",
                    "instrument",
                    "Station and target",
                    "Spectra",
                    "bar",
                )
            ],
        ),
        "F04": (
            "Native and effective axis coverage",
            coverage,
            [
                Panel(
                    "Instrument support intervals",
                    "axis_cm1",
                    "instrument_index",
                    "series",
                    "Raman shift (cm-1)",
                    "Instrument",
                    "line",
                    trace_group="trace_group",
                    y_tick_labels="instrument",
                )
            ],
        ),
        "F05": (
            "Robust instrument spectral summaries",
            spectra,
            [
                Panel(
                    "Native-intensity median",
                    "axis_cm1",
                    "median",
                    "instrument",
                    "Raman shift (cm-1)",
                    "Median intensity (AU)",
                    "line",
                    ("view", "native intensity"),
                ),
                Panel(
                    "Row-scaled median",
                    "axis_cm1",
                    "median",
                    "instrument",
                    "Raman shift (cm-1)",
                    "Median scaled intensity",
                    "line",
                    ("view", "row min-max"),
                ),
            ],
        ),
        "F06": (
            "Preprocessing preservation by instrument",
            preservation,
            [
                Panel(
                    "Shape correlation",
                    "representation_label",
                    "shape_correlation",
                    "instrument",
                    "Representation",
                    "Median correlation",
                    "line",
                ),
                Panel(
                    "Peak recall",
                    "representation_label",
                    "peak_recall",
                    "instrument",
                    "Representation",
                    "Median recall",
                    "line",
                ),
            ],
        ),
        "F07": (
            "PCA structure before and after row scaling",
            pca,
            [
                Panel(
                    "PC1 variance",
                    "representation_label",
                    "pc1_explained_variance",
                    "level",
                    "Representation",
                    "Explained fraction",
                    "bar",
                ),
                Panel(
                    "Components to 95%",
                    "representation_label",
                    "pca_components_to_95pct",
                    "level",
                    "Representation",
                    "Components",
                    "bar",
                ),
            ],
        ),
        "F08": (
            "PCA, UMAP, and t-SNE physical-master views",
            embedding,
            [
                Panel(
                    "PCA",
                    "pca_1",
                    "pca_2",
                    "target_analyte",
                    "PC1",
                    "PC2",
                    "scatter",
                ),
                Panel(
                    "UMAP",
                    "umap_1",
                    "umap_2",
                    "target_analyte",
                    "UMAP 1",
                    "UMAP 2",
                    "scatter",
                ),
                Panel(
                    "t-SNE",
                    "tsne_1",
                    "tsne_2",
                    "target_analyte",
                    "t-SNE 1",
                    "t-SNE 2",
                    "scatter",
                ),
            ],
        ),
        "F09": (
            "Clustering stability and metadata association",
            association_and_stability,
            [
                Panel(
                    "K-means association",
                    "representation_label",
                    "value",
                    "metric",
                    "Representation",
                    "NMI",
                    "bar",
                    ("row_type", "association"),
                ),
                Panel(
                    "Seed and parameter stability",
                    "representation_label",
                    "value",
                    "metric",
                    "Representation",
                    "Median stability",
                    "bar",
                    ("row_type", "stability"),
                ),
            ],
        ),
    }


def generate_p01_figures(
    output_root: Path,
    tables: dict[str, tuple[str, pd.DataFrame, list[Panel]]],
) -> pd.DataFrame:
    """Write and compile all registered P01 figure forms from identical tables."""

    data_dir = output_root / "figures" / "data"
    tikz_dir = output_root / "figures" / "tikz"
    pdf_dir = output_root / "figures" / "pdf"
    html_dir = output_root / "figures" / "html"
    for directory in (data_dir, tikz_dir, pdf_dir, html_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for figure_id, (title, frame, panels) in tables.items():
        stem = f"{figure_id}_{FIGURE_SLUGS[figure_id]}"
        data_path = data_dir / f"{stem}.csv"
        frame.to_csv(data_path, index=False, lineterminator="\n")
        data_hash = sha256_file(data_path)
        figure = _plotly_figure(title, frame, panels)
        html_path = html_dir / f"{stem}.html"
        document = figure.to_html(
            include_plotlyjs=True,
            full_html=True,
            div_id=f"atlas-{figure_id.lower()}",
            config={"responsive": True, "scrollZoom": True, "displaylogo": False},
        )
        marker = (
            f"<!-- data_sha256={data_hash}; ATLAS {figure_id}; "
            f"description={html.escape(title)} -->\n"
        )
        html_path.write_text(document.replace("<head>", f"<head>\n{marker}", 1))
        tex_path = tikz_dir / f"{stem}.tex"
        tex_path.write_text(_tikz_source(figure_id, title, data_hash, frame, panels))
        pdf_path, log_path = _compile(tex_path, pdf_dir)
        rows.append(
            {
                "figure_id": figure_id,
                "title": title,
                "data_path": data_path.relative_to(output_root).as_posix(),
                "tikz_path": tex_path.relative_to(output_root).as_posix(),
                "pdf_path": pdf_path.relative_to(output_root).as_posix(),
                "html_path": html_path.relative_to(output_root).as_posix(),
                "log_path": log_path.relative_to(output_root).as_posix(),
                "data_sha256": data_hash,
                "tikz_sha256": sha256_file(tex_path),
                "pdf_sha256": sha256_file(pdf_path),
                "html_sha256": sha256_file(html_path),
                "semantic_parity": data_hash in tex_path.read_text()
                and data_hash in html_path.read_text(),
                "compiled": pdf_path.stat().st_size > 0,
                "native_tikz": "\\includegraphics" not in tex_path.read_text(),
                "standalone_html": "</html>" in html_path.read_text()[-2000:].lower(),
            }
        )
    return pd.DataFrame(rows).sort_values("figure_id").reset_index(drop=True)
