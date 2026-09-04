# ruff: noqa: E501
"""Create native TikZ, vector, raster, and standalone HTML P04 figures."""

from __future__ import annotations

import html
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from atlas_sers.governance.canonical import sha256_file

COLORS = {
    "D0-ERM": "#009E73",
    "C-SELECTED": "#0072B2",
    "C-RBF-SVM": "#E69F00",
    "C-RANDOM-FOREST": "#CC79A7",
    "C-EXTRA-TREES": "#56B4E9",
}


def _tex(value: object) -> str:
    result = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        result = result.replace(source, replacement)
    return result


def _compile(tex_path: Path, pdf_path: Path, png_path: Path, log_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="p04-figure-") as temporary_name:
        temporary = Path(temporary_name)
        local = temporary / tex_path.name
        shutil.copy2(tex_path, local)
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", local.name],
            cwd=temporary,
            capture_output=True,
            text=True,
            check=False,
        )
        log_path.write_text(result.stdout + "\n" + result.stderr)
        if result.returncode:
            raise RuntimeError(f"P04 TikZ compilation failed for {tex_path.name}.")
        shutil.copy2(temporary / f"{tex_path.stem}.pdf", pdf_path)
    subprocess.run(
        ["pdftocairo", "-png", "-singlefile", "-r", "300", pdf_path, png_path.with_suffix("")],
        capture_output=True,
        text=True,
        check=True,
    )
    log_path.unlink(missing_ok=True)


def _write_html(figure: go.Figure, path: Path, *, digest: str, description: str) -> None:
    body = figure.to_html(
        full_html=True,
        include_plotlyjs=True,
        div_id=f"nato-sers-{path.stem.lower()}",
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )
    body = body.replace("https://cdn.plot.ly/un/", "data:,")
    body = body.replace(
        "<head>",
        f"<head>\n<!-- data_sha256={digest}; description={html.escape(description)} -->",
        1,
    )
    path.write_text("\n".join(line.rstrip() for line in body.splitlines()) + "\n")


def _f19_semantic() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0, "Input", "1 x 1401", "1 x 1401", "-", "-", "-"],
            [1, "Stem", "1 x 1401", "24 x 1401", "11", "1", "GroupNorm + GELU"],
            [2, "Residual stage 1", "24 x 1401", "24 x 1401", "7", "1, 2", "2 residual blocks"],
            [3, "Transition 1", "24 x 1401", "48 x 701", "5 / stride 2", "1", "GroupNorm + GELU"],
            [4, "Residual stage 2", "48 x 701", "48 x 701", "7", "1, 2", "2 residual blocks"],
            [5, "Transition 2", "48 x 701", "64 x 351", "5 / stride 2", "1", "GroupNorm + GELU"],
            [6, "Residual stage 3", "64 x 351", "64 x 351", "5", "1, 2", "2 residual blocks"],
            [
                7,
                "Ordered adaptive mean",
                "64 x 351",
                "64 x 16",
                "16 bins",
                "-",
                "deterministic means",
            ],
            [8, "Projection", "1024", "96 then 64", "-", "-", "GELU + dropout 0.2"],
            [9, "Station-local head", "64", "3 logits", "linear", "-", "softmax after temperature"],
        ],
        columns=[
            "display_order",
            "component",
            "input_shape",
            "output_shape",
            "kernel_or_pool",
            "dilation",
            "operation",
        ],
    ).assign(
        figure_id="F19",
        research_question_id="RQ-P01",
        scope="S",
        model_id="D0-ERM",
        preprocessing_policy_id="PP-U-MIN",
        representation_id="R_MIN_400_1800",
        trainable_parameters=208691,
        maximum_parameters=250000,
    )


def _f19_tex(data: pd.DataFrame, digest: str) -> str:
    nodes = []
    arrows = []
    for row in data.itertuples(index=False):
        x = (row.display_order if row.display_order < 5 else 9 - row.display_order) * 3.2
        y = 0 if row.display_order < 5 else -3
        fill = (
            "atlasGreen!15"
            if row.component not in {"Input", "Station-local head"}
            else "atlasBlue!12"
        )
        nodes.append(
            rf"\node[layer,fill={fill}] (n{row.display_order}) at ({x},{y}) "
            rf"{{\textbf{{{_tex(row.component)}}}\\{_tex(row.output_shape)}\\{_tex(row.operation)}}};"
        )
        if row.display_order:
            arrows.append(rf"\draw[flow] (n{row.display_order - 1}) -- (n{row.display_order});")
    return rf"""\documentclass[tikz,border=5pt]{{standalone}}
\pdfinfoomitdate=1
\pdftrailerid{{}}
\pdfsuppressptexinfo=-1
\usepackage{{tikz}}
\usetikzlibrary{{arrows.meta,positioning}}
\definecolor{{atlasBlue}}{{HTML}}{{0072B2}}
\definecolor{{atlasGreen}}{{HTML}}{{009E73}}
% NATO SERS F19; data_sha256={digest}
% (S) RQ-P01; PP-U-MIN; R_MIN_400_1800; 598 spectra / 69 physical masters.
\begin{{document}}
\begin{{tikzpicture}}[x=1cm,y=1cm,layer/.style={{draw,rounded corners,text width=2.55cm,minimum height=1.75cm,align=center,font=\sffamily\fontsize{{8}}{{10}}\selectfont}},flow/.style={{-{{Latex[length=2mm]}},thick}}]
{chr(10).join(nodes)}
{chr(10).join(arrows)}
\node[font=\sffamily\bfseries,anchor=south] at (current bounding box.north) {{Compact location-preserving D0-ERM encoder}};
\node[font=\sffamily\fontsize{{8}}{{10}}\selectfont,anchor=north,align=center,text width=15.8cm] at ([yshift=-2mm]current bounding box.south) {{(S) RQ-P01; 208,691 parameters; one-convolution residual branches; GroupNorm; PP-U-MIN.\\598 spectra from 69 physical masters; 1,401 channels from 400--1,800 cm$^{{-1}}$; source-only fitting.}};
\end{{tikzpicture}}
\end{{document}}
"""


def _f19_html(data: pd.DataFrame, path: Path, digest: str) -> None:
    labels = [
        f"<b>{row.component}</b><br>{row.output_shape}<br>{row.operation}"
        for row in data.itertuples(index=False)
    ]
    xs = [index * 3.2 if index < 5 else (9 - index) * 3.2 for index in range(len(data))]
    ys = [0 if index < 5 else -3 for index in range(len(data))]
    custom = data[
        ["input_shape", "output_shape", "kernel_or_pool", "dilation", "operation"]
    ].to_numpy()
    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker={"opacity": 0, "size": 28},
            text=labels,
            textposition="middle center",
            textfont={"size": 12},
            customdata=custom,
            hovertemplate=(
                "%{text}<br>input=%{customdata[0]}<br>output=%{customdata[1]}"
                "<br>kernel/pool=%{customdata[2]}<br>dilation=%{customdata[3]}"
                "<br>%{customdata[4]}<extra></extra>"
            ),
        )
    )
    for index, (x, y) in enumerate(zip(xs, ys, strict=True)):
        fig.add_shape(
            type="rect",
            x0=x - 1.4,
            x1=x + 1.4,
            y0=y - 0.9,
            y1=y + 0.9,
            line={"color": "#0072B2"},
            fillcolor="#e0f3ec" if index not in (0, 9) else "#e0eef6",
            layer="below",
        )
        if index:
            previous_x, previous_y = xs[index - 1], ys[index - 1]
            direction = float(np.sign(x - previous_x))
            fig.add_annotation(
                x=x - direction * 1.4,
                y=y + (0.9 if direction == 0 else 0),
                ax=previous_x + direction * 1.4,
                ay=previous_y - (0.9 if direction == 0 else 0),
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",
                showarrow=True,
                arrowhead=2,
                arrowwidth=2,
                arrowcolor="#009E73",
            )
    fig.update_layout(
        title="Compact location-preserving D0-ERM encoder",
        template="plotly_white",
        width=1400,
        height=600,
        xaxis={"visible": False, "range": [-1.7, 14.5]},
        yaxis={"visible": False, "range": [-4.25, 1.25]},
        annotations=[
            *list(fig.layout.annotations),
            {
                "text": (
                    "(S) RQ-P01 · PP-U-MIN · R_MIN_400_1800 · 598 spectra from 69 "
                    "physical masters · 208,691 trainable parameters · no BatchNorm · source-only fitting"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.15,
                "showarrow": False,
            },
        ],
    )
    _write_html(fig, path, digest=digest, description="F19 compact D0 architecture")


def _f20_tex(data: pd.DataFrame, digest: str) -> str:
    panels = []
    for index, experiment in enumerate(("EXP-N00-DEV", "EXP-N00-T3")):
        rows = data[data.experiment_id.eq(experiment)]
        epoch = float(rows.selected_epoch_median.iloc[0])
        train = " ".join(
            f"({row.epoch},{row.training_balanced_accuracy_mean:.8g})"
            for row in rows.itertuples(index=False)
        )
        validation = " ".join(
            f"({row.epoch},{row.validation_balanced_accuracy_mean:.8g})"
            for row in rows.itertuples(index=False)
        )
        qlow = " ".join(
            f"({row.epoch},{row.validation_balanced_accuracy_q25:.8g})"
            for row in rows.itertuples(index=False)
        )
        qhigh = " ".join(
            f"({row.epoch},{row.validation_balanced_accuracy_q75:.8g})"
            for row in rows.itertuples(index=False)
        )
        legend = (
            "\n\\legend{Training mean,Validation mean,Validation Q25,Validation Q75}"
            if index == 0
            else ""
        )
        panels.append(
            rf"""\nextgroupplot[title={{{"A. Source development" if index == 0 else "B. T3 source-only selection"}}},xlabel={{Epoch}},ylabel={{{"Balanced accuracy" if index == 0 else ""}}},xmin=1,xmax=200,ymin=0,ymax=1.02]
\addplot[atlasGreen,thick] coordinates {{{train}}};
\addplot[atlasBlue,thick,dashed] coordinates {{{validation}}};
\addplot[atlasBlue!55,thin,dotted] coordinates {{{qlow}}};
\addplot[atlasBlue!55,thin,dashdotted] coordinates {{{qhigh}}};
\addplot[black,densely dashed] coordinates {{({epoch},0) ({epoch},1.02)}};
\addplot[black,dotted] coordinates {{(1,0.333333) (200,0.333333)}};
{legend}
"""
        )
    return rf"""\documentclass[tikz,border=5pt]{{standalone}}
\pdfinfoomitdate=1
\pdftrailerid{{}}
\pdfsuppressptexinfo=-1
\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots}}
\pgfplotsset{{compat=1.18}}
\definecolor{{atlasBlue}}{{HTML}}{{0072B2}}
\definecolor{{atlasGreen}}{{HTML}}{{009E73}}
% NATO SERS F20; data_sha256={digest}
% (S) RQ-P01; source-only validation; curves aggregate selected-candidate inner fits.
\begin{{document}}
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.2cm}},width=8.2cm,height=6.2cm,grid=major,tick label style={{font=\sffamily\fontsize{{8}}{{10}}\selectfont}},label style={{font=\sffamily\fontsize{{8}}{{10}}\selectfont}},title style={{font=\small\bfseries}},legend style={{font=\sffamily\fontsize{{8}}{{10}}\selectfont,at={{(1.04,-0.22)}},anchor=north}}]
{"".join(panels)}
\end{{groupplot}}
\node[font=\sffamily\bfseries,anchor=south] at (current bounding box.north) {{D0 learning curves and source-selected epochs}};
\node[font=\sffamily\fontsize{{8}}{{10}}\selectfont,anchor=north,align=center,text width=16cm] at ([yshift=-3mm]current bounding box.south) {{(S) RQ-P01; PP-U-MIN; source-only validation; 598 spectra / 69 physical masters; 400--1,800 cm$^{{-1}}$.\\Vertical line: median selected checkpoint epoch. Later epochs include only still-running fits.\\Quartiles describe fit variation, not confidence intervals. Horizontal dotted line: three-class chance.}};
\end{{tikzpicture}}
\end{{document}}
"""


def _f20_html(data: pd.DataFrame, path: Path, digest: str) -> None:
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("A. Source development", "B. T3 source-only selection")
    )
    for column, experiment in enumerate(("EXP-N00-DEV", "EXP-N00-T3"), start=1):
        rows = data[data.experiment_id.eq(experiment)]
        for field, name, color, dash in (
            ("training_balanced_accuracy_mean", "Training mean", "#009E73", "solid"),
            ("validation_balanced_accuracy_mean", "Validation mean", "#0072B2", "dash"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=rows.epoch,
                    y=rows[field],
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=column == 1,
                    line={"color": color, "dash": dash},
                    customdata=rows[["run_count"]],
                    hovertemplate="epoch=%{x}<br>BA=%{y:.3f}<br>active fits=%{customdata[0]}<extra></extra>",
                ),
                row=1,
                col=column,
            )
        fig.add_trace(
            go.Scatter(
                x=pd.concat([rows.epoch, rows.epoch.iloc[::-1]]),
                y=pd.concat(
                    [
                        rows.validation_balanced_accuracy_q75,
                        rows.validation_balanced_accuracy_q25.iloc[::-1],
                    ]
                ),
                fill="toself",
                fillcolor="rgba(0,114,178,0.15)",
                line={"color": "rgba(0,0,0,0)"},
                name="Validation IQR",
                showlegend=column == 1,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )
        median = float(rows.selected_epoch_median.iloc[0])
        fig.add_vline(x=median, line_dash="dash", line_color="black", row=1, col=column)
        fig.add_hline(y=1 / 3, line_dash="dot", line_color="black", row=1, col=column)
    fig.update_xaxes(title_text="Epoch", range=[1, 200])
    fig.update_yaxes(range=[0, 1.02])
    fig.update_yaxes(title_text="Balanced accuracy", row=1, col=1)
    fig.update_layout(
        title="D0 learning curves and source-selected epochs",
        template="plotly_white",
        width=1400,
        height=700,
        margin={"b": 180},
        legend={"orientation": "h", "y": -0.14},
        annotations=[
            *list(fig.layout.annotations),
            {
                "text": (
                    "(S) RQ-P01 · PP-U-MIN · 598 spectra / 69 physical masters · 400–1,800 cm⁻¹<br>Source-only inner validation; "
                    "shading is the fit-level IQR; dotted line is 3-class chance; vertical line is median selected epoch"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.28,
                "showarrow": False,
            },
        ],
    )
    fig.add_annotation(
        text="Only fits still running contribute at later epochs; hover shows active-fit counts.",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.39,
        showarrow=False,
    )
    _write_html(fig, path, digest=digest, description="F20 D0 source-only learning curves")


def _f48_tex(data: pd.DataFrame, digest: str) -> str:
    semantic = data[(data.aggregation_id.eq("M01")) & ~data.domain.eq("__overall__")].copy()
    domains = sorted(semantic.domain.unique())
    y = {domain: len(domains) - index - 1 for index, domain in enumerate(domains)}
    labels = ",".join(_tex(value) for value in domains)
    ticks = ",".join(str(y[value]) for value in domains)
    extent = max(
        0.72, float(np.ceil(semantic[["lower_95", "upper_95"]].abs().max().max() * 20) / 20) + 0.05
    )
    panels = []
    colors = {
        "C-SELECTED": "atlasBlue",
        "C-RBF-SVM": "atlasOrange",
        "C-RANDOM-FOREST": "atlasPurple",
        "C-EXTRA-TREES": "atlasSky",
    }
    for index, model in enumerate(CLASSICAL_ORDER):
        rows = semantic[semantic.comparison_model_id.eq(model)]
        coordinates = " ".join(
            f"({row.estimate_d0_minus_classical_ba:.8g},{y[row.domain]}) += ({row.upper_95 - row.estimate_d0_minus_classical_ba:.8g},0) -= ({row.estimate_d0_minus_classical_ba - row.lower_95:.8g},0)"
            for row in rows.itertuples(index=False)
        )
        ylabels = f"yticklabels={{{labels}}}" if index % 2 == 0 else "yticklabels={}"
        xlabel = "D0 minus classical BA" if index >= 2 else ""
        panels.append(
            rf"""\nextgroupplot[title={{{chr(65 + index)}. {_tex(model)}}},xlabel={{{xlabel}}},xmin={-extent},xmax={extent},ymin=-1,ymax={len(domains)},ytick={{{ticks}}},{ylabels}]
\addplot+[only marks,mark=triangle*,color={colors[model]},mark options={{fill={colors[model]},draw={colors[model]}}},error bars/.cd,x dir=both,x explicit] coordinates {{{coordinates}}};
\addplot[black,dashdotted] coordinates {{(0,-1) (0,{len(domains)})}};
"""
        )
    return rf"""\documentclass[tikz,border=5pt]{{standalone}}
\pdfinfoomitdate=1
\pdftrailerid{{}}
\pdfsuppressptexinfo=-1
\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots}}
\pgfplotsset{{compat=1.18}}
\definecolor{{atlasBlue}}{{HTML}}{{0072B2}}
\definecolor{{atlasOrange}}{{HTML}}{{E69F00}}
\definecolor{{atlasPurple}}{{HTML}}{{CC79A7}}
\definecolor{{atlasSky}}{{HTML}}{{56B4E9}}
% NATO SERS F48; data_sha256={digest}
% (P) RQ-P01; M01 spectrum BA; points are domains; intervals resample physical masters.
\begin{{document}}
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 2,horizontal sep=0.7cm,vertical sep=1.2cm}},width=8.0cm,height=8.2cm,grid=major,tick label style={{font=\sffamily\fontsize{{8}}{{10}}\selectfont}},label style={{font=\sffamily\fontsize{{8}}{{10}}\selectfont}},title style={{font=\small\bfseries}},y tick label style={{align=right}}]
{"".join(panels)}
\end{{groupplot}}
\node[font=\sffamily\bfseries,anchor=south] at (current bounding box.north) {{Compact D0 versus classical methods across held instruments}};
\node[font=\sffamily\fontsize{{8}}{{10}}\selectfont,anchor=north,align=center,text width=16cm] at ([yshift=-3mm]current bounding box.south) {{(P) RQ-P01; PP-U-MIN; source-only fitting; 598 spectra / 69 physical masters; 400--1,800 cm$^{{-1}}$.\\Triangles: pooled out-of-fold spectrum-BA differences in 13 held-instrument domains.\\Intervals: 95\% paired master bootstrap (5,000 draws); zero: no difference; positive: D0 favoured.}};
\end{{tikzpicture}}
\end{{document}}
"""


CLASSICAL_ORDER = ("C-SELECTED", "C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES")


def _f48_html(data: pd.DataFrame, path: Path, digest: str) -> None:
    semantic = data[(data.aggregation_id.eq("M01")) & ~data.domain.eq("__overall__")].copy()
    extent = max(
        0.72, float(np.ceil(semantic[["lower_95", "upper_95"]].abs().max().max() * 20) / 20) + 0.05
    )
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        subplot_titles=[
            f"{chr(65 + index)}. {model}" for index, model in enumerate(CLASSICAL_ORDER)
        ],
    )
    for index, model in enumerate(CLASSICAL_ORDER):
        row = index // 2 + 1
        column = index % 2 + 1
        values = semantic[semantic.comparison_model_id.eq(model)].sort_values("domain")
        fig.add_trace(
            go.Scatter(
                x=values.estimate_d0_minus_classical_ba,
                y=values.domain,
                mode="markers",
                name=model,
                showlegend=False,
                marker={"symbol": "triangle-up", "color": COLORS[model], "size": 8},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": values.upper_95 - values.estimate_d0_minus_classical_ba,
                    "arrayminus": values.estimate_d0_minus_classical_ba - values.lower_95,
                },
                customdata=values[
                    ["independent_physical_masters", "bootstrap_draws", "lower_95", "upper_95"]
                ],
                hovertemplate=(
                    "%{y}<br>D0 - classical BA=%{x:.3f}<br>95% CI=[%{customdata[2]:.3f}, %{customdata[3]:.3f}]"
                    "<br>independent masters=%{customdata[0]}<br>bootstrap draws=%{customdata[1]}<extra></extra>"
                ),
            ),
            row=row,
            col=column,
        )
        fig.add_vline(x=0, line_dash="dashdot", line_color="black", row=row, col=column)
    fig.update_xaxes(title_text="D0 minus classical balanced accuracy", range=[-extent, extent])
    fig.update_yaxes(
        autorange="reversed", categoryorder="array", categoryarray=sorted(semantic.domain.unique())
    )
    fig.update_layout(
        title="Compact D0 versus classical methods across held instruments",
        template="plotly_white",
        width=1500,
        height=1100,
        margin={"b": 170},
        annotations=[
            *list(fig.layout.annotations),
            {
                "text": (
                    "(P) RQ-P01 · PP-U-MIN · source-only fitting · 598 spectra / 69 physical masters · 400–1,800 cm⁻¹<br>"
                    "Triangles: M01 spectrum-BA differences in 13 held-instrument domains; 95% paired master bootstrap (5,000 draws).<br>"
                    "Zero: no difference; positive: D0 favoured. Intervals condition on the observed domains and retained class support."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.14,
                "showarrow": False,
            },
        ],
    )
    _write_html(fig, path, digest=digest, description="F48 paired D0/classical held-domain effects")


def generate_p04_figures(
    *,
    learning_curves: pd.DataFrame,
    selected_epochs: pd.DataFrame,
    bootstrap_effects: pd.DataFrame,
    results_root: Path,
    plan_root: Path,
) -> pd.DataFrame:
    semantic_root = results_root / "semantic"
    semantic_root.mkdir(parents=True, exist_ok=True)
    for directory in ("data", "tikz", "html", "pdf", "png", "logs"):
        (plan_root / "figures" / directory).mkdir(parents=True, exist_ok=True)
    curve_semantic = learning_curves.merge(
        selected_epochs,
        on="experiment_id",
        how="left",
        validate="many_to_one",
    )
    common_metadata = {
        "research_question_id": "RQ-P01",
        "population_id": "primary_598",
        "population_spectra": 598,
        "population_physical_masters": 69,
        "preprocessing_policy_id": "PP-U-MIN",
        "representation_id": "R_MIN_400_1800",
        "information_regime": "source_only",
    }
    curve_semantic = curve_semantic.assign(figure_id="F20", scope="S", **common_metadata)
    effect_semantic = bootstrap_effects.assign(figure_id="F48", scope="P", **common_metadata)
    figure_inputs: list[tuple[str, pd.DataFrame]] = [
        ("F19_deep_architecture", _f19_semantic()),
        ("F20_learning_curves", curve_semantic),
        ("F48_deep_classical_comparison", effect_semantic),
    ]
    records = []
    for stem, semantic in figure_inputs:
        semantic_path = semantic_root / f"{stem}.csv"
        semantic.to_csv(semantic_path, index=False, lineterminator="\n", float_format="%.12g")
        data_path = plan_root / "figures/data" / f"{stem}.csv"
        data_path.write_bytes(semantic_path.read_bytes())
        digest = sha256_file(data_path)
        tex_path = plan_root / "figures/tikz" / f"{stem}.tex"
        html_path = plan_root / "figures/html" / f"{stem}.html"
        pdf_path = plan_root / "figures/pdf" / f"{stem}.pdf"
        png_path = plan_root / "figures/png" / f"{stem}.png"
        log_path = plan_root / "figures/logs" / f"{stem}.pdflatex.log"
        if stem.startswith("F19"):
            tex_path.write_text(_f19_tex(semantic, digest))
            _f19_html(semantic, html_path, digest)
            title = "Compact encoder architecture"
        elif stem.startswith("F20"):
            tex_path.write_text(_f20_tex(semantic, digest))
            _f20_html(semantic, html_path, digest)
            title = "Deep learning curves and selected epochs"
        else:
            tex_path.write_text(_f48_tex(semantic, digest))
            _f48_html(semantic, html_path, digest)
            title = "Compact D0 versus classical held-instrument performance"
        _compile(tex_path, pdf_path, png_path, log_path)
        if digest not in tex_path.read_text() or digest not in html_path.read_text():
            raise RuntimeError(f"{stem} semantic data hash parity failed.")
        records.append(
            {
                "figure_id": stem.split("_", 1)[0],
                "title": title,
                "semantic_sha256": digest,
                "semantic_path": semantic_path.relative_to(results_root.parent.parent).as_posix(),
                "tikz_sha256": sha256_file(tex_path),
                "pdf_sha256": sha256_file(pdf_path),
                "png_sha256": sha256_file(png_path),
                "html_sha256": sha256_file(html_path),
            }
        )
    return pd.DataFrame(records)
