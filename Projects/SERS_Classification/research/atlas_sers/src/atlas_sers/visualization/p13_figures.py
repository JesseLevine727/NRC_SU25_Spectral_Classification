# ruff: noqa: E501
"""Generate native TikZ and interactive P13 evidence figures from semantic tables."""

from __future__ import annotations

import html
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from atlas_sers.governance.canonical import sha256_file

COLORS = {
    "H_SERS_H_Kit": "#0072B2",
    "NRC_Canadian_SERS": "#E69F00",
    "pSERS_Metrohm_silver": "#009E73",
    "GaN_polymer": "#CC79A7",
}
TIKZ_COLORS = {
    "H_SERS_H_Kit": "atlasBlue",
    "NRC_Canadian_SERS": "atlasOrange",
    "pSERS_Metrohm_silver": "atlasGreen",
    "GaN_polymer": "atlasPurple",
}


@dataclass(frozen=True)
class FigurePayload:
    figure_id: str
    slug: str
    title: str
    caption: str
    semantic: pd.DataFrame
    tex: str
    interactive: go.Figure


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


def _header(*, figure_id: str, title: str, caption: str, digest: str) -> str:
    return rf"""\documentclass[tikz,border=5pt]{{standalone}}
\pdfinfoomitdate=1
\pdftrailerid{{}}
\pdfsuppressptexinfo=-1
\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots}}
\pgfplotsset{{compat=1.18}}
\definecolor{{atlasBlue}}{{HTML}}{{0072B2}}
\definecolor{{atlasOrange}}{{HTML}}{{E69F00}}
\definecolor{{atlasGreen}}{{HTML}}{{009E73}}
\definecolor{{atlasPurple}}{{HTML}}{{CC79A7}}
\definecolor{{atlasGray}}{{HTML}}{{777777}}
% NATO SERS figure {figure_id}; data_sha256={digest}
% title={_tex(title)}
% caption={_tex(caption)}
\begin{{document}}
"""


def _footer(title: str) -> str:
    return rf"""\node[font=\normalsize\bfseries,anchor=south] at (current bounding box.north) {{{_tex(title)}}};
\end{{tikzpicture}}
\end{{document}}
"""


def _domain_label(row: pd.Series) -> str:
    return f"{row.station} / {row.substrate_family} / {row.held_instrument}"


def _f45(domain_claims: pd.DataFrame, digest: str) -> FigurePayload:
    semantic = domain_claims[
        domain_claims.support_tier.astype(str).isin(
            ["confirmatory", "exploratory_low_support"]
        )
    ].copy()
    semantic["domain_label"] = semantic.apply(_domain_label, axis=1)
    semantic = semantic.sort_values(
        ["substrate_family", "station", "held_instrument"], kind="stable"
    ).reset_index(drop=True)
    semantic["plot_y"] = np.arange(len(semantic))[::-1]
    labels = ",".join(_tex(value) for value in semantic.domain_label)
    ticks = ",".join(str(value) for value in semantic.plot_y)
    caption = (
        "C-SELECTED with PP-U-MIN. Points and horizontal bars are estimates and 95% "
        "master-clustered bootstrap intervals. Portability requires held balanced-accuracy "
        "LCB at least 0.60 and matched source-minus-held loss UCB at most 0.10. Open gray "
        "marks at the plot boundary are predeclared unavailable endpoints. n is physical masters."
    )
    tex = _header(
        figure_id="F45",
        title="Held-instrument analyte recoverability by substrate",
        caption=caption,
        digest=digest,
    )
    tex += rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=2.0cm}},width=8.4cm,height=10.8cm,
tick label style={{font=\scriptsize}},label style={{font=\small}},title style={{font=\small\bfseries}},
ytick={{{ticks}}},yticklabels={{{labels}}},y tick label style={{font=\tiny,align=right}},grid=major]
\nextgroupplot[title={{A. Held-instrument recovery}},xlabel={{Balanced accuracy}},xmin=-0.08,xmax=1.02,ymin=-1,ymax={len(semantic)},xmajorgrids=true]
"""
    for substrate, group in semantic.groupby("substrate_family", sort=True):
        available = group[group.endpoint_status.astype(str).eq("complete")]
        coordinates = " ".join(
            f"({row.held_balanced_accuracy:.8g},{int(row.plot_y)}) += "
            f"({row.held_ucb95-row.held_balanced_accuracy:.8g},0) -= "
            f"({row.held_balanced_accuracy-row.held_lcb95:.8g},0)"
            for row in available.itertuples(index=False)
        )
        tex += (
            rf"\addplot+[only marks,mark=*,color={TIKZ_COLORS[str(substrate)]},"
            rf"error bars/.cd,x dir=both,x explicit] coordinates {{{coordinates}}};"
            "\n"
        )
        tex += rf"\addlegendentry{{{_tex(substrate)}}}" + "\n"
    unavailable = semantic[semantic.endpoint_status.astype(str).ne("complete")]
    if not unavailable.empty:
        coordinates = " ".join(f"(-0.05,{int(row.plot_y)})" for row in unavailable.itertuples())
        tex += rf"\addplot+[only marks,mark=o,color=atlasGray] coordinates {{{coordinates}}};\addlegendentry{{unavailable}}" + "\n"
    tex += rf"\addplot[black,dashdotted] coordinates {{(0.6,-1) (0.6,{len(semantic)})}};" + "\n"
    tex += rf"""\nextgroupplot[title={{B. Matched acquisition loss}},xlabel={{Source minus held balanced accuracy}},xmin=-1.08,xmax=1.02,ymin=-1,ymax={len(semantic)},yticklabels={{}},xmajorgrids=true]
"""
    for substrate, group in semantic.groupby("substrate_family", sort=True):
        available = group[group.endpoint_status.astype(str).eq("complete")]
        coordinates = " ".join(
            f"({row.source_minus_held_balanced_accuracy:.8g},{int(row.plot_y)}) += "
            f"({row.loss_ucb95-row.source_minus_held_balanced_accuracy:.8g},0) -= "
            f"({row.source_minus_held_balanced_accuracy-row.loss_lcb95:.8g},0)"
            for row in available.itertuples(index=False)
        )
        tex += (
            rf"\addplot+[only marks,mark=square*,color={TIKZ_COLORS[str(substrate)]},"
            rf"error bars/.cd,x dir=both,x explicit] coordinates {{{coordinates}}};"
            "\n"
        )
    if not unavailable.empty:
        coordinates = " ".join(f"(-1.03,{int(row.plot_y)})" for row in unavailable.itertuples())
        tex += rf"\addplot+[only marks,mark=o,color=atlasGray] coordinates {{{coordinates}}};" + "\n"
    tex += rf"\addplot[black,dashdotted] coordinates {{(0.1,-1) (0.1,{len(semantic)})}};" + "\n"
    tex += "\\end{groupplot}\n" + _footer(
        "Held-instrument analyte recoverability by substrate"
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("A. Held-instrument recovery", "B. Matched acquisition loss"),
        horizontal_spacing=0.08,
    )
    for substrate, group in semantic.groupby("substrate_family", sort=True):
        available = group[group.endpoint_status.astype(str).eq("complete")]
        custom = np.column_stack(
            [
                available.support_tier,
                available.held_masters,
                available.matched_masters,
                available.completion_state,
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=available.held_balanced_accuracy,
                y=available.domain_label,
                mode="markers",
                name=str(substrate),
                marker={"color": COLORS[str(substrate)], "symbol": "circle", "size": 8},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": available.held_ucb95 - available.held_balanced_accuracy,
                    "arrayminus": available.held_balanced_accuracy - available.held_lcb95,
                },
                customdata=custom,
                hovertemplate=(
                    "%{y}<br>Held BA=%{x:.3f}<br>tier=%{customdata[0]}"
                    "<br>n masters=%{customdata[1]}<br>state=%{customdata[3]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=available.source_minus_held_balanced_accuracy,
                y=available.domain_label,
                mode="markers",
                name=str(substrate),
                showlegend=False,
                marker={"color": COLORS[str(substrate)], "symbol": "square", "size": 8},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": available.loss_ucb95
                    - available.source_minus_held_balanced_accuracy,
                    "arrayminus": available.source_minus_held_balanced_accuracy
                    - available.loss_lcb95,
                },
                customdata=custom,
                hovertemplate=(
                    "%{y}<br>Source-held loss=%{x:.3f}<br>tier=%{customdata[0]}"
                    "<br>matched n=%{customdata[2]}<br>state=%{customdata[3]}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    if not unavailable.empty:
        unavailable_custom = unavailable[
            ["support_tier", "completion_state"]
        ].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=np.full(len(unavailable), -0.01),
                y=unavailable.domain_label,
                mode="markers",
                name="Unavailable",
                marker={"color": "#777777", "symbol": "circle-open", "size": 9},
                customdata=unavailable_custom,
                hovertemplate=(
                    "%{y}<br>unavailable endpoint<br>tier=%{customdata[0]}"
                    "<br>state=%{customdata[1]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.full(len(unavailable), -1.01),
                y=unavailable.domain_label,
                mode="markers",
                name="Unavailable",
                showlegend=False,
                marker={"color": "#777777", "symbol": "circle-open", "size": 9},
                customdata=unavailable_custom,
                hovertemplate=(
                    "%{y}<br>unavailable endpoint<br>tier=%{customdata[0]}"
                    "<br>state=%{customdata[1]}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    fig.add_vline(x=0.6, line_dash="dashdot", line_color="black", row=1, col=1)
    fig.add_vline(x=0.1, line_dash="dashdot", line_color="black", row=1, col=2)
    fig.update_xaxes(title_text="Balanced accuracy", range=[-0.02, 1.02], row=1, col=1)
    fig.update_xaxes(
        title_text="Source minus held balanced accuracy", range=[-1.02, 1.02], row=1, col=2
    )
    fig.update_layout(
        title="Held-instrument analyte recoverability by substrate",
        template="plotly_white",
        height=800,
        width=1450,
        font={"family": "Arial, sans-serif", "size": 12},
        legend={"orientation": "h", "y": -0.1},
    )
    return FigurePayload("F45", "substrate_recoverability", fig.layout.title.text, caption, semantic, tex, fig)


def _f46(crossover: pd.DataFrame, digest: str) -> FigurePayload:
    semantic = crossover[
        crossover.procedure_id.astype(str).isin(["C-SELECTED", "C-RBF-SVM"])
    ].copy()
    semantic = semantic.sort_values(
        ["support_tier", "crossover_block_id", "procedure_id"], kind="stable"
    ).reset_index(drop=True)
    blocks = crossover.drop_duplicates("crossover_block_id").sort_values(
        ["support_tier", "crossover_block_id"], kind="stable"
    )
    block_order = blocks.crossover_block_id.astype(str).tolist()
    y_lookup = {block: len(block_order) - index - 1 for index, block in enumerate(block_order)}
    semantic["plot_y"] = semantic.crossover_block_id.astype(str).map(y_lookup)
    labels = ",".join(_tex(value) for value in block_order)
    ticks = ",".join(str(y_lookup[value]) for value in block_order)
    caption = (
        "Same-master difference-in-differences, oriented as the substrate-B effect at instrument "
        "B minus at instrument A. Circles are C-SELECTED and squares fixed RBF SVM; horizontal "
        "bars are 95% master-bootstrap intervals. The distance panel is PP-U-MIN cosine-distance "
        "difference (substrate B minus A). Singleton blocks have point estimates only; unavailable "
        "predictive blocks remain visible in the semantic table."
    )
    title = "Same-master substrate by instrument crossover effects"
    tex = _header(figure_id="F46", title=title, caption=caption, digest=digest)
    tex += rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=3 by 1,horizontal sep=1.15cm}},width=6.1cm,height=14.5cm,
tick label style={{font=\tiny}},label style={{font=\scriptsize}},title style={{font=\scriptsize\bfseries}},
ytick={{{ticks}}},yticklabels={{{labels}}},y tick label style={{font=\tiny}},grid=major]
"""
    panels = (
        ("correctness_interaction", "A. Correctness interaction", "Correctness DiD"),
        ("true_probability_interaction", "B. True-class probability", "Probability DiD"),
    )
    for panel_index, (field, panel_title, x_label) in enumerate(panels):
        ylabels = "" if panel_index else f"yticklabels={{{labels}}},"
        tex += rf"\nextgroupplot[title={{{panel_title}}},xlabel={{{x_label}}},xmin=-2.08,xmax=2.08,ymin=-1,ymax={len(block_order)},{ylabels}]" + "\n"
        for procedure, marker, color in (
            ("C-SELECTED", "*", "atlasBlue"),
            ("C-RBF-SVM", "square*", "atlasOrange"),
        ):
            rows = semantic[
                semantic.procedure_id.astype(str).eq(procedure)
                & semantic.predictive_status.astype(str).eq("complete")
            ]
            coords = " ".join(
                f"({getattr(row, field):.8g},{int(row.plot_y)}) += "
                f"({getattr(row, field + '_upper_95')-getattr(row, field):.8g},0) -= "
                f"({getattr(row, field)-getattr(row, field + '_lower_95'):.8g},0)"
                if np.isfinite(getattr(row, field + "_lower_95"))
                else f"({getattr(row, field):.8g},{int(row.plot_y)})"
                for row in rows.itertuples(index=False)
            )
            tex += rf"\addplot+[only marks,mark={marker},color={color},error bars/.cd,x dir=both,x explicit] coordinates {{{coords}}};" + "\n"
            if panel_index == 0:
                tex += rf"\addlegendentry{{{_tex(procedure)}}}" + "\n"
        tex += rf"\addplot[black,dashdotted] coordinates {{(0,-1) (0,{len(block_order)})}};" + "\n"
    distance = semantic.drop_duplicates("crossover_block_id")
    tex += rf"\nextgroupplot[title={{C. Acquisition distance}},xlabel={{Cosine-distance B minus A}},xmin=-1.08,xmax=1.08,ymin=-1,ymax={len(block_order)},yticklabels={{}}]" + "\n"
    coords = " ".join(
        f"({row.representation_distance_difference:.8g},{int(row.plot_y)}) += "
        f"({row.representation_distance_upper_95-row.representation_distance_difference:.8g},0) -= "
        f"({row.representation_distance_difference-row.representation_distance_lower_95:.8g},0)"
        if np.isfinite(row.representation_distance_lower_95)
        else f"({row.representation_distance_difference:.8g},{int(row.plot_y)})"
        for row in distance.itertuples(index=False)
    )
    tex += rf"\addplot+[only marks,mark=diamond*,color=atlasGreen,error bars/.cd,x dir=both,x explicit] coordinates {{{coords}}};" + "\n"
    tex += rf"\addplot[black,dashdotted] coordinates {{(0,-1) (0,{len(block_order)})}};" + "\n"
    tex += "\\end{groupplot}\n" + _footer(title)

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=(
            "A. Correctness interaction",
            "B. True-class probability interaction",
            "C. Acquisition distance difference",
        ),
        horizontal_spacing=0.05,
    )
    for procedure, symbol, color in (
        ("C-SELECTED", "circle", "#0072B2"),
        ("C-RBF-SVM", "square", "#E69F00"),
    ):
        rows = semantic[
            semantic.procedure_id.astype(str).eq(procedure)
            & semantic.predictive_status.astype(str).eq("complete")
        ]
        custom = rows[
            ["station", "target_analyte", "support_tier", "physical_masters"]
        ].to_numpy()
        for column, field in enumerate(
            ("correctness_interaction", "true_probability_interaction"), start=1
        ):
            fig.add_trace(
                go.Scatter(
                    x=rows[field],
                    y=rows.crossover_block_id,
                    mode="markers",
                    name=procedure,
                    legendgroup=procedure,
                    showlegend=column == 1,
                    marker={"symbol": symbol, "color": color, "size": 8},
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": rows[f"{field}_upper_95"] - rows[field],
                        "arrayminus": rows[field] - rows[f"{field}_lower_95"],
                    },
                    customdata=custom,
                    hovertemplate=(
                        "%{y}<br>effect=%{x:.3f}<br>station=%{customdata[0]}"
                        "<br>analyte=%{customdata[1]}<br>tier=%{customdata[2]}"
                        "<br>n=%{customdata[3]}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
    distance = semantic.drop_duplicates("crossover_block_id")
    fig.add_trace(
        go.Scatter(
            x=distance.representation_distance_difference,
            y=distance.crossover_block_id,
            mode="markers",
            name="PP-U-MIN distance",
            marker={"symbol": "diamond", "color": "#009E73", "size": 8},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": distance.representation_distance_upper_95
                - distance.representation_distance_difference,
                "arrayminus": distance.representation_distance_difference
                - distance.representation_distance_lower_95,
            },
            hovertemplate="%{y}<br>distance difference=%{x:.4f}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    for column in (1, 2, 3):
        fig.add_vline(x=0, line_dash="dashdot", line_color="black", row=1, col=column)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=950,
        width=1700,
        font={"family": "Arial, sans-serif", "size": 11},
        legend={"orientation": "h", "y": -0.08},
    )
    return FigurePayload("F46", "substrate_instrument_crossover", title, caption, semantic, tex, fig)


def _f47(field: pd.DataFrame, digest: str) -> FigurePayload:
    semantic = field[field.procedure_id.astype(str).eq("C-SELECTED")].copy()
    semantic["row_label"] = semantic.substrate_family.astype(str) + " / " + semantic.endpoint_type.astype(str)
    semantic = semantic.sort_values(["substrate_family", "endpoint_type"], kind="stable").reset_index(drop=True)
    semantic["plot_y"] = np.arange(len(semantic))[::-1]
    labels = ",".join(_tex(value) for value in semantic.row_label)
    ticks = ",".join(str(value) for value in semantic.plot_y)
    caption = (
        "Recorded field-log outcomes at the master-substrate-instrument-view unit for C-SELECTED "
        "PP-U-MIN held predictions. Points are complete-case recorded detection/specificity; "
        "horizontal bars are worst-to-best bounds for genuinely missing logs, not confidence "
        "intervals. M and conflicting repeats are excluded from the definite endpoint. Agreement "
        "compares recorded success/failure with correct/incorrect analyte classification."
    )
    title = "Recorded field-trial detection completeness and model agreement"
    tex = _header(figure_id="F47", title=title, caption=caption, digest=digest)
    tex += rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=2.0cm}},width=8.2cm,height=6.4cm,
tick label style={{font=\scriptsize}},label style={{font=\small}},title style={{font=\small\bfseries}},
ytick={{{ticks}}},yticklabels={{{labels}}},y tick label style={{font=\tiny,align=right}},grid=major]
\nextgroupplot[title={{A. Recorded success}},xlabel={{Proportion}},xmin=-0.02,xmax=1.02,ymin=-1,ymax={len(semantic)}]
"""
    for substrate, group in semantic.groupby("substrate_family", sort=True):
        coords = " ".join(
            f"({row.complete_case_estimate:.8g},{int(row.plot_y)}) += "
            f"({row.best_case_missing_bound-row.complete_case_estimate:.8g},0) -= "
            f"({row.complete_case_estimate-row.worst_case_missing_bound:.8g},0)"
            for row in group.itertuples(index=False)
        )
        tex += rf"\addplot+[only marks,mark=*,color={TIKZ_COLORS[str(substrate)]},error bars/.cd,x dir=both,x explicit] coordinates {{{coords}}};\addlegendentry{{{_tex(substrate)}}}" + "\n"
    tex += rf"\nextgroupplot[title={{B. Model--field agreement}},xlabel={{Agreement proportion}},xmin=-0.02,xmax=1.02,ymin=-1,ymax={len(semantic)},yticklabels={{}}]" + "\n"
    for substrate, group in semantic.groupby("substrate_family", sort=True):
        coords = " ".join(
            f"({row.model_field_agreement:.8g},{int(row.plot_y)})"
            for row in group.itertuples(index=False)
        )
        tex += rf"\addplot+[only marks,mark=square*,color={TIKZ_COLORS[str(substrate)]}] coordinates {{{coords}}};" + "\n"
    tex += "\\end{groupplot}\n" + _footer(title)

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("A. Recorded success", "B. Model–field agreement"),
    )
    for substrate, group in semantic.groupby("substrate_family", sort=True):
        custom = group[
            [
                "definite_views",
                "missing_views",
                "ambiguous_or_conflicting_views",
                "successful_views",
                "model_prediction_views",
                "model_field_agreement_views",
            ]
        ].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=group.complete_case_estimate,
                y=group.row_label,
                mode="markers",
                name=str(substrate),
                marker={"color": COLORS[str(substrate)], "symbol": "circle", "size": 9},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": group.best_case_missing_bound - group.complete_case_estimate,
                    "arrayminus": group.complete_case_estimate - group.worst_case_missing_bound,
                },
                customdata=custom,
                hovertemplate=(
                    "%{y}<br>complete-case=%{x:.3f}<br>definite=%{customdata[0]}"
                    "<br>missing=%{customdata[1]}<br>ambiguous=%{customdata[2]}"
                    "<br>successes=%{customdata[3]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=group.model_field_agreement,
                y=group.row_label,
                mode="markers",
                name=str(substrate),
                showlegend=False,
                marker={"color": COLORS[str(substrate)], "symbol": "square", "size": 9},
                customdata=custom,
                hovertemplate=(
                    "%{y}<br>agreement=%{x:.3f}<br>model predictions=%{customdata[4]}"
                    "<br>agreement views=%{customdata[5]}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(range=[-0.02, 1.02], title_text="Proportion", row=1, col=1)
    fig.update_xaxes(range=[-0.02, 1.02], title_text="Agreement proportion", row=1, col=2)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=600,
        width=1400,
        font={"family": "Arial, sans-serif", "size": 12},
        legend={"orientation": "h", "y": -0.18},
    )
    return FigurePayload("F47", "recorded_detection_agreement", title, caption, semantic, tex, fig)


def _compile_tex(tex_path: Path, pdf_path: Path, png_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="p13-figure-") as temporary_name:
        temporary = Path(temporary_name)
        local_tex = temporary / tex_path.name
        shutil.copy2(tex_path, local_tex)
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                local_tex.name,
            ],
            cwd=temporary,
            capture_output=True,
            text=True,
            check=True,
        )
        shutil.copy2(temporary / f"{tex_path.stem}.pdf", pdf_path)
    subprocess.run(
        ["pdftocairo", "-png", "-singlefile", "-r", "300", pdf_path, png_path.with_suffix("")],
        capture_output=True,
        text=True,
        check=True,
    )


def generate_p13_figures(*, results_root: Path, plan_root: Path) -> pd.DataFrame:
    semantic_root = results_root / "semantic"
    semantic_root.mkdir(parents=True, exist_ok=True)
    for directory in ("data", "tikz", "html", "pdf", "png"):
        (plan_root / "figures" / directory).mkdir(parents=True, exist_ok=True)
    domain_claims = pd.read_csv(results_root / "tables/domain_claims.csv")
    crossover = pd.read_csv(results_root / "tables/crossover_effects.csv")
    field = pd.read_csv(results_root / "tables/field_log_results.csv")
    builders: list[tuple[str, Any, Any]] = [
        ("F45_substrate_recoverability.csv", domain_claims, _f45),
        ("F46_substrate_instrument_crossover.csv", crossover, _f46),
        ("F47_recorded_detection_agreement.csv", field, _f47),
    ]
    records: list[dict[str, Any]] = []
    for filename, source, builder in builders:
        semantic_path = semantic_root / filename
        source.to_csv(semantic_path, index=False, lineterminator="\n", float_format="%.12g")
        # Keep the conventional figure-local semantic file as a byte-identical copy so
        # the repository-wide parity validator can audit every figure uniformly.
        (plan_root / "figures/data" / filename).write_bytes(semantic_path.read_bytes())
        digest = sha256_file(semantic_path)
        payload = builder(source, digest)
        # The plotted subset is also explicit in the semantic table through procedure/status
        # fields; every source row remains in the released semantic CSV.
        tex_path = plan_root / "figures/tikz" / f"{payload.figure_id}_{payload.slug}.tex"
        html_path = plan_root / "figures/html" / f"{payload.figure_id}_{payload.slug}.html"
        pdf_path = plan_root / "figures/pdf" / f"{payload.figure_id}_{payload.slug}.pdf"
        png_path = plan_root / "figures/png" / f"{payload.figure_id}_{payload.slug}.png"
        tex_path.write_text(payload.tex)
        html_body = payload.interactive.to_html(
            full_html=True,
            include_plotlyjs=True,
            div_id=f"nato-sers-{payload.figure_id.lower()}",
            config={"displaylogo": False, "responsive": True},
        )
        # Plotly's embedded bundle carries an unused default topojson CDN URL.
        # These figures contain no geographic traces, so remove the fallback to
        # preserve a strictly offline artifact and satisfy the public boundary.
        html_body = html_body.replace("https://cdn.plot.ly/un/", "data:,")
        html_body = "\n".join(line.rstrip() for line in html_body.splitlines()) + "\n"
        html_path.write_text(
            html_body.replace(
                "<head>",
                "<head>\n"
                f"<!-- data_sha256={digest}; NATO SERS {payload.figure_id}; "
                f"description={html.escape(payload.title)} -->",
                1,
            )
        )
        _compile_tex(tex_path, pdf_path, png_path)
        if digest not in tex_path.read_text() or digest not in html_path.read_text():
            raise RuntimeError(f"{payload.figure_id} semantic hash parity failed.")
        records.append(
            {
                "figure_id": payload.figure_id,
                "title": payload.title,
                "caption": payload.caption,
                "semantic_path": semantic_path.relative_to(results_root.parent.parent).as_posix(),
                "semantic_sha256": digest,
                "tikz_path": tex_path.relative_to(plan_root).as_posix(),
                "tikz_sha256": sha256_file(tex_path),
                "pdf_path": pdf_path.relative_to(plan_root).as_posix(),
                "pdf_sha256": sha256_file(pdf_path),
                "png_path": png_path.relative_to(plan_root).as_posix(),
                "png_sha256": sha256_file(png_path),
                "html_path": html_path.relative_to(plan_root).as_posix(),
                "html_sha256": sha256_file(html_path),
            }
        )
    return pd.DataFrame(records)
