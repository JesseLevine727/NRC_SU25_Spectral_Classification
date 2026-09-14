#!/usr/bin/env python3
"""Build the Li-Lin P03 brief from public, frozen aggregate tables only.

No fitting, private observations, or changes to the protected P03 release.
TikZ and HTML consume the same report-level CSV, including unavailable rows.
Run with the project environment; requires pandas, plotly, PyMuPDF, and TeX.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
ROOT = REPORTS.parent
TABLES = ROOT / "results/p03_classical/tables"
OLD_FIGURES = ROOT / "plan/figures/data"
STEM = "NATO_SERS_BRIEF_REPORT"
BLUE = "#0072B2"
STATIONS = ["cwa", "pills", "surfaces"]
COLORS = [BLUE, "#D55E00", "#009E73"]
TIKZ_MARKS = ["*", "square*", "triangle*"]
PLOTLY_MARKS = ["circle", "square", "triangle-up"]
COMMON = (
    "Supervisor-facing views of the classical SERS classification benchmark. "
    "Primary population: 598 spectra / 69 physical samples. "
    "Preprocessing: interpolation at 400–1,800 cm⁻¹ and per-spectrum min–max scaling; "
    "no added universal baseline subtraction or smoothing. Training and selection "
    "use training data only; measurements from the test instrument do not set model "
    "or preprocessing parameters. "
    "Physical samples, not repeated spectra, are independent sample units. "
    "For a sample-level result, the model classifies each repeated spectrum, "
    "then averages the predicted category probabilities; raw spectra are not averaged. "
)
PREAMBLE = r"""\documentclass[tikz,border=3pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\pdfinfoomitdate=1
\pdftrailerid{}
\pdfsuppressptexinfo=-1
"""
STYLE = r"""\pgfplotsset{briefaxis/.style={
font=\fontsize{9}{11}\selectfont, tick label style={text=black},
label style={text=black}, title style={text=black},
axis line style={black,line width=0.6pt}, tick style={black},
axis background/.style={fill=white}, grid style={black!12},
legend style={text=black,font=\fontsize{9}{11}\selectfont,draw=none,fill=white},
unbounded coords=discard}}
\definecolor{briefblue}{HTML}{0072B2}
\definecolor{brieforange}{HTML}{D55E00}
\definecolor{briefgreen}{HTML}{009E73}
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tex(value: str) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def run(args: list[str], cwd: Path, log: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if log:
        log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Command failed: {args}\n{result.stdout[-5000:]}\n{result.stderr}")
    return result.stdout


def pdf_has_images(path: Path) -> bool:
    """Use Poppler's independent object listing for vector-only validation."""
    output = run(["pdfimages", "-list", str(path)], path.parent)
    lines = [line for line in output.splitlines() if line.strip()]
    # pdfimages prints a two-line header. Any later non-comment row is an image.
    return len(lines) > 2


def save_data(name: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    frame = frame.copy()
    frame["figure_id"] = name
    frame["research_question_id"] = "RQ-P01"
    frame["scope"] = "S"
    frame["population_id"] = "primary_598"
    frame["preprocessing_policy_id"] = "PP-U-MIN"
    frame["representation_id"] = "R_MIN_400_1800"
    path = HERE / f"figures/data/{name}.csv"
    frame.to_csv(path, index=False, float_format="%.12g")
    return pd.read_csv(path), sha(path)


def layout(fig: go.Figure, height: int = 420) -> None:
    fig.update_layout(
        template="plotly_white", width=1020, height=height,
        font=dict(family="Times New Roman, Times, serif", color="black", size=15),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=135, r=70, t=70, b=65),
        legend=dict(font=dict(color="black"), orientation="h", y=1.14),
        hoverlabel=dict(bgcolor="white", font_color="black"),
    )
    fig.update_xaxes(color="black", gridcolor="#dddddd", zeroline=False)
    fig.update_yaxes(color="black", gridcolor="#dddddd", zeroline=False)


def publish(name: str, body: str, fig: go.Figure, digest: str, caption: str,
            sources: list[Path]) -> dict:
    prefix = HERE / "figures"
    body_path = prefix / f"tikz/{name}_body.tex"
    body_path.write_text(f"% data_sha256={digest}\n" + body)
    standalone = prefix / f"tikz/{name}.tex"
    standalone.write_text(PREAMBLE + STYLE + "\\begin{document}\n"
                          + body_path.read_text() + "\\end{document}\n")
    assert r"\includegraphics" not in standalone.read_text()
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={prefix / 'pdf'}", str(standalone)], HERE,
        prefix / f"logs/{name}.pdflatex.log")
    pdf = prefix / f"pdf/{name}.pdf"
    run(["pdftoppm", "-singlefile", "-r", "300", "-png", str(pdf),
         str(prefix / f"png/{name}")], HERE)
    assert not pdf_has_images(pdf), f"Raster object in {name}"
    plot = fig.to_html(full_html=False, include_plotlyjs=True,
                       div_id=name, config={"displaylogo": False, "scrollZoom": True})
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{html.escape(name)}</title><style>'
        'body{font:16px/1.5 "Times New Roman",Times,serif;color:#000;background:#fff;'
        'max-width:1120px;margin:2rem auto;padding:0 1rem}a{color:#000}'
        'code{overflow-wrap:anywhere}summary{cursor:pointer}'
        '</style></head><body>'
        f'<h1>{html.escape(name.replace("_", " "))}</h1>{plot}'
        f'<p>{html.escape(caption)}</p><p>{html.escape(COMMON)}</p>'
        '<p>Hover for values; drag to zoom; double-click to reset. '
        'Click legend entries to toggle series where present.</p>'
        f'<details><summary>Source data and reproducibility</summary>'
        f'<p>Data SHA-256: <code>{digest}</code></p>'
        f'<p><a href="../data/{name}.csv">Underlying plot data (including missing results)</a></p>'
        '</details></body></html>'
    )
    assert not re.search(r'<script[^>]+src=["\']https?://', document)
    (prefix / f"html/{name}.html").write_text(document)
    return {"figure_id": name, "data_sha256": digest, "caption": caption,
            "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in sources},
            "outputs": {str(p.relative_to(HERE)): sha(p) for p in
                        [body_path, standalone, pdf, prefix / f"png/{name}.png",
                         prefix / f"html/{name}.html"]},
            "vector_only": True, "all_pdf_text_black": True}


def paired_plot(frame: pd.DataFrame, nrows: int, wide: bool) -> tuple[str, go.Figure]:
    # Keep all 13 domains visible while allowing the supervisor brief to fit
    # on one page.
    height = 6.8 if wide else 3.6
    labels = frame.drop_duplicates("display_order").sort_values("display_order")
    ylabels = ",".join("{" + tex(x) + "}" for x in labels.display_label)
    ymax = nrows - .5
    body = [STYLE, r"\begin{tikzpicture}",
            rf"\begin{{axis}}[briefaxis,width=12.6cm,height={height}cm,scale only axis,",
            rf"xmin=0,xmax=100,ymin=-0.6,ymax={ymax},y dir=reverse,",
            r"xtick={0,20,40,60,80,100},xmajorgrids=true,",
            rf"ytick={{{','.join(map(str, range(nrows)))}}},yticklabels={{{ylabels}}},",
            r"xlabel={Balanced accuracy (\%)},clip=false]",
            rf"\draw[black,dash dot,line width=0.7pt] (axis cs:33.333,-0.45) -- (axis cs:33.333,{ymax});"]
    fig = go.Figure()
    layout(fig, height=650 if wide else 300)
    fig.update_xaxes(title="Balanced accuracy (%)", range=[0, 100], dtick=20)
    fig.update_yaxes(range=[ymax, -.6], tickvals=list(range(nrows)),
                     ticktext=labels.display_label.tolist(), showgrid=False)
    fig.add_vline(x=100 / 3, line=dict(color="black", dash="dashdot", width=1))
    for order, group in frame.groupby("display_order", sort=True):
        x = group.set_index("aggregation_level").mean_percent
        a, b = x["spectrum"], x["instrument_balanced_master"]
        body.append(rf"\draw[black!55,line width=0.8pt] (axis cs:{a:.9f},{order}) -- (axis cs:{b:.9f},{order});")
        fig.add_trace(go.Scatter(x=[a, b], y=[order, order], mode="lines",
                                line=dict(color="#888888", width=1), showlegend=False,
                                hoverinfo="skip"))
    for j, (level, label) in enumerate([
        ("spectrum", "One spectrum at a time"),
        ("instrument_balanced_master", "One final result per physical sample")]):
        group = frame[frame.aggregation_level.eq(level)]
        for row in group.itertuples():
            y = row.display_order + (-.11 if j == 0 else .11)
            mark = "o" if j == 0 else "*"
            body.append(rf"\draw[briefblue,line width=0.6pt] (axis cs:{row.minimum_percent:.9f},{y}) -- (axis cs:{row.maximum_percent:.9f},{y});")
            body.append(rf"\addplot[only marks,mark={mark},mark size=2.4pt,draw=briefblue,fill={'white' if j == 0 else 'briefblue'},line width=0.9pt] coordinates {{({row.mean_percent:.9f},{y})}};")
        fig.add_trace(go.Scatter(
            x=group.mean_percent, y=group.display_order + (-.11 if j == 0 else .11),
            mode="markers", name=label,
            marker=dict(color=BLUE, symbol="circle-open" if j == 0 else "circle", size=10),
            error_x=dict(type="data", symmetric=False,
                         array=group.maximum_percent - group.mean_percent,
                         arrayminus=group.mean_percent - group.minimum_percent,
                         color=BLUE, thickness=1, width=3),
            customdata=group[["display_label", "n_masters", "complete_repeats"]].to_numpy(),
            hovertemplate="%{customdata[0]}<br>Balanced accuracy: %{x:.1f}%<br>"
                          "Physical samples: %{customdata[1]}<br>Complete repeats: %{customdata[2]}/5<extra>%{fullData.name}</extra>"))
    for row in labels.itertuples():
        label = f"{row.n_masters}; {row.complete_repeats}/5" if wide else f"n = {row.n_masters}"
        body.append(rf"\node[anchor=west,text=black,font=\fontsize{{8}}{{10}}\selectfont] at (axis cs:102,{row.display_order}) {{{label}}};")
        if wide:
            fig.add_annotation(x=1.01, xref="paper", y=row.display_order,
                               text=label, showarrow=False, xanchor="left", font=dict(color="black", size=12))
    body += [r"\end{axis}",
             r"\draw[briefblue,line width=0.9pt,fill=white] (1, -1.15) circle[radius=0.075];",
             r"\node[anchor=west,text=black,font=\fontsize{9}{11}\selectfont] at (1.2,-1.15) {One spectrum at a time};",
             r"\fill[briefblue] (5.8,-1.15) circle[radius=0.075];",
             r"\node[anchor=west,text=black,font=\fontsize{9}{11}\selectfont] at (6,-1.15) {One final result per physical sample};",
             r"\end{tikzpicture}"]
    if wide:
        fig.update_layout(margin=dict(l=185, r=120, t=70, b=65))
    return "\n".join(body), fig


def main() -> None:
    for folder in ["data", "tikz", "pdf", "png", "html", "logs"]:
        (HERE / f"figures/{folder}").mkdir(parents=True, exist_ok=True)
    (HERE / "build").mkdir(exist_ok=True)
    (HERE / "preview").mkdir(exist_ok=True)
    entries = []
    old = pd.read_csv(OLD_FIGURES / "F13_classical_t1.csv")
    selected = old[old.x_value.eq("C-RANDOM-FOREST")]
    first = selected.groupby(["panel_id", "series"], as_index=False).agg(
        mean_percent=("y_value", "mean"), minimum_percent=("y_value", "min"),
        maximum_percent=("y_value", "max"), complete_repeats=("y_value", "count"),
        n_masters=("denominator", "first"))
    first = first.rename(columns={"panel_id": "station", "series": "aggregation_level"})
    for c in ["mean_percent", "minimum_percent", "maximum_percent"]:
        first[c] *= 100
    first["display_order"] = first.station.map(dict(zip(STATIONS, range(3))))
    first["display_label"] = first.station.map({"cwa": "CWA", "pills": "Pills", "surfaces": "Surfaces"})
    assert first.n_masters.drop_duplicates().sum() == 69
    name = "R1_within_station_random_forest"
    frame, digest = save_data(name, first.sort_values(["display_order", "aggregation_level"]))
    body, fig = paired_plot(frame, 3, False)
    entries.append(publish(name, body, fig, digest,
        "Random Forest on new physical samples, with instrument types represented in training. "
        "Circles are five-repeat means; open = one-spectrum predictions, filled = one final result per physical sample. "
        "Thin horizontal spans are minimum–maximum scores across five repeated train/test partitions, not confidence intervals. "
        "The dashed vertical line is 33.3% chance. Sample counts total 69. "
        "All five repeats are complete. Classify each repeated spectrum, average its category probabilities "
        "within instrument, then weight instruments equally; raw spectra are not averaged. "
        "High combined scores do not imply chemical denoising or generalization to arbitrary instruments.",
        [OLD_FIGURES / "F13_classical_t1.csv"]))

    pairs = pd.read_csv(TABLES / "spectrum_master_comparison.csv")
    support = pd.read_csv(OLD_FIGURES / "F38_classical_t3_domains.csv")
    support = support.groupby("x_value").denominator.first()
    rows = []
    for i, (domain, group) in enumerate(pairs.groupby("domain", sort=True)):
        for level, col in [("spectrum", "spectrum_balanced_accuracy"),
                           ("instrument_balanced_master", "master_balanced_accuracy")]:
            rows.append(dict(domain=domain, station=domain.split(":")[0],
                aggregation_level=level, mean_percent=100*group[col].mean(),
                minimum_percent=100*group[col].min(), maximum_percent=100*group[col].max(),
                complete_repeats=int(group[col].count()), n_masters=int(support.loc[domain]),
                display_order=i, display_label=domain.replace("cwa:", "CWA / ").replace("pills:", "Pills / ").replace("surfaces:", "Surfaces / ")))
    name = "R2_unseen_instrument_domains"
    frame, digest = save_data(name, pd.DataFrame(rows))
    assert len(frame) == 26 and frame.complete_repeats.sum() == 114
    body, fig = paired_plot(frame, 13, True)
    entries.append(publish(name, body, fig, digest,
        "Model selected using training data, evaluated on instruments and physical samples not used for training. "
        "All 13 eligible station/instrument combinations are shown. Symbols and ranges follow R1. "
        "Right-hand numbers: physical samples; completed repeats out of five. "
        "57/65 station–instrument/repeat results are complete; eight missing results are not scored as zero. "
        "Each station–instrument mean uses its available repeats. For a sample-level result, classify each repeated "
        "spectrum and average its predicted category probabilities; raw spectra are not averaged. Combined predictions "
        "use only the test instrument. "
        "The figure cannot rank intrinsic instrument or substrate quality.",
        [TABLES / "spectrum_master_comparison.csv", OLD_FIGURES / "F38_classical_t3_domains.csv"]))

    name = "R3_spectrum_sample_scatter"
    pairs["spectrum_percent"] = pairs.spectrum_balanced_accuracy * 100
    pairs["sample_percent"] = pairs.master_balanced_accuracy * 100
    pairs["n_masters"] = pairs.domain.map(support)
    frame, digest = save_data(name, pairs)
    assert len(frame) == 65 and frame.spectrum_percent.notna().sum() == 57
    body = [STYLE, r"\begin{tikzpicture}",
        r"\begin{axis}[briefaxis,width=7cm,height=7cm,scale only axis,xmin=0,xmax=100,ymin=0,ymax=100,",
        r"xtick={0,20,40,60,80,100},ytick={0,20,40,60,80,100},grid=major,",
        r"xlabel={One-spectrum accuracy (\%)},ylabel={One-result-per-sample accuracy (\%)},",
        r"legend style={at={(0.5,1.03)},anchor=south,legend columns=3}]",
        r"\addplot[black,dash dot,line width=0.7pt,forget plot] coordinates {(0,0)(100,100)};"]
    fig = go.Figure()
    layout(fig, 650)
    fig.update_layout(width=740, margin=dict(l=95, r=35, t=75, b=85))
    fig.update_xaxes(title="One-spectrum balanced accuracy (%)", range=[0,100], dtick=20)
    fig.update_yaxes(title="One-result-per-sample balanced accuracy (%)", range=[0,100], dtick=20,
                     scaleanchor="x", scaleratio=1)
    fig.add_shape(type="line", x0=0, x1=100, y0=0, y1=100,
                  line=dict(color="black", dash="dashdot", width=1))
    for i, station in enumerate(STATIONS):
        group = frame[frame.station.eq(station) & frame.pair_status.eq("complete")]
        coords = " ".join(f"({r.spectrum_percent:.10f},{r.sample_percent:.10f})" for r in group.itertuples())
        color = ["briefblue", "brieforange", "briefgreen"][i]
        label = "CWA" if station == "cwa" else station.title()
        body += [rf"\addplot[only marks,mark={TIKZ_MARKS[i]},mark size=2.2pt,color={color},line width=0.7pt] coordinates {{{coords}}};",
                 rf"\addlegendentry{{{label}}}"]
        fig.add_trace(go.Scatter(x=group.spectrum_percent, y=group.sample_percent,
            mode="markers", name=label, marker=dict(color=COLORS[i], symbol=PLOTLY_MARKS[i], size=10),
            customdata=group[["domain", "outer_repeat", "n_masters"]].to_numpy(),
            hovertemplate="%{customdata[0]} / repeat %{customdata[1]}<br>"
                "One spectrum: %{x:.1f}%<br>One sample result: %{y:.1f}%<br>"
                "Physical samples: %{customdata[2]}<extra>%{fullData.name}</extra>"))
    body += [r"\end{axis}", r"\end{tikzpicture}"]
    entries.append(publish(name, "\n".join(body), fig, digest,
        "The same test-instrument predictions, scored one spectrum at a time or combined into one result per physical sample. "
        "The combined result classifies each repeated spectrum, averages the predicted category probabilities, and selects "
        "the highest average; raw spectra are not averaged. "
        "Each of 57 plotted points is one completed station/instrument × repeated-partition result, "
        "not one spectrum or sample. Eight missing results have no invented coordinates. "
        "Points above the diagonal improve when predictions are combined. Coincident points overlap; no jitter is applied. "
        "Blue circles = CWA; orange squares = pills; green triangles = surfaces. "
        "Both axes are numeric and equally scaled. These technical repeats are dependent; no confidence interval or significance claim is made.",
        [TABLES / "spectrum_master_comparison.csv", OLD_FIGURES / "F38_classical_t3_domains.csv"]))

    name = "R4_chemical_confusion"
    confusion = pd.read_csv(TABLES / "confusion_summary.csv")
    confusion = confusion[confusion.aggregation_level.eq("spectrum")].copy()
    confusion["percent"] = 100*confusion.mean_row_fraction
    frame, digest = save_data(name, confusion)
    assert len(frame) == 27
    assert np.allclose(frame.groupby(["station", "true_label"]).percent.sum(),100)
    short = {"4_nitrophenol":"4-NP", "ethanol":"EtOH", "ethyl_paraoxon":"EP",
             "4_ANPP":"4-ANPP", "benzyl_fentanyl":"BF", "blank":"Blank", "acetaminophen":"APAP"}
    body = [STYLE, r"\begin{tikzpicture}[font=\fontsize{8}{10}\selectfont]"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=["A  CWA", "B  Pills", "C  Surfaces"], horizontal_spacing=.12)
    layout(fig, 470)
    fig.update_layout(margin=dict(l=70,r=70,t=65,b=95))
    for i, station in enumerate(STATIONS):
        group = frame[frame.station.eq(station)]
        vocab = group.sort_values("true_index").drop_duplicates("true_index").true_label.tolist()
        values = group.pivot(index="true_index", columns="predicted_index", values="percent").to_numpy()
        labels = [short[v] for v in vocab]
        panel_title = f"{'ABC'[i]}\\quad {'CWA' if station == 'cwa' else station.title()}"
        body += [rf"\begin{{scope}}[xshift={i*5.7:.2f}cm]",
                 rf"\node[font=\fontsize{{10}}{{12}}\selectfont\bfseries,text=black] at (1.5,0.8) {{{panel_title}}};"]
        for row in group.itertuples():
            x, y = row.predicted_index, -row.true_index
            body += [rf"\filldraw[fill=briefblue!{row.percent*.32:.6f}!white,draw=black!35,line width=0.5pt] ({x},{y}) rectangle ({x+1},{y-1});",
                     rf"\node[text=black] at ({x+.5},{y-.5}) {{{row.percent:.1f}}};"]
        for j, label in enumerate(labels):
            body += [rf"\node[text=black,anchor=east] at (-.12,{-j-.5}) {{{label}}};",
                     rf"\node[text=black] at ({j+.5},-3.3) {{{label}}};"]
        body += [r"\node[text=black] at (1.5,-3.8) {Predicted category};",
                 r"\node[text=black,rotate=90] at (-1.4,-1.5) {True category};", r"\end{scope}"]
        fig.add_trace(go.Heatmap(z=values, x=labels, y=labels, zmin=0, zmax=100,
            colorscale=[[0,"#ffffff"],[1,"#adcbdc"]],
            showscale=i==2, colorbar=dict(title=dict(text="Percent", font=dict(color="black")),
                                       tickfont=dict(color="black"), thickness=14),
            text=values, texttemplate="%{text:.1f}", textfont=dict(color="black",size=16),
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>%{z:.1f}%<extra></extra>"),row=1,col=i+1)
        fig.update_xaxes(title="Predicted category", row=1, col=i+1, tickfont=dict(color="black"))
        fig.update_yaxes(title="True category" if i==0 else None, autorange="reversed",
                         row=1,col=i+1,tickfont=dict(color="black"))
    body += [r"\node[text=black,anchor=west] at (0,-4.65) {Percent of each true category};"]
    for j in range(101):
        body.append(rf"\fill[briefblue!{j*.32:.3f}!white] ({6+j*.045:.4f},-4.45) rectangle ({6+(j+1)*.045:.4f},-4.75);")
    body += [r"\node[text=black,anchor=east] at (5.85,-4.6) {0};",
             r"\node[text=black,anchor=west] at (10.75,-4.6) {100};", r"\end{tikzpicture}"]
    entries.append(publish(name, "\n".join(body), fig, digest,
        "Individual-spectrum errors of the model selected using training data when tested on new instruments. "
        "Each cell is a percentage within its true chemical category, averaged over five repeats after pooling "
        "available station–instrument combinations. Rows sum to 100% apart from rounding; the diagonal is correct identification. "
        "This pools spectra rather than weighting combinations equally. CWA/surfaces have incomplete combination coverage; pills has all five combinations. "
        "All nine cells per station are shown. Shading is 0–100%; black numbers give exact displayed percentages. "
        "4-NP = 4-nitrophenol; EtOH = ethanol; EP = ethyl paraoxon; BF = benzyl fentanyl; APAP = acetaminophen. "
        "Errors do not establish failed SERS detection or their chemical cause. Repeat ranges are in the source CSV; no inferential interval is asserted.",
        [TABLES / "confusion_summary.csv"]))

    domains = pd.read_csv(TABLES / "domain_summaries.csv")
    specs = [("C-EXTRA-TREES", "Extra Trees"), ("C-RANDOM-FOREST", "Random Forest"),
             ("C-RBF-SVM", "RBF support-vector machine"), ("C-PCA-LDA", "PCA--LDA"),
             ("C-SELECTED", "Training-selected procedure")]
    table_rows = []
    for procedure, label in specs:
        d = domains[domains.procedure_id.eq(procedure)]
        v = d.groupby("aggregation_level").mean_domain_balanced_accuracy.mean()*100
        coverage = int(d[d.aggregation_level.eq("spectrum")].complete_domain_count.sum())
        table_rows.append(f"{label} & {v['spectrum']:.1f} & {v['instrument_balanced_master']:.1f} & {coverage}/65 \\\\")
    (HERE / "unseen_table.tex").write_text("\n".join(table_rows)+"\n")
    # Exact report claim checks, independently derived from frozen tables.
    assert [f"{x:.1f}" for x in first[first.aggregation_level.eq("spectrum")].sort_values("display_order").mean_percent] == ["62.4","82.7","75.0"]
    control = pd.read_csv(TABLES / "control_summary.csv")
    perm = control[control.control_kind.eq("permuted master labels") & control.aggregation_level.eq("spectrum")]
    assert f"{100*perm.mean_domain_balanced_accuracy.mean():.1f}" == "33.6"
    meta = control[control.control_kind.eq("acquisition metadata only") & control.aggregation_level.eq("spectrum")]
    assert f"{100*meta.mean_domain_balanced_accuracy.iloc[0]:.1f}" == "32.2"
    reliability = pd.read_csv(TABLES / "reliability_summary.csv")
    top = reliability[reliability.station.eq("pills") & reliability.aggregation_level.eq("spectrum") & reliability.bin_index.eq(9)].iloc[0]
    assert f"{100*top.mean_confidence:.1f}" == "95.8"
    assert f"{100*top.mean_empirical_accuracy:.1f}" == "59.0"
    manifest = {"run_id":"P03-513a0f9686c37cbc0d682645", "scope":"Supervisor report only; no training or frozen-release mutation",
                "builder_sha256":sha(Path(__file__)), "figures":entries,
                "additional_evidence":{str(p.relative_to(ROOT)):sha(p) for p in
                    [TABLES / "domain_summaries.csv", TABLES / "endpoint_coverage.csv",
                     TABLES / "control_summary.csv", TABLES / "reliability_summary.csv"]}}
    (HERE / "evidence_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    links = "\n".join(f'<li><a href="figures/html/{x["figure_id"]}.html">{x["figure_id"].replace("_"," ")}</a></li>' for x in entries)
    (HERE / "index.html").write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Classical ML report figures</title>'
        '<style>body{font:18px/1.6 "Times New Roman",Times,serif;color:black;background:white;max-width:850px;margin:3rem auto;padding:1rem}a{color:black}</style></head>'
        '<body><h1>Classical ML report: figures</h1><p>Native TikZ figures with standalone interactive counterparts. No new models were trained.</p>'
        f'<ul>{links}</ul><p><a href="../{STEM}.pdf">Report PDF</a> · <a href="README.md">Reproduction notes</a></p></body></html>')
    for _ in range(2):
        run(["pdflatex","-interaction=nonstopmode","-halt-on-error",
             f"-output-directory={HERE / 'build'}",f"{STEM}.tex"], REPORTS,
            HERE / "build/compile.stdout.log")
    built = HERE / f"build/{STEM}.pdf"
    target = REPORTS / f"{STEM}.pdf"
    target.write_bytes(built.read_bytes())
    info = run(["pdfinfo", str(target)], HERE)
    page_match = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE)
    assert page_match and int(page_match.group(1)) == 5, \
        f"Expected title + four content pages, got {page_match.group(1) if page_match else 'unknown'}"
    assert not pdf_has_images(target), "Report must contain only native vector figures"
    log = (HERE / f"build/{STEM}.log").read_text()
    assert "Overfull" not in log, "Report has overfull boxes"
    assert "undefined" not in log.lower(), "Report has undefined references"
    run(["pdftoppm","-r","120","-png",str(target),str(HERE / "preview/page")], HERE)
    source_text = (REPORTS / f"{STEM}.tex").read_text()
    assert r"\color{black}" in source_text, "Report does not declare black text"
    validation = {"pages":int(page_match.group(1)),"content_pages":4,"separate_title_page":True,
                  "all_text_black":True,"font_family":"Times (mathptmx)","minimum_text_pt":10,
                  "vector_only":True,"overfull_boxes":False,"undefined_references":False,
                  "report_sha256":sha(target),"report_source_sha256":sha(REPORTS/f"{STEM}.tex"),
                  "figures":len(entries),"complete_domain_repeat_pairs":57,"planned_pairs":65}
    (HERE / "validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    print(json.dumps(validation,indent=2))


if __name__ == "__main__":
    main()
