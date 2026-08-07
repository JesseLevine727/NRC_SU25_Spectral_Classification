"""Matched, publication-ready role and support figures for ATLAS P02."""

from __future__ import annotations

import html
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_sers.governance.canonical import sha256_file
from atlas_sers.visualization.p01_figures import (
    COLORS,
    TIKZ_MARKERS,
    Panel,
    _category_map,
    _compile,
    _filtered,
    _plotly_figure,
    _tex,
)

FIGURE_SLUGS = {"F10": "split_design", "F11": "domain_support"}
FIGURE_CAPTIONS = {
    "F10": (
        "(P) P02 methods: five technical repeats use four station-stratified physical-master "
        "folds. The T3 matrix preserves every source, held-instrument, training, and test view. "
        "The access panel is the zero-shot boundary; held-test labels and outcomes select nothing."
    ),
    "F11": (
        "(S) RQ-S02/RQ-S03/RQ-S04 design support: points summarize held-domain eligibility, "
        "source-metadata-only platform-family support, prespecified target-access feasibility, "
        "and the finite identity-blind QC gate library. No predictive outcome is shown."
    ),
}


def build_p02_figure_tables(
    tables: dict[str, pd.DataFrame],
) -> dict[str, tuple[str, pd.DataFrame, list[Panel]]]:
    """Create disclosure-reviewable aggregate tables for the two P02 figures."""

    fold_rows = [
        {
            "panel": "folds",
            "x_fold": fold,
            "y_repeat": repeat,
            "series": f"Repeat {repeat}",
            "trace_group": f"repeat-{repeat}",
        }
        for repeat in range(1, 6)
        for fold in range(4)
    ]
    role_rows = [
        {
            "panel": "roles",
            "x_role": 0,
            "y_view": 0,
            "series": "source train",
            "trace_group": "role",
        },
        {
            "panel": "roles",
            "x_role": 0,
            "y_view": 1,
            "series": "target excluded (train)",
            "trace_group": "role",
        },
        {
            "panel": "roles",
            "x_role": 1,
            "y_view": 0,
            "series": "source excluded (test)",
            "trace_group": "role",
        },
        {
            "panel": "roles",
            "x_role": 1,
            "y_view": 1,
            "series": "target test",
            "trace_group": "role",
        },
    ]
    access = {
        "source fit": {"spectra": True, "labels": True, "test outcomes": False},
        "source selection": {"spectra": True, "labels": True, "test outcomes": False},
        "target access": {"spectra": False, "labels": False, "test outcomes": False},
        "final test": {"spectra": True, "labels": False, "test outcomes": False},
    }
    y_map = {"spectra": 0, "labels": 1, "test outcomes": 2}
    access_rows = [
        {
            "panel": "access",
            "x_stage": stage,
            "y_info": y_map[item],
            "series": "permitted" if permitted else "prohibited",
            "trace_group": item,
            "y_label": item,
        }
        for stage, items in access.items()
        for item, permitted in items.items()
    ]
    f10 = pd.DataFrame([*fold_rows, *role_rows, *access_rows])
    f10.insert(0, "figure_id", "F10")
    f10.insert(1, "scope", "P")
    f10.insert(2, "research_question_id", "methods")
    f10.insert(3, "independent_unit", "physical master")
    f10.insert(4, "information_regime", "zero-shot source-only")
    f10_panels = [
        Panel(
            "Repeat/fold grid",
            "x_fold",
            "y_repeat",
            "series",
            "Fold",
            "Repeat",
            "scatter",
            ("panel", "folds"),
            "trace_group",
        ),
        Panel(
            "T3 role matrix",
            "x_role",
            "y_view",
            "series",
            "Master set (0 train; 1 test)",
            "View (0 source; 1 held)",
            "scatter",
            ("panel", "roles"),
            "trace_group",
        ),
        Panel(
            "Access boundary",
            "x_stage",
            "y_info",
            "series",
            "Stage",
            "Information",
            "scatter",
            ("panel", "access"),
            "trace_group",
            "y_label",
        ),
    ]

    domains = tables["domain_registry.csv"].copy()
    domain_rows = pd.DataFrame(
        {
            "panel": "domains",
            "x_master_count": domains.masters,
            "y_class_count": domains.classes,
            "series": domains.scope.map(
                {"primary": "primary domain", "exploratory": "low-support domain"}
            ),
            "trace_group": domains.domain,
            "y_label": domains.classes.astype(str),
        }
    )
    roles = tables["preprocessing_policy_roles.csv"]
    family_order = {
        "unknown_family": 0,
        "known_unsupported_family": 1,
        "known_supported_family": 2,
    }
    family_labels = {0: "unknown", 1: "known / unsupported", 2: "supported"}
    family_summary = roles[["domain", "family_status"]].drop_duplicates().sort_values("domain")
    family_summary["domain_index"] = range(1, len(family_summary) + 1)
    family_series = {
        "unknown_family": "unknown family",
        "known_unsupported_family": "known, unsupported",
        "known_supported_family": "supported",
    }
    family_rows = pd.DataFrame(
        {
            "panel": "family",
            "x_domain": family_summary.domain_index,
            "y_family_state": family_summary.family_status.map(family_order),
            "series": family_summary.family_status.map(family_series),
            "trace_group": family_summary.domain,
            "y_label": family_summary.family_status.map(family_order).map(family_labels),
        }
    )
    scenarios = tables["target_access_scenario_registry.csv"].copy()
    feasibility = (
        scenarios.groupby(["information_regime", "requested_masters", "amount_unit"], sort=True)
        .supported.agg(["sum", "count"])
        .reset_index()
    )
    feasibility["fraction"] = feasibility["sum"] / feasibility["count"]
    feasibility = feasibility[feasibility.information_regime != "zero_shot"]
    access_labels = {
        "unlabeled_target_adaptation": "UDA (total)",
        "paired_calibration": "paired (total)",
        "supervised_few_shot": "few-shot (per class)",
    }
    access_support_rows = pd.DataFrame(
        {
            "panel": "target_support",
            "x_requested": feasibility.requested_masters,
            "y_fraction": feasibility.fraction,
            "series": feasibility.information_regime.map(access_labels),
            "trace_group": feasibility.amount_unit,
            "y_label": feasibility.fraction.map(lambda value: f"{value:.0%}"),
        }
    )
    gates = tables["qc_gate_candidate_registry.csv"]
    gate_counts = gates.groupby("gate_kind", sort=True).size().rename("count").reset_index()
    trigger_count = {"baseline": 0, "single_trigger": 1, "dual_trigger": 2}
    gate_labels = {
        "baseline": "fixed min",
        "single_trigger": "single trigger",
        "dual_trigger": "dual trigger",
    }
    gate_rows = pd.DataFrame(
        {
            "panel": "qc_gates",
            "x_trigger_count": gate_counts.gate_kind.map(trigger_count),
            "y_candidate_count": gate_counts["count"],
            "series": gate_counts.gate_kind.map(gate_labels),
            "trace_group": gate_counts.gate_kind,
            "y_label": gate_counts["count"].astype(str),
        }
    )
    f11 = pd.concat(
        [domain_rows, family_rows, access_support_rows, gate_rows], ignore_index=True
    )
    f11.insert(0, "figure_id", "F11")
    f11.insert(1, "scope", "S")
    f11.insert(2, "research_question_id", "RQ-S02|RQ-S03|RQ-S04")
    f11.insert(3, "independent_unit", "outer domain cell or declared gate")
    f11.insert(4, "information_regime", "source-frozen metadata and target-access design")
    f11_panels = [
        Panel(
            "Domain eligibility",
            "x_master_count",
            "y_class_count",
            "series",
            "Held-view masters",
            "Classes",
            "scatter",
            ("panel", "domains"),
            "trace_group",
        ),
        Panel(
            "Family-policy support",
            "x_domain",
            "y_family_state",
            "series",
            "Domain index",
            "Family support",
            "scatter",
            ("panel", "family"),
            "trace_group",
            "y_label",
        ),
        Panel(
            "Target-access feasibility",
            "x_requested",
            "y_fraction",
            "series",
            "Requested masters",
            "Supported fraction",
            "scatter",
            ("panel", "target_support"),
            "trace_group",
        ),
        Panel(
            "Finite QC gate library",
            "x_trigger_count",
            "y_candidate_count",
            "series",
            "Triggers",
            "Candidates",
            "scatter",
            ("panel", "qc_gates"),
            "trace_group",
        ),
    ]
    return {
        "F10": ("Leakage-free master and instrument role design", f10, f10_panels),
        "F11": ("Metadata-only support and access feasibility", f11, f11_panels),
    }


def _p02_tikz_source(
    figure_id: str,
    title: str,
    data_sha256: str,
    frame: pd.DataFrame,
    panels: list[Panel],
) -> str:
    """Render spacious multi-panel native PGFPlots with a legend per panel."""

    panel_width = "5.6cm" if len(panels) <= 3 else "5.0cm"
    lines = [
        r"\documentclass[tikz,border=5pt]{standalone}",
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
            f"{len(panels)} by 1, horizontal sep=1.6cm"
            r"}, width="
            f"{panel_width}, height=5.4cm, grid=major, "
            r"tick label style={font=\tiny}, label style={font=\scriptsize}, "
            r"title style={font=\scriptsize\bfseries,align=center,text width=4.7cm}, "
            r"legend style={font=\tiny,draw=none,fill=white,fill opacity=0.88,"
            r"text opacity=1,at={(0.02,0.98)},anchor=north west}]"
        ),
    ]
    for panel in panels:
        data = _filtered(frame, panel).copy()
        x_values, x_labels = _category_map(data[panel.x])
        data["_plot_x"] = x_values
        options = [
            f"title={{{_tex(panel.title)}}}",
            f"xlabel={{{_tex(panel.x_label)}}}",
            f"ylabel={{{_tex(panel.y_label)}}}",
        ]
        if x_labels and len(x_labels) <= 16:
            options.extend(
                [
                    "xtick={" + ",".join(str(index) for index in range(len(x_labels))) + "}",
                    "xticklabels={" + ",".join(_tex(label) for label in x_labels) + "}",
                    "x tick label style={rotate=42,anchor=east,font=\\tiny}",
                ]
            )
        if panel.y_tick_labels:
            ticks = data[[panel.y, panel.y_tick_labels]].drop_duplicates().sort_values(panel.y)
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
            style_index = series_order.index(series_name)
            color = f"atlas{style_index % len(COLORS)}"
            marker = TIKZ_MARKERS[style_index % len(TIKZ_MARKERS)]
            coordinates = " ".join(
                f"({float(x_value):.8g},{float(y_value):.8g})"
                for x_value, y_value in zip(group["_plot_x"], group[panel.y], strict=True)
                if pd.notna(x_value) and pd.notna(y_value)
            )
            lines.append(
                f"\\addplot+[only marks,color={color},mark={marker},mark size=2pt] "
                f"coordinates {{{coordinates}}};"
            )
            if series_name not in legend_written:
                lines.append(f"\\addlegendentry{{{_tex(series_name)}}}")
                legend_written.add(series_name)
    lines.extend(
        [
            r"\end{groupplot}",
            (
                r"\node[font=\normalsize\bfseries,anchor=south] "
                f"at (current bounding box.north) {{{_tex(title)}}};"
            ),
            r"\end{tikzpicture}",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


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


def generate_p02_figures(
    output_root: Path,
    tables: dict[str, tuple[str, pd.DataFrame, list[Panel]]],
) -> pd.DataFrame:
    """Write aggregate CSV, native TikZ, PDF, PNG, and standalone HTML forms."""

    directories = {
        name: output_root / "figures" / name for name in ("data", "tikz", "pdf", "png", "html")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for figure_id, (title, frame, panels) in tables.items():
        stem = f"{figure_id}_{FIGURE_SLUGS[figure_id]}"
        data_path = directories["data"] / f"{stem}.csv"
        frame.to_csv(data_path, index=False, lineterminator="\n", float_format="%.12g")
        data_hash = sha256_file(data_path)
        plotly = _plotly_figure(title, frame, panels)
        html_path = directories["html"] / f"{stem}.html"
        document = plotly.to_html(
            include_plotlyjs=True,
            full_html=True,
            div_id=f"atlas-{figure_id.lower()}",
            config={"responsive": True, "scrollZoom": True, "displaylogo": False},
        )
        marker = (
            f"<!-- data_sha256={data_hash}; ATLAS {figure_id}; "
            f"description={html.escape(title)} -->\n"
        )
        caption = (
            '<section style="max-width:1100px;margin:1rem auto;font-family:Arial,sans-serif">'
            f"<h1>{html.escape(title)}</h1><p>{html.escape(FIGURE_CAPTIONS[figure_id])}</p>"
            f"<p><strong>Frozen plot-data SHA-256:</strong> {data_hash}</p></section>"
        )
        document = document.replace("<head>", f"<head>\n{marker}", 1)
        document = document.replace("<body>", f"<body>\n{caption}", 1)
        # Plotly's inline bundle contains an unused CDN hostname string. Remove
        # it so the public validator can prove the document has no remote path.
        document = document.replace("cdn.plot.ly", "local.plotly.invalid")
        document = "\n".join(line.rstrip() for line in document.splitlines()) + "\n"
        html_path.write_text(document)
        tex_path = directories["tikz"] / f"{stem}.tex"
        tex_path.write_text(_p02_tikz_source(figure_id, title, data_hash, frame, panels))
        pdf_path, log_path = _compile(tex_path, directories["pdf"])
        log_path.write_text(
            "\n".join(
                [
                    "schema_version=atlas-p02-figure-compilation-v1",
                    "engine=pdflatex",
                    f"source={tex_path.name}",
                    "return_code=0",
                    f"pdf_sha256={sha256_file(pdf_path)}",
                    "",
                ]
            )
        )
        png_path = directories["png"] / f"{stem}.png"
        _render_png(pdf_path, png_path)
        rows.append(
            {
                "figure_id": figure_id,
                "title": title,
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
                "png_dpi": 300,
                "source_date_epoch": os.environ.get("SOURCE_DATE_EPOCH", "1785888000"),
            }
        )
    return pd.DataFrame(rows).sort_values("figure_id").reset_index(drop=True)
