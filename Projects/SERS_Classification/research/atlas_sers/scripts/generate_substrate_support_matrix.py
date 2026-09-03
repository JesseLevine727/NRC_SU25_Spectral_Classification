#!/usr/bin/env python3
"""Generate the NATO SERS sample × substrate × instrument support matrix.

The canonical public output uses recorded physical-master identifiers, as
approved by the project owner. An optional pseudonym mode remains available for
external derivative documents. Neither mode exports source paths, filenames,
observation IDs, or operator IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FIGURE_ID = "F44"
SLUG = "sample_substrate_instrument_matrix"
STATION_ORDER = ["cwa", "pills", "surfaces"]
TARGET_ORDER = {
    "cwa": ["4_nitrophenol", "ethanol", "ethyl_paraoxon"],
    "pills": ["4_ANPP", "benzyl_fentanyl", "blank"],
    "surfaces": ["4_ANPP", "acetaminophen", "benzyl_fentanyl"],
}
INSTRUMENT_ORDER = [
    "Agilent-1",
    "Agilent-3",
    "Mira-1",
    "Mira-2",
    "Mira-3",
    "Pendar-1",
    "Pendar-2",
    "Pendar-3",
    "RMX-1",
    "RMX-2",
]
INSTRUMENT_SHORT = {
    "Agilent-1": "A1",
    "Agilent-3": "A3",
    "Mira-1": "M1",
    "Mira-2": "M2",
    "Mira-3": "M3",
    "Pendar-1": "P1",
    "Pendar-2": "P2",
    "Pendar-3": "P3",
    "RMX-1": "R1",
    "RMX-2": "R2",
}
SUBSTRATE_ORDER = [
    "pSERS_Metrohm_silver",
    "H_SERS_H_Kit",
    "NRC_Canadian_SERS",
    "GaN_polymer",
]
SUBSTRATE_LABEL = {
    "pSERS_Metrohm_silver": "pSERS silver",
    "H_SERS_H_Kit": "H-SERS H-Kit",
    "NRC_Canadian_SERS": "NRC Canadian SERS",
    "GaN_polymer": "GaN / polymer",
}
GROUP_LABEL = {
    ("cwa", "4_nitrophenol"): "CWA: 4-NP",
    ("cwa", "ethanol"): "CWA: EtOH",
    ("cwa", "ethyl_paraoxon"): "CWA: EP",
    ("pills", "4_ANPP"): "Pills: 4-ANPP",
    ("pills", "benzyl_fentanyl"): "Pills: BF",
    ("pills", "blank"): "Pills: blank",
    ("surfaces", "4_ANPP"): "Surf.: 4-ANPP",
    ("surfaces", "acetaminophen"): "Surf.: APAP",
    ("surfaces", "benzyl_fentanyl"): "Surf.: BF",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tex(text: object) -> str:
    value = str(text)
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        value = value.replace(source, target)
    return value


def _master_label(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    return str(value)


def _load_matrix(
    manifest_path: Path, sample_label_mode: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path, low_memory=False)
    required = {
        "master_sample_id",
        "station",
        "target_analyte",
        "instrument",
        "sensor_family",
        "tier_unique_attributable_sers",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest lacks required columns: {missing}")

    selected = manifest[
        manifest["tier_unique_attributable_sers"]
        .astype(str)
        .str.lower()
        .eq("true")
    ].copy()
    if len(selected) != 598 or selected["master_sample_id"].nunique() != 69:
        raise ValueError(
            "Expected the frozen 598-spectrum, 69-master primary population; "
            f"received {len(selected)} spectra and "
            f"{selected['master_sample_id'].nunique()} masters."
        )

    station_rank = {value: index for index, value in enumerate(STATION_ORDER)}
    target_rank = {
        (station, target): index
        for station, targets in TARGET_ORDER.items()
        for index, target in enumerate(targets)
    }
    masters = selected[
        ["master_sample_id", "station", "target_analyte"]
    ].drop_duplicates()
    per_master_labels = masters.groupby("master_sample_id").size()
    if not per_master_labels.eq(1).all():
        bad = per_master_labels[~per_master_labels.eq(1)].index.astype(str).tolist()
        raise ValueError(f"Masters have conflicting station/target labels: {bad}")
    masters["station_rank"] = masters["station"].map(station_rank)
    masters["target_rank"] = [
        target_rank.get((station, target), 999)
        for station, target in zip(
            masters["station"], masters["target_analyte"], strict=True
        )
    ]
    masters["master_sort"] = pd.to_numeric(
        masters["master_sample_id"], errors="coerce"
    )
    masters["master_sort_text"] = masters["master_sample_id"].astype(str)
    masters = masters.sort_values(
        ["station_rank", "target_rank", "master_sort", "master_sort_text"],
        na_position="last",
    ).reset_index(drop=True)
    masters["sample_order"] = np.arange(1, len(masters) + 1)
    if sample_label_mode == "master-id":
        masters["sample_code"] = masters["master_sample_id"].map(_master_label)
    else:
        masters["sample_code"] = masters["sample_order"].map(
            lambda value: f"S{value:02d}"
        )

    code_map = masters.set_index("master_sample_id")["sample_code"]
    selected["sample_code"] = selected["master_sample_id"].map(code_map)
    counts = (
        selected.groupby(
            ["sample_code", "sensor_family", "instrument"], dropna=False
        )
        .size()
        .rename("replicate_count")
    )

    full_index = pd.MultiIndex.from_product(
        [masters["sample_code"], SUBSTRATE_ORDER, INSTRUMENT_ORDER],
        names=["sample_code", "substrate_family", "instrument"],
    )
    matrix = counts.reindex(full_index, fill_value=0).reset_index()
    matrix = matrix.merge(
        masters[["sample_code", "sample_order", "station", "target_analyte"]],
        on="sample_code",
        how="left",
        validate="many_to_one",
    )
    matrix["substrate_label"] = matrix["substrate_family"].map(SUBSTRATE_LABEL)
    matrix["substrate_order"] = matrix["substrate_family"].map(
        {value: index + 1 for index, value in enumerate(SUBSTRATE_ORDER)}
    )
    matrix["instrument_order"] = matrix["instrument"].map(
        {value: index + 1 for index, value in enumerate(INSTRUMENT_ORDER)}
    )
    matrix["instrument_short"] = matrix["instrument"].map(INSTRUMENT_SHORT)
    matrix["observed"] = matrix["replicate_count"].gt(0)
    matrix["display_level"] = matrix["replicate_count"].clip(upper=3).astype(int)
    matrix["group_label"] = [
        GROUP_LABEL.get((station, target), f"{station}: {target}")
        for station, target in zip(
            matrix["station"], matrix["target_analyte"], strict=True
        )
    ]
    matrix.insert(0, "figure_id", FIGURE_ID)
    matrix.insert(1, "scope", "S")
    matrix.insert(2, "research_question_id", "RQ-S07")

    public_masters = masters[
        ["sample_code", "sample_order", "station", "target_analyte"]
    ].copy()
    return matrix, public_masters


def _write_tikz(
    matrix: pd.DataFrame,
    masters: pd.DataFrame,
    data_hash: str,
    output_path: Path,
) -> None:
    level_color = {0: "matrixmissing", 1: "matrixone", 2: "matrixtwo", 3: "matrixthree"}
    panel_positions = {
        SUBSTRATE_ORDER[0]: (0.0, 0.0),
        SUBSTRATE_ORDER[1]: (8.6, 0.0),
        SUBSTRATE_ORDER[2]: (0.0, -14.7),
        SUBSTRATE_ORDER[3]: (8.6, -14.7),
    }
    group_bounds = (
        masters.groupby(["station", "target_analyte"], sort=False)
        .agg(start=("sample_order", "min"), end=("sample_order", "max"))
        .reset_index()
    )

    lines = [
        r"\documentclass[tikz,border=5pt]{standalone}",
        r"\pdfinfoomitdate=1",
        r"\pdftrailerid{}",
        r"\pdfsuppressptexinfo=-1",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{xcolor}",
        r"\definecolor{matrixmissing}{HTML}{F2F2F2}",
        r"\definecolor{matrixone}{HTML}{56B4E9}",
        r"\definecolor{matrixtwo}{HTML}{0072B2}",
        r"\definecolor{matrixthree}{HTML}{D55E00}",
        r"\definecolor{matrixgrid}{HTML}{B8B8B8}",
        rf"% {FIGURE_ID}; data_sha256={data_hash}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=0.62cm,y=0.18cm]",
        (
            r"\node[font=\sffamily\bfseries\large,anchor=south west] at (0,75.0) "
            r"{NATO field-trial SERS: sample $\times$ substrate $\times$ instrument coverage};"
        ),
        (
            r"\node[font=\sffamily\small,anchor=south west,text=black!70] "
            r"at (0,72.0) {69 physical-master split units; cell shade is the number "
            r"of stored spectra};"
        ),
    ]

    for substrate in SUBSTRATE_ORDER:
        x_offset, y_offset = panel_positions[substrate]
        subset = matrix[matrix["substrate_family"].eq(substrate)].copy()
        n_rows = int(subset["replicate_count"].sum())
        n_masters = int(subset.loc[subset["observed"], "sample_code"].nunique())
        lines.extend(
            [
                rf"\begin{{scope}}[shift={{({x_offset}cm,{y_offset}cm)}}]",
                (
                    r"\node[font=\sffamily\bfseries\small,anchor=south] at (5,70.4) {"
                    rf"{_tex(SUBSTRATE_LABEL[substrate])} ($n={n_rows}$ spectra; "
                    rf"$N={n_masters}$ samples)}};"
                ),
            ]
        )
        for row in subset.itertuples(index=False):
            x = int(row.instrument_order) - 1
            y = 69 - int(row.sample_order)
            color = level_color[int(row.display_level)]
            lines.append(
                rf"\filldraw[fill={color},draw=matrixgrid,line width=0.08pt] "
                rf"({x},{y}) rectangle ++(1,1);"
            )
        for instrument_index, instrument in enumerate(INSTRUMENT_ORDER):
            lines.append(
                r"\node[font=\sffamily\tiny,rotate=45,anchor=south west] at "
                rf"({instrument_index + 0.34},69.2) "
                rf"{{{_tex(INSTRUMENT_SHORT[instrument])}}};"
            )
        left_panel = substrate in {SUBSTRATE_ORDER[0], SUBSTRATE_ORDER[2]}
        if left_panel:
            for row in masters.itertuples(index=False):
                y = 69 - int(row.sample_order) + 0.5
                lines.append(
                    r"\node[font=\sffamily\fontsize{3.6}{3.8}\selectfont,anchor=east] "
                    rf"at (-0.12,{y}) {{{_tex(row.sample_code)}}};"
                )
            for group in group_bounds.itertuples(index=False):
                midpoint = 69 - (float(group.start) + float(group.end)) / 2 + 1
                label = GROUP_LABEL.get(
                    (group.station, group.target_analyte),
                    f"{group.station}: {group.target_analyte}",
                )
                lines.append(
                    r"\node[font=\sffamily\fontsize{3.5}{3.7}\selectfont,anchor=east,"
                    rf"text=black!70] at (-1.05,{midpoint}) {{{_tex(label)}}};"
                )
        for boundary in group_bounds["end"].iloc[:-1]:
            y = 69 - int(boundary)
            lines.append(
                rf"\draw[black!70,line width=0.35pt] (0,{y}) -- (10,{y});"
            )
        lines.extend(
            [
                r"\draw[black,line width=0.45pt] (0,0) rectangle (10,69);",
                r"\end{scope}",
            ]
        )

    lines.extend(
        [
            r"\begin{scope}[shift={(0cm,-15.8cm)}]",
            r"\filldraw[fill=matrixmissing,draw=matrixgrid] (0,0) rectangle ++(0.55,0.55);",
            r"\node[font=\sffamily\scriptsize,anchor=west] at (0.72,0.275) {missing};",
            r"\filldraw[fill=matrixone,draw=matrixgrid] (2.25,0) rectangle ++(0.55,0.55);",
            r"\node[font=\sffamily\scriptsize,anchor=west] at (2.97,0.275) {1 spectrum};",
            r"\filldraw[fill=matrixtwo,draw=matrixgrid] (5.35,0) rectangle ++(0.55,0.55);",
            r"\node[font=\sffamily\scriptsize,anchor=west] at (6.07,0.275) {2 spectra};",
            r"\filldraw[fill=matrixthree,draw=matrixgrid] (8.45,0) rectangle ++(0.55,0.55);",
            r"\node[font=\sffamily\scriptsize,anchor=west] at (9.17,0.275) {$\geq 3$ spectra};",
            (
                r"\node[font=\sffamily\scriptsize,anchor=west,text=black!70] "
                r"at (12.0,0.275) {A/M/P/R = Agilent/Mira/Pendar/RMX};"
            ),
            r"\end{scope}",
            r"\end{tikzpicture}",
            r"\end{document}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _colorscale() -> list[list[object]]:
    return [
        [0.0000, "#F2F2F2"],
        [0.1666, "#F2F2F2"],
        [0.1667, "#56B4E9"],
        [0.4999, "#56B4E9"],
        [0.5000, "#0072B2"],
        [0.8332, "#0072B2"],
        [0.8333, "#D55E00"],
        [1.0000, "#D55E00"],
    ]


def _write_html(
    matrix: pd.DataFrame,
    masters: pd.DataFrame,
    data_hash: str,
    output_path: Path,
    sample_label_mode: str,
) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[SUBSTRATE_LABEL[value] for value in SUBSTRATE_ORDER],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )
    locations = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for panel_index, (substrate, (row, column)) in enumerate(
        zip(SUBSTRATE_ORDER, locations, strict=True)
    ):
        subset = matrix[matrix["substrate_family"].eq(substrate)].copy()
        pivot = subset.pivot(
            index="sample_code", columns="instrument", values="display_level"
        ).reindex(index=masters["sample_code"], columns=INSTRUMENT_ORDER)
        counts = subset.pivot(
            index="sample_code", columns="instrument", values="replicate_count"
        ).reindex(index=masters["sample_code"], columns=INSTRUMENT_ORDER)
        meta = masters.set_index("sample_code").reindex(pivot.index)
        custom = np.empty((len(pivot), len(pivot.columns), 4), dtype=object)
        for y_index, sample_code in enumerate(pivot.index):
            for x_index, instrument in enumerate(pivot.columns):
                custom[y_index, x_index] = [
                    sample_code,
                    meta.loc[sample_code, "station"],
                    meta.loc[sample_code, "target_analyte"],
                    int(counts.loc[sample_code, instrument]),
                ]
        figure.add_trace(
            go.Heatmap(
                z=pivot.to_numpy(),
                x=INSTRUMENT_ORDER,
                y=pivot.index.tolist(),
                customdata=custom,
                zmin=0,
                zmax=3,
                colorscale=_colorscale(),
                colorbar=(
                    {
                        "title": "Spectra",
                        "tickvals": [0, 1, 2, 3],
                        "ticktext": ["missing", "1", "2", "≥3"],
                        "len": 0.40,
                        "y": 0.76,
                    }
                    if panel_index == 0
                    else None
                ),
                showscale=panel_index == 0,
                xgap=0.5,
                ygap=0.25,
                hovertemplate=(
                    "Sample: %{customdata[0]}<br>"
                    "Station: %{customdata[1]}<br>"
                    "Analyte: %{customdata[2]}<br>"
                    "Instrument: %{x}<br>"
                    f"Substrate: {html.escape(SUBSTRATE_LABEL[substrate])}<br>"
                    "Stored spectra: %{customdata[3]}<extra></extra>"
                ),
            ),
            row=row,
            col=column,
        )
        figure.update_yaxes(autorange="reversed", dtick=1, row=row, col=column)
        figure.update_xaxes(tickangle=-35, row=row, col=column)

    figure.update_layout(
        title={
            "text": "NATO field-trial SERS: sample × substrate × instrument coverage",
            "x": 0.5,
            "xanchor": "center",
        },
        width=1500,
        height=2400,
        margin={"l": 100, "r": 90, "t": 130, "b": 120},
        font={"family": "Arial, Helvetica, sans-serif", "size": 12},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    figure.update_annotations(font={"size": 15})
    plot = figure.to_html(
        full_html=False,
        include_plotlyjs="inline",
        config={"displaylogo": False, "responsive": True},
    )
    # Plotly's inline bundle contains a dormant CDN hostname string even though
    # no external script is loaded. Disable that fallback and satisfy the
    # repository's strictly offline HTML contract.
    plot = plot.replace("cdn.plot.ly", "offline.invalid")
    if sample_label_mode == "master-id":
        note_block = ""
    else:
        sample_note = (
            "69 physical-master split units, pseudonymized as S01–S69."
        )
        privacy_note = (
            "No raw master identifiers, filenames, source paths, observation "
            "identifiers, or operator identifiers are included."
        )
        hover_identifier = "sample code"
        note_block = f"""
  <div class="note">
    <p><strong>Scope:</strong> RQ-S07 secondary support audit. Each row is one of
    {sample_note} Each panel is a
    normalized substrate family; columns are the ten acquisition instruments.
    Cell color gives the number of stored spectra. Gray cells are unobserved
    combinations, not failed measurements.</p>
    <p><strong>Privacy:</strong> {privacy_note} Hover gives
    only {hover_identifier}, station, analyte, instrument, substrate family, and count.</p>
    <p><strong>Data SHA-256:</strong> <code>{data_hash}</code></p>
  </div>"""

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{FIGURE_ID}: NATO SERS sample × substrate × instrument coverage</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 1.2rem; color: #1f2933; }}
    .note {{ max-width: 1100px; line-height: 1.45; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <!-- data_sha256={data_hash}; NATO SERS {FIGURE_ID}; sample_label_mode={sample_label_mode} -->
  <h1>{FIGURE_ID}: NATO SERS sample × substrate × instrument coverage</h1>
{note_block}
  {plot}
</body>
</html>
"""
    document = "\n".join(line.rstrip() for line in document.splitlines()) + "\n"
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument(
        "--sample-label-mode",
        choices=("pseudonym", "master-id"),
        default="master-id",
    )
    args = parser.parse_args()

    figures_dir = args.plan_dir / "figures"
    data_dir = figures_dir / "data"
    tikz_dir = figures_dir / "tikz"
    html_dir = figures_dir / "html"
    for directory in (data_dir, tikz_dir, html_dir):
        directory.mkdir(parents=True, exist_ok=True)

    matrix, masters = _load_matrix(args.manifest, args.sample_label_mode)
    data_path = data_dir / f"{FIGURE_ID}_{SLUG}.csv"
    public_columns = [
        "figure_id",
        "scope",
        "research_question_id",
        "sample_code",
        "sample_order",
        "station",
        "target_analyte",
        "substrate_family",
        "substrate_label",
        "substrate_order",
        "instrument",
        "instrument_short",
        "instrument_order",
        "replicate_count",
        "observed",
        "display_level",
        "group_label",
    ]
    output_matrix = matrix[public_columns].copy()
    if args.sample_label_mode == "master-id":
        output_matrix = output_matrix.rename(columns={"sample_code": "master_sample_id"})
    output_matrix.to_csv(data_path, index=False, lineterminator="\n")
    data_hash = _sha256(data_path)

    tikz_path = tikz_dir / f"{FIGURE_ID}_{SLUG}.tex"
    html_path = html_dir / f"{FIGURE_ID}_{SLUG}.html"
    _write_tikz(matrix, masters, data_hash, tikz_path)
    _write_html(matrix, masters, data_hash, html_path, args.sample_label_mode)

    observed = matrix[matrix["observed"]].copy()
    master_instrument_counts = observed.groupby("sample_code")["instrument"].nunique()
    master_substrate_counts = observed.groupby("sample_code")["substrate_family"].nunique()
    crossover_masters: set[str] = set()
    crossover_cycles = 0
    for sample_code, sample_rows in observed.groupby("sample_code"):
        instrument_sets = {
            substrate: set(group["instrument"])
            for substrate, group in sample_rows.groupby("substrate_family")
        }
        for substrate_a, substrate_b in combinations(sorted(instrument_sets), 2):
            common_instruments = instrument_sets[substrate_a] & instrument_sets[substrate_b]
            if len(common_instruments) >= 2:
                crossover_masters.add(str(sample_code))
                crossover_cycles += comb(len(common_instruments), 2)

    substrate_summary = {}
    for substrate in SUBSTRATE_ORDER:
        subset = matrix[matrix["substrate_family"].eq(substrate)]
        subset_observed = subset[subset["observed"]]
        substrate_summary[substrate] = {
            "spectra": int(subset["replicate_count"].sum()),
            "physical_masters": int(subset_observed["sample_code"].nunique()),
            "instruments": int(subset_observed["instrument"].nunique()),
            "observed_cells": int(subset["observed"].sum()),
            "missing_cells": int((~subset["observed"]).sum()),
        }

    summary = {
        "figure_id": FIGURE_ID,
        "research_question_id": "RQ-S07",
        "scope": "S",
        "population": "primary_598",
        "independent_unit": "physical master group",
        "independent_samples": 69,
        "spectra": 598,
        "instruments": 10,
        "substrate_families": 4,
        "matrix_cells": int(len(matrix)),
        "observed_cells": int(matrix["observed"].sum()),
        "missing_cells": int((~matrix["observed"]).sum()),
        "masters_with_two_or_more_instruments": int((master_instrument_counts >= 2).sum()),
        "masters_with_two_or_more_substrates": int((master_substrate_counts >= 2).sum()),
        "masters_with_complete_two_substrate_two_instrument_crossover": len(crossover_masters),
        "two_substrate_two_instrument_crossover_cycles": int(crossover_cycles),
        "substrate_summary": substrate_summary,
        "data_sha256": data_hash,
        "privacy": "pseudonymous sample codes; no source identifiers or paths",
        "sample_label_mode": args.sample_label_mode,
    }
    if args.sample_label_mode == "master-id":
        summary["privacy"] = (
            "public project-owner-approved artifact with recorded master IDs; "
            "no source filenames, paths, observation IDs, or operator IDs"
        )
    (data_dir / f"{FIGURE_ID}_{SLUG}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
