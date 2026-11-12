#!/usr/bin/env python3
"""Add instrument-stratified preservation diagnostics to preprocessing artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from run_nato_sers_preprocessing_exploration import spectral_diagnostics, write_html


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    root = repository / "Workspace" / "nato_sers_raw_reexploration_v1"
    output = root / "preprocessing"
    raw_bundle = np.load(root / "arrays" / "unique_attributable_common_support_raw.npz")
    candidates = np.load(output / "preprocessing_candidates.npz")
    raw = raw_bundle["intensity"].astype(float)
    manifest = pd.read_csv(root / "tables" / "analysis_manifest.csv")
    names = [name for name in candidates.files if name != "axis_cm1"]
    rows = []
    for instrument in sorted(manifest.instrument.unique()):
        mask = manifest.instrument.eq(instrument).to_numpy()
        for name in names:
            rows.append({"instrument": instrument, "candidate": name,
                         **spectral_diagnostics(raw[mask], candidates[name][mask])})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "tables" / "preprocessing_metrics_by_instrument.csv", index=False)
    registry_path = output / "html_artifact_registry.json"
    registry = json.loads(registry_path.read_text())
    registry = [item for item in registry if not item["file"].startswith("instrument_metric_")]
    for metric, label in (
        ("median_raw_shape_correlation", "Median raw-shape correlation"),
        ("top_peak_location_recall_pm5cm1", "Prominent-peak recall within ±5 cm⁻¹"),
        ("median_baseline_span", "Residual baseline-span proxy"),
        ("median_first_difference_roughness", "First-difference roughness proxy"),
    ):
        table = frame.pivot(index="candidate", columns="instrument", values=metric)
        figure = go.Figure(go.Heatmap(
            z=table.to_numpy(), x=table.columns, y=table.index, colorscale="Viridis",
            text=np.round(table.to_numpy(), 3), texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>value=%{z:.5g}<extra></extra>",
        ))
        figure.update_layout(title=f"{label} by instrument")
        filename = f"instrument_metric_{metric}.html"
        write_html(output / "html" / filename, figure)
        registry.append({"file": filename, "title": f"{label} by instrument",
                         "description": "Stratified preservation/damage diagnostic; compare candidates within each instrument."})
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    links = "\n".join(
        f'<li><a href="html/{item["file"]}">{item["title"]}</a> — {item["description"]}</li>'
        for item in registry
    )
    (output / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>NATO SERS preprocessing</title>"
        "<style>body{font:16px Arial;max-width:1100px;margin:40px auto;line-height:1.5}li{margin:12px 0}</style>"
        f"</head><body><h1>NATO SERS preprocessing exploration</h1>"
        f"<p>No candidate is selected by visual appearance alone.</p><ol>{links}</ol></body></html>"
    )
    print(frame.groupby("candidate").agg(
        worst_peak_recall=("top_peak_location_recall_pm5cm1", "min"),
        worst_shape_correlation=("median_raw_shape_correlation", "min"),
        median_baseline_span=("median_baseline_span", "median"),
    ).sort_values("worst_peak_recall", ascending=False).to_string())


if __name__ == "__main__":
    main()
