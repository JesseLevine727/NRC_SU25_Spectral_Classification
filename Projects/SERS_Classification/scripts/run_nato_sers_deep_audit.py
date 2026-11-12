#!/usr/bin/env python3
"""Deep metadata, quality, replicate, alignment, and stability audit for NATO SERS."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

import build_nato_sers_field_trial as source_parser


COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#000000"]
PLOT_CONFIG = {"responsive": True, "scrollZoom": True, "displaylogo": False,
               "toImageButtonOptions": {"format": "svg", "scale": 1}}


def write_html(path: Path, figure: go.Figure) -> None:
    figure.update_layout(template="plotly_white", font={"family": "Arial", "size": 14},
                         margin={"l": 80, "r": 30, "t": 80, "b": 70})
    figure.write_html(path, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG)


def metadata_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "session", "team", "operator_initials", "recorded_time", "station", "instrument",
        "normal_raman_flag", "sensor_family", "sensor_variant", "instrument_result",
        "comments", "target_detected_log", "paper_sheet", "scenario", "master_description",
        "target_analyte", "sample_matrix", "carrier_geometry", "nominal_concentration",
        "instrument_serial", "firmware_package", "software_version", "integration_time",
        "auto_integration", "averages", "laser_power", "smart_tip_type", "system_suitability",
        "measurement_duration", "instrument_start_date", "instrument_start_time",
        "rmx_timestamp", "rmx_ccdbias", "rmx_ccdgain", "rmx_sigmaread", "rmx_laserwavenum",
        "rmx_wavenumcorrection", "rmx_scancount", "rmx_totalexposurems", "rmx_singleexposurems",
        "rmx_measurementstate", "rmx_scanmode", "rmx_replacelist", "rmx_peaks",
    ]
    rows = []
    for field in fields:
        if field not in frame:
            rows.append({"field": field, "present": False, "nonmissing": 0, "coverage": 0.0,
                         "unique_nonmissing": 0, "example_values": "[]", "coverage_by_instrument": "{}"})
            continue
        series = frame[field]
        examples = series.dropna().astype(str).value_counts().head(8).index.tolist()
        coverage = frame.groupby("instrument")[field].apply(lambda values: float(values.notna().mean())).to_dict()
        rows.append({"field": field, "present": True, "nonmissing": int(series.notna().sum()),
                     "coverage": float(series.notna().mean()), "unique_nonmissing": int(series.nunique(dropna=True)),
                     "example_values": json.dumps(examples), "coverage_by_instrument": json.dumps(coverage, sort_keys=True)})
    return pd.DataFrame(rows)


def quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["relative_noise"] = work.first_difference_noise_mad / work.intensity_range.clip(lower=np.finfo(float).eps)
    work["system_suitability_fail"] = work.get("system_suitability", pd.Series(index=work.index, dtype=object)).astype(str).str.lower().str.startswith("fail")
    metrics = ["intensity_range", "baseline_energy_fraction_proxy", "baseline_span_fraction_proxy",
               "relative_noise", "spike_fraction_proxy", "negative_fraction", "repeated_max_fraction",
               "leading_constant_points"]
    rows = []
    for instrument, group in work.groupby("instrument"):
        row = {"instrument": instrument, "n_spectra": len(group),
               "n_masters": group.master_sample_id.nunique(),
               "manual_severe_fraction": float(group.manual_severe_qc_flag.mean()),
               "manual_low_signal_fraction": float(group.manual_low_signal_or_noise_flag.mean()),
               "system_suitability_fail_fraction": float(group.system_suitability_fail.mean())}
        for metric in metrics:
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_q10"] = float(group[metric].quantile(.10))
            row[f"{metric}_q90"] = float(group[metric].quantile(.90))
        rows.append(row)
    return pd.DataFrame(rows), work


def lag_cm1(left: np.ndarray, right: np.ndarray, max_lag: int = 10) -> tuple[int, float]:
    left = savgol_filter(left, 11, 3, deriv=1)
    right = savgol_filter(right, 11, 3, deriv=1)
    left = (left - left.mean()) / max(left.std(), 1e-12)
    right = (right - right.mean()) / max(right.std(), 1e-12)
    values = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            values.append(float(np.mean(left[-lag:] * right[:lag])))
        elif lag > 0:
            values.append(float(np.mean(left[:-lag] * right[lag:])))
        else:
            values.append(float(np.mean(left * right)))
    best = int(np.argmax(values))
    return best - max_lag, values[best]


def replicate_pairs(manifest: pd.DataFrame, matrices: dict[str, np.ndarray]) -> pd.DataFrame:
    normalized = {name: matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
                  for name, matrix in matrices.items()}
    rows = []
    indexed = manifest.reset_index().rename(columns={"index": "matrix_index"})
    for master, group in indexed.groupby("master_sample_id"):
        for left, right in combinations(group.itertuples(), 2):
            i, j = left.matrix_index, right.matrix_index
            lag, peak_correlation = lag_cm1(matrices["despike_savgol_minmax"][i], matrices["despike_savgol_minmax"][j])
            row = {"master_sample_id": master, "target_analyte": left.target_analyte,
                   "instrument_left": left.instrument, "instrument_right": right.instrument,
                   "same_instrument": left.instrument == right.instrument,
                   "sensor_left": left.sensor_family, "sensor_right": right.sensor_family,
                   "same_sensor_family": left.sensor_family == right.sensor_family,
                   "estimated_peak_lag_cm1": lag, "peak_derivative_correlation": peak_correlation}
            for name, matrix in normalized.items():
                row[f"cosine_{name}"] = float(np.dot(matrix[i], matrix[j]))
            rows.append(row)
    return pd.DataFrame(rows)


def pair_separation(manifest: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    norm = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    rows = []
    for i, j in combinations(range(len(manifest)), 2):
        if manifest.master_sample_id.iloc[i] == manifest.master_sample_id.iloc[j]:
            category = "same physical master"
        elif manifest.target_analyte.iloc[i] == manifest.target_analyte.iloc[j]:
            category = "same target, different master"
        else:
            category = "different target"
        rows.append({"category": category, "cosine_similarity": float(np.dot(norm[i], norm[j]))})
    return pd.DataFrame(rows)


def cluster_stability(manifest: pd.DataFrame, candidates: dict[str, np.ndarray], seed: int) -> pd.DataFrame:
    rows = []; rng = np.random.default_rng(seed); masters = manifest.master_sample_id.unique()
    for name, matrix in candidates.items():
        features = PCA(n_components=.95, svd_solver="full").fit_transform(matrix)
        label_sets = []
        for run in range(30):
            selected_masters = rng.choice(masters, size=int(np.ceil(.8 * len(masters))), replace=False)
            selected = manifest.master_sample_id.isin(selected_masters).to_numpy()
            model = KMeans(n_clusters=7, n_init=20, random_state=seed + run).fit(features[selected])
            label_sets.append(model.predict(features))
        aris = [adjusted_rand_score(a, b) for a, b in combinations(label_sets, 2)]
        labels = label_sets[0]
        rows.append({"representation": name, "method": "KMeans k=7, 80% master subsampling",
                     "runs": 30, "median_pairwise_stability_ari": float(np.median(aris)),
                     "q10_pairwise_stability_ari": float(np.quantile(aris, .1)),
                     "q90_pairwise_stability_ari": float(np.quantile(aris, .9)),
                     "target_nmi": float(normalized_mutual_info_score(manifest.target_analyte, labels)),
                     "instrument_nmi": float(normalized_mutual_info_score(manifest.instrument, labels)),
                     "sensor_nmi": float(normalized_mutual_info_score(manifest.sensor_family, labels))})
        for min_cluster in (8, 12, 20):
            for min_samples in (3, 5, 10):
                labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster, min_samples=min_samples).fit_predict(features)
                nonnoise = labels != -1
                silhouette = np.nan
                if nonnoise.sum() > 2 and len(np.unique(labels[nonnoise])) > 1:
                    silhouette = silhouette_score(features[nonnoise], labels[nonnoise])
                rows.append({"representation": name,
                             "method": f"HDBSCAN mcs={min_cluster},ms={min_samples}", "runs": 1,
                             "clusters_including_noise": len(np.unique(labels)),
                             "noise_fraction": float(np.mean(labels == -1)), "silhouette_nonnoise": silhouette,
                             "target_nmi": float(normalized_mutual_info_score(manifest.target_analyte, labels)),
                             "instrument_nmi": float(normalized_mutual_info_score(manifest.instrument, labels)),
                             "sensor_nmi": float(normalized_mutual_info_score(manifest.sensor_family, labels))})
    return pd.DataFrame(rows)


def scatter(frame, x, y, color, title):
    fig = go.Figure(); values = frame[color].astype(str)
    hover = frame[["observation_uid", "master_sample_id", "target_analyte", "instrument", "sensor_family", "station"]].astype(str).to_numpy()
    for idx, category in enumerate(sorted(values.unique())):
        mask = values.eq(category)
        fig.add_trace(go.Scattergl(x=frame.loc[mask, x], y=frame.loc[mask, y], mode="markers", name=category,
                                   marker={"size": 8, "opacity": .75, "color": COLORS[idx % len(COLORS)]},
                                   customdata=hover[mask], hovertemplate="%{customdata[0]}<br>master=%{customdata[1]}<br>target=%{customdata[2]}<br>instrument=%{customdata[3]}<br>sensor=%{customdata[4]}<br>station=%{customdata[5]}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, legend_title=color); return fig


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path,
                        default=repository / "Workspace" / "nato_sers_raw_reexploration_v1")
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args(); root = args.workspace.resolve()
    output = root / "deep_audit"; tables = output / "tables"; html_dir = output / "html"
    tables.mkdir(parents=True, exist_ok=True); html_dir.mkdir(exist_ok=True)
    manifest = pd.read_csv(root / "tables" / "analysis_manifest.csv")
    all_manifest = pd.read_csv(root / "tables" / "recordings_manifest_fresh.csv")
    raw_bundle = np.load(root / "arrays" / "unique_attributable_common_support_raw.npz")
    axis, raw = raw_bundle["axis_cm1"], raw_bundle["intensity"]
    prep_bundle = np.load(root / "preprocessing" / "preprocessing_candidates.npz")
    candidates = {name: prep_bundle[name] for name in prep_bundle.files if name != "axis_cm1"}
    metadata = metadata_catalog(manifest); metadata.to_csv(tables / "metadata_field_catalog_selected.csv", index=False)
    metadata_catalog(all_manifest).to_csv(tables / "metadata_field_catalog_all_recordings.csv", index=False)
    source_root = Path(json.loads((root / "initial_audit_summary.json").read_text())["source_root"])
    workbook_rows = []
    for workbook in sorted(source_root.glob("*.xlsx")):
        for sheet, rows in source_parser.read_xlsx_rows(workbook).items():
            headers = []
            for _, row in source_parser.tabular_rows(rows):
                headers = list(row)
                break
            workbook_rows.append({"workbook": workbook.name, "sheet": sheet,
                                  "xml_rows": len(rows), "tabular_columns": json.dumps(headers),
                                  "tabular_column_count": len(headers)})
    pd.DataFrame(workbook_rows).to_csv(tables / "workbook_sheet_audit.csv", index=False)
    inventory = pd.read_csv(root / "tables" / "raw_file_inventory.csv")
    inventory.groupby(["top_level", "role", "extension"], dropna=False).size().reset_index(
        name="files").to_csv(tables / "file_role_summary.csv", index=False)
    note_lines = [line.strip() for line in (source_root / "Notes.txt").read_text().splitlines() if line.strip()]
    pd.DataFrame({"line_number": range(1, len(note_lines) + 1), "note_text": note_lines}).to_csv(
        tables / "source_notes_transcription.csv", index=False)
    quality, quality_rows = quality_summary(manifest); quality.to_csv(tables / "quality_by_instrument.csv", index=False)
    quality_rows.to_csv(tables / "quality_rows.csv", index=False)
    pair_matrices = {name: candidates[name] for name in
                     ["minimal_minmax", "despike_savgol_minmax", "arpls_savgol_minmax"]}
    pairs = replicate_pairs(manifest, pair_matrices); pairs.to_csv(tables / "within_master_pair_diagnostics.csv", index=False)
    separation = pair_separation(manifest, candidates["despike_savgol_minmax"])
    separation.groupby("category").cosine_similarity.agg(["count", "median", "mean", "std",
                                                           lambda x: x.quantile(.1), lambda x: x.quantile(.9)]).to_csv(
        tables / "pair_similarity_summary.csv")
    stability = cluster_stability(manifest, {name: candidates[name] for name in
        ["minimal_minmax", "despike_savgol_minmax", "arpls_savgol_minmax", "minimal_area"]}, args.seed)
    stability.to_csv(tables / "cluster_stability_and_hdbscan_grid.csv", index=False)

    registry = []
    def save(name, title, description, fig):
        write_html(html_dir / name, fig); registry.append({"file": name, "title": title, "description": description})

    # Every selected spectrum, separately by instrument, both raw and recommended-candidate display.
    target_colors = {target: COLORS[idx % len(COLORS)] for idx, target in enumerate(sorted(manifest.target_analyte.unique()))}
    for instrument_index, instrument in enumerate(sorted(manifest.instrument.unique())):
        mask = manifest.instrument.eq(instrument).to_numpy()
        for mode, matrix in (("raw", raw), ("smoothed_minmax", candidates["despike_savgol_minmax"])):
            fig = go.Figure(); seen = set()
            for row_index in np.flatnonzero(mask):
                target = manifest.target_analyte.iloc[row_index]
                fig.add_trace(go.Scattergl(x=axis, y=matrix[row_index], mode="lines", name=target,
                                           legendgroup=target, showlegend=target not in seen,
                                           line={"width": .8, "color": target_colors[target]}, opacity=.25,
                                           customdata=np.full((len(axis), 1), manifest.observation_uid.iloc[row_index]),
                                           hovertemplate="%{customdata[0]}<br>%{x:.0f} cm⁻¹<br>%{y:.5g}<extra></extra>")); seen.add(target)
            fig.update_layout(title=f"{instrument}: all {mask.sum()} selected spectra ({mode})",
                              xaxis_title="Raman shift (cm⁻¹)", yaxis_title="Intensity")
            save(f"{len(registry)+1:02d}_{instrument}_{mode}_all.html", f"{instrument}: all {mode} spectra",
                 "All unique attributable spectra; opacity reveals density and outliers.", fig)

    for metric, title, log_axis in (
        ("intensity_range", "Native intensity dynamic range", True),
        ("baseline_energy_fraction_proxy", "Baseline-energy fraction proxy", False),
        ("baseline_span_fraction_proxy", "Baseline-span fraction proxy", False),
        ("relative_noise", "First-difference noise relative to intensity range", True),
        ("spike_fraction_proxy", "Impulse/spike fraction proxy", True),
        ("negative_fraction", "Negative-intensity fraction", False),
        ("repeated_max_fraction", "Repeated-maximum/saturation proxy", True),
    ):
        fig = go.Figure()
        for idx, instrument in enumerate(sorted(manifest.instrument.unique())):
            values = quality_rows.loc[quality_rows.instrument.eq(instrument), metric]
            fig.add_trace(go.Box(y=values, name=instrument, boxpoints="outliers", marker_color=COLORS[idx % len(COLORS)]))
        fig.update_layout(title=title, yaxis_title=metric, yaxis_type="log" if log_axis else "linear")
        save(f"{len(registry)+1:02d}_quality_{metric}.html", title,
             "Instrument-stratified native-spectrum diagnostic; inspect outliers before any resampling.", fig)

    # Pair similarity and instrument-pair replicate heatmaps.
    fig = go.Figure()
    for idx, category in enumerate(["same physical master", "same target, different master", "different target"]):
        values = separation.loc[separation.category.eq(category), "cosine_similarity"]
        fig.add_trace(go.Violin(y=values, name=category, box_visible=True, meanline_visible=True,
                                points=False, line_color=COLORS[idx]))
    fig.update_layout(title="Does repeated-measure similarity exceed chemical-class similarity?",
                      yaxis_title="Cosine similarity after despike + smoothing + min–max")
    save(f"{len(registry)+1:02d}_pair_similarity.html", "Pair similarity hierarchy",
         "Contrasts repeated views, independent masters of the same target, and different targets.", fig)

    cross = pairs[~pairs.same_instrument].copy()
    cross["instrument_pair"] = cross.apply(lambda r: " | ".join(sorted([r.instrument_left, r.instrument_right])), axis=1)
    pair_summary = cross.groupby("instrument_pair").agg(
        n_pairs=("master_sample_id", "size"), median_cosine=("cosine_despike_savgol_minmax", "median"),
        median_signed_lag_cm1=("estimated_peak_lag_cm1", "median"),
        q10_signed_lag_cm1=("estimated_peak_lag_cm1", lambda x: float(np.quantile(x, .1))),
        q90_signed_lag_cm1=("estimated_peak_lag_cm1", lambda x: float(np.quantile(x, .9))),
        median_abs_lag_cm1=("estimated_peak_lag_cm1", lambda x: float(np.median(np.abs(x)))),
        q90_abs_lag_cm1=("estimated_peak_lag_cm1", lambda x: float(np.quantile(np.abs(x), .9))),
        fraction_abs_lag_le_1=("estimated_peak_lag_cm1", lambda x: float(np.mean(np.abs(x) <= 1))),
        median_peak_derivative_correlation=("peak_derivative_correlation", "median")).reset_index()
    pair_summary.to_csv(tables / "cross_instrument_pair_summary.csv", index=False)
    for value, title in (("median_cosine", "Cross-instrument replicate cosine similarity"),
                         ("median_abs_lag_cm1", "Estimated cross-instrument peak lag (absolute cm⁻¹)")):
        instruments = sorted(manifest.instrument.unique()); grid = np.full((len(instruments), len(instruments)), np.nan)
        for row in pair_summary.itertuples():
            left, right = row.instrument_pair.split(" | "); i, j = instruments.index(left), instruments.index(right)
            grid[i, j] = grid[j, i] = getattr(row, value)
        fig = go.Figure(go.Heatmap(z=grid, x=instruments, y=instruments, colorscale="Viridis",
                                   text=np.round(grid, 2), texttemplate="%{text}", hoverongaps=False))
        fig.update_layout(title=title)
        save(f"{len(registry)+1:02d}_{value}_heatmap.html", title,
             "Computed only where the same physical masters were measured by both instruments.", fig)

    # Candidate embedding diagnostics.
    embedding_specs = {
        "despike_savgol_minmax": (True, True),
        "arpls_savgol_minmax": (True, False),
        "minimal_area": (False, False),
    }
    embedding_rows = []
    for name, (run_umap, run_tsne) in embedding_specs.items():
        features = PCA(n_components=.95, svd_solver="full").fit_transform(candidates[name])
        pca2 = features[:, :2]
        methods = {"pca": pca2}
        if run_umap:
            methods["umap"] = umap.UMAP(n_components=2, n_neighbors=15, min_dist=.1, metric="cosine",
                                         random_state=args.seed, n_jobs=1).fit_transform(features)
        if run_tsne:
            methods["tsne"] = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                                     random_state=args.seed).fit_transform(features)
        frame = manifest.copy()
        for method, values in methods.items():
            frame[f"{method}_1"], frame[f"{method}_2"] = values[:, 0], values[:, 1]
            for color in ("target_analyte", "instrument", "sensor_family", "station"):
                save(f"{len(registry)+1:02d}_{name}_{method}_{color}.html",
                     f"{name} {method.upper()} by {color}",
                     "Candidate-preprocessed embedding; geometry is exploratory and hover preserves provenance.",
                     scatter(frame, f"{method}_1", f"{method}_2", color,
                             f"{name}: {method.upper()}, colored by {color}"))
            embedding_rows.append({"representation": name, "method": method,
                                   "components_to_95pct": features.shape[1]})
    pd.DataFrame(embedding_rows).to_csv(tables / "embedding_diagnostics.csv", index=False)

    (output / "html_artifact_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    links = "\n".join(f'<li><a href="html/{item["file"]}">{item["title"]}</a> — {item["description"]}</li>' for item in registry)
    (output / "index.html").write_text("<!doctype html><html><head><meta charset='utf-8'><title>NATO SERS deep audit</title>"
        "<style>body{font:16px Arial;max-width:1150px;margin:40px auto;line-height:1.5}li{margin:11px 0}</style></head>"
        f"<body><h1>NATO SERS deep audit</h1><p>All-spectrum, metadata, quality, replicate, alignment, stability, and candidate-embedding evidence.</p><ol>{links}</ol></body></html>")
    summary = {"selected_spectra": len(manifest), "physical_masters": manifest.master_sample_id.nunique(),
               "metadata_fields_audited": len(metadata), "within_master_pairs": len(pairs),
               "cross_instrument_pairs": len(cross), "html_plots": len(registry),
               "overall_alignment": {"median_abs_lag_cm1": float(np.median(np.abs(cross.estimated_peak_lag_cm1))),
                                     "q90_abs_lag_cm1": float(np.quantile(np.abs(cross.estimated_peak_lag_cm1), .9)),
                                     "fraction_abs_lag_le_1": float(np.mean(np.abs(cross.estimated_peak_lag_cm1) <= 1))}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(stability[stability.method.str.startswith("KMeans")].to_string(index=False))


if __name__ == "__main__":
    main()
