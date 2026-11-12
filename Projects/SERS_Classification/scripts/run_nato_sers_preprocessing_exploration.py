#!/usr/bin/env python3
"""Compare SERS preprocessing candidates without choosing by visual appeal alone."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import sparse
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse.linalg import spsolve
from scipy.ndimage import percentile_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PLOT_CONFIG = {"responsive": True, "scrollZoom": True, "displaylogo": False,
               "toImageButtonOptions": {"format": "svg", "scale": 1}}
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
          "#56B4E9", "#000000", "#6A3D9A", "#A6CEE3", "#FB9A99", "#B2DF8A"]


def row_minmax(matrix: np.ndarray) -> np.ndarray:
    low = matrix.min(axis=1, keepdims=True)
    span = np.maximum(matrix.max(axis=1, keepdims=True) - low, np.finfo(float).eps)
    return (matrix - low) / span


def row_snv(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    return centered / np.maximum(centered.std(axis=1, keepdims=True), np.finfo(float).eps)


def row_vector(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), np.finfo(float).eps)


def row_area(matrix: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Non-negative integrated-area normalization, robust to small negative offsets."""
    shifted = matrix - matrix.min(axis=1, keepdims=True)
    area = np.trapezoid(shifted, x=axis, axis=1)[:, None]
    return shifted / np.maximum(area, np.finfo(float).eps)


def despike(matrix: np.ndarray, window: int, threshold: float) -> np.ndarray:
    result = matrix.copy()
    half = window // 2
    for index, row in enumerate(matrix):
        median = pd.Series(row).rolling(window, center=True, min_periods=1).median().to_numpy()
        residual = row - median
        mad = np.median(np.abs(residual - np.median(residual)))
        flagged = np.abs(residual) > threshold * max(mad / 0.67448975, np.finfo(float).eps)
        # Replace only isolated impulses; contiguous Raman peaks are retained.
        neighborhood = np.convolve(flagged.astype(int), np.ones(3, dtype=int), mode="same")
        isolated = flagged & (neighborhood <= 2)
        result[index, isolated] = median[isolated]
    return result


def penalty_matrix(points: int) -> sparse.csc_matrix:
    difference = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(points - 2, points))
    return (difference.T @ difference).tocsc()


def asls_baseline(y: np.ndarray, penalty: sparse.csc_matrix, lam: float,
                  asymmetry: float, iterations: int) -> np.ndarray:
    weights = np.ones(len(y))
    for _ in range(iterations):
        system = sparse.diags(weights, format="csc") + lam * penalty
        baseline = spsolve(system, weights * y)
        weights = asymmetry * (y > baseline) + (1 - asymmetry) * (y <= baseline)
    return baseline


def arpls_baseline(y: np.ndarray, penalty: sparse.csc_matrix, lam: float,
                   iterations: int) -> np.ndarray:
    weights = np.ones(len(y))
    for _ in range(iterations):
        baseline = spsolve(sparse.diags(weights, format="csc") + lam * penalty, weights * y)
        negative = (y - baseline)[y < baseline]
        if len(negative) < 2 or np.std(negative) <= np.finfo(float).eps:
            break
        mean, std = float(np.mean(negative)), float(np.std(negative))
        argument = np.clip(2.0 * ((y - baseline) - (2 * std - mean)) / std, -60, 60)
        new_weights = 1.0 / (1.0 + np.exp(argument))
        if np.linalg.norm(new_weights - weights) / max(np.linalg.norm(weights), 1e-12) < 1e-3:
            weights = new_weights
            break
        weights = new_weights
    return baseline


def rubberband_baseline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    hull: list[int] = []
    for index in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            left = (x[b] - x[a]) * (y[index] - y[b])
            right = (y[b] - y[a]) * (x[index] - x[b])
            if left >= right:
                hull.pop()
            else:
                break
        hull.append(index)
    return np.interp(x, x[hull], y[hull])


def preprocess_all(axis: np.ndarray, raw: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    window = int(config["savgol_window"])
    polynomial = int(config["savgol_polynomial"])
    clean = despike(raw, int(config["despike_window"]), float(config["despike_mad_threshold"]))
    smooth = savgol_filter(clean, window, polynomial, axis=1, mode="interp")
    penalty = penalty_matrix(raw.shape[1])
    asls = np.empty_like(raw)
    arpls = np.empty_like(raw)
    rubber = np.empty_like(raw)
    for index, row in enumerate(clean):
        asls[index] = row - asls_baseline(
            row, penalty, float(config["baseline_lambda"]),
            float(config["baseline_asymmetry"]), int(config["baseline_iterations"])
        )
        arpls[index] = row - arpls_baseline(
            row, penalty, float(config["baseline_lambda"]), int(config["baseline_iterations"])
        )
        rubber[index] = row - rubberband_baseline(axis, row)
    arpls_smooth = savgol_filter(arpls, window, polynomial, axis=1, mode="interp")
    derivative = savgol_filter(clean, window, polynomial, deriv=1, delta=float(np.median(np.diff(axis))), axis=1)
    return {
        "minimal_minmax": row_minmax(raw),
        "minimal_snv": row_snv(raw),
        "minimal_vector": row_vector(raw),
        "minimal_area": row_area(raw, axis),
        "despike_minmax": row_minmax(clean),
        "despike_savgol_minmax": row_minmax(smooth),
        "rubberband_minmax": row_minmax(rubber),
        "asls_minmax": row_minmax(asls),
        "arpls_minmax": row_minmax(arpls),
        "arpls_savgol_minmax": row_minmax(arpls_smooth),
        "arpls_savgol_snv": row_snv(arpls_smooth),
        "savgol_derivative1_snv": row_snv(derivative),
    }


def peak_recall(reference: np.ndarray, candidate: np.ndarray) -> float:
    recalls = []
    for ref, test in zip(reference, candidate):
        ref_s = savgol_filter(row_minmax(ref[None, :])[0], 11, 3)
        test_s = savgol_filter(row_minmax(test[None, :])[0], 11, 3)
        peaks, props = find_peaks(ref_s, prominence=0.04, distance=4)
        if not len(peaks):
            continue
        order = np.argsort(props["prominences"])[-10:]
        peaks = peaks[order]
        found, _ = find_peaks(test_s, prominence=0.025, distance=3)
        recalls.append(np.mean([np.any(np.abs(found - point) <= 5) for point in peaks]))
    return float(np.mean(recalls))


def replicate_cosine(matrix: np.ndarray, manifest: pd.DataFrame) -> tuple[float, float]:
    normalized = row_vector(matrix)
    within, cross = [], []
    for _, group in manifest.reset_index().groupby("master_sample_id"):
        for left, right in combinations(group.itertuples(), 2):
            similarity = float(np.dot(normalized[left.index], normalized[right.index]))
            within.append(similarity)
            if left.instrument != right.instrument:
                cross.append(similarity)
    return float(np.mean(within)), float(np.mean(cross))


def grouped_diagnostic(matrix: np.ndarray, labels: pd.Series, groups: pd.Series,
                       folds: int, seed: int, stratified: bool) -> tuple[float, float]:
    splitter = (StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
                if stratified else GroupKFold(n_splits=folds))
    truth, predictions = [], []
    for train, test in splitter.split(matrix, labels, groups):
        model = make_pipeline(
            StandardScaler(),
            RidgeClassifier(alpha=1.0, class_weight="balanced", solver="lsqr"),
        )
        model.fit(matrix[train], labels.iloc[train])
        predictions.extend(model.predict(matrix[test]))
        truth.extend(labels.iloc[test])
    return (float(balanced_accuracy_score(truth, predictions)),
            float(f1_score(truth, predictions, average="macro", zero_division=0)))


def cluster_diagnostics(matrix: np.ndarray, manifest: pd.DataFrame, seed: int) -> dict[str, float]:
    features = PCA(n_components=0.95, svd_solver="full").fit_transform(matrix)
    labels_by_seed = []
    for offset in range(5):
        labels_by_seed.append(KMeans(n_clusters=7, n_init=10, random_state=seed + offset).fit_predict(features))
    labels = labels_by_seed[0]
    stability = [adjusted_rand_score(labels, other) for other in labels_by_seed[1:]]
    return {
        "kmeans7_silhouette": float(silhouette_score(features, labels)),
        "kmeans7_seed_stability_ari": float(np.mean(stability)),
        "kmeans7_target_nmi": float(normalized_mutual_info_score(manifest.target_analyte, labels)),
        "kmeans7_instrument_nmi": float(normalized_mutual_info_score(manifest.instrument, labels)),
        "kmeans7_sensor_nmi": float(normalized_mutual_info_score(manifest.sensor_family, labels)),
    }


def spectral_diagnostics(raw: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    scaled = row_minmax(candidate)
    reference = row_minmax(raw)
    correlation = [np.corrcoef(a, b)[0, 1] for a, b in zip(reference, scaled)]
    roughness = np.median(np.abs(np.diff(scaled, axis=1)), axis=1)
    baseline = percentile_filter(scaled, percentile=10, size=(1, 101), mode="nearest")
    baseline_span = np.ptp(baseline, axis=1)
    return {
        "median_raw_shape_correlation": float(np.nanmedian(correlation)),
        "top_peak_location_recall_pm5cm1": peak_recall(raw, candidate),
        "median_first_difference_roughness": float(np.median(roughness)),
        "median_baseline_span": float(np.median(baseline_span)),
    }


def write_html(path: Path, figure: go.Figure) -> None:
    figure.update_layout(template="plotly_white", font={"family": "Arial", "size": 14},
                         margin={"l": 80, "r": 30, "t": 80, "b": 70})
    figure.write_html(path, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG)


def make_plots(output: Path, axis: np.ndarray, raw: np.ndarray, manifest: pd.DataFrame,
               candidates: dict[str, np.ndarray], metrics: pd.DataFrame,
               range_metrics: pd.DataFrame) -> list[dict[str, str]]:
    html_dir = output / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    def save(name: str, title: str, description: str, fig: go.Figure) -> None:
        write_html(html_dir / name, fig)
        artifacts.append({"file": name, "title": title, "description": description})

    columns = ["target_balanced_accuracy", "instrument_balanced_accuracy",
               "cross_instrument_master_cosine", "top_peak_location_recall_pm5cm1",
               "median_baseline_span", "median_first_difference_roughness"]
    z = metrics.set_index("candidate")[columns]
    normalized = (z - z.min()) / (z.max() - z.min()).replace(0, 1)
    # Lower is preferable only for nuisance predictability, baseline span, and roughness.
    for col in ["instrument_balanced_accuracy", "median_baseline_span", "median_first_difference_roughness"]:
        normalized[col] = 1 - normalized[col]
    fig = go.Figure(go.Heatmap(z=normalized.to_numpy(), x=columns, y=normalized.index,
                               colorscale="Viridis", zmin=0, zmax=1,
                               text=np.round(z.to_numpy(), 3), texttemplate="%{text}",
                               hovertemplate="%{y}<br>%{x}<br>value=%{text}<extra></extra>"))
    fig.update_layout(title="Preprocessing diagnostic scorecard (color direction adjusted; text is raw value)")
    save("01_preprocessing_scorecard.html", "Preprocessing scorecard",
         "Color is better-to-worse per diagnostic; values are raw and no scalar winner is imposed.", fig)

    fig = go.Figure()
    for idx, row in metrics.iterrows():
        fig.add_trace(go.Scatter(x=[row.instrument_balanced_accuracy], y=[row.target_balanced_accuracy],
                                 mode="markers+text", text=[row.candidate], textposition="top center",
                                 marker={"size": 11, "color": COLORS[idx % len(COLORS)]},
                                 name=row.candidate))
    fig.update_layout(title="Chemistry recognition versus residual instrument predictability",
                      xaxis_title="Instrument balanced accuracy (lower nuisance leakage is preferable)",
                      yaxis_title="Target balanced accuracy (higher is preferable)", showlegend=False)
    save("02_chemistry_vs_instrument.html", "Chemistry vs instrument signal",
         "Both diagnostics use physical-master-grouped folds; this is exploratory, not a final test estimate.", fig)

    fig = go.Figure()
    for candidate, group in range_metrics.groupby("candidate"):
        fig.add_trace(go.Scatter(x=group["range"], y=group["target_balanced_accuracy"],
                                 mode="lines+markers", name=candidate))
    fig.update_layout(title="Range sensitivity under master-grouped target classification",
                      xaxis_title="Measured common interval (cm⁻¹)", yaxis_title="Balanced accuracy")
    save("03_range_sensitivity.html", "Spectral-range sensitivity",
         "Compares only intervals measured by all instruments; no extrapolated 100–399 cm⁻¹ values are introduced.", fig)

    base = candidates["minimal_minmax"]
    for instrument_index, instrument in enumerate(sorted(manifest.instrument.unique())):
        indices = np.flatnonzero(manifest.instrument.eq(instrument).to_numpy())
        center = np.median(base[indices], axis=0)
        chosen = indices[np.argmin(np.linalg.norm(base[indices] - center, axis=1))]
        fig = go.Figure()
        for idx, (name, matrix) in enumerate(candidates.items()):
            fig.add_trace(go.Scattergl(x=axis, y=row_minmax(matrix[chosen:chosen+1])[0], mode="lines",
                                       name=name, line={"width": 1.5, "color": COLORS[idx % len(COLORS)]}))
        record = manifest.iloc[chosen]
        fig.update_layout(title=f"{instrument}: preprocessing on a median-representative spectrum",
                          xaxis_title="Raman shift (cm⁻¹)", yaxis_title="Display-only min–max intensity")
        save(f"{instrument_index + 4:02d}_{instrument}_candidate_overlay.html",
             f"{instrument} preprocessing overlay",
             f"Representative {record.target_analyte} spectrum; every candidate is display-scaled identically.", fig)
    return artifacts


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=repository / "configs" / "nato_sers_preprocessing_exploration_v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    input_dir = repository / config["input_workspace"]
    output = repository / config["output_workspace"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)
    bundle = np.load(input_dir / "arrays" / "unique_attributable_common_support_raw.npz")
    axis, raw = bundle["axis_cm1"].astype(float), bundle["intensity"].astype(float)
    manifest = pd.read_csv(input_dir / "tables" / "analysis_manifest.csv")
    candidates = preprocess_all(axis, raw, config)
    np.savez_compressed(output / "preprocessing_candidates.npz", axis_cm1=axis, **{
        key: value.astype(np.float32) for key, value in candidates.items()
    })
    rows = []
    for name, matrix in candidates.items():
        row = {"candidate": name, **spectral_diagnostics(raw, matrix)}
        row["within_master_cosine"], row["cross_instrument_master_cosine"] = replicate_cosine(matrix, manifest)
        row["target_balanced_accuracy"], row["target_macro_f1"] = grouped_diagnostic(
            matrix, manifest.target_analyte, manifest.master_sample_id,
            int(config["diagnostic_folds"]), int(config["random_seed"]), True)
        row["instrument_balanced_accuracy"], row["instrument_macro_f1"] = grouped_diagnostic(
            matrix, manifest.instrument, manifest.master_sample_id,
            int(config["diagnostic_folds"]), int(config["random_seed"]), False)
        row.update(cluster_diagnostics(matrix, manifest, int(config["random_seed"])))
        rows.append(row)
        print(json.dumps(row), flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "tables" / "preprocessing_metrics.csv", index=False)

    range_rows = []
    for name in ["minimal_minmax", "despike_savgol_minmax", "arpls_minmax",
                 "arpls_savgol_minmax", "savgol_derivative1_snv"]:
        for low, high in config["ranges_cm1"]:
            mask = (axis >= low) & (axis <= high)
            bal, macro = grouped_diagnostic(candidates[name][:, mask], manifest.target_analyte,
                                            manifest.master_sample_id, int(config["diagnostic_folds"]),
                                            int(config["random_seed"]), True)
            range_rows.append({"candidate": name, "range": f"{low}–{high}",
                               "low_cm1": low, "high_cm1": high,
                               "target_balanced_accuracy": bal, "target_macro_f1": macro})
    range_metrics = pd.DataFrame(range_rows)
    range_metrics.to_csv(output / "tables" / "range_sensitivity.csv", index=False)
    artifacts = make_plots(output, axis, raw, manifest, candidates, metrics, range_metrics)
    (output / "html_artifact_registry.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    cards = "\n".join(f'<li><a href="html/{a["file"]}">{a["title"]}</a> — {a["description"]}</li>' for a in artifacts)
    (output / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>NATO SERS preprocessing</title>"
        "<style>body{font:16px Arial;max-width:1100px;margin:40px auto;line-height:1.5}li{margin:12px 0}</style>"
        f"</head><body><h1>NATO SERS preprocessing exploration</h1><p>No candidate is selected by visual appearance alone.</p><ol>{cards}</ol></body></html>"
    )
    provenance = {"protocol_version": config["protocol_version"],
                  "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  "input_raw_bundle_sha256": hashlib.sha256((input_dir / "arrays" / "unique_attributable_common_support_raw.npz").read_bytes()).hexdigest(),
                  "candidates": list(candidates), "html_plots": len(artifacts)}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
