#!/usr/bin/env python3
"""Fresh, provenance-first exploration of the raw NATO SERS field-trial archive.

This stage intentionally does not consume a prior frozen manifest or preprocessing
bundle. It re-parses the workbooks and readable vendor exports, inventories every
source file, derives native/effective support before interpolation, and produces
interactive HTML exploration for a transparent, uniquely attributable SERS tier.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import hdbscan
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
import sklearn
import umap
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d, median_filter, percentile_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

import build_nato_sers_field_trial as source_parser


PLOTLY_CONFIG = {
    "responsive": True,
    "scrollZoom": True,
    "displaylogo": False,
    "toImageButtonOptions": {"format": "svg", "scale": 1},
}
PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#F0E442",
    "#000000",
    "#6A3D9A",
    "#A6CEE3",
    "#B2DF8A",
    "#FB9A99",
]


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repository / "configs" / "nato_sers_raw_reexploration_v1.json",
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_ready) + "\n"
    )


def file_role(relative: Path) -> str:
    extension = relative.suffix.lower()
    if relative.name in {
        "01_Raman Team recordings_06Mar2024.xlsx",
        "MASTER SAMPLE LIST FOR NATO SERS CDT_MARCH 2024.xlsx",
    }:
        return "authoritative_workbook"
    if relative.name == "Notes.txt":
        return "source_note"
    if relative.parts and relative.parts[0] == "Z Converted":
        return "converted_duplicate_or_derivative"
    if extension == ".csv":
        return "readable_numeric_export"
    if extension == ".txt":
        return "readable_numeric_or_metadata_export"
    if extension == ".spc":
        return "native_binary_spectrum"
    if extension == ".pdf":
        return "instrument_report"
    if extension == ".prb":
        return "instrument_probe_bundle"
    if extension == ".cez":
        return "agilent_session_archive"
    return "other"


def instrument_from_relative(relative: Path) -> str | None:
    for part in relative.parts:
        cleaned = part.replace(" ", "-")
        for prefix in ("Agilent-", "Mira-", "Pendar-", "RMX-"):
            if cleaned.startswith(prefix) and cleaned[len(prefix) :].isdigit():
                return cleaned
    return None


def inventory_files(source_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in source_root.rglob("*") if item.is_file()):
        relative = path.relative_to(source_root)
        stat = path.stat()
        rows.append(
            {
                "relative_path": str(relative),
                "top_level": relative.parts[0],
                "instrument": instrument_from_relative(relative),
                "extension": path.suffix.lower() or "[none]",
                "role": file_role(relative),
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    frame = pd.DataFrame(rows)
    frame["hash_reference_count"] = frame.groupby("sha256")["sha256"].transform(
        "size"
    )
    frame["exact_content_duplicate"] = frame["hash_reference_count"] > 1
    return frame


def robust_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def constant_edge_count(values: np.ndarray, from_right: bool = False) -> int:
    values = np.asarray(values, dtype=float)
    if from_right:
        values = values[::-1]
    if len(values) < 2:
        return len(values)
    scale = max(float(np.nanmax(np.abs(values))), 1.0)
    tolerance = scale * 1e-12
    changes = np.flatnonzero(np.abs(np.diff(values)) > tolerance)
    if not len(changes):
        return len(values)
    return int(changes[0] + 1)


def spectrum_diagnostics(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    step = np.diff(x)
    leading = constant_edge_count(y)
    trailing = constant_edge_count(y, from_right=True)
    lead_index = min(leading, max(len(x) - 1, 0))
    trail_index = max(0, len(x) - 1 - trailing)
    effective_min = float(x[lead_index]) if leading >= 5 else float(x[0])
    effective_max = float(x[trail_index]) if trailing >= 5 else float(x[-1])
    first_difference = np.diff(y)
    second_difference = np.diff(y, n=2)
    noise_mad = robust_mad(first_difference) / 0.67448975 if len(y) > 2 else 0.0
    curvature_mad = robust_mad(second_difference) if len(y) > 3 else 0.0
    curvature_center = float(np.median(second_difference)) if len(y) > 3 else 0.0
    spike_threshold = max(12.0 * curvature_mad, np.finfo(float).eps)
    spike_count = int(
        np.sum(np.abs(second_difference - curvature_center) > spike_threshold)
    )
    baseline_window = min(101, len(y) if len(y) % 2 else len(y) - 1)
    baseline_window = max(3, baseline_window)
    baseline = gaussian_filter1d(
        percentile_filter(y, percentile=10, size=baseline_window, mode="nearest"),
        sigma=max(1.0, baseline_window / 5.0),
    )
    dynamic = max(float(np.ptp(y)), np.finfo(float).eps)
    maximum = float(np.max(y))
    repeated_max_fraction = float(np.mean(np.isclose(y, maximum, rtol=0, atol=0)))
    if not finite.all():
        numeric_status = "non_finite"
    elif len(x) < 2 or not np.all(step > 0):
        numeric_status = "invalid_axis"
    elif float(np.ptp(y)) == 0.0:
        numeric_status = "constant"
    else:
        numeric_status = "pass"
    return {
        "numeric_qc_status": numeric_status,
        "n_points": len(x),
        "finite_fraction": float(np.mean(finite)),
        "axis_min_cm1": float(np.min(x)),
        "axis_max_cm1": float(np.max(x)),
        "axis_step_median_cm1": float(np.median(step)) if len(step) else np.nan,
        "axis_step_min_cm1": float(np.min(step)) if len(step) else np.nan,
        "axis_step_max_cm1": float(np.max(step)) if len(step) else np.nan,
        "axis_strictly_increasing": bool(np.all(step > 0)),
        "axis_sha256": hashlib.sha256(np.asarray(x, dtype=np.float64).tobytes()).hexdigest(),
        "leading_constant_points": leading,
        "trailing_constant_points": trailing,
        "effective_axis_min_cm1": effective_min,
        "effective_axis_max_cm1": effective_max,
        "intensity_min": float(np.min(y)),
        "intensity_q01": float(np.quantile(y, 0.01)),
        "intensity_median": float(np.median(y)),
        "intensity_q99": float(np.quantile(y, 0.99)),
        "intensity_max": maximum,
        "intensity_range": dynamic,
        "intensity_std": float(np.std(y)),
        "negative_fraction": float(np.mean(y < 0)),
        "zero_fraction": float(np.mean(y == 0)),
        "repeated_max_fraction": repeated_max_fraction,
        "first_difference_noise_mad": noise_mad,
        "spike_count_proxy": spike_count,
        "spike_fraction_proxy": float(spike_count / max(len(y) - 2, 1)),
        "baseline_energy_fraction_proxy": float(
            np.linalg.norm(baseline) / max(np.linalg.norm(y), np.finfo(float).eps)
        ),
        "baseline_span_fraction_proxy": float(np.ptp(baseline) / dynamic),
        "intensity_sha256": hashlib.sha256(
            np.asarray(y, dtype=np.float64).tobytes()
        ).hexdigest(),
    }


def source_spectrum_inventory(
    sources: dict[str, dict[int, dict[str, Any]]],
) -> tuple[pd.DataFrame, dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    spectra: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    agilent_cache: dict[str, Any] = {}
    for instrument, scans in sorted(sources.items()):
        for scan_id, source in sorted(scans.items()):
            record = {"instrument": instrument, "source_scan_id": scan_id, **source}
            row: dict[str, Any] = {
                "instrument": instrument,
                "source_scan_id": scan_id,
                **source,
                "load_status": "pass",
            }
            try:
                x, y = source_parser.load_spectrum(record, agilent_cache)
                x = np.asarray(x, dtype=float)
                y = np.asarray(y, dtype=float)
                row.update(spectrum_diagnostics(x, y))
                spectra[(instrument, scan_id)] = (x, y)
            except Exception as error:
                row["load_status"] = f"error:{type(error).__name__}:{error}"
            rows.append(row)
    return pd.DataFrame(rows), spectra


def fresh_tiers(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["tier_all_parseable_sers"] = (
        result["is_named_sers"].fillna(False)
        & result["source_match_status"].eq("matched")
        & result["numeric_qc_status"].eq("pass")
    )
    result["tier_attributable_sers"] = (
        result["tier_all_parseable_sers"]
        & result["master_sample_id"].notna()
        & result["target_analyte"].notna()
        & ~result["target_analyte"].eq("unmapped")
    )
    result["tier_unique_attributable_sers"] = (
        result["tier_attributable_sers"]
        & ~result["source_reference_conflict"].fillna(False)
        & result["source_primary_reference"].fillna(False)
    )
    result["tier_notes_clear_sers"] = (
        result["tier_unique_attributable_sers"]
        & ~result["manual_severe_qc_flag"].fillna(False)
        & ~result["manual_low_signal_or_noise_flag"].fillna(False)
    )

    def reasons(row: pd.Series) -> str:
        values: list[str] = []
        if not bool(row.get("is_named_sers")):
            values.append("not_named_sers")
        if row.get("source_match_status") != "matched":
            values.append(str(row.get("source_match_status") or "source_unresolved"))
        numeric_status = row.get("numeric_qc_status")
        if numeric_status != "pass":
            values.append(
                "numeric_unavailable"
                if pd.isna(numeric_status)
                else str(numeric_status)
            )
        if pd.isna(row.get("master_sample_id")):
            values.append("master_sample_unresolved")
        if pd.isna(row.get("target_analyte")) or row.get("target_analyte") == "unmapped":
            values.append("target_unresolved")
        if bool(row.get("source_reference_conflict")):
            values.append("source_reused_with_conflicting_metadata")
        if not bool(row.get("source_primary_reference")) and row.get("source_match_status") == "matched":
            values.append("duplicate_source_reference")
        if bool(row.get("manual_severe_qc_flag")):
            values.append("severe_quality_note")
        if bool(row.get("manual_low_signal_or_noise_flag")):
            values.append("low_signal_or_noise_note")
        return ";".join(dict.fromkeys(values)) or "included"

    result["tier_reason_codes"] = result.apply(reasons, axis=1)
    return result


def attach_source_diagnostics(
    records: pd.DataFrame, source_inventory: pd.DataFrame
) -> pd.DataFrame:
    diagnostic_columns = [
        column
        for column in source_inventory.columns
        if column
        not in {
            "source_text_path",
            "source_spc_path",
            "source_dark_spc_path",
            "source_pdf_path",
            "source_prb_path",
            "source_format",
        }
    ]
    diagnostics = source_inventory[diagnostic_columns].copy()
    diagnostics["source_scan_id"] = pd.to_numeric(
        diagnostics["source_scan_id"], errors="coerce"
    ).astype("Int64")
    result = records.copy()
    result["source_scan_id"] = pd.to_numeric(
        result["source_scan_id"], errors="coerce"
    ).astype("Int64")
    overlap = [
        column
        for column in diagnostics.columns
        if column in result.columns and column not in {"instrument", "source_scan_id"}
    ]
    result = result.drop(columns=overlap)
    return result.merge(
        diagnostics,
        how="left",
        on=["instrument", "source_scan_id"],
        validate="many_to_one",
    )


def support_summary(source_inventory: pd.DataFrame) -> pd.DataFrame:
    valid = source_inventory[
        source_inventory["load_status"].eq("pass")
        & source_inventory["numeric_qc_status"].eq("pass")
    ]
    rows: list[dict[str, Any]] = []
    for instrument, group in valid.groupby("instrument", sort=True):
        rows.append(
            {
                "instrument": instrument,
                "n_source_spectra": len(group),
                "n_native_axis_profiles": group["axis_sha256"].nunique(),
                "nominal_axis_min_min": group["axis_min_cm1"].min(),
                "nominal_axis_min_max": group["axis_min_cm1"].max(),
                "nominal_axis_max_min": group["axis_max_cm1"].min(),
                "nominal_axis_max_max": group["axis_max_cm1"].max(),
                "effective_axis_min_min": group["effective_axis_min_cm1"].min(),
                "effective_axis_min_max": group["effective_axis_min_cm1"].max(),
                "effective_axis_max_min": group["effective_axis_max_cm1"].min(),
                "effective_axis_max_max": group["effective_axis_max_cm1"].max(),
                "median_step_cm1": group["axis_step_median_cm1"].median(),
                "median_points": group["n_points"].median(),
                "median_leading_constant_points": group[
                    "leading_constant_points"
                ].median(),
                "median_baseline_energy_fraction": group[
                    "baseline_energy_fraction_proxy"
                ].median(),
                "median_spike_fraction": group["spike_fraction_proxy"].median(),
                "median_negative_fraction": group["negative_fraction"].median(),
            }
        )
    return pd.DataFrame(rows)


def analysis_matrix(
    manifest: pd.DataFrame,
    spectra: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    step_cm1: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    selected = manifest[manifest["tier_unique_attributable_sers"]].copy()
    selected = selected.sort_values(
        ["session", "excel_row", "recording_subindex", "observation_uid"]
    ).reset_index(drop=True)
    common_start = math.ceil(float(selected["effective_axis_min_cm1"].max()) / step_cm1) * step_cm1
    common_end = math.floor(float(selected["effective_axis_max_cm1"].min()) / step_cm1) * step_cm1
    if common_end <= common_start:
        raise ValueError(f"No common effective support: {common_start}..{common_end}")
    axis = np.arange(common_start, common_end + step_cm1 / 2, step_cm1)
    matrix = np.empty((len(selected), len(axis)), dtype=np.float32)
    for row_index, row in selected.iterrows():
        key = (str(row["instrument"]), int(row["source_scan_id"]))
        x, y = spectra[key]
        if axis[0] < x.min() or axis[-1] > x.max():
            raise ValueError(f"Interpolation would extrapolate for {key}")
        matrix[row_index] = np.interp(axis, x, y).astype(np.float32)
    metadata = {
        "n_spectra": len(selected),
        "n_masters": int(selected["master_sample_id"].nunique()),
        "common_effective_start_cm1": float(common_start),
        "common_effective_end_cm1": float(common_end),
        "step_cm1": float(step_cm1),
        "n_features": len(axis),
        "selection_tier": "tier_unique_attributable_sers",
    }
    return selected, axis.astype(np.float32), matrix, metadata


def minmax_rows(matrix: np.ndarray) -> np.ndarray:
    low = np.min(matrix, axis=1, keepdims=True)
    span = np.max(matrix, axis=1, keepdims=True) - low
    return (matrix - low) / np.maximum(span, np.finfo(np.float32).eps)


def snv_rows(matrix: np.ndarray) -> np.ndarray:
    mean = np.mean(matrix, axis=1, keepdims=True)
    std = np.std(matrix, axis=1, keepdims=True)
    return (matrix - mean) / np.maximum(std, np.finfo(np.float32).eps)


def pca_for_clustering(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, PCA]:
    maximum = min(50, matrix.shape[0] - 1, matrix.shape[1])
    model = PCA(n_components=maximum, svd_solver="full", random_state=seed)
    embedded = model.fit_transform(matrix)
    cumulative = np.cumsum(model.explained_variance_ratio_)
    keep = min(max(int(np.searchsorted(cumulative, 0.95) + 1), 2), maximum)
    return embedded[:, :keep], model


def kmeans_search(
    features: np.ndarray, k_values: Iterable[int], seed: int
) -> tuple[pd.DataFrame, np.ndarray, int]:
    rows: list[dict[str, Any]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in k_values:
        if k >= len(features):
            continue
        model = KMeans(n_clusters=k, n_init=50, random_state=seed)
        labels = model.fit_predict(features)
        labels_by_k[k] = labels
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(features, labels)),
                "inertia": float(model.inertia_),
            }
        )
    frame = pd.DataFrame(rows)
    best_k = int(frame.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    return frame, labels_by_k[best_k], best_k


def cluster_association(
    manifest: pd.DataFrame,
    labels: np.ndarray,
    method: str,
    silhouette: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in (
        "target_analyte",
        "instrument",
        "sensor_family",
        "sensor_variant",
        "station",
        "master_sample_id",
        "manual_severe_qc_flag",
    ):
        values = manifest[field].fillna("<missing>").astype(str).to_numpy()
        rows.append(
            {
                "method": method,
                "metadata_field": field,
                "n_observations": len(labels),
                "n_clusters_including_noise": int(len(np.unique(labels))),
                "noise_fraction": float(np.mean(labels == -1)),
                "silhouette_nonnoise": silhouette,
                "adjusted_rand_index": float(adjusted_rand_score(values, labels)),
                "normalized_mutual_information": float(
                    normalized_mutual_info_score(values, labels)
                ),
            }
        )
    return rows


def run_embeddings(
    manifest: pd.DataFrame,
    raw_matrix: np.ndarray,
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
]:
    seed = int(config["random_seed"])
    raw_feature_scaled = StandardScaler().fit_transform(raw_matrix)
    minmax = minmax_rows(raw_matrix)
    snv = snv_rows(raw_matrix)
    raw_cluster_features, raw_pca_model = pca_for_clustering(raw_feature_scaled, seed)
    minmax_cluster_features, minmax_pca_model = pca_for_clustering(minmax, seed)
    minmax_pca2 = minmax_pca_model.transform(minmax)[:, :2]
    raw_pca2 = raw_pca_model.transform(raw_feature_scaled)[:, :2]

    umap_config = config["umap"]
    umap_embedding = umap.UMAP(
        n_components=2,
        n_neighbors=int(umap_config["n_neighbors"]),
        min_dist=float(umap_config["min_dist"]),
        metric=str(umap_config["metric"]),
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    ).fit_transform(minmax_cluster_features)
    perplexity = min(float(config["tsne"]["perplexity"]), (len(manifest) - 1) / 3)
    tsne_embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init=str(config["tsne"]["init"]),
        learning_rate=config["tsne"]["learning_rate"],
        random_state=seed,
        method="barnes_hut",
    ).fit_transform(minmax_cluster_features)

    kmeans_frame, kmeans_labels, best_k = kmeans_search(
        minmax_cluster_features, config["kmeans_k_values"], seed
    )
    hdb_config = config["hdbscan"]
    hdb_model = hdbscan.HDBSCAN(
        min_cluster_size=int(hdb_config["min_cluster_size"]),
        min_samples=int(hdb_config["min_samples"]),
        metric=str(hdb_config["metric"]),
        cluster_selection_method="eom",
        prediction_data=True,
    )
    hdb_labels = hdb_model.fit_predict(minmax_cluster_features)
    nonnoise = hdb_labels != -1
    hdb_silhouette = None
    if np.sum(nonnoise) >= 3 and len(np.unique(hdb_labels[nonnoise])) >= 2:
        hdb_silhouette = float(
            silhouette_score(minmax_cluster_features[nonnoise], hdb_labels[nonnoise])
        )
    kmeans_silhouette = float(
        silhouette_score(minmax_cluster_features, kmeans_labels)
    )

    embedding_frame = manifest[
        [
            "observation_uid",
            "master_sample_id",
            "target_analyte",
            "instrument",
            "sensor_family",
            "sensor_variant",
            "station",
            "manual_severe_qc_flag",
            "manual_low_signal_or_noise_flag",
        ]
    ].copy()
    for prefix, values in (
        ("raw_pca", raw_pca2),
        ("minmax_pca", minmax_pca2),
        ("minmax_umap", umap_embedding),
        ("minmax_tsne", tsne_embedding),
    ):
        embedding_frame[f"{prefix}_1"] = values[:, 0]
        embedding_frame[f"{prefix}_2"] = values[:, 1]
    embedding_frame["kmeans_cluster"] = kmeans_labels
    embedding_frame["hdbscan_cluster"] = hdb_labels
    embedding_frame["hdbscan_probability"] = hdb_model.probabilities_
    embedding_frame["hdbscan_outlier_score"] = hdb_model.outlier_scores_

    association = []
    association.extend(
        cluster_association(
            manifest, kmeans_labels, f"kmeans_k{best_k}_minmax_pca95", kmeans_silhouette
        )
    )
    association.extend(
        cluster_association(
            manifest, hdb_labels, "hdbscan_minmax_pca95", hdb_silhouette
        )
    )
    diagnostics = pd.DataFrame(
        [
            {
                "representation": "raw_channel_standardized",
                "pca_components_to_95pct": raw_cluster_features.shape[1],
                "pc1_explained_variance": raw_pca_model.explained_variance_ratio_[0],
                "pc2_explained_variance": raw_pca_model.explained_variance_ratio_[1],
            },
            {
                "representation": "per_spectrum_minmax",
                "pca_components_to_95pct": minmax_cluster_features.shape[1],
                "pc1_explained_variance": minmax_pca_model.explained_variance_ratio_[0],
                "pc2_explained_variance": minmax_pca_model.explained_variance_ratio_[1],
            },
        ]
    )
    arrays = {
        "minmax": minmax.astype(np.float32),
        "snv": snv.astype(np.float32),
        "raw_pca2": raw_pca2.astype(np.float32),
        "minmax_pca2": minmax_pca2.astype(np.float32),
        "minmax_umap": umap_embedding.astype(np.float32),
        "minmax_tsne": tsne_embedding.astype(np.float32),
    }
    return embedding_frame, kmeans_frame, pd.DataFrame(association), diagnostics, arrays


def run_master_embeddings(
    manifest: pd.DataFrame,
    minmax_matrix: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = int(config["random_seed"])
    work = manifest.reset_index(drop=True).copy()
    work["matrix_index"] = np.arange(len(work))
    master_rows: list[dict[str, Any]] = []
    master_spectra: list[np.ndarray] = []
    for master_id, group in work.groupby("master_sample_id", sort=True):
        indices = group["matrix_index"].to_numpy(int)
        targets = sorted(group["target_analyte"].dropna().astype(str).unique())
        stations = sorted(group["station"].dropna().astype(str).unique())
        master_rows.append(
            {
                "master_sample_id": master_id,
                "target_analyte": targets[0] if len(targets) == 1 else "|".join(targets),
                "station": stations[0] if len(stations) == 1 else "|".join(stations),
                "n_spectra": len(group),
                "n_instruments": group["instrument"].nunique(),
                "n_sensor_families": group["sensor_family"].nunique(),
                "instruments": "|".join(sorted(group["instrument"].dropna().astype(str).unique())),
                "sensor_families": "|".join(sorted(group["sensor_family"].dropna().astype(str).unique())),
            }
        )
        master_spectra.append(np.mean(minmax_matrix[indices], axis=0))
    frame = pd.DataFrame(master_rows)
    matrix = np.asarray(master_spectra, dtype=np.float32)
    features, pca_model = pca_for_clustering(matrix, seed)
    pca2 = pca_model.transform(matrix)[:, :2]
    neighbors = min(10, len(frame) - 1)
    umap2 = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    ).fit_transform(features)
    perplexity = min(15.0, (len(frame) - 1) / 3)
    tsne2 = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)
    k_frame, k_labels, best_k = kmeans_search(
        features,
        [value for value in config["kmeans_k_values"] if value < len(frame)],
        seed,
    )
    hdb_model = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    hdb_labels = hdb_model.fit_predict(features)
    for prefix, values in (("pca", pca2), ("umap", umap2), ("tsne", tsne2)):
        frame[f"master_{prefix}_1"] = values[:, 0]
        frame[f"master_{prefix}_2"] = values[:, 1]
    frame["kmeans_cluster"] = k_labels
    frame["hdbscan_cluster"] = hdb_labels
    association_rows: list[dict[str, Any]] = []
    for method, labels in ((f"kmeans_k{best_k}", k_labels), ("hdbscan", hdb_labels)):
        nonnoise = labels != -1
        silhouette = None
        if np.sum(nonnoise) >= 3 and len(np.unique(labels[nonnoise])) >= 2:
            silhouette = float(silhouette_score(features[nonnoise], labels[nonnoise]))
        for field in ("target_analyte", "station"):
            values = frame[field].fillna("<missing>").astype(str).to_numpy()
            association_rows.append(
                {
                    "method": method,
                    "metadata_field": field,
                    "n_masters": len(frame),
                    "n_clusters_including_noise": len(np.unique(labels)),
                    "noise_fraction": float(np.mean(labels == -1)),
                    "silhouette_nonnoise": silhouette,
                    "adjusted_rand_index": float(adjusted_rand_score(values, labels)),
                    "normalized_mutual_information": float(
                        normalized_mutual_info_score(values, labels)
                    ),
                }
            )
    return frame, k_frame, pd.DataFrame(association_rows)


def write_plot(path: Path, figure: go.Figure) -> None:
    figure.update_layout(
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 14},
        hoverlabel={"font_size": 13},
        margin={"l": 80, "r": 30, "t": 80, "b": 70},
    )
    figure.write_html(
        path,
        include_plotlyjs=True,
        full_html=True,
        config=PLOTLY_CONFIG,
    )


def categorical_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
    x_title: str,
    y_title: str,
) -> go.Figure:
    figure = go.Figure()
    values = frame[color].fillna("<missing>").astype(str)
    hover_columns = [
        "observation_uid",
        "master_sample_id",
        "target_analyte",
        "instrument",
        "sensor_family",
        "sensor_variant",
        "station",
    ]
    for index, category in enumerate(sorted(values.unique())):
        mask = values.eq(category)
        custom = frame.loc[mask, hover_columns].fillna("<missing>").astype(str).to_numpy()
        figure.add_trace(
            go.Scattergl(
                x=frame.loc[mask, x],
                y=frame.loc[mask, y],
                mode="markers",
                name=category,
                marker={
                    "size": 8,
                    "opacity": 0.78,
                    "color": PALETTE[index % len(PALETTE)],
                    "line": {"width": 0.3, "color": "white"},
                },
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>master=%{customdata[1]}"
                    "<br>target=%{customdata[2]}<br>instrument=%{customdata[3]}"
                    "<br>sensor=%{customdata[4]} / %{customdata[5]}"
                    "<br>station=%{customdata[6]}<br>x=%{x:.4g}, y=%{y:.4g}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title=color,
    )
    return figure


def master_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
) -> go.Figure:
    figure = go.Figure()
    values = frame[color].fillna("<missing>").astype(str)
    hover_columns = [
        "master_sample_id",
        "target_analyte",
        "station",
        "n_spectra",
        "n_instruments",
        "n_sensor_families",
        "instruments",
        "sensor_families",
    ]
    for index, category in enumerate(sorted(values.unique())):
        mask = values.eq(category)
        custom = frame.loc[mask, hover_columns].fillna("<missing>").astype(str).to_numpy()
        figure.add_trace(
            go.Scatter(
                x=frame.loc[mask, x],
                y=frame.loc[mask, y],
                mode="markers",
                name=category,
                marker={
                    "size": 10,
                    "opacity": 0.82,
                    "color": PALETTE[index % len(PALETTE)],
                    "line": {"width": 0.5, "color": "white"},
                },
                customdata=custom,
                hovertemplate=(
                    "<b>master %{customdata[0]}</b><br>target=%{customdata[1]}"
                    "<br>station=%{customdata[2]}<br>spectra=%{customdata[3]}"
                    "<br>instruments=%{customdata[4]} (%{customdata[6]})"
                    "<br>sensors=%{customdata[5]} (%{customdata[7]})"
                    "<br>x=%{x:.4g}, y=%{y:.4g}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        legend_title=color,
    )
    return figure


def make_html_plots(
    html_dir: Path,
    file_inventory: pd.DataFrame,
    manifest: pd.DataFrame,
    analysis_manifest: pd.DataFrame,
    source_inventory: pd.DataFrame,
    support: pd.DataFrame,
    axis: np.ndarray,
    raw_matrix: np.ndarray,
    embeddings: pd.DataFrame,
    kmeans_search_frame: pd.DataFrame,
    master_embeddings: pd.DataFrame,
    master_kmeans_search: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []

    def save(name: str, title: str, description: str, figure: go.Figure) -> None:
        write_plot(html_dir / name, figure)
        artifacts.append(
            {"file": name, "title": title, "description": description}
        )

    file_counts = (
        file_inventory.groupby(["top_level", "extension"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    figure = go.Figure()
    for index, extension in enumerate(sorted(file_counts["extension"].unique())):
        group = file_counts[file_counts["extension"].eq(extension)]
        figure.add_bar(
            x=group["top_level"],
            y=group["count"],
            name=extension,
            marker_color=PALETTE[index % len(PALETTE)],
            customdata=group[["extension"]],
            hovertemplate="%{x}<br>%{customdata[0]}: %{y}<extra></extra>",
        )
    figure.update_layout(
        barmode="stack",
        title="Raw archive file inventory",
        xaxis_title="Top-level source group",
        yaxis_title="Files",
    )
    save(
        "01_file_inventory.html",
        "Raw file inventory",
        "Counts native files, readable exports, reports, workbooks, and converted duplicates.",
        figure,
    )

    flow = manifest.copy()
    flow["sers_semantics"] = np.select(
        [
            flow["is_named_sers"].fillna(False),
            flow["is_ambiguous_sers"].fillna(False),
        ],
        ["named SERS sensor", "normal=N but sensor unnamed"],
        default="normal Raman / no named SERS",
    )
    flow_counts = (
        flow.groupby(["sers_semantics", "source_match_status"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    figure = go.Figure()
    for index, status in enumerate(sorted(flow_counts["source_match_status"].astype(str).unique())):
        group = flow_counts[flow_counts["source_match_status"].astype(str).eq(status)]
        figure.add_bar(
            x=group["sers_semantics"],
            y=group["count"],
            name=status,
            marker_color=PALETTE[index % len(PALETTE)],
        )
    figure.update_layout(
        barmode="stack",
        title="Recording-log semantics and source resolution",
        xaxis_title="Log interpretation",
        yaxis_title="Expanded recording observations",
    )
    save(
        "02_recording_flow.html",
        "Recording flow",
        "Separates named SERS, ambiguous rows, ordinary Raman, and source-match outcomes.",
        figure,
    )

    heat = pd.crosstab(
        analysis_manifest["target_analyte"], analysis_manifest["instrument"]
    )
    figure = go.Figure(
        go.Heatmap(
            z=heat.to_numpy(),
            x=heat.columns,
            y=heat.index,
            colorscale="Viridis",
            text=heat.to_numpy(),
            texttemplate="%{text}",
            hovertemplate="target=%{y}<br>instrument=%{x}<br>spectra=%{z}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Chemical × instrument support in the unique attributable SERS tier",
        xaxis_title="Instrument",
        yaxis_title="Master-list target analyte",
    )
    save(
        "03_chemical_instrument_support.html",
        "Chemical × instrument support",
        "Exposes missing cells and acquisition/chemistry confounding before modeling.",
        figure,
    )

    heat = pd.crosstab(
        analysis_manifest["target_analyte"], analysis_manifest["sensor_family"]
    )
    figure = go.Figure(
        go.Heatmap(
            z=heat.to_numpy(),
            x=heat.columns,
            y=heat.index,
            colorscale="Cividis",
            text=heat.to_numpy(),
            texttemplate="%{text}",
        )
    )
    figure.update_layout(
        title="Chemical × SERS sensor-family support",
        xaxis_title="Sensor family",
        yaxis_title="Master-list target analyte",
    )
    save(
        "04_chemical_sensor_support.html",
        "Chemical × sensor support",
        "Shows whether sensor and chemistry factors are sufficiently crossed.",
        figure,
    )

    coverage = source_inventory[
        source_inventory["load_status"].eq("pass")
        & source_inventory["numeric_qc_status"].eq("pass")
    ].copy()
    coverage = coverage.sort_values(["instrument", "source_scan_id"]).reset_index(drop=True)
    coverage["row"] = np.arange(len(coverage))
    figure = go.Figure()
    for index, (instrument, group) in enumerate(coverage.groupby("instrument", sort=True)):
        color = PALETTE[index % len(PALETTE)]
        for row in group.itertuples():
            figure.add_trace(
                go.Scattergl(
                    x=[row.axis_min_cm1, row.axis_max_cm1, None],
                    y=[row.row, row.row, None],
                    mode="lines",
                    line={"color": color, "width": 2},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        figure.add_trace(
            go.Scattergl(
                x=group["effective_axis_min_cm1"],
                y=group["row"],
                mode="markers",
                name=instrument,
                marker={"size": 5, "color": color},
                customdata=group[["source_scan_id", "effective_axis_max_cm1"]],
                hovertemplate=(
                    f"{instrument}<br>scan=%{{customdata[0]}}"
                    "<br>effective=%{x:.1f}–%{customdata[1]:.1f} cm⁻¹<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Native nominal coverage and effective low-support boundary",
        xaxis_title="Raman shift (cm⁻¹)",
        yaxis_title="Source spectrum index",
    )
    save(
        "05_native_axis_coverage.html",
        "Native axis coverage",
        "Lines show nominal file axes; markers show the effective lower boundary after leading-flat detection.",
        figure,
    )

    max_examples = int(config["html"]["max_raw_example_spectra_per_instrument"])
    instruments = sorted(analysis_manifest["instrument"].unique())
    rows = math.ceil(len(instruments) / 2)
    figure = make_subplots(rows=rows, cols=2, subplot_titles=instruments)
    for panel, instrument in enumerate(instruments):
        row = panel // 2 + 1
        column = panel % 2 + 1
        group = analysis_manifest[analysis_manifest["instrument"].eq(instrument)]
        indices = np.linspace(0, len(group) - 1, min(max_examples, len(group)), dtype=int)
        for offset, selected_index in enumerate(indices):
            record = group.iloc[selected_index]
            matrix_index = int(record.name)
            figure.add_trace(
                go.Scattergl(
                    x=axis,
                    y=raw_matrix[matrix_index],
                    mode="lines",
                    line={"width": 1.2, "color": PALETTE[offset % len(PALETTE)]},
                    name=str(record["target_analyte"]),
                    legendgroup=f"{instrument}-{offset}",
                    showlegend=False,
                    customdata=np.full((len(axis), 1), record["observation_uid"]),
                    hovertemplate="%{customdata[0]}<br>%{x:.0f} cm⁻¹<br>%{y:.5g}<extra></extra>",
                ),
                row=row,
                col=column,
            )
        figure.update_xaxes(title_text="cm⁻¹", row=row, col=column)
        figure.update_yaxes(title_text="Raw intensity", row=row, col=column)
    figure.update_layout(height=max(480, rows * 330), title="Raw common-support examples by instrument")
    save(
        "06_raw_spectra_examples.html",
        "Raw spectral examples",
        "Unnormalized intensities on the empirically shared measured interval; zoom and hover are enabled.",
        figure,
    )

    minmax = minmax_rows(raw_matrix)
    figure = go.Figure()
    for index, instrument in enumerate(instruments):
        mask = analysis_manifest["instrument"].eq(instrument).to_numpy()
        median = np.median(minmax[mask], axis=0)
        q10 = np.quantile(minmax[mask], 0.10, axis=0)
        q90 = np.quantile(minmax[mask], 0.90, axis=0)
        color = PALETTE[index % len(PALETTE)]
        figure.add_trace(
            go.Scattergl(x=axis, y=median, mode="lines", name=instrument, line={"color": color, "width": 2})
        )
        figure.add_trace(
            go.Scattergl(
                x=np.concatenate([axis, axis[::-1]]),
                y=np.concatenate([q90, q10[::-1]]),
                fill="toself",
                fillcolor=color.replace("#", "#") + "26" if False else "rgba(120,120,120,0.08)",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    figure.update_layout(
        title="Per-spectrum min–max summaries by instrument",
        xaxis_title="Raman shift (cm⁻¹)",
        yaxis_title="Scaled intensity",
    )
    save(
        "07_minmax_instrument_summaries.html",
        "Min–max instrument summaries",
        "Instrument medians with 10th–90th percentile envelopes; diagnostic only, not a selected protocol.",
        figure,
    )

    embedding_specs = [
        ("raw_pca", "Raw channel-standardized PCA"),
        ("minmax_pca", "Per-spectrum min–max PCA"),
        ("minmax_umap", "Per-spectrum min–max UMAP"),
        ("minmax_tsne", "Per-spectrum min–max t-SNE"),
    ]
    color_fields = ["target_analyte", "instrument", "sensor_family", "station"]
    counter = 8
    for prefix, method_title in embedding_specs:
        for color_field in color_fields:
            name = f"{counter:02d}_{prefix}_{color_field}.html"
            figure = categorical_scatter(
                embeddings,
                f"{prefix}_1",
                f"{prefix}_2",
                color_field,
                f"{method_title}, colored by {color_field}",
                f"{prefix} 1",
                f"{prefix} 2",
            )
            save(
                name,
                f"{method_title}: {color_field}",
                "Hover identifies the recording, physical master, chemistry, instrument, sensor, and station.",
                figure,
            )
            counter += 1

    figure = go.Figure(
        go.Scatter(
            x=kmeans_search_frame["k"],
            y=kmeans_search_frame["silhouette"],
            mode="lines+markers",
            marker={"size": 9, "color": "#0072B2"},
        )
    )
    figure.update_layout(
        title="K-means cluster-count search on min–max PCA95 features",
        xaxis_title="k",
        yaxis_title="Observation-level silhouette (descriptive)",
    )
    save(
        f"{counter:02d}_kmeans_k_search.html",
        "K-means k search",
        "Silhouette is exploratory and does not establish chemical validity.",
        figure,
    )
    counter += 1
    for cluster_field, title in (
        ("kmeans_cluster", "K-means assignments on min–max PCA"),
        ("hdbscan_cluster", "HDBSCAN assignments on min–max PCA"),
    ):
        figure = categorical_scatter(
            embeddings,
            "minmax_umap_1",
            "minmax_umap_2",
            cluster_field,
            title,
            "UMAP 1",
            "UMAP 2",
        )
        save(
            f"{counter:02d}_{cluster_field}.html",
            title,
            "Cluster labels are overlaid on UMAP for inspection; clustering itself used PCA95 features.",
            figure,
        )
        counter += 1

    for prefix, method_title in (
        ("master_pca", "Physical-master mean-spectrum PCA"),
        ("master_umap", "Physical-master mean-spectrum UMAP"),
        ("master_tsne", "Physical-master mean-spectrum t-SNE"),
    ):
        for color_field in ("target_analyte", "station"):
            figure = master_scatter(
                master_embeddings,
                f"{prefix}_1",
                f"{prefix}_2",
                color_field,
                f"{method_title}, colored by {color_field}",
            )
            save(
                f"{counter:02d}_{prefix}_{color_field}.html",
                f"{method_title}: {color_field}",
                "One point per physical master; each spectrum is min–max scaled before within-master averaging.",
                figure,
            )
            counter += 1

    figure = go.Figure(
        go.Scatter(
            x=master_kmeans_search["k"],
            y=master_kmeans_search["silhouette"],
            mode="lines+markers",
            marker={"size": 9, "color": "#009E73"},
        )
    )
    figure.update_layout(
        title="Physical-master K-means cluster-count search",
        xaxis_title="k",
        yaxis_title="Master-level silhouette",
    )
    save(
        f"{counter:02d}_master_kmeans_k_search.html",
        "Master-level K-means k search",
        "One independent point per physical master, avoiding repeated-row pseudo-replication.",
        figure,
    )
    counter += 1
    for cluster_field, title in (
        ("kmeans_cluster", "Master-level K-means assignments"),
        ("hdbscan_cluster", "Master-level HDBSCAN assignments"),
    ):
        figure = master_scatter(
            master_embeddings,
            "master_umap_1",
            "master_umap_2",
            cluster_field,
            title,
        )
        save(
            f"{counter:02d}_master_{cluster_field}.html",
            title,
            "Assignments use master-mean PCA95 features and are shown on the master UMAP.",
            figure,
        )
        counter += 1
    return artifacts


def write_index(
    output_dir: Path,
    artifacts: list[dict[str, str]],
    summary: dict[str, Any],
) -> None:
    cards = "\n".join(
        f"""<a class="card" href="html/{html.escape(item['file'])}">
<h2>{html.escape(item['title'])}</h2>
<p>{html.escape(item['description'])}</p>
<code>{html.escape(item['file'])}</code>
</a>"""
        for item in artifacts
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NATO SERS raw re-exploration</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1280px;margin:0 auto;padding:2rem;background:#f6f8fb;color:#17212b}}
h1{{margin-bottom:.3rem}} .lede{{font-size:1.1rem;color:#425466;max-width:900px}}
.summary{{display:flex;gap:1rem;flex-wrap:wrap;margin:1.5rem 0}} .pill{{background:white;border:1px solid #d9e1ea;border-radius:10px;padding:.8rem 1rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:1rem}}
.card{{display:block;background:white;border:1px solid #d9e1ea;border-radius:12px;padding:1rem;text-decoration:none;color:inherit;box-shadow:0 2px 8px rgba(0,0,0,.04)}}
.card:hover{{border-color:#0072B2;transform:translateY(-1px)}} .card h2{{font-size:1.05rem;margin:.1rem 0 .5rem}} .card p{{color:#526372;min-height:3.2rem}} code{{color:#00689d}}
</style></head><body>
<h1>NATO SERS raw-data re-exploration</h1>
<p class="lede">Fresh provenance-first exploration. Previous filtering and preprocessing decisions are not treated as inputs. Open any card for a standalone interactive Plotly view; zoom, pan, hover, and SVG export are enabled.</p>
<div class="summary">
<div class="pill"><b>{summary['raw_files']}</b><br>raw archive files</div>
<div class="pill"><b>{summary['expanded_log_observations']}</b><br>expanded log observations</div>
<div class="pill"><b>{summary['named_sers_observations']}</b><br>named-SERS log observations</div>
<div class="pill"><b>{summary['unique_attributable_sers']}</b><br>unique attributable spectra</div>
<div class="pill"><b>{summary['analysis_masters']}</b><br>physical masters</div>
<div class="pill"><b>{summary['common_effective_support_cm1'][0]:.0f}–{summary['common_effective_support_cm1'][1]:.0f}</b><br>shared measured cm⁻¹</div>
</div>
<div class="grid">{cards}</div>
</body></html>"""
    (output_dir / "index.html").write_text(document)


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    source_root = (args.source_root or Path(config["source_root"])).resolve()
    output_dir = (args.output_dir or repository / config["output_dir"]).resolve()
    tables_dir = output_dir / "tables"
    arrays_dir = output_dir / "arrays"
    html_dir = output_dir / "html"
    for directory in (output_dir, tables_dir, arrays_dir, html_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    file_inventory = inventory_files(source_root)
    file_inventory.to_csv(tables_dir / "raw_file_inventory.csv", index=False)

    master = source_parser.parse_master_samples(
        source_root / "MASTER SAMPLE LIST FOR NATO SERS CDT_MARCH 2024.xlsx"
    )
    records = source_parser.parse_recordings(
        source_root / "01_Raman Team recordings_06Mar2024.xlsx", master
    )
    sources = source_parser.index_sources(source_root)
    records = source_parser.attach_sources(records, sources)
    source_parser.add_reference_and_numeric_qc(records)
    # Exhaust the acquisition metadata already present in vendor PDF reports and
    # RMX TXT headers. This is deliberately performed before tier construction so
    # metadata availability can be audited for included and excluded observations.
    source_parser.add_acquisition_metadata(records)
    recordings = pd.DataFrame(records).sort_values(
        ["session", "excel_row", "recording_subindex"]
    )
    source_inventory, spectra = source_spectrum_inventory(sources)
    source_inventory.to_csv(tables_dir / "source_spectrum_inventory.csv", index=False)
    recordings = attach_source_diagnostics(recordings, source_inventory)
    recordings = fresh_tiers(recordings)
    recordings.to_csv(tables_dir / "recordings_manifest_fresh.csv", index=False)
    for tier in (
        "tier_all_parseable_sers",
        "tier_attributable_sers",
        "tier_unique_attributable_sers",
        "tier_notes_clear_sers",
    ):
        recordings[recordings[tier]].to_csv(tables_dir / f"{tier}.csv", index=False)

    support = support_summary(source_inventory)
    support.to_csv(tables_dir / "native_support_by_instrument.csv", index=False)
    analysis_manifest, axis, raw_matrix, matrix_metadata = analysis_matrix(
        recordings, spectra, float(config["common_grid_step_cm1"])
    )
    analysis_manifest.to_csv(tables_dir / "analysis_manifest.csv", index=False)
    np.savez_compressed(
        arrays_dir / "unique_attributable_common_support_raw.npz",
        axis_cm1=axis,
        intensity=raw_matrix,
        observation_uid=np.asarray(
            analysis_manifest["observation_uid"].astype(str).tolist(), dtype=str
        ),
    )
    embeddings, kmeans_frame, association, pca_diagnostics, arrays = run_embeddings(
        analysis_manifest, raw_matrix, config
    )
    master_embeddings, master_kmeans_frame, master_association = run_master_embeddings(
        analysis_manifest, arrays["minmax"], config
    )
    embeddings.to_csv(tables_dir / "initial_embeddings_and_clusters.csv", index=False)
    kmeans_frame.to_csv(tables_dir / "kmeans_k_search.csv", index=False)
    association.to_csv(tables_dir / "cluster_metadata_association.csv", index=False)
    pca_diagnostics.to_csv(tables_dir / "pca_diagnostics.csv", index=False)
    master_embeddings.to_csv(
        tables_dir / "master_embeddings_and_clusters.csv", index=False
    )
    master_kmeans_frame.to_csv(
        tables_dir / "master_kmeans_k_search.csv", index=False
    )
    master_association.to_csv(
        tables_dir / "master_cluster_metadata_association.csv", index=False
    )
    np.savez_compressed(
        arrays_dir / "initial_representations_and_embeddings.npz",
        **arrays,
        observation_uid=np.asarray(
            analysis_manifest["observation_uid"].astype(str).tolist(), dtype=str
        ),
    )

    duplicate_groups = (
        file_inventory[file_inventory["exact_content_duplicate"]]
        .groupby("sha256")
        .agg(
            references=("relative_path", "count"),
            size_bytes=("size_bytes", "first"),
            paths=("relative_path", lambda values: json.dumps(sorted(values))),
        )
        .reset_index()
    )
    duplicate_groups.to_csv(tables_dir / "exact_file_duplicate_groups.csv", index=False)
    reason_counts = (
        recordings["tier_reason_codes"].value_counts(dropna=False).rename_axis("reason_codes").reset_index(name="observations")
    )
    reason_counts.to_csv(tables_dir / "tier_reason_counts.csv", index=False)

    selected = recordings[recordings["tier_unique_attributable_sers"]]
    summary = {
        "protocol_version": config["protocol_version"],
        "source_root": str(source_root),
        "mirror_root": config.get("mirror_root"),
        "raw_files": len(file_inventory),
        "raw_bytes": int(file_inventory["size_bytes"].sum()),
        "exact_duplicate_file_references": int(file_inventory["exact_content_duplicate"].sum()),
        "exact_duplicate_content_groups": len(duplicate_groups),
        "expanded_log_observations": len(recordings),
        "named_sers_observations": int(recordings["is_named_sers"].sum()),
        "ambiguous_sers_observations": int(recordings["is_ambiguous_sers"].sum()),
        "all_parseable_sers": int(recordings["tier_all_parseable_sers"].sum()),
        "attributable_sers": int(recordings["tier_attributable_sers"].sum()),
        "unique_attributable_sers": int(recordings["tier_unique_attributable_sers"].sum()),
        "notes_clear_sers": int(recordings["tier_notes_clear_sers"].sum()),
        "analysis_masters": int(selected["master_sample_id"].nunique()),
        "analysis_instruments": int(selected["instrument"].nunique()),
        "analysis_sensor_families": int(selected["sensor_family"].nunique()),
        "analysis_targets": int(selected["target_analyte"].nunique()),
        "common_effective_support_cm1": [
            matrix_metadata["common_effective_start_cm1"],
            matrix_metadata["common_effective_end_cm1"],
        ],
        "common_grid_points": matrix_metadata["n_features"],
        "source_spectrum_load_status": source_inventory["load_status"].value_counts().to_dict(),
        "source_index_counts": {key: len(value) for key, value in sources.items()},
        "tier_reason_counts": dict(zip(reason_counts["reason_codes"], reason_counts["observations"])),
    }
    write_json(output_dir / "initial_audit_summary.json", summary)
    write_json(output_dir / "matrix_metadata.json", matrix_metadata)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "umap": umap.__version__,
        "hdbscan": getattr(hdbscan, "__version__", "unknown"),
        "plotly": __import__("plotly").__version__,
        "config_sha256": sha256_file(args.config),
        "source_workbook_hashes": {
            path.name: sha256_file(path) for path in sorted(source_root.glob("*.xlsx"))
        },
        "source_notes_sha256": sha256_file(source_root / "Notes.txt"),
    }
    write_json(output_dir / "environment_and_inputs.json", environment)

    artifacts = make_html_plots(
        html_dir,
        file_inventory,
        recordings,
        analysis_manifest,
        source_inventory,
        support,
        axis,
        raw_matrix,
        embeddings,
        kmeans_frame,
        master_embeddings,
        master_kmeans_frame,
        config,
    )
    write_json(output_dir / "html_artifact_registry.json", artifacts)
    write_index(output_dir, artifacts, summary)
    print(json.dumps(summary, indent=2, default=json_ready))
    print(json.dumps({"html_index": str(output_dir / "index.html"), "plots": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
