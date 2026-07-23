#!/usr/bin/env python3
"""Freeze and benchmark leakage-safe NATO SERS preprocessing representations.

This script consumes the audited 598-spectrum common-grid dataset and its
predefined master-sample folds. It creates a versioned preprocessing bundle
without modifying the source archive or the upstream audit products.

The frozen candidate set contains:

* conservative despiking + per-spectrum min-max scaling,
* conservative despiking + robust percentile min-max scaling,
* domain-blind AsLS or arPLS baseline correction + min-max scaling, and
* the SNV/Savitzky-Golay first- and second-derivative representations used by
  the prior Siamese and classical work.

Candidate selection uses only fixed simple classifiers and nested frozen
master-sample folds. Domain probes are balanced over observed target-domain
cells and are accompanied by target-only null probes so dataset confounding is
not mistaken for spectral instrument leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import OneHotEncoder


VERSION = "nato-sers-preprocessing-v1"
RANDOM_SEED = 2026
N_PCA_COMPONENTS = 32
CORRUPTIONS = ("scale_offset", "baseline", "noise", "spike", "shift", "composite")
CANDIDATE_ORDER = (
    "minimal_minmax",
    "robust_minmax",
    "asls_minmax",
    "arpls_minmax",
    "derivative_1",
    "derivative_2",
)
INTENSITY_CANDIDATES = (
    "minimal_minmax",
    "robust_minmax",
    "asls_minmax",
    "arpls_minmax",
)
DERIVATIVE_CANDIDATES = ("derivative_1", "derivative_2")


@dataclass(frozen=True)
class PreprocessingConfig:
    axis_min_cm1: int = 400
    axis_max_cm1: int = 1800
    axis_step_cm1: int = 1
    spike_min_prominence_fraction: float = 0.10
    spike_max_half_prominence_width_points: float = 1.25
    saturation_min_plateau_points: int = 3
    saturation_relative_tolerance: float = 1.0e-6
    robust_low_percentile: float = 1.0
    robust_high_percentile: float = 99.0
    asls_lambda: float = 1.0e6
    asls_asymmetry: float = 0.001
    asls_iterations: int = 12
    arpls_lambda: float = 1.0e6
    arpls_iterations: int = 30
    arpls_tolerance: float = 1.0e-3
    derivative_window_points: int = 17
    derivative_polynomial_order: int = 3
    injected_noise_fraction: float = 0.03
    injected_baseline_fraction: float = 0.30
    injected_spike_fraction: float = 0.80
    injected_shift_points: int = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    manifest = pd.read_csv(input_dir / "sers_core_manifest.csv")
    archive = np.load(input_dir / "sers_core_spectra_raw_common_grid.npz")
    ids = archive["observation_uid"].astype(str)
    positions = {value: index for index, value in enumerate(ids)}
    if set(manifest["observation_uid"]) != set(ids):
        raise ValueError("Core manifest and common-grid archive have different IDs")
    order = np.asarray([positions[value] for value in manifest["observation_uid"]])
    axis = archive["axis_cm1"].astype(float)
    raw = archive["intensity"][order].astype(float)
    folds = pd.read_csv(input_dir / "grouped_sample_cv_assignments.csv")
    fold_by_id = folds.set_index("observation_uid")["grouped_sample_fold_5"]
    if not set(manifest["observation_uid"]).issubset(fold_by_id.index):
        raise ValueError("Missing grouped fold assignments")
    aligned_folds = pd.DataFrame(
        {
            "observation_uid": manifest["observation_uid"],
            "grouped_sample_fold_5": manifest["observation_uid"].map(fold_by_id).astype(int),
        }
    )
    return manifest, axis, raw, aligned_folds


def minmax_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = values.min(axis=1)
    high = values.max(axis=1)
    scale = high - low
    transformed = (values - low[:, None]) / np.maximum(scale[:, None], 1.0e-12)
    return transformed, low, high


def robust_minmax_rows(
    values: np.ndarray, low_percentile: float, high_percentile: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = np.percentile(values, low_percentile, axis=1)
    high = np.percentile(values, high_percentile, axis=1)
    scale = high - low
    transformed = np.clip(
        (values - low[:, None]) / np.maximum(scale[:, None], 1.0e-12), 0.0, 1.0
    )
    return transformed, low, high


def snv(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.maximum(std, 1.0e-12)


def l2_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-12)


def detect_spikes(
    values: np.ndarray, config: PreprocessingConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Flag only high-prominence peaks narrower than one instrumental feature.

    These are conservative *candidate* cosmic spikes. The mask is preserved so
    no repaired point becomes invisible to later analysis.
    """

    mask = np.zeros_like(values, dtype=bool)
    prominence_fraction = np.zeros_like(values, dtype=np.float32)
    for row_index, row in enumerate(values):
        span = float(np.ptp(row))
        if not np.isfinite(span) or span <= 0:
            continue
        peaks, properties = find_peaks(
            row, prominence=config.spike_min_prominence_fraction * span
        )
        if not len(peaks):
            continue
        widths = peak_widths(row, peaks, rel_height=0.5)[0]
        selected = widths <= config.spike_max_half_prominence_width_points
        selected_peaks = peaks[selected]
        mask[row_index, selected_peaks] = True
        prominence_fraction[row_index, selected_peaks] = (
            properties["prominences"][selected] / span
        ).astype(np.float32)
    return mask, prominence_fraction


def repair_masked_points(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    repaired = values.copy()
    coordinates = np.arange(values.shape[1])
    for row_index in np.flatnonzero(mask.any(axis=1)):
        good = ~mask[row_index]
        if good.sum() < 2:
            continue
        repaired[row_index, mask[row_index]] = np.interp(
            coordinates[mask[row_index]], coordinates[good], values[row_index, good]
        )
    return repaired


def detect_saturation(values: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    mask = np.zeros_like(values, dtype=bool)
    for row_index, row in enumerate(values):
        span = float(np.ptp(row))
        tolerance = max(
            float(np.max(np.abs(row))) * 1.0e-10,
            span * config.saturation_relative_tolerance,
            1.0e-12,
        )
        at_maximum = np.abs(row - np.max(row)) <= tolerance
        padded = np.r_[False, at_maximum, False].astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends):
            if end - start >= config.saturation_min_plateau_points:
                mask[row_index, start:end] = True
    return mask


def second_difference_penalty(n_points: int, lam: float) -> sparse.csc_matrix:
    difference = sparse.diags(
        [np.ones(n_points - 2), -2 * np.ones(n_points - 2), np.ones(n_points - 2)],
        [0, 1, 2],
        shape=(n_points - 2, n_points),
        format="csc",
    )
    return (lam * (difference.T @ difference)).tocsc()


def asls_baseline_matrix(values: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    n_points = values.shape[1]
    penalty = second_difference_penalty(n_points, config.asls_lambda)
    output = np.empty_like(values)
    for row_index, row in enumerate(values):
        weights = np.ones(n_points)
        baseline = np.zeros(n_points)
        for _ in range(config.asls_iterations):
            weight_matrix = sparse.spdiags(weights, 0, n_points, n_points)
            baseline = spsolve(weight_matrix + penalty, weights * row)
            weights = config.asls_asymmetry * (row > baseline) + (
                1.0 - config.asls_asymmetry
            ) * (row <= baseline)
        output[row_index] = baseline
    return output


def arpls_baseline_matrix(values: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    n_points = values.shape[1]
    penalty = second_difference_penalty(n_points, config.arpls_lambda)
    output = np.empty_like(values)
    for row_index, row in enumerate(values):
        weights = np.ones(n_points)
        baseline = np.zeros(n_points)
        for _ in range(config.arpls_iterations):
            weight_matrix = sparse.spdiags(weights, 0, n_points, n_points)
            baseline = spsolve(weight_matrix + penalty, weights * row)
            residual = row - baseline
            negative = residual[residual < 0]
            if negative.size < 2 or float(np.std(negative)) <= 1.0e-12:
                break
            mean_negative = float(np.mean(negative))
            std_negative = float(np.std(negative))
            exponent = 2.0 * (
                residual - (2.0 * std_negative - mean_negative)
            ) / std_negative
            new_weights = 1.0 / (1.0 + np.exp(np.clip(exponent, -60.0, 60.0)))
            relative_change = np.linalg.norm(new_weights - weights) / max(
                np.linalg.norm(weights), 1.0e-12
            )
            weights = new_weights
            if relative_change < config.arpls_tolerance:
                break
        output[row_index] = baseline
    return output


def build_candidates(
    raw: np.ndarray, config: PreprocessingConfig
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    spike_mask, spike_prominence = detect_spikes(raw, config)
    saturation_mask = detect_saturation(raw, config)
    despiked = repair_masked_points(raw, spike_mask)
    minimal, raw_min, raw_max = minmax_rows(despiked)
    robust, robust_low, robust_high = robust_minmax_rows(
        despiked, config.robust_low_percentile, config.robust_high_percentile
    )

    asls_baseline = asls_baseline_matrix(despiked, config)
    asls_corrected = despiked - asls_baseline
    asls_scaled, asls_min, asls_max = minmax_rows(asls_corrected)

    arpls_baseline = arpls_baseline_matrix(despiked, config)
    arpls_corrected = despiked - arpls_baseline
    arpls_scaled, arpls_min, arpls_max = minmax_rows(arpls_corrected)

    normalized = snv(despiked)
    derivative_1 = l2_rows(
        savgol_filter(
            normalized,
            config.derivative_window_points,
            config.derivative_polynomial_order,
            deriv=1,
            axis=1,
        )
    )
    derivative_2 = l2_rows(
        savgol_filter(
            normalized,
            config.derivative_window_points,
            config.derivative_polynomial_order,
            deriv=2,
            axis=1,
        )
    )

    candidates = {
        "minimal_minmax": minimal,
        "robust_minmax": robust,
        "asls_minmax": asls_scaled,
        "arpls_minmax": arpls_scaled,
        "derivative_1": derivative_1,
        "derivative_2": derivative_2,
    }
    auxiliary = {
        "despiked": despiked,
        "spike_mask": spike_mask,
        "spike_prominence_fraction": spike_prominence,
        "saturation_mask": saturation_mask,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "robust_low": robust_low,
        "robust_high": robust_high,
        "asls_baseline": asls_baseline,
        "asls_min": asls_min,
        "asls_max": asls_max,
        "arpls_baseline": arpls_baseline,
        "arpls_min": arpls_min,
        "arpls_max": arpls_max,
    }
    return candidates, auxiliary


def stable_rng(observation_uid: str, corruption: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{RANDOM_SEED}|{observation_uid}|{corruption}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def shift_with_edge(row: np.ndarray, amount: int) -> np.ndarray:
    shifted = np.empty_like(row)
    if amount > 0:
        shifted[:amount] = row[0]
        shifted[amount:] = row[:-amount]
    elif amount < 0:
        shifted[amount:] = row[-1]
        shifted[:amount] = row[-amount:]
    else:
        shifted[:] = row
    return shifted


def corrupt_spectra(
    raw: np.ndarray,
    observation_uids: Iterable[str],
    corruption: str,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    corrupted = raw.copy()
    injected_spikes = np.zeros_like(raw, dtype=bool)
    coordinate = np.linspace(-1.0, 1.0, raw.shape[1])
    for row_index, (row, uid) in enumerate(zip(raw, observation_uids)):
        rng = stable_rng(str(uid), corruption)
        span = max(float(np.ptp(row)), 1.0e-12)
        work = row.copy()

        operations = {corruption} if corruption != "composite" else {
            "scale_offset",
            "baseline",
            "noise",
            "spike",
            "shift",
        }
        if "scale_offset" in operations:
            work = 1.7 * work + 0.15 * span
        if "baseline" in operations:
            slope = rng.uniform(-0.35, 0.35)
            curve = 0.65 * coordinate**2 + slope * coordinate
            curve -= curve.min()
            curve /= max(float(np.ptp(curve)), 1.0e-12)
            work = work + config.injected_baseline_fraction * span * curve
        if "noise" in operations:
            work = work + rng.normal(
                0.0, config.injected_noise_fraction * span, size=work.size
            )
        if "spike" in operations:
            positions = rng.choice(np.arange(10, work.size - 10), size=2, replace=False)
            work[positions] += config.injected_spike_fraction * span
            injected_spikes[row_index, positions] = True
        if "shift" in operations:
            amount = config.injected_shift_points if rng.random() >= 0.5 else -config.injected_shift_points
            work = shift_with_edge(work, amount)
            if injected_spikes[row_index].any():
                shifted_mask = shift_with_edge(
                    injected_spikes[row_index].astype(float), amount
                ) > 0.5
                injected_spikes[row_index] = shifted_mask
        corrupted[row_index] = work
    return corrupted, injected_spikes


def make_preprocessing_manifest(
    manifest: pd.DataFrame,
    folds: pd.DataFrame,
    auxiliary: dict[str, np.ndarray],
) -> pd.DataFrame:
    output = manifest.copy()
    output["dataset_version"] = VERSION
    output["grouped_sample_fold_5"] = folds["grouped_sample_fold_5"].to_numpy()
    output["candidate_spike_count"] = auxiliary["spike_mask"].sum(axis=1)
    output["candidate_spike_max_prominence_fraction"] = auxiliary[
        "spike_prominence_fraction"
    ].max(axis=1)
    output["saturation_point_count"] = auxiliary["saturation_mask"].sum(axis=1)
    output["despiked_min"] = auxiliary["raw_min"]
    output["despiked_max"] = auxiliary["raw_max"]
    output["despiked_range"] = auxiliary["raw_max"] - auxiliary["raw_min"]
    output["robust_p01"] = auxiliary["robust_low"]
    output["robust_p99"] = auxiliary["robust_high"]
    output["asls_corrected_min"] = auxiliary["asls_min"]
    output["asls_corrected_max"] = auxiliary["asls_max"]
    output["arpls_corrected_min"] = auxiliary["arpls_min"]
    output["arpls_corrected_max"] = auxiliary["arpls_max"]
    return output


def write_candidate_archive(
    path: Path,
    axis: np.ndarray,
    observation_uids: np.ndarray,
    raw: np.ndarray,
    candidates: dict[str, np.ndarray],
    auxiliary: dict[str, np.ndarray],
    selection: np.ndarray | None = None,
) -> None:
    if selection is None:
        selection = np.arange(len(observation_uids))
    arrays: dict[str, np.ndarray] = {
        "axis_cm1": axis.astype(np.float32),
        "observation_uid": observation_uids[selection].astype(str),
        "raw_common_grid": raw[selection].astype(np.float32),
        "despiked_common_grid": auxiliary["despiked"][selection].astype(np.float32),
        "spike_mask": auxiliary["spike_mask"][selection],
        "saturation_mask": auxiliary["saturation_mask"][selection],
        "asls_baseline": auxiliary["asls_baseline"][selection].astype(np.float32),
        "arpls_baseline": auxiliary["arpls_baseline"][selection].astype(np.float32),
    }
    arrays.update(
        {name: values[selection].astype(np.float32) for name, values in candidates.items()}
    )
    np.savez_compressed(path, **arrays)


def write_frozen_model_archive(
    path: Path,
    axis: np.ndarray,
    observation_uids: np.ndarray,
    candidates: dict[str, np.ndarray],
    selected_representations: list[str],
    selection: np.ndarray | None = None,
) -> None:
    """Write only the representations authorized for downstream comparisons."""
    if selection is None:
        selection = np.arange(len(observation_uids))
    arrays: dict[str, np.ndarray] = {
        "axis_cm1": axis.astype(np.float32),
        "observation_uid": observation_uids[selection].astype(str),
    }
    arrays.update(
        {
            name: candidates[name][selection].astype(np.float32)
            for name in selected_representations
        }
    )
    np.savez_compressed(path, **arrays)


def nested_assignments(manifest: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    folds = manifest["grouped_sample_fold_5"].to_numpy()
    for outer_fold in sorted(np.unique(folds)):
        frame = manifest[
            [
                "observation_uid",
                "master_sample_id",
                "target_analyte",
                "include_sers_qc_pass",
                "grouped_sample_fold_5",
            ]
        ].copy()
        frame.insert(0, "outer_fold", int(outer_fold))
        frame["outer_partition"] = np.where(
            folds == outer_fold, "test", "development"
        )
        frame["inner_validation_fold"] = np.where(
            folds == outer_fold, pd.NA, folds
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def pca_features(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, PCA]:
    n_components = min(N_PCA_COMPONENTS, train.shape[0] - 1, train.shape[1])
    model = PCA(
        n_components=n_components,
        whiten=True,
        svd_solver="randomized",
        random_state=RANDOM_SEED,
    )
    return model.fit_transform(train), model.transform(test), model


def supported_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    recalls = [float(np.mean(y_pred[y_true == cls] == cls)) for cls in classes]
    return float(np.mean(recalls))


def target_metrics(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, PCA]:
    train_pca, test_pca, pca = pca_features(train_features, test_features)
    logistic = LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=RANDOM_SEED
    )
    logistic.fit(train_pca, train_labels)
    logistic_prediction = logistic.predict(test_pca)

    centroid = NearestCentroid()
    centroid.fit(train_features, train_labels)
    centroid_prediction = centroid.predict(test_features)
    labels_for_f1 = np.unique(test_labels)
    metrics = {
        "target_pca_balanced_accuracy": supported_balanced_accuracy(
            test_labels, logistic_prediction
        ),
        "target_pca_macro_f1": float(
            f1_score(
                test_labels,
                logistic_prediction,
                labels=labels_for_f1,
                average="macro",
                zero_division=0,
            )
        ),
        "target_centroid_balanced_accuracy": supported_balanced_accuracy(
            test_labels, centroid_prediction
        ),
        "target_centroid_macro_f1": float(
            f1_score(
                test_labels,
                centroid_prediction,
                labels=labels_for_f1,
                average="macro",
                zero_division=0,
            )
        ),
    }
    return metrics, logistic_prediction, centroid_prediction, pca


def target_domain_cell_weights(targets: np.ndarray, domains: np.ndarray) -> np.ndarray:
    cells = pd.Series(list(zip(targets, domains)))
    counts = cells.value_counts()
    weights = np.asarray([1.0 / counts[cell] for cell in cells], dtype=float)
    return weights / np.mean(weights)


def cell_balanced_domain_accuracy(
    targets: np.ndarray, domains: np.ndarray, predictions: np.ndarray
) -> float:
    frame = pd.DataFrame(
        {"target": targets, "domain": domains, "correct": predictions == domains}
    )
    return float(frame.groupby(["target", "domain"])["correct"].mean().mean())


def domain_probe(
    train_pca: np.ndarray,
    test_pca: np.ndarray,
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    train_domains: np.ndarray,
    test_domains: np.ndarray,
) -> dict[str, float]:
    weights = target_domain_cell_weights(train_targets, train_domains)
    spectral_model = LogisticRegression(max_iter=3000, random_state=RANDOM_SEED)
    spectral_model.fit(train_pca, train_domains, sample_weight=weights)
    spectral_prediction = spectral_model.predict(test_pca)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_target_only = encoder.fit_transform(train_targets.reshape(-1, 1))
    test_target_only = encoder.transform(test_targets.reshape(-1, 1))
    null_model = LogisticRegression(max_iter=3000, random_state=RANDOM_SEED)
    null_model.fit(train_target_only, train_domains, sample_weight=weights)
    null_prediction = null_model.predict(test_target_only)

    spectral_cell_score = cell_balanced_domain_accuracy(
        test_targets, test_domains, spectral_prediction
    )
    null_cell_score = cell_balanced_domain_accuracy(
        test_targets, test_domains, null_prediction
    )
    return {
        "cell_balanced_accuracy": spectral_cell_score,
        "target_only_null_cell_balanced_accuracy": null_cell_score,
        "increment_over_target_only": spectral_cell_score - null_cell_score,
        "ordinary_balanced_accuracy": supported_balanced_accuracy(
            test_domains, spectral_prediction
        ),
    }


def row_correlation_normalize(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    return centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1.0e-12)


def geometry_metrics(values: np.ndarray, metadata: pd.DataFrame) -> dict[str, float | int]:
    if len(values) < 2:
        return {}
    normalized = row_correlation_normalize(values)
    correlation_distance = 1.0 - normalized @ normalized.T
    upper_i, upper_j = np.triu_indices(len(values), k=1)
    target = metadata["target_analyte"].astype(str).to_numpy()
    sample = metadata["master_sample_id"].to_numpy()
    instrument = metadata["instrument"].astype(str).to_numpy()
    sensor = metadata["sensor_family"].astype(str).to_numpy()

    categories = {
        "same_master_cross_instrument": (
            (sample[upper_i] == sample[upper_j])
            & (instrument[upper_i] != instrument[upper_j])
        ),
        "same_master_cross_sensor": (
            (sample[upper_i] == sample[upper_j]) & (sensor[upper_i] != sensor[upper_j])
        ),
        "same_target_different_master_cross_instrument": (
            (target[upper_i] == target[upper_j])
            & (sample[upper_i] != sample[upper_j])
            & (instrument[upper_i] != instrument[upper_j])
        ),
        "different_target": target[upper_i] != target[upper_j],
    }
    output: dict[str, float | int] = {}
    pair_values = correlation_distance[upper_i, upper_j]
    for name, selected in categories.items():
        values_for_category = pair_values[selected]
        output[f"geometry_{name}_pairs"] = int(values_for_category.size)
        output[f"geometry_{name}_mean_distance"] = (
            float(np.mean(values_for_category)) if values_for_category.size else np.nan
        )
        output[f"geometry_{name}_median_distance"] = (
            float(np.median(values_for_category)) if values_for_category.size else np.nan
        )
    same = output.get("geometry_same_master_cross_instrument_mean_distance", np.nan)
    different = output.get("geometry_different_target_mean_distance", np.nan)
    output["geometry_cross_instrument_separation_margin"] = (
        float(different - same)
        if np.isfinite(float(same)) and np.isfinite(float(different))
        else np.nan
    )
    return output


def match_peak_positions(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference_peaks, _ = find_peaks(reference, prominence=0.05, distance=5)
    candidate_peaks, _ = find_peaks(candidate, prominence=0.05, distance=5)
    if not len(reference_peaks):
        return {"reference_peaks": 0, "peak_recall_5cm": np.nan, "peak_shift_cm": np.nan}
    available = list(candidate_peaks)
    shifts: list[float] = []
    for peak in reference_peaks:
        if not available:
            continue
        differences = np.abs(np.asarray(available) - peak)
        best_index = int(np.argmin(differences))
        if differences[best_index] <= 5:
            shifts.append(float(differences[best_index]))
            available.pop(best_index)
    return {
        "reference_peaks": int(len(reference_peaks)),
        "peak_recall_5cm": float(len(shifts) / len(reference_peaks)),
        "peak_shift_cm": float(np.mean(shifts)) if shifts else np.nan,
    }


def peak_preservation_summary(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float | int]:
    rows = [match_peak_positions(ref, cand) for ref, cand in zip(reference, candidate)]
    frame = pd.DataFrame(rows)
    correlations = np.asarray(
        [
            np.corrcoef(ref, cand)[0, 1]
            if np.std(ref) > 0 and np.std(cand) > 0
            else np.nan
            for ref, cand in zip(reference, candidate)
        ]
    )
    return {
        "spectra_with_reference_peaks": int((frame["reference_peaks"] > 0).sum()),
        "mean_peak_recall_5cm": float(frame["peak_recall_5cm"].mean()),
        "median_peak_recall_5cm": float(frame["peak_recall_5cm"].median()),
        "mean_peak_shift_cm": float(frame["peak_shift_cm"].mean()),
        "median_spectral_correlation_to_minimal": float(np.nanmedian(correlations)),
    }


def evaluate_split(
    representation: str,
    clean: np.ndarray,
    corrupted: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, Any]:
    train_labels = manifest.loc[train_mask, "target_analyte"].astype(str).to_numpy()
    test_labels = manifest.loc[test_mask, "target_analyte"].astype(str).to_numpy()
    clean_train = clean[train_mask]
    clean_test = clean[test_mask]
    metrics, pca_prediction, centroid_prediction, pca = target_metrics(
        clean_train, clean_test, train_labels, test_labels
    )
    train_pca = pca.transform(clean_train)
    test_pca = pca.transform(clean_test)

    for domain_column, prefix in (
        ("instrument", "instrument_probe"),
        ("sensor_family", "sensor_probe"),
    ):
        probe = domain_probe(
            train_pca,
            test_pca,
            train_labels,
            test_labels,
            manifest.loc[train_mask, domain_column].astype(str).to_numpy(),
            manifest.loc[test_mask, domain_column].astype(str).to_numpy(),
        )
        metrics.update({f"{prefix}_{key}": value for key, value in probe.items()})

    logistic = LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=RANDOM_SEED
    )
    logistic.fit(train_pca, train_labels)
    centroid = NearestCentroid().fit(clean_train, train_labels)
    for corruption, values in corrupted.items():
        corrupted_test = values[test_mask]
        pca_corrupt_prediction = logistic.predict(pca.transform(corrupted_test))
        centroid_corrupt_prediction = centroid.predict(corrupted_test)
        metrics[f"corruption_{corruption}_pca_balanced_accuracy"] = (
            supported_balanced_accuracy(test_labels, pca_corrupt_prediction)
        )
        metrics[f"corruption_{corruption}_centroid_balanced_accuracy"] = (
            supported_balanced_accuracy(test_labels, centroid_corrupt_prediction)
        )
        metrics[f"corruption_{corruption}_pca_prediction_agreement"] = float(
            np.mean(pca_corrupt_prediction == pca_prediction)
        )
        metrics[f"corruption_{corruption}_centroid_prediction_agreement"] = float(
            np.mean(centroid_corrupt_prediction == centroid_prediction)
        )
        clean_norm = row_correlation_normalize(clean_test)
        corrupt_norm = row_correlation_normalize(corrupted_test)
        metrics[f"corruption_{corruption}_mean_shape_correlation"] = float(
            np.mean(np.sum(clean_norm * corrupt_norm, axis=1))
        )

    metrics.update(geometry_metrics(clean_test, manifest.loc[test_mask]))
    metrics.update(
        {
            "representation": representation,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_train_master_samples": int(
                manifest.loc[train_mask, "master_sample_id"].nunique()
            ),
            "n_test_master_samples": int(
                manifest.loc[test_mask, "master_sample_id"].nunique()
            ),
        }
    )
    return metrics


def benchmark_candidates(
    manifest: pd.DataFrame,
    candidates: dict[str, np.ndarray],
    corrupted_candidates: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    folds = manifest["grouped_sample_fold_5"].to_numpy()
    subsets = {
        "strict_core": np.ones(len(manifest), dtype=bool),
        "quality_pass": manifest["include_sers_qc_pass"].astype(bool).to_numpy(),
    }
    for subset_name, subset_mask in subsets.items():
        for outer_fold in sorted(np.unique(folds)):
            outer_train = subset_mask & (folds != outer_fold)
            outer_test = subset_mask & (folds == outer_fold)
            for representation in CANDIDATE_ORDER:
                record = evaluate_split(
                    representation,
                    candidates[representation],
                    corrupted_candidates[representation],
                    manifest,
                    outer_train,
                    outer_test,
                )
                record.update(
                    {
                        "subset": subset_name,
                        "stage": "outer_test",
                        "outer_fold": int(outer_fold),
                        "inner_validation_fold": pd.NA,
                    }
                )
                records.append(record)

            for inner_fold in sorted(set(np.unique(folds)) - {outer_fold}):
                inner_train = subset_mask & (folds != outer_fold) & (folds != inner_fold)
                inner_validation = subset_mask & (folds == inner_fold)
                for representation in CANDIDATE_ORDER:
                    record = evaluate_split(
                        representation,
                        candidates[representation],
                        corrupted_candidates[representation],
                        manifest,
                        inner_train,
                        inner_validation,
                    )
                    record.update(
                        {
                            "subset": subset_name,
                            "stage": "inner_validation",
                            "outer_fold": int(outer_fold),
                            "inner_validation_fold": int(inner_fold),
                        }
                    )
                    records.append(record)
    return pd.DataFrame(records)


def summarize_benchmarks(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in {"outer_fold", "inner_validation_fold"}
    ]
    summary = (
        metrics.groupby(["representation", "subset", "stage"])[numeric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def corruption_detection_summary(
    injected_masks: dict[str, np.ndarray],
    detected_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for corruption in CORRUPTIONS:
        injected = injected_masks[corruption]
        detected = detected_masks[corruption]
        true_positive = int(np.sum(injected & detected))
        injected_count = int(injected.sum())
        detected_count = int(detected.sum())
        output[corruption] = {
            "injected_spike_points": injected_count,
            "detected_candidate_points": detected_count,
            "injected_spike_recall": (
                true_positive / injected_count if injected_count else None
            ),
            "injected_spike_precision_among_detected": (
                true_positive / detected_count if detected_count else None
            ),
        }
    return output


def objective_table(
    metrics: pd.DataFrame, peak_metrics: pd.DataFrame
) -> pd.DataFrame:
    inner = metrics[metrics["stage"] == "inner_validation"]
    records: list[dict[str, Any]] = []
    for representation in CANDIDATE_ORDER:
        core = inner[
            (inner["representation"] == representation)
            & (inner["subset"] == "strict_core")
        ]
        quality = inner[
            (inner["representation"] == representation)
            & (inner["subset"] == "quality_pass")
        ]
        peak = peak_metrics[
            (peak_metrics["representation"] == representation)
            & (peak_metrics["subset"] == "strict_core")
        ]
        records.append(
            {
                "representation": representation,
                "family": (
                    "intensity" if representation in INTENSITY_CANDIDATES else "derivative"
                ),
                "target_core": core["target_pca_balanced_accuracy"].mean(),
                "target_quality": quality["target_pca_balanced_accuracy"].mean(),
                "target_centroid_core": core[
                    "target_centroid_balanced_accuracy"
                ].mean(),
                "instrument_leakage_increment": core[
                    "instrument_probe_increment_over_target_only"
                ].mean(),
                "sensor_leakage_increment": core[
                    "sensor_probe_increment_over_target_only"
                ].mean(),
                "composite_corruption_target": core[
                    "corruption_composite_pca_balanced_accuracy"
                ].mean(),
                "composite_prediction_agreement": core[
                    "corruption_composite_pca_prediction_agreement"
                ].mean(),
                "same_master_cross_instrument_distance": core[
                    "geometry_same_master_cross_instrument_mean_distance"
                ].mean(),
                "cross_instrument_separation_margin": core[
                    "geometry_cross_instrument_separation_margin"
                ].mean(),
                "peak_recall": (
                    peak["mean_peak_recall_5cm"].mean()
                    if representation in INTENSITY_CANDIDATES
                    else np.nan
                ),
                "spectral_correlation_to_minimal": (
                    peak["median_spectral_correlation_to_minimal"].mean()
                    if representation in INTENSITY_CANDIDATES
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def pareto_mask(frame: pd.DataFrame, objectives: dict[str, str]) -> np.ndarray:
    values = frame[list(objectives)].to_numpy(dtype=float).copy()
    for column_index, direction in enumerate(objectives.values()):
        if direction == "min":
            values[:, column_index] *= -1.0
    result = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        if not result[index]:
            continue
        dominated = np.all(values >= values[index], axis=1) & np.any(
            values > values[index], axis=1
        )
        dominated[index] = False
        if np.any(dominated):
            result[index] = False
    return result


def normalized_utility(frame: pd.DataFrame, objectives: dict[str, str]) -> np.ndarray:
    utility = np.zeros(len(frame), dtype=float)
    for column, direction in objectives.items():
        values = frame[column].to_numpy(dtype=float)
        low, high = np.nanmin(values), np.nanmax(values)
        normalized = (
            np.full(len(values), 0.5)
            if high - low <= 1.0e-12
            else (values - low) / (high - low)
        )
        utility += normalized if direction == "max" else 1.0 - normalized
    return utility / len(objectives)


def select_representations(objectives_frame: pd.DataFrame) -> dict[str, Any]:
    common_objectives = {
        "target_core": "max",
        "target_quality": "max",
        "composite_corruption_target": "max",
        "instrument_leakage_increment": "min",
        "sensor_leakage_increment": "min",
        "same_master_cross_instrument_distance": "min",
        "cross_instrument_separation_margin": "max",
    }
    frame = objectives_frame.copy()
    frame["global_pareto"] = pareto_mask(frame, common_objectives)
    frame["generalization_utility"] = normalized_utility(frame, common_objectives)
    frame["role_pareto"] = False
    frame["role_utility"] = np.nan
    selected: list[str] = []
    role_fronts: dict[str, list[str]] = {}
    for family in ("intensity", "derivative"):
        indices = frame.index[frame["family"] == family]
        family_frame = frame.loc[indices]
        family_objectives = dict(common_objectives)
        if family == "intensity":
            family_objectives.update(
                {
                    "peak_recall": "max",
                    "spectral_correlation_to_minimal": "max",
                }
            )
        family_mask = pareto_mask(family_frame, family_objectives)
        front_indices = family_frame.index[family_mask]
        frame.loc[front_indices, "role_pareto"] = True
        frame.loc[indices, "role_utility"] = normalized_utility(
            family_frame, family_objectives
        )
        ordered = frame.loc[front_indices].sort_values(
            ["role_utility", "target_core"], ascending=False
        )
        role_fronts[family] = ordered["representation"].tolist()

    # The minimally processed spectrum is the causal no-baseline control for
    # all reconstructive models. It is retained only when it is globally
    # Pareto-optimal, as it is for this frozen dataset. One corrected intensity
    # representation and one derivative control are then selected by utility
    # from their role-specific Pareto fronts.
    minimal_row = frame[frame["representation"] == "minimal_minmax"].iloc[0]
    if bool(minimal_row["global_pareto"]):
        selected.append("minimal_minmax")
    corrected_front = frame[
        frame["representation"].isin(role_fronts["intensity"])
        & frame["representation"].isin(("asls_minmax", "arpls_minmax"))
        & (frame["peak_recall"] >= 0.90)
        & (frame["spectral_correlation_to_minimal"] >= 0.90)
    ].sort_values(["generalization_utility", "target_core"], ascending=False)
    if not corrected_front.empty:
        selected.append(str(corrected_front.iloc[0]["representation"]))
    derivative_front = frame[
        frame["representation"].isin(role_fronts["derivative"])
    ].sort_values(["generalization_utility", "target_core"], ascending=False)
    if not derivative_front.empty:
        selected.append(str(derivative_front.iloc[0]["representation"]))
    frame["selected_frozen"] = frame["representation"].isin(selected)
    return {
        "selection_basis": (
            "Nested inner-validation Pareto selection. The globally Pareto-optimal "
            "minimal min-max representation is retained as the causal no-baseline "
            "control; corrected intensity candidates must preserve at least 90% of "
            "reference peaks within 5 cm^-1 and retain at least 0.90 median spectral "
            "correlation, after which one corrected representation and one derivative "
            "control are selected by equal-weight generalization utility from their "
            "role-specific Pareto fronts."
        ),
        "common_objectives": common_objectives,
        "intensity_role_additional_objectives": {
            "peak_recall": "max",
            "spectral_correlation_to_minimal": "max",
        },
        "corrected_intensity_peak_gates": {
            "minimum_peak_recall_5cm": 0.90,
            "minimum_median_spectral_correlation": 0.90,
        },
        "global_pareto": frame.loc[frame["global_pareto"], "representation"].tolist(),
        "role_pareto": role_fronts,
        "selected_frozen_representations": selected,
        "table": frame.to_dict(orient="records"),
    }


def peak_metrics_table(
    manifest: pd.DataFrame, candidates: dict[str, np.ndarray]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    folds = manifest["grouped_sample_fold_5"].to_numpy()
    subsets = {
        "strict_core": np.ones(len(manifest), dtype=bool),
        "quality_pass": manifest["include_sers_qc_pass"].astype(bool).to_numpy(),
    }
    for subset_name, subset in subsets.items():
        for fold in sorted(np.unique(folds)):
            mask = subset & (folds == fold)
            reference = candidates["minimal_minmax"][mask]
            for representation in CANDIDATE_ORDER:
                record: dict[str, Any] = {
                    "subset": subset_name,
                    "outer_fold": int(fold),
                    "representation": representation,
                    "applicable": representation in INTENSITY_CANDIDATES,
                }
                if representation in INTENSITY_CANDIDATES:
                    record.update(
                        peak_preservation_summary(reference, candidates[representation][mask])
                    )
                records.append(record)
    return pd.DataFrame(records)


def plot_tradeoffs(output: Path, objective_frame: pd.DataFrame, selection: dict[str, Any]) -> None:
    selected = set(selection["selected_frozen_representations"])
    leakage = 0.5 * (
        objective_frame["instrument_leakage_increment"]
        + objective_frame["sensor_leakage_increment"]
    )
    fig, axis = plt.subplots(figsize=(8, 6))
    colors = {"intensity": "#174A7E", "derivative": "#D1495B"}
    label_offsets = {
        "minimal_minmax": (5, -14),
        "robust_minmax": (5, 7),
        "asls_minmax": (5, 7),
        "arpls_minmax": (5, -14),
        "derivative_1": (5, 7),
        "derivative_2": (5, 7),
    }
    for row_index, row in objective_frame.iterrows():
        axis.scatter(
            leakage.iloc[row_index],
            row["target_core"],
            s=150 if row["representation"] in selected else 75,
            marker="*" if row["representation"] in selected else "o",
            color=colors[row["family"]],
            edgecolor="black",
            linewidth=0.7,
        )
        axis.annotate(
            row["representation"],
            (leakage.iloc[row_index], row["target_core"]),
            xytext=label_offsets[row["representation"]],
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Mean target-adjusted instrument/sensor leakage increment (lower is better)")
    axis.set_ylabel("Nested target balanced accuracy (higher is better)")
    axis.set_title("NATO SERS preprocessing trade-off; stars are frozen selections")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_corruption(output: Path, objective_frame: pd.DataFrame, selection: dict[str, Any]) -> None:
    selected = selection["selected_frozen_representations"]
    frame = objective_frame.set_index("representation").loc[selected]
    x = np.arange(len(frame))
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x - 0.18, frame["target_core"], 0.36, label="clean")
    axis.bar(
        x + 0.18,
        frame["composite_corruption_target"],
        0.36,
        label="composite corruption",
    )
    axis.set_xticks(x, frame.index, rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Nested target balanced accuracy")
    axis.set_title("Clean and corrupted performance of frozen representations")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_version_record(
    script_path: Path,
    input_dir: Path,
    config: PreprocessingConfig,
    manifest: pd.DataFrame,
    axis: np.ndarray,
) -> dict[str, Any]:
    input_files = [
        input_dir / "sers_core_manifest.csv",
        input_dir / "sers_qc_pass_manifest.csv",
        input_dir / "sers_core_spectra_raw_common_grid.npz",
        input_dir / "grouped_sample_cv_assignments.csv",
        input_dir / "domain_evaluation_partitions.csv",
        script_path,
    ]
    return {
        "dataset_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive_modified": False,
        "strict_core_observations": int(len(manifest)),
        "quality_pass_observations": int(manifest["include_sers_qc_pass"].sum()),
        "axis_cm1": {
            "minimum": float(axis.min()),
            "maximum": float(axis.max()),
            "step": float(np.median(np.diff(axis))),
            "points": int(len(axis)),
        },
        "candidate_order": list(CANDIDATE_ORDER),
        "config": config.__dict__,
        "fixed_baseline": {
            "pca_components": N_PCA_COMPONENTS,
            "pca_whiten": True,
            "classifier": "class-balanced logistic regression",
            "secondary_classifier": "nearest centroid",
            "seed": RANDOM_SEED,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "input_sha256": {str(path.resolve()): sha256_file(path) for path in input_files},
    }


def write_bundle_readme(
    output_path: Path,
    objective_frame: pd.DataFrame,
    selection: dict[str, Any],
    detection_summary: dict[str, Any],
) -> None:
    selected = selection["selected_frozen_representations"]
    rows = []
    for row in objective_frame.itertuples(index=False):
        rows.append(
            "| {name} | {target:.3f} | {quality:.3f} | {corrupt:.3f} | "
            "{instrument:.3f} | {sensor:.3f} | {distance:.3f} | {selected} |".format(
                name=row.representation,
                target=row.target_core,
                quality=row.target_quality,
                corrupt=row.composite_corruption_target,
                instrument=row.instrument_leakage_increment,
                sensor=row.sensor_leakage_increment,
                distance=row.same_master_cross_instrument_distance,
                selected="yes" if row.representation in selected else "no",
            )
        )
    observed_spikes = sum(
        item["candidate_spike_points"]
        for item in detection_summary["observed_candidate_flags_by_instrument"].values()
    )
    observed_saturated = sum(
        item["spectra_with_saturation_plateau"]
        for item in detection_summary["observed_candidate_flags_by_instrument"].values()
    )
    output_path.write_text(
        "\n".join(
            [
                "# NATO SERS frozen preprocessing v1",
                "",
                "This bundle freezes six auditable representations for the 598-spectrum "
                "strict core and the aligned 500-spectrum quality subset. All source "
                "spectra remain unchanged.",
                "",
                "## Frozen selections",
                "",
                *[f"- `{name}`" for name in selected],
                "",
                "`minimal_minmax` is the no-baseline reconstructive control. The "
                "baseline-corrected and derivative selections were chosen from "
                "role-specific nested inner-validation Pareto fronts.",
                "",
                "## Nested selection evidence",
                "",
                "| Representation | Target core | Target quality | Composite corruption | "
                "Instrument leakage increment | Sensor leakage increment | Same-master "
                "cross-instrument distance | Frozen |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
                *rows,
                "",
                "Higher target/corruption scores are better. Lower target-adjusted "
                "domain leakage and same-master distance are better. These are fixed "
                "PCA/logistic screening baselines, not VAE results.",
                "",
                "## Artifact detection",
                "",
                f"- Conservative candidate spike points in observed core: {observed_spikes}",
                f"- Spectra with numeric maximum plateaus: {observed_saturated}",
                "- Candidate spike locations are preserved in `spike_mask`; repaired "
                "values never overwrite `raw_common_grid`.",
                "- Synthetic injection recall and precision are in "
                "`artifact_detection_summary.json`.",
                "",
                "## Files",
                "",
                "- `candidate_spectra_core.npz` and `candidate_spectra_quality.npz`: "
                "raw, despiked, masks, baselines, and six representations;",
                "- `frozen_model_inputs_core.npz` and "
                "`frozen_model_inputs_quality.npz`: only the selected arrays "
                "authorized as downstream model inputs;",
                "- `core_preprocessing_manifest.csv` and "
                "`quality_preprocessing_manifest.csv`: provenance, flags, folds, and "
                "scaling metadata;",
                "- `benchmark_fold_metrics.csv`: all outer and nested-inner results;",
                "- `selection_objectives.csv` and `frozen_selection.json`: explicit "
                "selection evidence;",
                "- `dataset_version.json` and `artifact_hashes.json`: configuration, "
                "software, input hashes, and bundle hashes.",
                "",
                "## Rebuild and validate",
                "",
                "```bash",
                ".venv/bin/python scripts/freeze_nato_sers_preprocessing.py",
                ".venv/bin/python scripts/validate_nato_sers_preprocessing_freeze.py",
                "```",
                "",
                "The selected arrays are fixed inputs for subsequent AE, denoising-AE, "
                "VAE, disentangled-VAE, and Siamese-hybrid comparisons. Model selection "
                "must not alter these preprocessing choices using outer-test results.",
                "",
            ]
        )
    )


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repository / "Workspace" / "nato_sers_field_trial",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "nato_sers_field_trial"
        / "preprocessing_v1",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    config = PreprocessingConfig()

    manifest, axis, raw, folds = load_inputs(input_dir)
    expected_axis = np.arange(
        config.axis_min_cm1,
        config.axis_max_cm1 + config.axis_step_cm1,
        config.axis_step_cm1,
    )
    if not np.array_equal(axis, expected_axis):
        raise ValueError("Input archive does not use the declared 400--1800 cm^-1 grid")

    candidates, auxiliary = build_candidates(raw, config)
    preprocessing_manifest = make_preprocessing_manifest(manifest, folds, auxiliary)
    preprocessing_manifest.to_csv(
        output_dir / "core_preprocessing_manifest.csv", index=False
    )
    quality_indices = np.flatnonzero(
        preprocessing_manifest["include_sers_qc_pass"].astype(bool).to_numpy()
    )
    preprocessing_manifest.iloc[quality_indices].to_csv(
        output_dir / "quality_preprocessing_manifest.csv", index=False
    )

    observation_uids = manifest["observation_uid"].astype(str).to_numpy()
    write_candidate_archive(
        output_dir / "candidate_spectra_core.npz",
        axis,
        observation_uids,
        raw,
        candidates,
        auxiliary,
    )
    write_candidate_archive(
        output_dir / "candidate_spectra_quality.npz",
        axis,
        observation_uids,
        raw,
        candidates,
        auxiliary,
        quality_indices,
    )

    core_splits = preprocessing_manifest[
        [
            "observation_uid",
            "master_sample_id",
            "target_analyte",
            "instrument",
            "sensor_family",
            "include_sers_qc_pass",
            "grouped_sample_fold_5",
        ]
    ]
    core_splits.to_csv(output_dir / "core_split_assignments.csv", index=False)
    core_splits[core_splits["include_sers_qc_pass"]].to_csv(
        output_dir / "quality_split_assignments.csv", index=False
    )
    nested_assignments(preprocessing_manifest).to_csv(
        output_dir / "nested_group_cv_assignments.csv", index=False
    )
    domain_partitions = pd.read_csv(input_dir / "domain_evaluation_partitions.csv")
    domain_partitions.to_csv(
        output_dir / "domain_evaluation_partitions_core.csv", index=False
    )
    domain_partitions[domain_partitions["include_sers_qc_pass"]].to_csv(
        output_dir / "domain_evaluation_partitions_quality.csv", index=False
    )

    corrupted_candidates: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in CANDIDATE_ORDER
    }
    injected_masks: dict[str, np.ndarray] = {}
    detected_masks: dict[str, np.ndarray] = {}
    for corruption in CORRUPTIONS:
        corrupted_raw, injected_mask = corrupt_spectra(
            raw, observation_uids, corruption, config
        )
        corruption_candidates, corruption_auxiliary = build_candidates(
            corrupted_raw, config
        )
        injected_masks[corruption] = injected_mask
        detected_masks[corruption] = corruption_auxiliary["spike_mask"]
        for representation in CANDIDATE_ORDER:
            corrupted_candidates[representation][corruption] = (
                corruption_candidates[representation]
            )

    detection_summary = corruption_detection_summary(injected_masks, detected_masks)
    actual_spike_frame = pd.DataFrame(
        {
            "instrument": manifest["instrument"],
            "candidate_spike_count": auxiliary["spike_mask"].sum(axis=1),
            "saturation_point_count": auxiliary["saturation_mask"].sum(axis=1),
        }
    )
    detection_summary["observed_candidate_flags_by_instrument"] = {
        instrument: {
            "spectra_with_candidate_spikes": int((group["candidate_spike_count"] > 0).sum()),
            "candidate_spike_points": int(group["candidate_spike_count"].sum()),
            "spectra_with_saturation_plateau": int(
                (group["saturation_point_count"] > 0).sum()
            ),
        }
        for instrument, group in actual_spike_frame.groupby("instrument")
    }
    (output_dir / "artifact_detection_summary.json").write_text(
        json.dumps(detection_summary, indent=2, sort_keys=True, default=json_default)
        + "\n"
    )

    fold_metrics = benchmark_candidates(
        preprocessing_manifest, candidates, corrupted_candidates
    )
    fold_metrics.to_csv(output_dir / "benchmark_fold_metrics.csv", index=False)
    benchmark_summary = summarize_benchmarks(fold_metrics)
    benchmark_summary.to_csv(output_dir / "benchmark_summary.csv", index=False)

    peak_metrics = peak_metrics_table(preprocessing_manifest, candidates)
    peak_metrics.to_csv(output_dir / "peak_preservation_metrics.csv", index=False)
    objective_frame = objective_table(fold_metrics, peak_metrics)
    objective_frame.to_csv(output_dir / "selection_objectives.csv", index=False)
    selection = select_representations(objective_frame)
    (output_dir / "frozen_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    selected_representations = selection["selected_frozen_representations"]
    write_frozen_model_archive(
        output_dir / "frozen_model_inputs_core.npz",
        axis,
        observation_uids,
        candidates,
        selected_representations,
    )
    write_frozen_model_archive(
        output_dir / "frozen_model_inputs_quality.npz",
        axis,
        observation_uids,
        candidates,
        selected_representations,
        quality_indices,
    )

    plot_tradeoffs(figures / "preprocessing_tradeoff.png", objective_frame, selection)
    plot_corruption(figures / "selected_corruption_robustness.png", objective_frame, selection)

    version_record = build_version_record(
        Path(__file__).resolve(), input_dir, config, preprocessing_manifest, axis
    )
    version_record["selected_frozen_representations"] = selection[
        "selected_frozen_representations"
    ]
    (output_dir / "dataset_version.json").write_text(
        json.dumps(version_record, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    write_bundle_readme(
        output_dir / "README.md", objective_frame, selection, detection_summary
    )

    artifacts = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "artifact_hashes.json"
    )
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(
            {str(path.relative_to(output_dir)): sha256_file(path) for path in artifacts},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(selection, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
