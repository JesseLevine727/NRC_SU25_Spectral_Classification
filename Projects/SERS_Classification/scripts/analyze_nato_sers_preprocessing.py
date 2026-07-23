#!/usr/bin/env python3
"""Generate preprocessing diagnostics for the NATO SERS field-trial manifest.

This is an audit aid, not a declaration of final preprocessing.  It preserves
three explicit candidate representations: common-grid vendor output, per-
spectrum SNV, and uniform AsLS baseline correction followed by SNV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
SENSOR_ORDER = [
    "pSERS_Metrohm_silver",
    "NRC_Canadian_SERS",
    "H_SERS_H_Kit",
    "GaN_polymer",
]
TARGET_ORDER = [
    "benzyl_fentanyl",
    "4_ANPP",
    "blank",
    "acetaminophen",
    "ethyl_paraoxon",
    "4_nitrophenol",
    "ethanol",
]


def asls_baseline(
    y: np.ndarray, lam: float = 1.0e6, asymmetry: float = 0.001, iterations: int = 12
) -> np.ndarray:
    """Asymmetric least-squares baseline on a one-wavenumber grid."""

    n = y.size
    difference = sparse.diags(
        [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
        [0, 1, 2],
        shape=(n - 2, n),
        format="csc",
    )
    penalty = lam * (difference.T @ difference)
    weights = np.ones(n)
    baseline = np.zeros(n)
    for _ in range(iterations):
        weight_matrix = sparse.spdiags(weights, 0, n, n)
        baseline = spsolve(weight_matrix + penalty, weights * y)
        weights = asymmetry * (y > baseline) + (1 - asymmetry) * (y <= baseline)
    return baseline


def snv(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-12)


def load_aligned(input_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    manifest = pd.read_csv(input_dir / "sers_core_manifest.csv")
    archive = np.load(input_dir / "sers_core_spectra_raw_common_grid.npz")
    ids = archive["observation_uid"].astype(str)
    positions = {value: index for index, value in enumerate(ids)}
    missing = [value for value in manifest["observation_uid"] if value not in positions]
    if missing:
        raise ValueError(f"Manifest/NPZ observation mismatch: {missing[:5]}")
    order = np.asarray([positions[value] for value in manifest["observation_uid"]])
    return manifest, archive["axis_cm1"], archive["intensity"][order].astype(float)


def classifier() -> Any:
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=40, whiten=True, random_state=2026),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )


def cv_summary(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    random_state: int = 2026,
) -> dict[str, Any]:
    random_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grouped_cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_state
    )
    random_scores = cross_val_score(
        classifier(), features, labels, cv=random_cv, scoring="balanced_accuracy"
    )
    grouped_scores = cross_val_score(
        classifier(),
        features,
        labels,
        groups=groups,
        cv=grouped_cv,
        scoring="balanced_accuracy",
    )
    return {
        "random_spectrum_cv_balanced_accuracy": random_scores.tolist(),
        "grouped_physical_sample_cv_balanced_accuracy": grouped_scores.tolist(),
        "random_spectrum_cv_mean": float(random_scores.mean()),
        "grouped_physical_sample_cv_mean": float(grouped_scores.mean()),
    }


def leave_one_domain_out(
    features: np.ndarray, labels: np.ndarray, domains: np.ndarray
) -> dict[str, Any]:
    predictions: dict[str, Any] = {}
    splitter = LeaveOneGroupOut()
    for train, test in splitter.split(features, labels, groups=domains):
        domain = str(domains[test][0])
        model = classifier().fit(features[train], labels[train])
        predicted = model.predict(features[test])
        test_classes = np.unique(labels[test])
        recalls = [
            float(np.mean(predicted[labels[test] == class_name] == class_name))
            for class_name in test_classes
        ]
        predictions[domain] = {
            "n_test": int(len(test)),
            "n_test_classes": int(len(test_classes)),
            "balanced_accuracy": float(np.mean(recalls)),
        }
    return predictions


def leave_one_domain_and_sample_out(
    features: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    sample_ids: np.ndarray,
) -> dict[str, Any]:
    """Hold out a domain and remove its physical specimens from training.

    Test classes absent from the remaining training set are reported and left
    out of the balanced-accuracy calculation. They are impossible zero-shot
    classification cases, not ordinary errors.
    """

    predictions: dict[str, Any] = {}
    for domain in np.unique(domains):
        test_mask = domains == domain
        test_sample_ids = np.unique(sample_ids[test_mask])
        train_mask = (~test_mask) & (~np.isin(sample_ids, test_sample_ids))
        train_classes = np.unique(labels[train_mask])
        supported_test_mask = test_mask & np.isin(labels, train_classes)
        unsupported_classes = sorted(set(labels[test_mask]) - set(train_classes))
        result: dict[str, Any] = {
            "n_train": int(train_mask.sum()),
            "n_test_all": int(test_mask.sum()),
            "n_test_supported": int(supported_test_mask.sum()),
            "n_test_classes_all": int(np.unique(labels[test_mask]).size),
            "n_test_classes_supported": int(
                np.unique(labels[supported_test_mask]).size
            ),
            "unsupported_test_classes": unsupported_classes,
        }
        if train_mask.sum() and supported_test_mask.sum():
            model = classifier().fit(features[train_mask], labels[train_mask])
            predicted = model.predict(features[supported_test_mask])
            supported_labels = labels[supported_test_mask]
            recalls = [
                float(np.mean(predicted[supported_labels == class_name] == class_name))
                for class_name in np.unique(supported_labels)
            ]
            result["supported_class_balanced_accuracy"] = float(np.mean(recalls))
        else:
            result["supported_class_balanced_accuracy"] = None
        predictions[str(domain)] = result
    return predictions


def plot_representative_backgrounds(
    output: Path,
    manifest: pd.DataFrame,
    axis: np.ndarray,
    raw: np.ndarray,
    baselines: np.ndarray,
) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(13, 16), sharex=True)
    for plot_axis, instrument in zip(axes.flat, INSTRUMENT_ORDER):
        subset = np.flatnonzero(manifest["instrument"].to_numpy() == instrument)
        proxy = manifest.iloc[subset]["baseline_energy_fraction_proxy"].to_numpy()
        representative = subset[np.argmin(np.abs(proxy - np.median(proxy)))]
        y = raw[representative]
        baseline = baselines[representative]
        scale = max(np.percentile(y, 99) - np.percentile(y, 1), 1e-12)
        y_scaled = (y - np.percentile(y, 1)) / scale
        baseline_scaled = (baseline - np.percentile(y, 1)) / scale
        plot_axis.plot(axis, y_scaled, color="#174A7E", linewidth=1, label="vendor export")
        plot_axis.plot(
            axis,
            baseline_scaled,
            color="#D1495B",
            linewidth=1.2,
            linestyle="--",
            label="AsLS baseline",
        )
        plot_axis.set_title(
            f"{instrument}  |  median background proxy={np.median(proxy):.2f}"
        )
        plot_axis.set_ylabel("scaled intensity")
        plot_axis.grid(alpha=0.2)
    axes.flat[0].legend(loc="best", frameon=False)
    for plot_axis in axes[-1]:
        plot_axis.set_xlabel("Raman shift (cm$^{-1}$)")
    fig.suptitle(
        "Representative field-trial SERS spectrum and uniformly estimated baseline",
        fontsize=15,
        y=0.998,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def annotated_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
) -> None:
    image = axis.imshow(values, cmap="Blues", aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                str(int(values[row, column])),
                ha="center",
                va="center",
                color="white" if values[row, column] > values.max() * 0.55 else "black",
                fontsize=8,
            )
    axis.set_xticks(range(len(column_labels)), column_labels, rotation=45, ha="right")
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_title(title)
    plt.colorbar(image, ax=axis, fraction=0.04, pad=0.02)


def plot_balance(output: Path, manifest: pd.DataFrame) -> None:
    sensor_table = pd.crosstab(
        manifest["target_analyte"], manifest["sensor_family"]
    ).reindex(index=TARGET_ORDER, columns=SENSOR_ORDER, fill_value=0)
    instrument_table = pd.crosstab(
        manifest["target_analyte"], manifest["instrument"]
    ).reindex(index=TARGET_ORDER, columns=INSTRUMENT_ORDER, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1, 2]})
    annotated_heatmap(
        axes[0],
        sensor_table.to_numpy(),
        TARGET_ORDER,
        SENSOR_ORDER,
        "Target analyte × SERS sensor family",
    )
    annotated_heatmap(
        axes[1],
        instrument_table.to_numpy(),
        TARGET_ORDER,
        INSTRUMENT_ORDER,
        "Target analyte × Raman instrument",
    )
    fig.suptitle("Strict-core dataset balance (zero cells reveal confounding)", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repository / "Workspace" / "nato_sers_field_trial",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    figures = input_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    manifest, axis, raw = load_aligned(input_dir)
    baselines = np.vstack([asls_baseline(row) for row in raw])
    baseline_corrected = raw - baselines
    raw_snv = snv(raw)
    corrected_snv = snv(baseline_corrected)
    asls_baseline_energy_fraction = np.linalg.norm(baselines, axis=1) / np.maximum(
        np.linalg.norm(raw, axis=1), 1e-12
    )
    asls_baseline_span_fraction = np.ptp(baselines, axis=1) / np.maximum(
        np.ptp(raw, axis=1), 1e-12
    )
    np.savez_compressed(
        input_dir / "sers_core_spectra_preprocessing_candidates.npz",
        axis_cm1=axis.astype(np.float32),
        observation_uid=manifest["observation_uid"].astype(str).to_numpy(dtype=str),
        vendor_raw=raw.astype(np.float32),
        raw_snv=raw_snv.astype(np.float32),
        asls_baseline=baselines.astype(np.float32),
        asls_corrected=baseline_corrected.astype(np.float32),
        asls_corrected_snv=corrected_snv.astype(np.float32),
    )

    plot_representative_backgrounds(
        figures / "instrument_background_examples.png",
        manifest,
        axis,
        raw,
        baselines,
    )
    plot_balance(figures / "dataset_balance_heatmaps.png", manifest)

    labels = manifest["target_analyte"].to_numpy()
    groups = manifest["master_sample_id"].to_numpy()
    diagnostics: dict[str, Any] = {
        "warning": (
            "These are diagnostic linear baselines. AsLS lambda=1e6 and p=0.001 "
            "are candidates to validate, not frozen production settings."
        ),
        "n_core": int(len(manifest)),
        "target_classification": {
            "raw_snv": cv_summary(raw_snv, labels, groups),
            "asls_corrected_snv": cv_summary(corrected_snv, labels, groups),
        },
        "instrument_discriminability": {
            "interpretation": (
                "High grouped accuracy means instrument identity remains recoverable "
                "for unseen physical samples; it is not an analyte score."
            ),
            "raw_snv": cv_summary(
                raw_snv, manifest["instrument"].to_numpy(), groups
            ),
            "asls_corrected_snv": cv_summary(
                corrected_snv, manifest["instrument"].to_numpy(), groups
            ),
        },
        "median_background_proxy_by_instrument": {
            key: float(value)
            for key, value in manifest.groupby("instrument")[
                "baseline_energy_fraction_proxy"
            ].median().items()
        },
        "baseline_proxy_interpretation": (
            "The manifest proxy is a rolling lower-envelope estimate on each "
            "vendor's native axis. The AsLS metrics use the common 400--1800 cm^-1 "
            "grid. Their disagreement, especially for broad RMX structure, is "
            "evidence that no baseline estimate should be treated as ground truth."
        ),
        "median_manifest_lower_envelope_span_fraction_by_instrument": {
            key: float(value)
            for key, value in manifest.groupby("instrument")[
                "baseline_span_fraction_proxy"
            ].median().items()
        },
        "median_asls_baseline_energy_fraction_by_instrument": {
            instrument: float(np.median(asls_baseline_energy_fraction[index]))
            for instrument in INSTRUMENT_ORDER
            if (index := manifest["instrument"].to_numpy() == instrument).any()
        },
        "median_asls_baseline_span_fraction_by_instrument": {
            instrument: float(np.median(asls_baseline_span_fraction[index]))
            for instrument in INSTRUMENT_ORDER
            if (index := manifest["instrument"].to_numpy() == instrument).any()
        },
        "mira_system_suitability_counts": {
            f"{instrument} | {status}": int(count)
            for (instrument, status), count in manifest[
                manifest["instrument"].str.startswith("Mira")
            ]
            .groupby(["instrument", "system_suitability"])
            .size()
            .items()
        },
        "domain_transfer_protocol_notes": {
            "domain_only": (
                "Train on all other domains; the same master sample may occur in "
                "both train and test. This isolates domain shift but is not a "
                "deployment-style unseen-specimen result."
            ),
            "domain_and_sample": (
                "Hold out the domain and remove every test master sample from "
                "training. Scores use only target classes still represented in "
                "training; unsupported classes are listed explicitly."
            ),
        },
        "leave_one_instrument_out_domain_only": {
            "raw_snv": leave_one_domain_out(
                raw_snv, labels, manifest["instrument"].to_numpy()
            ),
            "asls_corrected_snv": leave_one_domain_out(
                corrected_snv, labels, manifest["instrument"].to_numpy()
            ),
        },
        "leave_one_instrument_out_domain_and_sample": {
            "raw_snv": leave_one_domain_and_sample_out(
                raw_snv,
                labels,
                manifest["instrument"].to_numpy(),
                groups,
            ),
            "asls_corrected_snv": leave_one_domain_and_sample_out(
                corrected_snv,
                labels,
                manifest["instrument"].to_numpy(),
                groups,
            ),
        },
        "leave_one_sensor_family_out_domain_only": {
            "raw_snv": leave_one_domain_out(
                raw_snv, labels, manifest["sensor_family"].to_numpy()
            ),
            "asls_corrected_snv": leave_one_domain_out(
                corrected_snv, labels, manifest["sensor_family"].to_numpy()
            ),
        },
        "leave_one_sensor_family_out_domain_and_sample": {
            "raw_snv": leave_one_domain_and_sample_out(
                raw_snv,
                labels,
                manifest["sensor_family"].to_numpy(),
                groups,
            ),
            "asls_corrected_snv": leave_one_domain_and_sample_out(
                corrected_snv,
                labels,
                manifest["sensor_family"].to_numpy(),
                groups,
            ),
        },
    }
    (input_dir / "preprocessing_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
