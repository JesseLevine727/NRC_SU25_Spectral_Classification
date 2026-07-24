#!/usr/bin/env python3
"""Consolidate, visualize, document, and hash SERS baseline protocol v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import confusion_matrix

import run_sers_representation_baselines as baseline
import sers_baseline_common as common


OKABE_ITO = {
    "Classical": "#0072B2",
    "Siamese": "#E69F00",
    "AE arPLS": "#56B4E9",
    "DAE arPLS": "#D55E00",
    "AE minimal": "#009E73",
    "DAE minimal": "#CC79A7",
}


def write_json(path: Path, value: Any) -> None:
    baseline.write_json(path, value)


def sha256_file(path: Path) -> str:
    return common.sha256_file(path)


def model_display(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(index=frame.index, dtype=object)
    classical = (frame["model_family"] == "classical") & (
        frame["model"] == "pca_logistic"
    )
    result.loc[classical & (frame["representation"] == "arpls_minmax")] = (
        "Classical"
    )
    result.loc[frame["model_family"] == "siamese"] = "Siamese"
    result.loc[
        (frame["model_family"] == "ae")
        & (frame["representation"] == "arpls_minmax")
    ] = "AE arPLS"
    result.loc[
        (frame["model_family"] == "dae")
        & (frame["representation"] == "arpls_minmax")
    ] = "DAE arPLS"
    result.loc[
        (frame["model_family"] == "ae")
        & (frame["representation"] == "minimal_minmax")
    ] = "AE minimal"
    result.loc[
        (frame["model_family"] == "dae")
        & (frame["representation"] == "minimal_minmax")
    ] = "DAE minimal"
    return result


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.3,
            "lines.markersize": 4,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(
        path.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)


def t_interval(values: Iterable[float]) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = len(array)
    if not n:
        return np.nan, np.nan, 0
    mean = float(np.mean(array))
    if n < 2:
        return mean, np.nan, n
    half = float(t.ppf(0.975, n - 1) * np.std(array, ddof=1) / np.sqrt(n))
    return mean, half, n


def aggregate_outer_units(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.copy()
    selected["cohort"] = selected["scenario"].str.extract(r"test_(.+)$")[0]
    selected["display_model"] = model_display(selected)
    selected = selected[selected["display_model"].notna()].copy()
    return (
        selected.groupby(
            ["cohort", "display_model", "outer_fold"], as_index=False
        )["balanced_accuracy_supported"]
        .mean()
        .rename(columns={"balanced_accuracy_supported": "balanced_accuracy"})
    )


def figure_outer(output_dir: Path, metrics: pd.DataFrame) -> None:
    units = aggregate_outer_units(metrics)
    cohorts = [
        ("strict_core", "Strict core (n=598)"),
        ("quality_pass", "Quality pass (n=500)"),
        ("field_quality_stress", "Field-quality stress (n=98)"),
    ]
    order = list(OKABE_ITO)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), sharey=True)
    rng = np.random.default_rng(2026)
    for panel, (ax, (cohort, title)) in enumerate(zip(axes, cohorts)):
        part = units[units["cohort"] == cohort]
        for index, label in enumerate(order):
            values = part.loc[
                part["display_model"] == label, "balanced_accuracy"
            ].to_numpy()
            if not len(values):
                continue
            jitter = rng.normal(0, 0.035, size=len(values))
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                color=OKABE_ITO[label],
                edgecolor="black",
                linewidth=0.25,
                s=15,
                alpha=0.65,
                zorder=2,
            )
            mean, half, _ = t_interval(values)
            ax.errorbar(
                index,
                mean,
                yerr=half,
                fmt="D",
                color=OKABE_ITO[label],
                markeredgecolor="black",
                markeredgewidth=0.35,
                capsize=2.5,
                zorder=3,
            )
        ax.axhline(1 / 7, color="0.45", linestyle=":", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(range(len(order)), order, rotation=55, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Model")
        ax.text(
            -0.16,
            1.05,
            chr(ord("A") + panel),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,
        )
    axes[0].set_ylabel("Balanced accuracy")
    fig.text(
        0.5,
        -0.06,
        "Points: outer folds after averaging neural seeds; diamonds: mean ± 95% t CI (n=5)",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "outer_performance")


def figure_corruption(output_dir: Path, frame: pd.DataFrame) -> None:
    data = frame.copy()
    data["cohort"] = data["scenario"].str.extract(r"test_(.+)$")[0]
    data = data[
        (data["cohort"] == "strict_core")
        & (data["corruption"] == "composite")
    ].copy()
    data["display_model"] = model_display(data)
    data = data[
        data["display_model"].isin(
            ["AE arPLS", "DAE arPLS", "AE minimal", "DAE minimal"]
        )
    ]
    units = (
        data.groupby(
            ["display_model", "severity", "outer_fold"], as_index=False
        )
        .agg(
            balanced_accuracy=("balanced_accuracy_supported", "mean"),
            reconstruction_mse=("reconstruction_mse", "mean"),
            prediction_agreement=("prediction_agreement", "mean"),
            latent_drift=("latent_cosine_drift", "mean"),
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1))
    panels = [
        ("balanced_accuracy", "Balanced accuracy", (0, 1)),
        ("reconstruction_mse", "Recovery MSE", None),
        ("prediction_agreement", "Clean/corrupted agreement", (0, 1)),
        ("latent_drift", "Latent cosine drift", None),
    ]
    markers = {
        "AE arPLS": "o",
        "DAE arPLS": "s",
        "AE minimal": "^",
        "DAE minimal": "D",
    }
    styles = {"AE arPLS": "--", "DAE arPLS": "-", "AE minimal": "--", "DAE minimal": "-"}
    for panel, (ax, (column, ylabel, ylim)) in enumerate(
        zip(axes.ravel(), panels)
    ):
        for label in markers:
            part = units[units["display_model"] == label]
            means = []
            halves = []
            for severity in (0.5, 1.0, 1.5):
                mean, half, _ = t_interval(
                    part.loc[part["severity"] == severity, column]
                )
                means.append(mean)
                halves.append(half)
            ax.errorbar(
                [0.5, 1.0, 1.5],
                means,
                yerr=halves,
                label=label,
                color=OKABE_ITO[label],
                marker=markers[label],
                linestyle=styles[label],
                capsize=2.5,
            )
        ax.set_xlabel("Composite corruption severity (× base)")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.text(
            -0.14,
            1.04,
            chr(ord("A") + panel),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,
        )
    axes[0, 0].legend(ncol=2, loc="lower left")
    fig.suptitle(
        "Strict-core held-out composite-corruption response (mean ± 95% t CI; n=5 outer folds)",
        y=1.01,
        fontsize=9,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "corruption_robustness")


def figure_domain(output_dir: Path, frame: pd.DataFrame) -> None:
    data = frame[
        (frame["evaluation_subset"] == "strict_core")
        & (frame["domain_protocol"] == "domain_and_sample")
    ].copy()
    data["display_model"] = model_display(data)
    data.loc[data["display_model"] == "Classical", "display_model"] = (
        "PCA (arPLS)"
    )
    order = ["PCA (arPLS)", "Siamese", "AE arPLS", "DAE arPLS"]
    data = data[data["display_model"].isin(order)]
    units = (
        data.groupby(
            ["domain_type", "heldout_domain", "display_model"], as_index=False
        )["balanced_accuracy_supported"]
        .mean()
    )
    instruments = sorted(
        units.loc[
            units["domain_type"] == "instrument", "heldout_domain"
        ].unique()
    )
    sensors = sorted(
        units.loc[
            units["domain_type"] == "sensor_family", "heldout_domain"
        ].unique()
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.1, 4.5),
        gridspec_kw={"height_ratios": [2.1, 1]},
    )
    for panel, (ax, (domain_type, domains, title)) in enumerate(
        zip(
            axes,
            [
                ("instrument", instruments, "Held-out instrument"),
                ("sensor_family", sensors, "Held-out sensor family"),
            ],
        )
    ):
        matrix = np.full((len(order), len(domains)), np.nan)
        for row, model in enumerate(order):
            for column, domain in enumerate(domains):
                values = units.loc[
                    (units["domain_type"] == domain_type)
                    & (units["heldout_domain"] == domain)
                    & (units["display_model"] == model),
                    "balanced_accuracy_supported",
                ]
                if len(values):
                    matrix[row, column] = values.iloc[0]
        image = ax.imshow(matrix, cmap="cividis", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(order)), order)
        ax.set_xticks(range(len(domains)), domains, rotation=35, ha="right")
        ax.set_xlabel(title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                text = "NA" if not np.isfinite(matrix[row, column]) else f"{matrix[row, column]:.2f}"
                ax.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=(
                        "white"
                        if np.isfinite(matrix[row, column])
                        and matrix[row, column] < 0.45
                        else "black"
                    ),
                )
        ax.text(
            -0.08,
            1.08,
            chr(ord("A") + panel),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=10,
        )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    colorbar.set_label("Balanced accuracy (supported classes)")
    fig.text(
        0.5,
        0.005,
        "NA: held-out test analytes absent from the same-sample-excluded training set",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(hspace=0.55, bottom=0.24, right=0.87)
    save_figure(fig, output_dir / "figures" / "strict_domain_transfer")


def figure_poster(output_dir: Path, metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    data = metrics.copy()
    data["display_model"] = model_display(data)
    data.loc[data["model_family"] == "classical", "display_model"] = pd.NA
    data.loc[
        (data["model_family"] == "classical")
        & (data["representation"] == "derivative_1")
        & (data["model"] == "nearest_centroid"),
        "display_model",
    ] = "Derivative centroid"
    data.loc[
        data["representation"] == "derivative_1_historical", "display_model"
    ] = "Historical Siamese"
    data = data[
        data["display_model"].isin(
            [
                "Derivative centroid",
                "Historical Siamese",
                "Siamese",
                "AE arPLS",
                "DAE arPLS",
            ]
        )
    ]
    colors = {
        **OKABE_ITO,
        "Historical Siamese": "#000000",
        "Derivative centroid": "#0072B2",
    }
    order = [
        "Derivative centroid",
        "Historical Siamese",
        "Siamese",
        "AE arPLS",
        "DAE arPLS",
    ]
    family_order = ["Ag", "Au", "PICO", "pSERS"]
    units = (
        data.groupby(
            ["display_model", "heldout_substrate_family"], as_index=False
        )["balanced_accuracy_supported"]
        .mean()
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    for label in order:
        part = units[units["display_model"] == label].set_index(
            "heldout_substrate_family"
        )
        axes[0].plot(
            range(len(family_order)),
            [part.loc[f, "balanced_accuracy_supported"] for f in family_order],
            marker="o",
            label=label,
            color=colors[label],
        )
    axes[0].set_xticks(range(len(family_order)), family_order)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("Held-out substrate family")
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].legend(fontsize=6, loc="lower right")
    axes[0].text(
        -0.15,
        1.04,
        "A",
        transform=axes[0].transAxes,
        fontweight="bold",
        fontsize=10,
    )

    pred = predictions[
        (predictions["heldout_substrate_family"] == "Ag")
        & (predictions["true_label"] == "4np")
    ].copy()
    pred["display_model"] = model_display(pred)
    pred.loc[pred["model_family"] == "classical", "display_model"] = pd.NA
    pred.loc[
        (pred["model_family"] == "classical")
        & (pred["representation"] == "derivative_1")
        & (pred["model"] == "nearest_centroid"),
        "display_model",
    ] = "Derivative centroid"
    pred.loc[
        pred["representation"] == "derivative_1_historical", "display_model"
    ] = "Historical Siamese"
    recall = (
        pred[pred["display_model"].isin(order)]
        .groupby("display_model")["correct"]
        .mean()
        .reindex(order)
    )
    axes[1].bar(
        range(len(order)),
        recall,
        color=[colors[label] for label in order],
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].set_xticks(range(len(order)), order, rotation=55, ha="right")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("4NP recall")
    axes[1].set_title("Held-out Ag / AgNP subgroup")
    axes[1].text(
        -0.15,
        1.04,
        "B",
        transform=axes[1].transAxes,
        fontweight="bold",
        fontsize=10,
    )
    fig.text(
        0.5,
        -0.08,
        "Poster rows are map locations without physical-preparation IDs; results are descriptive substrate transfer.",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "poster_transfer")


def uncertainty_summary(
    outer: pd.DataFrame,
    poster: pd.DataFrame,
    domain: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outer_units = aggregate_outer_units(outer)
    for keys, group in outer_units.groupby(["cohort", "display_model"]):
        mean, half, n = t_interval(group["balanced_accuracy"])
        rows.append(
            {
                "evaluation": "outer",
                "subset": keys[0],
                "domain_type": "",
                "protocol": "grouped_outer_fold",
                "model": keys[1],
                "mean_balanced_accuracy": mean,
                "ci95_half_width": half,
                "independent_unit": "outer_fold",
                "n_units": n,
            }
        )
    poster_copy = poster.copy()
    poster_copy["display_model"] = model_display(poster_copy)
    poster_copy.loc[
        poster_copy["representation"] == "derivative_1_historical",
        "display_model",
    ] = "Historical Siamese"
    poster_units = (
        poster_copy[poster_copy["display_model"].notna()]
        .groupby(
            ["display_model", "heldout_substrate_family"], as_index=False
        )["balanced_accuracy_supported"]
        .mean()
    )
    for label, group in poster_units.groupby("display_model"):
        mean, half, n = t_interval(group["balanced_accuracy_supported"])
        rows.append(
            {
                "evaluation": "poster",
                "subset": "chemical_only_275",
                "domain_type": "substrate_family",
                "protocol": "leave_one_substrate_family_out",
                "model": label,
                "mean_balanced_accuracy": mean,
                "ci95_half_width": half,
                "independent_unit": (
                    "substrate_family_descriptive_not_independent_preparation"
                ),
                "n_units": n,
            }
        )
    domain_copy = domain.copy()
    domain_copy["display_model"] = model_display(domain_copy)
    domain_copy = domain_copy[domain_copy["display_model"].notna()]
    domain_units = (
        domain_copy.groupby(
            [
                "evaluation_subset",
                "domain_protocol",
                "domain_type",
                "display_model",
                "heldout_domain",
            ],
            as_index=False,
        )["balanced_accuracy_supported"]
        .mean()
    )
    for keys, group in domain_units.groupby(
        [
            "evaluation_subset",
            "domain_protocol",
            "domain_type",
            "display_model",
        ]
    ):
        mean, half, n = t_interval(group["balanced_accuracy_supported"])
        rows.append(
            {
                "evaluation": "domain_transfer",
                "subset": keys[0],
                "domain_type": keys[2],
                "protocol": keys[1],
                "model": keys[3],
                "mean_balanced_accuracy": mean,
                "ci95_half_width": half,
                "independent_unit": "heldout_domain",
                "n_units": n,
            }
        )
    return pd.DataFrame(rows)


def per_class_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    grouping = [
        "evaluation_phase",
        "scenario",
        "model_family",
        "model",
        "representation",
        "seed",
    ]
    for keys, group in predictions.groupby(grouping, dropna=False):
        supported = group["test_class_supported"].astype(bool)
        selected = group[supported]
        if selected.empty:
            continue
        result = common.per_class_classification(
            selected["true_label"].astype(str),
            selected["predicted_label"].astype(str),
        )
        for key, value in zip(grouping, keys):
            result[key] = value
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def confusion_catalog(predictions: pd.DataFrame) -> dict[str, Any]:
    catalog: dict[str, Any] = {}
    grouping = [
        "evaluation_phase",
        "scenario",
        "model_family",
        "model",
        "representation",
        "seed",
    ]
    for keys, group in predictions.groupby(grouping, dropna=False):
        supported = group["test_class_supported"].astype(bool)
        selected = group[supported]
        if selected.empty:
            continue
        labels = sorted(
            set(selected["true_label"].astype(str))
            | set(selected["predicted_label"].astype(str))
        )
        matrix = confusion_matrix(
            selected["true_label"].astype(str),
            selected["predicted_label"].astype(str),
            labels=labels,
        )
        identifier = "__".join(str(value) for value in keys)
        catalog[identifier] = {
            "labels": labels,
            "matrix": matrix.tolist(),
            "n_supported": int(len(selected)),
        }
    return catalog


def consolidate(output_dir: Path) -> None:
    outer = pd.read_csv(output_dir / "outer_fold_metrics.csv")
    outer_predictions = pd.read_csv(output_dir / "outer_fold_predictions.csv")
    poster = pd.read_csv(output_dir / "poster_metrics.csv")
    poster_predictions = pd.read_csv(output_dir / "poster_predictions.csv")
    domain = pd.read_csv(output_dir / "domain_transfer_metrics.csv")
    domain_predictions = pd.read_csv(
        output_dir / "domain_transfer_predictions.csv"
    )
    control_predictions = pd.read_csv(output_dir / "control_predictions.csv")

    for phase, frame in (
        ("outer", outer_predictions),
        ("poster", poster_predictions),
        ("domain_transfer", domain_predictions),
        ("control", control_predictions),
    ):
        frame.insert(0, "evaluation_phase", phase)
    all_predictions = pd.concat(
        [
            outer_predictions,
            poster_predictions,
            domain_predictions,
            control_predictions,
        ],
        ignore_index=True,
        sort=False,
    )
    all_predictions.to_csv(
        output_dir / "per_spectrum_predictions.csv", index=False
    )
    per_class_metrics(all_predictions).to_csv(
        output_dir / "per_class_metrics.csv", index=False
    )
    write_json(
        output_dir / "confusion_matrices.json",
        confusion_catalog(all_predictions),
    )

    corruptions = []
    reconstructions = []
    histories = []
    for phase, prefix in (
        ("outer", "outer_fold"),
        ("poster", "poster"),
        ("domain_transfer", "domain_transfer"),
    ):
        corruption = pd.read_csv(
            output_dir / f"{prefix}_corruption_metrics.csv"
        )
        corruption.insert(0, "evaluation_phase", phase)
        corruptions.append(corruption)
        reconstruction = pd.read_csv(
            output_dir / f"{prefix}_reconstruction_metrics.csv"
        )
        reconstruction.insert(0, "evaluation_phase", phase)
        reconstructions.append(reconstruction)
        history = pd.read_csv(
            output_dir / f"{prefix}_training_histories.csv"
        )
        history.insert(0, "evaluation_phase", phase)
        histories.append(history)
    pd.concat(corruptions, ignore_index=True, sort=False).to_csv(
        output_dir / "corruption_metrics.csv", index=False
    )
    pd.concat(reconstructions, ignore_index=True, sort=False).to_csv(
        output_dir / "reconstruction_metrics.csv", index=False
    )
    ae_history = pd.read_csv(output_dir / "ae_search_training_histories.csv")
    ae_history.insert(0, "evaluation_phase", "ae_search")
    dae_history = pd.read_csv(output_dir / "dae_search_training_histories.csv")
    dae_history.insert(0, "evaluation_phase", "dae_search")
    pd.concat(
        [ae_history, dae_history, *histories],
        ignore_index=True,
        sort=False,
    ).to_csv(output_dir / "training_histories.csv", index=False)

    ae_selection = pd.read_csv(output_dir / "ae_selection_metrics.csv")
    ae_selection.insert(0, "selection_phase", "ae")
    dae_selection = pd.read_csv(output_dir / "dae_selection_metrics.csv")
    dae_selection.insert(0, "selection_phase", "dae")
    pd.concat(
        [ae_selection, dae_selection], ignore_index=True, sort=False
    ).to_csv(output_dir / "selection_metrics.csv", index=False)

    search = pd.concat(
        [
            pd.read_csv(output_dir / "ae_search_fold_metrics.csv"),
            pd.read_csv(output_dir / "dae_search_fold_metrics.csv"),
        ],
        ignore_index=True,
        sort=False,
    )
    search.to_csv(output_dir / "search_run_registry.csv", index=False)

    uncertainty_summary(outer, poster, domain).to_csv(
        output_dir / "uncertainty_summary.csv", index=False
    )
    failures = all_predictions[
        (~all_predictions["correct"].astype(bool))
        | (~all_predictions["test_class_supported"].astype(bool))
    ].copy()
    failures["failure_type"] = np.where(
        ~failures["test_class_supported"].astype(bool),
        "unsupported_true_class",
        "misclassification",
    )
    failures.to_csv(output_dir / "failure_cases.csv", index=False)

    figure_outer(output_dir, outer)
    figure_corruption(
        output_dir, pd.read_csv(output_dir / "outer_fold_corruption_metrics.csv")
    )
    figure_domain(output_dir, domain)
    figure_poster(output_dir, poster, poster_predictions)


def split_seed_assignments(output_dir: Path, bundle: Path, poster_csv: Path) -> None:
    rows: list[pd.DataFrame] = []
    for subset, manifest_name in (
        ("strict_core", "core_preprocessing_manifest.csv"),
        ("quality_pass", "quality_preprocessing_manifest.csv"),
        ("field_quality_stress", "field_quality_stress_manifest.csv"),
    ):
        frame = pd.read_csv(bundle / manifest_name)
        selected = pd.DataFrame(
            {
                "dataset": "NATO",
                "subset": subset,
                "observation_uid": frame["observation_uid"].astype(str),
                "grouping_key": frame["master_sample_id"].astype(str),
                "grouped_sample_fold_5": frame[
                    "grouped_sample_fold_5"
                ].astype(int),
                "selection_seed": 1729,
                "final_seeds": "1729|2718|3141",
            }
        )
        rows.append(selected)
    poster = common.load_poster_dataset(poster_csv).manifest
    rows.append(
        pd.DataFrame(
            {
                "dataset": "Poster",
                "subset": "chemical_only",
                "observation_uid": poster["observation_uid"].astype(str),
                "grouping_key": poster["substrate_family"].astype(str),
                "grouped_sample_fold_5": pd.NA,
                "selection_seed": pd.NA,
                "final_seeds": "1729|2718|3141",
            }
        )
    )
    pd.concat(rows, ignore_index=True).to_csv(
        output_dir / "split_seed_assignments.csv", index=False
    )


def decision_registry(output_dir: Path) -> dict[str, Any]:
    outer = pd.read_csv(output_dir / "outer_fold_metrics.csv")
    outer["cohort"] = outer["scenario"].str.extract(r"test_(.+)$")[0]
    outer["display_model"] = model_display(outer)
    outer_summary = (
        outer[outer["display_model"].notna()]
        .groupby(["cohort", "display_model"])[
            "balanced_accuracy_supported"
        ]
        .mean()
        .unstack()
        .to_dict(orient="index")
    )
    corruption = pd.read_csv(output_dir / "outer_fold_corruption_metrics.csv")
    corruption["cohort"] = corruption["scenario"].str.extract(
        r"test_(.+)$"
    )[0]
    corruption["display_model"] = model_display(corruption)
    composite = (
        corruption[
            (corruption["cohort"] == "strict_core")
            & (corruption["corruption"] == "composite")
            & (corruption["severity"] == 1.0)
            & corruption["display_model"].isin(
                ["AE arPLS", "DAE arPLS", "AE minimal", "DAE minimal"]
            )
        ]
        .groupby("display_model")
        .agg(
            balanced_accuracy=("balanced_accuracy_supported", "mean"),
            reconstruction_mse=("reconstruction_mse", "mean"),
            prediction_agreement=("prediction_agreement", "mean"),
            latent_drift=("latent_cosine_drift", "mean"),
        )
        .to_dict(orient="index")
    )
    domain = pd.read_csv(output_dir / "domain_transfer_metrics.csv")
    domain["display_model"] = model_display(domain)
    strict_domain = (
        domain[
            (domain["evaluation_subset"] == "strict_core")
            & (domain["domain_protocol"] == "domain_and_sample")
            & (domain["domain_type"] == "instrument")
            & domain["display_model"].notna()
        ]
        .groupby("display_model")["balanced_accuracy_supported"]
        .mean()
        .to_dict()
    )
    return {
        "baseline_protocol": common.PROTOCOL_VERSION,
        "reproducibility": {
            "training_device": "CUDA with deterministic kernels",
            "canonical_evaluation_device": "CPU",
            "reason": (
                "CUDA-trained checkpoint tensors reproduce exactly, but "
                "separate-process CUDA convolution inference showed "
                "approximately 1e-7 to 1e-5 floating variation. CPU replay "
                "from verified checkpoints is the canonical bitwise-stable "
                "evaluation layer."
            ),
        },
        "classical_reference": {
            "decision": "PCA-logistic is the primary clean classification reference.",
            "outer_balanced_accuracy": outer_summary,
        },
        "siamese": {
            "decision": (
                "Retain as deterministic metric-learning control; it does not "
                "provide reconstruction or explicit denoising and is not the "
                "strongest NATO classifier."
            ),
        },
        "deterministic_reconstruction": {
            "decision": (
                "Clean AE is a useful matched compression/reconstruction "
                "diagnostic but does not outperform the classical clean "
                "reference and fails the absolute repeatable-peak gate."
            ),
        },
        "denoising": {
            "decision": (
                "DAE adds substantial synthetic composite-corruption recovery "
                "and latent stability, but does not consistently improve clean, "
                "real-stress, or unseen-domain classification."
            ),
            "strict_core_composite_severity_1": composite,
        },
        "advancing_view": {
            "primary": "arpls_minmax",
            "mandatory_sensitivity_control": "minimal_minmax",
            "reason": (
                "Pre-outer inner-only decision prioritized target separation, "
                "instrument invariance, same-master geometry, and corruption "
                "stability. Minimal remains mandatory because it reconstructs "
                "clean spectra and repeatable peaks more faithfully."
            ),
        },
        "standard_vae_starting_point": {
            "representation": "arpls_minmax",
            "channels": [8, 16],
            "bottleneck_dimension": 64,
            "reconstruction_loss": "spectral_composite",
            "clean_curriculum": "clean",
            "denoising_comparator_curriculum": "mixed_uniform",
            "unchanged_from_baseline_selection": True,
        },
        "strict_instrument_and_sample_balanced_accuracy": strict_domain,
        "claim_limits": [
            "No model solved the 98-spectrum field-quality stress cohort.",
            "No AE passed the predeclared absolute repeatable-peak gate.",
            "Poster rows lack physical-preparation IDs.",
            "Unsupported unseen analyte classes are excluded from supported-class metrics and reported separately.",
            "Synthetic denoising success is not evidence of universal real-domain denoising.",
        ],
    }


def write_documentation(
    output_dir: Path,
    decisions: dict[str, Any],
) -> None:
    registry = f"""# SERS baseline decision registry

Bundle: `{common.PROTOCOL_VERSION}`  
Selection: closed before outer, stress, poster, and domain evaluation.  
VAE models: prohibited in this bundle and not run.
Training: deterministic CUDA. Canonical final inference: deterministic CPU replay from verified checkpoints.

## Final decisions

1. **Classical reference:** PCA-logistic is the primary clean classification reference. It remains strongest on NATO strict core and quality pass.
2. **Siamese:** retain as a deterministic metric-learning control. It learns similarity structure but has no decoder and no explicit denoising objective.
3. **Clean AE:** retain as a compression/reconstruction diagnostic. It does not beat the classical clean reference and every clean AE failed the absolute repeatable-peak gate.
4. **DAE:** retain as a robustness comparator. It substantially improves held-out synthetic composite-corruption recovery, agreement, and latent stability, but does not consistently improve clean, real-stress, or unseen-domain classification.
5. **Primary VAE input:** `arpls_minmax`.
6. **Mandatory sensitivity input:** `minimal_minmax`, because it reconstructs clean spectra and repeatable peaks more faithfully.
7. **Frozen VAE starting architecture:** channels `(8, 16)`, latent dimension `64`, spectral-composite reconstruction loss; clean curriculum for standard VAE and `mixed_uniform` only for the denoising comparator.

## Critical limits

- The 98 field-quality-stress spectra remain difficult for every model family.
- Absolute peak preservation is unresolved; no AE passed the registered peak gate.
- Poster substrate holdout is descriptive map-location transfer, not independent-preparation validation.
- Domain-and-sample tests can contain unsupported analytes. They are predicted and listed, but excluded from supported-class balanced accuracy.
- Denoising gains on synthetic corruptions must not be generalized to arbitrary instrument/substrate nuisance.

## Evaluation-metric correction

The first final-evaluation pass incorrectly defined repeatable test peaks across analytes. Before interpretation, the final metric code was corrected to the frozen rule: peaks repeat within the same master sample across instruments. Checkpoints and predictions were unchanged and replayed from verified state hashes.
"""
    (output_dir / "DECISION_REGISTRY.md").write_text(registry)
    readme = f"""# SERS representation baselines v1

This immutable result bundle establishes classical, deterministic Siamese, clean autoencoder, and denoising-autoencoder baselines before any VAE experiment.

## Outcome

The baseline does **not** show that reconstruction alone improves classification. PCA-logistic remains the strongest clean NATO reference. The arPLS DAE is materially more robust to registered synthetic corruptions than the matched clean AE, but it does not consistently improve real field-quality or unseen-domain classification. Peak preservation remains below the predeclared gate.

The next standard-VAE goal should begin from `arpls_minmax`, channels `(8, 16)`, latent dimension `64`, and the spectral-composite reconstruction loss. `minimal_minmax` is a mandatory sensitivity control.

Training uses strict deterministic CUDA. Because separate-process CUDA convolution inference retained tiny floating variation near a handful of decision and peak thresholds, all canonical final metrics, predictions, embeddings, reconstructions, and corruption outputs are replayed on CPU from state-hash-verified checkpoints. The clean rebuild matches that canonical layer exactly.

## Main records

- `predeclared_protocol.json`: frozen design and gates.
- `selected_configurations.json`: inner-only closed selection.
- `DECISION_REGISTRY.md`: final decisions and claim limits.
- `outer_fold_metrics.csv`: sealed NATO outer results.
- `domain_transfer_metrics.csv`: 56 domain scenarios.
- `poster_metrics.csv`: descriptive substrate-family transfer.
- `per_spectrum_predictions.csv`: all predictions and support flags.
- `corruption_metrics.csv`: corruption-by-severity results.
- `reconstruction_metrics.csv`: per-spectrum reconstruction evidence.
- `uncertainty_summary.csv`: fold/domain uncertainty and unit definitions.
- `artifact_hashes.json`: SHA-256 catalog.
- `validation_report.json`: automated integrity audit.

## Reproduce

Run selection, final evaluation, finalization, and validation with the project `.venv` and the commands recorded in `reproduction_commands.sh`.
"""
    (output_dir / "README.md").write_text(readme)
    write_json(output_dir / "final_decisions.json", decisions)


def provenance(
    output_dir: Path,
    bundle: Path,
    poster_csv: Path,
) -> None:
    source_paths = {
        "preprocessing_v2_artifact_hashes": bundle / "artifact_hashes.json",
        "preprocessing_v2_dataset_version": bundle / "dataset_version.json",
        "poster_source_csv": poster_csv,
        "protocol_source": Path(
            "configs/sers_representation_baselines_v1.json"
        ).resolve(),
        "selection_script": Path(
            "scripts/run_sers_representation_baselines.py"
        ).resolve(),
        "shared_harness": Path("scripts/sers_baseline_common.py").resolve(),
        "final_evaluation_script": Path(
            "scripts/run_sers_baseline_final.py"
        ).resolve(),
        "finalization_script": Path(__file__).resolve(),
        "bundle_validator": Path(
            "scripts/validate_sers_baseline_bundle.py"
        ).resolve(),
        "clean_rebuild_comparator": Path(
            "scripts/compare_sers_baseline_rebuild.py"
        ).resolve(),
    }
    write_json(
        output_dir / "provenance_audit.json",
        {
            "source_hashes": {
                key: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for key, path in source_paths.items()
            },
            "poster_rows": {
                "source": 503,
                "chemical_only": 275,
                "physical_preparation_ids_available": False,
            },
            "historical_poster_result": {
                "original_random_row_train_query_protocol": (
                    "98.76% label+substrate prediction; not comparable to "
                    "grouped substrate-family transfer."
                ),
                "saved_grouped_siamese_mean_balanced_accuracy": 0.975,
                "fresh_exact_reproducibility_qualification": (
                    "legacy implementation was nondeterministic; the new "
                    "shared harness uses strict deterministic kernels."
                ),
            },
            "metric_correction": (
                "Final repeatable-test peak definition corrected before "
                "interpretation; checkpoints/predictions unchanged and replayed."
            ),
            "canonical_inference": {
                "training_device": "cuda",
                "evaluation_device": "cpu",
                "reason": (
                    "Exact CUDA checkpoint tensors with tiny separate-process "
                    "CUDA inference variation; CPU replay is bitwise stable."
                ),
            },
        },
    )
    write_json(
        output_dir / "dataset_version.json",
        {
            "dataset_version": "sers-representation-baselines-v1",
            "nato_source_version": "nato-sers-preprocessing-v2",
            "nato_roles": {
                "strict_core": 598,
                "quality_pass": 500,
                "field_quality_stress": 98,
            },
            "poster_chemical_only_rows": 275,
            "selection_closed": True,
            "vae_models_run": False,
        },
    )


def hash_artifacts(output_dir: Path) -> None:
    excluded = {
        "artifact_hashes.json",
        "validation_report.json",
        "clean_rebuild_comparison.json",
    }
    files = [
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name not in excluded
        and "search_cache" not in path.parts
        and "run_cache" not in path.parts
    ]
    catalog = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(files)
    }
    write_json(output_dir / "artifact_hashes.json", catalog)


def reproduction_commands(output_dir: Path) -> None:
    text = f"""#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
.venv/bin/python scripts/run_sers_representation_baselines.py \\
  --stage selection --device cuda --output-dir {output_dir}
.venv/bin/python scripts/run_sers_baseline_final.py \\
  --stage all --device cuda --output-dir {output_dir}
# Canonical, bitwise-stable inference replay from the CUDA-trained checkpoints.
.venv/bin/python scripts/run_sers_baseline_final.py \\
  --stage all --device cpu --output-dir {output_dir}
.venv/bin/python scripts/finalize_sers_baseline_bundle.py \\
  --output-dir {output_dir}
.venv/bin/python scripts/validate_sers_baseline_bundle.py \\
  --output-dir {output_dir}
"""
    path = output_dir / "reproduction_commands.sh"
    path.write_text(text)
    path.chmod(0o755)


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_representation_baselines"
        / "baselines_v1",
    )
    parser.add_argument(
        "--nato-bundle",
        type=Path,
        default=repository
        / "Workspace"
        / "nato_sers_field_trial"
        / "preprocessing_v2",
    )
    parser.add_argument(
        "--poster-csv",
        type=Path,
        default=repository
        / "Workspace"
        / "data"
        / "processed"
        / "consolidated_SERS.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    output_dir = args.output_dir.resolve()
    bundle = args.nato_bundle.resolve()
    poster_csv = args.poster_csv.resolve()
    warnings.filterwarnings(
        "ignore",
        message="A single label was found in 'y_true' and 'y_pred'.*",
        category=UserWarning,
        module="sklearn.metrics._classification",
    )
    apply_style()
    consolidate(output_dir)
    split_seed_assignments(output_dir, bundle, poster_csv)
    decisions = decision_registry(output_dir)
    write_documentation(output_dir, decisions)
    provenance(output_dir, bundle, poster_csv)
    reproduction_commands(output_dir)
    hash_artifacts(output_dir)
    print(
        json.dumps(
            {
                "status": "finalized",
                "output_dir": str(output_dir),
                "figures": 4,
                "vae_models_run": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
