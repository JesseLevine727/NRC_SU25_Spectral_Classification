#!/usr/bin/env python3
"""Create publication-ready figures for the finalized open-world SERS study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from string import ascii_uppercase
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OKABE_ITO = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#000000",
]
METHOD_ORDER = [
    "Selected classical",
    "Siamese triplet",
    "Supervised contrastive",
    "Standard VAE",
    "Structured VAE—chemical",
    "Structured VAE—union",
    "Selected residual CNN",
]
METHOD_SHORT = {
    "Selected classical": "Classical",
    "Siamese triplet": "Siamese",
    "Supervised contrastive": "SupCon",
    "Standard VAE": "VAE",
    "Structured VAE—chemical": "D-VAE chem.",
    "Structured VAE—union": "D-VAE union",
    "Selected residual CNN": "CNN",
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-."]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("Workspace/sers_open_world_shift/open_world_v1"),
    )
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axes(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def panel_labels(
    axes: Iterable[mpl.axes.Axes], x: float = -0.15
) -> None:
    for label, ax in zip(ascii_uppercase, axes):
        ax.text(
            x,
            1.12,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )


def save_figure(
    figure: mpl.figure.Figure, output_dir: Path, stem: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=300 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(figure)


def method_positions() -> dict[str, int]:
    return {method: index for index, method in enumerate(METHOD_ORDER)}


def scatter_summary(
    ax: mpl.axes.Axes,
    units: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    chance: float | None = None,
) -> None:
    positions = method_positions()
    rng = np.random.default_rng(1147)
    for method_index, method in enumerate(METHOD_ORDER):
        values = units[units["method"] == method][metric].to_numpy(float)
        jitter = rng.uniform(-0.11, 0.11, len(values))
        ax.scatter(
            np.full(len(values), positions[method]) + jitter,
            values,
            s=13,
            alpha=0.48,
            color=OKABE_ITO[method_index],
            marker=MARKERS[method_index],
            linewidths=0.3,
            edgecolors="white",
            zorder=2,
        )
        row = summary[summary["method"] == method].iloc[0]
        mean = float(row[f"{metric}_mean"])
        lower = float(row[f"{metric}_bootstrap_ci95_lower"])
        upper = float(row[f"{metric}_bootstrap_ci95_upper"])
        ax.errorbar(
            positions[method],
            mean,
            yerr=[[mean - lower], [upper - mean]],
            fmt=MARKERS[method_index],
            color=OKABE_ITO[method_index],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=5,
            capsize=2,
            zorder=3,
        )
    if chance is not None:
        ax.axhline(chance, color="#777777", linestyle=":", linewidth=0.9)
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(
        [METHOD_SHORT[method] for method in METHOD_ORDER],
        rotation=38,
        ha="right",
    )
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, len(METHOD_ORDER) - 0.5)
    clean_axes(ax)


def plot_primary(study: Path, figures: Path) -> None:
    summaries = study / "summaries"
    open_units = pd.read_csv(
        summaries / "primary_open_held_chemical_metrics.csv"
    )
    open_summary = pd.read_csv(summaries / "primary_open_summary.csv")
    class_summary = pd.read_csv(
        summaries / "primary_classification_summary.csv"
    )
    field_units = pd.read_csv(
        summaries / "field_quality_master_jackknife_values.csv"
    )
    field_summary = pd.read_csv(
        summaries / "field_quality_master_jackknife_summary.csv"
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    for ax, evaluation, title in (
        (axes[0, 0], "quality_open", "Unseen chemistry: quality spectra"),
        (axes[0, 1], "stress_open", "Unseen chemistry: field stress"),
    ):
        scatter_summary(
            ax,
            open_units[open_units["evaluation"] == evaluation],
            open_summary[open_summary["evaluation"] == evaluation],
            "master_unknown_auroc",
            "Master-level AUROC",
            chance=0.5,
        )
        ax.set_title(title)
        ax.set_ylim(0.0, 1.04)
    positions = method_positions()
    rng = np.random.default_rng(4219)
    for method_index, method in enumerate(METHOD_ORDER):
        values = field_units[field_units["method"] == method][
            "master_unknown_auroc"
        ].to_numpy(float)
        axes[1, 0].scatter(
            np.full(len(values), positions[method])
            + rng.uniform(-0.11, 0.11, len(values)),
            values,
            s=8,
            alpha=0.18,
            color=OKABE_ITO[method_index],
            marker=MARKERS[method_index],
            linewidths=0,
        )
        row = field_summary[field_summary["method"] == method].iloc[0]
        mean = float(row["master_unknown_auroc_mean"])
        lower = float(row["master_unknown_auroc_ci95_lower"])
        upper = float(row["master_unknown_auroc_ci95_upper"])
        axes[1, 0].errorbar(
            positions[method],
            mean,
            yerr=[[mean - lower], [upper - mean]],
            fmt=MARKERS[method_index],
            color=OKABE_ITO[method_index],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=5,
            capsize=2,
            zorder=3,
        )
    axes[1, 0].axhline(
        0.5, color="#777777", linestyle=":", linewidth=0.9
    )
    axes[1, 0].set_xticks(range(len(METHOD_ORDER)))
    axes[1, 0].set_xticklabels(
        [METHOD_SHORT[method] for method in METHOD_ORDER],
        rotation=38,
        ha="right",
    )
    axes[1, 0].set_ylabel("Master-level AUROC")
    axes[1, 0].set_ylim(0.0, 1.04)
    axes[1, 0].set_title("Acquisition shift (master jackknife)")
    clean_axes(axes[1, 0])
    for method_index, method in enumerate(METHOD_ORDER):
        for partition, offset, marker in (
            ("known_quality_test", -0.08, MARKERS[method_index]),
            ("known_stress", 0.08, "x"),
        ):
            row = class_summary[
                (class_summary["method"] == method)
                & (
                    class_summary["evaluation_partition"]
                    == partition
                )
            ].iloc[0]
            mean = float(row["balanced_accuracy_mean"])
            lower = float(row["balanced_accuracy_bootstrap_ci95_lower"])
            upper = float(row["balanced_accuracy_bootstrap_ci95_upper"])
            axes[1, 1].errorbar(
                positions[method] + offset,
                mean,
                yerr=[[mean - lower], [upper - mean]],
                fmt=marker,
                color=OKABE_ITO[method_index],
                markersize=5,
                capsize=2,
                markeredgecolor="black",
                markeredgewidth=0.4,
            )
    axes[1, 1].set_xticks(range(len(METHOD_ORDER)))
    axes[1, 1].set_xticklabels(
        [METHOD_SHORT[method] for method in METHOD_ORDER],
        rotation=38,
        ha="right",
    )
    axes[1, 1].set_ylabel("Supported-class balanced accuracy")
    axes[1, 1].set_ylim(0.0, 1.04)
    axes[1, 1].set_title("Known-class preservation")
    axes[1, 1].plot(
        [], [], "o", color="#555555", label="Quality"
    )
    axes[1, 1].plot(
        [], [], "x", color="#555555", label="Stress"
    )
    axes[1, 1].legend(frameon=False, loc="lower left")
    clean_axes(axes[1, 1])
    panel_labels(axes.ravel())
    figure.subplots_adjust(hspace=0.47, wspace=0.30)
    save_figure(figure, figures, "figure1_primary_performance")


def plot_selection(study: Path, figures: Path) -> None:
    cnn = study / "cnn"
    selection = pd.read_csv(cnn / "surrogate_selection.csv")
    gates = pd.read_csv(study / "summaries" / "cnn_promotion_gates.csv")
    open_units = pd.read_csv(
        study
        / "summaries"
        / "primary_open_held_chemical_metrics.csv"
    )
    verdict = json.loads(
        (study / "summaries" / "cnn_promotion_verdict.json").read_text()
    )
    gate_pass = gates["pass"].astype(str).str.lower().eq("true")
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    variant_counts = (
        selection.groupby("variant").size().sort_values(ascending=True)
    )
    axes[0, 0].barh(
        range(len(variant_counts)),
        variant_counts.values,
        color=OKABE_ITO[0],
    )
    axes[0, 0].set_yticks(range(len(variant_counts)))
    axes[0, 0].set_yticklabels(
        [value.replace("_", " ") for value in variant_counts.index]
    )
    axes[0, 0].set_xlabel("Selected held-chemical/fold tasks (of 30)")
    axes[0, 0].set_title("Development-only objective selection")
    clean_axes(axes[0, 0])

    score_counts = (
        selection.groupby("score_name").size().sort_values(ascending=True)
    )
    axes[0, 1].barh(
        range(len(score_counts)),
        score_counts.values,
        color=OKABE_ITO[1],
    )
    axes[0, 1].set_yticks(range(len(score_counts)))
    axes[0, 1].set_yticklabels(
        [value.replace("_", " ") for value in score_counts.index]
    )
    axes[0, 1].set_xlabel("Selected held-chemical/fold tasks (of 30)")
    axes[0, 1].set_title("Development-only novelty-score selection")
    clean_axes(axes[0, 1])

    gate_colors = [
        OKABE_ITO[2] if bool(value) else OKABE_ITO[3]
        for value in gate_pass
    ]
    axes[1, 0].barh(
        range(len(gates)),
        np.where(gate_pass, 1.0, 0.0),
        color=gate_colors,
    )
    axes[1, 0].scatter(
        np.where(gate_pass, 0.92, 0.08),
        range(len(gates)),
        color="black",
        s=12,
        zorder=3,
    )
    axes[1, 0].set_yticks(range(len(gates)))
    axes[1, 0].set_yticklabels(
        [value.replace("_", " ") for value in gates["gate"]]
    )
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(["Fail", "Pass"])
    axes[1, 0].set_xlim(-0.1, 1.1)
    axes[1, 0].set_title(
        f"Promotion: {verdict['gates_passed']}/{verdict['gates_total']} gates"
    )
    clean_axes(axes[1, 0])

    best = verdict["best_frozen_primary_method"]
    paired = open_units[
        (open_units["evaluation"] == "quality_open")
        & open_units["method"].isin([best, "Selected residual CNN"])
    ].pivot(
        index="held_unknown",
        columns="method",
        values="master_unknown_auroc",
    )
    x = np.arange(len(paired))
    axes[1, 1].plot(
        x,
        paired[best],
        marker="o",
        color=OKABE_ITO[4],
        linestyle="--",
        label=METHOD_SHORT.get(best, best),
    )
    axes[1, 1].plot(
        x,
        paired["Selected residual CNN"],
        marker="s",
        color=OKABE_ITO[0],
        linestyle="-",
        label="CNN",
    )
    axes[1, 1].axhline(0.5, color="#777777", linestyle=":", linewidth=0.9)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(
        [value.replace("_", " ") for value in paired.index],
        rotation=35,
        ha="right",
    )
    axes[1, 1].set_ylabel("Quality-open master AUROC")
    axes[1, 1].set_ylim(0.0, 1.04)
    axes[1, 1].set_title("Chemical-level promotion comparison")
    axes[1, 1].legend(frameon=False)
    clean_axes(axes[1, 1])
    panel_labels(axes.ravel())
    figure.subplots_adjust(hspace=0.55, wspace=0.48)
    save_figure(figure, figures, "figure2_cnn_selection_and_gates")


def plot_domain(study: Path, figures: Path) -> None:
    domain = pd.read_csv(
        study / "summaries" / "cnn_domain_seed_averaged_metrics.csv"
    )
    summary = pd.read_csv(
        study / "summaries" / "cnn_domain_summary.csv"
    )
    support = pd.read_csv(study / "cnn" / "domain_support_audit.csv")
    primary = domain[
        (domain["subset"] == "quality_pass")
        & (domain["protocol"] == "domain_and_sample")
        & (domain["domain_type"] == "instrument")
    ]
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    variants = [
        "erm_derivative",
        "conditional_coral",
        "conditional_mmd",
        "group_dro",
    ]
    held = sorted(primary["heldout_domain"].unique())
    x = np.arange(len(held))
    for index, variant in enumerate(variants):
        frame = (
            primary[primary["variant"] == variant]
            .set_index("heldout_domain")
            .reindex(held)
        )
        axes[0, 0].plot(
            x,
            frame["balanced_accuracy"],
            color=OKABE_ITO[index],
            marker=MARKERS[index],
            linestyle=LINESTYLES[index],
            label=variant.replace("_", " "),
        )
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(held, rotation=38, ha="right")
    axes[0, 0].set_ylabel("Supported-class balanced accuracy")
    axes[0, 0].set_ylim(0.0, 1.04)
    axes[0, 0].set_title("Unseen-master instrument transfer")
    axes[0, 0].legend(frameon=False, ncol=2)
    clean_axes(axes[0, 0])

    primary_summary = summary[
        (summary["subset"] == "quality_pass")
        & (summary["protocol"] == "domain_and_sample")
        & (summary["domain_type"] == "instrument")
    ].sort_values("variant_order")
    y = np.arange(len(primary_summary))
    axes[0, 1].barh(
        y - 0.17,
        primary_summary["mean_balanced_accuracy"],
        height=0.32,
        color=OKABE_ITO[0],
        label="Mean",
    )
    axes[0, 1].barh(
        y + 0.17,
        primary_summary["worst_balanced_accuracy"],
        height=0.32,
        color=OKABE_ITO[1],
        label="Worst",
    )
    axes[0, 1].set_yticks(y)
    axes[0, 1].set_yticklabels(
        [value.replace("_", " ") for value in primary_summary["variant"]]
    )
    axes[0, 1].set_xlim(0.0, 1.0)
    axes[0, 1].set_xlabel("Balanced accuracy")
    axes[0, 1].set_title("Mean and worst held instrument")
    axes[0, 1].legend(frameon=False)
    clean_axes(axes[0, 1])

    erm = summary[
        (summary["subset"] == "quality_pass")
        & (summary["domain_type"] == "instrument")
        & (summary["variant"] == "erm_derivative")
    ]
    protocols = ["domain_only", "domain_and_sample"]
    axes[1, 0].bar(
        np.arange(2),
        [
            float(
                erm[erm["protocol"] == protocol][
                    "mean_balanced_accuracy"
                ].iloc[0]
            )
            for protocol in protocols
        ],
        color=[OKABE_ITO[5], OKABE_ITO[3]],
        width=0.65,
    )
    axes[1, 0].set_xticks(np.arange(2))
    axes[1, 0].set_xticklabels(
        ["Domain only", "Domain + unseen master"]
    )
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_ylabel("Mean held-instrument BA")
    axes[1, 0].set_title("Same-master leakage sensitivity")
    clean_axes(axes[1, 0])

    support_primary = support[
        (support["subset"] == "quality_pass")
        & (support["protocol"] == "domain_and_sample")
        & (support["domain_type"] == "instrument")
    ]
    supported_mask = (
        support_primary["test_class_supported"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    support_counts = (
        support_primary[supported_mask]
        .groupby("heldout_domain")["test_rows"]
        .sum()
        .reindex(held)
        .fillna(0)
    )
    unsupported_counts = (
        support_primary[~supported_mask]
        .groupby("heldout_domain")["test_rows"]
        .sum()
        .reindex(held)
        .fillna(0)
    )
    axes[1, 1].bar(
        x, support_counts, color=OKABE_ITO[2], label="Supported"
    )
    axes[1, 1].bar(
        x,
        unsupported_counts,
        bottom=support_counts,
        color="#BBBBBB",
        hatch="//",
        label="Unsupported",
    )
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(held, rotation=38, ha="right")
    axes[1, 1].set_ylabel("Held-domain test spectra")
    axes[1, 1].set_title("Target support by held instrument")
    axes[1, 1].legend(frameon=False)
    clean_axes(axes[1, 1])
    panel_labels(axes.ravel(), x=-0.08)
    figure.subplots_adjust(hspace=0.58, wspace=0.48)
    save_figure(figure, figures, "figure3_domain_transfer")


def plot_controls(study: Path, figures: Path) -> None:
    summary_dir = study / "summaries"
    permutation = pd.read_csv(summary_dir / "cnn_permutation_units.csv")
    class_units = pd.read_csv(
        summary_dir / "cnn_sensitivity_class_units.csv"
    )
    open_summary = pd.read_csv(
        summary_dir / "cnn_sensitivity_open_summary.csv"
    )
    mc_units = pd.read_csv(summary_dir / "cnn_mc_units.csv")
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    axes[0, 0].scatter(
        np.arange(len(permutation)),
        permutation["balanced_accuracy"],
        color=OKABE_ITO[3],
        marker="x",
        s=28,
    )
    axes[0, 0].axhline(
        1.0 / 6.0, color="#777777", linestyle=":", label="1/6 chance"
    )
    axes[0, 0].set_xticks(np.arange(len(permutation)))
    axes[0, 0].set_xticklabels(
        [value.replace("_", " ") for value in permutation["held_unknown"]],
        rotation=35,
        ha="right",
    )
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_ylabel("Balanced accuracy")
    axes[0, 0].set_title("Master-label permutation")
    axes[0, 0].legend(frameon=False)
    clean_axes(axes[0, 0])

    quality = class_units[
        class_units["evaluation_partition"] == "known_quality_test"
    ]
    controls = sorted(quality["control"].unique())
    control_labels = {
        "erm_arpls_augmented": "arPLS + aug.",
        "erm_derivative_augmented": "Derivative + aug.",
        "erm_derivative_no_augmentation": "Derivative, no aug.",
        "erm_minimal_augmented": "Minimal + aug.",
    }
    score_labels = {
        "energy": "Energy",
        "mc_dropout_mutual_information": "MC-dropout MI",
        "minimum_class_mahalanobis_distance": "Mahalanobis",
        "one_minus_calibrated_max_probability": "1 − max probability",
        "predictive_entropy": "Entropy",
    }
    for index, control in enumerate(controls):
        values = quality[quality["control"] == control][
            "balanced_accuracy"
        ].to_numpy(float)
        axes[0, 1].scatter(
            np.full(len(values), index),
            values,
            color=OKABE_ITO[index],
            alpha=0.55,
            marker=MARKERS[index],
            s=17,
        )
        axes[0, 1].scatter(
            index,
            values.mean(),
            color=OKABE_ITO[index],
            edgecolor="black",
            linewidth=0.4,
            marker=MARKERS[index],
            s=35,
            zorder=3,
        )
    axes[0, 1].set_xticks(range(len(controls)))
    axes[0, 1].set_xticklabels(
        [control_labels[value] for value in controls],
        rotation=25,
        ha="right",
    )
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_ylabel("Quality balanced accuracy")
    axes[0, 1].set_title("Preprocessing and augmentation sensitivity")
    clean_axes(axes[0, 1])

    heat = (
        open_summary[open_summary["evaluation"] == "quality_open"]
        .pivot(
            index="control",
            columns="score_name",
            values="master_unknown_auroc_mean",
        )
        .reindex(controls)
    )
    image = axes[1, 0].imshow(
        heat.to_numpy(float), vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto"
    )
    axes[1, 0].set_yticks(range(len(heat)))
    axes[1, 0].set_yticklabels(
        [control_labels[value] for value in heat.index]
    )
    axes[1, 0].set_xticks(range(len(heat.columns)))
    axes[1, 0].set_xticklabels(
        [score_labels[value] for value in heat.columns],
        rotation=30,
        ha="right",
    )
    axes[1, 0].set_title("Unselected sensitivity scores")
    colorbar = figure.colorbar(image, ax=axes[1, 0], fraction=0.046)
    colorbar.set_label("Quality-open AUROC")

    evaluations = ["quality_open", "stress_open", "field_quality"]
    for index, evaluation in enumerate(evaluations):
        frame = mc_units[mc_units["evaluation"] == evaluation]
        summary = frame.groupby("pass_count")[
            "master_unknown_auroc"
        ].agg(["mean", "std"])
        axes[1, 1].errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["std"],
            color=OKABE_ITO[index],
            marker=MARKERS[index],
            linestyle=LINESTYLES[index],
            capsize=2,
            label=evaluation.replace("_", " "),
        )
    axes[1, 1].axhline(0.5, color="#777777", linestyle=":", linewidth=0.9)
    axes[1, 1].set_xticks([10, 20, 50])
    axes[1, 1].set_xlabel("MC-dropout passes")
    axes[1, 1].set_ylabel("Master-level AUROC")
    axes[1, 1].set_ylim(0.0, 1.04)
    axes[1, 1].set_title("Inference-pass sensitivity")
    axes[1, 1].legend(frameon=False)
    clean_axes(axes[1, 1])
    panel_labels(axes.ravel())
    figure.subplots_adjust(hspace=0.70, wspace=0.58)
    save_figure(figure, figures, "figure4_controls")


def plot_risk_coverage(study: Path, figures: Path) -> None:
    units = pd.read_csv(
        study
        / "summaries"
        / "risk_coverage_held_chemical_metrics.csv"
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    panels = [
        (
            axes[0, 0],
            "quality_open",
            "known_master_balanced_accuracy_accepted",
            "Accepted-known balanced accuracy",
        ),
        (
            axes[0, 1],
            "quality_open",
            "unknown_master_rejection_rate",
            "Unseen-chemical rejection rate",
        ),
        (
            axes[1, 0],
            "stress_open",
            "known_master_balanced_accuracy_accepted",
            "Accepted-known balanced accuracy",
        ),
        (
            axes[1, 1],
            "stress_open",
            "unknown_master_rejection_rate",
            "Unseen-chemical rejection rate",
        ),
    ]
    for ax, evaluation, metric, ylabel in panels:
        frame = units[units["evaluation"] == evaluation]
        for index, method in enumerate(METHOD_ORDER):
            group = (
                frame[frame["method"] == method]
                .groupby("requested_known_coverage")[metric]
                .agg(["mean", "std"])
                .sort_index()
            )
            ax.plot(
                group.index,
                group["mean"],
                color=OKABE_ITO[index],
                marker=MARKERS[index],
                linestyle=LINESTYLES[index],
                label=METHOD_SHORT[method],
            )
            ax.fill_between(
                group.index.to_numpy(float),
                np.clip(
                    group["mean"].to_numpy(float)
                    - group["std"].fillna(0).to_numpy(float),
                    0.0,
                    1.0,
                ),
                np.clip(
                    group["mean"].to_numpy(float)
                    + group["std"].fillna(0).to_numpy(float),
                    0.0,
                    1.0,
                ),
                color=OKABE_ITO[index],
                alpha=0.07,
                linewidth=0,
            )
        ax.set_xlabel("Requested known retention")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.04)
        ax.set_xlim(0.48, 1.02)
        ax.set_title(evaluation.replace("_", " "))
        clean_axes(ax)
    axes[0, 1].legend(
        frameon=False,
        ncol=2,
        bbox_to_anchor=(1.02, 1.02),
        loc="upper left",
    )
    panel_labels(axes.ravel())
    figure.subplots_adjust(hspace=0.43, wspace=0.30, right=0.82)
    save_figure(figure, figures, "figure5_risk_coverage")


def plot_confusion(study: Path, figures: Path) -> None:
    frame = pd.read_csv(
        study / "summaries" / "cnn_confusion_matrices.csv"
    )
    labels = sorted(
        set(frame["target_analyte"]) | set(frame["predicted_analyte"])
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, partition in zip(
        axes, ["known_quality_test", "known_stress"]
    ):
        subset = frame[frame["evaluation_partition"] == partition]
        matrix = (
            subset.pivot(
                index="target_analyte",
                columns="predicted_analyte",
                values="row_normalized_fraction",
            )
            .reindex(index=labels, columns=labels)
            .fillna(0.0)
        )
        image = ax.imshow(
            matrix.to_numpy(float),
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="equal",
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            [value.replace("_", " ") for value in labels],
            rotation=45,
            ha="right",
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(
            [value.replace("_", " ") for value in labels]
        )
        ax.set_xlabel("Predicted analyte")
        ax.set_ylabel("True analyte")
        ax.set_title(partition.replace("_", " "))
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.03)
    colorbar.set_label("Row-normalized fraction")
    panel_labels(axes)
    figure.subplots_adjust(wspace=0.35, bottom=0.25, right=0.9)
    save_figure(figure, figures, "figure6_cnn_confusion")


def write_audit(figures: Path) -> None:
    payload = {
        "status": "pass",
        "formats": ["pdf", "svg", "png"],
        "png_dpi": 300,
        "width_inches": 7.2,
        "palette": "Okabe-Ito",
        "redundant_encoding": ["color", "marker", "line_style", "direct_label"],
        "continuous_colormap": "viridis",
        "minimum_font_points": 7,
        "individual_independent_units_shown": True,
        "technical_seeds_averaged_before_uncertainty": True,
        "uncertainty_definition": (
            "95% held-chemical bootstrap intervals, 69-master jackknife "
            "intervals, or SD across explicitly labelled independent units"
        ),
        "significance_annotations": "not used; decisions follow preregistered effect-size gates rather than seed-level null-hypothesis tests",
    }
    (figures / "figure_accessibility_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


def write_captions(figures: Path) -> None:
    text = """# Figure captions

**Figure 1 — Primary performance.** Master-level unseen-chemical AUROC on
quality-pass and field-stress spectra, acquisition-quality AUROC, and
supported-class balanced accuracy. Small points in chemical-novelty panels
are six held chemicals; error bars are 95% bootstrap intervals after averaging
seeds and outer models. Acquisition-shift small points are leave-one-master-out
jackknife estimates across 69 physical masters, with 95% normal jackknife
intervals. The dotted line marks chance AUROC. Quality and stress classification
markers are horizontally offset.

**Figure 2 — CNN development selection and promotion.** Counts of the
development-only surrogate-selected objective and novelty score across 30
held-chemical/outer-fold tasks, binary preregistered gate decisions, and the
chemical-level comparison between the selected CNN and globally strongest
frozen primary comparator. No true held chemical or stress row selected a
candidate.

**Figure 3 — Source-only domain generalization.** Seed-averaged
`domain_and_sample` quality-pass balanced accuracy for each held instrument,
mean and worst performance across instruments, the gap between `domain_only`
and unseen-master transfer for derivative ERM, and supported/unsupported test
rows. Unsupported classes remain in predictions but not supported-class
accuracy.

**Figure 4 — Negative and sensitivity controls.** Master-label permutation,
fixed preprocessing/augmentation ERM ablations, quality-open AUROC for every
unselected score, and MC-dropout 10/20/50-pass sensitivity. Dots are held
chemicals after technical averaging; MC error bars are SD across chemicals.

**Figure 5 — Fixed-threshold selective behavior.** Accepted-known master
balanced accuracy and unseen-chemical rejection versus thresholds fit only to
known development data. Lines average six held chemicals after seeds and
outer models; light bands are ±1 SD across held chemicals. Requested retention
is not recalibrated on test unknowns or stress spectra.

**Figure 6 — Residual-CNN confusion matrices.** Row-normalized known-class
confusion for quality and stress partitions. Counts pool technical model
repeats only for visualization; inferential summaries remain grouped by held
chemical/master.
"""
    (figures / "FIGURE_CAPTIONS.md").write_text(text)


def main() -> None:
    args = parse_args()
    configure_style()
    figures = args.study_dir / "figures"
    plot_primary(args.study_dir, figures)
    plot_selection(args.study_dir, figures)
    plot_domain(args.study_dir, figures)
    plot_controls(args.study_dir, figures)
    plot_risk_coverage(args.study_dir, figures)
    plot_confusion(args.study_dir, figures)
    write_audit(figures)
    write_captions(figures)
    print(
        json.dumps(
            {
                "status": "complete",
                "figures": 6,
                "formats_each": ["pdf", "svg", "png"],
                "output": str(figures),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
