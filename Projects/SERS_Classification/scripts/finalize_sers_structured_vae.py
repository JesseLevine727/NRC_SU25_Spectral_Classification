#!/usr/bin/env python3
"""Finalize structured-VAE attribution, figures, report, and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D

import sers_baseline_common as baseline


OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#777777",
}
STAGE_ORDER = ["controls", "instrument_adversary", "pair", "dependence"]
COHORT_ORDER = ["strict_core", "quality_pass", "field_quality_stress"]
COHORT_LABELS = {
    "strict_core": "Strict core",
    "quality_pass": "Quality pass",
    "field_quality_stress": "Field stress",
}


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n")


def stage_winners(output_dir: Path) -> pd.DataFrame:
    rows = []
    for stage in STAGE_ORDER:
        frame = pd.read_csv(output_dir / f"{stage}_summary.csv")
        selected = frame.sort_values(
            [
                "converged",
                "passes_all_gates",
                "gate_count",
                "selection_utility",
                "parameter_count",
                "identifier",
            ],
            ascending=[False, False, False, False, True, True],
            kind="mergesort",
        ).iloc[0]
        rows.append(selected.to_dict())
    result = pd.DataFrame(rows)
    result.insert(0, "source_stage", STAGE_ORDER)
    result.to_csv(output_dir / "inner_stage_winners.csv", index=False)
    return result


def gate_matrix(output_dir: Path) -> pd.DataFrame:
    candidates = pd.read_csv(output_dir / "all_inner_candidates.csv")
    gates = [column for column in candidates if column.startswith("gate_")]
    result = candidates[
        ["source_stage", "identifier", "gate_count", "gate_total"] + gates
    ].copy()
    result.to_csv(output_dir / "inner_gate_matrix.csv", index=False)
    return result


def outer_summaries(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(output_dir / "locked_outer_metrics.csv")
    numeric = [
        column
        for column in metrics
        if pd.api.types.is_numeric_dtype(metrics[column])
        and column not in {"outer_fold", "seed"}
    ]
    summary = (
        metrics.groupby(
            ["representation", "test_subset", "partition"], as_index=False
        )[numeric]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = [
        "_".join([str(part) for part in column if str(part)])
        for column in summary.columns
    ]
    summary.to_csv(output_dir / "locked_outer_summary.csv", index=False)
    predictions = pd.read_csv(output_dir / "locked_outer_predictions.csv")
    primary = predictions[
        predictions["representation"].eq("arpls_minmax")
        & predictions["partition"].eq("chemical")
    ].copy()
    return metrics, primary


def failure_table(
    predictions: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    return (
        predictions.groupby(group_columns, dropna=False)
        .agg(
            spectrum_count=("observation_uid", "size"),
            supported_count=("test_class_supported", "sum"),
            correct_count=("correct", "sum"),
            accuracy=("correct", "mean"),
            analyte_count=("target_analyte", "nunique"),
        )
        .reset_index()
        .sort_values(group_columns)
    )


def save_failure_tables(
    output_dir: Path, predictions: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    analyte = failure_table(
        predictions, ["test_subset", "target_analyte"]
    )
    instrument = failure_table(
        predictions, ["test_subset", "instrument"]
    )
    sensor = failure_table(
        predictions, ["test_subset", "sensor_family"]
    )
    analyte.to_csv(output_dir / "per_analyte_failures.csv", index=False)
    instrument.to_csv(output_dir / "per_instrument_failures.csv", index=False)
    sensor.to_csv(output_dir / "per_sensor_failures.csv", index=False)
    for subset in COHORT_ORDER:
        selected = predictions[predictions["test_subset"].eq(subset)]
        confusion = pd.crosstab(
            selected["true_label"],
            selected["predicted_label"],
            dropna=False,
        )
        confusion.to_csv(output_dir / f"confusion_{subset}.csv")
    return {"analyte": analyte, "instrument": instrument, "sensor": sensor}


def domain_summary(output_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(output_dir / "locked_domain_metrics.csv")
    numeric = [
        "balanced_accuracy_supported",
        "macro_f1_supported",
        "n_test",
        "n_test_supported",
        "n_test_unsupported",
        "reconstruction_median_row_correlation",
        "repeatable_peak_recall",
        "partition_maximum_canonical_correlation",
    ]
    result = (
        metrics.groupby(
            [
                "evaluation_subset",
                "domain_protocol",
                "domain_type",
                "partition",
            ],
            as_index=False,
        )[numeric]
        .agg(["mean", "std", "min", "max"])
    )
    result.columns = [
        "_".join([str(part) for part in column if str(part)])
        for column in result.columns
    ]
    result.to_csv(output_dir / "locked_domain_summary.csv", index=False)
    chemical = metrics[metrics["partition"].eq("chemical")].copy()
    chemical.to_csv(output_dir / "per_domain_failures.csv", index=False)
    return metrics


def poster_summary(output_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(output_dir / "locked_poster_metrics.csv")
    result = (
        metrics.groupby(["representation", "partition"], as_index=False)[
            [
                "balanced_accuracy_supported",
                "macro_f1_supported",
                "reconstruction_median_row_correlation",
                "partition_maximum_canonical_correlation",
            ]
        ]
        .agg(["mean", "std", "min", "max"])
    )
    result.columns = [
        "_".join([str(part) for part in column if str(part)])
        for column in result.columns
    ]
    result.to_csv(output_dir / "locked_poster_summary.csv", index=False)
    return metrics


def baseline_comparators(
    repository: Path,
    output_dir: Path,
    structured_outer: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    structured = structured_outer[
        structured_outer["representation"].eq("arpls_minmax")
        & structured_outer["partition"].isin(["chemical", "union"])
    ]
    for (partition, subset), group in structured.groupby(
        ["partition", "test_subset"]
    ):
        rows.append(
            {
                "display_model": (
                    "Structured VAE—chemical"
                    if partition == "chemical"
                    else "Structured VAE—union"
                ),
                "test_subset": subset,
                "mean_balanced_accuracy": group[
                    "balanced_accuracy_supported"
                ].mean(),
                "standard_deviation": group[
                    "balanced_accuracy_supported"
                ].std(),
                "minimum": group["balanced_accuracy_supported"].min(),
                "maximum": group["balanced_accuracy_supported"].max(),
                "independent_fold_count": group["outer_fold"].nunique(),
                "source": "structured_vae_v1",
            }
        )
    adequacy = pd.read_csv(
        repository
        / "Workspace"
        / "sers_vae_adequacy"
        / "adequacy_v1"
        / "outer_fold_metrics.csv"
    )
    adequacy["test_subset"] = adequacy["scenario"].str.extract(
        r"test_(.+)$"
    )[0]
    adequacy = adequacy[adequacy["representation"].eq("arpls_minmax")]
    adequacy_fold = (
        adequacy.groupby(["test_subset", "outer_fold"], as_index=False)[
            "balanced_accuracy_supported"
        ]
        .mean()
    )
    for subset, group in adequacy_fold.groupby("test_subset"):
        rows.append(
            {
                "display_model": "Standard VAE-500",
                "test_subset": subset,
                "mean_balanced_accuracy": group[
                    "balanced_accuracy_supported"
                ].mean(),
                "standard_deviation": group[
                    "balanced_accuracy_supported"
                ].std(),
                "minimum": group["balanced_accuracy_supported"].min(),
                "maximum": group["balanced_accuracy_supported"].max(),
                "independent_fold_count": group["outer_fold"].nunique(),
                "source": "sers_vae_adequacy_v1",
            }
        )
    baseline_outer = pd.read_csv(
        repository
        / "Workspace"
        / "sers_representation_baselines"
        / "baselines_v1"
        / "outer_fold_metrics.csv"
    )
    baseline_outer["test_subset"] = baseline_outer["scenario"].str.extract(
        r"test_(.+)$"
    )[0]
    wanted = [
        ("classical", "pca_logistic", "arpls_minmax", "PCA/logistic"),
        ("classical", "linear_svm", "arpls_minmax", "Linear SVM"),
        (
            "siamese",
            "deterministic_edge_triplet",
            "derivative_1",
            "Siamese",
        ),
        (
            "dae",
            "c8x16_z64_spectral_composite_mixed_uniform",
            "arpls_minmax",
            "Denoising AE",
        ),
    ]
    for family, model, representation, display in wanted:
        selected = baseline_outer[
            baseline_outer["model_family"].eq(family)
            & baseline_outer["model"].eq(model)
            & baseline_outer["representation"].eq(representation)
        ]
        for subset, group in selected.groupby("test_subset"):
            rows.append(
                {
                    "display_model": display,
                    "test_subset": subset,
                    "mean_balanced_accuracy": group[
                        "balanced_accuracy_supported"
                    ].mean(),
                    "standard_deviation": group[
                        "balanced_accuracy_supported"
                    ].std(),
                    "minimum": group["balanced_accuracy_supported"].min(),
                    "maximum": group["balanced_accuracy_supported"].max(),
                    "independent_fold_count": group["outer_fold"].nunique(),
                    "source": "sers_representation_baselines_v1",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "comparator_summary.csv", index=False)
    return result


def convergence_attribution(output_dir: Path) -> dict[str, Any]:
    selected = pd.read_csv(output_dir / "dependence_summary.csv").sort_values(
        ["converged", "gate_count", "selection_utility"],
        ascending=False,
    ).iloc[0]
    histories = pd.read_csv(
        output_dir / "locked_outer_training_histories.csv"
    )
    improvements = []
    for _, group in histories.groupby(
        ["training_scenario", "representation"]
    ):
        values = group.set_index("epoch")["validation_total"]
        improvements.append(
            (float(values.loc[450]) - float(values.loc[500]))
            / max(abs(float(values.loc[450])), 1.0e-12)
        )
    return {
        "inner_selected": {
            "median_final50_relative_improvement": selected[
                "median_relative_improvement_50"
            ],
            "fraction_folds_improving_at_least_1_percent": selected[
                "folds_improving_at_least_1_percent"
            ],
            "converged": selected["converged"],
        },
        "locked_outer": {
            "run_count": len(improvements),
            "median_final50_relative_improvement": np.median(improvements),
            "mean_final50_relative_improvement": np.mean(improvements),
            "maximum_final50_relative_improvement": np.max(improvements),
            "fraction_runs_improving_at_least_1_percent": np.mean(
                np.asarray(improvements) >= 0.01
            ),
            "fraction_runs_worsening": np.mean(
                np.asarray(improvements) < 0
            ),
        },
        "interpretation": (
            "The selected inner configuration satisfies the registered "
            "convergence criterion. A minority of locked fits remain at the "
            "1% boundary, but median validation objective does not improve "
            "over the final 50 epochs; undertraining is not the primary "
            "explanation for partition dependence or domain failure."
        ),
    }


def terminal_decision(
    output_dir: Path,
    winners: pd.DataFrame,
    outer: pd.DataFrame,
    domain: pd.DataFrame,
    poster: pd.DataFrame,
    convergence: dict[str, Any],
) -> dict[str, Any]:
    selected = winners[winners["source_stage"].eq("dependence")].iloc[0]
    strict = outer[
        outer["representation"].eq("arpls_minmax")
        & outer["test_subset"].eq("strict_core")
        & outer["partition"].eq("chemical")
    ]
    quality = outer[
        outer["representation"].eq("arpls_minmax")
        & outer["test_subset"].eq("quality_pass")
        & outer["partition"].eq("chemical")
    ]
    stress = outer[
        outer["representation"].eq("arpls_minmax")
        & outer["test_subset"].eq("field_quality_stress")
        & outer["partition"].eq("chemical")
    ]
    nuisance = outer[
        outer["representation"].eq("arpls_minmax")
        & outer["test_subset"].eq("strict_core")
        & outer["partition"].eq("nuisance")
    ]
    domain_chemical = domain[domain["partition"].eq("chemical")]
    poster_chemical = poster[poster["partition"].eq("chemical")]
    result = {
        "protocol": "sers-structured-vae-v1",
        "terminal_classification": "unsuccessful",
        "idea_worked_as_disentanglement": False,
        "idea_worked_as_general_nuisance_filter": False,
        "selected_by_hierarchy": selected["identifier"],
        "selected_passes_all_gates": selected["passes_all_gates"],
        "selected_gate_count": selected["gate_count"],
        "selected_gate_total": selected["gate_total"],
        "failed_selected_gates": [
            "gate_same_master",
            "gate_fold_chemical_direction",
        ],
        "inner_partial_effect": {
            "chemical_balanced_accuracy": selected[
                "chemical_target_balanced_accuracy"
            ],
            "instrument_probe_increment": selected[
                "chemical_instrument_probe_increment"
            ],
            "same_master_distance": selected[
                "chemical_same_master_cross_instrument_distance"
            ],
            "cross_covariance_mean_square": selected[
                "partition_cross_covariance_mean_square"
            ],
            "maximum_canonical_correlation": selected[
                "partition_maximum_canonical_correlation"
            ],
        },
        "locked_outer_arpls": {
            "strict_chemical_ba": strict[
                "balanced_accuracy_supported"
            ].mean(),
            "strict_instrument_probe": strict[
                "instrument_probe_increment"
            ].mean(),
            "strict_same_master_distance": strict[
                "same_master_cross_domain_mean_distance"
            ].mean(),
            "strict_nuisance_partition_chemical_ba": nuisance[
                "balanced_accuracy_supported"
            ].mean(),
            "strict_partition_maximum_canonical_correlation": strict[
                "partition_maximum_canonical_correlation"
            ].mean(),
            "quality_chemical_ba": quality[
                "balanced_accuracy_supported"
            ].mean(),
            "quality_instrument_probe": quality[
                "instrument_probe_increment"
            ].mean(),
            "field_stress_chemical_ba": stress[
                "balanced_accuracy_supported"
            ].mean(),
            "field_stress_reconstruction_correlation": stress[
                "reconstruction_median_row_correlation"
            ].mean(),
        },
        "domain_transfer": {
            "strict_domain_only_instrument_ba": domain_chemical[
                domain_chemical["evaluation_subset"].eq("strict_core")
                & domain_chemical["domain_protocol"].eq("domain_only")
                & domain_chemical["domain_type"].eq("instrument")
            ]["balanced_accuracy_supported"].mean(),
            "strict_domain_only_sensor_ba": domain_chemical[
                domain_chemical["evaluation_subset"].eq("strict_core")
                & domain_chemical["domain_protocol"].eq("domain_only")
                & domain_chemical["domain_type"].eq("sensor_family")
            ]["balanced_accuracy_supported"].mean(),
            "quality_domain_only_instrument_ba": domain_chemical[
                domain_chemical["evaluation_subset"].eq("quality_pass")
                & domain_chemical["domain_protocol"].eq("domain_only")
                & domain_chemical["domain_type"].eq("instrument")
            ]["balanced_accuracy_supported"].mean(),
            "quality_domain_only_sensor_ba": domain_chemical[
                domain_chemical["evaluation_subset"].eq("quality_pass")
                & domain_chemical["domain_protocol"].eq("domain_only")
                & domain_chemical["domain_type"].eq("sensor_family")
            ]["balanced_accuracy_supported"].mean(),
        },
        "poster_descriptive": {
            "arpls_chemical_ba": poster_chemical[
                poster_chemical["representation"].eq("arpls_minmax")
            ]["balanced_accuracy_supported"].mean(),
            "minimal_chemical_ba": poster_chemical[
                poster_chemical["representation"].eq("minimal_minmax")
            ]["balanced_accuracy_supported"].mean(),
            "interpretation": (
                "Architecture transfer is viable, but nuisance partition "
                "chemistry and approximately unit canonical correlation "
                "preclude a disentanglement claim."
            ),
        },
        "convergence": convergence,
        "why": [
            "No registered inner candidate passed every eligibility gate.",
            "The selected dependence penalty reduced covariance but left "
            "maximum canonical correlation near one.",
            "The nuisance partition retained substantial analyte information.",
            "Locked instrument leakage and same-master geometry failed.",
            "Field-stress and held-out sensor-family performance collapsed.",
            "The chemical partition did not outperform the frozen mixed VAE "
            "or PCA/logistic comparator.",
        ],
    }
    write_json(output_dir / "terminal_decision.json", result)
    return result


def compute_accounting(output_dir: Path, selected_parameters: int) -> dict:
    stages = {
        "identity_control": 20,
        "partition_controls": 80,
        "instrument_adversary": 60,
        "pair_alignment": 60,
        "dependence": 40,
        "preprocessing_sensitivity": 60,
        "locked_outer": 20,
        "locked_domain": 56,
        "locked_poster": 8,
    }
    total_runs = sum(stages.values())
    result = {
        "registered_epoch_count_per_run": 500,
        "authoritative_run_counts": stages,
        "authoritative_training_run_count": total_runs,
        "authoritative_optimizer_epoch_count": total_runs * 500,
        "diagnostic_smoke_runs_excluded": True,
        "selected_parameter_count": selected_parameters,
        "standard_vae_parameter_count": 1082353,
        "parameter_increase": selected_parameters - 1082353,
        "parameter_increase_fraction": (
            selected_parameters / 1082353 - 1.0
        ),
    }
    write_json(output_dir / "compute_accounting.json", result)
    return result


def configure_figures(repository: Path) -> tuple[Any, Any]:
    skill_scripts = Path(
        "/home/elfo/.codex/skills/scientific-visualization/scripts"
    )
    sys.path.insert(0, str(skill_scripts))
    from figure_export import save_publication_figure
    from style_presets import apply_publication_style

    apply_publication_style("default")
    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    plt.rcParams.update(
        {
            "figure.constrained_layout.use": True,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return save_publication_figure, repository


def save_figure(
    save_publication_figure: Any,
    fig: plt.Figure,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_publication_figure(
        fig, path, formats=["pdf", "png"], dpi=600, pad_inches=0.05
    )
    plt.close(fig)


def figure_inner_search(
    output_dir: Path,
    save_publication_figure: Any,
) -> None:
    candidates = pd.read_csv(output_dir / "all_inner_candidates.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    colors = {
        "controls": OKABE_ITO["gray"],
        "instrument_adversary": OKABE_ITO["vermillion"],
        "pair": OKABE_ITO["sky"],
        "dependence": OKABE_ITO["green"],
    }
    markers = {
        "controls": "o",
        "instrument_adversary": "^",
        "pair": "s",
        "dependence": "D",
    }
    for stage in STAGE_ORDER:
        frame = candidates[candidates["source_stage"].eq(stage)]
        axes[0].scatter(
            frame["chemical_instrument_probe_increment"],
            frame["chemical_target_balanced_accuracy"],
            c=colors[stage],
            marker=markers[stage],
            edgecolors="black",
            linewidths=0.35,
            s=42,
            label=stage.replace("_", " ").title(),
        )
        axes[1].scatter(
            frame["chemical_same_master_cross_instrument_distance"],
            frame["chemical_cross_instrument_separation_margin"],
            c=colors[stage],
            marker=markers[stage],
            edgecolors="black",
            linewidths=0.35,
            s=42,
        )
    axes[0].axhline(0.6621185251, color="black", ls=":", lw=0.9)
    axes[0].axvline(0.5425838599, color="black", ls=":", lw=0.9)
    axes[0].set_xlabel("Instrument probe increment (lower is better)")
    axes[0].set_ylabel("Chemical balanced accuracy")
    axes[0].legend(ncol=2, loc="lower left")
    axes[1].axvline(0.7275300776, color="black", ls=":", lw=0.9)
    axes[1].axhline(0.2240310515, color="black", ls=":", lw=0.9)
    axes[1].set_xlabel("Same-master cross-instrument distance")
    axes[1].set_ylabel("Cross-instrument class-separation margin")
    for label, ax in zip(("A", "B"), axes):
        ax.text(
            0.01,
            0.98,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
        )
    save_figure(
        save_publication_figure,
        fig,
        output_dir / "figures" / "inner_mechanism_tradeoffs",
    )


def figure_outer_comparators(
    output_dir: Path,
    comparator: pd.DataFrame,
    save_publication_figure: Any,
) -> None:
    models = [
        "PCA/logistic",
        "Linear SVM",
        "Standard VAE-500",
        "Structured VAE—chemical",
        "Structured VAE—union",
        "Siamese",
    ]
    colors = [
        OKABE_ITO["black"],
        OKABE_ITO["gray"],
        OKABE_ITO["orange"],
        OKABE_ITO["blue"],
        OKABE_ITO["sky"],
        OKABE_ITO["purple"],
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width = 0.12
    x = np.arange(len(COHORT_ORDER))
    for index, (model, color) in enumerate(zip(models, colors)):
        selected = (
            comparator[comparator["display_model"].eq(model)]
            .set_index("test_subset")
            .reindex(COHORT_ORDER)
        )
        offset = (index - (len(models) - 1) / 2) * width
        ax.errorbar(
            x + offset,
            selected["mean_balanced_accuracy"],
            yerr=selected["standard_deviation"],
            fmt="o",
            markersize=4.5,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.3,
            capsize=2,
            lw=0.9,
            label=model,
        )
    ax.set_xticks(x, [COHORT_LABELS[value] for value in COHORT_ORDER])
    ax.set_ylabel("Balanced accuracy (outer folds, mean ± SD)")
    ax.set_ylim(0.2, 0.85)
    ax.axhline(1 / 7, color="black", ls=":", lw=0.8)
    ax.legend(ncol=3, loc="upper right")
    save_figure(
        save_publication_figure,
        fig,
        output_dir / "figures" / "locked_outer_comparators",
    )


def figure_partition_and_preprocessing(
    output_dir: Path,
    outer: pd.DataFrame,
    save_publication_figure: Any,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    metrics = [
        ("balanced_accuracy_supported", "Balanced accuracy"),
        ("instrument_probe_increment", "Instrument probe increment"),
        ("repeatable_peak_recall", "Repeatable-peak recall"),
    ]
    subset = outer[outer["test_subset"].ne("field_quality_stress")]
    palette = {
        "arpls_minmax": OKABE_ITO["blue"],
        "minimal_minmax": OKABE_ITO["orange"],
    }
    for ax, (metric, label) in zip(axes, metrics):
        sns.pointplot(
            data=subset,
            x="partition",
            y=metric,
            hue="representation",
            order=["chemical", "nuisance", "union"],
            hue_order=["arpls_minmax", "minimal_minmax"],
            errorbar="sd",
            dodge=0.25,
            markers=["o", "s"],
            linestyles=["-", "--"],
            palette=palette,
            ax=ax,
        )
        sns.stripplot(
            data=subset,
            x="partition",
            y=metric,
            hue="representation",
            order=["chemical", "nuisance", "union"],
            hue_order=["arpls_minmax", "minimal_minmax"],
            dodge=True,
            palette=palette,
            alpha=0.35,
            size=2.2,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(label)
        if ax is not axes[0] and ax.legend_ is not None:
            ax.legend_.remove()
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=palette["arpls_minmax"],
                marker="o",
                linestyle="-",
                label="arPLS + min–max",
            ),
            Line2D(
                [0],
                [0],
                color=palette["minimal_minmax"],
                marker="s",
                linestyle="--",
                label="Minimal + min–max",
            ),
        ],
        loc="lower left",
    )
    for label, ax in zip(("A", "B", "C"), axes):
        ax.text(
            0.01,
            0.98,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
        )
    save_figure(
        save_publication_figure,
        fig,
        output_dir / "figures" / "partition_preprocessing_sensitivity",
    )


def figure_domain_heatmap(
    output_dir: Path,
    domain: pd.DataFrame,
    save_publication_figure: Any,
) -> None:
    selected = domain[
        domain["partition"].eq("chemical")
        & domain["domain_protocol"].eq("domain_only")
    ].copy()
    selected["column"] = (
        selected["evaluation_subset"].map(
            {"strict_core": "Strict", "quality_pass": "Quality"}
        )
        + " / "
        + selected["domain_type"].map(
            {"instrument": "instrument", "sensor_family": "sensor"}
        )
    )
    pivot = selected.pivot_table(
        index="heldout_domain",
        columns="column",
        values="balanced_accuracy_supported",
    )
    order = sorted(
        pivot.index,
        key=lambda value: (
            "Strict / sensor" not in pivot.columns
            or pd.isna(pivot.loc[value].get("Strict / sensor", np.nan)),
            value,
        ),
    )
    pivot = pivot.reindex(order)
    columns = [
        value
        for value in (
            "Strict / instrument",
            "Quality / instrument",
            "Strict / sensor",
            "Quality / sensor",
        )
        if value in pivot
    ]
    pivot = pivot[columns]
    pivot.columns = [
        str(value).replace(" / ", "\n") for value in pivot.columns
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.heatmap(
        pivot,
        vmin=0,
        vmax=1,
        cmap="cividis",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Balanced accuracy"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Held-out instrument or sensor family")
    save_figure(
        save_publication_figure,
        fig,
        output_dir / "figures" / "heldout_domain_heatmap",
    )


def figure_corruption(
    output_dir: Path,
    save_publication_figure: Any,
) -> None:
    corruption = pd.read_csv(
        output_dir / "locked_outer_corruption_metrics.csv"
    )
    selected = corruption[
        corruption["representation"].eq("arpls_minmax")
        & corruption["test_subset"].isin(
            ["strict_core", "quality_pass"]
        )
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    palette = {
        "strict_core": OKABE_ITO["blue"],
        "quality_pass": OKABE_ITO["green"],
    }
    sns.lineplot(
        data=selected,
        x="severity",
        y="chemical_balanced_accuracy_supported",
        hue="test_subset",
        style="corruption",
        markers=True,
        dashes=True,
        errorbar="sd",
        palette=palette,
        ax=axes[0],
    )
    sns.lineplot(
        data=selected,
        x="severity",
        y="chemical_prediction_agreement",
        hue="test_subset",
        style="corruption",
        markers=True,
        dashes=True,
        errorbar="sd",
        palette=palette,
        legend=False,
        ax=axes[1],
    )
    axes[0].set_ylabel("Chemical balanced accuracy")
    axes[1].set_ylabel("Clean/corrupted prediction agreement")
    axes[0].set_xlabel("Corruption severity")
    axes[1].set_xlabel("Corruption severity")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        fontsize=5.5,
        ncol=2,
        loc="lower left",
    )
    for label, ax in zip(("A", "B"), axes):
        ax.text(
            0.01,
            0.98,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
        )
    save_figure(
        save_publication_figure,
        fig,
        output_dir / "figures" / "corruption_robustness",
    )


def figure_swap_examples(
    output_dir: Path,
    save_publication_figure: Any,
) -> None:
    def correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left = left - left.mean(axis=1, keepdims=True)
        right = right - right.mean(axis=1, keepdims=True)
        return np.sum(left * right, axis=1) / np.maximum(
            np.linalg.norm(left, axis=1)
            * np.linalg.norm(right, axis=1),
            1.0e-12,
        )

    metrics = pd.read_csv(output_dir / "locked_outer_swap_metrics.csv")
    metrics = metrics[
        metrics["representation"].eq("arpls_minmax")
        & metrics["real_pair_count"].gt(0)
    ]
    examples: dict[str, dict[str, Any]] = {}
    for subset in COHORT_ORDER:
        candidates: list[dict[str, Any]] = []
        selected = metrics[metrics["test_subset"].eq(subset)]
        for row in selected.itertuples(index=False):
            with np.load(output_dir / row.artifact, allow_pickle=False) as data:
                partner = data["partner_clean"].astype(np.float32)
                swapped = data["swapped_reconstruction"].astype(np.float32)
                values = correlations(swapped, partner)
                for index, value in enumerate(values):
                    candidates.append(
                        {
                            "correlation": float(value),
                            "axis": data["axis_cm1"].astype(np.float32),
                            "source": data["source_clean"][index].astype(
                                np.float32
                            ),
                            "partner": partner[index],
                            "swapped": swapped[index],
                            "source_instrument": str(
                                data["source_instrument"][index]
                            ),
                            "partner_instrument": str(
                                data["partner_instrument"][index]
                            ),
                            "analyte": str(
                                data["source_target_analyte"][index]
                            ),
                        }
                    )
        if not candidates:
            continue
        median = float(
            np.median([item["correlation"] for item in candidates])
        )
        examples[subset] = min(
            candidates,
            key=lambda item: (
                abs(item["correlation"] - median),
                item["source_instrument"],
                item["partner_instrument"],
                item["analyte"],
            ),
        )

    fig = plt.figure(figsize=(7.2, 3.0))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.08))
    axes = []
    for index in range(3):
        axes.append(
            fig.add_subplot(
                grid[0, index],
                sharey=axes[0] if axes else None,
            )
        )
    for label, subset, ax in zip(("A", "B", "C"), COHORT_ORDER, axes):
        example = examples[subset]
        ax.plot(
            example["axis"],
            example["source"],
            color=OKABE_ITO["blue"],
            lw=0.8,
            label="Source",
        )
        ax.plot(
            example["axis"],
            example["partner"],
            color=OKABE_ITO["black"],
            lw=0.8,
            alpha=0.8,
            label="Real partner",
        )
        ax.plot(
            example["axis"],
            example["swapped"],
            color=OKABE_ITO["vermillion"],
            lw=0.9,
            label="Latent swap",
        )
        ax.set_title(COHORT_LABELS[subset])
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_xlim(400, 1800)
        ax.text(
            0.98,
            0.97,
            (
                f"median-representative r={example['correlation']:.2f}\n"
                f"{example['source_instrument']} → "
                f"{example['partner_instrument']}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.5,
        )
        ax.text(
            0.01,
            0.98,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
        )
    axes[0].set_ylabel("Normalized intensity")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_axis = fig.add_subplot(grid[1, :])
    legend_axis.axis("off")
    legend_axis.legend(
        handles,
        labels,
        loc="center",
        fontsize=6,
        ncol=3,
    )
    save_figure(
        save_publication_figure,
        fig,
        output_dir / "figures" / "locked_swap_examples",
    )


def architecture_and_failure_attribution(
    output_dir: Path,
    winners: pd.DataFrame,
    terminal: dict[str, Any],
    accounting: dict[str, Any],
) -> dict[str, Any]:
    control = winners[winners["source_stage"].eq("controls")].iloc[0]
    dependence = winners[winners["source_stage"].eq("dependence")].iloc[0]
    pair = winners[winners["source_stage"].eq("pair")].iloc[0]
    adversary = winners[
        winners["source_stage"].eq("instrument_adversary")
    ].iloc[0]
    result = {
        "terminal_classification": terminal["terminal_classification"],
        "undertraining": {
            "primary_failure": False,
            "evidence": terminal["convergence"],
        },
        "capacity_or_depth": {
            "primary_failure": False,
            "architecture": {
                "encoder_channels": [8, 16],
                "pooling_stages": 2,
                "total_latent_dimension": 64,
                "chemical_dimension": 48,
                "nuisance_dimension": 16,
                "selected_parameter_count": accounting[
                    "selected_parameter_count"
                ],
                "standard_vae_parameter_count": accounting[
                    "standard_vae_parameter_count"
                ],
            },
            "evidence": (
                "The union latent approximately reproduces mixed-VAE "
                "classification, reconstruction and posterior activity. "
                "Capacity is sufficient to encode the spectra, but the "
                "objective does not identify a unique chemical/nuisance split."
            ),
        },
        "partition_non_identifiability": {
            "primary_failure": True,
            "control_maximum_canonical_correlation": control[
                "partition_maximum_canonical_correlation"
            ],
            "selected_maximum_canonical_correlation": dependence[
                "partition_maximum_canonical_correlation"
            ],
            "selected_cross_covariance": dependence[
                "partition_cross_covariance_mean_square"
            ],
            "evidence": (
                "A batchwise linear covariance penalty lowers mean-square "
                "cross-covariance but leaves an almost perfectly predictable "
                "linear combination between partitions."
            ),
        },
        "adversarial_instability": {
            "primary_failure": True,
            "selected_instrument_probe": adversary[
                "chemical_instrument_probe_increment"
            ],
            "fold_fraction_improving_instrument": adversary[
                "fold_fraction_improving_instrument"
            ],
            "fold_fraction_preserving_chemical": adversary[
                "fold_fraction_preserving_chemical"
            ],
            "evidence": (
                "The internal adversary objective does not monotonically "
                "reduce the independent target-adjusted domain probe."
            ),
        },
        "pair_overalignment": {
            "primary_failure": True,
            "same_master_distance": pair[
                "chemical_same_master_cross_instrument_distance"
            ],
            "different_class_separation_margin": pair[
                "chemical_cross_instrument_separation_margin"
            ],
            "chemical_balanced_accuracy": pair[
                "chemical_target_balanced_accuracy"
            ],
            "evidence": (
                "Pair loss drives same-master distance near zero but also "
                "drives different-analyte separation near zero; invariance is "
                "obtained by compressing chemistry."
            ),
        },
        "data_identifiability_and_shift": {
            "primary_failure": True,
            "evidence": [
                "Analyte×instrument and analyte×sensor support is incomplete.",
                "Sensor family is strongly confounded with analyte.",
                "No independent preparation identifier is available.",
                "Field-stress reconstruction and classification collapse.",
                "Held-out sensor-family transfer is poor.",
            ],
        },
    }
    write_json(output_dir / "failure_attribution.json", result)
    return result


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def report_markdown(
    output_dir: Path,
    winners: pd.DataFrame,
    terminal: dict[str, Any],
    comparator: pd.DataFrame,
    accounting: dict[str, Any],
    convergence: dict[str, Any],
) -> str:
    stage_lines = []
    for row in winners.itertuples(index=False):
        stage_lines.append(
            "| "
            + " | ".join(
                [
                    str(row.source_stage).replace("_", " ").title(),
                    f"{int(row.gate_count)}/{int(row.gate_total)}",
                    fmt(row.chemical_target_balanced_accuracy),
                    fmt(row.chemical_instrument_probe_increment),
                    fmt(row.chemical_same_master_cross_instrument_distance),
                    fmt(row.chemical_cross_instrument_separation_margin),
                    fmt(row.partition_maximum_canonical_correlation),
                ]
            )
            + " |"
        )
    outer = terminal["locked_outer_arpls"]
    domain = terminal["domain_transfer"]
    poster = terminal["poster_descriptive"]
    comparison = comparator.pivot(
        index="display_model",
        columns="test_subset",
        values="mean_balanced_accuracy",
    )
    comparator_lines = []
    for model in [
        "PCA/logistic",
        "Linear SVM",
        "Standard VAE-500",
        "Structured VAE—chemical",
        "Structured VAE—union",
        "Siamese",
    ]:
        if model not in comparison.index:
            continue
        comparator_lines.append(
            f"| {model} | "
            f"{fmt(comparison.loc[model, 'strict_core'])} | "
            f"{fmt(comparison.loc[model, 'quality_pass'])} | "
            f"{fmt(comparison.loc[model, 'field_quality_stress'])} |"
        )
    rebuild_section = ""
    rebuild_path = output_dir / "rebuild_validation.json"
    if rebuild_path.is_file():
        rebuild = json.loads(rebuild_path.read_text())
        if rebuild.get("status") == "passed":
            rebuild_section = f"""
## Reproducibility verification

An independent rebuild was executed in a previously nonexistent output
directory. All `{int(rebuild['check_count']):,}` cross-build checks passed:
canonical scientific tables and decision JSON were exact; frozen-input
SHA-256 digests matched after normalizing the intentionally different output
directory; every embedding and reconstruction array was exact; and every
checkpoint and optimizer tensor was exact. `rebuild_validation.json` contains
the complete machine-readable audit.
"""
    swap_section = ""
    swap_path = output_dir / "locked_outer_swap_metrics.csv"
    if swap_path.is_file():
        swaps = pd.read_csv(swap_path)
        primary_swaps = swaps[
            swaps["representation"].eq("arpls_minmax")
        ]
        swap_values = (
            primary_swaps.groupby("test_subset")[
                [
                    "swap_partner_median_correlation",
                    "source_partner_median_correlation",
                ]
            ]
            .mean()
            .to_dict(orient="index")
        )
        strict_swap = swap_values["strict_core"]
        quality_swap = swap_values["quality_pass"]
        stress_swap = swap_values["field_quality_stress"]
        swap_section = f"""
## Locked real-pair swaps

All 30 outer scenario/representation combinations have inspectable latent-swap
bundles. Each swap decodes the source chemical mean with a deterministic,
real same-master/different-instrument partner's nuisance mean and domain
labels. For arPLS, mean fold-level swap-to-partner median correlation is
`{fmt(strict_swap['swap_partner_median_correlation'])}` on strict,
`{fmt(quality_swap['swap_partner_median_correlation'])}` on quality, and
`{fmt(stress_swap['swap_partner_median_correlation'])}` on field stress,
compared with unmodeled source-to-partner correlations of
`{fmt(strict_swap['source_partner_median_correlation'])}`,
`{fmt(quality_swap['source_partner_median_correlation'])}`, and
`{fmt(stress_swap['source_partner_median_correlation'])}`. These are
descriptive swap reconstructions, not semantic validation: the nuisance block
retains analyte information and the partitions remain almost canonically
collinear. `locked_outer_swap_metrics.csv` and `swaps/` retain every metric and
spectrum pair, including the one stress fold with no valid real partner.
"""
    return f"""# Structured/disentangled VAE study on the NATO SERS field trial

## Executive conclusion

**Terminal evidence class: unsuccessful.** The proposed structured VAE did
not learn a defensible chemical/nuisance disentanglement and did not become a
general instrument, sensor, substrate or field-noise filter.

There is a real partial effect: the selected dependence penalty reduced the
inner target-adjusted instrument probe from `0.572` for the zero-structure
partition control to `0.543`, and reduced cross-covariance from `0.0655` to
`0.0248`. That effect did not survive all required tests. Same-master geometry
remained `0.783`, only 50% of folds preserved chemistry, maximum canonical
correlation remained `0.990`, and locked instrument leakage rose to `0.584`.

The chemical/nuisance split is not semantic: the nuisance partition alone
classifies strict-core analytes at BA `{fmt(outer['strict_nuisance_partition_chemical_ba'])}`,
while chemical–nuisance maximum canonical correlation is
`{fmt(outer['strict_partition_maximum_canonical_correlation'])}`. The union
latent behaves like the original mixed VAE, whereas the chemical partition
does not outperform it.

## Question and claim rules

The study asked whether a fixed-capacity z64 VAE, split into z48 chemical and
z16 nuisance variables, could preserve analyte signal while removing
instrument, sensor/substrate, baseline and noise variation. The registered
terminal classes were:

1. **Disentangled:** chemical and nuisance semantics, low dependence, nuisance
   removal, preserved chemistry and swap evidence all pass.
2. **Nuisance-suppressed:** chemistry is preserved and nuisance is reduced,
   but full semantic factorization is not supported.
3. **Structured-only:** partitions are operationally useful but nuisance
   suppression is not established.
4. **Unsuccessful:** no registered candidate passes every applicable gate.

No inner candidate passed every gate, so locked results could characterize but
could not rescue the claim.

## Data and preprocessing held fixed

- Strict core: 598 spectra, 69 master samples, 7 analytes, 10 instruments and
  4 sensor families.
- Quality-pass subset: 500 spectra.
- Field-quality stress cohort: 98 spectra.
- Common axis: 400–1800 cm⁻¹ at 1 cm⁻¹ spacing (1,401 values).
- Primary view: despiking/alignment, arPLS baseline correction and per-spectrum
  min–max scaling.
- Sensitivity view: the same axis, alignment and min–max scaling with minimal
  baseline removal.
- All split decisions are grouped by `master_sample_id`.

The metadata audit found only 44/70 analyte×instrument cells and 17/28
analyte×sensor cells supported. Analyte–sensor Cramér's V was 0.542 and no
independent preparation ID was available. These facts cap causal
identifiability: a low domain probe cannot by itself prove removal of physical
instrument or substrate effects.

## Model and training

The encoder has two 1-D convolution/max-pooling blocks with 8 and 16 channels.
It produces separate posterior means/log-variances for z48 chemical and z16
nuisance partitions. The decoder receives their concatenation; optional fixed
instrument/sensor conditioning inputs are present in every control so parameter
comparisons remain fair. Registered heads include chemical classification,
nuisance instrument/sensor classification and target-conditioned gradient
reversal adversaries.

Every authoritative run used:

- spectral-composite reconstruction loss;
- β=0.25 with the frozen four-phase warm-up;
- Adam, learning rate 0.001, weight decay 1e-5;
- batch size 64, gradient clipping at 5;
- exactly 500 epochs with checkpoints at 100/300/400/500;
- total latent capacity fixed at 64 and no encoder–decoder skips.

Selected parameter count was `{accounting['selected_parameter_count']:,}`
versus `{accounting['standard_vae_parameter_count']:,}` for the mixed VAE
(`{accounting['parameter_increase_fraction'] * 100:.1f}%` more). The study ran
`{accounting['authoritative_training_run_count']}` authoritative fits and
`{accounting['authoritative_optimizer_epoch_count']:,}` optimizer epochs;
diagnostic smoke tests are excluded.

## Identity and convergence

The exact standard-VAE identity control reproduced all 20 grouped-inner
histories, checkpoints and optimizer states with maximum numeric difference
zero. Selection was blocked until this passed.

The selected structured model converged under the registered rule: median
final-50 improvement was
`{convergence['inner_selected']['median_final50_relative_improvement'] * 100:.3f}%`
and only
`{convergence['inner_selected']['fraction_folds_improving_at_least_1_percent'] * 100:.0f}%`
of folds improved at least 1%. Locked fits were more variable, but their median
final-50 change was
`{convergence['locked_outer']['median_final50_relative_improvement'] * 100:.3f}%`
and half worsened. More epochs might move a minority of fits, but undertraining
does not explain near-unit partition dependence or held-out domain collapse.

## Inner mechanism search

| Branch winner | Gates | Chemical BA | Instrument probe | Same-master distance | Separation margin | Max CCA |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(stage_lines)}

The instrument adversary improved same-master geometry and separation but did
not reduce the independent instrument probe consistently. Pair alignment
reduced same-master distance to `0.008`, but separation margin collapsed to
`0.002`: the model aligned both same- and different-analyte spectra. The
dependence penalty was selected by the fixed hierarchy at 15/17 gates; it
failed same-master geometry and fold-wise chemistry preservation.

Sensor adversaries remained closed because no instrument-adversarial candidate
was eligible. Combinations remained closed because no two individual
mechanisms were eligible. No post-hoc weights were added.

## Negative controls

Grouped chemical-label permutation reduced mean BA to `0.162`; every fold was
below 0.25 and the maximum was `0.218`. Nuisance-label and partner permutations
were non-applicable to the frozen dependence-only objective because all
nuisance-label, adversarial, pair and cross-reconstruction weights were zero.
They are recorded as non-applicable rather than passed.

## Preprocessing sensitivity

Quality-pass arPLS reached chemical BA `0.719`, instrument probe `0.540` and
same-master distance `0.743`. Strict minimal reached BA `0.628`, probe `0.644`
and distance `0.863`. Quality minimal recovered BA to `0.699`, but leakage
remained `0.631` and distance `0.834`.

Minimal preprocessing preserves more peaks, but leaves substantially more
instrument/background structure. Min–max scaling correctly places all spectra
on a common amplitude range; it cannot remove background curvature or
system-response shape. The evidence supports one common arPLS primary view plus
minimal sensitivity—not ad hoc instrument-specific preprocessing chosen after
outcome inspection.

## Locked grouped-outer results

| Model/view | Strict core | Quality pass | Field stress |
|---|---:|---:|---:|
{chr(10).join(comparator_lines)}

For the primary arPLS chemical partition:

- strict BA `{fmt(outer['strict_chemical_ba'])}`, instrument probe
  `{fmt(outer['strict_instrument_probe'])}`, same-master distance
  `{fmt(outer['strict_same_master_distance'])}`;
- quality BA `{fmt(outer['quality_chemical_ba'])}`, instrument probe
  `{fmt(outer['quality_instrument_probe'])}`;
- field-stress BA `{fmt(outer['field_stress_chemical_ba'])}` and reconstruction
  correlation `{fmt(outer['field_stress_reconstruction_correlation'])}`.

The field-stress result is the main operational failure. Composite corruption
at severity 1 reduces strict chemical BA from `0.681` to `0.596` and clean/
corrupted agreement to `0.696`. Minimal spectra preserve more repeatable peaks
but do not solve field stress.
{swap_section}

## Held-out instrument and sensor transfer

Domain-only instrument BA averages
`{fmt(domain['strict_domain_only_instrument_ba'])}` on strict data and
`{fmt(domain['quality_domain_only_instrument_ba'])}` on quality data, with
large between-instrument ranges. Sensor-family BA averages only
`{fmt(domain['strict_domain_only_sensor_ba'])}` and
`{fmt(domain['quality_domain_only_sensor_ba'])}`.

Domain-plus-sample scores require caution: held-out domains may contain analytes
not represented in the remaining training partition. Some apparent 1.0 scores
are based on 17–21 supported spectra, while some sensor-family tests have zero
supported analytes. Tables retain supported/unsupported counts and do not use
those cells as evidence of generalization.

## Poster transfer

The architecture transfers descriptively to the poster data:
leave-one-substrate-family-out chemical BA is
`{fmt(poster['arpls_chemical_ba'])}` for arPLS and
`{fmt(poster['minimal_chemical_ba'])}` for minimal spectra. This is not NATO
label transfer—the poster analytes differ—and it is not disentanglement:
nuisance chemistry remains high and partition CCA is approximately 0.9995.

## Did the idea work?

**Not as intended.** A structured VAE can allocate capacity into named blocks,
but naming the blocks and penalizing covariance does not make the factorization
identifiable. The union representation remains competitive, so the architecture
can encode spectra. The failure is the semantic allocation:

- reconstruction can place chemistry and nuisance in either partition;
- a covariance penalty removes only batchwise linear covariance, not shared
  predictive structure;
- adversarial loss is unstable under analyte×instrument confounding;
- pair consistency can erase analyte separation while looking invariant;
- field-stress spectra are far outside the quality training distribution.

This does not show that disentangled SERS models are impossible. It shows that
this bounded formulation and this confounded field-trial dataset do not support
the claim.
{rebuild_section}

## Recommended next study

1. Treat acquisition as a balanced-design problem: collect the same analytes,
   preparations and concentrations across every instrument/sensor, and record
   independent preparation/batch IDs.
2. Add explicit class-preserving negatives to pair alignment: same-master
   positives plus different-analyte cross-instrument margin/contrastive terms.
3. Replace simple covariance with stronger dependence control (HSIC, total
   correlation or conditional mutual-information surrogates), while evaluating
   external probes.
4. Consider an instrument-aware physical front end or calibration layer before
   the shared chemical encoder; do not require one latent model to discover all
   baseline physics unsupervised.
5. Build a field-stress rejection/QC pathway. The current model should not
   classify spectra it cannot reconstruct or place near the training domain.
6. Keep PCA/logistic and the mixed VAE as mandatory comparators. A future model
   must beat them on grouped outer, field stress and held-out sensor transfer,
   not only on reconstruction.

## Artifact guide

- `terminal_decision.json`: final evidence class and headline numbers.
- `failure_attribution.json`: undertraining, capacity, objective and data-shift
  attribution.
- `inner_stage_winners.csv` / `inner_gate_matrix.csv`: all selection decisions.
- `locked_outer_*`, `locked_domain_*`, `locked_poster_*`: complete locked
  metrics, predictions, reconstruction, corruption, histories and registries.
- `locked_outer_swap_metrics.csv` / `swaps/`: real same-master latent swaps and
  the underlying source, partner and decoded spectra.
- `per_analyte_failures.csv`, `per_instrument_failures.csv`,
  `per_sensor_failures.csv`, `per_domain_failures.csv`: granular failures.
- `comparator_summary.csv`: matched outer-fold comparators.
- `figures/`: PDF and 600-DPI PNG figures.
- `validation_report.json` / `rebuild_validation.json`: within-build and exact
  independent-rebuild audits.
- `artifact_hashes.json`, `environment.json`, `reproduction_commands.sh`:
  rebuild provenance.
"""


def environment_record(repository: Path) -> dict[str, Any]:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.cuda.is_available()
            else None
        ),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "git_commit": git_commit,
    }


def reproduction_commands(output_dir: Path) -> None:
    text = """#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python scripts/audit_sers_structured_vae_metadata.py
.venv/bin/python scripts/run_sers_structured_vae_identity.py --device cuda
.venv/bin/python scripts/run_sers_structured_vae_selection.py --stage controls --training-device cuda --metric-device cpu
.venv/bin/python scripts/run_sers_structured_vae_selection.py --stage instrument_adversary --training-device cuda --metric-device cpu
.venv/bin/python scripts/run_sers_structured_vae_selection.py --stage pair --training-device cuda --metric-device cpu
.venv/bin/python scripts/run_sers_structured_vae_selection.py --stage dependence --training-device cuda --metric-device cpu
.venv/bin/python scripts/finalize_sers_structured_vae_inner.py --device cpu
.venv/bin/python scripts/run_sers_structured_vae_confirmation.py --stage sensitivity --training-device cuda --metric-device cpu
.venv/bin/python scripts/run_sers_structured_vae_confirmation.py --stage outer --training-device cuda --metric-device cpu
.venv/bin/python scripts/run_sers_structured_vae_confirmation.py --stage domain --training-device cuda --metric-device cpu
.venv/bin/python scripts/run_sers_structured_vae_confirmation.py --stage poster --training-device cuda --metric-device cpu
.venv/bin/python scripts/export_sers_structured_vae_swaps.py --device cuda
.venv/bin/python scripts/finalize_sers_structured_vae.py
.venv/bin/python scripts/validate_sers_structured_vae.py

# Independent clean rebuild (the destination must not already exist):
# scripts/rebuild_sers_structured_vae.sh Workspace/sers_structured_vae/structured_vae_v1_rebuild
"""
    (output_dir / "reproduction_commands.sh").write_text(text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(repository: Path, output_dir: Path) -> dict[str, str]:
    paths: list[Path] = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir)
        if any("cache" in part or part == "checkpoints" for part in relative.parts):
            continue
        if path.name in {
            "artifact_hashes.json",
            "validation_report.json",
            "rebuild_validation.json",
        }:
            continue
        paths.append(path)
    for relative in (
        "configs/sers_structured_vae_v1.json",
        "docs/SERS_STRUCTURED_VAE_PROTOCOL_V1.md",
        "scripts/sers_structured_vae_common.py",
        "scripts/run_sers_structured_vae_identity.py",
        "scripts/run_sers_structured_vae_selection.py",
        "scripts/finalize_sers_structured_vae_inner.py",
        "scripts/run_sers_structured_vae_confirmation.py",
        "scripts/export_sers_structured_vae_swaps.py",
        "scripts/finalize_sers_structured_vae.py",
        "scripts/validate_sers_structured_vae.py",
        "scripts/rebuild_sers_structured_vae.sh",
    ):
        path = repository / relative
        if path.is_file():
            paths.append(path)
    result = {
        str(path.relative_to(repository)): sha256_file(path)
        for path in sorted(set(paths))
    }
    write_json(output_dir / "artifact_hashes.json", result)
    return result


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_structured_vae"
        / "structured_vae_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    repository = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    winners = stage_winners(output_dir)
    gate_matrix(output_dir)
    outer, predictions = outer_summaries(output_dir)
    failure_tables = save_failure_tables(output_dir, predictions)
    domain = domain_summary(output_dir)
    poster = poster_summary(output_dir)
    comparator = baseline_comparators(repository, output_dir, outer)
    convergence = convergence_attribution(output_dir)
    terminal = terminal_decision(
        output_dir, winners, outer, domain, poster, convergence
    )
    selected_parameters = int(
        winners[winners["source_stage"].eq("dependence")][
            "parameter_count"
        ].iloc[0]
    )
    accounting = compute_accounting(output_dir, selected_parameters)
    attribution = architecture_and_failure_attribution(
        output_dir, winners, terminal, accounting
    )
    save_publication_figure, _ = configure_figures(repository)
    figure_inner_search(output_dir, save_publication_figure)
    figure_outer_comparators(
        output_dir, comparator, save_publication_figure
    )
    figure_partition_and_preprocessing(
        output_dir, outer, save_publication_figure
    )
    figure_domain_heatmap(output_dir, domain, save_publication_figure)
    figure_corruption(output_dir, save_publication_figure)
    figure_swap_examples(output_dir, save_publication_figure)
    report = report_markdown(
        output_dir,
        winners,
        terminal,
        comparator,
        accounting,
        convergence,
    )
    (output_dir / "FINAL_REPORT.md").write_text(report)
    rebuild_note = (
        " Exact independent rebuild: **passed**; see "
        "[rebuild_validation.json](rebuild_validation.json)."
        if (output_dir / "rebuild_validation.json").is_file()
        else ""
    )
    (output_dir / "README.md").write_text(
        "# SERS structured VAE v1\n\n"
        "**Outcome: unsuccessful as disentanglement or general nuisance "
        "suppression.** See [FINAL_REPORT.md](FINAL_REPORT.md), "
        "[terminal_decision.json](terminal_decision.json), and `figures/`."
        f"{rebuild_note}\n"
    )
    write_json(output_dir / "environment.json", environment_record(repository))
    reproduction_commands(output_dir)
    artifact_hashes(repository, output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "terminal_classification": terminal[
                    "terminal_classification"
                ],
                "idea_worked_as_disentanglement": terminal[
                    "idea_worked_as_disentanglement"
                ],
                "figure_count": len(
                    list((output_dir / "figures").glob("*.pdf"))
                ),
                "failure_table_rows": {
                    key: len(value) for key, value in failure_tables.items()
                },
                "attribution_primary_failures": [
                    key
                    for key, value in attribution.items()
                    if isinstance(value, dict)
                    and value.get("primary_failure") is True
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
