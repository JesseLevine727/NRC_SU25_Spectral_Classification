#!/usr/bin/env python3
"""Finalize the supervised-contrastive SERS experiment and decision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

import sers_baseline_common as baseline
import sers_classical_benchmark_common as classical


COLORS = {
    "Classical": "#000000",
    "Siamese": "#E69F00",
    "Contrastive successor": "#0072B2",
    "strict_core": "#0072B2",
    "quality_pass": "#009E73",
    "field_quality_stress": "#D55E00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Workspace/sers_supervised_contrastive/contrastive_v1"
        ),
    )
    parser.add_argument(
        "--classical-bundle",
        type=Path,
        default=Path(
            "Workspace/sers_classical_benchmark/classical_benchmark_v2"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/sers_supervised_contrastive_v1.json"),
    )
    return parser.parse_args()


def mean_ci(values: pd.Series) -> dict[str, float | int]:
    values = values.dropna().astype(float)
    n = len(values)
    mean = float(values.mean()) if n else np.nan
    sd = float(values.std(ddof=1)) if n > 1 else np.nan
    half = (
        float(t.ppf(0.975, n - 1) * sd / math.sqrt(n))
        if n > 1
        else np.nan
    )
    return {
        "mean": mean,
        "sd": sd,
        "ci95_half_width": half,
        "n_independent_units": n,
    }


def deep_outer_summary(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[
        ~frame["variant"].str.startswith("preprocessing_sensitivity_")
    ].copy()
    per_fold = (
        frame.groupby(
            ["variant", "evaluation_subset", "outer_fold"], as_index=False
        )
        .agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            accuracy=("accuracy", "mean"),
            negative_log_likelihood=("negative_log_likelihood", "mean"),
            expected_calibration_error_10=(
                "expected_calibration_error_10",
                "mean",
            ),
            different_minus_same_margin=(
                "different_minus_same_margin",
                "mean",
            ),
            embedding_effective_rank=("embedding_effective_rank", "mean"),
            analyte_probe_balanced_accuracy=(
                "analyte_probe_balanced_accuracy",
                "mean",
            ),
            instrument_probe_balanced_accuracy=(
                "instrument_probe_balanced_accuracy",
                "mean",
            ),
            sensor_probe_balanced_accuracy=(
                "sensor_probe_balanced_accuracy",
                "mean",
            ),
        )
    )
    rows: list[dict[str, Any]] = []
    for keys, group in per_fold.groupby(
        ["variant", "evaluation_subset"], sort=True
    ):
        variant, subset = keys
        for metric in (
            "balanced_accuracy",
            "macro_f1",
            "accuracy",
            "negative_log_likelihood",
            "expected_calibration_error_10",
            "different_minus_same_margin",
            "embedding_effective_rank",
            "analyte_probe_balanced_accuracy",
            "instrument_probe_balanced_accuracy",
            "sensor_probe_balanced_accuracy",
        ):
            rows.append(
                {
                    "variant": variant,
                    "evaluation_subset": subset,
                    "metric": metric,
                    "independent_unit": "outer master-group fold after averaging seeds",
                    **mean_ci(group[metric]),
                }
            )
    return pd.DataFrame(rows)


def preprocessing_sensitivity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    sensitivity = frame[
        (frame["variant"] == "full_domain_aware")
        | frame["variant"].str.startswith("preprocessing_sensitivity_")
    ].copy()
    per_fold = (
        sensitivity.groupby(
            ["representation", "evaluation_subset", "outer_fold"],
            as_index=False,
        )
        .agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
        )
    )
    rows: list[dict[str, Any]] = []
    for (representation, subset), group in per_fold.groupby(
        ["representation", "evaluation_subset"], sort=True
    ):
        for metric in ("balanced_accuracy", "macro_f1"):
            rows.append(
                {
                    "representation": representation,
                    "evaluation_subset": subset,
                    "metric": metric,
                    "independent_unit": (
                        "outer master-group fold after averaging seeds"
                    ),
                    **mean_ci(group[metric]),
                }
            )
    return pd.DataFrame(rows)


def prior_siamese_summary(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "test_strict_core": "strict_core",
        "test_quality_pass": "quality_pass",
        "test_field_quality_stress": "field_quality_stress",
    }
    frame = frame.copy()
    frame["evaluation_subset"] = frame["scenario"].map(
        lambda value: next(
            subset for token, subset in mapping.items() if token in value
        )
    )
    per_fold = (
        frame.groupby(["evaluation_subset", "outer_fold"], as_index=False)
        .agg(
            balanced_accuracy=("balanced_accuracy_supported", "mean"),
            macro_f1=("macro_f1_supported", "mean"),
        )
    )
    rows = []
    for subset, group in per_fold.groupby("evaluation_subset"):
        for metric in ("balanced_accuracy", "macro_f1"):
            rows.append(
                {
                    "model": "Siamese",
                    "evaluation_subset": subset,
                    "metric": metric,
                    **mean_ci(group[metric]),
                }
            )
    return pd.DataFrame(rows)


def classical_summary(bundle: Path) -> pd.DataFrame:
    frame = pd.read_csv(bundle / "outer_metrics.csv")
    rows = []
    for subset, group in frame.groupby("evaluation_subset"):
        for metric in ("balanced_accuracy", "macro_f1"):
            rows.append(
                {
                    "model": "Classical",
                    "evaluation_subset": subset,
                    "metric": metric,
                    **mean_ci(group[metric]),
                }
            )
    return pd.DataFrame(rows)


def model_summary(
    deep: pd.DataFrame, siamese: pd.DataFrame, classical_frame: pd.DataFrame
) -> pd.DataFrame:
    full = deep[
        (deep["variant"] == "full_domain_aware")
        & deep["metric"].isin(["balanced_accuracy", "macro_f1"])
    ].copy()
    full["model"] = "Contrastive successor"
    return pd.concat(
        [
            classical_frame,
            siamese,
            full[
                [
                    "model",
                    "evaluation_subset",
                    "metric",
                    "mean",
                    "sd",
                    "ci95_half_width",
                    "n_independent_units",
                ]
            ],
        ],
        ignore_index=True,
    )


def paired_gate(
    output_dir: Path,
    protocol: dict[str, Any],
    outer: pd.DataFrame,
    domain: pd.DataFrame,
    siamese: pd.DataFrame,
    classical_bundle: Path,
) -> dict[str, Any]:
    full = outer[outer["variant"] == "full_domain_aware"].copy()
    deep_fold = (
        full.groupby(["evaluation_subset", "outer_fold"], as_index=False)[
            "balanced_accuracy"
        ]
        .mean()
    )
    classical_outer = pd.read_csv(classical_bundle / "outer_metrics.csv")
    siamese = siamese.copy()
    mapping = {
        "test_strict_core": "strict_core",
        "test_quality_pass": "quality_pass",
        "test_field_quality_stress": "field_quality_stress",
    }
    siamese["evaluation_subset"] = siamese["scenario"].map(
        lambda value: next(
            subset for token, subset in mapping.items() if token in value
        )
    )
    siamese_fold = (
        siamese.groupby(["evaluation_subset", "outer_fold"], as_index=False)
        .agg(balanced_accuracy=("balanced_accuracy_supported", "mean"))
    )
    cohort_differences: dict[str, Any] = {}
    for subset in ("strict_core", "quality_pass", "field_quality_stress"):
        deep_values = deep_fold[
            deep_fold["evaluation_subset"] == subset
        ].sort_values("outer_fold")["balanced_accuracy"].to_numpy()
        classical_values = classical_outer[
            classical_outer["evaluation_subset"] == subset
        ].sort_values("outer_fold")["balanced_accuracy"].to_numpy()
        siamese_values = siamese_fold[
            siamese_fold["evaluation_subset"] == subset
        ].sort_values("outer_fold")["balanced_accuracy"].to_numpy()
        cohort_differences[subset] = {
            "successor_mean": float(deep_values.mean()),
            "classical_mean": float(classical_values.mean()),
            "siamese_mean": float(siamese_values.mean()),
            "successor_minus_classical": float(
                np.mean(deep_values - classical_values)
            ),
            "successor_minus_siamese": float(
                np.mean(deep_values - siamese_values)
            ),
        }
    # Seed-direction evidence against matched historical Siamese seeds.
    seed_rows = []
    for seed in protocol["fixed_training"]["final_seeds"]:
        deep_seed = full[full["declared_seed"] == seed]
        siamese_seed = siamese[siamese["seed"] == seed]
        deep_mean = float(deep_seed["balanced_accuracy"].mean())
        siamese_mean = float(
            siamese_seed["balanced_accuracy_supported"].mean()
        )
        seed_rows.append(
            {
                "seed": seed,
                "successor_mean_across_cohorts_and_folds": deep_mean,
                "siamese_mean_across_cohorts_and_folds": siamese_mean,
                "difference": deep_mean - siamese_mean,
            }
        )
    positive_seed_count = sum(row["difference"] > 0 for row in seed_rows)

    classical_domain = pd.read_csv(classical_bundle / "domain_metrics.csv")
    deep_domain = (
        domain.groupby(
            ["subset", "protocol", "domain_type", "heldout_domain"],
            as_index=False,
        )
        .agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            n_supported=("n_supported", "mean"),
        )
    )
    domain_advantages = []
    for protocol_name in ("domain_only", "domain_and_sample"):
        for domain_type in ("instrument", "sensor_family"):
            for subset in ("strict_core", "quality_pass"):
                left = deep_domain[
                    (deep_domain["subset"] == subset)
                    & (deep_domain["protocol"] == protocol_name)
                    & (deep_domain["domain_type"] == domain_type)
                ][
                    ["heldout_domain", "balanced_accuracy", "n_supported"]
                ].rename(
                    columns={"balanced_accuracy": "deep"}
                )
                right = classical_domain[
                    (classical_domain["subset"] == subset)
                    & (classical_domain["protocol"] == protocol_name)
                    & (classical_domain["domain_type"] == domain_type)
                ][
                    ["heldout_domain", "balanced_accuracy", "n_supported"]
                ].rename(
                    columns={
                        "balanced_accuracy": "classical",
                        "n_supported": "classical_n_supported",
                    }
                )
                merged = left.merge(right, on="heldout_domain").dropna(
                    subset=["deep", "classical"]
                )
                merged = merged[
                    (merged["n_supported"] > 0)
                    & (merged["classical_n_supported"] > 0)
                ]
                domain_advantages.append(
                    {
                        "subset": subset,
                        "protocol": protocol_name,
                        "domain_type": domain_type,
                        "n_domains": len(merged),
                        "successor_minus_classical": (
                            float((merged["deep"] - merged["classical"]).mean())
                            if len(merged)
                            else np.nan
                        ),
                    }
                )
    primary_deep = deep_domain[
        (deep_domain["protocol"] == "domain_and_sample")
        & (deep_domain["domain_type"] == "instrument")
        & (deep_domain["n_supported"] > 0)
    ][
        ["subset", "heldout_domain", "balanced_accuracy", "n_supported"]
    ].rename(columns={"balanced_accuracy": "deep"})
    primary_classical = classical_domain[
        (classical_domain["protocol"] == "domain_and_sample")
        & (classical_domain["domain_type"] == "instrument")
        & (classical_domain["n_supported"] > 0)
    ][
        ["subset", "heldout_domain", "balanced_accuracy", "n_supported"]
    ].rename(
        columns={
            "balanced_accuracy": "classical",
            "n_supported": "classical_n_supported",
        }
    )
    primary_domain_pairs = primary_deep.merge(
        primary_classical,
        on=["subset", "heldout_domain"],
    ).dropna(subset=["deep", "classical"])
    primary_domain_advantage = float(
        (
            primary_domain_pairs["deep"]
            - primary_domain_pairs["classical"]
        ).mean()
    )
    siamese_domain = pd.read_csv(
        output_dir / "siamese_control_domain_metrics.csv"
    )
    primary_deep_seeded = domain[
        (domain["protocol"] == "domain_and_sample")
        & (domain["domain_type"] == "instrument")
        & (domain["n_supported"] > 0)
    ][
        [
            "subset",
            "heldout_domain",
            "declared_seed",
            "balanced_accuracy",
        ]
    ].rename(columns={"balanced_accuracy": "deep"})
    primary_siamese_seeded = siamese_domain[
        (siamese_domain["domain_protocol"] == "domain_and_sample")
        & (siamese_domain["domain_type"] == "instrument")
        & (siamese_domain["n_test_supported"] > 0)
    ][
        [
            "evaluation_subset",
            "heldout_domain",
            "seed",
            "balanced_accuracy_supported",
        ]
    ].rename(
        columns={
            "evaluation_subset": "subset",
            "seed": "declared_seed",
            "balanced_accuracy_supported": "siamese",
        }
    )
    primary_siamese_pairs = primary_deep_seeded.merge(
        primary_siamese_seeded,
        on=["subset", "heldout_domain", "declared_seed"],
    ).dropna(subset=["deep", "siamese"])
    primary_domain_advantage_vs_siamese = float(
        (
            primary_siamese_pairs["deep"]
            - primary_siamese_pairs["siamese"]
        ).mean()
    )
    primary_domain_seed_differences_vs_siamese = []
    for seed, frame in primary_siamese_pairs.groupby(
        "declared_seed", sort=True
    ):
        primary_domain_seed_differences_vs_siamese.append(
            {
                "seed": int(seed),
                "n_domain_cohort_pairs": len(frame),
                "difference": float(
                    (frame["deep"] - frame["siamese"]).mean()
                ),
            }
        )

    deep_selective = pd.read_csv(
        output_dir / "outer_selective_metrics.csv"
    )
    deep_stress_selective = deep_selective[
        (deep_selective["variant"] == "full_domain_aware")
        & (
            deep_selective["evaluation_subset"]
            == "field_quality_stress"
        )
        & deep_selective["rejection_selected"].astype(bool)
        & np.isclose(deep_selective["requested_coverage"], 0.8)
    ]
    deep_stress_fold = (
        deep_stress_selective.groupby("outer_fold", as_index=False)[
            "accuracy"
        ]
        .mean()
        .sort_values("outer_fold")
    )
    classical_selective = pd.read_csv(
        classical_bundle / "selective_metrics.csv"
    )
    classical_stress_fold = classical_selective[
        (
            classical_selective["evaluation_subset"]
            == "field_quality_stress"
        )
        & np.isclose(classical_selective["requested_coverage"], 0.8)
    ].sort_values("outer_fold")
    stress_selective_pairs = deep_stress_fold.merge(
        classical_stress_fold[["outer_fold", "accuracy"]],
        on="outer_fold",
        suffixes=("_successor", "_classical"),
    )
    field_stress_selective_advantage = float(
        (
            stress_selective_pairs["accuracy_successor"]
            - stress_selective_pairs["accuracy_classical"]
        ).mean()
    )
    minimum = float(protocol["success_gate"]["minimum_mean_advantage_for_materiality"])
    maximum_loss = float(
        protocol["success_gate"][
            "maximum_allowed_strict_or_quality_balanced_accuracy_loss"
        ]
    )
    no_clean_degradation = all(
        cohort_differences[subset]["successor_minus_classical"]
        >= -maximum_loss
        for subset in ("strict_core", "quality_pass")
    )
    stress_advantage = cohort_differences["field_quality_stress"][
        "successor_minus_classical"
    ]
    secondary_best_domain_advantage = float(
        np.nanmax(
            [row["successor_minus_classical"] for row in domain_advantages]
        )
    )
    material_shift_advantage = max(
        field_stress_selective_advantage, primary_domain_advantage
    ) >= minimum
    reproducibly_beats_siamese = positive_seed_count >= 2
    supported = (
        no_clean_degradation
        and material_shift_advantage
        and reproducibly_beats_siamese
    )
    return {
        "successor_supported": supported,
        "cohort_differences": cohort_differences,
        "seed_differences_vs_siamese": seed_rows,
        "positive_seed_count": positive_seed_count,
        "domain_advantages": domain_advantages,
        "primary_domain_pair_count": len(primary_domain_pairs),
        "primary_domain_advantage": primary_domain_advantage,
        "primary_domain_advantage_vs_siamese": (
            primary_domain_advantage_vs_siamese
        ),
        "primary_domain_seed_differences_vs_siamese": (
            primary_domain_seed_differences_vs_siamese
        ),
        "secondary_best_domain_advantage": (
            secondary_best_domain_advantage
        ),
        "field_stress_full_coverage_advantage": stress_advantage,
        "field_stress_selective_pair_count": len(
            stress_selective_pairs
        ),
        "field_stress_selective_advantage_at_80_percent": (
            field_stress_selective_advantage
        ),
        "gate_no_clean_degradation": no_clean_degradation,
        "gate_material_shift_advantage": material_shift_advantage,
        "gate_reproducibly_beats_siamese": reproducibly_beats_siamese,
        "minimum_material_advantage": minimum,
        "maximum_clean_loss": maximum_loss,
    }


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def consolidate_training_histories(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_parts: list[pd.DataFrame] = []
    for path in sorted(
        (output_dir / "selection_shards").glob("*/*__history.csv")
    ):
        selection_parts.append(pd.read_csv(path))
    selection = pd.concat(selection_parts, ignore_index=True, sort=False)
    selection.to_csv(
        output_dir / "selection_training_histories.csv", index=False
    )

    final_parts: list[pd.DataFrame] = []
    for history_path in sorted(
        (output_dir / "training_cache").glob("*__history.csv")
    ):
        stem = history_path.name.removesuffix("__history.csv")
        metadata = json.loads(
            (output_dir / "training_cache" / f"{stem}.json").read_text()
        )
        history = pd.read_csv(history_path)
        history["scenario"] = metadata["scenario"]
        history["variant"] = metadata["spec"]["name"]
        history["candidate_id"] = metadata["candidate_id"]
        history["run_seed"] = metadata["seed"]
        history["representation"] = metadata["spec"]["representation"]
        history["architecture"] = metadata["spec"]["architecture"]
        final_parts.append(history)
    final = pd.concat(final_parts, ignore_index=True, sort=False)
    final.to_csv(
        output_dir / "final_training_histories.csv", index=False
    )
    return selection, final


def write_confusions_and_failures(
    output_dir: Path, predictions: pd.DataFrame
) -> None:
    full = predictions[
        predictions["variant"] == "full_domain_aware"
    ].copy()
    labels = sorted(
        set(full["target_analyte"].astype(str))
        | set(full["predicted_analyte"].astype(str))
    )
    supported = full[full["supported"].astype(bool)]
    matrices: dict[str, Any] = {
        "labels": labels,
        "note": "Pooled matrices sum the three declared seeds; seed- and fold-specific matrices preserve the independent runs.",
        "pooled_across_seeds": {},
        "by_seed": {},
        "by_fold_and_seed": {},
    }
    for subset, frame in supported.groupby(
        "evaluation_subset", sort=True
    ):
        table = pd.crosstab(
            frame["target_analyte"].astype(str),
            frame["predicted_analyte"].astype(str),
        ).reindex(index=labels, columns=labels, fill_value=0)
        matrices["pooled_across_seeds"][str(subset)] = (
            table.astype(int).values.tolist()
        )
        matrices["by_seed"][str(subset)] = {}
        matrices["by_fold_and_seed"][str(subset)] = {}
        for seed, seed_frame in frame.groupby("declared_seed", sort=True):
            seed_table = pd.crosstab(
                seed_frame["target_analyte"].astype(str),
                seed_frame["predicted_analyte"].astype(str),
            ).reindex(index=labels, columns=labels, fill_value=0)
            matrices["by_seed"][str(subset)][str(int(seed))] = (
                seed_table.astype(int).values.tolist()
            )
        for (outer_fold, seed), run_frame in frame.groupby(
            ["outer_fold", "declared_seed"], sort=True
        ):
            run_table = pd.crosstab(
                run_frame["target_analyte"].astype(str),
                run_frame["predicted_analyte"].astype(str),
            ).reindex(index=labels, columns=labels, fill_value=0)
            key = f"outer_{int(outer_fold)}__seed_{int(seed)}"
            matrices["by_fold_and_seed"][str(subset)][key] = (
                run_table.astype(int).values.tolist()
            )
    classical.write_json(
        output_dir / "successor_confusion_matrices.json", matrices
    )
    failures = full[
        (~full["correct"].astype(bool))
        | (~full["supported"].astype(bool))
    ].copy()
    failures.to_csv(
        output_dir / "successor_failure_cases.csv", index=False
    )


def plot_diagnostics(
    output_dir: Path,
    selection_history: pd.DataFrame,
    stage2_selection: pd.DataFrame,
    outer: pd.DataFrame,
    deep_selective: pd.DataFrame,
    siamese_selective: pd.DataFrame,
    classical_bundle: Path,
) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    global_candidate = str(
        stage2_selection.loc[
            stage2_selection["scope"] == "global", "candidate_id"
        ].iloc[0]
    )
    convergence = selection_history[
        (selection_history["stage"] == "stage2")
        & (selection_history["candidate_id"] == global_candidate)
    ].dropna(subset=["validation_balanced_accuracy"])
    convergence_summary = (
        convergence.groupby("epoch")["validation_balanced_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    half = (
        1.96
        * convergence_summary["std"]
        / np.sqrt(convergence_summary["count"].clip(lower=1))
    )
    axes[0, 0].plot(
        convergence_summary["epoch"],
        convergence_summary["mean"],
        color=COLORS["Contrastive successor"],
        linewidth=1.3,
    )
    axes[0, 0].fill_between(
        convergence_summary["epoch"],
        convergence_summary["mean"] - half,
        convergence_summary["mean"] + half,
        color=COLORS["Contrastive successor"],
        alpha=0.18,
        linewidth=0,
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Validation balanced accuracy")
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].set_title("Nested-development convergence")

    variants = [
        "cross_entropy_only",
        "supervised_contrastive_only",
        "ce_plus_supervised_contrastive",
        "full_domain_aware",
    ]
    variant_labels = {
        "cross_entropy_only": "CE",
        "supervised_contrastive_only": "SupCon",
        "ce_plus_supervised_contrastive": "CE+SupCon",
        "full_domain_aware": "Full",
    }
    marker_cycle = ["o", "s", "^", "D"]
    strict = outer[
        (outer["evaluation_subset"] == "strict_core")
        & outer["variant"].isin(variants)
    ]
    for variant, marker in zip(variants, marker_cycle):
        frame = strict[strict["variant"] == variant]
        axes[0, 1].scatter(
            frame["different_minus_same_margin"],
            frame["embedding_effective_rank"],
            s=17,
            alpha=0.55,
            marker=marker,
            label=variant_labels[variant],
        )
    axes[0, 1].axhline(4.0, color="0.5", linestyle=":", linewidth=0.8)
    axes[0, 1].axvline(0.0, color="0.5", linestyle=":", linewidth=0.8)
    axes[0, 1].set_xlabel("Different − same analyte distance")
    axes[0, 1].set_ylabel("Embedding effective rank")
    axes[0, 1].set_title("Collapse and class geometry")
    axes[0, 1].legend(frameon=False, ncol=2, fontsize=6)

    full = outer[outer["variant"] == "full_domain_aware"]
    for subset, frame in full.groupby("evaluation_subset", sort=True):
        axes[1, 0].scatter(
            frame["instrument_probe_balanced_accuracy"],
            frame["balanced_accuracy"],
            color=COLORS[subset],
            s=17,
            alpha=0.55,
            label=subset.replace("_", " ").title(),
        )
    axes[1, 0].set_xlabel("Instrument-probe balanced accuracy")
    axes[1, 0].set_ylabel("Chemical balanced accuracy")
    axes[1, 0].set_xlim(0, 1.02)
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].set_title("Chemistry versus instrument information")
    axes[1, 0].legend(frameon=False, fontsize=6)

    classical_selective = pd.read_csv(
        classical_bundle / "selective_metrics.csv"
    )
    curves = [
        (
            "Classical",
            classical_selective[
                classical_selective["evaluation_subset"]
                == "field_quality_stress"
            ],
            "outer_fold",
            None,
        ),
        (
            "Siamese",
            siamese_selective[
                siamese_selective["evaluation_subset"]
                == "field_quality_stress"
            ],
            "outer_fold",
            "declared_seed",
        ),
        (
            "Contrastive successor",
            deep_selective[
                (deep_selective["variant"] == "full_domain_aware")
                & (
                    deep_selective["evaluation_subset"]
                    == "field_quality_stress"
                )
                & deep_selective["rejection_selected"].astype(bool)
            ],
            "outer_fold",
            "declared_seed",
        ),
    ]
    for name, frame, fold_column, seed_column in curves:
        if seed_column is not None:
            per_fold = (
                frame.groupby(
                    ["requested_coverage", fold_column], as_index=False
                )["selective_risk"]
                .mean()
            )
        else:
            per_fold = frame[
                ["requested_coverage", fold_column, "selective_risk"]
            ]
        summary = (
            per_fold.groupby("requested_coverage")["selective_risk"]
            .mean()
            .sort_index()
        )
        axes[1, 1].plot(
            summary.index,
            summary.values,
            marker="o",
            markersize=3,
            color=COLORS[name],
            label=name,
        )
    axes[1, 1].set_xlabel("Requested coverage")
    axes[1, 1].set_ylabel("Field-stress selective risk")
    axes[1, 1].set_ylim(0, 1.02)
    axes[1, 1].set_title("Locked abstention comparison")
    axes[1, 1].legend(frameon=False, fontsize=6)
    for label, axis in zip(("A", "B", "C", "D"), axes.flat):
        axis.text(
            -0.15,
            1.07,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
            va="top",
        )
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "training_diagnostics")


def plot_model_comparison(
    output_dir: Path,
    models: pd.DataFrame,
    deep_summary: pd.DataFrame,
    ood: pd.DataFrame,
) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    subsets = ["strict_core", "quality_pass", "field_quality_stress"]
    model_order = ["Classical", "Siamese", "Contrastive successor"]
    x = np.arange(len(subsets))
    offsets = [-0.18, 0.0, 0.18]
    for model, offset in zip(model_order, offsets):
        table = models[
            (models["model"] == model)
            & (models["metric"] == "balanced_accuracy")
        ].set_index("evaluation_subset").loc[subsets]
        axes[0].errorbar(
            x + offset,
            table["mean"],
            yerr=table["ci95_half_width"],
            fmt="o",
            capsize=2.5,
            color=COLORS[model],
            label=model,
            markersize=4,
        )
    axes[0].set_xticks(
        x, ["Strict", "Quality", "Stress"], rotation=15, ha="right"
    )
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].set_title("Locked model comparison")
    axes[0].legend(frameon=False, fontsize=6.5)

    ablation_names = [
        "cross_entropy_only",
        "supervised_contrastive_only",
        "ce_plus_supervised_contrastive",
        "full_domain_aware",
    ]
    labels = ["CE", "SupCon", "CE+SupCon", "Full"]
    for subset in subsets:
        table = deep_summary[
            (deep_summary["evaluation_subset"] == subset)
            & (deep_summary["metric"] == "balanced_accuracy")
            & deep_summary["variant"].isin(ablation_names)
        ].set_index("variant").loc[ablation_names]
        axes[1].plot(
            np.arange(len(labels)),
            table["mean"],
            marker="o",
            color=COLORS[subset],
            label=subset.replace("_", " ").title(),
        )
    axes[1].set_xticks(
        np.arange(len(labels)), labels, rotation=20, ha="right"
    )
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_title("Objective ablation")
    axes[1].legend(frameon=False, fontsize=6)

    ood_fold = (
        ood.groupby(["score", "outer_fold"], as_index=False)["ood_auroc"]
        .mean()
    )
    scores = sorted(ood_fold["score"].unique())
    for index, score in enumerate(scores):
        values = ood_fold.loc[
            ood_fold["score"] == score, "ood_auroc"
        ].astype(float)
        jitter = np.linspace(-0.05, 0.05, len(values))
        axes[2].scatter(
            np.full(len(values), index) + jitter,
            values,
            color="#56B4E9",
            alpha=0.55,
            s=13,
        )
        summary = mean_ci(values)
        axes[2].errorbar(
            index,
            summary["mean"],
            yerr=summary["ci95_half_width"],
            fmt="D",
            color="black",
            capsize=2.5,
            markersize=4,
        )
    axes[2].set_xticks(
        np.arange(len(scores)),
        [score.replace("_", "\n") for score in scores],
        rotation=15,
        ha="right",
    )
    axes[2].set_ylim(0, 1.02)
    axes[2].set_ylabel("Field-stress OOD AUROC")
    axes[2].set_title("Quality rejection")
    axes[2].axhline(0.5, color="0.55", linestyle=":", linewidth=0.8)
    for label, axis in zip(("A", "B", "C"), axes):
        axis.text(
            -0.16,
            1.07,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
            va="top",
        )
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "model_comparison")


def format_value(
    models: pd.DataFrame, model: str, subset: str
) -> str:
    row = models[
        (models["model"] == model)
        & (models["evaluation_subset"] == subset)
        & (models["metric"] == "balanced_accuracy")
    ].iloc[0]
    return f'{row["mean"]:.3f} ± {row["ci95_half_width"]:.3f}'


def write_report(
    output_dir: Path,
    models: pd.DataFrame,
    gate: dict[str, Any],
    outer: pd.DataFrame,
    siamese_diagnostics: pd.DataFrame,
    ood: pd.DataFrame,
    attribution: pd.DataFrame,
    negative: pd.DataFrame,
    stage1: pd.DataFrame,
    stage2: pd.DataFrame,
    classical_champions: dict[str, Any],
    preprocessing_summary: pd.DataFrame,
) -> None:
    status = "SUPPORTED" if gate["successor_supported"] else "NOT SUPPORTED"
    full_outer = (
        outer.loc[
            outer["variant"] == "full_domain_aware",
            [
                "outer_fold",
                "representation",
                "architecture",
                "embedding_dimension",
                "total_parameters",
            ],
        ]
        .drop_duplicates()
        .sort_values("outer_fold")
    )
    outer_configuration_counts = (
        full_outer.groupby(
            [
                "representation",
                "architecture",
                "embedding_dimension",
                "total_parameters",
            ],
            as_index=False,
        )["outer_fold"]
        .nunique()
        .sort_values(
            ["outer_fold", "representation", "embedding_dimension"],
            ascending=[False, True, True],
        )
    )
    outer_configuration_text = "; ".join(
        (
            f"`{row.representation}` + "
            f"`{row.architecture}{int(row.embedding_dimension)}` in "
            f"{int(row.outer_fold)}/5 folds "
            f"({int(row.total_parameters):,} total parameters)"
        )
        for row in outer_configuration_counts.itertuples()
    )
    global_configuration = stage2.iloc[0]
    lines = [
        "# Supervised-contrastive NATO SERS experiment",
        "",
        "## Terminal decision",
        "",
        f"**Successor promotion: {status}.**",
        "",
        "This experiment tests domain-robust classification and abstention. It does not claim physical denoising or chemical/nuisance disentanglement.",
        "",
        "## Locked outer results",
        "",
        "| Model | Strict BA | Quality BA | Field-stress BA |",
        "|---|---:|---:|---:|",
    ]
    for model in ("Classical", "Siamese", "Contrastive successor"):
        lines.append(
            f"| {model} | {format_value(models, model, 'strict_core')} | "
            f"{format_value(models, model, 'quality_pass')} | "
            f"{format_value(models, model, 'field_quality_stress')} |"
        )
    lines.extend(
        [
            "",
            "Uncertainty is a 95% interval over five outer master-group folds after averaging repeated seeds. Spectral rows are not treated as independent uncertainty units.",
            "",
            "## Promotion gates",
            "",
            f"- No material strict/quality degradation: **{gate['gate_no_clean_degradation']}**.",
            f"- Material held-domain and/or stress advantage: **{gate['gate_material_shift_advantage']}**.",
            f"- Beats the Siamese control in at least two of three seed directions: **{gate['gate_reproducibly_beats_siamese']}** ({gate['positive_seed_count']}/3).",
            f"- Primary held-instrument plus new-sample advantage over classical: {gate['primary_domain_advantage']:.3f} across {gate['primary_domain_pair_count']} supported domain/cohort pairs.",
            f"- Primary held-instrument plus new-sample advantage over the historical Siamese: {gate['primary_domain_advantage_vs_siamese']:.3f}.",
            f"- Field-stress selective-accuracy advantage at 80% requested coverage: {gate['field_stress_selective_advantage_at_80_percent']:.3f} across {gate['field_stress_selective_pair_count']} outer folds.",
            f"- Full-coverage field-stress balanced-accuracy difference (secondary): {gate['field_stress_full_coverage_advantage']:.3f}.",
            "",
            "## Selected model",
            "",
            f"Global held-domain configuration: Stage 1 selected `{stage1.iloc[0]['representation']}` with supervised-contrastive weight {stage1.iloc[0]['supervised_contrastive_weight']} and pair-margin weight {stage1.iloc[0]['pair_margin_weight']}; Stage 2 selected `{global_configuration['architecture']}` with {int(global_configuration['embedding_dimension'])} embedding dimensions ({int(global_configuration['total_parameters']):,} total parameters).",
            f"Nested outer configurations: {outer_configuration_text}. Each outer fold used only its own inner-fold selection.",
            "",
            "## Representation and Siamese-control diagnostics",
            "",
            f"- Successor different-minus-same-analyte distance margin: {outer.loc[outer['variant'] == 'full_domain_aware', 'different_minus_same_margin'].mean():.3f}; historical Siamese: {siamese_diagnostics['different_minus_same_margin'].mean():.3f}.",
            f"- Successor embedding effective rank: {outer.loc[outer['variant'] == 'full_domain_aware', 'embedding_effective_rank'].mean():.2f}; historical Siamese: {siamese_diagnostics['embedding_effective_rank'].mean():.2f}.",
            f"- Historical Siamese leave-one-master-out analyte probe balanced accuracy: {siamese_diagnostics['analyte_heldout_master_probe_balanced_accuracy'].mean():.3f} where supported.",
            f"- Historical Siamese cross-fitted correctness-confidence ECE10: {siamese_diagnostics['correctness_ece_10'].mean():.3f}. This is correctness calibration from nearest-prototype distance, not multiclass probability calibration.",
            f"- Historical Siamese encoder parameters: {int(siamese_diagnostics['encoder_parameters'].iloc[0]):,}. Successor parameter counts are reported per nested/global configuration above because architecture selection was fold-specific.",
            "",
            "## Frozen preprocessing sensitivity",
            "",
            "Each representation below uses all five outer folds: when a representation was selected it contributes the full-model row; otherwise it contributes the registered sensitivity row. This prevents partial-fold aggregation.",
            "",
            "| Representation | Strict BA | Quality BA | Stress BA |",
            "|---|---:|---:|---:|",
        ]
    )
    for representation in (
        "arpls_minmax",
        "minimal_minmax",
        "derivative_1",
    ):
        values = {}
        for subset in (
            "strict_core",
            "quality_pass",
            "field_quality_stress",
        ):
            row = preprocessing_summary[
                (preprocessing_summary["representation"] == representation)
                & (
                    preprocessing_summary["evaluation_subset"]
                    == subset
                )
                & (preprocessing_summary["metric"] == "balanced_accuracy")
            ].iloc[0]
            values[subset] = (
                f"{row['mean']:.3f} ± {row['ci95_half_width']:.3f}"
            )
        lines.append(
            f"| `{representation}` | {values['strict_core']} | "
            f"{values['quality_pass']} | "
            f"{values['field_quality_stress']} |"
        )
    lines.extend(
        [
            "",
            "## Direct answers",
            "",
            f"- **Which classical model won?** Strict: `{classical_champions['strict_core']['model_family']}` on `{classical_champions['strict_core']['representation']}` with `{classical_champions['strict_core']['parameters_json']}`. Quality/stress-development: `{classical_champions['quality_pass']['model_family']}` on `{classical_champions['quality_pass']['representation']}` with `{classical_champions['quality_pass']['parameters_json']}`.",
            f"- **Does this dataset support preferring the tested deep model?** {'Yes, within the registered transfer and clean-performance gates.' if gate['successor_supported'] else 'No. The registered evidence does not justify replacing the classical champion with this successor.'}",
            f"- **Did supervised contrastive learning improve the Siamese control?** {'Yes under the registered two-of-three seed rule.' if gate['gate_reproducibly_beats_siamese'] else 'No reproducible improvement under the registered two-of-three seed rule.'}",
            f"- **Did the gain transfer?** Primary domain difference versus classical {gate['primary_domain_advantage']:.3f}; field-stress selective difference at 80% coverage {gate['field_stress_selective_advantage_at_80_percent']:.3f}.",
            f"- **Publication interpretation:** {'A positive cross-instrument classification-and-abstention result is supported, without a disentanglement claim.' if gate['successor_supported'] else 'The publishable result is a leakage-safe grouped benchmark and mechanistic negative/ablation study, not a claimed invariant or disentangled representation.'}",
            "- **What crossed data are still needed?** More independent master samples for every analyte; the same physical samples measured across every instrument and sensor family; balanced sensor-family support; raw vendor spectra before proprietary baseline removal; reference Raman/SERS spectra for chemicals and blank substrates; and controlled concentration, substrate-lot, acquisition-time, and environmental replicates.",
            "",
            "## OOD and attribution evidence",
            "",
        ]
    )
    for score, group in ood.groupby("score"):
        lines.append(
            f"- `{score}`: field-stress AUROC {group['ood_auroc'].mean():.3f}, AUPRC {group['ood_auprc'].mean():.3f} across folds and seeds."
        )
    selected_counts = (
        ood[["outer_fold", "selected_rejection_score"]]
        .drop_duplicates()["selected_rejection_score"]
        .value_counts()
    )
    lines.append(
        "- Development-selected rejection scores by outer fold: "
        + ", ".join(
            f"`{name}` in {int(count)} fold(s)"
            for name, count in selected_counts.items()
        )
        + "."
    )
    lines.extend(
        [
            f"- Same-master cross-instrument attribution Jaccard: {attribution['same_master_cross_instrument_attribution_jaccard'].mean():.3f}.",
            f"- Same-analyte, different-master attribution Jaccard: {attribution['same_analyte_different_master_attribution_jaccard'].mean():.3f}.",
            "",
            "## Negative controls",
            "",
        ]
    )
    for control, group in negative.groupby("control"):
        lines.append(
            f"- `{control}`: mean balanced accuracy {group['balanced_accuracy'].mean():.3f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The full model is promoted only if every registered gate above is satisfied. Otherwise, the result remains a controlled comparison showing which objective or domain relationship helped, without reinterpreting a failure as invariance or disentanglement. Held-sensor values with few supported classes must be read together with their support counts.",
            "",
            "The held-domain models never train on their test instrument or sensor, and held-domain outcomes never select the configuration. However, the global architecture was chosen by master-group CV across the archive's available domain identities, matching the classical benchmark. This is locked leave-one-domain-out transfer, not a substitute for a genuinely external instrument acquisition.",
            "",
            "## Main figure",
            "",
            "- `figures/model_comparison.pdf` and 600-DPI PNG: locked model comparison, objective ablation, and field-stress rejection.",
            "- `figures/training_diagnostics.pdf` and 600-DPI PNG: nested-development convergence, collapse geometry, chemistry-versus-instrument probes, and locked risk–coverage curves.",
            "- `successor_confusion_matrices.json`: pooled, seed-specific, and fold/seed-specific chemical confusions.",
            "- `successor_failure_cases.csv`: every unsupported or incorrect full-successor outer prediction.",
            "- `rejection_decisions_at_80.csv`: accepted and rejected examples at the locked 80% coverage endpoint.",
            "- `siamese_control_diagnostics.csv` and `siamese_control_failures.csv`: reconstructed geometry, collapse, grouped-probe, correctness-calibration, and failure evidence for the immutable historical control.",
        ]
    )
    (output_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")


def write_hashes(output_dir: Path) -> None:
    excluded = {
        "artifact_hashes.json",
        "validation_report.json",
        "clean_rebuild_comparison.json",
    }
    hashes = {
        str(path.relative_to(output_dir)): baseline.sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
        and path.name not in excluded
        and "training_cache" not in path.parts
        and "selection_shards" not in path.parts
    }
    classical.write_json(output_dir / "artifact_hashes.json", hashes)


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text())
    outer = pd.read_csv(args.output_dir / "outer_metrics.csv")
    domain = pd.read_csv(args.output_dir / "domain_metrics.csv")
    siamese = pd.read_csv(args.output_dir / "siamese_control_metrics.csv")
    siamese_diagnostics = pd.read_csv(
        args.output_dir / "siamese_control_diagnostics.csv"
    )
    ood = pd.read_csv(args.output_dir / "outer_ood_metrics.csv")
    attribution = pd.read_csv(args.output_dir / "attribution_metrics.csv")
    negative = pd.read_csv(args.output_dir / "negative_control_metrics.csv")
    stage1 = pd.read_csv(args.output_dir / "stage1_selection.csv")
    stage2 = pd.read_csv(args.output_dir / "stage2_selection.csv")
    classical_champions = json.loads(
        (args.classical_bundle / "global_champions.json").read_text()
    )
    predictions = pd.read_csv(
        args.output_dir / "outer_predictions.csv"
    )
    deep_selective = pd.read_csv(
        args.output_dir / "outer_selective_metrics.csv"
    )
    siamese_selective = pd.read_csv(
        args.output_dir / "siamese_control_selective_metrics.csv"
    )
    deep_summary = deep_outer_summary(outer)
    preprocessing_summary = preprocessing_sensitivity_summary(outer)
    siamese_summary = prior_siamese_summary(siamese)
    classical_frame = classical_summary(args.classical_bundle)
    models = model_summary(deep_summary, siamese_summary, classical_frame)
    gate = paired_gate(
        args.output_dir,
        protocol,
        outer,
        domain,
        siamese,
        args.classical_bundle,
    )
    deep_summary.to_csv(
        args.output_dir / "outer_uncertainty_summary.csv", index=False
    )
    models.to_csv(
        args.output_dir / "locked_model_comparison.csv", index=False
    )
    preprocessing_summary.to_csv(
        args.output_dir / "preprocessing_sensitivity_summary.csv",
        index=False,
    )
    classical.write_json(args.output_dir / "terminal_decision.json", gate)
    write_confusions_and_failures(args.output_dir, predictions)
    selection_history, _ = consolidate_training_histories(
        args.output_dir
    )
    plot_model_comparison(args.output_dir, models, deep_summary, ood)
    plot_diagnostics(
        args.output_dir,
        selection_history,
        stage2,
        outer,
        deep_selective,
        siamese_selective,
        args.classical_bundle,
    )
    global_stage1 = stage1[stage1["scope"] == "global"]
    global_stage2 = stage2[stage2["scope"] == "global"]
    write_report(
        args.output_dir,
        models,
        gate,
        outer,
        siamese_diagnostics,
        ood,
        attribution,
        negative,
        global_stage1,
        global_stage2,
        classical_champions,
        preprocessing_summary,
    )
    write_hashes(args.output_dir)
    print(
        json.dumps(
            {
                "status": "finalized",
                "successor_supported": gate["successor_supported"],
                "strict": format_value(
                    models, "Contrastive successor", "strict_core"
                ),
                "quality": format_value(
                    models, "Contrastive successor", "quality_pass"
                ),
                "stress": format_value(
                    models,
                    "Contrastive successor",
                    "field_quality_stress",
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
