#!/usr/bin/env python3
"""Finalize, plot, and document the NATO SERS classical benchmark v2."""

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

import sers_classical_benchmark_common as common


OKABE_ITO = {
    "strict_core": "#0072B2",
    "quality_pass": "#009E73",
    "field_quality_stress": "#D55E00",
    "instrument": "#56B4E9",
    "sensor_family": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Workspace/sers_classical_benchmark/classical_benchmark_v2"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/sers_classical_benchmark_v2.json"),
    )
    return parser.parse_args()


def mean_ci(values: pd.Series) -> dict[str, float | int]:
    clean = values.dropna().astype(float)
    n = len(clean)
    mean = float(clean.mean()) if n else np.nan
    sd = float(clean.std(ddof=1)) if n > 1 else np.nan
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


def outer_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "balanced_accuracy",
        "macro_f1",
        "accuracy",
        "negative_log_likelihood",
        "brier_multiclass",
        "expected_calibration_error_10",
    ]
    for keys, group in frame.groupby(
        ["training_subset", "evaluation_subset"], sort=True
    ):
        training_subset, evaluation_subset = keys
        for metric in metrics:
            rows.append(
                {
                    "training_subset": training_subset,
                    "evaluation_subset": evaluation_subset,
                    "metric": metric,
                    "independent_unit": "outer master-group fold",
                    **mean_ci(group[metric]),
                }
            )
    return pd.DataFrame(rows)


def domain_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["subset", "protocol", "domain_type"], sort=True
    ):
        subset, domain_protocol, domain_type = keys
        for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
            rows.append(
                {
                    "subset": subset,
                    "protocol": domain_protocol,
                    "domain_type": domain_type,
                    "metric": metric,
                    "independent_unit": "heldout domain",
                    **mean_ci(group[metric]),
                }
            )
    return pd.DataFrame(rows)


def learning_summary(frame: pd.DataFrame) -> pd.DataFrame:
    # Repeated subsamples share an outer test fold. Average repeats first so
    # the uncertainty unit remains the independent outer master-group fold.
    per_fold = (
        frame.groupby(
            ["subset", "training_group_fraction", "outer_fold"], as_index=False
        )
        .agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            n_train_groups=("n_train_groups", "mean"),
        )
    )
    rows: list[dict[str, Any]] = []
    for keys, group in per_fold.groupby(
        ["subset", "training_group_fraction"], sort=True
    ):
        subset, fraction = keys
        for metric in ("balanced_accuracy", "macro_f1"):
            rows.append(
                {
                    "subset": subset,
                    "training_group_fraction": fraction,
                    "mean_train_groups": float(group["n_train_groups"].mean()),
                    "metric": metric,
                    "independent_unit": "outer master-group fold after averaging subsample seeds",
                    **mean_ci(group[metric]),
                }
            )
    return pd.DataFrame(rows)


def configure_plot_style() -> None:
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
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_main_summary(
    output_dir: Path,
    outer: pd.DataFrame,
    summary: pd.DataFrame,
    learning: pd.DataFrame,
    learning_raw: pd.DataFrame,
) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    order = ["strict_core", "quality_pass", "field_quality_stress"]
    labels = ["Strict core", "Quality pass", "Field stress"]
    x = np.arange(len(order))
    for index, subset in enumerate(order):
        values = outer.loc[
            outer["evaluation_subset"] == subset, "balanced_accuracy"
        ].astype(float)
        jitter = np.linspace(-0.08, 0.08, len(values))
        axes[0].scatter(
            np.full(len(values), index) + jitter,
            values,
            s=18,
            color=OKABE_ITO[subset],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        record = summary[
            (summary["evaluation_subset"] == subset)
            & (summary["metric"] == "balanced_accuracy")
        ].iloc[0]
        axes[0].errorbar(
            index,
            record["mean"],
            yerr=record["ci95_half_width"],
            fmt="D",
            color="black",
            markersize=4,
            capsize=3,
            linewidth=1,
            zorder=4,
        )
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].set_title("Nested outer-fold performance")
    axes[0].axhline(1 / 7, color="0.55", linestyle=":", linewidth=0.8)
    axes[0].text(
        0.02,
        1 / 7 + 0.02,
        "Seven-class chance",
        color="0.4",
        transform=axes[0].get_yaxis_transform(),
        fontsize=6.5,
    )

    markers = {"strict_core": "o", "quality_pass": "s"}
    for subset in ("strict_core", "quality_pass"):
        table = learning[
            (learning["subset"] == subset)
            & (learning["metric"] == "balanced_accuracy")
        ].sort_values("training_group_fraction")
        axes[1].errorbar(
            table["mean_train_groups"],
            table["mean"],
            yerr=table["ci95_half_width"],
            label=subset.replace("_", " ").title(),
            marker=markers[subset],
            color=OKABE_ITO[subset],
            capsize=3,
            linewidth=1.4,
            markersize=4,
        )
        fold_points = (
            learning_raw.groupby(
                ["subset", "training_group_fraction", "outer_fold"],
                as_index=False,
            )
            .agg(
                balanced_accuracy=("balanced_accuracy", "mean"),
                n_train_groups=("n_train_groups", "mean"),
            )
        )
        fold_points = fold_points[fold_points["subset"] == subset]
        axes[1].scatter(
            fold_points["n_train_groups"],
            fold_points["balanced_accuracy"],
            color=OKABE_ITO[subset],
            alpha=0.22,
            s=10,
        )
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Independent training master samples")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_title("Master-sample learning curve")
    axes[1].legend(frameon=False)
    for label, axis in zip(("A", "B"), axes):
        axis.text(
            -0.13,
            1.06,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
            va="top",
        )
    fig.tight_layout()
    save_figure(
        fig, output_dir / "figures" / "classical_benchmark_summary"
    )


def plot_selective_domain(
    output_dir: Path,
    selective: pd.DataFrame,
    domain: pd.DataFrame,
) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for subset in ("strict_core", "quality_pass", "field_quality_stress"):
        table = selective[
            selective["evaluation_subset"] == subset
        ].groupby("requested_coverage", as_index=False).agg(
            realized_coverage=("realized_coverage", "mean"),
            accuracy=("accuracy", "mean"),
            sd=("accuracy", "std"),
        )
        axes[0].plot(
            table["realized_coverage"],
            table["accuracy"],
            marker="o",
            markersize=3.5,
            color=OKABE_ITO[subset],
            label=subset.replace("_", " ").title(),
        )
    axes[0].set_xlim(0.48, 1.02)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("Coverage retained")
    axes[0].set_ylabel("Selective accuracy")
    axes[0].set_title("Confidence-based abstention")
    axes[0].legend(frameon=False)

    domain_order = [
        ("instrument", "domain_only"),
        ("instrument", "domain_and_sample"),
        ("sensor_family", "domain_only"),
        ("sensor_family", "domain_and_sample"),
    ]
    positions = np.arange(len(domain_order))
    for subset, offset in (("strict_core", -0.12), ("quality_pass", 0.12)):
        means = []
        lower = []
        upper = []
        for domain_type, domain_protocol in domain_order:
            values = domain.loc[
                (domain["subset"] == subset)
                & (domain["domain_type"] == domain_type)
                & (domain["protocol"] == domain_protocol),
                "balanced_accuracy",
            ].dropna().astype(float)
            mean = float(values.mean())
            means.append(mean)
            lower.append(mean - float(values.min()))
            upper.append(float(values.max()) - mean)
            jitter = np.linspace(-0.035, 0.035, len(values))
            category_index = domain_order.index((domain_type, domain_protocol))
            axes[1].scatter(
                np.full(len(values), category_index + offset) + jitter,
                values,
                s=12,
                color=OKABE_ITO[subset],
                alpha=0.35,
                edgecolor="none",
                zorder=2,
            )
        axes[1].errorbar(
            positions + offset,
            means,
            yerr=np.vstack([lower, upper]),
            fmt="D",
            capsize=3,
            color=OKABE_ITO[subset],
            label=subset.replace("_", " ").title(),
            markersize=4,
            linewidth=1,
            zorder=3,
        )
    axes[1].set_xticks(
        positions,
        [
            "Instrument\nDomain only",
            "Instrument\n+ new sample",
            "Sensor\nDomain only",
            "Sensor\n+ new sample",
        ],
        rotation=15,
        ha="right",
    )
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Supported-class balanced accuracy")
    axes[1].set_title("Held-domain generalization\n(mean and observed range)")
    axes[1].legend(frameon=False)
    for label, axis in zip(("A", "B"), axes):
        axis.text(
            -0.13,
            1.06,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
            va="top",
        )
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "selective_domain_summary")


def format_metric(
    summary: pd.DataFrame, subset: str, metric: str = "balanced_accuracy"
) -> str:
    row = summary[
        (summary["evaluation_subset"] == subset)
        & (summary["metric"] == metric)
    ].iloc[0]
    return f'{row["mean"]:.3f} ± {row["ci95_half_width"]:.3f}'


def write_report(
    output_dir: Path,
    champions: dict[str, Any],
    summary: pd.DataFrame,
    domain_summary_frame: pd.DataFrame,
    learning: pd.DataFrame,
    ood: pd.DataFrame,
    negative: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    strict = champions["strict_core"]
    quality = champions["quality_pass"]
    strict_learning = learning[
        (learning["subset"] == "strict_core")
        & (learning["metric"] == "balanced_accuracy")
    ].sort_values("training_group_fraction")
    first = strict_learning.iloc[0]
    last = strict_learning.iloc[-1]
    domain_ba = domain_summary_frame[
        domain_summary_frame["metric"] == "balanced_accuracy"
    ]
    calibration_nll = (
        calibration.groupby(
            ["evaluation_subset", "calibration"], as_index=False
        )["negative_log_likelihood"]
        .mean()
        .pivot(
            index="evaluation_subset",
            columns="calibration",
            values="negative_log_likelihood",
        )
    )
    lines = [
        "# NATO SERS classical benchmark v2",
        "",
        "## Status",
        "",
        "This bundle is the locked classical foundation for the supervised-contrastive CNN experiment. It evaluates chemical classification and selective prediction; it does not claim physical chemical/nuisance disentanglement.",
        "",
        "## Selected configurations",
        "",
        f"- Strict-core champion: `{strict['model_family']}` on `{strict['representation']}` with `{strict['parameters_json']}`.",
        f"- Quality-pass champion: `{quality['model_family']}` on `{quality['representation']}` with `{quality['parameters_json']}`.",
        "- Selection used 20 nested inner folds and did not use outer, field-stress, or held-domain outcomes.",
        "",
        "## Master-group outer performance",
        "",
        "| Evaluation | Balanced accuracy, mean ± 95% CI half-width |",
        "|---|---:|",
        f"| Strict core | {format_metric(summary, 'strict_core')} |",
        f"| Quality pass | {format_metric(summary, 'quality_pass')} |",
        f"| Field-quality stress | {format_metric(summary, 'field_quality_stress')} |",
        "",
        "The uncertainty unit is the outer master-sample fold (n=5), not an individual spectrum.",
        "",
        "## Data adequacy",
        "",
        f"Strict-core balanced accuracy changed from {first['mean']:.3f} using about {first['mean_train_groups']:.1f} training master samples to {last['mean']:.3f} using about {last['mean_train_groups']:.1f}. The learning curve is the evidence used to judge whether added independent samples are likely to matter more than model capacity.",
        "",
        "## Held-domain evaluation",
        "",
        "| Subset | Protocol | Domain | Mean supported-class BA | 95% CI half-width | Held domains |",
        "|---|---|---|---:|---:|---:|",
    ]
    for _, row in domain_ba.sort_values(
        ["subset", "domain_type", "protocol"]
    ).iterrows():
        lines.append(
            f"| {row['subset']} | {row['protocol']} | {row['domain_type']} | "
            f"{row['mean']:.3f} | {row['ci95_half_width']:.3f} | "
            f"{int(row['n_independent_units'])} |"
        )
    lines.extend(
        [
            "",
            "Sensor-family confidence intervals can be extremely wide because only a few held sensor families retain supported analytes. These intervals must not be read as precision estimates from hundreds of independent spectra.",
            "",
            "## Calibration, abstention, and field stress",
            "",
            f"The mean field-stress OOD AUROC from one minus calibrated maximum probability was {ood['ood_auroc'].mean():.3f}; mean AUPRC was {ood['ood_auprc'].mean():.3f}. These values measure stress detection, not chemical classification.",
        ]
    )
    for subset in calibration_nll.index:
        if {
            "uncalibrated",
            "temperature_scaled",
        }.issubset(calibration_nll.columns):
            lines.append(
                f"- {subset}: mean NLL {calibration_nll.loc[subset, 'uncalibrated']:.3f} before and {calibration_nll.loc[subset, 'temperature_scaled']:.3f} after temperature scaling."
            )
    lines.extend(
        [
            "",
            "## Negative control",
            "",
            f"Master-group label permutation produced mean outer balanced accuracy {negative['balanced_accuracy'].mean():.3f} (maximum {negative['balanced_accuracy'].max():.3f}), compared with seven-class chance of 0.143.",
            "",
            "## Decision for the next stage",
            "",
            "The selected classical configurations, fold predictions, calibration temperatures, held-domain results, and learning curves are now the immutable comparison bar. The supervised-contrastive CNN must be trained on these same master-group partitions and is useful only if it improves the current Siamese control and offers a reproducible held-domain and/or field-stress selective advantage without materially degrading strict/quality performance.",
            "",
            "## Figures",
            "",
            "- `figures/classical_benchmark_summary.pdf`: grouped outer performance and master-sample learning curves.",
            "- `figures/selective_domain_summary.pdf`: abstention and held-domain results.",
            "",
            "Outer and learning-curve error bars are 95% intervals over master-group folds. Held-domain points show individual domains with mean and observed range because sensor-family counts are too small for stable visual intervals. Vector PDF and 600-DPI PNG exports use a colorblind-safe palette and redundant markers.",
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
        str(path.relative_to(output_dir)): common.sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
        and path.name not in excluded
        and "search_shards" not in path.parts
    }
    common.write_json(output_dir / "artifact_hashes.json", hashes)


def write_confusion_matrices(
    output_dir: Path, predictions: pd.DataFrame
) -> None:
    labels = sorted(
        set(predictions["target_analyte"].astype(str))
        | set(predictions["predicted_analyte"].astype(str))
    )
    matrices: dict[str, Any] = {"labels": labels, "pooled": {}, "folds": {}}
    supported = predictions[predictions["supported"].astype(bool)]
    for subset, frame in supported.groupby(
        "evaluation_subset", sort=True
    ):
        table = pd.crosstab(
            frame["target_analyte"].astype(str),
            frame["predicted_analyte"].astype(str),
        ).reindex(index=labels, columns=labels, fill_value=0)
        matrices["pooled"][str(subset)] = table.astype(int).values.tolist()
        matrices["folds"][str(subset)] = {}
        for outer_fold, fold_frame in frame.groupby("outer_fold", sort=True):
            fold_table = pd.crosstab(
                fold_frame["target_analyte"].astype(str),
                fold_frame["predicted_analyte"].astype(str),
            ).reindex(index=labels, columns=labels, fill_value=0)
            matrices["folds"][str(subset)][str(int(outer_fold))] = (
                fold_table.astype(int).values.tolist()
            )
    common.write_json(output_dir / "confusion_matrices.json", matrices)


def main() -> None:
    args = parse_args()
    common.load_protocol(args.protocol)
    outer = pd.read_csv(args.output_dir / "outer_metrics.csv")
    domain = pd.read_csv(args.output_dir / "domain_metrics.csv")
    learning_raw = pd.read_csv(
        args.output_dir / "learning_curve_metrics.csv"
    )
    selective = pd.read_csv(args.output_dir / "selective_metrics.csv")
    ood = pd.read_csv(args.output_dir / "field_stress_ood_metrics.csv")
    negative = pd.read_csv(
        args.output_dir / "negative_control_metrics.csv"
    )
    calibration = pd.read_csv(
        args.output_dir / "calibration_metrics.csv"
    )
    predictions = pd.read_csv(args.output_dir / "predictions.csv")
    champions = json.loads(
        (args.output_dir / "global_champions.json").read_text()
    )
    summary = outer_summary(outer)
    domains = domain_summary(domain)
    learning = learning_summary(learning_raw)
    summary.to_csv(args.output_dir / "uncertainty_summary.csv", index=False)
    domains.to_csv(args.output_dir / "domain_summary.csv", index=False)
    learning.to_csv(
        args.output_dir / "learning_curve_summary.csv", index=False
    )
    write_confusion_matrices(args.output_dir, predictions)
    plot_main_summary(
        args.output_dir, outer, summary, learning, learning_raw
    )
    plot_selective_domain(args.output_dir, selective, domain)
    write_report(
        args.output_dir,
        champions,
        summary,
        domains,
        learning,
        ood,
        negative,
        calibration,
    )
    write_hashes(args.output_dir)
    print(
        json.dumps(
            {
                "status": "finalized",
                "strict_balanced_accuracy": format_metric(
                    summary, "strict_core"
                ),
                "quality_balanced_accuracy": format_metric(
                    summary, "quality_pass"
                ),
                "stress_balanced_accuracy": format_metric(
                    summary, "field_quality_stress"
                ),
                "hashed_artifacts": len(
                    json.loads(
                        (args.output_dir / "artifact_hashes.json").read_text()
                    )
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
