#!/usr/bin/env python3
"""Summarize and report the NATO SERS random-forest addendum."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sers_random_forest_common as common


METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "accuracy",
    "negative_log_likelihood",
    "brier_multiclass",
    "expected_calibration_error_10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Workspace/sers_random_forest_addendum/random_forest_v1"
        ),
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path(
            "Workspace/sers_supervised_contrastive/contrastive_v1/"
            "locked_model_comparison.csv"
        ),
    )
    return parser.parse_args()


def mean_ci(values: pd.Series) -> dict[str, float | int]:
    clean = values.dropna().astype(float)
    n = len(clean)
    mean = float(clean.mean()) if n else np.nan
    sd = float(clean.std(ddof=1)) if n > 1 else 0.0
    return {
        "mean": mean,
        "sd": sd,
        "ci95_half_width": float(1.96 * sd / math.sqrt(n)) if n else np.nan,
        "n_independent_units": n,
        "minimum": float(clean.min()) if n else np.nan,
        "maximum": float(clean.max()) if n else np.nan,
    }


def outer_summary(frame: pd.DataFrame) -> pd.DataFrame:
    # Forest seeds are technical repeats; physical-master folds are independent.
    fold_means = (
        frame.groupby(
            ["training_subset", "evaluation_subset", "outer_fold"],
            as_index=False,
        )[list(METRICS)]
        .mean()
    )
    rows: list[dict[str, Any]] = []
    for (training, evaluation), group in fold_means.groupby(
        ["training_subset", "evaluation_subset"], sort=True
    ):
        for metric in METRICS:
            rows.append(
                {
                    "training_subset": training,
                    "evaluation_subset": evaluation,
                    "metric": metric,
                    "independent_unit": "master-group outer fold",
                    **mean_ci(group[metric]),
                }
            )
    return pd.DataFrame(rows)


def domain_summary(frame: pd.DataFrame) -> pd.DataFrame:
    domain_means = (
        frame.groupby(
            ["subset", "protocol", "domain_type", "heldout_domain"],
            as_index=False,
        )[list(METRICS)]
        .mean()
    )
    rows: list[dict[str, Any]] = []
    for keys, group in domain_means.groupby(
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


def comparison_table(
    outer: pd.DataFrame, comparison_path: Path
) -> pd.DataFrame:
    prior = pd.read_csv(comparison_path)
    prior = prior[
        prior["metric"].isin(["balanced_accuracy", "macro_f1"])
    ].copy()
    rf = outer[
        outer["metric"].isin(["balanced_accuracy", "macro_f1"])
    ][
        [
            "evaluation_subset",
            "metric",
            "mean",
            "sd",
            "ci95_half_width",
            "n_independent_units",
        ]
    ].copy()
    rf.insert(0, "model", "Random forest")
    return pd.concat([prior, rf], ignore_index=True).sort_values(
        ["evaluation_subset", "metric", "model"], kind="stable"
    )


def selection_summary(selection: pd.DataFrame) -> pd.DataFrame:
    parameters = selection["parameters_json"].map(json.loads)
    expanded = pd.json_normalize(parameters)
    frame = pd.concat(
        [selection[["subset", "outer_fold", "representation"]], expanded],
        axis=1,
    )
    return (
        frame.groupby(
            [
                "subset",
                "representation",
                "max_features",
                "max_depth",
                "min_samples_leaf",
                "master_weighting",
            ],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "outer_folds_selected"})
        .sort_values(
            ["subset", "outer_folds_selected"],
            ascending=[True, False],
            kind="stable",
        )
    )


def importance_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            ["band_start_cm1", "band_stop_cm1_exclusive"], as_index=False
        )
        .agg(
            mean_ba_drop=("importance_mean_ba_drop", "mean"),
            sd_across_folds=("importance_mean_ba_drop", "std"),
            folds=("outer_fold", "nunique"),
        )
        .sort_values("mean_ba_drop", ascending=False, kind="stable")
    )


def value(
    summary: pd.DataFrame,
    subset: str,
    metric: str = "balanced_accuracy",
) -> pd.Series:
    row = summary[
        (summary["evaluation_subset"] == subset)
        & (summary["metric"] == metric)
    ]
    if len(row) != 1:
        raise ValueError(f"Missing unique summary row: {subset}, {metric}")
    return row.iloc[0]


def fmt(row: pd.Series) -> str:
    return f"{row['mean']:.3f} ± {row['ci95_half_width']:.3f}"


def write_report(
    output_dir: Path,
    outer: pd.DataFrame,
    domains: pd.DataFrame,
    comparisons: pd.DataFrame,
    selection: pd.DataFrame,
    importance: pd.DataFrame,
    negative: pd.DataFrame,
    ood: pd.DataFrame,
) -> None:
    strict = value(outer, "strict_core")
    quality = value(outer, "quality_pass")
    stress = value(outer, "field_quality_stress")
    ba = comparisons[comparisons["metric"] == "balanced_accuracy"].pivot(
        index="model", columns="evaluation_subset", values="mean"
    )
    lines = [
        "# NATO SERS random-forest addendum v1 — final report",
        "",
        "## Headline",
        "",
        (
            "A random forest was selected inside each master-group outer fold. "
            "Its locked balanced accuracy was "
            f"{fmt(strict)} on all 598 spectra, {fmt(quality)} on the "
            f"500 quality-pass spectra, and {fmt(stress)} on the 98 "
            "field-quality-stress spectra. Intervals use five physical-master "
            "folds after averaging the three declared forest seeds."
        ),
        "",
        "## Locked comparison",
        "",
        "| Model | Strict BA | Quality BA | Stress BA |",
        "|---|---:|---:|---:|",
    ]
    for model in (
        "Classical",
        "Siamese",
        "Contrastive successor",
        "Random forest",
    ):
        lines.append(
            f"| {model} | {ba.loc[model, 'strict_core']:.3f} | "
            f"{ba.loc[model, 'quality_pass']:.3f} | "
            f"{ba.loc[model, 'field_quality_stress']:.3f} |"
        )
    lines.extend(
        [
            "",
            "These are not row-random train/test splits: no physical master "
            "sample crosses a fold. Seed repeats are not counted as independent "
            "samples.",
            "",
            "## What was selected",
            "",
        ]
    )
    for _, row in selection.iterrows():
        depth = (
            "None"
            if pd.isna(row["max_depth"])
            else str(int(row["max_depth"]))
        )
        lines.append(
            f"- {row['subset']}: {int(row['outer_folds_selected'])}/5 folds "
            f"selected {row['representation']}, max_features="
            f"{row['max_features']}, max_depth={depth}, min_leaf="
            f"{int(row['min_samples_leaf'])}, {row['master_weighting']}."
        )
    lines.extend(
        [
            "",
            "## Field shift and uncertainty",
            "",
            (
                f"- Confidence-only quality-versus-stress detection: mean AUROC "
                f"{ood['ood_auroc'].mean():.3f} over fold/seed runs. This tests "
                "whether low maximum probability flags bad field spectra; it "
                "does not prove denoising."
            ),
            (
                f"- Master-label permutation control: mean balanced accuracy "
                f"{negative['balanced_accuracy'].mean():.3f}, maximum "
                f"{negative['balanced_accuracy'].max():.3f}; seven-class chance "
                "is 0.143."
            ),
            "",
            "## Held-domain results",
            "",
            "| Training subset | Protocol | Held domain | BA mean | 95% half-width | Domains |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for _, row in domains[
        domains["metric"] == "balanced_accuracy"
    ].sort_values(["subset", "domain_type", "protocol"]).iterrows():
        lines.append(
            f"| {row['subset']} | {row['protocol']} | "
            f"{row['domain_type']} | {row['mean']:.3f} | "
            f"{row['ci95_half_width']:.3f} | "
            f"{int(row['n_independent_units'])} |"
        )
    lines.extend(
        [
            "",
            "Instrument/sensor means average the declared forest seeds within "
            "each held domain first. Wide intervals reflect very few domains "
            "and incomplete analyte-domain support.",
            "",
            "## Predictive spectral regions",
            "",
        ]
    )
    for _, row in importance.head(10).iterrows():
        lines.append(
            f"- {int(row['band_start_cm1'])}–"
            f"{int(row['band_stop_cm1_exclusive'])} cm⁻¹: mean held-fold BA "
            f"drop {row['mean_ba_drop']:.3f}."
        )
    lines.extend(
        [
            "",
            "Band permutation measures model reliance, not causal chemical "
            "assignment. Adjacent Raman variables are correlated, so importance "
            "can be divided among neighboring bands.",
            "",
            "## Interpretation",
            "",
            "The forest is a nonlinear classifier over preprocessed intensity "
            "variables. It does not learn a cleaned spectrum, reconstruct a "
            "chemical-only signal, or establish chemical/nuisance factorization. "
            "Its value is as a rigorous small-data baseline and as evidence for "
            "where performance fails under real field and domain shift.",
        ]
    )
    (output_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")


def write_hashes(output_dir: Path) -> None:
    excluded = {
        "artifact_hashes.json",
        "validation_report.json",
        "clean_rebuild_comparison.json",
    }
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name not in excluded
        and "search_shards" not in path.parts
    )
    common.write_json(
        output_dir / "artifact_hashes.json",
        {
            str(path.relative_to(output_dir)): common.sha256_file(path)
            for path in files
        },
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    outer = outer_summary(pd.read_csv(output_dir / "outer_metrics.csv"))
    domains = domain_summary(pd.read_csv(output_dir / "domain_metrics.csv"))
    comparisons = comparison_table(outer, args.comparison)
    selection = selection_summary(
        pd.read_csv(output_dir / "outer_selection.csv")
    )
    importance = importance_summary(
        pd.read_csv(output_dir / "band_permutation_importance.csv")
    )
    negative = pd.read_csv(output_dir / "negative_control_metrics.csv")
    ood = pd.read_csv(output_dir / "field_stress_ood_metrics.csv")
    outer.to_csv(output_dir / "outer_summary.csv", index=False)
    domains.to_csv(output_dir / "domain_summary.csv", index=False)
    comparisons.to_csv(output_dir / "locked_model_comparison.csv", index=False)
    selection.to_csv(output_dir / "selection_summary.csv", index=False)
    importance.to_csv(output_dir / "band_importance_summary.csv", index=False)
    write_report(
        output_dir,
        outer,
        domains,
        comparisons,
        selection,
        importance,
        negative,
        ood,
    )
    write_hashes(output_dir)


if __name__ == "__main__":
    main()
