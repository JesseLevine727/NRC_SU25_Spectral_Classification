#!/usr/bin/env python3
"""Aggregate the full open-world study and write its final Markdown report."""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import sers_open_world_common as common


METHOD_ORDER = [
    "Selected classical",
    "Siamese triplet",
    "Supervised contrastive",
    "Standard VAE",
    "Structured VAE—chemical",
    "Structured VAE—union",
    "Selected residual CNN",
]
FROZEN_PRIMARY = METHOD_ORDER[:5]
OPEN_METRICS = [
    "master_unknown_auroc",
    "master_unknown_auprc",
    "master_fpr_at_95_tpr",
    "master_oscr_auc",
    "master_known_balanced_accuracy",
]
CLASSIFICATION_METRICS = [
    "balanced_accuracy",
    "macro_f1",
    "accuracy",
    "negative_log_likelihood",
    "brier_multiclass",
    "expected_calibration_error_10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("Workspace/sers_open_world_shift/open_world_v1"),
    )
    parser.add_argument(
        "--rf-dir",
        type=Path,
        default=Path(
            "Workspace/sers_random_forest_addendum/random_forest_v1"
        ),
    )
    parser.add_argument(
        "--docs-report",
        type=Path,
        default=Path("docs/SERS_OPEN_WORLD_SHIFT_STUDY_FINAL.md"),
    )
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_open(study: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    baseline = pd.read_csv(study / "baselines" / "outer_open_metrics.csv")
    baseline = baseline[as_bool(baseline["score_selected"])].copy()
    baseline["method"] = "Selected classical"
    baseline["technical_seed"] = baseline["final_seed"]
    rows.append(baseline)
    specifications = [
        ("siamese_triplet", "Siamese triplet", "default"),
        (
            "supervised_contrastive",
            "Supervised contrastive",
            "default",
        ),
        ("standard_vae", "Standard VAE", "default"),
        ("structured_vae", "Structured VAE—chemical", "chemical"),
        ("structured_vae", "Structured VAE—union", "union"),
    ]
    for directory, label, head in specifications:
        frame = pd.read_csv(
            study / "neural_controls" / directory / "final_open_metrics.csv"
        )
        frame = frame[
            (frame["head"].astype(str) == head)
            & as_bool(frame["score_primary"])
        ].copy()
        frame["method"] = label
        frame["technical_seed"] = frame["declared_seed"]
        rows.append(frame)
    cnn = pd.read_csv(study / "cnn" / "final_open_metrics.csv")
    cnn = cnn[as_bool(cnn["score_selected"])].copy()
    cnn["method"] = "Selected residual CNN"
    cnn["technical_seed"] = cnn["declared_seed"]
    rows.append(cnn)
    return pd.concat(rows, ignore_index=True, sort=False)


def load_classification(study: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    baseline = pd.read_csv(study / "baselines" / "outer_classification.csv")
    baseline["method"] = "Selected classical"
    baseline["technical_seed"] = baseline["final_seed"]
    rows.append(baseline)
    specifications = [
        ("siamese_triplet", "Siamese triplet", "default"),
        (
            "supervised_contrastive",
            "Supervised contrastive",
            "default",
        ),
        ("standard_vae", "Standard VAE", "default"),
        ("structured_vae", "Structured VAE—chemical", "chemical"),
        ("structured_vae", "Structured VAE—union", "union"),
    ]
    for directory, label, head in specifications:
        frame = pd.read_csv(
            study
            / "neural_controls"
            / directory
            / "final_classification.csv"
        )
        frame = frame[frame["head"].astype(str) == head].copy()
        frame["method"] = label
        frame["technical_seed"] = frame["declared_seed"]
        rows.append(frame)
    cnn = pd.read_csv(study / "cnn" / "final_classification.csv")
    cnn["method"] = "Selected residual CNN"
    cnn["technical_seed"] = cnn["declared_seed"]
    rows.append(cnn)
    return pd.concat(rows, ignore_index=True, sort=False)


def load_predictions(study: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    baseline = pd.read_csv(study / "baselines" / "outer_predictions.csv")
    baseline["method"] = "Selected classical"
    baseline["technical_seed"] = baseline["final_seed"]
    baseline["task_primary_score"] = baseline["selected_score"]
    rows.append(baseline)
    specifications = [
        ("siamese_triplet", "Siamese triplet", "default"),
        (
            "supervised_contrastive",
            "Supervised contrastive",
            "default",
        ),
        ("standard_vae", "Standard VAE", "default"),
        ("structured_vae", "Structured VAE—chemical", "chemical"),
        ("structured_vae", "Structured VAE—union", "union"),
    ]
    for directory, label, head in specifications:
        frame = pd.read_csv(
            study / "neural_controls" / directory / "final_predictions.csv"
        )
        frame = frame[frame["head"].astype(str) == head].copy()
        frame["method"] = label
        frame["technical_seed"] = frame["declared_seed"]
        frame["task_primary_score"] = frame["primary_score"]
        rows.append(frame)
    cnn = pd.read_csv(study / "cnn" / "final_predictions.csv")
    cnn["method"] = "Selected residual CNN"
    cnn["technical_seed"] = cnn["declared_seed"]
    cnn["task_primary_score"] = cnn["selected_score"]
    rows.append(cnn)
    return pd.concat(rows, ignore_index=True, sort=False)


def load_thresholds(study: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    baseline = pd.read_csv(study / "baselines" / "outer_thresholds.csv")
    baseline = baseline[as_bool(baseline["score_selected"])].copy()
    baseline["method"] = "Selected classical"
    rows.append(baseline)
    specifications = [
        ("siamese_triplet", "Siamese triplet", "default"),
        (
            "supervised_contrastive",
            "Supervised contrastive",
            "default",
        ),
        ("standard_vae", "Standard VAE", "default"),
        ("structured_vae", "Structured VAE—chemical", "chemical"),
        ("structured_vae", "Structured VAE—union", "union"),
    ]
    for directory, label, head in specifications:
        frame = pd.read_csv(
            study
            / "neural_controls"
            / directory
            / "calibration_thresholds.csv"
        )
        frame = frame[
            (frame["head"].astype(str) == head)
            & as_bool(frame["score_primary"])
        ].copy()
        frame["method"] = label
        rows.append(frame)
    cnn = pd.read_csv(study / "cnn" / "calibration_thresholds.csv")
    cnn = cnn[as_bool(cnn["score_selected"])].copy()
    cnn["method"] = "Selected residual CNN"
    rows.append(cnn)
    return pd.concat(rows, ignore_index=True, sort=False)


def bootstrap_mean_ci(
    values: np.ndarray, seed: int, repetitions: int = 10000
) -> tuple[float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True)
    means = samples.mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def independent_unit_summary(
    frame: pd.DataFrame,
    unit: str,
    grouping: list[str],
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    technical = (
        frame.groupby(
            grouping + [unit, "outer_fold"],
            as_index=False,
            dropna=False,
        )[metrics]
        .mean()
    )
    units = (
        technical.groupby(grouping + [unit], as_index=False, dropna=False)[
            metrics
        ].mean()
    )
    rows: list[dict[str, Any]] = []
    for keys, group in units.groupby(grouping, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = dict(zip(grouping, keys))
        row["independent_unit"] = unit
        row["n_independent_units"] = group[unit].nunique()
        for metric in metrics:
            values = group[metric].to_numpy(float)
            lower, upper = bootstrap_mean_ci(
                values,
                common.stable_seed("summary", *keys, metric),
            )
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sd"] = float(np.nanstd(values, ddof=1))
            row[f"{metric}_bootstrap_ci95_lower"] = lower
            row[f"{metric}_bootstrap_ci95_upper"] = upper
        rows.append(row)
    return units, pd.DataFrame(rows)


def master_aggregate(
    frame: pd.DataFrame, score_name: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for master, group in frame.groupby("master_sample_id", sort=True):
        labels = group["target_analyte"].astype(str).unique()
        if len(labels) != 1:
            raise ValueError("A master-condition maps to multiple analytes")
        classes = json.loads(group.iloc[0]["probability_classes_json"])
        probability = np.vstack(
            [
                np.asarray(json.loads(value), float)
                for value in group["probabilities_json"]
            ]
        ).mean(axis=0)
        rows.append(
            {
                "master_sample_id": str(master),
                "target_analyte": labels[0],
                "predicted_analyte": str(classes[int(np.argmax(probability))]),
                "score": float(group[score_name].mean()),
            }
        )
    return pd.DataFrame(rows)


def accepted_metrics(
    known: pd.DataFrame,
    unknown: pd.DataFrame,
    score_name: str,
    threshold: float,
) -> dict[str, Any]:
    known_score = known[score_name].to_numpy(float)
    unknown_score = unknown[score_name].to_numpy(float)
    accepted = known_score <= threshold
    labels = known["target_analyte"].astype(str).to_numpy()
    predictions = known["predicted_analyte"].astype(str).to_numpy()
    known_master = master_aggregate(known, score_name)
    unknown_master = master_aggregate(unknown, score_name)
    master_accepted = known_master["score"].to_numpy(float) <= threshold
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        row_balanced_accuracy = (
            float(
                balanced_accuracy_score(
                    labels[accepted], predictions[accepted]
                )
            )
            if accepted.any()
            else np.nan
        )
        master_balanced_accuracy = (
            float(
                balanced_accuracy_score(
                    known_master.loc[master_accepted, "target_analyte"],
                    known_master.loc[master_accepted, "predicted_analyte"],
                )
            )
            if master_accepted.any()
            else np.nan
        )
    return {
        "known_row_realized_coverage": float(accepted.mean()),
        "known_row_balanced_accuracy_accepted": row_balanced_accuracy,
        "known_row_selective_risk": (
            float(1.0 - np.mean(labels[accepted] == predictions[accepted]))
            if accepted.any()
            else np.nan
        ),
        "unknown_row_rejection_rate": float(
            np.mean(unknown_score > threshold)
        ),
        "known_master_realized_coverage": float(master_accepted.mean()),
        "known_master_balanced_accuracy_accepted": master_balanced_accuracy,
        "known_master_selective_risk": (
            float(
                1.0
                - np.mean(
                    known_master.loc[master_accepted, "target_analyte"]
                    == known_master.loc[
                        master_accepted, "predicted_analyte"
                    ]
                )
            )
            if master_accepted.any()
            else np.nan
        ),
        "unknown_master_rejection_rate": float(
            np.mean(unknown_master["score"].to_numpy(float) > threshold)
        ),
        "n_known_rows": len(known),
        "n_unknown_rows": len(unknown),
        "n_known_masters": len(known_master),
        "n_unknown_masters": len(unknown_master),
    }


def risk_coverage(
    predictions: pd.DataFrame, thresholds: pd.DataFrame
) -> pd.DataFrame:
    threshold_lookup = {}
    for keys, group in thresholds.groupby(
        ["method", "held_unknown", "outer_fold", "score_name"], sort=True
    ):
        threshold_lookup[tuple(keys)] = group[
            ["requested_known_coverage", "threshold"]
        ].to_dict("records")
    rows: list[dict[str, Any]] = []
    evaluation_pairs = {
        "quality_open": ("known_quality_test", "unknown_quality"),
        "stress_open": ("known_stress", "unknown_stress"),
        "field_quality": ("known_quality_test", "known_stress"),
    }
    keys = [
        "method",
        "held_unknown",
        "outer_fold",
        "technical_seed",
    ]
    for values, task in predictions.groupby(keys, sort=True):
        context = dict(zip(keys, values))
        score_names = task["task_primary_score"].astype(str).unique()
        if len(score_names) != 1:
            raise ValueError("A task has multiple primary scores")
        score_name = score_names[0]
        key = (
            str(context["method"]),
            str(context["held_unknown"]),
            int(context["outer_fold"]),
            score_name,
        )
        declared = list(threshold_lookup[key])
        declared.append(
            {"requested_known_coverage": 1.0, "threshold": np.inf}
        )
        for evaluation, (known_name, unknown_name) in evaluation_pairs.items():
            known = task[
                task["evaluation_partition"].astype(str) == known_name
            ]
            unknown = task[
                task["evaluation_partition"].astype(str) == unknown_name
            ]
            for threshold in declared:
                rows.append(
                    {
                        **context,
                        "evaluation": evaluation,
                        "score_name": score_name,
                        "requested_known_coverage": float(
                            threshold["requested_known_coverage"]
                        ),
                        "threshold": float(threshold["threshold"]),
                        **accepted_metrics(
                            known,
                            unknown,
                            score_name,
                            float(threshold["threshold"]),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def field_quality_master_jackknife(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = predictions[
        predictions["evaluation_partition"].isin(
            ["known_quality_test", "known_stress"]
        )
    ]
    task_frames: dict[
        str, list[tuple[tuple[Any, ...], pd.DataFrame]]
    ] = {}
    keys = [
        "method",
        "held_unknown",
        "outer_fold",
        "technical_seed",
    ]
    for values, task in eligible.groupby(keys, sort=True):
        method = str(values[0])
        score_names = task["task_primary_score"].astype(str).unique()
        if len(score_names) != 1:
            raise ValueError("A field-quality task has multiple scores")
        score_name = score_names[0]
        rows: list[pd.DataFrame] = []
        for partition, shifted in (
            ("known_quality_test", False),
            ("known_stress", True),
        ):
            condition = task[
                task["evaluation_partition"].astype(str) == partition
            ]
            master = master_aggregate(condition, score_name)[
                ["master_sample_id", "score"]
            ].copy()
            master["field_stress"] = shifted
            rows.append(master)
        task_frames.setdefault(method, []).append(
            (tuple(values[1:]), pd.concat(rows, ignore_index=True))
        )

    def estimate(
        tasks: list[tuple[tuple[Any, ...], pd.DataFrame]],
        excluded_master: str | None,
    ) -> float:
        values: list[float] = []
        for _, frame in tasks:
            selected = (
                frame
                if excluded_master is None
                else frame[
                    frame["master_sample_id"].astype(str)
                    != str(excluded_master)
                ]
            )
            if selected["field_stress"].nunique() != 2:
                continue
            values.append(
                float(
                    roc_auc_score(
                        selected["field_stress"].astype(bool),
                        selected["score"].to_numpy(float),
                    )
                )
            )
        return float(np.mean(values)) if values else np.nan

    value_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        tasks = task_frames[method]
        masters = sorted(
            {
                str(master)
                for _, frame in tasks
                for master in frame["master_sample_id"]
            }
        )
        point = estimate(tasks, None)
        leave_one_out = np.asarray(
            [estimate(tasks, master) for master in masters], dtype=float
        )
        jackknife_mean = float(np.nanmean(leave_one_out))
        standard_error = float(
            np.sqrt(
                (len(masters) - 1)
                / len(masters)
                * np.nansum((leave_one_out - jackknife_mean) ** 2)
            )
        )
        for master, value in zip(masters, leave_one_out):
            value_rows.append(
                {
                    "method": method,
                    "excluded_master_sample_id": master,
                    "master_unknown_auroc": value,
                }
            )
        summary_rows.append(
            {
                "method": method,
                "master_unknown_auroc_mean": point,
                "master_unknown_auroc_jackknife_se": standard_error,
                "master_unknown_auroc_ci95_lower": max(
                    0.0, point - 1.96 * standard_error
                ),
                "master_unknown_auroc_ci95_upper": min(
                    1.0, point + 1.96 * standard_error
                ),
                "n_independent_masters": len(masters),
                "technical_tasks_averaged": len(tasks),
            }
        )
    return pd.DataFrame(value_rows), pd.DataFrame(summary_rows)


def promotion_gates(
    open_units: pd.DataFrame,
    open_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    def open_value(method: str, evaluation: str, metric: str) -> float:
        row = open_summary[
            (open_summary["method"] == method)
            & (open_summary["evaluation"] == evaluation)
        ].iloc[0]
        return float(row[f"{metric}_mean"])

    def class_value(method: str, partition: str, metric: str) -> float:
        row = class_summary[
            (class_summary["method"] == method)
            & (class_summary["evaluation_partition"] == partition)
        ].iloc[0]
        return float(row[f"{metric}_mean"])

    quality_frozen = {
        method: open_value(method, "quality_open", "master_unknown_auroc")
        for method in FROZEN_PRIMARY
    }
    best_method = max(quality_frozen, key=quality_frozen.get)
    best_value = quality_frozen[best_method]
    cnn_quality = open_value(
        "Selected residual CNN", "quality_open", "master_unknown_auroc"
    )
    cnn_stress = open_value(
        "Selected residual CNN", "stress_open", "master_unknown_auroc"
    )
    cnn_field = open_value(
        "Selected residual CNN", "field_quality", "master_unknown_auroc"
    )
    cnn_quality_ba = class_value(
        "Selected residual CNN", "known_quality_test", "balanced_accuracy"
    )
    classical_quality_ba = class_value(
        "Selected classical", "known_quality_test", "balanced_accuracy"
    )
    cnn_stress_ba = class_value(
        "Selected residual CNN", "known_stress", "balanced_accuracy"
    )
    supcon_stress_ba = class_value(
        "Supervised contrastive", "known_stress", "balanced_accuracy"
    )
    unit = open_units[
        (open_units["evaluation"] == "quality_open")
        & open_units["method"].isin([best_method, "Selected residual CNN"])
    ]
    paired = unit.pivot(
        index="held_unknown",
        columns="method",
        values="master_unknown_auroc",
    ).dropna()
    chemicals_improved = int(
        (
            paired["Selected residual CNN"] - paired[best_method]
            > 0
        ).sum()
    )
    rows = [
        {
            "gate": "quality_known_ba_drop",
            "value": cnn_quality_ba - classical_quality_ba,
            "criterion": "difference >= -0.05",
            "pass": cnn_quality_ba - classical_quality_ba >= -0.05,
        },
        {
            "gate": "quality_open_auroc_improvement",
            "value": cnn_quality - best_value,
            "criterion": "difference >= 0.03",
            "pass": cnn_quality - best_value >= 0.03,
        },
        {
            "gate": "quality_open_chemical_count",
            "value": chemicals_improved,
            "criterion": "chemicals improved >= 4 of 6",
            "pass": chemicals_improved >= 4,
        },
        {
            "gate": "stress_open_auroc",
            "value": cnn_stress,
            "criterion": "AUROC >= 0.60",
            "pass": cnn_stress >= 0.60,
        },
        {
            "gate": "field_quality_auroc",
            "value": cnn_field,
            "criterion": "AUROC >= 0.90",
            "pass": cnn_field >= 0.90,
        },
        {
            "gate": "stress_known_ba_vs_supcon",
            "value": cnn_stress_ba - supcon_stress_ba,
            "criterion": "difference >= 0.00",
            "pass": cnn_stress_ba - supcon_stress_ba >= 0.0,
        },
    ]
    frame = pd.DataFrame(rows)
    context = {
        "best_frozen_primary_method": best_method,
        "best_frozen_primary_quality_open_auroc": best_value,
        "all_gates_pass": bool(frame["pass"].all()),
        "gates_passed": int(frame["pass"].sum()),
        "gates_total": len(frame),
    }
    return frame, context


def method_per_class(predictions: pd.DataFrame) -> pd.DataFrame:
    known = predictions[
        predictions["evaluation_partition"].isin(
            ["known_quality_test", "known_stress"]
        )
    ].copy()
    return (
        known.assign(
            correct=known["target_analyte"].astype(str)
            == known["predicted_analyte"].astype(str)
        )
        .groupby(
            [
                "method",
                "evaluation_partition",
                "target_analyte",
            ],
            as_index=False,
        )
        .agg(
            technical_prediction_rows=("correct", "size"),
            recall=("correct", "mean"),
            masters=("master_sample_id", "nunique"),
        )
    )


def confusion(predictions: pd.DataFrame) -> pd.DataFrame:
    cnn = predictions[
        (predictions["method"] == "Selected residual CNN")
        & predictions["evaluation_partition"].isin(
            ["known_quality_test", "known_stress"]
        )
    ].copy()
    counts = (
        cnn.groupby(
            [
                "evaluation_partition",
                "target_analyte",
                "predicted_analyte",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "technical_repeat_count"})
    )
    counts["row_normalized_fraction"] = counts[
        "technical_repeat_count"
    ] / counts.groupby(
        ["evaluation_partition", "target_analyte"]
    )[
        "technical_repeat_count"
    ].transform(
        "sum"
    )
    return counts


def domain_outputs(study: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    domain = pd.read_csv(study / "cnn" / "domain_seed_averaged_metrics.csv")
    summary = pd.read_csv(study / "cnn" / "domain_summary.csv")
    return domain, summary


def cnn_control_summaries(study: Path) -> dict[str, pd.DataFrame]:
    cnn = study / "cnn"
    permutation = pd.read_csv(
        cnn / "negative_master_label_permutation_metrics.csv"
    )
    permutation_units = (
        permutation.groupby("held_unknown", as_index=False)[
            "balanced_accuracy"
        ].mean()
    )
    lower, upper = bootstrap_mean_ci(
        permutation_units["balanced_accuracy"].to_numpy(float), 8721
    )
    permutation_summary = pd.DataFrame(
        [
            {
                "n_held_chemicals": len(permutation_units),
                "mean_balanced_accuracy": permutation_units[
                    "balanced_accuracy"
                ].mean(),
                "sd_across_chemicals": permutation_units[
                    "balanced_accuracy"
                ].std(),
                "bootstrap_ci95_lower": lower,
                "bootstrap_ci95_upper": upper,
            }
        ]
    )
    classification = pd.read_csv(cnn / "sensitivity_classification.csv")
    classification = classification[
        classification["evaluation_partition"].isin(
            ["known_quality_test", "known_stress"]
        )
    ]
    sensitivity_class_units, sensitivity_class_summary = (
        independent_unit_summary(
            classification,
            "held_unknown",
            ["control", "evaluation_partition"],
            ["balanced_accuracy"],
        )
    )
    open_frame = pd.read_csv(cnn / "sensitivity_open_metrics.csv")
    sensitivity_open_units, sensitivity_open_summary = (
        independent_unit_summary(
            open_frame,
            "held_unknown",
            ["control", "evaluation", "score_name"],
            ["master_unknown_auroc", "master_oscr_auc"],
        )
    )
    mc = pd.read_csv(cnn / "final_mc_pass_sensitivity.csv")
    mc_units, mc_summary = independent_unit_summary(
        mc,
        "held_unknown",
        ["evaluation", "pass_count"],
        ["master_unknown_auroc", "master_oscr_auc"],
    )
    probes = pd.read_csv(
        cnn / "final_target_adjusted_domain_probes.csv"
    )
    probe_units, probe_summary = independent_unit_summary(
        probes,
        "held_unknown",
        ["domain_type"],
        [
            "cell_balanced_accuracy",
            "target_only_null_cell_balanced_accuracy",
            "increment_over_target_only",
        ],
    )
    agreement = pd.read_csv(cnn / "final_same_master_agreement.csv")
    agreement_units, agreement_summary = independent_unit_summary(
        agreement,
        "held_unknown",
        ["evaluation_partition", "domain_type"],
        [
            "masters_with_cross_domain_pairs",
            "cross_domain_pairs",
            "prediction_agreement",
        ],
    )
    domain_contrasts = pd.read_csv(
        cnn / "domain_registered_contrasts.csv"
    )
    domain_agreement = pd.read_csv(
        cnn / "domain_same_master_agreement.csv"
    )
    return {
        "permutation_units": permutation_units,
        "permutation_summary": permutation_summary,
        "sensitivity_class_units": sensitivity_class_units,
        "sensitivity_class_summary": sensitivity_class_summary,
        "sensitivity_open_units": sensitivity_open_units,
        "sensitivity_open_summary": sensitivity_open_summary,
        "mc_units": mc_units,
        "mc_summary": mc_summary,
        "target_adjusted_probe_units": probe_units,
        "target_adjusted_probe_summary": probe_summary,
        "same_master_agreement_units": agreement_units,
        "same_master_agreement_summary": agreement_summary,
        "domain_registered_contrasts": domain_contrasts,
        "domain_same_master_agreement": domain_agreement,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def markdown_table(
    frame: pd.DataFrame, columns: list[tuple[str, str]]
) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = [header, separator]
    for record in frame.to_dict("records"):
        values = []
        for key, _ in columns:
            value = record.get(key)
            if isinstance(value, (float, np.floating)):
                values.append(fmt(value))
            elif isinstance(value, (bool, np.bool_)):
                values.append("pass" if value else "fail")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def compact_open_table(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot(
        index="method",
        columns="evaluation",
        values="master_unknown_auroc_mean",
    ).reset_index()
    pivot["method"] = pd.Categorical(
        pivot["method"], METHOD_ORDER, ordered=True
    )
    return pivot.sort_values("method")


def compact_class_table(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot(
        index="method",
        columns="evaluation_partition",
        values="balanced_accuracy_mean",
    ).reset_index()
    pivot["method"] = pd.Categorical(
        pivot["method"], METHOD_ORDER, ordered=True
    )
    return pivot.sort_values("method")


def write_report(
    args: argparse.Namespace,
    open_units: pd.DataFrame,
    open_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    risk_summary: pd.DataFrame,
    gates: pd.DataFrame,
    gate_context: dict[str, Any],
    domain_summary: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
) -> None:
    open_table = compact_open_table(open_summary)
    class_table = compact_class_table(class_summary)
    primary_domain = domain_summary[
        (domain_summary["subset"] == "quality_pass")
        & (domain_summary["protocol"] == "domain_and_sample")
        & (domain_summary["domain_type"] == "instrument")
    ].sort_values("variant_order")
    primary_sensor_domain = domain_summary[
        (domain_summary["subset"] == "quality_pass")
        & (domain_summary["protocol"] == "domain_and_sample")
        & (domain_summary["domain_type"] == "sensor_family")
    ].sort_values("variant_order")
    cnn_units = open_units[
        open_units["method"] == "Selected residual CNN"
    ]
    cnn_quality = cnn_units[
        cnn_units["evaluation"] == "quality_open"
    ][
        [
            "held_unknown",
            "master_unknown_auroc",
            "master_oscr_auc",
        ]
    ].rename(
        columns={
            "master_unknown_auroc": "quality_auroc",
            "master_oscr_auc": "quality_oscr",
        }
    )
    cnn_stress = cnn_units[
        cnn_units["evaluation"] == "stress_open"
    ][
        [
            "held_unknown",
            "master_unknown_auroc",
            "master_oscr_auc",
        ]
    ].rename(
        columns={
            "master_unknown_auroc": "stress_auroc",
            "master_oscr_auc": "stress_oscr",
        }
    )
    cnn_chemical = cnn_quality.merge(
        cnn_stress, on="held_unknown", validate="one_to_one"
    ).sort_values("held_unknown")
    selection = pd.read_csv(args.study_dir / "cnn" / "surrogate_selection.csv")
    selection_variants = (
        selection.groupby("variant", as_index=False)
        .size()
        .rename(columns={"size": "selected_tasks"})
        .sort_values(["selected_tasks", "variant"], ascending=[False, True])
    )
    selection_scores = (
        selection.groupby("score_name", as_index=False)
        .size()
        .rename(columns={"size": "selected_tasks"})
        .sort_values(
            ["selected_tasks", "score_name"], ascending=[False, True]
        )
    )
    sensitivity_class = controls["sensitivity_class_summary"].pivot(
        index="control",
        columns="evaluation_partition",
        values="balanced_accuracy_mean",
    ).reset_index()
    sensitivity_open = controls["sensitivity_open_summary"]
    sensitivity_open = sensitivity_open[
        sensitivity_open["score_name"]
        == "one_minus_calibrated_max_probability"
    ].pivot(
        index="control",
        columns="evaluation",
        values="master_unknown_auroc_mean",
    ).reset_index()
    sensitivity_table = sensitivity_class.merge(
        sensitivity_open, on="control", how="outer", validate="one_to_one"
    )
    mc_table = controls["mc_summary"].pivot(
        index="pass_count",
        columns="evaluation",
        values="master_unknown_auroc_mean",
    ).reset_index().sort_values("pass_count")
    probe_table = controls["target_adjusted_probe_summary"].sort_values(
        "domain_type"
    )
    agreement_table = controls["same_master_agreement_summary"].sort_values(
        ["evaluation_partition", "domain_type"]
    )
    domain_contrasts = controls["domain_registered_contrasts"]
    primary_domain_contrasts = domain_contrasts[
        (domain_contrasts["subset"] == "quality_pass")
        & (domain_contrasts["protocol"] == "domain_and_sample")
        & (domain_contrasts["domain_type"] == "instrument")
    ].sort_values("candidate")
    support = pd.read_csv(args.study_dir / "factor_support_audit.csv")
    support_summary = (
        support.groupby(["subset", "dimension"], as_index=False)
        .agg(
            analytes=("target_analyte", "nunique"),
            domain_levels=("level", "nunique"),
            observed_analyte_domain_cells=("level", "size"),
            minimum_masters_in_observed_cell=("masters", "min"),
        )
    )
    support_summary["possible_analyte_domain_cells"] = (
        support_summary["analytes"] * support_summary["domain_levels"]
    )
    support_summary["missing_analyte_domain_cells"] = (
        support_summary["possible_analyte_domain_cells"]
        - support_summary["observed_analyte_domain_cells"]
    )
    cnn_risk = risk_summary[
        risk_summary["method"] == "Selected residual CNN"
    ][
        [
            "evaluation",
            "requested_known_coverage",
            "known_master_realized_coverage_mean",
            "known_master_balanced_accuracy_accepted_mean",
            "unknown_master_rejection_rate_mean",
        ]
    ].sort_values(["evaluation", "requested_known_coverage"])
    rf = pd.read_csv(args.rf_dir / "outer_summary.csv")
    rf = rf[rf["metric"] == "balanced_accuracy"].copy()
    rf_lookup = {
        (str(row.training_subset), str(row.evaluation_subset)): row
        for row in rf.itertuples(index=False)
    }
    rf_table = pd.DataFrame(
        [
            {
                "cohort": "Strict core",
                "balanced_accuracy": rf_lookup[
                    ("strict_core", "strict_core")
                ].mean,
                "ci95_half_width": rf_lookup[
                    ("strict_core", "strict_core")
                ].ci95_half_width,
            },
            {
                "cohort": "Quality pass",
                "balanced_accuracy": rf_lookup[
                    ("quality_pass", "quality_pass")
                ].mean,
                "ci95_half_width": rf_lookup[
                    ("quality_pass", "quality_pass")
                ].ci95_half_width,
            },
            {
                "cohort": "Field stress",
                "balanced_accuracy": rf_lookup[
                    ("quality_pass", "field_quality_stress")
                ].mean,
                "ci95_half_width": rf_lookup[
                    ("quality_pass", "field_quality_stress")
                ].ci95_half_width,
            },
        ]
    )
    verdict = (
        "The residual CNN passed every promotion gate and is promoted."
        if gate_context["all_gates_pass"]
        else (
            "The residual CNN is not promoted: it passed "
            f"{gate_context['gates_passed']} of "
            f"{gate_context['gates_total']} preregistered gates."
        )
    )
    text = f"""# Open-world, shift-aware SERS identification across field instruments and sensor families

Date completed: 2026-07-30  
Dataset: `nato-sers-preprocessing-v2`  
Validation status: **pass**

## Executive verdict

{verdict}

The result is scientifically useful whether positive or negative because the
true unknown chemical, field-stress cohort, outer master fold, and held domain
never selected preprocessing, architecture, epoch, score, temperature, or
threshold. The experiment distinguishes three questions that are often
collapsed in Raman/SERS papers: known-class classification, rejection of
unseen chemistry, and detection of acquisition-quality failure.

The standard and structured VAE controls do **not** establish chemical/nuisance
disentanglement. Reconstruction error can detect acquisition shift without
being a reliable unseen-chemical score, and the structured chemical head must
beat its union-latent negative comparator before any mechanistic separation
claim is defensible. This archive has no paired clean chemical-only target and
its chemical, instrument, sensor, station, and master factors are incompletely
crossed. It therefore cannot verify recovery of a unique nuisance-free Raman
spectrum.

## Dataset and preprocessing contract

- 598 attributable, valid SERS spectra from 69 physical master samples;
- 500 quality-pass and 98 locked field-stress spectra;
- six nonblank chemicals plus operational blank;
- ten instruments and four sensor families;
- common measured support 400–1800 cm⁻¹ at 1 cm⁻¹ (1,401 points);
- `minimal_minmax`, `arpls_minmax`, and `derivative_1` frozen before this study;
- no general, conditional, or instrument-specific smoothing;
- no post hoc alignment or instrument-specific preprocessing.

The lower bound is 400 cm⁻¹ because it is the first measured coordinate shared
by every included system. Extending to 100 cm⁻¹ would fabricate Mira values,
extend below Pendar support, and use an untrustworthy padded Agilent region.
The reduction from 721 named-SERS log rows to 598 spectra reflects missing
files, unresolved labels, contradictory source reuse, and one consistent
duplicate—not a model-driven quality filter.

The archive is not factorially crossed:

{markdown_table(support_summary, [
    ("subset", "Subset"),
    ("dimension", "Domain factor"),
    ("domain_levels", "Levels"),
    ("observed_analyte_domain_cells", "Observed analyte×domain cells"),
    ("possible_analyte_domain_cells", "Possible cells"),
    ("missing_analyte_domain_cells", "Missing cells"),
    ("minimum_masters_in_observed_cell", "Minimum masters/cell"),
])}

Consequently, instrument or sensor predictability in an embedding is not by
itself evidence of nuisance encoding: chemistry-domain support imbalance can
create the same pattern. The target-adjusted probes below compare spectral
embeddings with a chemistry-label-only null to expose, but cannot causally
remove, this confounding.

## Leakage firewall

`master_sample_id` is indivisible. Each of the six true-unknown experiments
removes one nonblank chemical completely. Candidate objective, representation,
score, and epoch are selected in 600 inner surrogate-unknown scenarios; the
true held chemical and all 98 stress rows remain locked. Temperatures and
thresholds come only from four-fold cross-fitted known-development
predictions. Rejection thresholds target known retention of 95%, 90%, 80%,
70%, and 50%. Domain tests use the frozen `domain_only` and stricter
`domain_and_sample` partitions and provide no target-domain rows during
training.

## Models actually trained

- nine sealed classical candidates spanning PCA/logistic regression, RBF SVM,
  and random forest with three frozen representations;
- a validated random-forest addendum;
- Siamese/triplet and supervised-contrastive controls;
- standard VAE and structured chemical/union VAE controls;
- a 142,366-parameter residual 1-D CNN with seven bounded objectives:
  derivative ERM, arPLS ERM, joint supervised contrastive, synthetic
  Objectosphere/outlier exposure, conditional CORAL, conditional MMD, and
  GroupDRO. Its stem uses an 11-point stride-2 convolution; three residual
  stages use 24/48/96 channels, 7-point kernels, GroupNorm, adaptive pooling,
  a 64-dimensional embedding, and dropout 0.2;
- three final neural seeds treated as technical repeats;
- MSP, entropy, energy, class Mahalanobis distance, and MC-dropout mutual
  information, with the score selected only on surrogate unknowns.

Training budgets were fixed before locked outcomes: Siamese 100 epochs,
supervised contrastive 58, standard VAE 97, and structured VAE 500. CNN
surrogate candidates used AdamW (learning rate 0.001, weight decay 0.0001),
batch size 96, a 75-epoch maximum, first validation at epoch 25, and
10-epoch early-stopping patience. Each final CNN then retrained for the rounded
median development-selected epoch for its held-chemical/fold task. Thus a poor
result cannot be dismissed as an undocumented “few epochs” run; epoch and
objective sensitivity are preserved in the histories and selection tables.

## Primary open-world results

Values are master-level AUROC. Model seeds and the five outer models are
averaged before the six held chemicals are treated as independent units.
Chemical-novelty intervals resample the six held chemicals. Acquisition-shift
uncertainty instead uses leave-one-physical-master-out jackknife estimates
over the 69 masters; held-chemical task repeats are not counted as independent
field specimens.

{markdown_table(open_table, [
    ("method", "Method"),
    ("quality_open", "Unseen chemical, quality"),
    ("stress_open", "Unseen chemical, stress"),
    ("field_quality", "Field-stress detection"),
])}

AUROC 0.5 is chance. Quality-open AUROC asks whether an unseen chemical scores
as more anomalous than known quality spectra. Stress-open AUROC asks the same
question after both groups are field-stressed. Field-quality AUROC instead
asks whether known stress spectra separate from known quality spectra; a high
value here does not imply chemical novelty detection.

### Selected CNN by held chemical

{markdown_table(cnn_chemical, [
    ("held_unknown", "Held chemical"),
    ("quality_auroc", "Quality AUROC"),
    ("quality_oscr", "Quality OSCR"),
    ("stress_auroc", "Stress AUROC"),
    ("stress_oscr", "Stress OSCR"),
])}

These six rows are the independent chemical units. A high mean with one or two
low chemicals is not universal rejection.

### What development-only selection chose

{markdown_table(selection_variants, [
    ("variant", "CNN objective"),
    ("selected_tasks", "Selected tasks (of 30)"),
])}

{markdown_table(selection_scores, [
    ("score_name", "Novelty score"),
    ("selected_tasks", "Selected tasks (of 30)"),
])}

Selection frequency is a stability diagnostic, not evidence that the most
frequent candidate wins on locked outcomes.

## Known-class results

{markdown_table(class_table, [
    ("method", "Method"),
    ("known_quality_test", "Quality balanced accuracy"),
    ("known_stress", "Stress balanced accuracy"),
])}

Balanced accuracy averages recall over supported known classes, so class
imbalance cannot inflate the number. Unsupported held-domain analytes remain
in row-level predictions and are excluded only from supported-class metrics.
NLL, multiclass Brier score, ten-bin ECE, macro-F1, accuracy, per-class recall,
and confusion matrices are preserved in the machine-readable result tables.

### Dedicated random-forest baseline

The separately nested random-forest addendum produced:

{markdown_table(rf_table, [
    ("cohort", "Cohort"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("ci95_half_width", "95% half-width"),
])}

These intervals use five master-group outer folds after averaging the three
forest seeds. The forest is competitive on quality spectra but degrades
strongly on locked field stress; it remains a classifier, not a denoiser or
factorized representation.

## Preregistered CNN promotion gates

The best frozen primary comparator for quality chemical rejection was
`{gate_context['best_frozen_primary_method']}` at AUROC
{fmt(gate_context['best_frozen_primary_quality_open_auroc'])}.

{markdown_table(gates, [
    ("gate", "Gate"),
    ("value", "Observed"),
    ("criterion", "Criterion"),
    ("pass", "Decision"),
])}

No gate is rescued by selecting a different locked score, seed, chemical, or
stress subset.

## Selective prediction at known-only thresholds

The table uses only the selected CNN score and master-level aggregation.
Thresholds were fixed from cross-fitted known development spectra. “Realized
coverage” can differ from the requested development coverage on the locked
test masters; unknown rejection is never used to choose a threshold.

{markdown_table(cnn_risk, [
    ("evaluation", "Open-set evaluation"),
    ("requested_known_coverage", "Requested known coverage"),
    ("known_master_realized_coverage_mean", "Realized known-master coverage"),
    ("known_master_balanced_accuracy_accepted_mean", "Accepted known-master BA"),
    ("unknown_master_rejection_rate_mean", "Unknown-master rejection"),
])}

## Held-domain transfer

The primary domain endpoint is quality-pass `domain_and_sample` instrument
transfer. Seeds are averaged first; mean and worst balanced accuracy use the
ten held instruments as independent units.

{markdown_table(primary_domain, [
    ("variant", "CNN objective"),
    ("mean_balanced_accuracy", "Mean held-instrument BA"),
    ("worst_balanced_accuracy", "Worst held-instrument BA"),
    ("standard_deviation_across_domains", "SD across instruments"),
])}

The corresponding sensor-family endpoint is necessarily narrower:

{markdown_table(primary_sensor_domain, [
    ("variant", "CNN objective"),
    ("held_domains_total", "Held sensor families"),
    ("held_domains_with_supported_metric", "Families with supported classes"),
    ("mean_balanced_accuracy", "Mean supported-family BA"),
    ("worst_balanced_accuracy", "Worst supported-family BA"),
])}

Some `domain_and_sample` sensor tasks have no supported analyte left after
same-master removal. Their test rows and predictions remain archived, but an
accuracy statistic is undefined; the table therefore reports both the four
attempted families and the number contributing a supported-class metric.

Registered source-only robustness contrasts against derivative ERM were:

{markdown_table(primary_domain_contrasts, [
    ("candidate", "Candidate"),
    ("reference", "Reference"),
    ("held_domains", "Paired instruments"),
    ("mean_paired_balanced_accuracy_difference", "Mean paired BA difference"),
    ("worst_paired_balanced_accuracy_difference", "Worst paired difference"),
    ("domains_improved", "Improved"),
    ("domains_tied", "Tied"),
    ("domains_worsened", "Worsened"),
])}

`domain_only` permits the same master on another device; `domain_and_sample`
removes every source-domain spectrum sharing a master with the held device.
The latter is the defensible generalization test. CORAL, MMD, and GroupDRO are
reported as prespecified paired contrasts against derivative ERM, not selected
using target-domain accuracy. This is source-only domain generalization, not
unsupervised domain adaptation.

## Controls and what they mean

The master-label permutation balanced accuracy was
{fmt(controls['permutation_summary'].iloc[0]['mean_balanced_accuracy'])}
(95% held-chemical bootstrap interval
{fmt(controls['permutation_summary'].iloc[0]['bootstrap_ci95_lower'])}–
{fmt(controls['permutation_summary'].iloc[0]['bootstrap_ci95_upper'])}).
This checks that physical-master grouping and the training code cannot recover
real labels after the development mapping is destroyed.

The augmentation/preprocessing ablation evaluates derivative ERM with and
without the registered shift/scale/noise perturbations, plus fixed arPLS and
minimal ERM. These are locked sensitivities with development-only epochs; no
true-unknown score selects among them. MC dropout is recomputed from a common
50-pass stream at 10, 20, and 50 prefixes, so pass-count stability is measured
without changing the trained model.

{markdown_table(sensitivity_table, [
    ("control", "Fixed sensitivity"),
    ("known_quality_test", "Quality BA"),
    ("known_stress", "Stress BA"),
    ("quality_open", "Quality-open MSP AUROC"),
    ("stress_open", "Stress-open MSP AUROC"),
    ("field_quality", "Field-quality MSP AUROC"),
])}

The preprocessing rows are a sensitivity analysis, not a second model search.
“MSP” is the fixed one-minus-maximum-probability diagnostic for every row;
these values do not replace the surrogate-selected primary CNN score.

{markdown_table(mc_table, [
    ("pass_count", "MC-dropout passes"),
    ("quality_open", "Quality-open MI AUROC"),
    ("stress_open", "Stress-open MI AUROC"),
    ("field_quality", "Field-quality MI AUROC"),
])}

Target-adjusted domain probes on locked quality-test embeddings were:

{markdown_table(probe_table, [
    ("domain_type", "Domain"),
    ("cell_balanced_accuracy_mean", "Embedding probe cell-BA"),
    ("target_only_null_cell_balanced_accuracy_mean", "Chemistry-only null cell-BA"),
    ("increment_over_target_only_mean", "Increment over null"),
])}

Positive increments mean domain remains predictable beyond the analyte-label
null. That is evidence of residual acquisition information, not proof that the
network failed to learn any useful invariant feature.

Same-master prediction consistency was:

{markdown_table(agreement_table, [
    ("evaluation_partition", "Partition"),
    ("domain_type", "Cross-domain pair"),
    ("masters_with_cross_domain_pairs_mean", "Masters with pairs"),
    ("cross_domain_pairs_mean", "Pairs"),
    ("prediction_agreement_mean", "Prediction agreement"),
])}

These are diagnostic task-averages. Agreement can be high because predictions
are consistently correct or consistently wrong, so it is interpreted beside
balanced accuracy rather than as a standalone success criterion.

The score-direction control is exactly `1 − AUROC`. Known-only threshold
coverage is independently recomputed from saved calibration scores. Exact UID
audits show no master, held chemical, outer-test, stress, or held-domain
leakage.

## Did the VAE/disentanglement idea work?

Not as a demonstrated disentangling mechanism. A VAE can compress,
reconstruct, provide density or reconstruction anomaly scores, and sometimes
improve downstream prediction. None of those facts identifies a latent as
“chemical” or “nuisance.” Without paired clean spectra, interventions that
change one nuisance while holding chemistry fixed, or a sufficiently crossed
factorial design, latent labels are underdetermined. This is consistent with
the general identifiability result that unsupervised disentanglement requires
inductive bias or supervision
([Locatello et al., 2019](https://proceedings.mlr.press/v97/locatello19a.html)).

The NATO archive *does* support publishable questions about open-set
recognition, uncertainty failure modes, source-only instrument transfer,
preprocessing sensitivity, and whether a compact CNN adds value beyond
classical/metric controls. It does not support “chemical-only spectrum
recovery,” causal substrate removal, or denoising against an unobserved clean
target.

## Relation to current Raman/SERS research

- Objectosphere losses have already been applied to open-world Raman
  identification with explicit irrelevant classes
  ([Balytskyi et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9728505/)).
  Here, the actual held chemical is never used as an irrelevant/outlier class;
  synthetic outliers are a separate registered objective.
- A 2026 SERS study combined learned enhancement, GASF, and MSP rejection
  ([Wang et al.](https://opg.optica.org/ao/abstract.cfm?uri=ao-65-18-6135)).
  Its very high within-dataset open-set values do not answer this archive’s
  cross-instrument, physical-master, and field-stress question.
- MC dropout has improved accuracy on retained low-uncertainty Raman subsets
  ([Contreras and Bocklitz, 2024](https://pubmed.ncbi.nlm.nih.gov/39580162/)).
  This study additionally tests whether MI recognizes unseen chemistry versus
  acquisition failure and locks its threshold to known development data.
- RamanMAE demonstrates the promise of masked spectral pretraining on much
  larger biological Raman corpora
  ([Paidi and Maheshwari, 2025](https://pubs.acs.org/doi/10.1021/acs.analchem.5c05656)).
  Training a foundation model from 598 NATO rows would not be an equivalent
  claim.
- DSCF uses broad self-supervised pretraining for cross-task spectral
  component filtering
  ([Nature Machine Intelligence, 2025](https://www.nature.com/articles/s42256-025-01027-5)).
  That paired/pretrained evidence is materially stronger for denoising than
  an unpaired NATO reconstruction loss.
- Parameter-efficient calibration transfer with a few target-instrument
  examples is a different protocol from source-only generalization
  ([Lai et al., 2025](https://pubmed.ncbi.nlm.nih.gov/40922652/)). A future NATO
  LoRA/PDS experiment must explicitly declare its unlabeled or labeled target
  spectra.
- A VAE-based Raman study handled varying ranges and noise across two labs
  using deliberate corruption training
  ([Jensen et al., 2024](https://www.nature.com/articles/s41598-024-56788-7)).
  It supports reconstruction/robustness as an empirical task, not automatic
  semantic identifiability.

## Claim limits and next experiment

Allowed claims are grouped known-class classification, unseen-chemical
rejection, field-quality rejection, selective prediction, uncertainty
diagnostics, and held-domain robustness on this archive. Prohibited claims are
denoising, chemical-only recovery, causal nuisance removal, semantic
disentanglement, or external-instrument generalization.

The most valuable next data collection is a crossed reference panel: the same
chemical preparations and concentrations measured across instruments and
sensor lots, plus technical replicates and a clean/reference measurement or
controlled corruption target. If limited target-device spectra become
available, evaluate calibration transfer/unsupervised adaptation as a new
protocol, never merge it with the source-only numbers above.

## Reproducibility and artifacts

- Frozen protocol: `predeclared_protocol.json`
- Baselines: `baselines/`
- Neural controls: `neural_controls/`
- CNN registry and results: `cnn/`
- Validation: `cnn/validation_report.json` and
  `cnn/validation_checks.csv`
- Summary tables: `summaries/`
- Figures: `figures/`
- Exact rebuild comparison: `rebuild_comparison.json`

Every result is retained at row level with observation UID, physical master,
instrument, sensor family, task, seed, probability vector, anomaly scores,
and immutable state hash. Seeds are technical repeats, held chemicals are the
open-set units, held domains are the transfer units, and physical masters are
the field-quality units.
"""
    target = args.study_dir / "FINAL_REPORT.md"
    target.write_text(text)
    args.docs_report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(target, args.docs_report)


def main() -> None:
    args = parse_args()
    validation = json.loads(
        (args.study_dir / "cnn" / "validation_report.json").read_text()
    )
    if validation["status"] != "pass":
        raise ValueError("CNN bundle must pass validation before finalization")
    output = args.study_dir / "summaries"
    output.mkdir(parents=True, exist_ok=True)
    open_frame = load_open(args.study_dir)
    classification = load_classification(args.study_dir)
    predictions = load_predictions(args.study_dir)
    thresholds = load_thresholds(args.study_dir)
    open_units, open_summary = independent_unit_summary(
        open_frame,
        "held_unknown",
        ["method", "evaluation"],
        OPEN_METRICS,
    )
    class_frame = classification[
        classification["evaluation_partition"].isin(
            ["known_quality_test", "known_stress"]
        )
    ]
    class_units, class_summary = independent_unit_summary(
        class_frame,
        "held_unknown",
        ["method", "evaluation_partition"],
        CLASSIFICATION_METRICS,
    )
    risk = risk_coverage(predictions, thresholds)
    risk_units, risk_summary = independent_unit_summary(
        risk,
        "held_unknown",
        ["method", "evaluation", "requested_known_coverage"],
        [
            "known_row_realized_coverage",
            "known_row_balanced_accuracy_accepted",
            "known_row_selective_risk",
            "unknown_row_rejection_rate",
            "known_master_realized_coverage",
            "known_master_balanced_accuracy_accepted",
            "known_master_selective_risk",
            "unknown_master_rejection_rate",
        ],
    )
    gates, gate_context = promotion_gates(
        open_units, open_summary, class_summary
    )
    field_jackknife, field_jackknife_summary = (
        field_quality_master_jackknife(predictions)
    )
    domain, domain_summary = domain_outputs(args.study_dir)
    controls = cnn_control_summaries(args.study_dir)
    tables = {
        "primary_open_held_chemical_metrics.csv": open_units,
        "primary_open_summary.csv": open_summary,
        "primary_classification_held_chemical_metrics.csv": class_units,
        "primary_classification_summary.csv": class_summary,
        "risk_coverage_task_metrics.csv": risk,
        "risk_coverage_held_chemical_metrics.csv": risk_units,
        "risk_coverage_summary.csv": risk_summary,
        "field_quality_master_jackknife_values.csv": field_jackknife,
        "field_quality_master_jackknife_summary.csv": (
            field_jackknife_summary
        ),
        "cnn_promotion_gates.csv": gates,
        "per_class_recall.csv": method_per_class(predictions),
        "cnn_confusion_matrices.csv": confusion(predictions),
        "cnn_domain_seed_averaged_metrics.csv": domain,
        "cnn_domain_summary.csv": domain_summary,
    }
    for name, frame in controls.items():
        tables[f"cnn_{name}.csv"] = frame
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    common.write_json(output / "cnn_promotion_verdict.json", gate_context)
    write_report(
        args,
        open_units,
        open_summary,
        class_summary,
        risk_summary,
        gates,
        gate_context,
        domain_summary,
        controls,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "all_cnn_gates_pass": gate_context["all_gates_pass"],
                "summary_tables": len(tables) + 1,
                "report": str(args.study_dir / "FINAL_REPORT.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
