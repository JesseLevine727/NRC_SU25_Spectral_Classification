#!/usr/bin/env python3
"""Validate completeness, leakage firewalls, and metrics for the CNN study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sers_open_world_common as common
from run_sers_open_world_cnn import SCORES


HELD = {
    "4_ANPP",
    "4_nitrophenol",
    "acetaminophen",
    "benzyl_fentanyl",
    "ethanol",
    "ethyl_paraoxon",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("Workspace/nato_sers_field_trial/preprocessing_v2"),
    )
    parser.add_argument(
        "--partition-dir",
        type=Path,
        default=Path("Workspace/sers_open_world_shift/open_world_v1"),
    )
    parser.add_argument(
        "--cnn-dir",
        type=Path,
        default=Path("Workspace/sers_open_world_shift/open_world_v1/cnn"),
    )
    return parser.parse_args()


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, identifier: str, condition: bool, details: Any) -> None:
        self.rows.append(
            {
                "check": identifier,
                "status": "pass" if bool(condition) else "fail",
                "details": json.dumps(
                    common.json_clean(details), sort_keys=True
                ),
            }
        )

    @property
    def passed(self) -> bool:
        return all(row["status"] == "pass" for row in self.rows)


def exact_groups(
    audit: Audit,
    identifier: str,
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
    uid: str = "observation_uid",
) -> None:
    observed_groups = {
        tuple(key if isinstance(key, tuple) else (key,)): set(
            group[uid].astype(str)
        )
        for key, group in observed.groupby(keys, sort=True)
    }
    expected_groups = {
        tuple(key if isinstance(key, tuple) else (key,)): set(
            group[uid].astype(str)
        )
        for key, group in expected.groupby(keys, sort=True)
    }
    duplicate_free = not observed.duplicated(keys + [uid]).any()
    audit.check(
        identifier,
        duplicate_free and observed_groups == expected_groups,
        {
            "observed_groups": len(observed_groups),
            "expected_groups": len(expected_groups),
            "duplicate_free": duplicate_free,
            "matching_groups": sum(
                observed_groups.get(key) == value
                for key, value in expected_groups.items()
            ),
        },
    )


def probability_audit(
    audit: Audit, identifier: str, predictions: pd.DataFrame
) -> None:
    valid = True
    maximum_error = 0.0
    class_mismatch = 0
    for row in predictions.itertuples(index=False):
        classes = json.loads(row.probability_classes_json)
        probabilities = np.asarray(json.loads(row.probabilities_json), float)
        valid &= (
            len(classes) == len(probabilities)
            and np.isfinite(probabilities).all()
            and (probabilities >= 0).all()
            and (probabilities <= 1).all()
        )
        maximum_error = max(maximum_error, abs(float(probabilities.sum()) - 1.0))
        if str(classes[int(np.argmax(probabilities))]) != str(
            row.predicted_analyte
        ):
            class_mismatch += 1
    audit.check(
        identifier,
        valid and maximum_error <= 2.0e-6 and class_mismatch == 0,
        {
            "rows": len(predictions),
            "maximum_probability_sum_error": maximum_error,
            "argmax_class_mismatches": class_mismatch,
        },
    )


def validate_surrogate(audit: Audit, cnn_dir: Path) -> None:
    metrics = pd.read_csv(cnn_dir / "surrogate_metrics.csv")
    scenario_keys = [
        "held_unknown",
        "outer_fold",
        "inner_fold",
        "surrogate_unknown",
    ]
    fit_keys = scenario_keys + ["variant"]
    audit.check(
        "cnn_surrogate_dimensions",
        len(metrics) == 21000
        and metrics.groupby(scenario_keys).ngroups == 600
        and metrics.groupby(fit_keys).ngroups == 4200
        and metrics["variant"].nunique() == 7
        and set(metrics["score_name"]) == set(SCORES),
        {
            "rows": len(metrics),
            "scenarios": metrics.groupby(scenario_keys).ngroups,
            "fitted_models": metrics.groupby(fit_keys).ngroups,
            "variants": metrics["variant"].nunique(),
            "scores": sorted(metrics["score_name"].unique()),
        },
    )
    numeric = [
        "master_oscr_auc",
        "master_known_balanced_accuracy",
        "master_unknown_auroc",
        "best_epoch",
    ]
    audit.check(
        "cnn_surrogate_metrics_finite",
        np.isfinite(metrics[numeric].to_numpy(float)).all(),
        {"numeric_columns": numeric},
    )
    consistency = metrics.groupby(fit_keys).agg(
        states=("state_sha256", "nunique"),
        epochs=("best_epoch", "nunique"),
        score_rows=("score_name", "nunique"),
    )
    audit.check(
        "cnn_surrogate_fit_row_consistency",
        (consistency == np.asarray([1, 1, 5])).all().all(),
        {
            "inconsistent_fits": int(
                (
                    (consistency.states != 1)
                    | (consistency.epochs != 1)
                    | (consistency.score_rows != 5)
                ).sum()
            )
        },
    )
    selection = pd.read_csv(cnn_dir / "surrogate_selection.csv")
    audit.check(
        "cnn_surrogate_selection_complete",
        len(selection) == 30
        and set(selection["held_unknown"]) == HELD
        and set(selection["outer_fold"]) == set(range(5))
        and (selection["scenarios_completed"] == 20).all(),
        {
            "rows": len(selection),
            "held_unknowns": sorted(selection["held_unknown"].unique()),
            "outer_folds": sorted(selection["outer_fold"].unique()),
        },
    )
    shard_paths = sorted(cnn_dir.glob("surrogate_metrics__*.csv"))
    cache_paths = sorted(
        (cnn_dir / "cache" / "surrogate").glob(
            "surrogate_metrics__*.csv"
        )
    )
    cache_by_name = {path.name: path for path in cache_paths}
    audit.check(
        "cnn_surrogate_resumable_shards_complete",
        len(shard_paths) == 6
        and len(cache_paths) == 6
        and all(
            path.name in cache_by_name
            and path.read_bytes() == cache_by_name[path.name].read_bytes()
            for path in shard_paths
        ),
        {
            "shards": len(shard_paths),
            "checkpoint_copies": len(cache_paths),
            "byte_identical_pairs": sum(
                path.name in cache_by_name
                and path.read_bytes()
                == cache_by_name[path.name].read_bytes()
                for path in shard_paths
            ),
        },
    )


def validate_calibration(
    audit: Audit, cnn_dir: Path, partition_dir: Path
) -> None:
    metrics = pd.read_csv(cnn_dir / "calibration_metrics.csv")
    thresholds = pd.read_csv(cnn_dir / "calibration_thresholds.csv")
    scores = pd.read_csv(cnn_dir / "calibration_scores.csv")
    registry = pd.read_csv(cnn_dir / "calibration_run_registry.csv")
    audit.check(
        "cnn_calibration_dimensions",
        len(metrics) == 30
        and len(thresholds) == 30 * 5 * 5
        and len(registry) == 30 * 4
        and registry["state_sha256"].nunique() == 120,
        {
            "metrics": len(metrics),
            "thresholds": len(thresholds),
            "run_registry": len(registry),
            "states": registry["state_sha256"].nunique(),
        },
    )
    partitions = pd.read_csv(
        partition_dir / "known_threshold_crossfit_partitions.csv"
    )
    expected = partitions[
        partitions["partition"] == "calibration_known"
    ][["held_unknown", "outer_fold", "observation_uid"]].drop_duplicates()
    exact_groups(
        audit,
        "cnn_calibration_exact_crossfit_uid_coverage",
        scores,
        expected,
        ["held_unknown", "outer_fold"],
    )
    threshold_ok = True
    maximum_error = 0.0
    for row in thresholds.itertuples(index=False):
        frame = scores[
            (scores["held_unknown"] == row.held_unknown)
            & (scores["outer_fold"] == row.outer_fold)
        ]
        expected_threshold = common.threshold_from_known(
            frame[row.score_name].to_numpy(float),
            float(row.requested_known_coverage),
        )
        error = abs(expected_threshold - float(row.threshold))
        maximum_error = max(maximum_error, error)
        threshold_ok &= error <= 1.0e-12
    audit.check(
        "cnn_known_only_thresholds_recomputed_exactly",
        threshold_ok,
        {
            "thresholds": len(thresholds),
            "maximum_absolute_error": maximum_error,
        },
    )
    audit.check(
        "cnn_calibration_values_finite",
        np.isfinite(metrics["temperature"]).all()
        and np.isfinite(scores[list(SCORES)].to_numpy(float)).all()
        and np.isfinite(thresholds["threshold"]).all(),
        {},
    )


def expected_final_predictions(partition_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(partition_dir / "open_set_outer_partitions.csv")
    mapping = {
        "known_quality_test": "known_quality_test",
        "unknown_quality": "unknown_quality",
        "known_stress": "known_stress",
        "unknown_stress": "unknown_stress",
    }
    expected = frame[frame["partition"].isin(mapping)].copy()
    expected["evaluation_partition"] = expected["partition"].map(mapping)
    seeds = pd.DataFrame({"declared_seed": [1729, 2718, 3141]})
    expected["_join"] = 1
    seeds["_join"] = 1
    return expected.merge(seeds, on="_join").drop(columns="_join")


def validate_final(
    audit: Audit, cnn_dir: Path, partition_dir: Path
) -> None:
    registry = pd.read_csv(cnn_dir / "final_run_registry.csv")
    classification = pd.read_csv(cnn_dir / "final_classification.csv")
    open_metrics = pd.read_csv(cnn_dir / "final_open_metrics.csv")
    threshold = pd.read_csv(cnn_dir / "final_threshold_metrics.csv")
    predictions = pd.read_csv(cnn_dir / "final_predictions.csv")
    mc = pd.read_csv(cnn_dir / "final_mc_pass_sensitivity.csv")
    probes = pd.read_csv(cnn_dir / "final_target_adjusted_domain_probes.csv")
    agreement = pd.read_csv(cnn_dir / "final_same_master_agreement.csv")
    histories = pd.read_csv(cnn_dir / "final_histories.csv")
    audit.check(
        "cnn_final_dimensions",
        len(registry) == 90
        and len(classification) == 360
        and len(open_metrics) == 1350
        and len(threshold) == 9000
        and len(mc) == 810
        and len(probes) == 180
        and len(agreement) == 360,
        {
            "run_registry": len(registry),
            "classification": len(classification),
            "open_metrics": len(open_metrics),
            "threshold_metrics": len(threshold),
            "mc_sensitivity": len(mc),
            "domain_probes": len(probes),
            "same_master_agreement": len(agreement),
        },
    )
    expected = expected_final_predictions(partition_dir)
    exact_groups(
        audit,
        "cnn_final_exact_locked_uid_coverage",
        predictions,
        expected,
        [
            "held_unknown",
            "outer_fold",
            "declared_seed",
            "evaluation_partition",
        ],
    )
    probability_audit(audit, "cnn_final_probability_integrity", predictions)
    held_logic = (
        (
            predictions["evaluation_partition"].isin(
                ["unknown_quality", "unknown_stress"]
            )
            == (
                predictions["target_analyte"].astype(str)
                == predictions["held_unknown"].astype(str)
            )
        ).all()
    )
    audit.check(
        "cnn_final_held_chemical_partition_logic",
        held_logic,
        {"rows": len(predictions)},
    )
    direction_error = np.nanmax(
        np.abs(
            open_metrics["direction_control_auroc"].to_numpy(float)
            - (1.0 - open_metrics["master_unknown_auroc"].to_numpy(float))
        )
    )
    audit.check(
        "cnn_score_direction_control",
        direction_error <= 1.0e-12,
        {"maximum_absolute_error": direction_error},
    )
    primary_counts = open_metrics.groupby(
        ["held_unknown", "outer_fold", "declared_seed", "evaluation"]
    )["score_selected"].sum()
    audit.check(
        "cnn_one_selected_score_per_locked_evaluation",
        (primary_counts == 1).all(),
        {
            "groups": len(primary_counts),
            "invalid_groups": int((primary_counts != 1).sum()),
        },
    )
    registered_mc = open_metrics[
        open_metrics["score_name"] == "mc_dropout_mutual_information"
    ]
    sensitivity_mc = mc[mc["pass_count"] == 20]
    keys = [
        "held_unknown",
        "outer_fold",
        "declared_seed",
        "evaluation",
    ]
    matched = registered_mc.merge(
        sensitivity_mc,
        on=keys,
        suffixes=("_registered", "_sensitivity"),
        validate="one_to_one",
    )
    columns = [
        "master_unknown_auroc",
        "master_unknown_auprc",
        "master_oscr_auc",
    ]
    mc_error = max(
        float(
            np.nanmax(
                np.abs(
                    matched[f"{column}_registered"].to_numpy(float)
                    - matched[f"{column}_sensitivity"].to_numpy(float)
                )
            )
        )
        for column in columns
    )
    audit.check(
        "cnn_mc_dropout_20_pass_reproduction",
        len(matched) == 270 and mc_error <= 1.0e-12,
        {"matched_rows": len(matched), "maximum_metric_error": mc_error},
    )
    audit.check(
        "cnn_final_numeric_finiteness",
        np.isfinite(
            open_metrics[
                [
                    "master_unknown_auroc",
                    "master_unknown_auprc",
                    "master_oscr_auc",
                ]
            ].to_numpy(float)
        ).all()
        and np.isfinite(
            classification.loc[
                classification["n_supported"] > 0,
                ["balanced_accuracy", "macro_f1", "accuracy"],
            ].to_numpy(float)
        ).all(),
        {},
    )
    history_lengths = (
        histories.groupby(
            ["held_unknown", "outer_fold", "declared_seed"],
            as_index=False,
        )
        .agg(
            epochs_recorded=("epoch", "max"),
            selected_epoch=("selected_epoch", "first"),
        )
    )
    audit.check(
        "cnn_final_training_histories_complete",
        len(history_lengths) == 90
        and (
            history_lengths["epochs_recorded"].astype(int)
            == history_lengths["selected_epoch"].astype(int)
        ).all(),
        {
            "runs": len(history_lengths),
            "minimum_epochs": int(history_lengths["epochs_recorded"].min()),
            "maximum_epochs": int(history_lengths["epochs_recorded"].max()),
        },
    )


def validate_controls(audit: Audit, cnn_dir: Path) -> None:
    permutation = pd.read_csv(
        cnn_dir / "negative_master_label_permutation_metrics.csv"
    )
    permutation_registry = pd.read_csv(
        cnn_dir / "negative_master_label_permutation_run_registry.csv"
    )
    sensitivity_registry = pd.read_csv(
        cnn_dir / "sensitivity_run_registry.csv"
    )
    sensitivity_classification = pd.read_csv(
        cnn_dir / "sensitivity_classification.csv"
    )
    sensitivity_open = pd.read_csv(cnn_dir / "sensitivity_open_metrics.csv")
    sensitivity_predictions = pd.read_csv(
        cnn_dir / "sensitivity_predictions.csv"
    )
    audit.check(
        "cnn_control_dimensions",
        len(permutation) == 90
        and len(permutation_registry) == 90
        and len(sensitivity_registry) == 360
        and len(sensitivity_classification) == 1440
        and len(sensitivity_open) == 5400
        and set(sensitivity_registry["control"])
        == {
            "erm_derivative_augmented",
            "erm_derivative_no_augmentation",
            "erm_arpls_augmented",
            "erm_minimal_augmented",
        },
        {
            "permutation": len(permutation),
            "permutation_runs": len(permutation_registry),
            "sensitivity_runs": len(sensitivity_registry),
            "classification": len(sensitivity_classification),
            "open_metrics": len(sensitivity_open),
        },
    )
    expected = expected_final_predictions(cnn_dir.parent)
    controls = pd.DataFrame(
        {
            "control": sorted(sensitivity_registry["control"].unique())
        }
    )
    expected["_join"] = 1
    controls["_join"] = 1
    expected = expected.merge(controls, on="_join").drop(columns="_join")
    exact_groups(
        audit,
        "cnn_sensitivity_exact_locked_uid_coverage",
        sensitivity_predictions,
        expected,
        [
            "held_unknown",
            "outer_fold",
            "declared_seed",
            "control",
            "evaluation_partition",
        ],
    )
    probability_audit(
        audit, "cnn_sensitivity_probability_integrity", sensitivity_predictions
    )
    audit.check(
        "cnn_master_label_permutation_is_nontrivial",
        (permutation["master_label_unchanged_fraction"] < 0.5).all()
        and np.isfinite(permutation["balanced_accuracy"]).all(),
        {
            "maximum_unchanged_master_fraction": float(
                permutation["master_label_unchanged_fraction"].max()
            ),
            "mean_balanced_accuracy": float(
                permutation["balanced_accuracy"].mean()
            ),
        },
    )
    permutation_cache = cnn_dir / "cache" / "controls_permutation"
    sensitivity_cache = cnn_dir / "cache" / "controls_sensitivity"
    audit.check(
        "cnn_control_resumable_caches_complete",
        len(list(permutation_cache.glob("*.json"))) == 90
        and len(list(sensitivity_cache.glob("*.json"))) == 360,
        {
            "permutation_json": len(
                list(permutation_cache.glob("*.json"))
            ),
            "sensitivity_json": len(
                list(sensitivity_cache.glob("*.json"))
            ),
        },
    )


def validate_domain(
    audit: Audit, cnn_dir: Path, partition_dir: Path
) -> None:
    epochs = pd.read_csv(cnn_dir / "domain_epoch_registry.csv")
    metrics = pd.read_csv(cnn_dir / "domain_metrics.csv")
    predictions = pd.read_csv(cnn_dir / "domain_predictions.csv")
    seed_averaged = pd.read_csv(cnn_dir / "domain_seed_averaged_metrics.csv")
    summary = pd.read_csv(cnn_dir / "domain_summary.csv")
    contrasts = pd.read_csv(cnn_dir / "domain_registered_contrasts.csv")
    agreement = pd.read_csv(cnn_dir / "domain_same_master_agreement.csv")
    support = pd.read_csv(cnn_dir / "domain_support_audit.csv")
    run_registry = pd.read_csv(cnn_dir / "domain_run_registry.csv")
    per_class = pd.read_csv(cnn_dir / "domain_per_class_metrics.csv")
    selective = pd.read_csv(cnn_dir / "domain_selective_metrics.csv")
    audit.check(
        "cnn_domain_dimensions",
        len(epochs) == 7
        and (epochs["surrogate_fitted_models"] == 600).all()
        and len(metrics) == 1176
        and len(seed_averaged) == 392
        and len(summary) == 56
        and len(contrasts) == 48
        and len(agreement) == 84,
        {
            "epoch_rows": len(epochs),
            "metrics": len(metrics),
            "seed_averaged": len(seed_averaged),
            "summary": len(summary),
            "contrasts": len(contrasts),
            "agreement": len(agreement),
            "run_registry": len(run_registry),
            "support_scenarios": support.groupby(
                ["subset", "protocol", "domain_type", "heldout_domain"]
            ).ngroups,
            "per_class_rows": len(per_class),
            "selective_rows": len(selective),
        },
    )
    audit.check(
        "cnn_domain_diagnostic_artifacts_complete",
        len(run_registry) == 1176
        and support.groupby(
            ["subset", "protocol", "domain_type", "heldout_domain"]
        ).ngroups
        == 56
        and not support.duplicated(
            [
                "subset",
                "protocol",
                "domain_type",
                "heldout_domain",
                "target_analyte",
            ]
        ).any()
        and len(per_class) > 0
        and len(selective) > 0,
        {
            "run_registry": len(run_registry),
            "support_rows": len(support),
            "support_scenarios": support.groupby(
                ["subset", "protocol", "domain_type", "heldout_domain"]
            ).ngroups,
            "per_class_rows": len(per_class),
            "selective_rows": len(selective),
        },
    )
    expected_parts: list[pd.DataFrame] = []
    for subset, filename in (
        ("strict_core", "domain_evaluation_partitions_core.csv"),
        ("quality_pass", "domain_evaluation_partitions_quality.csv"),
    ):
        frame = pd.read_csv(partition_dir / filename)
        frame = frame[frame["partition"] == "test"].copy()
        frame["subset"] = subset
        expected_parts.append(frame)
    expected = pd.concat(expected_parts, ignore_index=True)
    variants = pd.DataFrame({"variant": sorted(metrics["variant"].unique())})
    seeds = pd.DataFrame({"declared_seed": [1729, 2718, 3141]})
    expected["_join"] = 1
    variants["_join"] = 1
    seeds["_join"] = 1
    expected = (
        expected.merge(variants, on="_join")
        .merge(seeds, on="_join")
        .drop(columns="_join")
        .rename(
            columns={
                "protocol": "protocol",
                "heldout_domain": "heldout_domain",
            }
        )
    )
    exact_groups(
        audit,
        "cnn_domain_exact_test_uid_coverage",
        predictions,
        expected,
        [
            "subset",
            "protocol",
            "domain_type",
            "heldout_domain",
            "variant",
            "declared_seed",
        ],
    )
    probability_audit(audit, "cnn_domain_probability_integrity", predictions)
    supported_counts = (
        predictions.groupby(
            [
                "subset",
                "protocol",
                "domain_type",
                "heldout_domain",
                "variant",
                "declared_seed",
            ],
            as_index=False,
        )["supported"]
        .sum()
        .rename(columns={"supported": "prediction_n_supported"})
    )
    matched = metrics.merge(
        supported_counts,
        on=[
            "subset",
            "protocol",
            "domain_type",
            "heldout_domain",
            "variant",
            "declared_seed",
        ],
        validate="one_to_one",
    )
    audit.check(
        "cnn_domain_unsupported_accounting",
        (
            matched["n_supported"].astype(int)
            == matched["prediction_n_supported"].astype(int)
        ).all()
        and (
            matched["n_supported"].astype(int)
            + matched["n_unsupported"].astype(int)
            == matched["n_test"].astype(int)
        ).all(),
        {
            "tasks": len(matched),
            "unsupported_rows": int(metrics["n_unsupported"].sum()),
        },
    )
    cache = cnn_dir / "cache" / "domain"
    audit.check(
        "cnn_domain_resumable_cache_complete",
        len(list(cache.glob("*.json"))) == 1176
        and len(list(cache.glob("*.npz"))) == 1176,
        {
            "json": len(list(cache.glob("*.json"))),
            "npz": len(list(cache.glob("*.npz"))),
        },
    )
    shard_names = (
        "domain_metrics",
        "domain_predictions",
        "domain_per_class_metrics",
        "domain_selective_metrics",
        "domain_support_audit",
        "domain_run_registry",
    )
    shard_inventory = {
        name: sorted(cnn_dir.glob(f"{name}__shard_*_of_06.csv"))
        for name in shard_names
    }
    shard_metrics = sum(
        len(pd.read_csv(path))
        for path in shard_inventory["domain_metrics"]
    )
    shard_runs = sum(
        len(pd.read_csv(path))
        for path in shard_inventory["domain_run_registry"]
    )
    audit.check(
        "cnn_domain_shards_complete",
        all(len(paths) == 6 for paths in shard_inventory.values())
        and shard_metrics == 1176
        and shard_runs == 1176,
        {
            "files_per_artifact": {
                name: len(paths)
                for name, paths in shard_inventory.items()
            },
            "metric_rows": shard_metrics,
            "run_rows": shard_runs,
        },
    )


def validate_frozen_controls(audit: Audit, partition_dir: Path) -> None:
    baseline_dir = partition_dir / "baselines"
    baseline = pd.read_csv(baseline_dir / "outer_open_metrics.csv")
    baseline_selection = pd.read_csv(
        baseline_dir / "surrogate_selection.csv"
    )
    baseline_runs = baseline[
        ["held_unknown", "outer_fold", "final_seed"]
    ].drop_duplicates()
    expected_baseline_runs = int(
        np.where(
            baseline_selection["model_family"] == "random_forest",
            3,
            1,
        ).sum()
    )
    audit.check(
        "frozen_control_baselines_complete",
        baseline_selection[
            ["held_unknown", "outer_fold"]
        ].drop_duplicates().shape[0]
        == 30
        and baseline_runs[
            ["held_unknown", "outer_fold"]
        ].drop_duplicates().shape[0]
        == 30
        and len(baseline_runs) == expected_baseline_runs,
        {
            "tasks": baseline_runs[
                ["held_unknown", "outer_fold"]
            ].drop_duplicates().shape[0],
            "runs": len(baseline_runs),
            "expected_candidate_specific_runs": expected_baseline_runs,
            "metric_rows": len(baseline),
        },
    )
    expected = {
        "siamese_triplet": 90,
        "supervised_contrastive": 90,
        "standard_vae": 90,
        "structured_vae": 90,
    }
    for method, count in expected.items():
        directory = partition_dir / "neural_controls" / method
        frame = pd.read_csv(directory / "final_open_metrics.csv")
        runs = frame[
            ["held_unknown", "outer_fold", "declared_seed"]
        ].drop_duplicates()
        audit.check(
            f"frozen_control_{method}_complete",
            len(runs) == count,
            {"runs": len(runs), "metric_rows": len(frame)},
        )


def required_files(cnn_dir: Path) -> list[str]:
    return [
        "surrogate_metrics.csv",
        "surrogate_selection.csv",
        "calibration_metrics.csv",
        "calibration_thresholds.csv",
        "calibration_scores.csv",
        "calibration_run_registry.csv",
        "final_run_registry.csv",
        "final_histories.csv",
        "final_classification.csv",
        "final_open_metrics.csv",
        "final_threshold_metrics.csv",
        "final_predictions.csv",
        "final_mc_pass_sensitivity.csv",
        "final_target_adjusted_domain_probes.csv",
        "final_same_master_agreement.csv",
        "negative_master_label_permutation_metrics.csv",
        "negative_master_label_permutation_run_registry.csv",
        "sensitivity_run_registry.csv",
        "sensitivity_classification.csv",
        "sensitivity_open_metrics.csv",
        "sensitivity_predictions.csv",
        "domain_epoch_registry.csv",
        "domain_metrics.csv",
        "domain_predictions.csv",
        "domain_run_registry.csv",
        "domain_per_class_metrics.csv",
        "domain_selective_metrics.csv",
        "domain_support_audit.csv",
        "domain_seed_averaged_metrics.csv",
        "domain_summary.csv",
        "domain_registered_contrasts.csv",
        "domain_same_master_agreement.csv",
        "predeclared_cnn_registry.json",
        "predeclared_cnn_domain_registry.json",
        "predeclared_cnn_controls_registry.json",
        "predeclared_protocol.json",
        "input_hashes.json",
        "domain_input_hashes.json",
        "environment.json",
    ]


def main() -> None:
    args = parse_args()
    audit = Audit()
    missing = [name for name in required_files(args.cnn_dir) if not (args.cnn_dir / name).exists()]
    audit.check("cnn_required_artifacts_exist", not missing, {"missing": missing})
    if not missing:
        validate_surrogate(audit, args.cnn_dir)
        validate_calibration(audit, args.cnn_dir, args.partition_dir)
        validate_final(audit, args.cnn_dir, args.partition_dir)
        validate_controls(audit, args.cnn_dir)
        validate_domain(audit, args.cnn_dir, args.partition_dir)
        validate_frozen_controls(audit, args.partition_dir)
    report = {
        "status": "pass" if audit.passed else "fail",
        "checks": len(audit.rows),
        "passed": sum(row["status"] == "pass" for row in audit.rows),
        "failed": sum(row["status"] == "fail" for row in audit.rows),
    }
    pd.DataFrame(audit.rows).to_csv(
        args.cnn_dir / "validation_checks.csv", index=False
    )
    common.write_json(args.cnn_dir / "validation_report.json", report)
    print(json.dumps(report, indent=2))
    if not audit.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
