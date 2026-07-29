#!/usr/bin/env python3
"""Validate the supervised-contrastive SERS result bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sers_baseline_common as baseline
import sers_contrastive_common as common


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
        "--protocol",
        type=Path,
        default=Path("configs/sers_supervised_contrastive_v1.json"),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("Workspace/nato_sers_field_trial/preprocessing_v2"),
    )
    parser.add_argument("--require-clean-rebuild", action="store_true")
    return parser.parse_args()


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.rows.append(
            {"check": name, "passed": bool(condition), "detail": detail}
        )

    def require(self) -> None:
        failed = [row for row in self.rows if not row["passed"]]
        if failed:
            raise AssertionError(json.dumps(failed, indent=2))


def main() -> None:
    args = parse_args()
    protocol = common.load_protocol(args.protocol)
    dataset = baseline.load_nato_dataset(args.bundle)
    manifest = dataset.manifest
    audit = Audit()

    copied = args.output_dir / "predeclared_protocol.json"
    audit.check("protocol_copy_exists", copied.exists())
    audit.check(
        "protocol_copy_exact",
        copied.exists() and copied.read_bytes() == args.protocol.read_bytes(),
    )
    required = (
        "input_hashes.json",
        "environment.json",
        "final_environment.json",
        "final_input_hashes.json",
        "stage1_candidate_registry.csv",
        "stage1_inner_metrics.csv",
        "stage1_selection.csv",
        "stage2_candidate_registry.csv",
        "stage2_inner_metrics.csv",
        "stage2_selection.csv",
        "outer_metrics.csv",
        "outer_predictions.csv",
        "outer_per_class_metrics.csv",
        "outer_selective_metrics.csv",
        "outer_temperatures.csv",
        "rejection_selection_metrics.csv",
        "rejection_decisions_at_80.csv",
        "outer_ood_metrics.csv",
        "outer_run_registry.csv",
        "attribution_metrics.csv",
        "attribution_peaks.csv",
        "siamese_control_metrics.csv",
        "siamese_control_domain_metrics.csv",
        "siamese_control_predictions.csv",
        "siamese_control_diagnostics.csv",
        "siamese_control_selective_metrics.csv",
        "siamese_control_failures.csv",
        "siamese_control_provenance.json",
        "domain_metrics.csv",
        "domain_predictions.csv",
        "domain_run_registry.csv",
        "negative_control_metrics.csv",
        "selection_training_histories.csv",
        "final_training_histories.csv",
        "outer_uncertainty_summary.csv",
        "locked_model_comparison.csv",
        "preprocessing_sensitivity_summary.csv",
        "terminal_decision.json",
        "successor_confusion_matrices.json",
        "successor_failure_cases.csv",
        "FINAL_REPORT.md",
        "figures/model_comparison.pdf",
        "figures/model_comparison.png",
        "figures/training_diagnostics.pdf",
        "figures/training_diagnostics.png",
        "artifact_hashes.json",
    )
    for relative in required:
        audit.check(
            f"required::{relative}", (args.output_dir / relative).exists()
        )
    hash_path = args.output_dir / "artifact_hashes.json"
    if hash_path.exists():
        hashes = json.loads(hash_path.read_text())
        for relative, expected in hashes.items():
            path = args.output_dir / relative
            audit.check(f"hashed_exists::{relative}", path.exists())
            if path.exists():
                audit.check(
                    f"hash_exact::{relative}",
                    baseline.sha256_file(path) == expected,
                )

    stage1_registry = pd.read_csv(
        args.output_dir / "stage1_candidate_registry.csv"
    )
    stage1 = pd.read_csv(args.output_dir / "stage1_inner_metrics.csv")
    stage1_selection = pd.read_csv(
        args.output_dir / "stage1_selection.csv"
    )
    stage2 = pd.read_csv(args.output_dir / "stage2_inner_metrics.csv")
    stage2_selection = pd.read_csv(
        args.output_dir / "stage2_selection.csv"
    )
    outer = pd.read_csv(args.output_dir / "outer_metrics.csv")
    predictions = pd.read_csv(args.output_dir / "outer_predictions.csv")
    selective = pd.read_csv(
        args.output_dir / "outer_selective_metrics.csv"
    )
    temperatures = pd.read_csv(args.output_dir / "outer_temperatures.csv")
    rejection_selection = pd.read_csv(
        args.output_dir / "rejection_selection_metrics.csv"
    )
    rejection_decisions = pd.read_csv(
        args.output_dir / "rejection_decisions_at_80.csv"
    )
    ood = pd.read_csv(args.output_dir / "outer_ood_metrics.csv")
    registry = pd.read_csv(args.output_dir / "outer_run_registry.csv")
    attribution = pd.read_csv(args.output_dir / "attribution_metrics.csv")
    peaks = pd.read_csv(args.output_dir / "attribution_peaks.csv")
    siamese = pd.read_csv(args.output_dir / "siamese_control_metrics.csv")
    siamese_domain = pd.read_csv(
        args.output_dir / "siamese_control_domain_metrics.csv"
    )
    siamese_predictions = pd.read_csv(
        args.output_dir / "siamese_control_predictions.csv"
    )
    siamese_diagnostics = pd.read_csv(
        args.output_dir / "siamese_control_diagnostics.csv"
    )
    siamese_selective = pd.read_csv(
        args.output_dir / "siamese_control_selective_metrics.csv"
    )
    siamese_failures = pd.read_csv(
        args.output_dir / "siamese_control_failures.csv"
    )
    domain = pd.read_csv(args.output_dir / "domain_metrics.csv")
    domain_predictions = pd.read_csv(
        args.output_dir / "domain_predictions.csv"
    )
    domain_registry = pd.read_csv(
        args.output_dir / "domain_run_registry.csv"
    )
    negative = pd.read_csv(
        args.output_dir / "negative_control_metrics.csv"
    )
    selection_history = pd.read_csv(
        args.output_dir / "selection_training_histories.csv"
    )
    final_history = pd.read_csv(
        args.output_dir / "final_training_histories.csv"
    )

    audit.check("stage1_candidate_count", len(stage1_registry) == 12)
    audit.check(
        "stage1_candidate_unique",
        stage1_registry["candidate_id"].nunique() == 12,
    )
    audit.check("stage1_fit_count", len(stage1) == 12 * 20)
    audit.check(
        "stage1_run_unique",
        not stage1.duplicated(
            ["outer_fold", "inner_validation_fold", "candidate_id"]
        ).any(),
    )
    audit.check("stage1_selection_count", len(stage1_selection) == 6)
    audit.check(
        "stage1_selections_eligible",
        stage1_selection["candidate_id"].isin(
            stage1.loc[stage1["eligible"], "candidate_id"]
        ).all(),
    )
    manifest_folds = (
        manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    )
    quality_pass_mask = (
        manifest["include_sers_qc_pass"].astype(bool).to_numpy()
    )
    expected_partition_counts = {}
    for outer_fold in range(5):
        for inner_fold in sorted(set(range(5)) - {outer_fold}):
            expected_partition_counts[(outer_fold, inner_fold)] = (
                int(
                    (
                        quality_pass_mask
                        & (manifest_folds != outer_fold)
                        & (manifest_folds != inner_fold)
                    ).sum()
                ),
                int(
                    (
                        quality_pass_mask
                        & (manifest_folds == inner_fold)
                    ).sum()
                ),
            )
    for stage_name, frame in (("stage1", stage1), ("stage2", stage2)):
        partition_counts_exact = all(
            (
                int(row.n_train),
                int(row.n_validation),
            )
            == expected_partition_counts[
                (int(row.outer_fold), int(row.inner_validation_fold))
            ]
            for row in frame.itertuples()
        )
        audit.check(
            f"{stage_name}_quality_only_partition_counts_exact",
            partition_counts_exact,
        )

    audit.check("stage2_fit_count", len(stage2) == 160)
    audit.check(
        "stage2_run_unique",
        not stage2.duplicated(
            [
                "scope",
                "outer_fold",
                "inner_validation_fold",
                "candidate_id",
            ]
        ).any(),
    )
    audit.check("stage2_selection_count", len(stage2_selection) == 6)
    audit.check(
        "stage2_selected_gates",
        (
            stage2_selection[
                [
                    "mean_distance_margin",
                    "mean_effective_rank",
                ]
            ]
            .astype(float)
            .to_numpy()
            > np.asarray([0.0, 4.0])
        ).all(),
    )

    expected_variants = 9
    audit.check("outer_registry_count", len(registry) == 2 * 5 * expected_variants * 3)
    audit.check(
        "outer_metric_count", len(outer) == 3 * 5 * expected_variants * 3
    )
    audit.check(
        "outer_variants_per_training_scenario",
        (
            outer.groupby(
                ["training_subset", "outer_fold", "declared_seed"]
            )["variant"].nunique()
            == expected_variants
        ).all(),
    )
    preprocessing_rows = outer[
        (outer["variant"] == "full_domain_aware")
        | outer["variant"].str.startswith("preprocessing_sensitivity_")
    ]
    audit.check(
        "preprocessing_sensitivity_complete_by_representation",
        (
            preprocessing_rows.groupby(
                [
                    "representation",
                    "evaluation_subset",
                    "outer_fold",
                    "declared_seed",
                ]
            ).size()
            == 1
        ).all()
        and preprocessing_rows.groupby(
            ["representation", "evaluation_subset"]
        )["outer_fold"].nunique().eq(5).all(),
    )
    audit.check(
        "outer_scenarios",
        set(outer["evaluation_subset"])
        == {"strict_core", "quality_pass", "field_quality_stress"},
    )
    for metric in (
        "balanced_accuracy",
        "macro_f1",
        "accuracy",
        "different_minus_same_margin",
        "embedding_effective_rank",
        "analyte_probe_balanced_accuracy",
    ):
        audit.check(
            f"outer_finite::{metric}",
            np.isfinite(outer[metric].astype(float).to_numpy()).all(),
        )
    expected_predictions = 598 * 9 * 3 + 500 * 9 * 3 + 98 * 9 * 3
    audit.check(
        "outer_prediction_count",
        len(predictions) == expected_predictions,
        f"observed={len(predictions)} expected={expected_predictions}",
    )
    audit.check(
        "outer_confidence_bounds",
        predictions["confidence"].between(0.0, 1.0).all(),
    )
    audit.check(
        "selective_count",
        len(selective)
        == (
            len(outer)
            + 2 * int((outer["variant"] == "full_domain_aware").sum())
        )
        * len(protocol["calibration_and_ood"]["selective_coverages"]),
    )
    audit.check("temperature_count", len(temperatures) == 10)
    audit.check(
        "temperature_bounds",
        temperatures["temperature"].between(0.05, 20.0).all(),
    )
    expected_rejection_selection = (
        2
        * 5
        * 3
        * len(protocol["calibration_and_ood"]["selective_coverages"])
    )
    audit.check(
        "rejection_selection_count",
        len(rejection_selection) == expected_rejection_selection,
    )
    audit.check(
        "rejection_selection_one_per_outer_subset",
        (
            rejection_selection[
                rejection_selection["selected"].astype(bool)
            ]
            .groupby(["subset", "outer_fold"])["rejection_score"]
            .nunique()
            == 1
        ).all(),
    )
    audit.check(
        "selected_rejection_applied_to_stress",
        len(
            selective[
                (selective["variant"] == "full_domain_aware")
                & (
                    selective["evaluation_subset"]
                    == "field_quality_stress"
                )
                & selective["rejection_selected"].astype(bool)
                & np.isclose(selective["requested_coverage"], 0.8)
            ]
        )
        == 5 * 3,
    )
    audit.check(
        "rejection_decision_count",
        len(rejection_decisions) == 3588,
    )
    audit.check(
        "rejection_decision_selected_scores_match",
        rejection_decisions.groupby(
            ["training_subset", "outer_fold"]
        )["rejection_score"].nunique().eq(1).all(),
    )
    audit.check(
        "rejection_decision_supported_coverage",
        rejection_decisions[
            rejection_decisions["supported"].astype(bool)
        ]
        .groupby(
            [
                "evaluation_subset",
                "outer_fold",
                "declared_seed",
            ]
        )["accepted"]
        .mean()
        .between(0.8, 0.87)
        .all(),
    )
    audit.check("ood_count", len(ood) == 5 * 3 * 3)
    audit.check(
        "ood_one_selected_score_per_fold_seed",
        (
            ood.groupby(["outer_fold", "declared_seed"])["selected"]
            .sum()
            .astype(int)
            == 1
        ).all(),
    )
    audit.check(
        "ood_scores",
        set(ood["score"])
        == {
            "one_minus_max_probability",
            "energy",
            "class_mahalanobis",
        },
    )
    audit.check("attribution_metric_count", len(attribution) == 15)
    audit.check("attribution_peak_count", len(peaks) == 1196 * 30)
    audit.check(
        "attribution_axis_bounds",
        peaks["wavenumber_cm1"].between(400.0, 1800.0).all(),
    )
    audit.check("siamese_control_count", len(siamese) == 45)
    audit.check("siamese_domain_control_count", len(siamese_domain) == 168)
    audit.check(
        "siamese_prediction_count", len(siamese_predictions) == 3588
    )
    audit.check(
        "siamese_diagnostic_count", len(siamese_diagnostics) == 45
    )
    audit.check(
        "siamese_selective_count",
        len(siamese_selective)
        == 45
        * len(protocol["calibration_and_ood"]["selective_coverages"]),
    )
    audit.check("siamese_failures_present", len(siamese_failures) > 0)
    audit.check(
        "siamese_encoder_parameter_count",
        (
            siamese_diagnostics["encoder_parameters"]
            == int(
                protocol["legacy_encoder"]["expected_encoder_parameters"]
            )
        ).all(),
    )
    for metric in (
        "different_minus_same_margin",
        "embedding_effective_rank",
        "correctness_negative_log_likelihood",
        "correctness_brier",
        "correctness_ece_10",
    ):
        audit.check(
            f"siamese_diagnostic_finite::{metric}",
            np.isfinite(
                siamese_diagnostics[metric].astype(float).to_numpy()
            ).all(),
        )
    siamese_analyte_supported = (
        siamese_diagnostics[
            "analyte_heldout_master_probe_n_supported"
        ].astype(int)
        > 0
    )
    audit.check(
        "siamese_analyte_probe_finite_when_supported",
        np.isfinite(
            siamese_diagnostics.loc[
                siamese_analyte_supported,
                "analyte_heldout_master_probe_balanced_accuracy",
            ].astype(float)
        ).all(),
    )

    expected_domain_scenarios = 0
    expected_domain_prediction_rows = 0
    membership: dict[tuple[str, str, str, str], set[str]] = {}
    for subset, filename in (
        ("strict_core", "domain_evaluation_partitions_core.csv"),
        ("quality_pass", "domain_evaluation_partitions_quality.csv"),
    ):
        partitions = pd.read_csv(args.bundle / filename)
        groups = partitions.groupby(
            ["protocol", "domain_type", "heldout_domain"]
        )
        expected_domain_scenarios += groups.ngroups
        for keys, frame in groups:
            test = set(
                frame.loc[
                    frame["partition"] == "test", "observation_uid"
                ].astype(str)
            )
            membership[(subset, *keys)] = test
            expected_domain_prediction_rows += len(test) * 3
    audit.check(
        "domain_metric_count", len(domain) == expected_domain_scenarios * 3
    )
    audit.check(
        "domain_registry_count",
        len(domain_registry) == expected_domain_scenarios * 3,
    )
    audit.check(
        "domain_prediction_count",
        len(domain_predictions) == expected_domain_prediction_rows,
    )
    membership_ok = True
    for keys, frame in domain_predictions.groupby(
        ["subset", "protocol", "domain_type", "heldout_domain"]
    ):
        # Three seeds repeat each authoritative test UID exactly three times.
        counts = frame["observation_uid"].astype(str).value_counts()
        if set(counts.index) != membership[keys] or not (counts == 3).all():
            membership_ok = False
            break
    audit.check("domain_membership_exact", membership_ok)
    supported = domain["n_supported"].astype(int) > 0
    audit.check(
        "domain_metrics_finite_when_supported",
        np.isfinite(
            domain.loc[
                supported, ["balanced_accuracy", "macro_f1", "accuracy"]
            ]
            .astype(float)
            .to_numpy()
        ).all(),
    )
    audit.check("negative_control_count", len(negative) == 10)
    audit.check(
        "negative_controls",
        set(negative["control"])
        == {
            "master_group_analyte_label_permutation",
            "randomized_domain_relationships",
        },
    )
    audit.check(
        "selection_history_all_runs_present",
        selection_history[
            [
                "stage",
                "scope",
                "outer_fold",
                "inner_validation_fold",
                "candidate_id",
                "seed",
            ]
        ]
        .drop_duplicates()
        .shape[0]
        == 240 + 160,
    )
    audit.check(
        "final_history_all_runs_present",
        final_history[
            ["scenario", "candidate_id", "run_seed"]
        ]
        .drop_duplicates()
        .shape[0]
        == 270 + 40 + expected_domain_scenarios * 3 + 10,
    )

    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    groups = manifest["master_sample_id"].astype(str).to_numpy()
    split_ok = True
    for outer_fold in range(5):
        if set(groups[folds != outer_fold]) & set(
            groups[folds == outer_fold]
        ):
            split_ok = False
    audit.check("outer_master_groups_disjoint", split_ok)

    decision = json.loads(
        (args.output_dir / "terminal_decision.json").read_text()
    )
    audit.check(
        "terminal_decision_boolean",
        isinstance(decision.get("successor_supported"), bool),
    )
    audit.check(
        "terminal_primary_domain_pairs_present",
        int(decision.get("primary_domain_pair_count", 0)) > 0,
    )
    audit.check(
        "terminal_primary_domain_siamese_comparison_finite",
        np.isfinite(
            float(
                decision.get(
                    "primary_domain_advantage_vs_siamese", np.nan
                )
            )
        ),
    )
    audit.check(
        "terminal_stress_selective_folds",
        int(decision.get("field_stress_selective_pair_count", 0)) == 5,
    )
    if args.require_clean_rebuild:
        comparison = args.output_dir / "clean_rebuild_comparison.json"
        audit.check("clean_rebuild_exists", comparison.exists())
        if comparison.exists():
            audit.check(
                "clean_rebuild_exact",
                json.loads(comparison.read_text()).get("status")
                == "exact_match",
            )

    audit.require()
    report = {
        "status": "pass",
        "checks": len(audit.rows),
        "passed": len(audit.rows),
        "failed": 0,
        "details": audit.rows,
    }
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "checks": len(audit.rows),
                "selection_fits": len(stage1) + len(stage2),
                "outer_runs": len(registry),
                "domain_runs": len(domain_registry),
                "successor_supported": decision["successor_supported"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
