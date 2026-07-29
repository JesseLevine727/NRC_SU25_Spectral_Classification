#!/usr/bin/env python3
"""Independently validate the NATO SERS classical benchmark v2 bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sers_classical_benchmark_common as common


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
            {
                "check": name,
                "passed": bool(condition),
                "detail": detail,
            }
        )

    def require(self) -> None:
        failures = [row for row in self.rows if not row["passed"]]
        if failures:
            raise AssertionError(json.dumps(failures, indent=2))


def finite_metrics(
    audit: Audit, frame: pd.DataFrame, columns: list[str], stem: str
) -> None:
    for column in columns:
        audit.check(
            f"{stem}_{column}_finite",
            column in frame
            and np.isfinite(frame[column].astype(float).to_numpy()).all(),
        )


def finite_when_supported(
    audit: Audit, frame: pd.DataFrame, columns: list[str], stem: str
) -> None:
    supported = frame["n_supported"].astype(int) > 0
    for column in columns:
        values = frame.loc[supported, column].astype(float).to_numpy()
        audit.check(
            f"{stem}_{column}_finite_when_supported",
            column in frame and np.isfinite(values).all(),
        )
        unsupported_values = frame.loc[~supported, column]
        audit.check(
            f"{stem}_{column}_undefined_without_support",
            unsupported_values.isna().all(),
        )


def main() -> None:
    args = parse_args()
    protocol = common.load_protocol(args.protocol)
    dataset = common.baseline.load_nato_dataset(args.bundle)
    manifest = dataset.manifest
    audit = Audit()

    common.baseline.verify_hash_catalog(args.bundle)
    copied_protocol = args.output_dir / "predeclared_protocol.json"
    audit.check("copied_protocol_exists", copied_protocol.exists())
    audit.check(
        "copied_protocol_exact",
        copied_protocol.exists()
        and copied_protocol.read_bytes() == args.protocol.read_bytes(),
    )

    hashes_path = args.output_dir / "artifact_hashes.json"
    audit.check("artifact_hash_catalog_exists", hashes_path.exists())
    if hashes_path.exists():
        hashes = json.loads(hashes_path.read_text())
        for relative_path, expected in hashes.items():
            path = args.output_dir / relative_path
            audit.check(f"hash_file_exists::{relative_path}", path.exists())
            if path.exists():
                audit.check(
                    f"hash_exact::{relative_path}",
                    common.sha256_file(path) == expected,
                )

    required = protocol["required_outputs"]
    for name in required:
        audit.check(
            f"required_output::{name}",
            name == "validation_report.json"
            or (args.output_dir / name).exists(),
            "created by this validator" if name == "validation_report.json" else "",
        )
    for name in (
        "field_stress_ood_metrics.csv",
        "outer_domain_slices.csv",
        "domain_per_class_metrics.csv",
        "domain_selective_metrics.csv",
        "uncertainty_summary.csv",
        "domain_summary.csv",
        "learning_curve_summary.csv",
        "confusion_matrices.json",
        "figures/classical_benchmark_summary.pdf",
        "figures/classical_benchmark_summary.png",
        "figures/selective_domain_summary.pdf",
        "figures/selective_domain_summary.png",
    ):
        audit.check(f"derived_output::{name}", (args.output_dir / name).exists())

    candidates = pd.read_csv(args.output_dir / "candidate_registry.csv")
    inner = pd.read_csv(args.output_dir / "inner_fold_metrics.csv")
    selection = pd.read_csv(args.output_dir / "outer_selection.csv")
    outer = pd.read_csv(args.output_dir / "outer_metrics.csv")
    predictions = pd.read_csv(args.output_dir / "predictions.csv")
    per_class = pd.read_csv(args.output_dir / "per_class_metrics.csv")
    calibration = pd.read_csv(args.output_dir / "calibration_metrics.csv")
    selective = pd.read_csv(args.output_dir / "selective_metrics.csv")
    domain = pd.read_csv(args.output_dir / "domain_metrics.csv")
    domain_predictions = pd.read_csv(args.output_dir / "domain_predictions.csv")
    learning = pd.read_csv(args.output_dir / "learning_curve_metrics.csv")
    negative = pd.read_csv(args.output_dir / "negative_control_metrics.csv")

    expected_candidates = len(common.candidate_grid(protocol))
    audit.check("candidate_count", len(candidates) == expected_candidates)
    audit.check(
        "candidate_ids_unique",
        candidates["candidate_id"].nunique() == expected_candidates,
    )
    audit.check(
        "candidate_families",
        set(candidates["model_family"])
        == set(protocol["candidate_grid"]),
    )
    audit.check(
        "candidate_representations",
        set(candidates["representation"])
        == set(protocol["immutable_input"]["representations"]),
    )

    audit.check(
        "inner_metric_count",
        len(inner) == expected_candidates * 2 * 5 * 4,
        f"observed={len(inner)}",
    )
    audit.check("all_inner_candidates_pass", (inner["status"] == "pass").all())
    audit.check(
        "inner_unique_runs",
        not inner.duplicated(
            [
                "subset",
                "outer_fold",
                "inner_validation_fold",
                "candidate_id",
            ]
        ).any(),
    )
    finite_metrics(
        audit,
        inner,
        ["balanced_accuracy", "macro_f1", "accuracy"],
        "inner",
    )

    audit.check("outer_selection_count", len(selection) == 10)
    audit.check(
        "outer_selection_unique",
        not selection.duplicated(["subset", "outer_fold"]).any(),
    )
    selected_ids = set(selection["candidate_id"])
    audit.check(
        "selected_candidates_registered",
        selected_ids.issubset(set(candidates["candidate_id"])),
    )
    champions = json.loads(
        (args.output_dir / "global_champions.json").read_text()
    )
    audit.check(
        "global_champion_subsets",
        set(champions) == {"strict_core", "quality_pass"},
    )

    audit.check("outer_metric_count", len(outer) == 15)
    audit.check(
        "outer_scenarios",
        set(outer["evaluation_subset"])
        == {"strict_core", "quality_pass", "field_quality_stress"},
    )
    finite_metrics(
        audit,
        outer,
        [
            "balanced_accuracy",
            "macro_f1",
            "accuracy",
            "negative_log_likelihood",
            "brier_multiclass",
            "expected_calibration_error_10",
        ],
        "outer",
    )
    audit.check("prediction_count", len(predictions) == 1196)
    for training_subset, evaluation_subset, expected in (
        ("strict_core", "strict_core", 598),
        ("quality_pass", "quality_pass", 500),
        ("quality_pass", "field_quality_stress", 98),
    ):
        frame = predictions[
            (predictions["training_subset"] == training_subset)
            & (predictions["evaluation_subset"] == evaluation_subset)
        ]
        audit.check(
            f"prediction_rows::{training_subset}::{evaluation_subset}",
            len(frame) == expected,
        )
        audit.check(
            f"prediction_uid_unique::{training_subset}::{evaluation_subset}",
            frame["observation_uid"].nunique() == expected,
        )
    audit.check(
        "stress_trained_only_on_quality",
        (
            predictions.loc[
                predictions["evaluation_subset"] == "field_quality_stress",
                "training_subset",
            ]
            == "quality_pass"
        ).all(),
    )
    audit.check(
        "confidence_bounds",
        predictions["confidence"].between(0.0, 1.0).all(),
    )
    audit.check("per_class_nonempty", len(per_class) > 0)
    audit.check("calibration_count", len(calibration) == 30)
    audit.check(
        "calibration_methods",
        set(calibration["calibration"])
        == {"uncalibrated", "temperature_scaled"},
    )
    audit.check(
        "selective_count",
        len(selective)
        == 15 * len(protocol["evaluation"]["selective_coverages"]),
    )

    expected_domain_scenarios = 0
    membership: dict[tuple[str, str, str, str], set[str]] = {}
    for subset, filename in (
        ("strict_core", "domain_evaluation_partitions_core.csv"),
        ("quality_pass", "domain_evaluation_partitions_quality.csv"),
    ):
        partitions = pd.read_csv(args.bundle / filename)
        expected_domain_scenarios += partitions.groupby(
            ["protocol", "domain_type", "heldout_domain"]
        ).ngroups
        for keys, frame in partitions.groupby(
            ["protocol", "domain_type", "heldout_domain"]
        ):
            membership[(subset, *keys)] = set(
                frame.loc[
                    frame["partition"] == "test", "observation_uid"
                ].astype(str)
            )
    audit.check(
        "domain_metric_count", len(domain) == expected_domain_scenarios
    )
    finite_when_supported(
        audit,
        domain,
        ["balanced_accuracy", "macro_f1", "accuracy"],
        "domain",
    )
    domain_membership_ok = True
    for keys, frame in domain_predictions.groupby(
        ["subset", "protocol", "domain_type", "heldout_domain"]
    ):
        if set(frame["observation_uid"].astype(str)) != membership[keys]:
            domain_membership_ok = False
            break
    audit.check("domain_prediction_membership_exact", domain_membership_ok)

    expected_learning = (
        2
        * 5
        * len(protocol["learning_curves"]["training_group_fractions"])
        * len(protocol["learning_curves"]["repeats"])
    )
    audit.check("learning_curve_count", len(learning) == expected_learning)
    finite_metrics(
        audit,
        learning,
        ["balanced_accuracy", "macro_f1", "accuracy"],
        "learning",
    )
    audit.check("negative_control_count", len(negative) == 5)
    finite_metrics(
        audit,
        negative,
        ["balanced_accuracy", "macro_f1", "accuracy"],
        "negative",
    )

    # Prove the split contract directly from the authoritative manifest.
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    groups = manifest["master_sample_id"].astype(str).to_numpy()
    split_ok = True
    for outer_fold in range(5):
        train_groups = set(groups[folds != outer_fold])
        test_groups = set(groups[folds == outer_fold])
        if train_groups & test_groups:
            split_ok = False
    audit.check("outer_master_groups_disjoint", split_ok)
    audit.check(
        "strict_quality_stress_counts",
        len(manifest) == 598
        and int(manifest["include_sers_qc_pass"].sum()) == 500
        and int(manifest["field_quality_stress"].sum()) == 98,
    )

    if args.require_clean_rebuild:
        rebuild_path = args.output_dir / "clean_rebuild_comparison.json"
        audit.check("clean_rebuild_comparison_exists", rebuild_path.exists())
        if rebuild_path.exists():
            rebuild = json.loads(rebuild_path.read_text())
            audit.check(
                "clean_rebuild_exact", rebuild.get("status") == "exact_match"
            )

    audit.require()
    report = {
        "status": "pass",
        "checks": len(audit.rows),
        "passed": sum(row["passed"] for row in audit.rows),
        "failed": 0,
        "details": audit.rows,
    }
    common.write_json(args.output_dir / "validation_report.json", report)
    print(
        json.dumps(
            {
                "status": "PASS",
                "checks": report["checks"],
                "candidates": expected_candidates,
                "inner_fits": len(inner),
                "outer_predictions": len(predictions),
                "domain_scenarios": len(domain),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
