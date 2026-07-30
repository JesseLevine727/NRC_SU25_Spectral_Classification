#!/usr/bin/env python3
"""Validate counts, leakage boundaries, metrics, and hashes for the RF addendum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sers_random_forest_common as common


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
        "--bundle",
        type=Path,
        default=Path("Workspace/nato_sers_field_trial/preprocessing_v2"),
    )
    return parser.parse_args()


def check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def main() -> None:
    args = parse_args()
    output = args.output_dir
    checks: list[dict[str, Any]] = []
    protocol = common.load_protocol(output / "predeclared_protocol.json")
    registry = pd.read_csv(output / "candidate_registry.csv")
    inner = pd.read_csv(output / "inner_fold_metrics.csv")
    selection = pd.read_csv(output / "outer_selection.csv")
    outer = pd.read_csv(output / "outer_metrics.csv")
    predictions = pd.read_csv(output / "predictions.csv")
    domains = pd.read_csv(output / "domain_metrics.csv")
    negative = pd.read_csv(output / "negative_control_metrics.csv")
    impurity = pd.read_csv(output / "impurity_importance.csv")
    band = pd.read_csv(output / "band_permutation_importance.csv")
    dataset = common.baseline.load_nato_dataset(args.bundle)
    manifest = dataset.manifest

    check(checks, "candidate_count", len(registry) == 72, f"n={len(registry)}")
    check(checks, "inner_fit_count", len(inner) == 2880, f"n={len(inner)}")
    check(
        checks,
        "all_inner_pass",
        bool((inner["status"] == "pass").all()),
        f"failures={(inner['status'] != 'pass').sum()}",
    )
    check(checks, "selection_count", len(selection) == 10, f"n={len(selection)}")
    check(checks, "outer_metric_count", len(outer) == 45, f"n={len(outer)}")
    check(
        checks,
        "outer_prediction_count",
        len(predictions) == 3588,
        f"n={len(predictions)}",
    )
    check(checks, "domain_metric_count", len(domains) == 168, f"n={len(domains)}")
    check(checks, "negative_count", len(negative) == 15, f"n={len(negative)}")
    check(
        checks,
        "impurity_count",
        len(impurity) == 5 * 3 * 1401,
        f"n={len(impurity)}",
    )
    check(checks, "band_count", len(band) == 5 * 71, f"n={len(band)}")

    bounded = True
    for frame in (inner, outer, domains, negative):
        for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
            values = frame[metric].dropna().to_numpy(float)
            bounded &= bool(((values >= 0.0) & (values <= 1.0)).all())
    check(checks, "metrics_bounded", bounded, "classification metrics in [0,1]")

    unique_master_label = (
        manifest.groupby("master_sample_id")["target_analyte"].nunique().max()
        == 1
    )
    check(
        checks,
        "one_label_per_master",
        bool(unique_master_label),
        "max labels per master checked",
    )
    folds = manifest["grouped_sample_fold_5"].astype(int)
    fold_disjoint = True
    for outer_fold in range(5):
        train = set(manifest.loc[folds != outer_fold, "master_sample_id"])
        test = set(manifest.loc[folds == outer_fold, "master_sample_id"])
        fold_disjoint &= not bool(train & test)
    check(
        checks,
        "outer_master_disjoint",
        fold_disjoint,
        "all five frozen master folds",
    )
    expected_subsets = {
        ("strict_core", "strict_core"),
        ("quality_pass", "quality_pass"),
        ("quality_pass", "field_quality_stress"),
    }
    actual_subsets = set(
        outer[["training_subset", "evaluation_subset"]]
        .itertuples(index=False, name=None)
    )
    check(
        checks,
        "locked_subsets_present",
        actual_subsets == expected_subsets,
        str(sorted(actual_subsets)),
    )
    stress_not_in_search = "field_quality_stress" not in set(inner["subset"])
    check(
        checks,
        "stress_excluded_from_selection",
        stress_not_in_search,
        "search subsets inspected",
    )
    domain_not_in_search = not any(
        name in inner.columns
        for name in ("heldout_domain", "domain_type", "protocol")
    )
    check(
        checks,
        "domain_excluded_from_selection",
        domain_not_in_search,
        "search columns inspected",
    )
    seed_counts = outer.groupby(
        ["training_subset", "evaluation_subset", "outer_fold"]
    )["final_seed"].nunique()
    check(
        checks,
        "three_outer_seeds",
        bool((seed_counts == 3).all()),
        f"range={seed_counts.min()}..{seed_counts.max()}",
    )

    hashes = json.loads((output / "artifact_hashes.json").read_text())
    mismatches = [
        relative
        for relative, expected in hashes.items()
        if common.sha256_file(output / relative) != expected
    ]
    check(
        checks,
        "artifact_hashes",
        not mismatches,
        f"mismatches={mismatches}",
    )
    report = {
        "protocol_version": protocol["protocol_version"],
        "status": "pass" if all(item["pass"] for item in checks) else "fail",
        "checks": checks,
    }
    common.write_json(output / "validation_report.json", report)
    if report["status"] != "pass":
        failed = [item for item in checks if not item["pass"]]
        raise RuntimeError(f"Validation failed: {failed}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
