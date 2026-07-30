#!/usr/bin/env python3
"""Validate every frozen NATO SERS open-world partition invariant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import freeze_sers_open_world_partitions as freezer
import sers_baseline_common as baseline
import sers_classical_benchmark_common as classical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("Workspace/nato_sers_field_trial/preprocessing_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Workspace/sers_open_world_shift/open_world_v1"),
    )
    return parser.parse_args()


def add(
    checks: list[dict[str, Any]], name: str, passed: bool, detail: str
) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def scenario_complete(
    frame: pd.DataFrame, columns: list[str], expected_rows: int
) -> tuple[bool, str]:
    counts = frame.groupby(columns).agg(
        rows=("observation_uid", "size"),
        unique_uids=("observation_uid", "nunique"),
    )
    passed = bool(
        (counts["rows"] == expected_rows).all()
        and (counts["unique_uids"] == expected_rows).all()
    )
    return passed, (
        f"scenarios={len(counts)}, row_range={counts.rows.min()}.."
        f"{counts.rows.max()}, uid_range={counts.unique_uids.min()}.."
        f"{counts.unique_uids.max()}"
    )


def master_disjoint(
    frame: pd.DataFrame,
    scenario_columns: list[str],
    left: str,
    right: str,
) -> tuple[bool, int]:
    overlaps = 0
    for _, group in frame.groupby(scenario_columns, sort=False):
        left_groups = set(
            group.loc[group["partition"] == left, "master_sample_id"]
        )
        right_groups = set(
            group.loc[group["partition"] == right, "master_sample_id"]
        )
        overlaps += len(left_groups & right_groups)
    return overlaps == 0, overlaps


def main() -> None:
    args = parse_args()
    output = args.output_dir
    protocol = freezer.load_protocol(output / "predeclared_protocol.json")
    dataset = baseline.load_nato_dataset(args.bundle)
    manifest = dataset.manifest
    outer = pd.read_csv(output / "open_set_outer_partitions.csv")
    threshold = pd.read_csv(
        output / "known_threshold_crossfit_partitions.csv"
    )
    surrogate = pd.read_csv(
        output / "surrogate_unknown_inner_partitions.csv"
    )
    checks: list[dict[str, Any]] = []
    expected_rows = len(manifest)

    for name, frame, columns, scenarios, rows in (
        (
            "outer",
            outer,
            ["held_unknown", "outer_fold"],
            30,
            30 * expected_rows,
        ),
        (
            "threshold",
            threshold,
            ["held_unknown", "outer_fold", "calibration_fold"],
            120,
            120 * expected_rows,
        ),
        (
            "surrogate",
            surrogate,
            [
                "held_unknown",
                "outer_fold",
                "inner_fold",
                "surrogate_unknown",
            ],
            600,
            600 * expected_rows,
        ),
    ):
        complete, detail = scenario_complete(frame, columns, expected_rows)
        add(checks, f"{name}_scenario_completeness", complete, detail)
        add(
            checks,
            f"{name}_total_rows",
            len(frame) == rows,
            f"expected={rows}, actual={len(frame)}",
        )
        add(
            checks,
            f"{name}_scenario_count",
            frame.groupby(columns).ngroups == scenarios,
            f"expected={scenarios}, actual={frame.groupby(columns).ngroups}",
        )

    held = set(protocol["classes"]["held_unknown_chemicals"])
    actual_held = set(outer["held_unknown"].astype(str))
    add(
        checks,
        "six_true_unknowns",
        actual_held == held and "blank" not in actual_held,
        str(sorted(actual_held)),
    )
    surrogate_pairs_valid = bool(
        (surrogate["held_unknown"] != surrogate["surrogate_unknown"]).all()
        and not (surrogate["surrogate_unknown"] == "blank").any()
    )
    add(
        checks,
        "surrogate_unknown_validity",
        surrogate_pairs_valid,
        "held differs from surrogate and blank is never surrogate",
    )

    outer_expected = {
        "known_development_quality",
        "known_quality_test",
        "unknown_quality",
        "known_stress",
        "unknown_stress",
    }
    threshold_expected = {
        "train_known",
        "calibration_known",
        "excluded_outer_known_test",
        "locked_held_unknown",
        "locked_stress",
    }
    surrogate_expected = {
        "train_known",
        "validation_known",
        "validation_surrogate_unknown",
        "excluded_surrogate_training",
        "excluded_outer_known_test",
        "locked_held_unknown",
        "locked_stress",
    }
    for name, actual, expected in (
        ("outer", set(outer["partition"]), outer_expected),
        ("threshold", set(threshold["partition"]), threshold_expected),
        ("surrogate", set(surrogate["partition"]), surrogate_expected),
    ):
        add(
            checks,
            f"{name}_partition_vocabulary",
            actual == expected,
            str(sorted(actual)),
        )

    for name, frame, columns, pairs in (
        (
            "outer",
            outer,
            ["held_unknown", "outer_fold"],
            [
                ("known_development_quality", "known_quality_test"),
                ("known_development_quality", "unknown_quality"),
            ],
        ),
        (
            "threshold",
            threshold,
            ["held_unknown", "outer_fold", "calibration_fold"],
            [
                ("train_known", "calibration_known"),
                ("train_known", "excluded_outer_known_test"),
                ("calibration_known", "excluded_outer_known_test"),
            ],
        ),
        (
            "surrogate",
            surrogate,
            [
                "held_unknown",
                "outer_fold",
                "inner_fold",
                "surrogate_unknown",
            ],
            [
                ("train_known", "validation_known"),
                ("train_known", "validation_surrogate_unknown"),
                ("validation_known", "validation_surrogate_unknown"),
                ("train_known", "excluded_outer_known_test"),
            ],
        ),
    ):
        for left, right in pairs:
            disjoint, overlaps = master_disjoint(frame, columns, left, right)
            add(
                checks,
                f"{name}_{left}_vs_{right}_master_disjoint",
                disjoint,
                f"overlapping_masters={overlaps}",
            )

    true_unknown_leak = surrogate[
        (surrogate["target_analyte"] == surrogate["held_unknown"])
        & surrogate["partition"].isin(
            ["train_known", "validation_known", "validation_surrogate_unknown"]
        )
    ]
    add(
        checks,
        "true_unknown_never_selection",
        true_unknown_leak.empty,
        f"leaked_rows={len(true_unknown_leak)}",
    )
    surrogate_train_leak = surrogate[
        (surrogate["target_analyte"] == surrogate["surrogate_unknown"])
        & (surrogate["partition"] == "train_known")
    ]
    add(
        checks,
        "surrogate_unknown_never_train",
        surrogate_train_leak.empty,
        f"leaked_rows={len(surrogate_train_leak)}",
    )
    stress_leak = pd.concat(
        [
            threshold[
                threshold["partition"].isin(
                    ["train_known", "calibration_known"]
                )
                & threshold["field_quality_stress"].astype(bool)
            ],
            surrogate[
                surrogate["partition"].isin(
                    [
                        "train_known",
                        "validation_known",
                        "validation_surrogate_unknown",
                    ]
                )
                & surrogate["field_quality_stress"].astype(bool)
            ],
        ]
    )
    add(
        checks,
        "stress_never_selection_or_training",
        stress_leak.empty,
        f"leaked_rows={len(stress_leak)}",
    )

    threshold_coverage_ok = True
    threshold_coverage_detail: list[str] = []
    for (held_unknown, outer_fold), group in threshold.groupby(
        ["held_unknown", "outer_fold"]
    ):
        eligible = group[
            ~group["target_analyte"].eq(held_unknown)
            & group["include_sers_qc_pass"].astype(bool)
            & ~group["grouped_sample_fold_5"].eq(outer_fold)
        ]
        counts = eligible.groupby("observation_uid")["partition"].agg(
            calibration=lambda values: int((values == "calibration_known").sum()),
            train=lambda values: int((values == "train_known").sum()),
        )
        good = bool(
            (counts["calibration"] == 1).all() and (counts["train"] == 3).all()
        )
        threshold_coverage_ok &= good
        if not good:
            threshold_coverage_detail.append(f"{held_unknown}/o{outer_fold}")
    add(
        checks,
        "threshold_crossfit_exact_coverage",
        threshold_coverage_ok,
        f"failed={threshold_coverage_detail}",
    )

    surrogate_coverage_ok = True
    failed_surrogate: list[str] = []
    for keys, group in surrogate.groupby(
        ["held_unknown", "outer_fold", "surrogate_unknown"]
    ):
        held_unknown, outer_fold, surrogate_unknown = keys
        eligible = group[
            group["target_analyte"].eq(surrogate_unknown)
            & group["include_sers_qc_pass"].astype(bool)
            & ~group["grouped_sample_fold_5"].eq(outer_fold)
        ]
        counts = eligible.groupby("observation_uid")["partition"].agg(
            validation=lambda values: int(
                (values == "validation_surrogate_unknown").sum()
            )
        )
        good = bool((counts["validation"] == 1).all())
        surrogate_coverage_ok &= good
        if not good:
            failed_surrogate.append("/".join(map(str, keys)))
    add(
        checks,
        "surrogate_unknown_exact_crossfit_coverage",
        surrogate_coverage_ok,
        f"failed={failed_surrogate}",
    )

    class_support_ok = True
    support_failures: list[str] = []
    for keys, group in surrogate.groupby(
        [
            "held_unknown",
            "outer_fold",
            "inner_fold",
            "surrogate_unknown",
        ]
    ):
        train = group[group["partition"] == "train_known"]
        validation = group[group["partition"] == "validation_known"]
        expected_known = (
            set(protocol["classes"]["held_unknown_chemicals"])
            | {"blank"}
        ) - {keys[0], keys[3]}
        if (
            set(train["target_analyte"]) != expected_known
            or set(validation["target_analyte"]) != expected_known
        ):
            class_support_ok = False
            support_failures.append("/".join(map(str, keys)))
    add(
        checks,
        "all_surrogate_scenarios_have_expected_known_classes",
        class_support_ok,
        f"failed={support_failures[:10]}",
    )

    copied_domains_ok = True
    domain_details: list[str] = []
    for filename in protocol["immutable_input"]["domain_partition_files"]:
        source = args.bundle / filename
        copied = output / filename
        same = source.read_bytes() == copied.read_bytes()
        copied_domains_ok &= same
        domain_details.append(f"{filename}={same}")
    add(
        checks,
        "domain_partitions_byte_identical",
        copied_domains_ok,
        ", ".join(domain_details),
    )

    hashes = json.loads((output / "artifact_hashes.json").read_text())
    mismatches = [
        relative
        for relative, expected in hashes.items()
        if classical.sha256_file(output / relative) != expected
    ]
    add(
        checks,
        "artifact_hashes",
        not mismatches,
        f"mismatches={mismatches}",
    )
    result = {
        "protocol_version": protocol["protocol_version"],
        "status": "pass" if all(item["pass"] for item in checks) else "fail",
        "checks": checks,
    }
    classical.write_json(output / "validation_report.json", result)
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise RuntimeError(
            f"Partition validation failed: "
            f"{[item for item in checks if not item['pass']]}"
        )


if __name__ == "__main__":
    main()
