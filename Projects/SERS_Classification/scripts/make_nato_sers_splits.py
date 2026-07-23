#!/usr/bin/env python3
"""Create deterministic, leakage-aware evaluation splits for NATO SERS data.

The unit of independence in the field-trial archive is the master sample, not
the individual spectrum. Repeated spectra from the same master sample must
therefore remain together for ordinary cross-validation. Domain-transfer
partitions are also emitted for every sensor family and Raman instrument.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def grouped_sample_folds(
    manifest: pd.DataFrame, n_splits: int = 5, seed: int = 2026
) -> pd.DataFrame:
    samples = (
        manifest.groupby(["master_sample_id", "target_analyte"], as_index=False)
        .size()
        .rename(columns={"size": "observation_count"})
        .sort_values(["target_analyte", "master_sample_id"])
    )
    if samples.groupby("master_sample_id")["target_analyte"].nunique().max() != 1:
        raise ValueError("A master sample maps to more than one target analyte")

    rng = np.random.default_rng(seed)
    fold_by_sample: dict[int, int] = {}
    global_load = np.zeros(n_splits, dtype=int)
    for _, group in samples.groupby("target_analyte", sort=True):
        # Greedily balance spectrum counts within each target while keeping
        # every physical sample intact. Randomized tie order is reproducible.
        group = group.copy()
        group["tie_break"] = rng.random(len(group))
        group = group.sort_values(
            ["observation_count", "tie_break"], ascending=[False, True]
        )
        target_load = np.zeros(n_splits, dtype=int)
        for row in group.itertuples(index=False):
            least_target = np.flatnonzero(target_load == target_load.min())
            fold = int(least_target[np.argmin(global_load[least_target])])
            sample_id = int(row.master_sample_id)
            count = int(row.observation_count)
            fold_by_sample[sample_id] = fold
            target_load[fold] += count
            global_load[fold] += count

    result = manifest[
        ["observation_uid", "master_sample_id", "target_analyte", "include_sers_qc_pass"]
    ].copy()
    result["master_sample_id"] = result["master_sample_id"].astype(int)
    result["grouped_sample_fold_5"] = result["master_sample_id"].map(fold_by_sample)
    return result


def domain_partitions(
    manifest: pd.DataFrame, domain_column: str, domain_type: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    summary: dict[str, Any] = {}
    sample_ids = manifest["master_sample_id"]

    for domain in sorted(manifest[domain_column].unique()):
        is_test = manifest[domain_column] == domain
        test_sample_ids = set(sample_ids[is_test].astype(int))
        test_classes = set(manifest.loc[is_test, "target_analyte"])

        for protocol in ("domain_only", "domain_and_sample"):
            part = manifest[
                ["observation_uid", "master_sample_id", "target_analyte", "include_sers_qc_pass"]
            ].copy()
            part.insert(0, "heldout_domain", domain)
            part.insert(0, "domain_type", domain_type)
            part.insert(0, "protocol", protocol)

            if protocol == "domain_only":
                part["partition"] = np.where(is_test, "test", "train")
            else:
                same_sample = sample_ids.astype(int).isin(test_sample_ids)
                part["partition"] = np.select(
                    [is_test, (~is_test) & same_sample],
                    ["test", "excluded_same_master_sample"],
                    default="train",
                )

            train_classes = set(
                part.loc[part["partition"] == "train", "target_analyte"]
            )
            part["test_class_supported"] = np.where(
                part["partition"] == "test",
                part["target_analyte"].isin(train_classes),
                pd.NA,
            )
            frames.append(part)

            counts = part["partition"].value_counts().to_dict()
            supported_test = part[
                (part["partition"] == "test")
                & (part["target_analyte"].isin(train_classes))
            ]
            summary[f"{domain_type}:{domain}:{protocol}"] = {
                "partition_counts": {key: int(value) for key, value in counts.items()},
                "test_master_samples": len(test_sample_ids),
                "test_classes": sorted(test_classes),
                "train_classes": sorted(train_classes),
                "unsupported_test_classes": sorted(test_classes - train_classes),
                "supported_test_observations": int(len(supported_test)),
            }

    return pd.concat(frames, ignore_index=True), summary


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repository / "Workspace" / "nato_sers_field_trial",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    manifest = pd.read_csv(input_dir / "sers_core_manifest.csv")

    grouped = grouped_sample_folds(manifest)
    grouped.to_csv(input_dir / "grouped_sample_cv_assignments.csv", index=False)

    sensor_partitions, sensor_summary = domain_partitions(
        manifest, "sensor_family", "sensor_family"
    )
    instrument_partitions, instrument_summary = domain_partitions(
        manifest, "instrument", "instrument"
    )
    domain_frame = pd.concat(
        [sensor_partitions, instrument_partitions], ignore_index=True
    )
    domain_frame.to_csv(input_dir / "domain_evaluation_partitions.csv", index=False)

    fold_summary = {
        str(fold): {
            "observations": int(len(group)),
            "master_samples": int(group["master_sample_id"].nunique()),
            "target_observations": {
                key: int(value)
                for key, value in group["target_analyte"].value_counts().items()
            },
            "target_master_samples": {
                key: int(value)
                for key, value in group.drop_duplicates("master_sample_id")[
                    "target_analyte"
                ].value_counts().items()
            },
        }
        for fold, group in grouped.groupby("grouped_sample_fold_5")
    }
    summary = {
        "grouped_sample_cv": {
            "description": (
                "Five deterministic folds stratified at the master-sample level; "
                "all repeated spectra of one sample remain in one fold."
            ),
            "seed": 2026,
            "folds": fold_summary,
        },
        "domain_protocols": {
            "domain_only": (
                "Test on one held-out sensor/instrument; other-domain spectra of "
                "the same master sample may remain in training. Use as a paired "
                "domain-shift diagnostic, not as the deployment headline."
            ),
            "domain_and_sample": (
                "Test on one held-out sensor/instrument and exclude every test "
                "master sample from training. Unsupported target classes are "
                "reported rather than folded into balanced accuracy."
            ),
        },
        "partitions": {**sensor_summary, **instrument_summary},
    }
    (input_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
