#!/usr/bin/env python3
"""Build the locked, metadata-only P13 support registries."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

PROTOCOL_VERSION = "nato-sers-p13-v1-locked"
APPROVAL_DATE = "2026-09-04"
CONFIRMATORY_MINIMUM = 3
EXPLORATORY_MINIMUM = 2
MINIMUM_SOURCE_INSTRUMENTS = 2

DOMAIN_COLUMNS = [
    "protocol_version",
    "domain_id",
    "station",
    "substrate_family",
    "held_instrument",
    "station_class_count",
    "held_rows",
    "held_masters",
    "held_class_support",
    "minimum_held_masters_per_class",
    "source_pool_rows",
    "source_pool_masters",
    "source_pool_class_support",
    "minimum_source_pool_masters_per_class",
    "minimum_training_masters_per_class_across_outer_splits",
    "minimum_source_instruments_per_class_across_outer_splits",
    "paired_class_support",
    "minimum_paired_source_held_masters_per_class",
    "support_tier",
    "support_reason",
]

CROSSOVER_COLUMNS = [
    "protocol_version",
    "crossover_block_id",
    "station",
    "target_analyte",
    "substrate_a",
    "substrate_b",
    "instrument_a",
    "instrument_b",
    "physical_masters",
    "support_tier",
    "support_reason",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count_map(frame: pd.DataFrame, classes: list[str], column: str) -> dict[str, int]:
    return {
        target: int(frame.loc[frame["target_analyte"].eq(target), column].nunique())
        for target in classes
    }


def _serialize_counts(counts: dict[str, int]) -> str:
    return "|".join(f"{target}:{count}" for target, count in counts.items())


def _validate_inputs(manifest: pd.DataFrame, splits: pd.DataFrame) -> None:
    manifest_columns = {
        "master_sample_id",
        "station",
        "target_analyte",
        "sensor_family",
        "instrument",
    }
    split_columns = {"outer_repeat", "station", "master_sample_id", "outer_fold"}
    if missing := sorted(manifest_columns - set(manifest.columns)):
        raise ValueError(f"Primary manifest is missing columns: {missing}")
    if missing := sorted(split_columns - set(splits.columns)):
        raise ValueError(f"P02 split registry is missing columns: {missing}")
    if len(manifest) != 598 or manifest["master_sample_id"].nunique() != 69:
        raise ValueError("P13 requires the frozen 598-spectrum, 69-master population.")
    if splits["outer_repeat"].nunique() != 5:
        raise ValueError("P13 requires all five P02 outer repeats.")
    folds_per_repeat = splits.groupby(["station", "outer_repeat"])["outer_fold"].nunique()
    if not folds_per_repeat.eq(4).all():
        raise ValueError("P13 requires four P02 outer folds per station and repeat.")


def _support_tier(
    *, held_minimum: int, training_minimum: int, instrument_minimum: int, paired_minimum: int
) -> tuple[str, str]:
    confirmatory = (
        held_minimum >= CONFIRMATORY_MINIMUM
        and training_minimum >= CONFIRMATORY_MINIMUM
        and instrument_minimum >= MINIMUM_SOURCE_INSTRUMENTS
        and paired_minimum >= CONFIRMATORY_MINIMUM
    )
    if confirmatory:
        return "confirmatory", "confirmatory_supported"

    exploratory = (
        held_minimum >= EXPLORATORY_MINIMUM
        and training_minimum >= EXPLORATORY_MINIMUM
    )
    if exploratory:
        return "exploratory_low_support", "below_confirmatory_three_master_threshold"

    reasons = []
    if held_minimum < EXPLORATORY_MINIMUM:
        reasons.append("held_masters_per_class_lt_2")
    if training_minimum < EXPLORATORY_MINIMUM:
        reasons.append("training_masters_per_class_lt_2")
    if instrument_minimum < MINIMUM_SOURCE_INSTRUMENTS:
        reasons.append("source_instruments_per_class_lt_2")
    if paired_minimum < EXPLORATORY_MINIMUM:
        reasons.append("paired_source_held_masters_per_class_lt_2")
    return "unsupported_by_design", "|".join(reasons)


def build_domain_support(manifest: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    domain_index = 0
    split_keys = (
        splits[["outer_repeat", "station", "master_sample_id", "outer_fold"]]
        .drop_duplicates()
        .sort_values(["station", "outer_repeat", "outer_fold", "master_sample_id"])
    )

    for (station, substrate), group in manifest.groupby(
        ["station", "sensor_family"], sort=True
    ):
        classes = sorted(
            manifest.loc[manifest["station"].eq(station), "target_analyte"].unique()
        )
        if len(classes) != 3:
            raise ValueError(f"Station {station} does not have exactly three target classes.")

        for held_instrument in sorted(group["instrument"].unique()):
            domain_index += 1
            held = group[group["instrument"].eq(held_instrument)]
            source_pool = group[~group["instrument"].eq(held_instrument)]
            held_counts = _count_map(held, classes, "master_sample_id")
            source_counts = _count_map(source_pool, classes, "master_sample_id")

            held_master_ids = {
                target: set(
                    held.loc[held["target_analyte"].eq(target), "master_sample_id"]
                )
                for target in classes
            }
            source_master_ids = {
                target: set(
                    source_pool.loc[
                        source_pool["target_analyte"].eq(target), "master_sample_id"
                    ]
                )
                for target in classes
            }
            paired_counts = {
                target: len(held_master_ids[target] & source_master_ids[target])
                for target in classes
            }

            training_minima: list[int] = []
            instrument_minima: list[int] = []
            station_splits = split_keys[split_keys["station"].eq(station)]
            for (_, _), split in station_splits.groupby(
                ["outer_repeat", "outer_fold"], sort=True
            ):
                evaluation_masters = set(split["master_sample_id"])
                training = source_pool[
                    ~source_pool["master_sample_id"].isin(evaluation_masters)
                ]
                training_counts = _count_map(training, classes, "master_sample_id")
                source_instrument_counts = _count_map(training, classes, "instrument")
                training_minima.append(min(training_counts.values()))
                instrument_minima.append(min(source_instrument_counts.values()))

            held_minimum = min(held_counts.values())
            training_minimum = min(training_minima)
            instrument_minimum = min(instrument_minima)
            paired_minimum = min(paired_counts.values())
            support_tier, support_reason = _support_tier(
                held_minimum=held_minimum,
                training_minimum=training_minimum,
                instrument_minimum=instrument_minimum,
                paired_minimum=paired_minimum,
            )
            rows.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "domain_id": f"P13-DOM-{domain_index:03d}",
                    "station": station,
                    "substrate_family": substrate,
                    "held_instrument": held_instrument,
                    "station_class_count": len(classes),
                    "held_rows": len(held),
                    "held_masters": held["master_sample_id"].nunique(),
                    "held_class_support": _serialize_counts(held_counts),
                    "minimum_held_masters_per_class": held_minimum,
                    "source_pool_rows": len(source_pool),
                    "source_pool_masters": source_pool["master_sample_id"].nunique(),
                    "source_pool_class_support": _serialize_counts(source_counts),
                    "minimum_source_pool_masters_per_class": min(source_counts.values()),
                    "minimum_training_masters_per_class_across_outer_splits": (
                        training_minimum
                    ),
                    "minimum_source_instruments_per_class_across_outer_splits": (
                        instrument_minimum
                    ),
                    "paired_class_support": _serialize_counts(paired_counts),
                    "minimum_paired_source_held_masters_per_class": paired_minimum,
                    "support_tier": support_tier,
                    "support_reason": support_reason,
                }
            )

    frame = pd.DataFrame(rows, columns=DOMAIN_COLUMNS)
    expected = {
        "confirmatory": 13,
        "exploratory_low_support": 3,
        "unsupported_by_design": 18,
    }
    if frame["support_tier"].value_counts().to_dict() != expected:
        raise ValueError("P13 domain support counts differ from the approved freeze.")
    return frame


def build_crossover_support(manifest: pd.DataFrame) -> pd.DataFrame:
    unique_views = manifest[
        [
            "master_sample_id",
            "station",
            "target_analyte",
            "sensor_family",
            "instrument",
        ]
    ].drop_duplicates()
    rows: list[dict[str, object]] = []
    block_index = 0

    for (station, analyte), group in unique_views.groupby(
        ["station", "target_analyte"], sort=True
    ):
        substrates = sorted(group["sensor_family"].unique())
        for substrate_a, substrate_b in combinations(substrates, 2):
            relevant = group[group["sensor_family"].isin([substrate_a, substrate_b])]
            instruments = sorted(relevant["instrument"].unique())
            for instrument_a, instrument_b in combinations(instruments, 2):
                required = {
                    (substrate_a, instrument_a),
                    (substrate_a, instrument_b),
                    (substrate_b, instrument_a),
                    (substrate_b, instrument_b),
                }
                masters = []
                for master_id, master_rows in relevant.groupby("master_sample_id", sort=True):
                    observed = set(
                        zip(
                            master_rows["sensor_family"],
                            master_rows["instrument"],
                            strict=True,
                        )
                    )
                    if required <= observed:
                        masters.append(master_id)
                if not masters:
                    continue

                block_index += 1
                master_count = len(masters)
                if master_count >= CONFIRMATORY_MINIMUM:
                    tier = "confirmatory"
                    reason = "at_least_three_paired_masters"
                elif master_count == EXPLORATORY_MINIMUM:
                    tier = "exploratory_low_support"
                    reason = "exactly_two_paired_masters"
                else:
                    tier = "descriptive_singleton"
                    reason = "one_paired_master_no_interval_claim"
                rows.append(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "crossover_block_id": f"P13-X-{block_index:03d}",
                        "station": station,
                        "target_analyte": analyte,
                        "substrate_a": substrate_a,
                        "substrate_b": substrate_b,
                        "instrument_a": instrument_a,
                        "instrument_b": instrument_b,
                        "physical_masters": master_count,
                        "support_tier": tier,
                        "support_reason": reason,
                    }
                )

    frame = pd.DataFrame(rows, columns=CROSSOVER_COLUMNS)
    expected = {
        "confirmatory": 8,
        "exploratory_low_support": 7,
        "descriptive_singleton": 19,
    }
    if frame["support_tier"].value_counts().to_dict() != expected:
        raise ValueError("P13 crossover support counts differ from the approved freeze.")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--master-splits", required=True, type=Path)
    parser.add_argument("--plan-dir", required=True, type=Path)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    splits = pd.read_csv(args.master_splits)
    _validate_inputs(manifest, splits)
    domains = build_domain_support(manifest, splits)
    crossovers = build_crossover_support(manifest)

    registry_dir = args.plan_dir / "registries"
    registry_dir.mkdir(parents=True, exist_ok=True)
    domain_path = registry_dir / "p13_domain_support_registry.csv"
    crossover_path = registry_dir / "p13_crossover_support_registry.csv"
    domains.to_csv(domain_path, index=False, lineterminator="\n")
    crossovers.to_csv(crossover_path, index=False, lineterminator="\n")

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "approval_date": APPROVAL_DATE,
        "approval_authority": "project_owner",
        "input_hashes": {
            "primary_manifest_sha256": _sha256(args.manifest),
            "p02_master_split_registry_sha256": _sha256(args.master_splits),
        },
        "registry_hashes": {
            "p13_domain_support_registry_sha256": _sha256(domain_path),
            "p13_crossover_support_registry_sha256": _sha256(crossover_path),
        },
        "thresholds": {
            "tau_minimum_balanced_accuracy": 0.60,
            "delta_maximum_source_to_held_loss": 0.10,
            "confirmatory_minimum_held_masters_per_class": 3,
            "confirmatory_minimum_training_masters_per_class_per_outer_split": 3,
            "confirmatory_minimum_source_instruments_per_class_per_outer_split": 2,
            "confirmatory_minimum_paired_source_held_masters_per_class": 3,
            "exploratory_minimum_held_masters_per_class": 2,
            "exploratory_minimum_training_masters_per_class_per_outer_split": 2,
            "exploratory_source_instrument_and_pair_support": (
                "reported but not used as an eligibility gate"
            ),
        },
        "domain_counts": domains["support_tier"].value_counts().sort_index().to_dict(),
        "crossover_block_counts": (
            crossovers["support_tier"].value_counts().sort_index().to_dict()
        ),
        "p03_outcome_access_disclosure": (
            "P03 outcomes were known before P13 was proposed and locked."
        ),
        "p13_outcome_access_disclosure": (
            "No P13 predictive, crossover-effect, or field-log outcome was calculated "
            "or used to choose these rules."
        ),
    }
    summary_path = registry_dir / "p13_support_freeze_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
