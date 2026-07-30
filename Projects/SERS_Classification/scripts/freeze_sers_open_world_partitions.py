#!/usr/bin/env python3
"""Freeze the NATO SERS open-world, calibration, and surrogate partitions."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

import sers_baseline_common as baseline
import sers_classical_benchmark_common as classical


PROTOCOL_VERSION = "sers-open-world-shift-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/sers_open_world_shift_v1.json"),
    )
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


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unexpected open-world protocol")
    if protocol.get("status_before_execution") != (
        "predeclared_partitions_before_open_set_outcomes"
    ):
        raise ValueError("Open-world partitions were not predeclared")
    return protocol


def outer_partition(
    manifest: pd.DataFrame, held_unknown: str, outer_fold: int
) -> np.ndarray:
    labels = manifest["target_analyte"].astype(str).to_numpy()
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    quality = manifest["include_sers_qc_pass"].astype(bool).to_numpy()
    stress = manifest["field_quality_stress"].astype(bool).to_numpy()
    held = labels == held_unknown
    result = np.full(len(manifest), "", dtype=object)
    result[quality & held] = "unknown_quality"
    result[stress & held] = "unknown_stress"
    result[stress & ~held] = "known_stress"
    result[quality & ~held & (folds == outer_fold)] = "known_quality_test"
    result[quality & ~held & (folds != outer_fold)] = (
        "known_development_quality"
    )
    if np.any(result == ""):
        raise ValueError("Unassigned outer rows")
    return result


def threshold_partition(
    manifest: pd.DataFrame,
    held_unknown: str,
    outer_fold: int,
    calibration_fold: int,
) -> np.ndarray:
    labels = manifest["target_analyte"].astype(str).to_numpy()
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    quality = manifest["include_sers_qc_pass"].astype(bool).to_numpy()
    stress = manifest["field_quality_stress"].astype(bool).to_numpy()
    held = labels == held_unknown
    result = np.full(len(manifest), "", dtype=object)
    result[held] = "locked_held_unknown"
    result[stress & ~held] = "locked_stress"
    result[quality & ~held & (folds == outer_fold)] = (
        "excluded_outer_known_test"
    )
    result[quality & ~held & (folds == calibration_fold)] = (
        "calibration_known"
    )
    result[
        quality
        & ~held
        & (folds != outer_fold)
        & (folds != calibration_fold)
    ] = "train_known"
    if np.any(result == ""):
        raise ValueError("Unassigned threshold rows")
    return result


def surrogate_partition(
    manifest: pd.DataFrame,
    held_unknown: str,
    outer_fold: int,
    inner_fold: int,
    surrogate_unknown: str,
) -> np.ndarray:
    labels = manifest["target_analyte"].astype(str).to_numpy()
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    quality = manifest["include_sers_qc_pass"].astype(bool).to_numpy()
    stress = manifest["field_quality_stress"].astype(bool).to_numpy()
    held = labels == held_unknown
    surrogate = labels == surrogate_unknown
    result = np.full(len(manifest), "", dtype=object)
    result[held] = "locked_held_unknown"
    result[stress & ~held] = "locked_stress"
    result[quality & ~held & (folds == outer_fold)] = (
        "excluded_outer_known_test"
    )
    development = quality & ~held & (folds != outer_fold)
    result[development & surrogate & (folds == inner_fold)] = (
        "validation_surrogate_unknown"
    )
    result[development & surrogate & (folds != inner_fold)] = (
        "excluded_surrogate_training"
    )
    result[development & ~surrogate & (folds == inner_fold)] = (
        "validation_known"
    )
    result[development & ~surrogate & (folds != inner_fold)] = "train_known"
    if np.any(result == ""):
        raise ValueError("Unassigned surrogate rows")
    return result


def records_for_scenario(
    manifest: pd.DataFrame,
    context: dict[str, Any],
    partitions: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "observation_uid",
        "master_sample_id",
        "target_analyte",
        "grouped_sample_fold_5",
        "include_sers_qc_pass",
        "field_quality_stress",
        "instrument",
        "sensor_family",
    ]
    frame = manifest[columns].copy()
    for key, value in reversed(list(context.items())):
        frame.insert(0, key, value)
    frame["partition"] = partitions
    return frame


def partition_summary(
    frame: pd.DataFrame, scenario_columns: list[str]
) -> pd.DataFrame:
    group_columns = [*scenario_columns, "partition"]
    return (
        frame.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            rows=("observation_uid", "size"),
            masters=("master_sample_id", "nunique"),
            analytes=("target_analyte", "nunique"),
            instruments=("instrument", "nunique"),
            sensor_families=("sensor_family", "nunique"),
        )
        .sort_values(group_columns, kind="stable")
    )


def support_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dimensions = [
        ("instrument", "instrument"),
        ("sensor_family", "sensor_family"),
        ("outer_fold", "grouped_sample_fold_5"),
    ]
    for dimension, column in dimensions:
        for quality_name, mask in (
            ("strict_core", np.ones(len(manifest), dtype=bool)),
            (
                "quality_pass",
                manifest["include_sers_qc_pass"].astype(bool).to_numpy(),
            ),
            (
                "field_quality_stress",
                manifest["field_quality_stress"].astype(bool).to_numpy(),
            ),
        ):
            subset = manifest.loc[mask]
            for (analyte, level), group in subset.groupby(
                ["target_analyte", column], dropna=False, sort=True
            ):
                rows.append(
                    {
                        "subset": quality_name,
                        "dimension": dimension,
                        "target_analyte": str(analyte),
                        "level": str(level),
                        "rows": len(group),
                        "masters": group["master_sample_id"].nunique(),
                    }
                )
    return pd.DataFrame(rows)


def write_hashes(output_dir: Path) -> None:
    excluded = {"artifact_hashes.json", "validation_report.json"}
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name not in excluded
    )
    classical.write_json(
        output_dir / "artifact_hashes.json",
        {
            str(path.relative_to(output_dir)): classical.sha256_file(path)
            for path in paths
        },
    )


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "predeclared_protocol.json"
    if snapshot.exists() and snapshot.read_bytes() != args.protocol.read_bytes():
        raise ValueError("Output contains a different protocol snapshot")
    if not snapshot.exists():
        shutil.copyfile(args.protocol, snapshot)
    dataset = baseline.load_nato_dataset(args.bundle)
    manifest = dataset.manifest
    expected = protocol["immutable_input"]
    observed = {
        "strict_core_rows": len(manifest),
        "quality_pass_rows": int(manifest["include_sers_qc_pass"].sum()),
        "field_quality_stress_rows": int(
            manifest["field_quality_stress"].sum()
        ),
        "master_samples": manifest["master_sample_id"].nunique(),
    }
    for key, value in observed.items():
        if value != int(expected[key]):
            raise ValueError(f"{key}: expected {expected[key]}, got {value}")

    held_chemicals = list(protocol["classes"]["held_unknown_chemicals"])
    outer_frames: list[pd.DataFrame] = []
    threshold_frames: list[pd.DataFrame] = []
    surrogate_frames: list[pd.DataFrame] = []
    outer_registry: list[dict[str, Any]] = []
    threshold_registry: list[dict[str, Any]] = []
    surrogate_registry: list[dict[str, Any]] = []
    for held_unknown in held_chemicals:
        surrogate_chemicals = [
            label for label in held_chemicals if label != held_unknown
        ]
        for outer_fold in range(int(protocol["partition_design"]["outer_folds"])):
            outer_context = {
                "held_unknown": held_unknown,
                "outer_fold": outer_fold,
            }
            outer_frame = records_for_scenario(
                manifest,
                outer_context,
                outer_partition(manifest, held_unknown, outer_fold),
            )
            outer_frames.append(outer_frame)
            counts = outer_frame["partition"].value_counts().to_dict()
            outer_registry.append({**outer_context, **counts})
            inner_folds = [fold for fold in range(5) if fold != outer_fold]
            for inner_fold in inner_folds:
                threshold_context = {
                    "held_unknown": held_unknown,
                    "outer_fold": outer_fold,
                    "calibration_fold": inner_fold,
                }
                threshold_frame = records_for_scenario(
                    manifest,
                    threshold_context,
                    threshold_partition(
                        manifest, held_unknown, outer_fold, inner_fold
                    ),
                )
                threshold_frames.append(threshold_frame)
                threshold_registry.append(
                    {
                        **threshold_context,
                        **threshold_frame["partition"].value_counts().to_dict(),
                    }
                )
                for surrogate_unknown in surrogate_chemicals:
                    surrogate_context = {
                        "held_unknown": held_unknown,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "surrogate_unknown": surrogate_unknown,
                    }
                    surrogate_frame = records_for_scenario(
                        manifest,
                        surrogate_context,
                        surrogate_partition(
                            manifest,
                            held_unknown,
                            outer_fold,
                            inner_fold,
                            surrogate_unknown,
                        ),
                    )
                    surrogate_frames.append(surrogate_frame)
                    surrogate_registry.append(
                        {
                            **surrogate_context,
                            **surrogate_frame[
                                "partition"
                            ].value_counts().to_dict(),
                        }
                    )

    outer = pd.concat(outer_frames, ignore_index=True)
    threshold = pd.concat(threshold_frames, ignore_index=True)
    surrogate = pd.concat(surrogate_frames, ignore_index=True)
    outer.to_csv(output / "open_set_outer_partitions.csv", index=False)
    threshold.to_csv(
        output / "known_threshold_crossfit_partitions.csv", index=False
    )
    surrogate.to_csv(
        output / "surrogate_unknown_inner_partitions.csv", index=False
    )
    pd.DataFrame(outer_registry).to_csv(
        output / "open_set_outer_scenario_registry.csv", index=False
    )
    pd.DataFrame(threshold_registry).to_csv(
        output / "known_threshold_scenario_registry.csv", index=False
    )
    pd.DataFrame(surrogate_registry).to_csv(
        output / "surrogate_unknown_scenario_registry.csv", index=False
    )
    partition_summary(outer, ["held_unknown", "outer_fold"]).to_csv(
        output / "open_set_outer_partition_summary.csv", index=False
    )
    partition_summary(
        threshold, ["held_unknown", "outer_fold", "calibration_fold"]
    ).to_csv(
        output / "known_threshold_partition_summary.csv", index=False
    )
    partition_summary(
        surrogate,
        [
            "held_unknown",
            "outer_fold",
            "inner_fold",
            "surrogate_unknown",
        ],
    ).to_csv(
        output / "surrogate_unknown_partition_summary.csv", index=False
    )
    support_audit(manifest).to_csv(
        output / "factor_support_audit.csv", index=False
    )

    copied_domains: dict[str, Any] = {}
    for filename in expected["domain_partition_files"]:
        source = args.bundle / filename
        destination = output / filename
        shutil.copyfile(source, destination)
        copied_domains[filename] = {
            "source": str(source.resolve()),
            "source_sha256": classical.sha256_file(source),
            "copied_sha256": classical.sha256_file(destination),
            "rows": len(pd.read_csv(destination)),
        }
    classical.write_json(output / "domain_partition_registry.json", copied_domains)
    classical.write_json(
        output / "input_hashes.json",
        {
            "protocol": {
                "path": str(args.protocol.resolve()),
                "sha256": classical.sha256_file(args.protocol),
            },
            "dataset_hash_catalog": {
                "path": str((args.bundle / "artifact_hashes.json").resolve()),
                "sha256": classical.sha256_file(
                    args.bundle / "artifact_hashes.json"
                ),
            },
            "dataset_version": {
                "path": str((args.bundle / "dataset_version.json").resolve()),
                "sha256": classical.sha256_file(
                    args.bundle / "dataset_version.json"
                ),
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": classical.sha256_file(Path(__file__).resolve()),
            },
        },
    )
    classical.write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    )
    write_hashes(output)
    print(
        json.dumps(
            {
                "outer_scenarios": len(outer_registry),
                "threshold_scenarios": len(threshold_registry),
                "surrogate_scenarios": len(surrogate_registry),
                "outer_rows": len(outer),
                "threshold_rows": len(threshold),
                "surrogate_rows": len(surrogate),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
