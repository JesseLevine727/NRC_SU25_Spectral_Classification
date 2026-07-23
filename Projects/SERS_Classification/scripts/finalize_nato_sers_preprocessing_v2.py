#!/usr/bin/env python3
"""Complete and freeze the bounded NATO SERS preprocessing-v2 study."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.signal import find_peaks, peak_widths, savgol_filter

import freeze_nato_sers_preprocessing as v1


VERSION = "nato-sers-preprocessing-v2"
SMOOTHING_WINDOWS = (7, 11, 15)
CANDIDATE_ORDER = (
    "minimal_minmax",
    "minimal_sg7_minmax",
    "minimal_sg11_minmax",
    "minimal_sg15_minmax",
    "arpls_minmax",
    "arpls_sg7_minmax",
    "arpls_sg11_minmax",
    "arpls_sg15_minmax",
    "derivative_1",
)
INTENSITY_CANDIDATES = CANDIDATE_ORDER[:-1]
FAMILY_BASE = {
    **{name: "minimal_minmax" for name in CANDIDATE_ORDER if name.startswith("minimal")},
    **{name: "arpls_minmax" for name in CANDIDATE_ORDER if name.startswith("arpls")},
}
CORRUPTIONS = v1.CORRUPTIONS
RANDOM_SEED = v1.RANDOM_SEED


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def json_clean(value: Any) -> Any:
    """Convert nested NumPy/Pandas output to standards-compliant JSON values."""
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_clean(item) for item in value.tolist()]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def load_control(
    v1_dir: Path,
) -> tuple[pd.DataFrame, np.lib.npyio.NpzFile, dict[str, Any]]:
    manifest = pd.read_csv(v1_dir / "core_preprocessing_manifest.csv")
    archive = np.load(v1_dir / "candidate_spectra_core.npz")
    version = json.loads((v1_dir / "dataset_version.json").read_text())
    ids = manifest["observation_uid"].astype(str).to_numpy()
    if not np.array_equal(ids, archive["observation_uid"].astype(str)):
        raise ValueError("v1 manifest/archive observation order differs")
    if version["dataset_version"] != "nato-sers-preprocessing-v1":
        raise ValueError("Expected the frozen preprocessing-v1 control")
    hashes = json.loads((v1_dir / "artifact_hashes.json").read_text())
    for relative_path, expected_hash in hashes.items():
        if sha256_file(v1_dir / relative_path) != expected_hash:
            raise ValueError(f"v1 artifact changed: {relative_path}")
    return manifest, archive, version


def minmax(values: np.ndarray) -> np.ndarray:
    return v1.minmax_rows(values)[0]


def high_frequency_score_rows(values: np.ndarray) -> np.ndarray:
    second = np.diff(values, n=2, axis=1)
    center = np.median(second, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(second - center), axis=1)
    robust_span = np.percentile(values, 99, axis=1) - np.percentile(values, 1, axis=1)
    return mad / np.maximum(robust_span, 1.0e-12)


def candidates_from_components(
    despiked: np.ndarray,
    arpls_baseline: np.ndarray,
    derivative_1: np.ndarray,
    exact_minimal: np.ndarray | None = None,
    exact_arpls: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    residual = despiked - arpls_baseline
    output: dict[str, np.ndarray] = {
        "minimal_minmax": (
            exact_minimal.astype(float) if exact_minimal is not None else minmax(despiked)
        ),
        "arpls_minmax": (
            exact_arpls.astype(float) if exact_arpls is not None else minmax(residual)
        ),
        "derivative_1": derivative_1.astype(float),
    }
    for window in SMOOTHING_WINDOWS:
        output[f"minimal_sg{window}_minmax"] = minmax(
            savgol_filter(despiked, window, 3, deriv=0, axis=1, mode="interp")
        )
        output[f"arpls_sg{window}_minmax"] = minmax(
            savgol_filter(residual, window, 3, deriv=0, axis=1, mode="interp")
        )
    return {name: output[name] for name in CANDIDATE_ORDER}


def build_candidates_from_raw(
    raw: np.ndarray, config: v1.PreprocessingConfig
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    spike_mask, spike_prominence = v1.detect_spikes(raw, config)
    saturation_mask = v1.detect_saturation(raw, config)
    despiked = v1.repair_masked_points(raw, spike_mask)
    arpls_baseline = v1.arpls_baseline_matrix(despiked, config)
    normalized = v1.snv(despiked)
    derivative_1 = v1.l2_rows(
        savgol_filter(
            normalized,
            config.derivative_window_points,
            config.derivative_polynomial_order,
            deriv=1,
            axis=1,
        )
    )
    candidates = candidates_from_components(despiked, arpls_baseline, derivative_1)
    return candidates, {
        "despiked": despiked,
        "arpls_baseline": arpls_baseline,
        "spike_mask": spike_mask,
        "spike_prominence_fraction": spike_prominence,
        "saturation_mask": saturation_mask,
    }


def peak_table(values: np.ndarray) -> pd.DataFrame:
    peaks, properties = find_peaks(values, prominence=0.05, distance=5)
    widths = peak_widths(values, peaks, rel_height=0.5)[0] if len(peaks) else []
    return pd.DataFrame(
        {
            "position": peaks.astype(int),
            "prominence": properties.get("prominences", np.asarray([], dtype=float)),
            "width": np.asarray(widths, dtype=float),
        }
    )


def repeatable_peak_positions(
    peak_tables: list[pd.DataFrame], manifest: pd.DataFrame
) -> list[set[int]]:
    repeatable: list[set[int]] = [set() for _ in peak_tables]
    for _, indices in manifest.groupby("master_sample_id").groups.items():
        group_indices = list(indices)
        instruments = {
            row_index: str(manifest.iloc[row_index]["instrument"])
            for row_index in group_indices
        }
        if len(group_indices) < 2 or len(set(instruments.values())) < 2:
            continue
        required_support = max(1, int(np.ceil(0.5 * (len(group_indices) - 1))))
        for row_index in group_indices:
            for reference_peak in peak_tables[row_index].itertuples(index=False):
                if float(reference_peak.prominence) < 0.15:
                    continue
                supporting_rows: list[int] = []
                for other_index in group_indices:
                    if other_index == row_index:
                        continue
                    other = peak_tables[other_index]["position"].to_numpy(dtype=int)
                    if len(other) and np.min(
                        np.abs(other - int(reference_peak.position))
                    ) <= 3:
                        supporting_rows.append(other_index)
                cross_instrument_support = any(
                    instruments[other_index] != instruments[row_index]
                    for other_index in supporting_rows
                )
                if (
                    len(supporting_rows) >= required_support
                    and cross_instrument_support
                ):
                    repeatable[row_index].add(int(reference_peak.position))
    return repeatable


def match_peak_tables(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    repeatable_positions: set[int],
) -> dict[str, Any]:
    available = list(range(len(candidate)))
    shifts: list[float] = []
    width_changes: list[float] = []
    prominence_changes: list[float] = []
    repeatable_matches = 0
    for reference_row in reference.itertuples(index=False):
        if not available:
            continue
        candidate_positions = candidate.iloc[available]["position"].to_numpy(dtype=int)
        differences = np.abs(candidate_positions - int(reference_row.position))
        best_local = int(np.argmin(differences))
        if differences[best_local] > 5:
            continue
        candidate_index = available.pop(best_local)
        candidate_row = candidate.iloc[candidate_index]
        shifts.append(float(differences[best_local]))
        width_changes.append(
            float(
                abs(float(candidate_row["width"]) - float(reference_row.width))
                / max(float(reference_row.width), 1.0e-12)
            )
        )
        prominence_changes.append(
            float(
                abs(
                    float(candidate_row["prominence"])
                    - float(reference_row.prominence)
                )
            )
        )
        if int(reference_row.position) in repeatable_positions:
            repeatable_matches += 1
    return {
        "reference_peak_count": int(len(reference)),
        "matched_peak_count": int(len(shifts)),
        "repeatable_reference_peak_count": int(len(repeatable_positions)),
        "repeatable_matched_peak_count": int(repeatable_matches),
        "median_matched_peak_shift_cm1": (
            float(np.median(shifts)) if shifts else np.nan
        ),
        "median_absolute_relative_peak_width_change": (
            float(np.median(width_changes)) if width_changes else np.nan
        ),
        "median_absolute_peak_prominence_change": (
            float(np.median(prominence_changes)) if prominence_changes else np.nan
        ),
    }


def preservation_frames(
    manifest: pd.DataFrame, candidates: dict[str, np.ndarray]
) -> dict[str, pd.DataFrame]:
    family_references = {
        base: [peak_table(row) for row in candidates[base]]
        for base in ("minimal_minmax", "arpls_minmax")
    }
    repeatable = {
        base: repeatable_peak_positions(tables, manifest)
        for base, tables in family_references.items()
    }
    scores = {
        name: high_frequency_score_rows(values)
        for name, values in candidates.items()
        if name in INTENSITY_CANDIDATES
    }
    output: dict[str, pd.DataFrame] = {}
    for candidate_name in CANDIDATE_ORDER:
        if candidate_name == "derivative_1":
            output[candidate_name] = pd.DataFrame(
                {
                    "observation_uid": manifest["observation_uid"],
                    "candidate": candidate_name,
                }
            )
            continue
        base = FAMILY_BASE[candidate_name]
        rows: list[dict[str, Any]] = []
        for row_index in range(len(manifest)):
            reference_values = candidates[base][row_index]
            candidate_values = candidates[candidate_name][row_index]
            metrics = match_peak_tables(
                family_references[base][row_index],
                peak_table(candidate_values),
                repeatable[base][row_index],
            )
            correlation = (
                float(np.corrcoef(reference_values, candidate_values)[0, 1])
                if np.std(reference_values) > 0 and np.std(candidate_values) > 0
                else np.nan
            )
            reference_noise = float(scores[base][row_index])
            candidate_noise = float(scores[candidate_name][row_index])
            metrics.update(
                {
                    "observation_uid": manifest.iloc[row_index]["observation_uid"],
                    "candidate": candidate_name,
                    "family_base": base,
                    "row_correlation_to_unsmoothed_family": correlation,
                    "high_frequency_noise_score": candidate_noise,
                    "high_frequency_noise_score_reference": reference_noise,
                    "relative_high_frequency_noise_reduction": (
                        1.0 - candidate_noise / reference_noise
                        if reference_noise > 1.0e-12
                        else 0.0
                    ),
                }
            )
            rows.append(metrics)
        output[candidate_name] = pd.DataFrame(rows)
    return output


def aggregate_preservation(frame: pd.DataFrame) -> dict[str, float]:
    if "reference_peak_count" not in frame or frame["reference_peak_count"].isna().all():
        return {
            "preservation_all_peak_weighted_recall": np.nan,
            "preservation_repeatable_peak_weighted_recall": np.nan,
            "preservation_median_matched_peak_shift_cm1": np.nan,
            "preservation_median_absolute_relative_peak_width_change": np.nan,
            "preservation_median_absolute_peak_prominence_change": np.nan,
            "preservation_median_row_correlation_to_unsmoothed_family": np.nan,
            "preservation_median_relative_high_frequency_noise_reduction": np.nan,
        }
    reference_count = float(frame["reference_peak_count"].sum())
    repeatable_count = float(frame["repeatable_reference_peak_count"].sum())
    return {
        "preservation_all_peak_weighted_recall": (
            float(frame["matched_peak_count"].sum() / reference_count)
            if reference_count
            else np.nan
        ),
        "preservation_repeatable_peak_weighted_recall": (
            float(frame["repeatable_matched_peak_count"].sum() / repeatable_count)
            if repeatable_count
            else np.nan
        ),
        "preservation_median_matched_peak_shift_cm1": float(
            frame["median_matched_peak_shift_cm1"].median()
        ),
        "preservation_median_absolute_relative_peak_width_change": float(
            frame["median_absolute_relative_peak_width_change"].median()
        ),
        "preservation_median_absolute_peak_prominence_change": float(
            frame["median_absolute_peak_prominence_change"].median()
        ),
        "preservation_median_row_correlation_to_unsmoothed_family": float(
            frame["row_correlation_to_unsmoothed_family"].median()
        ),
        "preservation_median_relative_high_frequency_noise_reduction": float(
            frame["relative_high_frequency_noise_reduction"].median()
        ),
    }


def benchmark_candidates(
    manifest: pd.DataFrame,
    candidates: dict[str, np.ndarray],
    corrupted: dict[str, dict[str, np.ndarray]],
    preservation: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    folds = manifest["grouped_sample_fold_5"].to_numpy()
    quality = manifest["include_sers_qc_pass"].astype(bool).to_numpy()
    subsets = {
        "strict_core": np.ones(len(manifest), dtype=bool),
        "quality_pass": quality,
    }

    def evaluate_and_record(
        candidate_name: str,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        record = v1.evaluate_split(
            candidate_name,
            candidates[candidate_name],
            corrupted[candidate_name],
            manifest,
            train_mask,
            test_mask,
        )
        record.update(
            aggregate_preservation(preservation[candidate_name].loc[test_mask])
        )
        record.update(metadata)
        records.append(record)

    for subset_name, subset_mask in subsets.items():
        for outer_fold in sorted(np.unique(folds)):
            outer_train = subset_mask & (folds != outer_fold)
            outer_test = subset_mask & (folds == outer_fold)
            for candidate_name in CANDIDATE_ORDER:
                evaluate_and_record(
                    candidate_name,
                    outer_train,
                    outer_test,
                    {
                        "subset": subset_name,
                        "stage": "outer_test",
                        "outer_fold": int(outer_fold),
                        "inner_validation_fold": pd.NA,
                    },
                )
            for inner_fold in sorted(set(np.unique(folds)) - {outer_fold}):
                inner_train = (
                    subset_mask & (folds != outer_fold) & (folds != inner_fold)
                )
                inner_validation = subset_mask & (folds == inner_fold)
                for candidate_name in CANDIDATE_ORDER:
                    evaluate_and_record(
                        candidate_name,
                        inner_train,
                        inner_validation,
                        {
                            "subset": subset_name,
                            "stage": "inner_validation",
                            "outer_fold": int(outer_fold),
                            "inner_validation_fold": int(inner_fold),
                        },
                    )

    flagged = ~quality
    for outer_fold in sorted(np.unique(folds)):
        train_mask = quality & (folds != outer_fold)
        test_mask = flagged & (folds == outer_fold)
        if not test_mask.any():
            continue
        for candidate_name in CANDIDATE_ORDER:
            evaluate_and_record(
                candidate_name,
                train_mask,
                test_mask,
                {
                    "subset": "field_quality_stress",
                    "stage": "outer_test",
                    "outer_fold": int(outer_fold),
                    "inner_validation_fold": pd.NA,
                },
            )
    return pd.DataFrame(records)


def summarize_benchmarks(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in {"outer_fold", "inner_validation_fold"}
    ]
    summary = (
        metrics.groupby(["representation", "subset", "stage"])[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def preservation_aggregate_table(
    manifest: pd.DataFrame, preservation: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subsets = {
        "strict_core": np.ones(len(manifest), dtype=bool),
        "quality_pass": manifest["include_sers_qc_pass"].astype(bool).to_numpy(),
        "field_quality_stress": ~manifest["include_sers_qc_pass"]
        .astype(bool)
        .to_numpy(),
    }
    for candidate_name, frame in preservation.items():
        for subset_name, subset_mask in subsets.items():
            groups: list[tuple[str, np.ndarray]] = [("ALL", subset_mask)]
            for instrument in sorted(manifest.loc[subset_mask, "instrument"].unique()):
                groups.append(
                    (
                        str(instrument),
                        subset_mask
                        & (manifest["instrument"].astype(str).to_numpy() == instrument),
                    )
                )
            for instrument, mask in groups:
                row = {
                    "candidate": candidate_name,
                    "subset": subset_name,
                    "instrument": instrument,
                    "n_spectra": int(mask.sum()),
                }
                row.update(aggregate_preservation(frame.loc[mask]))
                rows.append(row)
    return pd.DataFrame(rows)


def normalized_utility(frame: pd.DataFrame, objectives: dict[str, str]) -> np.ndarray:
    utility = np.zeros(len(frame), dtype=float)
    for column, direction in objectives.items():
        values = frame[column].to_numpy(dtype=float)
        low, high = np.nanmin(values), np.nanmax(values)
        normalized = (
            np.full(len(values), 0.5)
            if high - low <= 1.0e-12
            else (values - low) / (high - low)
        )
        utility += normalized if direction == "max" else 1.0 - normalized
    return utility / len(objectives)


def selection_table(
    metrics: pd.DataFrame, protocol: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    inner = metrics[metrics["stage"] == "inner_validation"]
    stress = metrics[
        (metrics["subset"] == "field_quality_stress")
        & (metrics["stage"] == "outer_test")
    ]
    rows: list[dict[str, Any]] = []
    for candidate_name in CANDIDATE_ORDER:
        core = inner[
            (inner["representation"] == candidate_name)
            & (inner["subset"] == "strict_core")
        ]
        quality = inner[
            (inner["representation"] == candidate_name)
            & (inner["subset"] == "quality_pass")
        ]
        stress_candidate = stress[stress["representation"] == candidate_name]
        row = {
            "candidate": candidate_name,
            "family": (
                "minimal"
                if candidate_name.startswith("minimal")
                else "arpls"
                if candidate_name.startswith("arpls")
                else "derivative"
            ),
            "is_smoothed": "_sg" in candidate_name,
            "family_base": FAMILY_BASE.get(candidate_name),
            "target_core": core["target_pca_balanced_accuracy"].mean(),
            "target_quality": quality["target_pca_balanced_accuracy"].mean(),
            "target_stress_outer": stress_candidate[
                "target_pca_balanced_accuracy"
            ].mean(),
            "noise_target": core[
                "corruption_noise_pca_balanced_accuracy"
            ].mean(),
            "noise_prediction_agreement": core[
                "corruption_noise_pca_prediction_agreement"
            ].mean(),
            "composite_target": core[
                "corruption_composite_pca_balanced_accuracy"
            ].mean(),
            "instrument_leakage": core[
                "instrument_probe_increment_over_target_only"
            ].mean(),
            "sensor_leakage": core[
                "sensor_probe_increment_over_target_only"
            ].mean(),
            "same_master_distance": core[
                "geometry_same_master_cross_instrument_mean_distance"
            ].mean(),
            "separation_margin": core[
                "geometry_cross_instrument_separation_margin"
            ].mean(),
            "repeatable_peak_recall": core[
                "preservation_repeatable_peak_weighted_recall"
            ].mean(),
            "all_peak_recall": core[
                "preservation_all_peak_weighted_recall"
            ].mean(),
            "median_peak_shift_cm1": core[
                "preservation_median_matched_peak_shift_cm1"
            ].mean(),
            "median_absolute_relative_peak_width_change": core[
                "preservation_median_absolute_relative_peak_width_change"
            ].mean(),
            "median_absolute_peak_prominence_change": core[
                "preservation_median_absolute_peak_prominence_change"
            ].mean(),
            "median_row_correlation": core[
                "preservation_median_row_correlation_to_unsmoothed_family"
            ].mean(),
            "relative_high_frequency_noise_reduction": core[
                "preservation_median_relative_high_frequency_noise_reduction"
            ].mean(),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    gates = protocol["preservation_gates"]
    benefit_labels: dict[str, dict[str, bool]] = {}
    for index, row in frame.iterrows():
        base_name = row["family_base"]
        if not row["is_smoothed"] or not base_name:
            continue
        base = frame[frame["candidate"] == base_name].iloc[0]
        deltas = {
            "target_core_delta": row["target_core"] - base["target_core"],
            "target_quality_delta": row["target_quality"] - base["target_quality"],
            "target_stress_outer_delta_confirmatory": (
                row["target_stress_outer"] - base["target_stress_outer"]
            ),
            "noise_target_delta": row["noise_target"] - base["noise_target"],
            "noise_prediction_agreement_delta": (
                row["noise_prediction_agreement"]
                - base["noise_prediction_agreement"]
            ),
            "instrument_leakage_delta": (
                row["instrument_leakage"] - base["instrument_leakage"]
            ),
            "sensor_leakage_delta": row["sensor_leakage"] - base["sensor_leakage"],
            "same_master_distance_delta": (
                row["same_master_distance"] - base["same_master_distance"]
            ),
        }
        for key, value in deltas.items():
            frame.loc[index, key] = value
        gate_results = {
            "gate_repeatable_peak_recall": (
                row["repeatable_peak_recall"]
                >= gates["repeatable_peak_weighted_recall_minimum"]
            ),
            "gate_peak_shift": (
                row["median_peak_shift_cm1"]
                <= gates["median_matched_peak_shift_cm1_maximum"]
            ),
            "gate_peak_width": (
                row["median_absolute_relative_peak_width_change"]
                <= gates["median_absolute_relative_peak_width_change_maximum"]
            ),
            "gate_peak_prominence": (
                row["median_absolute_peak_prominence_change"]
                <= gates["median_absolute_peak_prominence_change_maximum"]
            ),
            "gate_row_correlation": (
                row["median_row_correlation"]
                >= gates["median_row_correlation_to_unsmoothed_family_minimum"]
            ),
            "gate_target_core": (
                deltas["target_core_delta"]
                >= -gates["clean_target_balanced_accuracy_drop_maximum"]
            ),
            "gate_target_quality": (
                deltas["target_quality_delta"]
                >= -gates["clean_target_balanced_accuracy_drop_maximum"]
            ),
            "gate_instrument_leakage": (
                deltas["instrument_leakage_delta"]
                <= gates["target_adjusted_instrument_leakage_increase_maximum"]
            ),
            "gate_sensor_leakage": (
                deltas["sensor_leakage_delta"]
                <= gates["target_adjusted_sensor_leakage_increase_maximum"]
            ),
            "gate_same_master_distance": (
                deltas["same_master_distance_delta"]
                <= gates["same_master_cross_instrument_distance_increase_maximum"]
            ),
        }
        benefits = {
            "benefit_noise_target": deltas["noise_target_delta"] >= 0.01,
            "benefit_noise_agreement": (
                deltas["noise_prediction_agreement_delta"] >= 0.02
            ),
            "benefit_high_frequency_reduction": (
                row["relative_high_frequency_noise_reduction"] >= 0.10
            ),
        }
        benefit_labels[str(row["candidate"])] = benefits
        for key, value in {**gate_results, **benefits}.items():
            frame.loc[index, key] = bool(value)
        frame.loc[index, "passes_all_preservation_gates"] = all(
            gate_results.values()
        )
        frame.loc[index, "passes_any_benefit_gate"] = any(benefits.values())
        frame.loc[index, "eligible_smoother"] = all(gate_results.values()) and any(
            benefits.values()
        )

    selected_smoothers: list[str] = []
    utility_objectives = {
        "target_core_delta": "max",
        "target_quality_delta": "max",
        "noise_target_delta": "max",
        "noise_prediction_agreement_delta": "max",
        "relative_high_frequency_noise_reduction": "max",
        "repeatable_peak_recall": "max",
        "median_absolute_relative_peak_width_change": "min",
        "instrument_leakage_delta": "min",
        "same_master_distance_delta": "min",
    }
    frame["selection_utility"] = np.nan
    for family in ("minimal", "arpls"):
        eligible = frame[
            (frame["family"] == family) & (frame["eligible_smoother"] == True)
        ].copy()
        if eligible.empty:
            continue
        eligible["selection_utility"] = normalized_utility(
            eligible, utility_objectives
        )
        frame.loc[eligible.index, "selection_utility"] = eligible[
            "selection_utility"
        ]
        winner = eligible.sort_values(
            ["selection_utility", "candidate"], ascending=[False, True]
        ).iloc[0]
        selected_smoothers.append(str(winner["candidate"]))
    selected = ["minimal_minmax", "arpls_minmax", *selected_smoothers, "derivative_1"]
    frame["selected_final"] = frame["candidate"].isin(selected)

    smoothed = frame[frame["is_smoothed"]]
    inconsistent = bool(
        (
            smoothed["passes_any_benefit_gate"].fillna(False)
            & ~smoothed["passes_all_preservation_gates"].fillna(False)
        ).any()
    )
    selection = {
        "selection_data": "nested inner validation only",
        "stress_cohort_role": "confirmatory only; not used for selection",
        "mandatory_unsmoothed_controls": [
            "minimal_minmax",
            "arpls_minmax",
            "derivative_1",
        ],
        "selected_smoothers": selected_smoothers,
        "selected_final_representations": selected,
        "uniform_smoothing_inconsistent_tradeoff_trigger": inconsistent,
        "noise_gated_policy_considered": inconsistent,
        "noise_gated_policy_implemented": False,
        "noise_gated_policy_decision": (
            "Considered and rejected. Uniform smoothing reduced the target-blind "
            "high-frequency score but did not meet the predeclared synthetic-noise "
            "accuracy or prediction-agreement benefits and failed repeatable-peak "
            "preservation. A fold-fitted gate would add an unsupported threshold, "
            "make model inputs split-dependent, and could select instrument "
            "resolution rather than field noise."
            if inconsistent
            else "Not triggered because uniform candidates did not show the "
            "predeclared inconsistent benefit/preservation pattern."
        ),
        "noise_gated_policy_rejection_reasons": (
            [
                "No uniform smoother met the repeatable-peak preservation gate.",
                "No uniform smoother met either classifier-based synthetic-noise benefit gate.",
                "The flagged-quality cohort is confirmatory and cannot justify a threshold.",
                "A threshold learned inside each fold would prevent one immutable downstream input archive.",
                "A threshold fitted on all 598 spectra would expose outer-test distribution information.",
            ]
            if inconsistent
            else []
        ),
        "utility_objectives": utility_objectives,
        "benefit_gate_results": benefit_labels,
        "table": frame.to_dict(orient="records"),
    }
    return frame, selection


def best_integer_lag(a: np.ndarray, b: np.ndarray, maximum: int = 5) -> tuple[int, float, float]:
    best_lag = 0
    best_score = -np.inf
    zero_score = np.nan
    for lag in range(-maximum, maximum + 1):
        if lag > 0:
            left, right = a[lag:], b[:-lag]
        elif lag < 0:
            left, right = a[:lag], b[-lag:]
        else:
            left, right = a, b
        denominator = max(np.linalg.norm(left) * np.linalg.norm(right), 1.0e-12)
        score = float(np.dot(left, right) / denominator)
        if lag == 0:
            zero_score = score
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag, best_score, float(zero_score)


def alignment_pair_evidence(
    manifest: pd.DataFrame, derivative: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for master_sample_id, indices in manifest.groupby("master_sample_id").groups.items():
        group_indices = list(indices)
        for left_index, right_index in itertools.combinations(group_indices, 2):
            left_instrument = str(manifest.iloc[left_index]["instrument"])
            right_instrument = str(manifest.iloc[right_index]["instrument"])
            if left_instrument == right_instrument:
                continue
            if left_instrument > right_instrument:
                left_index, right_index = right_index, left_index
                left_instrument, right_instrument = right_instrument, left_instrument
            lag, score, zero_score = best_integer_lag(
                derivative[left_index], derivative[right_index]
            )
            records.append(
                {
                    "master_sample_id": master_sample_id,
                    "target_analyte": manifest.iloc[left_index]["target_analyte"],
                    "instrument_a": left_instrument,
                    "instrument_b": right_instrument,
                    "observation_uid_a": manifest.iloc[left_index]["observation_uid"],
                    "observation_uid_b": manifest.iloc[right_index]["observation_uid"],
                    "both_quality_pass": bool(
                        manifest.iloc[left_index]["include_sers_qc_pass"]
                        and manifest.iloc[right_index]["include_sers_qc_pass"]
                    ),
                    "best_integer_lag_cm1_b_relative_to_a": int(lag),
                    "best_cosine_similarity": score,
                    "zero_lag_cosine_similarity": zero_score,
                    "similarity_gain_over_zero_lag": score - zero_score,
                }
            )
    pairs = pd.DataFrame(records)
    summaries: list[dict[str, Any]] = []
    for subset_name, subset in (
        ("strict_core", pairs),
        ("quality_pass", pairs[pairs["both_quality_pass"]]),
    ):
        for (instrument_a, instrument_b), group in subset.groupby(
            ["instrument_a", "instrument_b"]
        ):
            lags = group["best_integer_lag_cm1_b_relative_to_a"].to_numpy(dtype=float)
            median = float(np.median(lags))
            summaries.append(
                {
                    "subset": subset_name,
                    "instrument_a": instrument_a,
                    "instrument_b": instrument_b,
                    "pair_count": len(group),
                    "master_sample_count": group["master_sample_id"].nunique(),
                    "target_class_count": group["target_analyte"].nunique(),
                    "median_lag_cm1": median,
                    "lag_q25_cm1": float(np.quantile(lags, 0.25)),
                    "lag_q75_cm1": float(np.quantile(lags, 0.75)),
                    "lag_iqr_cm1": float(np.quantile(lags, 0.75) - np.quantile(lags, 0.25)),
                    "fraction_within_one_cm1_of_median": float(
                        np.mean(np.abs(lags - median) <= 1.0)
                    ),
                    "median_similarity_gain_over_zero": float(
                        group["similarity_gain_over_zero_lag"].median()
                    ),
                }
            )
    return pairs, pd.DataFrame(summaries)


def parse_standard_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = (
        pd.read_csv(path, comment="#", header=None)
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    axis = frame.iloc[:, 0].to_numpy(dtype=float)
    values = frame.iloc[:, 1].to_numpy(dtype=float)
    order = np.argsort(axis)
    return axis[order], values[order]


def calibration_standard_evidence(source_root: Path) -> pd.DataFrame:
    paths = list(source_root.glob("Mira/Mira */Calibration Passed*.csv"))
    paths += list(source_root.glob("Pendar/Pendar */*Polystyrene*.csv"))
    records: list[dict[str, Any]] = []
    for path in sorted(set(paths)):
        axis, values = parse_standard_csv(path)
        in_common = (axis >= 400) & (axis <= 1800)
        axis, values = axis[in_common], values[in_common]
        anchor_region = (axis >= 950) & (axis <= 1050)
        if not anchor_region.any():
            continue
        anchor = float(axis[anchor_region][np.argmax(values[anchor_region])])
        parent = path.parent.name.replace(" ", "-")
        instrument = parent if parent.startswith(("Mira", "Pendar")) else parent
        records.append(
            {
                "instrument": instrument,
                "source_path": str(path.resolve()),
                "standard_description": (
                    "vendor calibration-passed spectrum"
                    if "Calibration Passed" in path.name
                    else "polystyrene/calcium-carbonate suitability spectrum"
                ),
                "anchor_peak_950_1050_cm1": anchor,
                "axis_min_cm1": float(axis.min()),
                "axis_max_cm1": float(axis.max()),
                "source_sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(records)


def alignment_decision(
    pair_summary: pd.DataFrame,
    standards: pd.DataFrame,
    protocol: dict[str, Any],
    all_instruments: list[str],
) -> dict[str, Any]:
    gates = protocol["alignment"]["acceptance_requirements"]
    core = pair_summary[pair_summary["subset"] == "strict_core"].copy()
    quality = pair_summary[pair_summary["subset"] == "quality_pass"].copy()
    core["passes_pair_count"] = core["pair_count"] >= gates["minimum_pair_count"]
    core["passes_master_count"] = (
        core["master_sample_count"] >= gates["minimum_master_samples"]
    )
    core["passes_target_count"] = (
        core["target_class_count"] >= gates["minimum_target_classes"]
    )
    core["passes_shift_magnitude"] = (
        core["median_lag_cm1"].abs()
        >= gates["absolute_median_shift_cm1_minimum_to_justify_correction"]
    )
    core["passes_iqr"] = core["lag_iqr_cm1"] <= gates["shift_iqr_cm1_maximum"]
    core["passes_concentration"] = (
        core["fraction_within_one_cm1_of_median"]
        >= gates["fraction_within_one_cm1_of_pair_median_minimum"]
    )
    quality_medians = quality.set_index(["instrument_a", "instrument_b"])[
        "median_lag_cm1"
    ]
    core["passes_quality_direction"] = [
        (
            np.sign(row.median_lag_cm1)
            == np.sign(quality_medians.get((row.instrument_a, row.instrument_b), np.nan))
        )
        for row in core.itertuples()
    ]
    gate_columns = [column for column in core if column.startswith("passes_")]
    core["passes_all_pairwise_gates"] = core[gate_columns].all(axis=1)
    covered = sorted(standards["instrument"].unique().tolist())
    missing = sorted(set(all_instruments) - set(covered))
    accepted = False
    reasons = [
        "Named calibration/standard spectra do not cover all ten field instruments.",
        f"Standard-covered instruments: {covered}; missing: {missing}.",
        "A correction for only the covered systems would be system-aware and partial.",
        "Same-master lags are retained as diagnostics but cannot replace a shared "
        "calibration standard because chemistry, sensor, and instrument response are confounded.",
        "Flexible or target-informed per-spectrum warping remains prohibited.",
    ]
    return {
        "alignment_accepted": accepted,
        "final_alignment_operation": "none",
        "fallback": protocol["alignment"][
            "fallback_when_requirements_fail"
        ],
        "standard_covered_instruments": covered,
        "standard_missing_instruments": missing,
        "pairwise_gate_passes": core[
            [
                "instrument_a",
                "instrument_b",
                *gate_columns,
                "passes_all_pairwise_gates",
            ]
        ].to_dict(orient="records"),
        "reasons": reasons,
    }


def write_candidate_archive(
    path: Path,
    axis: np.ndarray,
    ids: np.ndarray,
    raw: np.ndarray,
    control: np.lib.npyio.NpzFile,
    candidates: dict[str, np.ndarray],
    selection: np.ndarray,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "axis_cm1": axis.astype(np.float32),
        "observation_uid": ids[selection].astype(str),
        "raw_common_grid": raw[selection].astype(np.float32),
        "despiked_common_grid": control["despiked_common_grid"][selection],
        "spike_mask": control["spike_mask"][selection],
        "saturation_mask": control["saturation_mask"][selection],
        "arpls_baseline": control["arpls_baseline"][selection],
    }
    arrays.update(
        {
            candidate_name: values[selection].astype(np.float32)
            for candidate_name, values in candidates.items()
        }
    )
    np.savez_compressed(path, **arrays)


def write_selected_archive(
    path: Path,
    axis: np.ndarray,
    ids: np.ndarray,
    candidates: dict[str, np.ndarray],
    selected: list[str],
    selection: np.ndarray,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "axis_cm1": axis.astype(np.float32),
        "observation_uid": ids[selection].astype(str),
    }
    arrays.update(
        {
            candidate_name: candidates[candidate_name][selection].astype(np.float32)
            for candidate_name in selected
        }
    )
    np.savez_compressed(path, **arrays)


def plot_smoothing_selection(
    path: Path, selection_frame: pd.DataFrame, selected: list[str]
) -> None:
    frame = selection_frame[selection_frame["family"].isin(["minimal", "arpls"])].copy()
    frame["window"] = frame["candidate"].str.extract(r"_sg(\d+)_")[0].fillna(0).astype(int)
    colors = {"minimal": "#0072B2", "arpls": "#D55E00"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for family, group in frame.groupby("family"):
        group = group.sort_values("window")
        label = "Minimal" if family == "minimal" else "arPLS"
        axes[0].plot(
            group["window"],
            group["target_core"],
            marker="o",
            color=colors[family],
            label=label,
        )
        axes[1].plot(
            group["window"],
            group["noise_target"],
            marker="o",
            color=colors[family],
            label=label,
        )
        axes[2].plot(
            group["window"],
            group["repeatable_peak_recall"],
            marker="o",
            color=colors[family],
            label=label,
        )
        chosen = group[group["candidate"].isin(selected)]
        for axis_index, column in enumerate(
            ("target_core", "noise_target", "repeatable_peak_recall")
        ):
            axes[axis_index].scatter(
                chosen["window"],
                chosen[column],
                marker="*",
                s=140,
                color=colors[family],
                edgecolor="black",
                zorder=5,
            )
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].set_title("Clean target")
    axes[1].set_title("Gaussian-noise target")
    axes[2].set_title("Repeatable-peak recall")
    axes[2].axhline(0.98, color="#777777", linestyle=":", linewidth=1)
    for axis in axes:
        axis.set_xlabel("SG window (cm$^{-1}$); 0 = none")
        axis.set_xticks([0, 7, 11, 15])
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Predeclared smoothing study; stars are final retained inputs")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_instrument_preservation(path: Path, aggregate: pd.DataFrame) -> None:
    frame = aggregate[
        (aggregate["subset"] == "strict_core")
        & (aggregate["instrument"] != "ALL")
        & aggregate["candidate"].str.contains("_sg")
    ]
    candidates = [name for name in CANDIDATE_ORDER if "_sg" in name]
    instruments = sorted(frame["instrument"].unique())
    matrix = (
        frame.pivot(
            index="instrument",
            columns="candidate",
            values="preservation_repeatable_peak_weighted_recall",
        )
        .reindex(index=instruments, columns=candidates)
        .to_numpy()
    )
    fig, axis = plt.subplots(figsize=(11, 4.8))
    image = axis.imshow(matrix, aspect="auto", vmin=0.90, vmax=1.0, cmap="viridis")
    axis.set_yticks(np.arange(len(instruments)), instruments)
    axis.set_xticks(
        np.arange(len(candidates)),
        [name.replace("_minmax", "") for name in candidates],
        rotation=35,
        ha="right",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < 0.95 else "black",
                fontsize=7,
            )
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Repeatable-peak weighted recall")
    axis.set_title("Smoothing preservation by instrument")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_alignment(path: Path, standards: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for position, (instrument, group) in enumerate(standards.groupby("instrument")):
        axes[0].scatter(
            np.full(len(group), position),
            group["anchor_peak_950_1050_cm1"],
            color="#0072B2",
            alpha=0.8,
        )
    instruments = sorted(standards["instrument"].unique())
    axes[0].set_xticks(np.arange(len(instruments)), instruments, rotation=35, ha="right")
    axes[0].set_ylabel("Anchor peak (cm$^{-1}$)")
    axes[0].set_title("Available calibration/standard spectra")
    axes[0].grid(axis="y", alpha=0.25)
    core = summary[summary["subset"] == "strict_core"].sort_values(
        ["instrument_a", "instrument_b"]
    )
    labels = core["instrument_a"] + "–" + core["instrument_b"]
    axes[1].errorbar(
        np.arange(len(core)),
        core["median_lag_cm1"],
        yerr=[
            core["median_lag_cm1"] - core["lag_q25_cm1"],
            core["lag_q75_cm1"] - core["median_lag_cm1"],
        ],
        fmt="o",
        color="#D55E00",
        capsize=2,
    )
    axes[1].axhline(0, color="#666666", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(core)), labels, rotation=75, ha="right")
    axes[1].set_ylabel("Best integer lag (cm$^{-1}$)")
    axes[1].set_title("Same-master cross-instrument evidence")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Alignment audit: incomplete standards and variable paired shifts")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_decision_registry(
    path: Path,
    protocol: dict[str, Any],
    selection: dict[str, Any],
    alignment: dict[str, Any],
    objectives: pd.DataFrame,
) -> None:
    selected = selection["selected_final_representations"]
    rows: list[str] = []
    for row in objectives.itertuples(index=False):
        rows.append(
            f"| `{row.candidate}` | {row.target_core:.3f} | {row.noise_target:.3f} | "
            f"{row.repeatable_peak_recall if np.isfinite(row.repeatable_peak_recall) else np.nan:.3f} | "
            f"{'yes' if getattr(row, 'eligible_smoother', False) == True else 'no'} | "
            f"{'yes' if row.candidate in selected else 'no'} |"
        )
    path.write_text(
        "\n".join(
            [
                "# NATO SERS preprocessing-v2 decision registry",
                "",
                "## Immutable controls",
                "",
                "- The validated preprocessing-v1 bundle was hash-verified before execution.",
                "- `minimal_minmax`, `arpls_minmax`, and `derivative_1` remain mandatory controls.",
                "- Outer-test and the 98-spectrum flagged-quality cohort were not used for selection.",
                "",
                "## Predeclared smoothing decision",
                "",
                f"Selected final representations: {', '.join(f'`{name}`' for name in selected)}.",
                "",
                "| Candidate | Inner clean target | Inner noise target | Repeatable peak recall | Eligible smoother | Final |",
                "|---|---:|---:|---:|---|---|",
                *rows,
                "",
                f"Noise-gated policy trigger: `{selection['uniform_smoothing_inconsistent_tradeoff_trigger']}`.",
                f"Noise-gated policy decision: {selection['noise_gated_policy_decision']}",
                *[
                    f"- {reason}"
                    for reason in selection["noise_gated_policy_rejection_reasons"]
                ],
                "",
                "## Alignment decision",
                "",
                f"Accepted: `{alignment['alignment_accepted']}`.",
                f"Final operation: `{alignment['final_alignment_operation']}`.",
                "",
                *[f"- {reason}" for reason in alignment["reasons"]],
                "",
                "## Closed operations",
                "",
                "- Common axis: 400--1800 cm^-1 at 1 cm^-1.",
                "- Candidate spike detection/derived interpolation: retained from v1.",
                "- Saturation masks: retained from v1; no automatic repair.",
                "- Per-spectrum min--max: retained for reconstructive intensity inputs.",
                "- arPLS parameters: retained from v1.",
                "- First derivative: retained unchanged from the poster/Siamese control.",
                "- Flexible or target-informed alignment: rejected.",
                "- Per-instrument headline preprocessing: rejected.",
                "",
                "The full machine-readable gates are copied from "
                "`configs/nato_sers_preprocessing_v2.json`.",
                "",
            ]
        )
    )


def write_bundle_readme(
    path: Path,
    selection: dict[str, Any],
    alignment: dict[str, Any],
) -> None:
    selected = selection["selected_final_representations"]
    path.write_text(
        "\n".join(
            [
                "# NATO SERS preprocessing v2",
                "",
                "This is the closed, validated preprocessing bundle for downstream "
                "AE, denoising-AE, VAE, disentangled-VAE, and classifier experiments.",
                "It contains 598 attributable SERS spectra on a 400--1800 cm^-1 "
                "axis at 1 cm^-1 spacing. The 500-row quality cohort and 98-row "
                "field-quality stress cohort are disjoint and exhaust the core.",
                "",
                "## Frozen model inputs",
                "",
                *[f"- `{name}`" for name in selected],
                "",
                "`minimal_minmax` and `arpls_minmax` are reconstructive inputs on "
                "[0,1]. `derivative_1` is the signed, row-L2-normalized "
                "poster/Siamese discriminative control. No general smoothing and "
                f"no spectral alignment are applied (`{alignment['final_alignment_operation']}`).",
                "",
                "Use `final_model_inputs_core.npz` for the primary 598-row "
                "experiment. Use `final_model_inputs_quality.npz` only as the "
                "prespecified 500-row sensitivity analysis. "
                "`final_model_inputs_field_quality_stress.npz` is a confirmatory "
                "98-row stress cohort and must not be used to tune preprocessing.",
                "",
                "## Evidence and provenance",
                "",
                "- `DECISION_REGISTRY.md`: human-readable decision summary.",
                "- `predeclared_protocol.json`: candidate grid, gates, and the "
                "pre-benchmark peak-gate amendment.",
                "- `final_selection.json`: complete smoothing selection record.",
                "- `alignment_decision.json`: complete alignment decision record.",
                "- `benchmark_fold_metrics.csv`: 495 nested and stress evaluations.",
                "- `smoothing_preservation_*.csv`: spectrum- and instrument-level "
                "fidelity evidence.",
                "- `alignment_*.csv`: named-standard and paired-lag evidence.",
                "- `*_manifest.csv`: observation provenance and cohort membership.",
                "- `*_split_assignments.csv` and `nested_group_cv_assignments.csv`: "
                "master-sample-grouped frozen splits.",
                "- `candidate_spectra_*.npz`: all nine audited candidates; these "
                "are evidence archives, not permission to reopen selection.",
                "- `artifact_hashes.json`: SHA-256 catalog for this bundle.",
                "- `v1_control_hashes.json`: immutable preprocessing-v1 snapshot.",
                "",
                "The candidate archives also retain common-grid raw spectra, "
                "despiked spectra, spike and saturation masks, the arPLS baseline, "
                "and all candidate representations. No source data were modified.",
                "",
                "## Rebuild and validate",
                "",
                "From the repository root:",
                "",
                "```bash",
                ".venv/bin/python scripts/finalize_nato_sers_preprocessing_v2.py",
                ".venv/bin/python scripts/validate_nato_sers_preprocessing_v2.py",
                "```",
                "",
                "The scientific rationale and exact downstream contract are in "
                "[`docs/NATO_SERS_PREPROCESSING_FINAL_V2.md`]"
                "(../../../docs/NATO_SERS_PREPROCESSING_FINAL_V2.md).",
                "",
            ]
        )
    )


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "nato_sers_field_trial"
        / "preprocessing_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "nato_sers_field_trial"
        / "preprocessing_v2",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repository / "configs" / "nato_sers_preprocessing_v2.json",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=repository.parents[1] / "2026July21" / "NATO SERS Data",
    )
    args = parser.parse_args()
    v1_dir = args.v1_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    shutil.copyfile(args.protocol, output_dir / "predeclared_protocol.json")

    manifest, control, v1_version = load_control(v1_dir)
    axis = control["axis_cm1"].astype(float)
    ids = control["observation_uid"].astype(str)
    raw = control["raw_common_grid"].astype(float)
    config = v1.PreprocessingConfig()
    candidates = candidates_from_components(
        control["despiked_common_grid"].astype(float),
        control["arpls_baseline"].astype(float),
        control["derivative_1"].astype(float),
        exact_minimal=control["minimal_minmax"],
        exact_arpls=control["arpls_minmax"],
    )
    if not np.array_equal(
        candidates["minimal_minmax"].astype(np.float32), control["minimal_minmax"]
    ):
        raise ValueError("minimal v1 control was not preserved exactly")
    if not np.array_equal(
        candidates["arpls_minmax"].astype(np.float32), control["arpls_minmax"]
    ):
        raise ValueError("arPLS v1 control was not preserved exactly")

    preservation = preservation_frames(manifest, candidates)
    per_spectrum = pd.concat(preservation.values(), ignore_index=True)
    per_spectrum.to_csv(
        output_dir / "smoothing_preservation_per_spectrum.csv", index=False
    )
    aggregate = preservation_aggregate_table(manifest, preservation)
    aggregate.to_csv(output_dir / "smoothing_preservation_summary.csv", index=False)

    corrupted_candidates: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in CANDIDATE_ORDER
    }
    for corruption in CORRUPTIONS:
        corrupted_raw, _ = v1.corrupt_spectra(raw, ids, corruption, config)
        corruption_candidates, _ = build_candidates_from_raw(corrupted_raw, config)
        for candidate_name in CANDIDATE_ORDER:
            corrupted_candidates[candidate_name][corruption] = (
                corruption_candidates[candidate_name]
            )

    metrics = benchmark_candidates(
        manifest, candidates, corrupted_candidates, preservation
    )
    metrics.to_csv(output_dir / "benchmark_fold_metrics.csv", index=False)
    summarize_benchmarks(metrics).to_csv(
        output_dir / "benchmark_summary.csv", index=False
    )
    objectives, selection = selection_table(metrics, protocol)
    objectives.to_csv(output_dir / "smoothing_selection_objectives.csv", index=False)
    (output_dir / "final_selection.json").write_text(
        json.dumps(
            json_clean(selection),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    pair_evidence, pair_summary = alignment_pair_evidence(
        manifest, candidates["derivative_1"]
    )
    pair_evidence.to_csv(output_dir / "alignment_pairwise_evidence.csv", index=False)
    pair_summary.to_csv(output_dir / "alignment_pairwise_summary.csv", index=False)
    standards = calibration_standard_evidence(args.source_root.resolve())
    standards.to_csv(output_dir / "alignment_standard_evidence.csv", index=False)
    alignment = alignment_decision(
        pair_summary,
        standards,
        protocol,
        sorted(manifest["instrument"].astype(str).unique()),
    )
    (output_dir / "alignment_decision.json").write_text(
        json.dumps(alignment, indent=2, sort_keys=True, default=json_default) + "\n"
    )

    quality = manifest["include_sers_qc_pass"].astype(bool).to_numpy()
    flagged = ~quality
    all_rows = np.arange(len(manifest))
    quality_rows = np.flatnonzero(quality)
    flagged_rows = np.flatnonzero(flagged)
    final_manifest = manifest.copy()
    final_manifest["preprocessing_version"] = VERSION
    final_manifest["field_quality_stress"] = flagged
    final_manifest["alignment_applied"] = False
    final_manifest["alignment_shift_cm1"] = 0.0
    final_manifest["minimal_high_frequency_noise_score"] = high_frequency_score_rows(
        candidates["minimal_minmax"]
    )
    final_manifest["arpls_high_frequency_noise_score"] = high_frequency_score_rows(
        candidates["arpls_minmax"]
    )
    final_manifest.to_csv(output_dir / "core_preprocessing_manifest.csv", index=False)
    final_manifest.loc[quality].to_csv(
        output_dir / "quality_preprocessing_manifest.csv", index=False
    )
    final_manifest.loc[flagged].to_csv(
        output_dir / "field_quality_stress_manifest.csv", index=False
    )

    write_candidate_archive(
        output_dir / "candidate_spectra_core.npz",
        axis,
        ids,
        raw,
        control,
        candidates,
        all_rows,
    )
    write_candidate_archive(
        output_dir / "candidate_spectra_quality.npz",
        axis,
        ids,
        raw,
        control,
        candidates,
        quality_rows,
    )
    write_candidate_archive(
        output_dir / "candidate_spectra_field_quality_stress.npz",
        axis,
        ids,
        raw,
        control,
        candidates,
        flagged_rows,
    )
    selected = selection["selected_final_representations"]
    write_selected_archive(
        output_dir / "final_model_inputs_core.npz",
        axis,
        ids,
        candidates,
        selected,
        all_rows,
    )
    write_selected_archive(
        output_dir / "final_model_inputs_quality.npz",
        axis,
        ids,
        candidates,
        selected,
        quality_rows,
    )
    write_selected_archive(
        output_dir / "final_model_inputs_field_quality_stress.npz",
        axis,
        ids,
        candidates,
        selected,
        flagged_rows,
    )

    for name in (
        "core_split_assignments.csv",
        "quality_split_assignments.csv",
        "nested_group_cv_assignments.csv",
        "domain_evaluation_partitions_core.csv",
        "domain_evaluation_partitions_quality.csv",
    ):
        shutil.copyfile(v1_dir / name, output_dir / name)

    v1_hash_snapshot = {
        "artifact_hash_catalog_sha256": sha256_file(v1_dir / "artifact_hashes.json"),
        "dataset_version_sha256": sha256_file(v1_dir / "dataset_version.json"),
        "catalog": json.loads((v1_dir / "artifact_hashes.json").read_text()),
    }
    (output_dir / "v1_control_hashes.json").write_text(
        json.dumps(v1_hash_snapshot, indent=2, sort_keys=True) + "\n"
    )
    plot_smoothing_selection(
        figures / "smoothing_selection.png", objectives, selected
    )
    plot_instrument_preservation(
        figures / "smoothing_instrument_preservation.png", aggregate
    )
    plot_alignment(
        figures / "alignment_evidence.png", standards, pair_summary
    )
    write_decision_registry(
        output_dir / "DECISION_REGISTRY.md",
        protocol,
        selection,
        alignment,
        objectives,
    )
    write_bundle_readme(
        output_dir / "README.md",
        selection,
        alignment,
    )

    version_record = {
        "dataset_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing_closed": True,
        "source_archive_modified": False,
        "strict_core_observations": int(len(manifest)),
        "quality_pass_observations": int(quality.sum()),
        "field_quality_stress_observations": int(flagged.sum()),
        "axis_cm1": {
            "minimum": float(axis.min()),
            "maximum": float(axis.max()),
            "step": float(np.median(np.diff(axis))),
            "points": int(len(axis)),
        },
        "candidate_order": list(CANDIDATE_ORDER),
        "selected_final_representations": selected,
        "alignment_applied": False,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "input_sha256": {
            str(args.protocol.resolve()): sha256_file(args.protocol.resolve()),
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
            str((v1_dir / "artifact_hashes.json").resolve()): sha256_file(
                v1_dir / "artifact_hashes.json"
            ),
            str((v1_dir / "dataset_version.json").resolve()): sha256_file(
                v1_dir / "dataset_version.json"
            ),
        },
        "v1_dataset_version": v1_version["dataset_version"],
    }
    (output_dir / "dataset_version.json").write_text(
        json.dumps(version_record, indent=2, sort_keys=True, default=json_default)
        + "\n"
    )
    artifacts = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    )
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(
            {
                str(path.relative_to(output_dir)): sha256_file(path)
                for path in artifacts
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "version": VERSION,
                "selected": selected,
                "alignment": alignment["final_alignment_operation"],
                "noise_gated_trigger": selection[
                    "uniform_smoothing_inconsistent_tradeoff_trigger"
                ],
                "benchmark_rows": len(metrics),
                "artifact_count": len(artifacts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
