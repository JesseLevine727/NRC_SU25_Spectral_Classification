#!/usr/bin/env python3
"""Validate every invariant of the frozen NATO SERS preprocessing bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_CANDIDATES = {
    "minimal_minmax",
    "robust_minmax",
    "asls_minmax",
    "arpls_minmax",
    "derivative_1",
    "derivative_2",
}
MINMAX_CANDIDATES = {
    "minimal_minmax",
    "robust_minmax",
    "asls_minmax",
    "arpls_minmax",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_archive(
    archive: np.lib.npyio.NpzFile,
    expected_rows: int,
    expected_ids: np.ndarray,
) -> None:
    assert archive["axis_cm1"].shape == (1401,)
    assert np.array_equal(archive["axis_cm1"], np.arange(400, 1801))
    assert archive["observation_uid"].shape == (expected_rows,)
    assert np.array_equal(archive["observation_uid"].astype(str), expected_ids)
    assert len(np.unique(expected_ids)) == expected_rows
    assert archive["raw_common_grid"].shape == (expected_rows, 1401)
    assert archive["despiked_common_grid"].shape == (expected_rows, 1401)
    assert archive["spike_mask"].shape == (expected_rows, 1401)
    assert archive["saturation_mask"].shape == (expected_rows, 1401)
    assert archive["spike_mask"].dtype == np.bool_
    assert archive["saturation_mask"].dtype == np.bool_

    present_candidates = EXPECTED_CANDIDATES & set(archive.files)
    assert present_candidates == EXPECTED_CANDIDATES
    for candidate in EXPECTED_CANDIDATES:
        values = archive[candidate]
        assert values.shape == (expected_rows, 1401)
        assert np.isfinite(values).all(), candidate
        if candidate in MINMAX_CANDIDATES:
            assert float(values.min()) >= -1.0e-6, candidate
            assert float(values.max()) <= 1.0 + 1.0e-6, candidate
            assert np.allclose(values.min(axis=1), 0.0, atol=2.0e-6), candidate
            assert np.allclose(values.max(axis=1), 1.0, atol=2.0e-6), candidate
        else:
            assert np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=2.0e-5), candidate


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "nato_sers_field_trial"
        / "preprocessing_v1",
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=repository / "Workspace" / "nato_sers_field_trial",
    )
    args = parser.parse_args()
    bundle = args.bundle_dir.resolve()
    upstream = args.upstream_dir.resolve()

    required_files = {
        "README.md",
        "dataset_version.json",
        "artifact_hashes.json",
        "artifact_detection_summary.json",
        "core_preprocessing_manifest.csv",
        "quality_preprocessing_manifest.csv",
        "core_split_assignments.csv",
        "quality_split_assignments.csv",
        "nested_group_cv_assignments.csv",
        "domain_evaluation_partitions_core.csv",
        "domain_evaluation_partitions_quality.csv",
        "candidate_spectra_core.npz",
        "candidate_spectra_quality.npz",
        "frozen_model_inputs_core.npz",
        "frozen_model_inputs_quality.npz",
        "benchmark_fold_metrics.csv",
        "benchmark_summary.csv",
        "peak_preservation_metrics.csv",
        "selection_objectives.csv",
        "frozen_selection.json",
        "figures/preprocessing_tradeoff.png",
        "figures/selected_corruption_robustness.png",
    }
    missing = sorted(name for name in required_files if not (bundle / name).is_file())
    assert not missing, f"Missing bundle files: {missing}"

    core_manifest = pd.read_csv(bundle / "core_preprocessing_manifest.csv")
    quality_manifest = pd.read_csv(bundle / "quality_preprocessing_manifest.csv")
    assert len(core_manifest) == 598
    assert len(quality_manifest) == 500
    assert core_manifest["observation_uid"].is_unique
    assert quality_manifest["observation_uid"].is_unique
    assert set(quality_manifest["observation_uid"]) == set(
        core_manifest.loc[
            core_manifest["include_sers_qc_pass"].astype(bool), "observation_uid"
        ]
    )
    assert set(core_manifest["grouped_sample_fold_5"]) == set(range(5))
    assert (
        core_manifest.groupby("master_sample_id")["grouped_sample_fold_5"]
        .nunique()
        .max()
        == 1
    )
    assert set(
        core_manifest.groupby("grouped_sample_fold_5")["target_analyte"].nunique()
    ) == {7}
    assert set(
        quality_manifest.groupby("grouped_sample_fold_5")["target_analyte"].nunique()
    ) == {7}

    core = np.load(bundle / "candidate_spectra_core.npz")
    quality = np.load(bundle / "candidate_spectra_quality.npz")
    assert_archive(
        core, 598, core_manifest["observation_uid"].astype(str).to_numpy()
    )
    assert_archive(
        quality, 500, quality_manifest["observation_uid"].astype(str).to_numpy()
    )
    quality_positions = {
        uid: index for index, uid in enumerate(core["observation_uid"].astype(str))
    }
    order = np.asarray(
        [quality_positions[uid] for uid in quality["observation_uid"].astype(str)]
    )
    for candidate in EXPECTED_CANDIDATES:
        assert np.array_equal(core[candidate][order], quality[candidate]), candidate

    upstream_archive = np.load(upstream / "sers_core_spectra_raw_common_grid.npz")
    upstream_positions = {
        uid: index
        for index, uid in enumerate(upstream_archive["observation_uid"].astype(str))
    }
    upstream_order = np.asarray(
        [upstream_positions[uid] for uid in core["observation_uid"].astype(str)]
    )
    assert np.array_equal(
        core["raw_common_grid"], upstream_archive["intensity"][upstream_order]
    )
    assert np.array_equal(
        core_manifest["candidate_spike_count"].to_numpy(),
        core["spike_mask"].sum(axis=1),
    )
    assert np.array_equal(
        core_manifest["saturation_point_count"].to_numpy(),
        core["saturation_mask"].sum(axis=1),
    )

    nested = pd.read_csv(bundle / "nested_group_cv_assignments.csv")
    assert len(nested) == 598 * 5
    for outer_fold, group in nested.groupby("outer_fold"):
        assert len(group) == 598
        test = group[group["outer_partition"] == "test"]
        development = group[group["outer_partition"] == "development"]
        assert set(test["master_sample_id"]).isdisjoint(development["master_sample_id"])
        assert set(test["grouped_sample_fold_5"]) == {outer_fold}
        assert development["inner_validation_fold"].notna().all()

    domain_core = pd.read_csv(bundle / "domain_evaluation_partitions_core.csv")
    domain_quality = pd.read_csv(bundle / "domain_evaluation_partitions_quality.csv")
    assert len(domain_core) == 598 * 28
    assert len(domain_quality) == 500 * 28

    metrics = pd.read_csv(bundle / "benchmark_fold_metrics.csv")
    assert len(metrics) == 6 * 2 * (5 + 5 * 4)
    assert set(metrics["representation"]) == EXPECTED_CANDIDATES
    assert set(metrics["subset"]) == {"strict_core", "quality_pass"}
    assert set(metrics["stage"]) == {"inner_validation", "outer_test"}
    assert len(metrics[metrics["stage"] == "outer_test"]) == 6 * 2 * 5
    assert len(metrics[metrics["stage"] == "inner_validation"]) == 6 * 2 * 5 * 4
    required_metric_columns = {
        "target_pca_balanced_accuracy",
        "target_centroid_balanced_accuracy",
        "instrument_probe_increment_over_target_only",
        "sensor_probe_increment_over_target_only",
        "geometry_same_master_cross_instrument_mean_distance",
        "geometry_cross_instrument_separation_margin",
        "corruption_composite_pca_balanced_accuracy",
        "corruption_spike_pca_prediction_agreement",
        "corruption_noise_mean_shape_correlation",
    }
    assert required_metric_columns.issubset(metrics.columns)
    bounded_columns = [
        column
        for column in metrics.columns
        if "balanced_accuracy" in column
        or "prediction_agreement" in column
        or "mean_shape_correlation" in column
    ]
    for column in bounded_columns:
        values = metrics[column].dropna().to_numpy()
        if "mean_shape_correlation" in column:
            assert np.all((-1.0 - 1e-6 <= values) & (values <= 1.0 + 1e-6)), column
        else:
            assert np.all((0.0 <= values) & (values <= 1.0)), column

    peaks = pd.read_csv(bundle / "peak_preservation_metrics.csv")
    assert len(peaks) == 6 * 2 * 5
    intensity_peaks = peaks[peaks["representation"].isin(MINMAX_CANDIDATES)]
    assert intensity_peaks["mean_peak_recall_5cm"].notna().all()
    derivative_peaks = peaks[~peaks["representation"].isin(MINMAX_CANDIDATES)]
    assert derivative_peaks["mean_peak_recall_5cm"].isna().all()

    selection = json.loads((bundle / "frozen_selection.json").read_text())
    selected = selection["selected_frozen_representations"]
    assert 2 <= len(selected) <= 3
    assert len(selected) == len(set(selected))
    assert set(selected).issubset(EXPECTED_CANDIDATES)
    assert "minimal_minmax" in selected
    assert any(name in {"asls_minmax", "arpls_minmax"} for name in selected)
    assert any(name in {"derivative_1", "derivative_2"} for name in selected)
    selected_rows = [
        row for row in selection["table"] if row["representation"] in selected
    ]
    assert all(row["global_pareto"] for row in selected_rows)
    frozen_core = np.load(bundle / "frozen_model_inputs_core.npz")
    frozen_quality = np.load(bundle / "frozen_model_inputs_quality.npz")
    expected_frozen_keys = {"axis_cm1", "observation_uid", *selected}
    assert set(frozen_core.files) == expected_frozen_keys
    assert set(frozen_quality.files) == expected_frozen_keys
    assert np.array_equal(frozen_core["axis_cm1"], core["axis_cm1"])
    assert np.array_equal(frozen_quality["axis_cm1"], quality["axis_cm1"])
    assert np.array_equal(frozen_core["observation_uid"], core["observation_uid"])
    assert np.array_equal(
        frozen_quality["observation_uid"], quality["observation_uid"]
    )
    for representation in selected:
        assert np.array_equal(frozen_core[representation], core[representation])
        assert np.array_equal(
            frozen_quality[representation], quality[representation]
        )

    detection = json.loads((bundle / "artifact_detection_summary.json").read_text())
    assert detection["spike"]["injected_spike_points"] == 598 * 2
    assert detection["spike"]["injected_spike_recall"] is not None
    assert detection["composite"]["injected_spike_recall"] is not None

    version = json.loads((bundle / "dataset_version.json").read_text())
    assert version["dataset_version"] == "nato-sers-preprocessing-v1"
    assert version["strict_core_observations"] == 598
    assert version["quality_pass_observations"] == 500
    assert version["axis_cm1"] == {
        "minimum": 400.0,
        "maximum": 1800.0,
        "step": 1.0,
        "points": 1401,
    }
    assert set(version["candidate_order"]) == EXPECTED_CANDIDATES
    assert version["selected_frozen_representations"] == selected
    for path_string, expected_hash in version["input_sha256"].items():
        assert sha256_file(Path(path_string)) == expected_hash, path_string

    hashes = json.loads((bundle / "artifact_hashes.json").read_text())
    for relative_path, expected_hash in hashes.items():
        assert sha256_file(bundle / relative_path) == expected_hash, relative_path
    assert required_files - {"artifact_hashes.json"} <= set(hashes)

    print(
        json.dumps(
            {
                "status": "PASS",
                "bundle": str(bundle),
                "strict_core": len(core_manifest),
                "quality_pass": len(quality_manifest),
                "candidates": sorted(EXPECTED_CANDIDATES),
                "selected": selected,
                "benchmark_rows": len(metrics),
                "hashes_verified": len(hashes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
