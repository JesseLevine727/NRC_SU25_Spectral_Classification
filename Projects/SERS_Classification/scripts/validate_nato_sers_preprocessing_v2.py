#!/usr/bin/env python3
"""Independently validate the closed NATO SERS preprocessing-v2 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


CANDIDATES = (
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
INTENSITY = CANDIDATES[:-1]
MANDATORY = {"minimal_minmax", "arpls_minmax", "derivative_1"}


def load_strict_json(path: Path) -> object:
    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"Non-standard JSON constant {value} in {path}")

    return json.loads(
        path.read_text(),
        parse_constant=reject_nonstandard_constant,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def minmax(values: np.ndarray) -> np.ndarray:
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    return (values - low) / np.maximum(high - low, 1.0e-12)


def assert_candidate_archive(
    archive: np.lib.npyio.NpzFile,
    expected_ids: np.ndarray,
    expected_rows: int,
) -> None:
    assert archive["axis_cm1"].shape == (1401,)
    assert np.array_equal(archive["axis_cm1"], np.arange(400, 1801))
    assert np.array_equal(archive["observation_uid"].astype(str), expected_ids)
    assert archive["raw_common_grid"].shape == (expected_rows, 1401)
    assert archive["despiked_common_grid"].shape == (expected_rows, 1401)
    assert archive["spike_mask"].shape == (expected_rows, 1401)
    assert archive["saturation_mask"].shape == (expected_rows, 1401)
    assert archive["arpls_baseline"].shape == (expected_rows, 1401)
    assert archive["spike_mask"].dtype == np.bool_
    assert archive["saturation_mask"].dtype == np.bool_
    assert set(CANDIDATES).issubset(archive.files)
    for candidate in CANDIDATES:
        values = archive[candidate]
        assert values.shape == (expected_rows, 1401), candidate
        assert np.isfinite(values).all(), candidate
        if candidate in INTENSITY:
            assert np.allclose(values.min(axis=1), 0.0, atol=2.0e-6), candidate
            assert np.allclose(values.max(axis=1), 1.0, atol=2.0e-6), candidate
            assert float(values.min()) >= -2.0e-6, candidate
            assert float(values.max()) <= 1.0 + 2.0e-6, candidate
        else:
            assert np.allclose(
                np.linalg.norm(values, axis=1), 1.0, atol=2.0e-5
            ), candidate


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "nato_sers_field_trial"
        / "preprocessing_v2",
    )
    parser.add_argument(
        "--v1-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "nato_sers_field_trial"
        / "preprocessing_v1",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repository / "configs" / "nato_sers_preprocessing_v2.json",
    )
    args = parser.parse_args()
    bundle = args.bundle_dir.resolve()
    v1_dir = args.v1_dir.resolve()

    required = {
        "DECISION_REGISTRY.md",
        "README.md",
        "predeclared_protocol.json",
        "dataset_version.json",
        "artifact_hashes.json",
        "v1_control_hashes.json",
        "core_preprocessing_manifest.csv",
        "quality_preprocessing_manifest.csv",
        "field_quality_stress_manifest.csv",
        "candidate_spectra_core.npz",
        "candidate_spectra_quality.npz",
        "candidate_spectra_field_quality_stress.npz",
        "final_model_inputs_core.npz",
        "final_model_inputs_quality.npz",
        "final_model_inputs_field_quality_stress.npz",
        "core_split_assignments.csv",
        "quality_split_assignments.csv",
        "nested_group_cv_assignments.csv",
        "domain_evaluation_partitions_core.csv",
        "domain_evaluation_partitions_quality.csv",
        "benchmark_fold_metrics.csv",
        "benchmark_summary.csv",
        "smoothing_preservation_per_spectrum.csv",
        "smoothing_preservation_summary.csv",
        "smoothing_selection_objectives.csv",
        "final_selection.json",
        "alignment_pairwise_evidence.csv",
        "alignment_pairwise_summary.csv",
        "alignment_standard_evidence.csv",
        "alignment_decision.json",
        "figures/smoothing_selection.png",
        "figures/smoothing_instrument_preservation.png",
        "figures/alignment_evidence.png",
    }
    missing = sorted(name for name in required if not (bundle / name).is_file())
    assert not missing, missing

    protocol = load_strict_json(bundle / "predeclared_protocol.json")
    assert protocol == load_strict_json(args.protocol)
    assert protocol["protocol_version"] == "nato-sers-preprocessing-v2"
    assert protocol["smoothing"]["windows_points_and_cm1"] == [7, 11, 15]
    assert protocol["smoothing"]["polynomial_order"] == 3
    assert protocol["preservation_gates"]["all_peak_recall_is_hard_gate"] is False
    assert protocol["alignment"]["flexible_per_spectrum_warping_allowed"] is False
    assert protocol["alignment"]["target_informed_alignment_allowed"] is False

    v1_hashes = load_strict_json(v1_dir / "artifact_hashes.json")
    v1_snapshot = load_strict_json(bundle / "v1_control_hashes.json")
    assert v1_snapshot["catalog"] == v1_hashes
    assert v1_snapshot["artifact_hash_catalog_sha256"] == sha256_file(
        v1_dir / "artifact_hashes.json"
    )
    assert v1_snapshot["dataset_version_sha256"] == sha256_file(
        v1_dir / "dataset_version.json"
    )
    for relative_path, expected_hash in v1_hashes.items():
        assert sha256_file(v1_dir / relative_path) == expected_hash, relative_path

    manifest = pd.read_csv(bundle / "core_preprocessing_manifest.csv")
    quality_manifest = pd.read_csv(bundle / "quality_preprocessing_manifest.csv")
    stress_manifest = pd.read_csv(bundle / "field_quality_stress_manifest.csv")
    assert len(manifest) == 598
    assert len(quality_manifest) == 500
    assert len(stress_manifest) == 98
    assert manifest["observation_uid"].is_unique
    assert quality_manifest["observation_uid"].is_unique
    assert stress_manifest["observation_uid"].is_unique
    assert set(quality_manifest["observation_uid"]).isdisjoint(
        stress_manifest["observation_uid"]
    )
    assert set(quality_manifest["observation_uid"]) | set(
        stress_manifest["observation_uid"]
    ) == set(manifest["observation_uid"])
    assert quality_manifest["include_sers_qc_pass"].astype(bool).all()
    assert (~stress_manifest["include_sers_qc_pass"].astype(bool)).all()
    assert stress_manifest["field_quality_stress"].astype(bool).all()
    assert not manifest["alignment_applied"].astype(bool).any()
    assert (manifest["alignment_shift_cm1"] == 0.0).all()
    assert set(manifest["grouped_sample_fold_5"]) == set(range(5))
    assert (
        manifest.groupby("master_sample_id")["grouped_sample_fold_5"].nunique().max()
        == 1
    )

    core = np.load(bundle / "candidate_spectra_core.npz")
    quality = np.load(bundle / "candidate_spectra_quality.npz")
    stress = np.load(bundle / "candidate_spectra_field_quality_stress.npz")
    core_ids = manifest["observation_uid"].astype(str).to_numpy()
    quality_ids = quality_manifest["observation_uid"].astype(str).to_numpy()
    stress_ids = stress_manifest["observation_uid"].astype(str).to_numpy()
    assert_candidate_archive(core, core_ids, 598)
    assert_candidate_archive(quality, quality_ids, 500)
    assert_candidate_archive(stress, stress_ids, 98)

    positions = {uid: index for index, uid in enumerate(core_ids)}
    quality_rows = np.asarray([positions[uid] for uid in quality_ids])
    stress_rows = np.asarray([positions[uid] for uid in stress_ids])
    for key in core.files:
        if key == "axis_cm1":
            assert np.array_equal(core[key], quality[key])
            assert np.array_equal(core[key], stress[key])
        else:
            assert np.array_equal(core[key][quality_rows], quality[key]), key
            assert np.array_equal(core[key][stress_rows], stress[key]), key

    v1_archive = np.load(v1_dir / "candidate_spectra_core.npz")
    for key in (
        "axis_cm1",
        "observation_uid",
        "raw_common_grid",
        "despiked_common_grid",
        "spike_mask",
        "saturation_mask",
        "arpls_baseline",
        "minimal_minmax",
        "arpls_minmax",
        "derivative_1",
    ):
        assert np.array_equal(core[key], v1_archive[key]), key

    despiked = core["despiked_common_grid"].astype(float)
    residual = despiked - core["arpls_baseline"].astype(float)
    for window in (7, 11, 15):
        expected_minimal = minmax(
            savgol_filter(despiked, window, 3, axis=1, mode="interp")
        ).astype(np.float32)
        expected_arpls = minmax(
            savgol_filter(residual, window, 3, axis=1, mode="interp")
        ).astype(np.float32)
        assert np.array_equal(
            expected_minimal, core[f"minimal_sg{window}_minmax"]
        ), window
        assert np.array_equal(
            expected_arpls, core[f"arpls_sg{window}_minmax"]
        ), window

    preservation = pd.read_csv(bundle / "smoothing_preservation_per_spectrum.csv")
    assert len(preservation) == 598 * len(CANDIDATES)
    assert set(preservation["candidate"]) == set(CANDIDATES)
    assert set(preservation["observation_uid"]) == set(core_ids)
    aggregate = pd.read_csv(bundle / "smoothing_preservation_summary.csv")
    assert set(aggregate["subset"]) == {
        "strict_core",
        "quality_pass",
        "field_quality_stress",
    }

    metrics = pd.read_csv(bundle / "benchmark_fold_metrics.csv")
    assert len(metrics) == 495
    assert set(metrics["representation"]) == set(CANDIDATES)
    assert set(metrics["subset"]) == {
        "strict_core",
        "quality_pass",
        "field_quality_stress",
    }
    assert len(
        metrics[
            (metrics["subset"].isin(["strict_core", "quality_pass"]))
            & (metrics["stage"] == "inner_validation")
        ]
    ) == 9 * 2 * 5 * 4
    assert len(
        metrics[
            (metrics["subset"].isin(["strict_core", "quality_pass"]))
            & (metrics["stage"] == "outer_test")
        ]
    ) == 9 * 2 * 5
    assert len(
        metrics[
            (metrics["subset"] == "field_quality_stress")
            & (metrics["stage"] == "outer_test")
        ]
    ) == 9 * 5
    assert metrics[
        (metrics["subset"] == "field_quality_stress")
        & (metrics["stage"] == "inner_validation")
    ].empty

    objectives = pd.read_csv(bundle / "smoothing_selection_objectives.csv")
    selection = load_strict_json(bundle / "final_selection.json")
    selected = selection["selected_final_representations"]
    assert set(selected) >= MANDATORY
    assert len(selected) <= 5
    assert selection["stress_cohort_role"].startswith("confirmatory")
    assert selection["noise_gated_policy_considered"] is True
    assert selection["noise_gated_policy_implemented"] is False
    assert selection["noise_gated_policy_rejection_reasons"]
    smoothers = objectives[objectives["is_smoothed"].astype(bool)]
    assert not smoothers["eligible_smoother"].astype(bool).any()
    assert not selection["selected_smoothers"]
    assert selected == ["minimal_minmax", "arpls_minmax", "derivative_1"]
    assert (
        smoothers["repeatable_peak_recall"]
        < protocol["preservation_gates"][
            "repeatable_peak_weighted_recall_minimum"
        ]
    ).all()

    for cohort, archive, ids in (
        ("core", core, core_ids),
        ("quality", quality, quality_ids),
        ("field_quality_stress", stress, stress_ids),
    ):
        frozen = np.load(bundle / f"final_model_inputs_{cohort}.npz")
        assert set(frozen.files) == {"axis_cm1", "observation_uid", *selected}
        assert np.array_equal(frozen["axis_cm1"], archive["axis_cm1"])
        assert np.array_equal(frozen["observation_uid"].astype(str), ids)
        for candidate in selected:
            assert np.array_equal(frozen[candidate], archive[candidate]), candidate

    alignment = load_strict_json(bundle / "alignment_decision.json")
    standards = pd.read_csv(bundle / "alignment_standard_evidence.csv")
    pair_evidence = pd.read_csv(bundle / "alignment_pairwise_evidence.csv")
    pair_summary = pd.read_csv(bundle / "alignment_pairwise_summary.csv")
    assert alignment["alignment_accepted"] is False
    assert alignment["final_alignment_operation"] == "none"
    assert len(alignment["standard_covered_instruments"]) == 5
    assert len(alignment["standard_missing_instruments"]) == 5
    assert len(standards) == 17
    assert set(standards["anchor_peak_950_1050_cm1"]).issubset({1002.0, 1003.0, 1004.0})
    assert len(pair_evidence) == 2473
    assert set(pair_summary["subset"]) == {"strict_core", "quality_pass"}
    assert pair_evidence["best_integer_lag_cm1_b_relative_to_a"].between(-5, 5).all()

    nested = pd.read_csv(bundle / "nested_group_cv_assignments.csv")
    assert len(nested) == 598 * 5
    for _, group in nested.groupby("outer_fold"):
        test = group[group["outer_partition"] == "test"]
        development = group[group["outer_partition"] == "development"]
        assert set(test["master_sample_id"]).isdisjoint(
            development["master_sample_id"]
        )

    version = load_strict_json(bundle / "dataset_version.json")
    assert version["dataset_version"] == "nato-sers-preprocessing-v2"
    assert version["preprocessing_closed"] is True
    assert version["strict_core_observations"] == 598
    assert version["quality_pass_observations"] == 500
    assert version["field_quality_stress_observations"] == 98
    assert version["selected_final_representations"] == selected
    assert version["alignment_applied"] is False
    for input_path, expected_hash in version["input_sha256"].items():
        assert sha256_file(Path(input_path)) == expected_hash, input_path

    hashes = load_strict_json(bundle / "artifact_hashes.json")
    assert required - {"artifact_hashes.json"} <= set(hashes)
    for relative_path, expected_hash in hashes.items():
        assert sha256_file(bundle / relative_path) == expected_hash, relative_path

    print(
        json.dumps(
            {
                "status": "PASS",
                "bundle": str(bundle),
                "strict_core": 598,
                "quality_pass": 500,
                "field_quality_stress": 98,
                "candidate_representations": len(CANDIDATES),
                "selected_final": selected,
                "smoothing": "rejected",
                "alignment": "rejected",
                "benchmark_rows": len(metrics),
                "v1_hashes_verified": len(v1_hashes),
                "v2_hashes_verified": len(hashes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
