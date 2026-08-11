from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation.p03_controls import (
    build_metadata_only_classifier,
    metadata_feature_frame,
    permute_master_labels,
)
from atlas_sers.governance.canonical import sha256_value
from atlas_sers.models.classical import UnsupportedCandidate


def _metadata() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(18):
        label = ("a", "b", "c")[index % 3]
        for replicate in range(2):
            rows.append(
                {
                    "observation_uid": f"row-{index}-{replicate}",
                    "master_sample_id": f"master-{index}",
                    "station": "station-a" if index < 12 else "station-b",
                    "target_analyte": label,
                    "instrument": f"platform-{1 + index % 2}",
                    "sensor_family": f"sensor-{index % 2}",
                    "sensor_variant": f"variant-{index % 3}",
                    "source_format": "text" if index % 2 else "sheet",
                    "team": f"team-{index % 2}",
                    "averages": np.nan if index == 0 else 2 + index % 3,
                    "laser_power": 10 + index,
                    "n_points": 1401,
                    "axis_min_cm1": 400,
                    "axis_max_cm1": 1800,
                    "axis_step_median_cm1": 1.0,
                    "leading_constant_points": index % 2,
                    "trailing_constant_points": 0,
                    "finite_fraction": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _parameters() -> dict[str, object]:
    return {
        "base_parameters": {"C": 1.0, "l1_ratio": 0.5},
        "categorical_features": [
            "instrument",
            "instrument_family",
            "sensor_family",
            "sensor_variant",
            "source_format",
            "team",
        ],
        "numeric_features": [
            "averages",
            "laser_power",
            "n_points",
            "axis_min_cm1",
            "axis_max_cm1",
            "axis_step_median_cm1",
            "leading_constant_points",
            "trailing_constant_points",
            "finite_fraction",
        ],
    }


def test_master_permutation_is_deterministic_master_safe_and_blocked_by_station() -> None:
    metadata = _metadata()
    first = permute_master_labels(metadata, seed=20261001)
    second = permute_master_labels(metadata, seed=20261001)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.mapping_sha256 == second.mapping_sha256
    assert first.label_sha256 == second.label_sha256
    assert first.masters == 18
    assigned = metadata.assign(permuted=first.labels)
    assert assigned.groupby("master_sample_id").permuted.nunique().eq(1).all()
    for station, block in assigned.groupby("station"):
        del station
        original = (
            block[["master_sample_id", "target_analyte"]]
            .drop_duplicates()
            .target_analyte.value_counts()
            .sort_index()
        )
        permuted = (
            block[["master_sample_id", "permuted"]]
            .drop_duplicates()
            .permuted.value_counts()
            .sort_index()
        )
        pd.testing.assert_series_equal(original, permuted, check_names=False)


def test_metadata_allowlist_derives_platform_family_and_rejects_missing_fields() -> None:
    metadata = _metadata()
    parameters = _parameters()
    features = metadata_feature_frame(
        metadata,
        categorical_features=parameters["categorical_features"],
        numeric_features=parameters["numeric_features"],
    )
    assert set(features.instrument_family) == {"platform"}
    assert list(features) == [
        *parameters["categorical_features"],
        *parameters["numeric_features"],
    ]
    with pytest.raises(ValueError, match="misses frozen features"):
        metadata_feature_frame(
            metadata.drop(columns="team"),
            categorical_features=parameters["categorical_features"],
            numeric_features=parameters["numeric_features"],
        )


def test_metadata_classifier_never_accepts_spectra_or_refits_on_target_batch() -> None:
    metadata = _metadata()
    fit = metadata[metadata.station == "station-a"].reset_index(drop=True)
    model = build_metadata_only_classifier(_parameters(), seed=20260805).fit(
        fit,
        fit.target_analyte.to_numpy(),
        observation_uids=fit.observation_uid.to_numpy(),
        master_ids=fit.master_sample_id.to_numpy(),
    )
    assert model.fit_audit is not None
    assert model.fit_audit.observation_uid_sha256 == sha256_value(
        sorted(fit.observation_uid)
    )
    state = model.source_state_sha256_
    target = metadata[metadata.station == "station-b"].copy()
    target.loc[:, "instrument"] = "unseen-platform-99"
    assert model.scores(target).shape == (len(target), 3)
    assert model.probabilities(target).shape == (len(target), 3)
    assert model.source_state_sha256_ == state
    with pytest.raises(UnsupportedCandidate, match="requires_dataframe"):
        model.scores(np.ones((2, 10)))  # type: ignore[arg-type]
