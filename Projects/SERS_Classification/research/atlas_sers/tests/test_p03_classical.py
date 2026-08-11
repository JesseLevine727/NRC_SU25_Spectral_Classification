from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation.classical import (
    apply_temperature,
    classification_metrics,
    fit_temperature,
    instrument_balanced_master_probabilities,
    select_lexicographic_candidate,
)
from atlas_sers.models.classical import UnsupportedCandidate, build_classical_estimator


def _classification_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260805)
    labels = np.repeat(np.asarray(["a", "b", "c"]), 20)
    centers = {"a": -1.0, "b": 0.0, "c": 1.0}
    x = np.vstack([rng.normal(centers[label], 0.25, 20) for label in labels])
    observations = np.asarray([f"row-{index:03d}" for index in range(len(labels))])
    masters = np.asarray([f"master-{index // 2:03d}" for index in range(len(labels))])
    return x, labels, observations, masters


@pytest.mark.parametrize(
    ("model_id", "parameters", "externally_calibrated"),
    [
        ("C-PRIOR", {"prior": "empirical"}, False),
        ("C-SPECTRAL-MATCH", {"metric": "pearson"}, False),
        (
            "C-NEAREST-CENTROID",
            {"metric": "euclidean", "shrink_threshold": None},
            False,
        ),
        (
            "C-NEAREST-CENTROID",
            {"metric": "cosine", "shrink_threshold": 0.01},
            False,
        ),
        ("C-PCA-LDA", {"pca_components": 5, "lda": "svd"}, False),
        (
            "C-PLS-DA",
            {"components": 2, "head": "class-balanced multinomial logistic"},
            False,
        ),
        ("C-LOGREG-EN", {"C": 1, "l1_ratio": 0.5}, False),
        ("C-RBF-SVM", {"C": 1, "gamma": "scale", "class_weight": "balanced"}, True),
        (
            "C-RANDOM-FOREST",
            {
                "n_estimators": 10,
                "max_features": "sqrt",
                "min_samples_leaf": 1,
                "class_weight": "balanced",
            },
            False,
        ),
        (
            "C-EXTRA-TREES",
            {
                "n_estimators": 10,
                "max_features": "sqrt",
                "min_samples_leaf": 1,
                "class_weight": "balanced",
                "bootstrap": False,
            },
            False,
        ),
    ],
)
def test_every_classical_family_has_audited_scores(
    model_id: str, parameters: dict[str, object], externally_calibrated: bool
) -> None:
    x, labels, observations, masters = _classification_data()
    model = build_classical_estimator(
        model_id,
        parameters,
        fit_samples=len(x),
        features=x.shape[1],
        seed=20260805,
    ).fit(
        x,
        labels,
        observation_uids=observations,
        master_ids=masters,
    )
    assert model.fit_audit is not None
    assert model.fit_audit.observations == 60
    assert model.fit_audit.masters == 30
    assert model.scores(x[:4]).shape == (4, 3)
    assert model.predict(x[:4]).shape == (4,)
    assert model.serialized_size_bytes() > 0
    if externally_calibrated:
        with pytest.raises(UnsupportedCandidate, match="cross_fitted_calibration"):
            model.probabilities(x[:4])
    else:
        probabilities = model.probabilities(x[:4])
        assert probabilities.shape == (4, 3)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_rank_limited_candidates_are_unsupported_not_silently_changed() -> None:
    with pytest.raises(UnsupportedCandidate, match="pca_components_exceed_rank"):
        build_classical_estimator(
            "C-PCA-LDA",
            {"pca_components": 40, "lda": "svd"},
            fit_samples=10,
            features=20,
        )
    x, labels, observations, masters = _classification_data()
    model = build_classical_estimator(
        "C-PLS-DA",
        {"components": 16, "head": "class-balanced multinomial logistic"},
        fit_samples=10,
        features=20,
    )
    selected = np.r_[0:4, 20:24, 40:44]
    with pytest.raises(UnsupportedCandidate, match="pls_components_exceed_rank"):
        model.fit(
            x[selected],
            labels[selected],
            observation_uids=observations[selected],
            master_ids=masters[selected],
        )


def test_source_covariance_augmentation_never_fits_or_transforms_target_statistics() -> None:
    x, labels, observations, masters = _classification_data()
    domains = np.tile(np.asarray(["unit-a", "unit-b"]), len(x) // 2)
    shifted = x.copy()
    shifted[domains == "unit-b"] = shifted[domains == "unit-b"] * np.linspace(
        0.7, 1.3, x.shape[1]
    ) + 0.4
    model = build_classical_estimator(
        "C-SOURCE-CORAL",
        {
            "base_model_id": "C-PCA-LDA",
            "base_parameters": {"pca_components": 5, "lda": "svd"},
            "rank_cap": 20,
            "ridge_fraction": 1e-3,
        },
        fit_samples=len(shifted),
        features=shifted.shape[1],
        seed=20260805,
    ).fit(
        shifted,
        labels,
        observation_uids=observations,
        master_ids=masters,
        domain_ids=domains,
    )
    assert model.fit_audit is not None
    assert model.fit_audit.domain_uid_sha256 is not None
    estimator = model.estimator
    assert estimator.source_domain_order_ == ("unit-a", "unit-b")
    assert estimator.augmented_masters_ == 30
    assert estimator.augmented_observations_ == 90
    frozen_state = estimator.source_domain_state_sha256_
    assert model.scores(np.full((1, x.shape[1]), 50.0)).shape == (1, 3)
    assert model.scores(np.full((7, x.shape[1]), -50.0)).shape == (7, 3)
    assert estimator.source_domain_state_sha256_ == frozen_state

    unsupported = build_classical_estimator(
        "C-SOURCE-CORAL",
        {
            "base_model_id": "C-PCA-LDA",
            "base_parameters": {"pca_components": 5, "lda": "svd"},
        },
        fit_samples=len(shifted),
        features=shifted.shape[1],
    )
    with pytest.raises(UnsupportedCandidate, match="requires_two_source_domains"):
        unsupported.fit(
            shifted,
            labels,
            observation_uids=observations,
            master_ids=masters,
            domain_ids=np.asarray(["one-unit"] * len(shifted)),
        )


def test_temperature_uses_traceable_cross_fitted_scores() -> None:
    scores = np.asarray([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [4.0, 0.0, 0.0]])
    labels = np.asarray(["a", "b", "c", "b"])
    observations = np.asarray(["o1", "o2", "o3", "o4"])
    masters = np.asarray(["m1", "m2", "m3", "m4"])
    calibration = fit_temperature(
        scores,
        labels,
        class_vocabulary=["a", "b", "c"],
        observation_uids=observations,
        master_ids=masters,
    )
    assert calibration.optimizer_success
    assert calibration.temperature > 0
    assert calibration.observations == calibration.masters == 4
    assert len(calibration.state_sha256) == 64
    probabilities = apply_temperature(scores, calibration)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_instrument_balanced_master_aggregation_prevents_replicate_dominance() -> None:
    probabilities = np.vstack(
        [np.tile([0.9, 0.1], (10, 1)), np.asarray([[0.1, 0.9]]), np.asarray([[0.2, 0.8]])]
    )
    result = instrument_balanced_master_probabilities(
        probabilities=probabilities,
        true_labels=np.asarray(["a"] * 11 + ["b"]),
        master_ids=np.asarray(["m1"] * 11 + ["m2"]),
        instruments=np.asarray(["i1"] * 10 + ["i2", "i1"]),
        class_vocabulary=["a", "b"],
    )
    first = np.asarray(result.loc[result.master_sample_id == "m1", "probabilities"].iloc[0])
    np.testing.assert_allclose(first, [0.5, 0.5])
    assert int(result.loc[result.master_sample_id == "m1", "instrument_views"].iloc[0]) == 2


def test_metrics_and_lexicographic_selection_follow_registry_order() -> None:
    metrics = classification_metrics(
        np.asarray(["a", "a", "b", "b"]),
        np.asarray(["a", "b", "b", "b"]),
        class_vocabulary=["a", "b"],
        probabilities=np.asarray([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.2, 0.8]]),
    )
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["per_class_recall"] == {"a": 0.5, "b": 1.0}
    assert metrics["negative_log_likelihood"] is not None
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "simple",
                "model_id": "m",
                "complexity_rank": 0,
                "declared_candidate_order": 0,
            },
            {
                "candidate_id": "complex",
                "model_id": "m",
                "complexity_rank": 1,
                "declared_candidate_order": 1,
            },
        ]
    )
    units = pd.DataFrame(
        [
            {
                "candidate_id": candidate,
                "selection_unit_id": unit,
                "balanced_accuracy": 0.8,
                "macro_f1": 0.7,
            }
            for candidate in ("simple", "complex")
            for unit in ("u1", "u2")
        ]
    )
    winner, trace = select_lexicographic_candidate(units, candidates)
    assert winner.candidate_id == "simple"
    assert trace.loc[trace.candidate_id == "simple", "selected"].item()


def test_selection_preserves_failures_and_requires_every_declared_seed() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "stable",
                "model_id": "deterministic",
                "complexity_rank": 0,
                "declared_candidate_order": 0,
                "seed_count": 1,
            },
            {
                "candidate_id": "fragile",
                "model_id": "forest",
                "complexity_rank": 1,
                "declared_candidate_order": 1,
                "seed_count": 2,
            },
        ]
    )
    units = pd.DataFrame(
        [
            {
                "candidate_id": "stable",
                "selection_unit_id": unit,
                "seed": "deterministic",
                "status": "complete",
                "balanced_accuracy": 0.7,
                "macro_f1": 0.7,
            }
            for unit in ("u1", "u2")
        ]
        + [
            {
                "candidate_id": "fragile",
                "selection_unit_id": unit,
                "seed": seed,
                "status": "fit_failure" if unit == "u2" and seed == 2 else "complete",
                "balanced_accuracy": None if unit == "u2" and seed == 2 else 0.99,
                "macro_f1": None if unit == "u2" and seed == 2 else 0.99,
            }
            for unit in ("u1", "u2")
            for seed in (1, 2)
        ]
    )
    winner, trace = select_lexicographic_candidate(units, candidates)
    assert winner.candidate_id == "stable"
    failed = trace[trace.candidate_id == "fragile"].iloc[0]
    assert not failed.complete_support
    assert failed.failure_rows == 1
    assert "fit_failure" in failed.terminal_statuses
