from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation.p04_runtime import _augment, select_candidate  # noqa: E402
from atlas_sers.models.deep import (  # noqa: E402
    CompactSERSClassifier,
    DeterministicAdaptiveAveragePool1d,
    architecture_audit,
)


def test_locked_architecture_shape_budget_and_normalization() -> None:
    audit = architecture_audit(class_count=3, batch_size=2)

    assert audit.input_shape == (2, 1, 1401)
    assert audit.transition_1_shape == (2, 48, 701)
    assert audit.transition_2_shape == (2, 64, 351)
    assert audit.pooled_shape == (2, 64, 16)
    assert audit.embedding_shape == (2, 64)
    assert audit.logits_shape == (2, 3)
    assert audit.trainable_parameters == 208691
    assert audit.batch_normalization_modules == 0
    assert not any(
        "BatchNorm" in type(module).__name__
        for module in CompactSERSClassifier(3).modules()
    )


def test_deterministic_pool_matches_adaptive_average_forward() -> None:
    values = torch.arange(2 * 5 * 351, dtype=torch.float32).reshape(2, 5, 351)
    observed = DeterministicAdaptiveAveragePool1d(16)(values)
    expected = torch.nn.functional.adaptive_avg_pool1d(values, 16)

    assert torch.allclose(observed, expected)


def test_training_augmentation_is_replayable_finite_and_row_scaled() -> None:
    values = np.vstack([np.linspace(0, 1, 1401), np.linspace(1, 0, 1401)]).astype(
        np.float32
    )
    uids = np.asarray(["a", "b"])
    levels = np.asarray([0.0, 0.001, 0.002, 0.003])
    first_digest = hashlib.sha256()
    second_digest = hashlib.sha256()
    first = _augment(
        values,
        uids,
        rng=np.random.default_rng(17),
        noise_levels=levels,
        digest=first_digest,
    )
    second = _augment(
        values,
        uids,
        rng=np.random.default_rng(17),
        noise_levels=levels,
        digest=second_digest,
    )

    assert np.array_equal(first, second)
    assert first_digest.hexdigest() == second_digest.hexdigest()
    assert np.isfinite(first).all()
    assert np.allclose(first.min(axis=1), 0)
    assert np.allclose(first.max(axis=1), 1)


def test_candidate_selection_requires_all_units_and_seeds() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "hyperparameter_sha256": "a" * 64,
                "complexity_rank": 0,
                "declared_candidate_order": 0,
            },
            {
                "candidate_id": "b",
                "learning_rate": 0.0003,
                "weight_decay": 0.001,
                "hyperparameter_sha256": "b" * 64,
                "complexity_rank": 0,
                "declared_candidate_order": 1,
            },
        ]
    )
    records = []
    for candidate, score in (("a", 0.7), ("b", 0.8)):
        for unit in ("u0", "u1"):
            for seed in (1, 2, 3):
                records.append(
                    {
                        "candidate_id": candidate,
                        "selection_unit_id": unit,
                        "seed": seed,
                        "status": "complete",
                        "best_validation_balanced_accuracy": score,
                        "best_validation_macro_f1": score - 0.01,
                    }
                )
    records[-1]["status"] = "fit_failure"
    winner, trace = select_candidate(pd.DataFrame(records), candidates, 2)

    assert winner is not None
    assert winner.candidate_id == "a"
    assert not bool(trace.loc[trace.candidate_id.eq("b"), "complete_support"].iloc[0])
