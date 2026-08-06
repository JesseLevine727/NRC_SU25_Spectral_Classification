from __future__ import annotations

import numpy as np
import pytest

from atlas_sers.governance.canonical import canonical_json_bytes, sha256_value
from atlas_sers.governance.runs import RunIdentity, deterministic_run_id


def identity() -> RunIdentity:
    return RunIdentity(
        protocol_version="atlas-sers-research-master-plan-v1",
        experiment_id="EXP-C09-T3",
        task_id="T3-ZS",
        information_regime="zero-shot source-only",
        outer_repeat=0,
        outer_fold=0,
        held_domain="pills:instrument-1",
        population_id="primary",
        representation_id="R_MIN_400_1800",
        model_id="C-SELECTED",
        hyperparameter_sha256="1" * 64,
        seed=20260805,
        code_sha256="2" * 64,
        config_sha256="3" * 64,
        input_sha256="4" * 64,
    )


def test_canonical_json_is_order_independent() -> None:
    left = {"b": [2, 1], "a": {"z": True, "y": None}}
    right = {"a": {"y": None, "z": True}, "b": [2, 1]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_value(left) == sha256_value(right)


def test_canonical_json_normalizes_numpy_scalars() -> None:
    value = {
        "boolean": np.bool_(True),
        "float": np.float64(1.25),
        "integer": np.int64(7),
    }
    assert canonical_json_bytes(value) == b'{"boolean":true,"float":1.25,"integer":7}'


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_version", "atlas-sers-research-master-plan-v2"),
        ("experiment_id", "EXP-C10-T3"),
        ("task_id", "T1-CWA"),
        ("information_regime", "closed set"),
        ("outer_repeat", 1),
        ("outer_fold", 1),
        ("held_domain", "pills:instrument-2"),
        ("population_id", "notes-clear"),
        ("representation_id", "R_SG_400_1800"),
        ("model_id", "C-RBF-SVM"),
        ("hyperparameter_sha256", "5" * 64),
        ("seed", 20260817),
        ("code_sha256", "6" * 64),
        ("config_sha256", "7" * 64),
        ("input_sha256", "8" * 64),
    ],
)
def test_every_scientific_identity_field_changes_run_id(field: str, value: object) -> None:
    baseline = identity()
    assert deterministic_run_id(baseline) != deterministic_run_id(baseline.changed(field, value))


def test_identical_state_reproduces_run_id() -> None:
    assert deterministic_run_id(identity()) == deterministic_run_id(identity())


def test_invalid_hash_is_rejected() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        deterministic_run_id(identity().changed("code_sha256", "invalid"))
