"""Deterministic synthetic tests for the P05 bounded smoke kernel."""

from __future__ import annotations

import inspect
import math
import time

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import p05_smoke  # noqa: E402
from atlas_sers.evaluation.p05_sampling import Observation  # noqa: E402
from atlas_sers.evaluation.p05_smoke import train_smoke_fit  # noqa: E402

RECIPES = ("D0-M", "D1", "D2", "D3")
NUMERIC_KEYS = (
    "chemical_ce",
    "total_loss",
    "supcon_loss",
    "paired_loss",
    "gradient_norm_mean",
    "gradient_norm_max",
    "head_gradient_norm_mean",
    "backbone_gradient_norm_mean",
    "clipped_fraction",
    "embedding_variance",
    "embedding_norm_mean",
    "train_ba",
    "train_nll",
    "train_predicted_class_count",
    "optimizer_steps",
    "eligible_anchor_count",
    "zero_positive_anchor_count",
    "paired_master_count",
)


def _normalized_row(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=1401).astype(np.float32)
    lo = float(vec.min())
    hi = float(vec.max())
    return ((vec - lo) / (hi - lo)).astype(np.float32)


def _role(specs, seed: int):
    rng = np.random.default_rng(seed)
    values = []
    observations = []
    metadata = []
    for index, (label, master, instrument) in enumerate(specs):
        uid = f"uid-{index:03d}"
        values.append(_normalized_row(rng))
        observations.append(
            Observation(
                uid=uid,
                master=master,
                station="station-1",
                target=label,
                instrument=instrument,
                substrate="na",
            )
        )
        metadata.append(
            {
                "observation_uid": uid,
                "first_difference_noise_mad": 0.01,
                "intensity_range": 1.0,
            }
        )
    return (
        np.asarray(values, dtype=np.float32),
        observations,
        pd.DataFrame(metadata),
    )


def _dense():
    specs = []
    for label in ("A", "B", "C"):
        for master_index in range(2):
            for instrument in ("i0", "i1"):
                specs.append((label, f"dense-{label}-{master_index}", instrument))
    return _role(specs, 20260805)


def _sparse():
    specs = [
        ("A", "sparse-A-0", "i0"),
        ("A", "sparse-A-1", "i0"),
        ("B", "sparse-B-0", "i0"),
        ("C", "sparse-C-0", "i0"),
    ]
    return _role(specs, 20260805)


def _fit(values, observations, metadata, recipe, seed, **kwargs):
    kwargs.setdefault("epochs", 2)
    kwargs.setdefault("batches_per_epoch", 2)
    return train_smoke_fit(
        values=values,
        observations=observations,
        noise_metadata=metadata,
        role_id="role-1",
        recipe=recipe,
        seed=seed,
        device="cpu",
        **kwargs,
    )


def _numeric(history):
    return [{key: row[key] for key in NUMERIC_KEYS} for row in history]


@pytest.mark.parametrize(
    "recipe,expected",
    [("D0-M", 208691), ("D1", 212851), ("D2", 208691), ("D3", 212851)],
)
def test_dense_fit_finite_parameter_counts(recipe, expected):
    values, observations, metadata = _dense()
    result = _fit(values, observations, metadata, recipe, 20260805)
    assert result.status == "complete"
    assert result.reason_code is None
    assert result.parameter_count == expected
    assert result.optimizer_steps == 4
    assert len(result.history) == 2
    for row in result.history:
        for key in NUMERIC_KEYS:
            assert math.isfinite(float(row[key]))
    head_mean = sum(row["head_gradient_norm_mean"] for row in result.history)
    if expected == 212851:
        assert head_mean > 0.0
    else:
        assert head_mean == 0.0
    assert result.finite_gradient_batches == 4
    assert result.nonzero_gradient_elements > 0


def test_common_initial_backbone_across_recipes():
    values, observations, metadata = _dense()
    digests = {
        recipe: _fit(values, observations, metadata, recipe, 20260805).initial_backbone_digest
        for recipe in RECIPES
    }
    assert len(set(digests.values())) == 1
    assert all(digest is not None for digest in digests.values())


@pytest.mark.parametrize("left,right", [("D0-M", "D2"), ("D1", "D3")])
def test_sparse_auxiliary_reduction_is_exact(left, right):
    values, observations, metadata = _sparse()
    first = _fit(values, observations, metadata, left, 20260805)
    second = _fit(values, observations, metadata, right, 20260805)
    assert first.status == second.status == "complete"
    assert first.final_state_digest == second.final_state_digest
    assert _numeric(first.history) == _numeric(second.history)


def test_sparse_supcon_support_counts():
    values, observations, metadata = _sparse()
    result = _fit(values, observations, metadata, "D3", 20260805)
    assert result.status == "complete"
    assert result.supcon_support["enabled"] == 1
    assert result.supcon_support["available_batches"] == 4
    assert result.supcon_support["zero_positive_anchors"] >= 2
    assert result.paired_support["enabled"] == 1
    assert result.paired_support["available_batches"] == 0
    assert result.history[0]["paired_available_batches"] == 0


def test_sparse_replay_is_exact():
    values, observations, metadata = _sparse()
    first = _fit(values, observations, metadata, "D3", 20260805)
    second = _fit(values, observations, metadata, "D3", 20260805)
    assert first.final_state_digest == second.final_state_digest
    assert first.history == second.history
    assert first.sampling_digest == second.sampling_digest
    assert first.augmentation_digest == second.augmentation_digest
    assert first.pair_digest == second.pair_digest


def test_changed_seed_changes_streams_and_state():
    values, observations, metadata = _sparse()
    first = _fit(values, observations, metadata, "D3", 20260805)
    second = _fit(values, observations, metadata, "D3", 20260817)
    assert first.final_state_digest != second.final_state_digest
    assert first.sampling_digest != second.sampling_digest


def test_callback_receives_epoch_copies():
    values, observations, metadata = _dense()
    seen = []
    result = _fit(values, observations, metadata, "D1", 20260805, on_epoch=seen.append)
    assert result.status == "complete"
    assert len(seen) == len(result.history) == 2
    assert seen == result.history
    assert {"epoch", "chemical_ce", "total_loss", "train_ba"} <= set(seen[0])


def test_reject_nonfinite_values():
    values, observations, metadata = _dense()
    for bad in (np.nan, np.inf):
        corrupted = values.copy()
        corrupted[0, 0] = bad
        with pytest.raises(ValueError):
            _fit(corrupted, observations, metadata, "D0-M", 20260805)


def test_reject_infinite_deadline():
    values, observations, metadata = _dense()
    with pytest.raises(ValueError):
        _fit(values, observations, metadata, "D0-M", 20260805, global_deadline=float("inf"))


def test_reject_unsorted_uids():
    values, observations, metadata = _dense()
    reversed_rows = list(reversed(observations))
    with pytest.raises(ValueError):
        _fit(values, reversed_rows, metadata.iloc[::-1].reset_index(drop=True), "D0-M", 20260805)


def test_reject_missing_classes():
    values, observations, metadata = _dense()
    keep = [index for index, row in enumerate(observations) if row.target != "C"]
    with pytest.raises(ValueError):
        _fit(
            values[keep],
            [observations[index] for index in keep],
            metadata.iloc[keep].reset_index(drop=True),
            "D0-M",
            20260805,
        )


def test_reject_epoch_and_batch_bounds():
    values, observations, metadata = _dense()
    with pytest.raises(ValueError):
        _fit(values, observations, metadata, "D0-M", 20260805, epochs=9)
    with pytest.raises(ValueError):
        _fit(values, observations, metadata, "D0-M", 20260805, batches_per_epoch=5)


def test_reject_padded_role_id():
    values, observations, metadata = _dense()
    with pytest.raises(ValueError):
        train_smoke_fit(
            values=values,
            observations=observations,
            noise_metadata=metadata,
            role_id=" role ",
            recipe="D0-M",
            seed=20260805,
            device="cpu",
        )


def test_reject_cuda_budget_above_ceiling():
    values, observations, metadata = _dense()
    with pytest.raises(ValueError):
        _fit(
            values,
            observations,
            metadata,
            "D0-M",
            20260805,
            maximum_cuda_allocated_bytes=p05_smoke.DEFAULT_CUDA_BYTES + 1,
        )


def test_callback_failure_retains_history_and_state():
    values, observations, metadata = _dense()

    def boom(_record):
        raise RuntimeError("kaboom")

    result = _fit(values, observations, metadata, "D1", 20260805, on_epoch=boom)
    assert result.status == "fit_failure"
    assert result.reason_code is not None and result.reason_code.startswith("fit_")
    assert len(result.history) == 1
    assert result.optimizer_steps == 2
    assert result.state_dict is not None
    assert result.final_state_digest is not None
    assert result.traceback_digest is not None


def test_global_deadline_expired_before_model_allocation():
    values, observations, metadata = _dense()
    result = _fit(
        values,
        observations,
        metadata,
        "D0-M",
        20260805,
        global_deadline=time.perf_counter() - 1.0,
    )
    assert result.status == "resource_failure"
    assert result.reason_code == "global_deadline_exceeded"
    assert result.history == []
    assert result.state_dict is None


def test_cuda_unavailable_is_resource_failure(monkeypatch):
    values, observations, metadata = _dense()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = train_smoke_fit(
        values=values,
        observations=observations,
        noise_metadata=metadata,
        role_id="role-1",
        recipe="D0-M",
        seed=20260805,
        device="cuda",
        epochs=2,
        batches_per_epoch=2,
    )
    assert result.status == "resource_failure"
    assert result.reason_code == "cuda_unavailable"


def test_cuda_peak_memory_cap_enforced(monkeypatch):
    values, observations, metadata = _dense()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 10**12)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 10**12)
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *a, **k: self)
    result = train_smoke_fit(
        values=values,
        observations=observations,
        noise_metadata=metadata,
        role_id="role-1",
        recipe="D0-M",
        seed=20260805,
        device="cuda",
        epochs=2,
        batches_per_epoch=2,
    )
    assert result.status == "resource_failure"
    assert result.reason_code == "cuda_memory_exceeded"


def test_determinism_configuration_applied():
    values, observations, metadata = _sparse()
    _fit(values, observations, metadata, "D0-M", 20260805)
    assert torch.get_num_threads() == 1
    assert torch.are_deterministic_algorithms_enabled()


def test_api_exposes_no_held_data_arguments():
    parameters = set(inspect.signature(train_smoke_fit).parameters)
    forbidden = {
        "validation_values",
        "validation_metadata",
        "test_values",
        "test_metadata",
        "held_values",
        "held_metadata",
    }
    assert not (parameters & forbidden)
