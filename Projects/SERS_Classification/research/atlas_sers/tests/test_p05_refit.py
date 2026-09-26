"""Deterministic synthetic tests for the P05 source-only refit kernel."""

from __future__ import annotations

import inspect
import time

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import p05_development  # noqa: E402
from atlas_sers.evaluation.p04_runtime import _state_hash  # noqa: E402
from atlas_sers.evaluation.p05_development import train_development_fit  # noqa: E402
from atlas_sers.evaluation.p05_refit import (  # noqa: E402
    MAXIMUM_REFIT_EPOCHS,
    MINIMUM_REFIT_EPOCHS,
    train_refit,
)
from atlas_sers.evaluation.p05_sampling import Observation  # noqa: E402
from atlas_sers.evaluation.p05_smoke import DEFAULT_CUDA_BYTES  # noqa: E402
from atlas_sers.models.acquisition import AcquisitionClassifier  # noqa: E402

RECIPES = ("D0-M", "D1", "D2", "D3")
PROJECTION_RECIPES = ("D1", "D3")
BATCHES_PER_EPOCH = 4
REFIT_EPOCHS = 30
EXPECTED_PARAMETERS = {"D0-M": 208691, "D1": 212851, "D2": 208691, "D3": 212851}

COMMON_HISTORY_KEYS = (
    "epoch",
    "chemical_ce",
    "total_loss",
    "supcon_loss",
    "paired_loss",
    "supcon_enabled",
    "paired_enabled",
    "supcon_available_batches",
    "paired_available_batches",
    "eligible_anchor_count",
    "zero_positive_anchor_count",
    "paired_master_count",
    "paired_pair_count",
    "gradient_norm_mean",
    "gradient_norm_max",
    "head_gradient_norm_mean",
    "backbone_gradient_norm_mean",
    "clipped_fraction",
    "zero_gradient_batches",
    "epoch_optimizer_steps",
    "total_optimizer_steps",
    "sampling_digest",
    "augmentation_digest",
    "pair_digest",
)
NUMERIC_HISTORY_KEYS = tuple(
    key for key in COMMON_HISTORY_KEYS if key not in {"epoch", "supcon_enabled", "paired_enabled"}
)


def _normalized_row(rng):
    vector = rng.normal(size=1401).astype(np.float32)
    low = float(vector.min())
    high = float(vector.max())
    return ((vector - low) / (high - low)).astype(np.float32)


def _role(entries, seed, *, station="station-1"):
    rng = np.random.default_rng(seed)
    values, observations, metadata = [], [], []
    for uid, label, master, instrument in sorted(entries, key=lambda item: item[0]):
        values.append(_normalized_row(rng))
        observations.append(Observation(uid, master, station, label, instrument, "na"))
        metadata.append(
            {
                "observation_uid": uid,
                "first_difference_noise_mad": 0.01,
                "intensity_range": 1.0,
            }
        )
    return np.asarray(values, dtype=np.float32), observations, pd.DataFrame(metadata)


def _sparse_role(seed=20260925):
    return _role(
        [
            ("fit-A-0-i0", "A", "fit-A-0", "i0"),
            ("fit-A-1-i0", "A", "fit-A-1", "i0"),
            ("fit-B-0-i0", "B", "fit-B-0", "i0"),
            ("fit-C-0-i0", "C", "fit-C-0", "i0"),
        ],
        seed,
    )


def _sparse_validation(seed=20260926):
    return _role(
        [
            ("val-A-0-i0", "A", "val-A-0", "i0"),
            ("val-B-0-i0", "B", "val-B-0", "i0"),
            ("val-C-0-i0", "C", "val-C-0", "i0"),
        ],
        seed,
    )


def _dense_role(seed=20260925):
    entries = [
        (f"fit-{label}-{master}-{instrument}", label, f"fit-{label}-{master}", instrument)
        for label in ("A", "B", "C")
        for master in range(2)
        for instrument in ("i0", "i1")
    ]
    return _role(entries, seed)


def _dense_validation(seed=20260926):
    entries = [
        (f"val-{label}-{master}-i0", label, f"val-{label}-{master}", "i0")
        for label in ("A", "B", "C")
        for master in range(2)
    ]
    return _role(entries, seed)


def _refit(fit, recipe, *, seed=20260925, epochs=REFIT_EPOCHS, **overrides):
    values, observations, metadata = fit
    arguments = {
        "values": values,
        "observations": observations,
        "noise_metadata": metadata,
        "role_id": "role-1",
        "recipe": recipe,
        "seed": seed,
        "epochs": epochs,
        "device": "cpu",
    }
    arguments.update(overrides)
    return train_refit(**arguments)


def _install_development_stop(monkeypatch):
    monkeypatch.setattr(p05_development, "MAXIMUM_EPOCHS", REFIT_EPOCHS)


def _development_run(fit, validation, recipe, *, seed=20260925, device="cpu"):
    values, observations, metadata = fit
    validation_values, validation_observations, _ = validation
    return train_development_fit(
        values=values,
        observations=observations,
        noise_metadata=metadata,
        validation_values=validation_values,
        validation_observations=validation_observations,
        role_id="role-1",
        recipe=recipe,
        seed=seed,
        device=device,
        maximum_fit_seconds=120.0,
    )


@pytest.mark.parametrize("recipe", RECIPES)
def test_refit_matches_development_at_thirty_epochs(recipe, monkeypatch):
    fit, validation = _dense_role(), _dense_validation()
    _install_development_stop(monkeypatch)
    development = _development_run(fit, validation, recipe)
    refit = _refit(fit, recipe)
    assert refit.status == development.status == "complete"
    assert refit.epochs_completed == development.epochs_completed == REFIT_EPOCHS
    assert refit.optimizer_steps == REFIT_EPOCHS * BATCHES_PER_EPOCH
    assert refit.parameter_count == EXPECTED_PARAMETERS[recipe]
    assert _state_hash(refit.terminal_state_dict) == _state_hash(development.terminal_state_dict)
    for name, tensor in refit.terminal_state_dict.items():
        assert torch.equal(tensor, development.terminal_state_dict[name])
    left = [{key: row[key] for key in COMMON_HISTORY_KEYS} for row in refit.history]
    right = [{key: row[key] for key in COMMON_HISTORY_KEYS} for row in development.history]
    assert left == right
    assert refit.supcon_support == development.supcon_support
    assert refit.paired_support == development.paired_support
    assert refit.sampling_digest == development.sampling_digest
    assert refit.augmentation_digest == development.augmentation_digest
    assert refit.pair_digest == development.pair_digest
    if recipe in PROJECTION_RECIPES:
        assert refit.terminal_head_digest == development.terminal_head_digest is not None
    else:
        assert refit.terminal_head_digest is None


@pytest.mark.parametrize("left,right", [("D0-M", "D2"), ("D1", "D3")])
def test_pairfree_auxiliary_equivalence(left, right):
    fit = _sparse_role()
    first = _refit(fit, left)
    second = _refit(fit, right)
    assert first.status == second.status == "complete"
    assert first.paired_support["available_batches"] == 0
    assert second.paired_support["available_batches"] == 0
    assert _state_hash(first.terminal_state_dict) == _state_hash(second.terminal_state_dict)
    first_rows = [{key: row[key] for key in NUMERIC_HISTORY_KEYS} for row in first.history]
    second_rows = [{key: row[key] for key in NUMERIC_HISTORY_KEYS} for row in second.history]
    assert first_rows == second_rows


def test_refit_api_surface_has_no_validation_inputs():
    parameters = set(inspect.signature(train_refit).parameters)
    required = {
        "values",
        "observations",
        "noise_metadata",
        "role_id",
        "recipe",
        "seed",
        "epochs",
        "device",
        "maximum_fit_seconds",
    }
    forbidden = {
        "validation_values",
        "validation_observations",
        "validation_noise_metadata",
        "test_values",
        "test_observations",
        "held_values",
        "calibration_values",
        "patience",
        "minimum_epochs",
        "maximum_epochs",
        "batches_per_epoch",
    }
    assert required <= parameters
    assert not (parameters & forbidden)


def test_epoch_bounds_acceptance_without_full_training():
    fit = _sparse_role()
    assert MINIMUM_REFIT_EPOCHS == 30
    assert MAXIMUM_REFIT_EPOCHS == 200
    for epochs in (MINIMUM_REFIT_EPOCHS, MAXIMUM_REFIT_EPOCHS):
        result = _refit(fit, "D0-M", epochs=epochs, maximum_fit_seconds=1e-9)
        assert result.epochs == epochs
        assert result.status == "resource_failure"
        assert result.reason_code == "deadline_exceeded"
        assert result.optimizer_steps == 0


def test_reject_invalid_roles():
    values, observations, metadata = _sparse_role()
    nonfinite = values.copy()
    nonfinite[0, 0] = float("nan")
    out_of_range = values.copy()
    out_of_range[0, :] *= 2.0
    keep = [index for index, row in enumerate(observations) if row.target != "C"]
    candidates = [
        (nonfinite, observations, metadata),
        (out_of_range, observations, metadata),
        (values, observations[:-1], metadata),
        (values, list(reversed(observations)), metadata.iloc[::-1].reset_index(drop=True)),
        (values, observations, metadata.iloc[::-1].reset_index(drop=True)),
        (
            values[keep],
            [observations[index] for index in keep],
            metadata.iloc[keep].reset_index(drop=True),
        ),
    ]
    for candidate in candidates:
        with pytest.raises(ValueError):
            _refit(candidate, "D0-M")


def test_reject_invalid_parameters():
    fit = _sparse_role()
    for bad in (29, 0, -1, 201):
        with pytest.raises(ValueError):
            _refit(fit, "D0-M", epochs=bad)
    for bad in (True, 30.0, "30", None):
        with pytest.raises(TypeError):
            _refit(fit, "D0-M", epochs=bad)
    with pytest.raises(TypeError):
        _refit(fit, "D0-M", seed=True)
    with pytest.raises(ValueError):
        _refit(fit, "D0-M", maximum_cuda_allocated_bytes=0)
    with pytest.raises(ValueError):
        _refit(fit, "D0-M", maximum_cuda_allocated_bytes=DEFAULT_CUDA_BYTES + 1)
    with pytest.raises(TypeError):
        _refit(fit, "D0-M", maximum_cuda_allocated_bytes=True)
    for bad in (0.0, -1.0, 120.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            _refit(fit, "D0-M", maximum_fit_seconds=bad)
    with pytest.raises(TypeError):
        _refit(fit, "D0-M", maximum_fit_seconds="120")
    with pytest.raises(ValueError):
        _refit(fit, "D0-M", global_deadline=float("inf"))


def test_expired_budgets_yield_zero_steps():
    fit = _sparse_role()
    expired = _refit(fit, "D0-M", maximum_fit_seconds=1e-9)
    assert expired.status == "resource_failure"
    assert expired.reason_code == "deadline_exceeded"
    assert expired.optimizer_steps == 0
    assert expired.history == []
    assert expired.terminal_state_dict is None
    global_expired = _refit(fit, "D0-M", global_deadline=time.perf_counter() - 1.0)
    assert global_expired.status == "resource_failure"
    assert global_expired.reason_code == "global_deadline_exceeded"
    assert global_expired.optimizer_steps == 0
    assert global_expired.history == []


def test_callback_failure_preserves_partial_updates():
    fit = _sparse_role()

    def boom(record):
        raise RuntimeError("callback boom")

    result = _refit(fit, "D1", on_epoch=boom)
    assert result.status == "fit_failure"
    assert result.reason_code == "callback_RuntimeError"
    assert result.epochs_completed == 1
    assert result.optimizer_steps == BATCHES_PER_EPOCH
    assert result.terminal_state_dict is not None
    assert result.terminal_state_digest is not None
    assert result.traceback_digest is not None


def test_terminal_checkpoint_roundtrip(tmp_path):
    fit = _sparse_role()
    result = _refit(fit, "D1")
    assert result.status == "complete"
    assert result.terminal_state_dict is not None
    path = tmp_path / "refit_terminal.pt"
    torch.save(result.terminal_state_dict, path)
    reloaded = torch.load(path, weights_only=True, map_location="cpu")
    assert _state_hash(reloaded) == result.terminal_state_digest
    model = AcquisitionClassifier(class_count=3, use_projection=True)
    model.load_state_dict(reloaded)
    assert model.batch_normalization_modules() == 0


def test_initial_backbone_shared_across_recipes():
    fit = _sparse_role()

    def stop(record):
        raise RuntimeError("stop after first epoch")

    results = {recipe: _refit(fit, recipe, on_epoch=stop) for recipe in RECIPES}
    for result in results.values():
        assert result.initial_state_digest is not None
        assert result.initial_backbone_digest is not None
    assert len({result.initial_backbone_digest for result in results.values()}) == 1
    assert results["D0-M"].initial_head_digest is None
    assert results["D2"].initial_head_digest is None
    for recipe in PROJECTION_RECIPES:
        assert results[recipe].initial_head_digest is not None
    assert results["D1"].initial_head_digest == results["D3"].initial_head_digest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
@pytest.mark.parametrize("recipe", RECIPES)
def test_cuda_development_refit_terminal_parity(monkeypatch, recipe):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    fit = _sparse_role()
    _install_development_stop(monkeypatch)
    cpu = _development_run(fit, _sparse_validation(), recipe, device="cuda")
    cuda = _refit(fit, recipe, device="cuda")
    assert cpu.status == cuda.status == "complete"
    assert cpu.optimizer_steps == cuda.optimizer_steps == REFIT_EPOCHS * BATCHES_PER_EPOCH
    assert cpu.sampling_digest == cuda.sampling_digest
    assert cpu.augmentation_digest == cuda.augmentation_digest
    assert cpu.pair_digest == cuda.pair_digest
    assert set(cpu.terminal_state_dict) == set(cuda.terminal_state_dict)
    for name, tensor in cpu.terminal_state_dict.items():
        assert torch.equal(tensor, cuda.terminal_state_dict[name])
