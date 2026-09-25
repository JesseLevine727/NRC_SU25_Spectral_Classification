"""Deterministic synthetic tests for the P05 development training kernel."""

from __future__ import annotations

import inspect
import io
import math
import time

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import (  # noqa: E402
    p05_core_run,
    p05_development,
)
from atlas_sers.evaluation.p04_runtime import (  # noqa: E402
    _noise_quantiles,
    _state_hash,
)
from atlas_sers.evaluation.p05_development import (  # noqa: E402
    train_development_fit,
)
from atlas_sers.evaluation.p05_sampling import Observation  # noqa: E402
from atlas_sers.models.acquisition import AcquisitionClassifier  # noqa: E402

RECIPES = ("D0-M", "D1", "D2", "D3")
PROJECTION_RECIPES = ("D1", "D3")
EXPECTED_PARAMETERS = {
    "D0-M": 208691,
    "D1": 212851,
    "D2": 208691,
    "D3": 212851,
}
MINIMUM_EPOCHS = 30
MAXIMUM_EPOCHS = 200
BATCHES_PER_EPOCH = 4

NUMERIC_HISTORY_KEYS = (
    "chemical_ce",
    "total_loss",
    "supcon_loss",
    "paired_loss",
    "gradient_norm_mean",
    "gradient_norm_max",
    "head_gradient_norm_mean",
    "backbone_gradient_norm_mean",
    "clipped_fraction",
    "zero_gradient_batches",
    "train_balanced_accuracy",
    "train_nll",
    "train_macro_f1",
    "train_predicted_class_count",
    "validation_balanced_accuracy",
    "validation_nll",
    "validation_macro_f1",
    "validation_predicted_class_count",
    "eligible_anchor_count",
    "zero_positive_anchor_count",
    "paired_master_count",
    "paired_pair_count",
    "epoch_optimizer_steps",
    "total_optimizer_steps",
)


def _normalized_row(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=1401).astype(np.float32)
    low = float(vector.min())
    high = float(vector.max())
    return ((vector - low) / (high - low)).astype(np.float32)


def _role(entries, seed: int, *, station: str = "station-1"):
    ordered = sorted(entries, key=lambda item: item[0])
    rng = np.random.default_rng(seed)
    values = []
    observations = []
    metadata = []
    for uid, label, master, instrument in ordered:
        values.append(_normalized_row(rng))
        observations.append(
            Observation(
                uid=uid,
                master=master,
                station=station,
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


def _small_roles(seed: int = 20260925):
    fit = _role(
        [
            ("fit-A-0-i0", "A", "fit-A-0", "i0"),
            ("fit-A-1-i0", "A", "fit-A-1", "i0"),
            ("fit-B-0-i0", "B", "fit-B-0", "i0"),
            ("fit-C-0-i0", "C", "fit-C-0", "i0"),
        ],
        seed,
    )
    validation = _role(
        [
            ("val-A-0-i0", "A", "val-A-0", "i0"),
            ("val-B-0-i0", "B", "val-B-0", "i0"),
            ("val-C-0-i0", "C", "val-C-0", "i0"),
        ],
        seed + 1,
    )
    return fit, validation


def _dense_roles(seed: int = 20260925):
    fit_entries = []
    validation_entries = []
    for label in ("A", "B", "C"):
        for master in range(2):
            for instrument in ("i0", "i1"):
                fit_entries.append(
                    (
                        f"fit-{label}-{master}-{instrument}",
                        label,
                        f"fit-{label}-{master}",
                        instrument,
                    )
                )
            validation_entries.append(
                (
                    f"val-{label}-{master}-i0",
                    label,
                    f"val-{label}-{master}",
                    "i0",
                )
            )
    return _role(fit_entries, seed), _role(validation_entries, seed + 1)


def _plateau(epoch: int) -> tuple[float, float]:
    return (0.9, 0.1) if epoch == 1 else (0.5, 1.0)


def _improving(epoch: int) -> tuple[float, float]:
    return 0.5 + 0.001 * epoch, 1.0 - 0.001 * epoch


def _early_best(epoch: int) -> tuple[float, float]:
    if epoch <= 4:
        return 0.5 + 0.05 * epoch, 1.0 - 0.05 * epoch
    return 0.5, 1.0


def _install_metrics(monkeypatch, fit_count: int, trajectory) -> None:
    state = {"epoch": 0}

    def fake(model, values, labels, classes, device):
        if values.shape[0] == fit_count:
            return {
                "balanced_accuracy": 0.6,
                "macro_f1": 0.6,
                "negative_log_likelihood": 0.5,
                "predicted_class_count": 3,
            }
        state["epoch"] += 1
        balanced_accuracy, nll = trajectory(state["epoch"])
        return {
            "balanced_accuracy": float(balanced_accuracy),
            "macro_f1": float(balanced_accuracy),
            "negative_log_likelihood": float(nll),
            "predicted_class_count": 3,
        }

    monkeypatch.setattr(p05_development, "_metric_pass", fake)


def _run(fit, validation, recipe, seed, monkeypatch, trajectory=_plateau, **kwargs):
    fit_values, fit_observations, fit_metadata = fit
    validation_values, validation_observations, _ = validation
    _install_metrics(monkeypatch, fit_values.shape[0], trajectory)
    kwargs.setdefault("maximum_fit_seconds", 120.0)
    return train_development_fit(
        values=fit_values,
        observations=fit_observations,
        noise_metadata=fit_metadata,
        validation_values=validation_values,
        validation_observations=validation_observations,
        role_id="role-1",
        recipe=recipe,
        seed=seed,
        device="cpu",
        **kwargs,
    )


def _call(fit, validation, **overrides):
    fit_values, fit_observations, fit_metadata = fit
    validation_values, validation_observations, _ = validation
    arguments = {
        "values": fit_values,
        "observations": fit_observations,
        "noise_metadata": fit_metadata,
        "validation_values": validation_values,
        "validation_observations": validation_observations,
        "role_id": "role-1",
        "recipe": "D0-M",
        "seed": 20260925,
        "device": "cpu",
        "maximum_fit_seconds": 120.0,
    }
    arguments.update(overrides)
    return train_development_fit(**arguments)


def _numeric_history(history):
    return [
        {key: row[key] for key in NUMERIC_HISTORY_KEYS}
        for row in history
    ]


def test_fixture_roles_are_disjoint():
    fit, validation = _small_roles()
    assert not ({row.uid for row in fit[1]} & {row.uid for row in validation[1]})
    assert not ({row.master for row in fit[1]} & {row.master for row in validation[1]})


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_parameter_counts_and_minimum_stopping(recipe, monkeypatch):
    fit, validation = _small_roles()
    result = _run(fit, validation, recipe, 20260925, monkeypatch)
    assert result.status == "complete"
    assert result.reason_code is None
    assert result.parameter_count == EXPECTED_PARAMETERS[recipe]
    assert result.epochs_completed == MINIMUM_EPOCHS
    assert result.optimizer_steps == MINIMUM_EPOCHS * BATCHES_PER_EPOCH
    assert result.best_epoch == 1
    assert result.best_validation_predicted_class_count is not None
    assert len(result.history) == MINIMUM_EPOCHS
    for row in result.history:
        for key in NUMERIC_HISTORY_KEYS:
            assert math.isfinite(float(row[key]))
    if recipe in PROJECTION_RECIPES:
        assert sum(row["head_gradient_norm_mean"] for row in result.history) > 0.0
    else:
        assert all(row["head_gradient_norm_mean"] == 0.0 for row in result.history)


def test_maximum_epochs_and_shared_prefix(monkeypatch):
    fit, validation = _small_roles()
    long_result = _run(
        fit, validation, "D0-M", 20260925, monkeypatch, trajectory=_improving
    )
    assert long_result.status == "complete"
    assert long_result.epochs_completed == MAXIMUM_EPOCHS
    assert long_result.optimizer_steps == MAXIMUM_EPOCHS * BATCHES_PER_EPOCH
    assert long_result.best_epoch == MAXIMUM_EPOCHS

    short_fit, short_validation = _small_roles()
    short_result = _run(
        short_fit, short_validation, "D0-M", 20260925, monkeypatch, trajectory=_plateau
    )
    assert short_result.epochs_completed == MINIMUM_EPOCHS
    shared = MINIMUM_EPOCHS - 1
    assert (
        short_result.history[shared]["sampling_digest"]
        == long_result.history[shared]["sampling_digest"]
    )
    assert (
        short_result.history[shared]["augmentation_digest"]
        == long_result.history[shared]["augmentation_digest"]
    )
    assert (
        short_result.history[shared]["pair_digest"]
        == long_result.history[shared]["pair_digest"]
    )
    assert (
        short_result.history[shared]["sampling_digest"]
        != long_result.history[MAXIMUM_EPOCHS - 1]["sampling_digest"]
    )


def test_patience_tie_and_pre_minimum_best(monkeypatch):
    fit, validation = _small_roles()
    result = _run(
        fit, validation, "D0-M", 20260925, monkeypatch, trajectory=_early_best
    )
    assert result.status == "complete"
    assert result.epochs_completed == MINIMUM_EPOCHS
    assert result.best_epoch == 4
    assert result.best_state_digest != result.terminal_state_digest
    assert _state_hash(result.state_dict) == result.best_state_digest
    assert _state_hash(result.terminal_state_dict) == result.terminal_state_digest


def test_repeatable_histories_and_state(monkeypatch):
    fit, validation = _small_roles()
    first = _run(fit, validation, "D3", 20260925, monkeypatch)
    second = _run(fit, validation, "D3", 20260925, monkeypatch)
    assert first.history == second.history
    assert _state_hash(first.state_dict) == _state_hash(second.state_dict)
    assert first.best_state_digest == second.best_state_digest
    assert first.terminal_state_digest == second.terminal_state_digest
    assert first.sampling_digest == second.sampling_digest
    assert first.augmentation_digest == second.augmentation_digest
    assert first.pair_digest == second.pair_digest


@pytest.mark.parametrize("left,right", [("D0-M", "D2"), ("D1", "D3")])
def test_sparse_auxiliary_equivalence(left, right, monkeypatch):
    fit, validation = _small_roles()
    first = _run(fit, validation, left, 20260925, monkeypatch)
    second = _run(fit, validation, right, 20260925, monkeypatch)
    assert first.status == second.status == "complete"
    assert _state_hash(first.state_dict) == _state_hash(second.state_dict)
    assert _numeric_history(first.history) == _numeric_history(second.history)


def test_sparse_support_accounting(monkeypatch):
    fit, validation = _small_roles()
    result = _run(fit, validation, "D3", 20260925, monkeypatch)
    assert result.status == "complete"
    assert result.supcon_support["enabled"] == 1
    assert result.supcon_support["available_batches"] == MINIMUM_EPOCHS * BATCHES_PER_EPOCH
    assert result.supcon_support["eligible_anchors"] == MINIMUM_EPOCHS * BATCHES_PER_EPOCH * 2
    assert result.supcon_support["zero_positive_anchors"] == MINIMUM_EPOCHS * BATCHES_PER_EPOCH * 2
    assert result.paired_support["enabled"] == 1
    assert result.paired_support["available_batches"] == 0
    assert all(row["paired_available_batches"] == 0 for row in result.history)


def test_dense_paired_support_available(monkeypatch):
    fit, validation = _dense_roles()
    result = _run(fit, validation, "D3", 20260925, monkeypatch)
    assert result.status == "complete"
    assert result.paired_support["enabled"] == 1
    assert result.paired_support["available_batches"] == MINIMUM_EPOCHS * BATCHES_PER_EPOCH
    assert result.paired_support["eligible_masters"] == MINIMUM_EPOCHS * BATCHES_PER_EPOCH * 6
    assert result.paired_support["pairs"] == MINIMUM_EPOCHS * BATCHES_PER_EPOCH * 6
    assert result.history[0]["paired_master_count"] == 24


def test_zero_gradient_batches_are_recorded_not_failed(monkeypatch):
    fit, validation = _small_roles()
    monkeypatch.setattr(
        p05_development,
        "weighted_cross_entropy",
        lambda logits, labels, weights: (logits * 0.0).sum(),
    )
    result = _run(fit, validation, "D0-M", 20260925, monkeypatch)
    assert result.status == "complete"
    assert result.optimizer_steps == MINIMUM_EPOCHS * BATCHES_PER_EPOCH
    assert result.zero_gradient_batches == MINIMUM_EPOCHS * BATCHES_PER_EPOCH
    assert result.nonzero_gradient_elements == 0
    assert all(row["zero_gradient_batches"] == 4 for row in result.history)


def test_nonfinite_parameter_after_update_preserves_terminal_state(monkeypatch):
    fit, validation = _small_roles()
    real_adamw = torch.optim.AdamW

    class Poisoned(real_adamw):
        def step(self, *args, **kwargs):
            output = super().step(*args, **kwargs)
            with torch.no_grad():
                for parameter in self.param_groups[0]["params"]:
                    parameter[0] = float("nan")
                    break
            return output

    monkeypatch.setattr(torch.optim, "AdamW", Poisoned)
    result = _run(fit, validation, "D0-M", 20260925, monkeypatch)
    assert result.status == "numerical_failure"
    assert result.reason_code == "nonfinite_parameter"
    assert result.optimizer_steps == 1
    assert result.state_capture_failed is False
    assert result.terminal_state_dict is not None
    assert result.terminal_state_digest is not None
    assert any(
        bool(torch.isnan(tensor).any()) for tensor in result.terminal_state_dict.values()
    )


def test_finalize_prediction_failure_preserves_terminal_not_best(monkeypatch):
    fit, validation = _small_roles()

    def boom(model, values, device):
        raise RuntimeError("prediction boom")

    monkeypatch.setattr(p05_development, "_predict_logits", boom)
    result = _run(
        fit, validation, "D0-M", 20260925, monkeypatch, trajectory=_early_best
    )
    assert result.status == "fit_failure"
    assert result.reason_code == "finalize_RuntimeError"
    assert result.state_dict is None
    assert result.validation_logits is None
    assert result.validation_uids == ()
    assert result.best_state_dict is not None
    assert result.terminal_state_dict is not None
    assert result.best_state_digest != result.terminal_state_digest


def test_capture_failure_marker_does_not_mask_original(monkeypatch):
    fit, validation = _small_roles()

    def broken_clone(model):
        raise RuntimeError("clone boom")

    monkeypatch.setattr(p05_development, "_clone_state", broken_clone)
    result = _run(fit, validation, "D0-M", 20260925, monkeypatch)
    assert result.status == "fit_failure"
    assert result.reason_code == "setup_RuntimeError"
    assert result.state_capture_failed is True
    assert result.terminal_state_dict is None
    assert result.best_state_dict is None


def test_callback_failure_preserves_history_and_counts(monkeypatch):
    fit, validation = _small_roles()

    def boom(record):
        raise RuntimeError("callback boom")

    result = _run(fit, validation, "D1", 20260925, monkeypatch, on_epoch=boom)
    assert result.status == "fit_failure"
    assert result.reason_code == "callback_RuntimeError"
    assert result.epochs_completed == 1
    assert result.optimizer_steps == BATCHES_PER_EPOCH
    assert result.terminal_state_dict is not None
    assert result.traceback_digest is not None


def test_maximum_fit_seconds_expiry():
    fit, validation = _small_roles()
    result = _call(fit, validation, maximum_fit_seconds=1e-9)
    assert result.status == "resource_failure"
    assert result.reason_code == "deadline_exceeded"
    assert result.history == []
    assert result.state_dict is None


def test_global_deadline_expiry():
    fit, validation = _small_roles()
    result = _call(fit, validation, global_deadline=time.perf_counter() - 1.0)
    assert result.status == "resource_failure"
    assert result.reason_code == "global_deadline_exceeded"
    assert result.history == []
    assert result.state_dict is None


def test_cuda_memory_cap_enforced(monkeypatch):
    fit, validation = _small_roles()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 10**12)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 10**12)
    result = _call(fit, validation, device="cuda")
    assert result.status == "resource_failure"
    assert result.reason_code == "cuda_memory_exceeded"


def test_inputs_not_mutated(monkeypatch):
    fit, validation = _small_roles()
    fit_values, fit_observations, fit_metadata = fit
    values_copy = fit_values.copy()
    metadata_copy = fit_metadata.copy(deep=True)
    observation_snapshot = [
        (row.uid, row.master, row.target, row.instrument, row.station)
        for row in fit_observations
    ]
    result = _run(fit, validation, "D3", 20260925, monkeypatch)
    assert result.status == "complete"
    assert np.array_equal(fit_values, values_copy)
    pd.testing.assert_frame_equal(fit_metadata, metadata_copy)
    assert [
        (row.uid, row.master, row.target, row.instrument, row.station)
        for row in fit_observations
    ] == observation_snapshot


def test_source_only_noise_derivation(monkeypatch):
    fit, validation = _small_roles()
    _, _, fit_metadata = fit
    expected = _noise_quantiles(fit_metadata.reset_index(drop=True))
    result = _run(fit, validation, "D0-M", 20260925, monkeypatch)
    assert result.source_noise_levels == tuple(
        float(value) for value in expected.tolist()
    )
    altered_validation = _role(
        [
            ("val-A-0-i0", "A", "val-A-0", "i0"),
            ("val-B-0-i0", "B", "val-B-0", "i0"),
            ("val-C-0-i0", "C", "val-C-0", "i0"),
        ],
        12345,
    )
    second = _run(fit, altered_validation, "D0-M", 20260925, monkeypatch)
    assert second.source_noise_levels == result.source_noise_levels
    assert second.augmentation_digest == result.augmentation_digest
    assert second.sampling_digest == result.sampling_digest


@pytest.mark.parametrize("recipe,use_projection", [("D0-M", False), ("D1", True)])
def test_best_state_serialization_roundtrip(recipe, use_projection, monkeypatch):
    fit, validation = _small_roles()
    result = _run(fit, validation, recipe, 20260925, monkeypatch)
    assert result.status == "complete"
    buffer = io.BytesIO()
    torch.save(result.state_dict, buffer)
    buffer.seek(0)
    reloaded = torch.load(buffer, weights_only=True)
    assert _state_hash(reloaded) == result.best_state_digest
    model = AcquisitionClassifier(class_count=3, use_projection=use_projection)
    model.load_state_dict(reloaded)
    assert model.batch_normalization_modules() == 0


def test_development_api_surface():
    parameters = set(inspect.signature(train_development_fit).parameters)
    required = {
        "values",
        "observations",
        "noise_metadata",
        "validation_values",
        "validation_observations",
        "role_id",
        "recipe",
        "seed",
        "device",
        "maximum_fit_seconds",
    }
    forbidden = {
        "epochs",
        "batches_per_epoch",
        "patience",
        "minimum_epochs",
        "maximum_epochs",
        "validation_noise_metadata",
        "test_values",
        "test_observations",
        "held_values",
        "calibration_values",
    }
    assert required <= parameters
    assert not (parameters & forbidden)


def test_determinism_configuration_applied(monkeypatch):
    fit, validation = _small_roles()
    _run(fit, validation, "D0-M", 20260925, monkeypatch)
    assert torch.get_num_threads() == 1
    assert torch.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_reject_nonfinite_fitting_values(bad):
    fit, validation = _small_roles()
    values = fit[0].copy()
    values[0, 0] = bad
    with pytest.raises(ValueError):
        _call((values, fit[1], fit[2]), validation)


def test_reject_out_of_range_values():
    fit, validation = _small_roles()
    values = fit[0].copy()
    values[0, :] *= 2.0
    with pytest.raises(ValueError):
        _call((values, fit[1], fit[2]), validation)


def test_reject_misaligned_observations():
    fit, validation = _small_roles()
    with pytest.raises(ValueError):
        _call((fit[0], fit[1][:-1], fit[2]), validation)


def test_reject_unsorted_uids():
    fit, validation = _small_roles()
    with pytest.raises(ValueError):
        _call(
            (
                fit[0],
                list(reversed(fit[1])),
                fit[2].iloc[::-1].reset_index(drop=True),
            ),
            validation,
        )


def test_reject_noise_metadata_alignment():
    fit, validation = _small_roles()
    reversed_frame = fit[2].iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        _call((fit[0], fit[1], reversed_frame), validation)


def test_reject_missing_fitting_class():
    fit, validation = _small_roles()
    keep = [index for index, row in enumerate(fit[1]) if row.target != "C"]
    with pytest.raises(ValueError):
        _call(
            (
                fit[0][keep],
                [fit[1][index] for index in keep],
                fit[2].iloc[keep].reset_index(drop=True),
            ),
            validation,
        )


def test_reject_vocabulary_mismatch():
    fit, _ = _small_roles()
    other = _role(
        [
            ("val-A-0-i0", "A", "val-A-0", "i0"),
            ("val-B-0-i0", "B", "val-B-0", "i0"),
            ("val-D-0-i0", "D", "val-D-0", "i0"),
        ],
        999,
    )
    with pytest.raises(ValueError):
        _call(fit, other)


def test_reject_station_mismatch():
    fit, _ = _small_roles()
    other = _role(
        [
            ("val-A-0-i0", "A", "val-A-0", "i0"),
            ("val-B-0-i0", "B", "val-B-0", "i0"),
            ("val-C-0-i0", "C", "val-C-0", "i0"),
        ],
        999,
        station="station-2",
    )
    with pytest.raises(ValueError):
        _call(fit, other)


def test_reject_uid_overlap():
    fit, _ = _small_roles()
    other = _role(
        [
            ("fit-A-0-i0", "A", "val-A-0", "i0"),
            ("val-B-0-i0", "B", "val-B-0", "i0"),
            ("val-C-0-i0", "C", "val-C-0", "i0"),
        ],
        999,
    )
    with pytest.raises(ValueError):
        _call(fit, other)


def test_reject_master_overlap():
    fit, _ = _small_roles()
    other = _role(
        [
            ("val-A-0-i2", "A", "fit-A-0", "i2"),
            ("val-B-0-i0", "B", "val-B-0", "i0"),
            ("val-C-0-i0", "C", "val-C-0", "i0"),
        ],
        999,
    )
    with pytest.raises(ValueError):
        _call(fit, other)


def _patience_beyond_minimum(epoch: int) -> tuple[float, float]:
    if epoch < 35:
        return 0.5 + 0.005 * epoch, 1.0
    return 0.8, 0.2


def _tie_ba_priority(epoch: int) -> tuple[float, float]:
    if epoch == 1:
        return 0.5, 0.1
    return 0.6, 5.0


def _tie_nll_resolves(epoch: int) -> tuple[float, float]:
    if epoch == 1:
        return 0.6, 1.0
    return 0.6, 0.5


def _tie_exact_retains_earliest(epoch: int) -> tuple[float, float]:
    return 0.6, 0.5


def test_patience_beyond_minimum_stops_at_tie_twenty(monkeypatch):
    fit, validation = _small_roles()
    result = _run(
        fit,
        validation,
        "D0-M",
        20260925,
        monkeypatch,
        trajectory=_patience_beyond_minimum,
    )
    assert result.status == "complete"
    assert result.best_epoch == 35
    assert result.epochs_completed == 55
    assert result.optimizer_steps == 55 * BATCHES_PER_EPOCH
    assert result.history[34]["improved"] is True
    assert result.history[34]["nonimproving_epochs"] == 0
    assert result.history[54]["nonimproving_epochs"] == 20
    assert all(result.history[index]["improved"] is False for index in range(35, 55))


@pytest.mark.parametrize(
    "trajectory,expected_best_epoch",
    [
        (_tie_ba_priority, 2),
        (_tie_nll_resolves, 2),
        (_tie_exact_retains_earliest, 1),
    ],
)
def test_epoch_tie_ordering(trajectory, expected_best_epoch, monkeypatch):
    fit, validation = _small_roles()
    result = _run(
        fit, validation, "D0-M", 20260925, monkeypatch, trajectory=trajectory
    )
    assert result.status == "complete"
    assert result.best_epoch == expected_best_epoch
    assert result.epochs_completed == MINIMUM_EPOCHS
    best_row = result.history[expected_best_epoch - 1]
    assert best_row["improved"] is True
    expected_ba, expected_nll = trajectory(expected_best_epoch)
    assert best_row["validation_balanced_accuracy"] == expected_ba
    assert best_row["validation_nll"] == expected_nll


def test_filesystem_checkpoint_roundtrip_best_and_terminal(tmp_path, monkeypatch):
    fit, validation = _small_roles()
    result = _run(
        fit, validation, "D1", 20260925, monkeypatch, trajectory=_early_best
    )
    assert result.status == "complete"
    assert result.best_epoch == 4
    assert result.best_state_dict is not None
    assert result.terminal_state_dict is not None
    assert result.best_state_digest != result.terminal_state_digest

    best_path = tmp_path / "best.pt"
    terminal_path = tmp_path / "terminal.pt"
    p05_core_run._save_state(torch, result.best_state_dict, best_path)
    p05_core_run._save_state(torch, result.terminal_state_dict, terminal_path)
    assert best_path.is_file()
    assert terminal_path.is_file()

    best_loaded = torch.load(best_path, weights_only=True, map_location="cpu")
    terminal_loaded = torch.load(terminal_path, weights_only=True, map_location="cpu")
    assert _state_hash(best_loaded["state_dict"]) == result.best_state_digest
    assert _state_hash(terminal_loaded["state_dict"]) == result.terminal_state_digest
    assert _state_hash(best_loaded["state_dict"]) != _state_hash(
        terminal_loaded["state_dict"]
    )
    assert _state_hash(best_loaded["state_dict"]) == _state_hash(result.state_dict)

    model = AcquisitionClassifier(class_count=3, use_projection=True)
    model.load_state_dict(best_loaded["state_dict"])
    assert model.batch_normalization_modules() == 0
