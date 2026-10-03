"""Bounded tests for strict P08 training-record consistency verification."""

from __future__ import annotations

import copy
import hashlib
import io
import json

import numpy as np
import pytest
import torch

from atlas_sers.evaluation import p04_runtime, p05_development, p05_pilot
from atlas_sers.evaluation import p08_training_record as module
from atlas_sers.evaluation.p04_runtime import _state_hash
from atlas_sers.evaluation.p05_development import DevelopmentFitResult
from atlas_sers.evaluation.p08_plan import SEEDS
from atlas_sers.evaluation.p08_source_artifacts import (
    EXPECTED_PARAMETER_COUNTS,
    PROJECTION_RECIPES,
    verify_neural_checkpoint_bytes,
)
from atlas_sers.evaluation.p08_source_predictions import verify_source_prediction_values
from atlas_sers.evaluation.p08_training_record import (
    HISTORY_FIELDS,
    RECORD_FIELDS,
    SCHEMA_VERSION,
    TrainingRecordError,
    require_scientific_execution,
    verify_neural_training_record,
)
from atlas_sers.governance.canonical import sha256_value
from atlas_sers.models.acquisition import AcquisitionClassifier
from tests.test_p08_source_predictions import valid_kwargs

FLAGS = {"D0-M": (False, False), "D1": (True, False), "D2": (False, True), "D3": (True, True)}
THREE_CLASSES = ("class-x", "class-y", "class-z")


def _hex(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _derive(val_ba, nll):
    improved = []
    best_epochs = []
    nonimproving = []
    best_key = None
    best = None
    count = 0
    for index in range(len(val_ba)):
        key = (-val_ba[index], nll[index])
        if best_key is None or key < best_key:
            best_key = key
            best = index + 1
            count = 0
            improved.append(True)
        else:
            count += 1
            improved.append(False)
        best_epochs.append(best)
        nonimproving.append(count)
    return improved, best_epochs, nonimproving


def build_history(val_ba, nll, *, recipe="D0-M", predicted=1, zero=None):
    length = len(val_ba)
    if zero is None:
        zero = [0] * length
    improved, best_epochs, nonimproving = _derive(val_ba, nll)
    supcon, paired = FLAGS[recipe]
    entries = []
    for index in range(length):
        epoch = index + 1
        entries.append(
            {
                "epoch": epoch,
                "epoch_optimizer_steps": 4,
                "total_optimizer_steps": 4 * epoch,
                "zero_gradient_batches": zero[index],
                "train_balanced_accuracy": 0.5,
                "validation_balanced_accuracy": val_ba[index],
                "validation_nll": nll[index],
                "validation_macro_f1": 0.5,
                "validation_predicted_class_count": predicted,
                "improved": improved[index],
                "best_epoch": best_epochs[index],
                "nonimproving_epochs": nonimproving[index],
                "sampling_digest": _hex(f"sampling-{index}"),
                "augmentation_digest": _hex(f"augmentation-{index}"),
                "pair_digest": _hex(f"pair-{index}"),
                "supcon_enabled": supcon,
                "paired_enabled": paired,
            }
        )
    return entries


def build_record(
    val_ba=None,
    nll=None,
    *,
    recipe="D0-M",
    seed=SEEDS[0],
    role_id="role-invented",
    predicted=1,
    zero=None,
    best_digest=None,
    terminal_digest=None,
    **fields,
):
    if val_ba is None:
        val_ba = [0.5] * 30
    if nll is None:
        nll = [0.5] * len(val_ba)
    history = build_history(val_ba, nll, recipe=recipe, predicted=predicted, zero=zero)
    length = len(history)
    best_epoch = history[-1]["best_epoch"]
    selected = history[best_epoch - 1]
    projection = PROJECTION_RECIPES[recipe]
    parameter_count = EXPECTED_PARAMETER_COUNTS[(3, projection)]
    if best_digest is None:
        best_digest = _hex("best-state")
    if terminal_digest is None:
        terminal_digest = _hex("terminal-state")
    if best_epoch == length:
        terminal_digest = best_digest
    zero_total = sum(entry["zero_gradient_batches"] for entry in history)
    record = {
        "status": "complete",
        "reason_code": None,
        "history": history,
        "epochs_completed": length,
        "parameter_count": parameter_count,
        "optimizer_steps": 4 * length,
        "zero_gradient_batches": zero_total,
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": selected["validation_balanced_accuracy"],
        "best_validation_nll": selected["validation_nll"],
        "best_validation_macro_f1": selected["validation_macro_f1"],
        "best_validation_predicted_class_count": selected["validation_predicted_class_count"],
        "best_training_balanced_accuracy": selected["train_balanced_accuracy"],
        "collapse": selected["validation_predicted_class_count"] < 2,
        "best_state_digest": best_digest,
        "terminal_state_digest": terminal_digest,
        "state_capture_failed": False,
        "augmentation_digest": history[-1]["augmentation_digest"],
        "sampling_digest": history[-1]["sampling_digest"],
        "pair_digest": history[-1]["pair_digest"],
        "finite_gradient_batches": 4 * length,
        "nonzero_gradient_elements": 4 * length - zero_total,
        "role_id": role_id,
        "recipe": recipe,
        "seed": seed,
        "elapsed_seconds": 1.0,
        "peak_cuda_bytes": 0,
        "traceback_digest": None,
    }
    record.update(fields)
    return record


def kwargs_for(record, **overrides):
    values = {
        "expected_role_id": record["role_id"],
        "expected_recipe_id": record["recipe"],
        "expected_seed": record["seed"],
        "class_count": 3,
        "expected_best_state_sha256": record["best_state_digest"],
        "expected_terminal_state_sha256": record["terminal_state_digest"],
    }
    values.update(overrides)
    return values


def verify(record, **overrides):
    return verify_neural_training_record(record, **kwargs_for(record, **overrides))


def reason(record, **overrides):
    with pytest.raises(TrainingRecordError) as info:
        verify(record, **overrides)
    return info.value.reason_code


# --------------------------------------------------------------------------- #
# Acceptance
# --------------------------------------------------------------------------- #


def test_field_sets_exact():
    assert len(RECORD_FIELDS) == 28
    assert len(set(RECORD_FIELDS)) == 28
    assert len(HISTORY_FIELDS) == 17
    assert len(set(HISTORY_FIELDS)) == 17
    record = build_record()
    assert set(record) == set(RECORD_FIELDS)
    assert all(set(entry) == set(HISTORY_FIELDS) for entry in record["history"])


@pytest.mark.parametrize("recipe", sorted(FLAGS))
def test_all_recipes_accept(recipe):
    record = build_record(recipe=recipe)
    result = verify(record)
    assert result["training_record_consistent"] is True
    assert result["declared_checkpoint_links_consistent"] is True
    assert result["execution_authorized"] is False


def test_minimum_constant_history_and_tie_retains_earlier():
    record = build_record()
    result = verify(record)
    assert result["epochs_completed"] == 30
    assert result["best_epoch"] == 1
    assert result["collapse"] is True


def test_later_patience_stop_accepted():
    val_ba = [0.5 + 0.001 * index for index in range(40)] + [0.5 + 0.001 * 39] * 20
    record = build_record(val_ba=val_ba)
    result = verify(record)
    assert result["epochs_completed"] == 60
    assert result["best_epoch"] == 40


def test_maximum_continuously_improving_history_accepted():
    val_ba = [0.5 + 0.0001 * index for index in range(200)]
    record = build_record(val_ba=val_ba)
    result = verify(record)
    assert result["epochs_completed"] == 200
    assert result["best_epoch"] == 200


def test_ba_priority_over_nll():
    val_ba = [0.5, 0.6] + [0.6] * 28
    nll = [0.1, 100.0] + [100.0] * 28
    result = verify(build_record(val_ba=val_ba, nll=nll))
    assert result["best_epoch"] == 2


def test_nll_tiebreak():
    val_ba = [0.5, 0.5] + [0.5] * 28
    nll = [0.5, 0.4] + [0.4] * 28
    result = verify(build_record(val_ba=val_ba, nll=nll))
    assert result["best_epoch"] == 2


def test_finite_collapse_and_zero_gradients_accepted():
    record = build_record(
        zero=[4] * 30,
        best_digest=_hex("same"),
        terminal_digest=_hex("same"),
    )
    result = verify(record)
    assert result["collapse"] is True
    assert result["optimizer_steps"] == 120


def test_three_predicted_classes_not_collapsed_accepted():
    record = build_record(predicted=3)
    result = verify(record)
    assert result["collapse"] is False


def test_report_flags_and_hashes():
    record = build_record()
    result = verify(record)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["physical_role_isolation_verified"] is False
    assert result["training_completion_verified"] is False
    assert result["prediction_parity_verified"] is False
    assert result["live_resources_verified"] is False
    assert result["class_count"] == 3
    assert result["record_sha256"] == sha256_value(record)
    assert result["role_sha256"] == sha256_value(record["role_id"])
    body = {key: value for key, value in result.items() if key != "report_sha256"}
    assert result["report_sha256"] == sha256_value(body)


def test_report_does_not_leak_role_text():
    secret = "SECRET-ROLE-9"
    record = build_record(role_id=secret)
    rendered = json.dumps(verify(record))
    assert secret not in rendered


# --------------------------------------------------------------------------- #
# Sequence, stopping and summary selection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "field,bad",
    [
        ("improved", True),
        ("best_epoch", 5),
        ("nonimproving_epochs", 0),
    ],
)
def test_false_sequence_fields_rejected(field, bad):
    record = build_record()
    record["history"][10][field] = bad
    assert reason(record) == "history_sequence_mismatch"


def test_history_stopping_rule_mismatch():
    record = build_record(val_ba=[0.5] * 31)
    assert reason(record) == "history_stopping_mismatch"


def test_history_stopped_early_rejected():
    val_ba = [0.5 + 0.0001 * index for index in range(30)]
    record = build_record(val_ba=val_ba)
    assert reason(record) == "history_stopping_mismatch"


def test_selected_summary_mismatch_rejected():
    record = build_record()
    record["best_validation_macro_f1"] = 0.75
    assert reason(record) == "history_sequence_mismatch"


# --------------------------------------------------------------------------- #
# Cardinality, containers, counters, metrics
# --------------------------------------------------------------------------- #


def test_history_too_short_rejected():
    record = build_record()
    record["history"] = record["history"][:29]
    assert reason(record) == "invalid_history"


def test_history_too_long_rejected():
    record = build_record(val_ba=[0.5] * 201)
    assert reason(record) == "invalid_history"


class _DictSubclass(dict):
    pass


class _ListSubclass(list):
    pass


class _StrSubclass(str):
    pass


class _IntSubclass(int):
    pass


def test_missing_record_field_rejected():
    broken = build_record()
    broken.pop("pair_digest")
    assert reason(broken) == "invalid_record"


def test_extra_record_field_rejected():
    broken = build_record()
    broken["extra_field"] = "x"
    assert reason(broken) == "invalid_record"


def test_missing_history_field_rejected():
    broken = build_record()
    broken["history"][0].pop("pair_digest")
    assert reason(broken) == "invalid_history"


def test_extra_history_field_rejected():
    broken = build_record()
    broken["history"][0]["extra_field"] = "x"
    assert reason(broken) == "invalid_history"


def test_oversize_record_rejected_before_keysets(monkeypatch):
    record = build_record()
    record["extra_field"] = "x"

    def forbidden_set(*args, **kwargs):
        raise AssertionError("keyset constructed before cardinality check")

    monkeypatch.setattr(module, "set", forbidden_set, raising=False)
    assert reason(record) == "invalid_record"


def test_missing_record_field_rejected_before_keysets(monkeypatch):
    record = build_record()
    record.pop("pair_digest")

    def forbidden_set(*args, **kwargs):
        raise AssertionError("keyset constructed before cardinality check")

    monkeypatch.setattr(module, "set", forbidden_set, raising=False)
    assert reason(record) == "invalid_record"


def test_history_wrong_cardinality_rejected_before_keysets(monkeypatch):
    record = build_record()
    record["history"][0]["extra_field"] = "x"
    real_set = set

    def guarded_set(value=()):
        if isinstance(value, dict) and value is not record:
            raise AssertionError("history entry keyset constructed before cardinality check")
        return real_set(value)

    monkeypatch.setattr(module, "set", guarded_set, raising=False)
    assert reason(record) == "invalid_history"


def test_strict_container_types_rejected():
    record = build_record()
    with pytest.raises(TrainingRecordError) as info:
        verify_neural_training_record(_DictSubclass(record), **kwargs_for(record))
    assert info.value.reason_code == "invalid_record"

    broken = build_record()
    broken["history"] = _ListSubclass(broken["history"])
    assert reason(broken) == "invalid_history"

    broken = build_record()
    broken["history"][0] = _DictSubclass(broken["history"][0])
    assert reason(broken) == "invalid_history"

    broken = build_record()
    value = broken.pop("role_id")
    broken[_StrSubclass("role_id")] = value
    assert reason(broken) == "invalid_record"

    broken = build_record()
    broken["best_validation_macro_f1"] = _IntSubclass(1)
    assert reason(broken) == "invalid_metrics"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("optimizer_steps", -4),
        ("nonzero_gradient_elements", -1),
        ("peak_cuda_bytes", -1),
        ("elapsed_seconds", -0.5),
    ],
)
def test_negative_bounds_rejected(field, bad):
    record = build_record()
    record[field] = bad
    assert reason(record) == "invalid_counters"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("optimizer_steps", 116),
        ("epochs_completed", 29),
        ("finite_gradient_batches", 0),
        ("zero_gradient_batches", 1),
        ("parameter_count", 1),
    ],
)
def test_wrong_summary_counters_rejected(field, bad):
    record = build_record()
    record[field] = bad
    assert reason(record) == "invalid_counters"


def test_counter_zero_active_with_nonzero_rejected():
    record = build_record(zero=[4] * 30)
    assert record["optimizer_steps"] - record["zero_gradient_batches"] == 0
    record["nonzero_gradient_elements"] = 1
    assert reason(record) == "invalid_counters"


def test_counter_nonzero_below_active_rejected():
    record = build_record()
    assert record["optimizer_steps"] - record["zero_gradient_batches"] == 120
    record["nonzero_gradient_elements"] = 119
    assert reason(record) == "invalid_counters"


def test_counter_nonzero_above_parameter_active_bound_rejected():
    record = build_record(zero=[4] * 29 + [3])
    active = record["optimizer_steps"] - record["zero_gradient_batches"]
    assert active == 1
    record["nonzero_gradient_elements"] = record["parameter_count"] + 1
    assert (
        record["nonzero_gradient_elements"] < record["parameter_count"] * record["optimizer_steps"]
    )
    assert reason(record) == "invalid_counters"


def test_counter_exact_lower_bound_accepted():
    record = build_record()
    assert record["optimizer_steps"] == 120
    assert record["zero_gradient_batches"] == 0
    assert record["nonzero_gradient_elements"] == 120
    assert verify(record)["training_record_consistent"] is True


def test_counter_exact_upper_bound_accepted():
    record = build_record()
    record["nonzero_gradient_elements"] = record["parameter_count"] * record["optimizer_steps"]
    assert verify(record)["training_record_consistent"] is True


def test_counter_valid_mixed_zero_and_nonzero_accepted():
    record = build_record(zero=[1] * 30)
    assert record["zero_gradient_batches"] == 30
    assert record["optimizer_steps"] - record["zero_gradient_batches"] == 90
    assert record["nonzero_gradient_elements"] == 90
    assert verify(record)["training_record_consistent"] is True


def test_counter_all_zero_gradient_batches_accepted():
    record = build_record(zero=[4] * 30)
    assert record["zero_gradient_batches"] == 120
    assert record["optimizer_steps"] - record["zero_gradient_batches"] == 0
    assert record["nonzero_gradient_elements"] == 0
    assert verify(record)["training_record_consistent"] is True


def test_history_zero_gradient_out_of_range_rejected():
    record = build_record()
    record["history"][0]["zero_gradient_batches"] = 5
    assert reason(record) == "invalid_history"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("best_validation_balanced_accuracy", True),
        ("best_validation_balanced_accuracy", "0.5"),
        ("best_validation_balanced_accuracy", float("nan")),
        ("best_validation_balanced_accuracy", 1.5),
        ("best_validation_macro_f1", -0.1),
        ("best_validation_nll", -0.1),
        ("best_validation_nll", float("inf")),
    ],
)
def test_summary_metric_bad_rejected(field, bad):
    record = build_record()
    record[field] = bad
    assert reason(record) == "invalid_metrics"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("validation_balanced_accuracy", True),
        ("validation_balanced_accuracy", 2.0),
        ("validation_nll", -0.5),
        ("validation_macro_f1", "x"),
        ("train_balanced_accuracy", float("nan")),
    ],
)
def test_history_metric_bad_rejected(field, bad):
    record = build_record()
    record["history"][3][field] = bad
    assert reason(record) == "invalid_history"


def test_class_count_two_rejected():
    record = build_record()
    assert reason(record, class_count=2) == "invalid_expected_inputs"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("epochs_completed", True),
        ("optimizer_steps", 120.0),
        ("zero_gradient_batches", False),
        ("finite_gradient_batches", 120.0),
        ("nonzero_gradient_elements", True),
        ("parameter_count", 1.0),
        ("peak_cuda_bytes", 0.0),
    ],
)
def test_summary_counter_coercion_rejected(field, bad):
    record = build_record()
    record[field] = bad
    assert reason(record) == "invalid_counters"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("epoch", True),
        ("epoch_optimizer_steps", 4.0),
        ("total_optimizer_steps", True),
        ("zero_gradient_batches", False),
        ("validation_predicted_class_count", 1.0),
        ("best_epoch", True),
        ("nonimproving_epochs", 0.0),
    ],
)
def test_history_counter_coercion_rejected(field, bad):
    record = build_record()
    record["history"][0][field] = bad
    assert reason(record) == "invalid_history"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("peak_cuda_bytes", module._MAXIMUM_CUDA_BYTES + 1),
        ("elapsed_seconds", module._MAXIMUM_FIT_SECONDS + 0.5),
    ],
)
def test_resource_bound_exceeded_rejected(field, bad):
    record = build_record()
    record[field] = bad
    assert reason(record) == "invalid_counters"


@pytest.mark.parametrize("bad", ["role\x00x", "role\x01x", "\ud800role", "role\udfff"])
def test_control_or_surrogate_expected_role_rejected(bad):
    record = build_record()
    assert reason(record, expected_role_id=bad) == "invalid_expected_inputs"


@pytest.mark.parametrize("bad", ["role\x01x", "\ud800role", "role\udfff"])
def test_control_or_surrogate_record_role_rejected(bad):
    record = build_record(role_id=bad)
    assert reason(record, expected_role_id="valid-role") == "identity_mismatch"


@pytest.mark.parametrize("bad", ["A" * 64, "g" * 64, "0" * 63, "0" * 65])
def test_malformed_summary_digest_rejected(bad):
    record = build_record()
    record["best_state_digest"] = bad
    assert reason(record, expected_best_state_sha256=_hex("valid-expected")) == "invalid_digests"


@pytest.mark.parametrize("recipe", sorted(FLAGS))
@pytest.mark.parametrize("field", ["supcon_enabled", "paired_enabled"])
def test_wrong_recipe_flags_rejected(recipe, field):
    record = build_record(recipe=recipe)
    record["history"][0][field] = not record["history"][0][field]
    assert reason(record) == "invalid_recipe_flags"


# --------------------------------------------------------------------------- #
# Identity, completion and digests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides",
    [
        {"expected_role_id": ""},
        {"expected_role_id": " padded "},
        {"expected_role_id": "x" * 257},
        {"expected_role_id": ["not", "hashable"]},
        {"expected_recipe_id": "D4"},
        {"expected_recipe_id": ["D0-M"]},
        {"expected_seed": 1},
        {"expected_seed": True},
        {"expected_seed": ["x"]},
        {"class_count": 2},
        {"class_count": True},
        {"expected_best_state_sha256": "Z" * 64},
        {"expected_best_state_sha256": None},
        {"expected_terminal_state_sha256": b"x" * 64},
    ],
)
def test_invalid_expected_inputs_rejected(overrides):
    record = build_record()
    assert reason(record, **overrides) == "invalid_expected_inputs"


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "failed"},
        {"status": 5},
        {"reason_code": "deadline_exceeded"},
        {"traceback_digest": "a" * 64},
        {"state_capture_failed": True},
    ],
)
def test_non_complete_record_rejected(overrides):
    record = build_record()
    record.update(overrides)
    assert reason(record) == "record_not_complete"


def test_identity_mismatch_rejected():
    record = build_record(role_id="role-a")
    assert reason(record, expected_role_id="role-b") == "identity_mismatch"


def test_state_digest_mismatch_rejected():
    record = build_record()
    assert reason(record, expected_best_state_sha256=_hex("other")) == "checkpoint_link_mismatch"
    assert (
        reason(record, expected_terminal_state_sha256=_hex("other")) == "checkpoint_link_mismatch"
    )


def test_last_trace_digest_mismatch_rejected():
    record = build_record()
    record["history"][-1]["sampling_digest"] = _hex("tampered")
    assert reason(record) == "invalid_digests"


def test_summary_digest_syntax_rejected():
    record = build_record()
    record["augmentation_digest"] = "bad"
    assert reason(record) == "invalid_digests"


def test_final_epoch_requires_matching_state_digests():
    val_ba = [0.5 + 0.0001 * index for index in range(200)]
    record = build_record(val_ba=val_ba)
    assert verify(record)["best_epoch"] == 200
    record["terminal_state_digest"] = _hex("different")
    assert reason(record) == "checkpoint_link_mismatch"


# --------------------------------------------------------------------------- #
# Ownership, sanitization and no side effects
# --------------------------------------------------------------------------- #


def test_caller_inputs_not_mutated():
    record = build_record()
    before = copy.deepcopy(record)
    verify(record)
    assert record == before


def test_mutation_after_snapshot_does_not_change_report(monkeypatch):
    record = build_record()
    baseline = verify(record)
    real = module.sha256_value
    state = {"calls": 0}

    def hooked(value):
        state["calls"] += 1
        if state["calls"] == 1:
            record["history"][3]["validation_balanced_accuracy"] = 0.999
            record["role_id"] = "role-mutated"
        return real(value)

    monkeypatch.setattr(module, "sha256_value", hooked)
    assert verify(record) == baseline
    assert state["calls"] >= 1


def test_snapshot_isolated_from_mutation_during_stopping(monkeypatch):
    record = build_record()
    baseline = verify(record)
    real = module._validate_stopping
    state = {"calls": 0}

    def hooked(history, expected_best_epoch):
        state["calls"] += 1
        if state["calls"] == 1:
            record["history"][2]["validation_balanced_accuracy"] = 0.999
            record["history"][4]["validation_macro_f1"] = 0.999
            record["role_id"] = "role-mutated"
            record["best_epoch"] = 7
        return real(history, expected_best_epoch)

    monkeypatch.setattr(module, "_validate_stopping", hooked)
    assert verify(record) == baseline
    assert state["calls"] >= 1


def test_unexpected_exception_sanitized(monkeypatch):
    record = build_record()

    def boom(*args, **kwargs):
        raise RuntimeError("SYNTHETIC_PRIVATE_DETAIL")

    monkeypatch.setattr(module, "_validate_expected", boom)
    with pytest.raises(TrainingRecordError) as info:
        verify(record)
    assert info.value.reason_code == "verification_failed"
    assert "SYNTHETIC_PRIVATE_DETAIL" not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True


def test_keyboard_interrupt_propagates_same_object(monkeypatch):
    record = build_record()
    interrupt = KeyboardInterrupt("stop")

    def raising(*args, **kwargs):
        raise interrupt

    monkeypatch.setattr(module, "_snapshot_record", raising)
    with pytest.raises(KeyboardInterrupt) as info:
        verify(record)
    assert info.value is interrupt


def test_system_exit_propagates_same_object(monkeypatch):
    record = build_record()
    exit_signal = SystemExit(7)

    def raising(*args, **kwargs):
        raise exit_signal

    monkeypatch.setattr(module, "_snapshot_record", raising)
    with pytest.raises(SystemExit) as info:
        verify(record)
    assert info.value is exit_signal


@pytest.mark.parametrize("bad", [None, ["x"], 5, "not-a-code"])
def test_error_sanitizes_unknown_codes(bad):
    assert TrainingRecordError(bad).reason_code == "unlisted_reason_code"


def test_error_accepts_allowlisted_code():
    error = TrainingRecordError("invalid_record")
    assert error.reason_code == "invalid_record"
    assert str(error) == "invalid_record"


def test_no_filesystem_or_scientific_calls(monkeypatch):
    record = build_record()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("forbidden call attempted")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(AcquisitionClassifier, "__init__", forbidden)
    monkeypatch.setattr(AcquisitionClassifier, "forward", forbidden)
    monkeypatch.setattr(p05_development, "train_development_fit", forbidden)
    monkeypatch.setattr(p04_runtime, "_metric_values", forbidden)
    monkeypatch.setattr(p05_development, "_metric_values", forbidden)
    monkeypatch.setattr(p05_development, "_predict_logits", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden, raising=False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden, raising=False)
    verify(record)
    assert calls == []
    assert not hasattr(module, "torch")
    assert not hasattr(module, "numpy")


def test_execution_always_denied():
    for call in (
        lambda: require_scientific_execution(),
        lambda: require_scientific_execution(execution_authorized=True),
        lambda: require_scientific_execution(None, authorized=True, token="x"),
    ):
        with pytest.raises(TrainingRecordError) as info:
            call()
        assert info.value.reason_code == "scientific_execution_not_authorized"


# --------------------------------------------------------------------------- #
# Composition with the accepted checkpoint and prediction boundaries
# --------------------------------------------------------------------------- #


def _make_state(class_count, use_projection, seed=17):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AcquisitionClassifier(class_count, use_projection=use_projection)
    model = model.to(device=torch.device("cpu"), dtype=torch.float32)
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _serialize(state):
    buffer = io.BytesIO()
    torch.save({"state_dict": state}, buffer)
    return buffer.getvalue()


def _verify_checkpoint(state, recipe):
    blob = _serialize(state)
    return verify_neural_checkpoint_bytes(
        blob,
        expected_file_sha256=hashlib.sha256(blob).hexdigest(),
        expected_state_sha256=_state_hash(state),
        class_count=3,
        recipe_id=recipe,
    )


@pytest.mark.parametrize("recipe", sorted(FLAGS))
def test_composition_three_boundaries_still_deny(recipe):
    sp_kwargs = valid_kwargs(model_id=recipe, classes=THREE_CLASSES)
    fit_job = sp_kwargs["fit_job"]
    record_recipe = fit_job["model_id"]
    record_seed = fit_job["seed"]
    assert record_recipe == recipe
    assert record_seed == SEEDS[0]

    role_id = "p08-role-" + record_recipe
    assert fit_job["unit_id"] != role_id

    use_projection = PROJECTION_RECIPES[record_recipe]
    assert use_projection == PROJECTION_RECIPES[fit_job["model_id"]]
    best_state = _make_state(3, use_projection, seed=101)
    terminal_state = _make_state(3, use_projection, seed=202)
    best_report = _verify_checkpoint(best_state, record_recipe)
    terminal_report = _verify_checkpoint(terminal_state, record_recipe)
    assert best_report["checkpoint_content_verified"] is True
    assert terminal_report["checkpoint_content_verified"] is True

    history = build_history([0.5] * 30, [0.5] * 30, recipe=record_recipe, predicted=1)
    result = DevelopmentFitResult(
        status="complete",
        history=history,
        epochs_completed=30,
        parameter_count=EXPECTED_PARAMETER_COUNTS[(3, use_projection)],
        optimizer_steps=120,
        zero_gradient_batches=0,
        best_epoch=1,
        best_validation_balanced_accuracy=0.5,
        best_validation_nll=0.5,
        best_validation_macro_f1=0.5,
        best_validation_predicted_class_count=1,
        best_training_balanced_accuracy=0.5,
        collapse=True,
        best_state_digest=best_report["checkpoint_state_sha256"],
        terminal_state_digest=terminal_report["checkpoint_state_sha256"],
        augmentation_digest=history[-1]["augmentation_digest"],
        sampling_digest=history[-1]["sampling_digest"],
        pair_digest=history[-1]["pair_digest"],
        finite_gradient_batches=120,
        nonzero_gradient_elements=120,
        role_id=role_id,
        recipe=record_recipe,
        seed=record_seed,
        elapsed_seconds=1.0,
        peak_cuda_bytes=0,
    )
    summary = p05_pilot._result_private_summary(result)
    record = {name: summary[name] for name in RECORD_FIELDS}
    record["history"] = [
        {name: entry[name] for name in HISTORY_FIELDS} for entry in summary["history"]
    ]
    assert record["recipe"] == fit_job["model_id"]
    assert record["seed"] == fit_job["seed"]

    sp_report = verify_source_prediction_values(**sp_kwargs)
    assert sp_report["declared_job_pair_verified"] is True
    assert sp_report["source_prediction_structure_verified"] is True

    record_report = verify_neural_training_record(
        record,
        expected_role_id=role_id,
        expected_recipe_id=record_recipe,
        expected_seed=record_seed,
        class_count=3,
        expected_best_state_sha256=best_report["checkpoint_state_sha256"],
        expected_terminal_state_sha256=terminal_report["checkpoint_state_sha256"],
    )
    assert record_report["training_record_consistent"] is True
    assert record_report["declared_checkpoint_links_consistent"] is True

    for report in (best_report, terminal_report, sp_report, record_report):
        assert report["execution_authorized"] is False
        assert report["prediction_parity_verified"] is False
    assert sp_report["external_authentication_verified"] is False
    assert record_report["external_authentication_verified"] is False
    assert record_report["training_completion_verified"] is False

    wrong_scores = np.full((len(THREE_CLASSES), len(THREE_CLASSES)), 0.0, dtype=np.float64)
    wrong_scores[0, 0] = 1e308
    wrong_kwargs = dict(sp_kwargs)
    wrong_kwargs["scores"] = wrong_scores
    wrong_report = verify_source_prediction_values(**wrong_kwargs)
    assert wrong_report["source_prediction_structure_verified"] is True
    assert wrong_report["prediction_parity_verified"] is False
