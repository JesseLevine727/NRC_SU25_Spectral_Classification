"""Strict inherited training-record consistency verification (P08-T085).

This module checks that a caller-extracted, in-memory projection of one
completed P05 development fit is internally consistent and agrees with the
declared training identity and independently verified checkpoint state digests.
It is a pure metadata reader: it opens no files, builds no model, computes no
metric, draws no random number, runs no fit and produces no prediction.

Non-claims
----------
* ``training_record_consistent`` and ``declared_checkpoint_links_consistent``
  record only internal agreement against caller-supplied pins.  They are not
  evidence of provenance, real fitting, numerical parity or resource bounds.
* ``external_authentication_verified``, ``physical_role_isolation_verified``,
  ``training_completion_verified``, ``prediction_parity_verified`` and
  ``live_resources_verified`` are always ``False``.
* Declared elapsed-time and CUDA-memory values are replay constraints copied
  from the inherited inner kernel contract, not live enforcement or new scope
  limits.
* Only scalars, counts and hashes leave this interface.  No metric value, role
  identifier, recipe identifier, seed, raw history or label is returned.
* Execution through this module is never authorized.
"""

from __future__ import annotations

import math

from atlas_sers.evaluation.p05_pilot import (
    P05PilotError,
    _check_stopping,
    _expected_best_epoch,
)
from atlas_sers.evaluation.p08_plan import SEEDS
from atlas_sers.evaluation.p08_source_artifacts import (
    EXPECTED_PARAMETER_COUNTS,
    PROJECTION_RECIPES,
)
from atlas_sers.governance.canonical import sha256_value

SCHEMA_VERSION = "nato-sers-p08-neural-training-record-v1"

RECORD_FIELDS = (
    "status",
    "reason_code",
    "history",
    "epochs_completed",
    "parameter_count",
    "optimizer_steps",
    "zero_gradient_batches",
    "best_epoch",
    "best_validation_balanced_accuracy",
    "best_validation_nll",
    "best_validation_macro_f1",
    "best_validation_predicted_class_count",
    "best_training_balanced_accuracy",
    "collapse",
    "best_state_digest",
    "terminal_state_digest",
    "state_capture_failed",
    "augmentation_digest",
    "sampling_digest",
    "pair_digest",
    "finite_gradient_batches",
    "nonzero_gradient_elements",
    "role_id",
    "recipe",
    "seed",
    "elapsed_seconds",
    "peak_cuda_bytes",
    "traceback_digest",
)

HISTORY_FIELDS = (
    "epoch",
    "epoch_optimizer_steps",
    "total_optimizer_steps",
    "zero_gradient_batches",
    "train_balanced_accuracy",
    "validation_balanced_accuracy",
    "validation_nll",
    "validation_macro_f1",
    "validation_predicted_class_count",
    "improved",
    "best_epoch",
    "nonimproving_epochs",
    "sampling_digest",
    "augmentation_digest",
    "pair_digest",
    "supcon_enabled",
    "paired_enabled",
)

_TRACE_FIELDS = ("sampling_digest", "augmentation_digest", "pair_digest")

_RECIPE_FLAGS = {
    "D0-M": (False, False),
    "D1": (True, False),
    "D2": (False, True),
    "D3": (True, True),
}

_EPOCH_MINIMUM = 30
_EPOCH_MAXIMUM = 200
_BATCHES_PER_EPOCH = 4
_MAXIMUM_FIT_SECONDS = 120.0
_MAXIMUM_CUDA_BYTES = 4294967296
_MAXIMUM_IDENTIFIER_LENGTH = 256
_HEX_DIGITS = frozenset("0123456789abcdef")

_REASON_CODES = frozenset(
    {
        "scientific_execution_not_authorized",
        "invalid_record",
        "record_not_complete",
        "invalid_expected_inputs",
        "identity_mismatch",
        "invalid_metrics",
        "invalid_counters",
        "invalid_history",
        "invalid_digests",
        "invalid_recipe_flags",
        "history_sequence_mismatch",
        "history_stopping_mismatch",
        "checkpoint_link_mismatch",
        "verification_failed",
        "unlisted_reason_code",
    }
)

__all__ = [
    "HISTORY_FIELDS",
    "RECORD_FIELDS",
    "SCHEMA_VERSION",
    "TrainingRecordError",
    "require_scientific_execution",
    "verify_neural_training_record",
]


class TrainingRecordError(ValueError):
    """ValueError carrying one static allowlisted reason code."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "unlisted_reason_code"
        super().__init__(reason_code)
        self.reason_code = reason_code


def _fail(reason_code):
    raise TrainingRecordError(reason_code) from None


def _is_exact_bool(value):
    return type(value) is bool


def _is_finite_number(value):
    if type(value) is not int and type(value) is not float:
        return False
    return math.isfinite(value)


def _is_unit_number(value):
    return _is_finite_number(value) and 0.0 <= float(value) <= 1.0


def _is_lower_hex64(value):
    return type(value) is str and len(value) == 64 and all(ch in _HEX_DIGITS for ch in value)


def _is_identifier(value):
    if type(value) is not str:
        return False
    if not value or value != value.strip() or len(value) > _MAXIMUM_IDENTIFIER_LENGTH:
        return False
    for character in value:
        if ord(character) < 0x20:
            return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _validate_expected(
    *,
    expected_role_id,
    expected_recipe_id,
    expected_seed,
    class_count,
    expected_best_state_sha256,
    expected_terminal_state_sha256,
):
    if not _is_identifier(expected_role_id):
        _fail("invalid_expected_inputs")
    if type(expected_recipe_id) is not str or expected_recipe_id not in _RECIPE_FLAGS:
        _fail("invalid_expected_inputs")
    if type(expected_seed) is not int or expected_seed not in SEEDS:
        _fail("invalid_expected_inputs")
    if type(class_count) is not int or class_count != 3:
        _fail("invalid_expected_inputs")
    if not _is_lower_hex64(expected_best_state_sha256):
        _fail("invalid_expected_inputs")
    if not _is_lower_hex64(expected_terminal_state_sha256):
        _fail("invalid_expected_inputs")


def _snapshot_record(record):
    """Return a private defensive copy of a caller-owned record.

    The caller retains ownership of ``record`` and MUST NOT mutate ``record``
    or any of its history entries while this snapshot is being captured.  The
    top-level record and every history entry are copied into fresh
    dictionaries, so a mutation performed only after this capture completes
    cannot alter the checked report.  This is defensive copying for sequential
    callers; it is not an atomic-concurrency or locking guarantee and does not
    make concurrent mutation safe.
    """
    if type(record) is not dict:
        _fail("invalid_record")
    if len(record) != len(RECORD_FIELDS):
        _fail("invalid_record")
    for key in record:
        if type(key) is not str or key not in RECORD_FIELDS:
            _fail("invalid_record")
    if set(record) != set(RECORD_FIELDS):
        _fail("invalid_record")

    history = record["history"]
    if type(history) is not list:
        _fail("invalid_history")
    if not (_EPOCH_MINIMUM <= len(history) <= _EPOCH_MAXIMUM):
        _fail("invalid_history")
    for entry in history:
        if type(entry) is not dict:
            _fail("invalid_history")
        if len(entry) != len(HISTORY_FIELDS):
            _fail("invalid_history")
        for key in entry:
            if type(key) is not str or key not in HISTORY_FIELDS:
                _fail("invalid_history")
        if set(entry) != set(HISTORY_FIELDS):
            _fail("invalid_history")

    snapshot = {}
    for name in RECORD_FIELDS:
        if name == "history":
            snapshot["history"] = [dict(entry) for entry in history]
        else:
            snapshot[name] = record[name]
    return snapshot


def _validate_completion(snapshot):
    if type(snapshot["status"]) is not str or snapshot["status"] != "complete":
        _fail("record_not_complete")
    if snapshot["reason_code"] is not None or snapshot["traceback_digest"] is not None:
        _fail("record_not_complete")
    if snapshot["state_capture_failed"] is not False:
        _fail("record_not_complete")


def _validate_identity(snapshot, *, expected_role_id, expected_recipe_id, expected_seed):
    if not _is_identifier(snapshot["role_id"]) or snapshot["role_id"] != expected_role_id:
        _fail("identity_mismatch")
    if type(snapshot["recipe"]) is not str or snapshot["recipe"] != expected_recipe_id:
        _fail("identity_mismatch")
    if type(snapshot["seed"]) is not int or snapshot["seed"] != expected_seed:
        _fail("identity_mismatch")


def _validate_snapshot(snapshot, *, expected_role_id, expected_recipe_id, expected_seed):
    _validate_completion(snapshot)
    _validate_identity(
        snapshot,
        expected_role_id=expected_role_id,
        expected_recipe_id=expected_recipe_id,
        expected_seed=expected_seed,
    )

    projection = PROJECTION_RECIPES[expected_recipe_id]
    parameter_count = EXPECTED_PARAMETER_COUNTS[(3, projection)]

    for name in (
        "best_validation_balanced_accuracy",
        "best_validation_macro_f1",
        "best_training_balanced_accuracy",
    ):
        if not _is_unit_number(snapshot[name]):
            _fail("invalid_metrics")
    if not _is_finite_number(snapshot["best_validation_nll"]) or (
        float(snapshot["best_validation_nll"]) < 0.0
    ):
        _fail("invalid_metrics")
    predicted = snapshot["best_validation_predicted_class_count"]
    if type(predicted) is not int or not (1 <= predicted <= 3):
        _fail("invalid_metrics")
    if not _is_exact_bool(snapshot["collapse"]) or snapshot["collapse"] != (predicted < 2):
        _fail("invalid_metrics")

    history = snapshot["history"]
    epochs = snapshot["epochs_completed"]
    if (
        type(epochs) is not int
        or epochs != len(history)
        or not (_EPOCH_MINIMUM <= epochs <= _EPOCH_MAXIMUM)
    ):
        _fail("invalid_counters")
    steps = snapshot["optimizer_steps"]
    if type(steps) is not int or steps != _BATCHES_PER_EPOCH * epochs:
        _fail("invalid_counters")
    if (
        type(snapshot["parameter_count"]) is not int
        or snapshot["parameter_count"] != parameter_count
    ):
        _fail("invalid_counters")
    if (
        type(snapshot["finite_gradient_batches"]) is not int
        or snapshot["finite_gradient_batches"] != steps
    ):
        _fail("invalid_counters")
    zero = snapshot["zero_gradient_batches"]
    if type(zero) is not int or not (0 <= zero <= steps):
        _fail("invalid_counters")
    nonzero = snapshot["nonzero_gradient_elements"]
    if type(nonzero) is not int or not (0 <= nonzero <= parameter_count * steps):
        _fail("invalid_counters")
    active_gradient_batches = steps - zero
    if not (active_gradient_batches <= nonzero <= parameter_count * active_gradient_batches):
        _fail("invalid_counters")
    peak = snapshot["peak_cuda_bytes"]
    if type(peak) is not int or not (0 <= peak <= _MAXIMUM_CUDA_BYTES):
        _fail("invalid_counters")
    elapsed = snapshot["elapsed_seconds"]
    if not _is_finite_number(elapsed) or not (0.0 <= float(elapsed) <= _MAXIMUM_FIT_SECONDS):
        _fail("invalid_counters")

    for name in (
        "best_state_digest",
        "terminal_state_digest",
        "augmentation_digest",
        "sampling_digest",
        "pair_digest",
    ):
        if not _is_lower_hex64(snapshot[name]):
            _fail("invalid_digests")

    flags = _RECIPE_FLAGS[expected_recipe_id]
    best_key = None
    expected_best_epoch = None
    nonimproving = 0

    for index, entry in enumerate(history, start=1):
        if type(entry["epoch"]) is not int or entry["epoch"] != index:
            _fail("invalid_history")
        if (
            type(entry["epoch_optimizer_steps"]) is not int
            or entry["epoch_optimizer_steps"] != _BATCHES_PER_EPOCH
        ):
            _fail("invalid_history")
        if (
            type(entry["total_optimizer_steps"]) is not int
            or entry["total_optimizer_steps"] != _BATCHES_PER_EPOCH * index
        ):
            _fail("invalid_history")
        epoch_zero = entry["zero_gradient_batches"]
        if type(epoch_zero) is not int or not (0 <= epoch_zero <= _BATCHES_PER_EPOCH):
            _fail("invalid_history")
        if not _is_unit_number(entry["train_balanced_accuracy"]):
            _fail("invalid_history")
        if not _is_unit_number(entry["validation_balanced_accuracy"]):
            _fail("invalid_history")
        if not _is_finite_number(entry["validation_nll"]) or float(entry["validation_nll"]) < 0.0:
            _fail("invalid_history")
        if not _is_unit_number(entry["validation_macro_f1"]):
            _fail("invalid_history")
        epoch_predicted = entry["validation_predicted_class_count"]
        if type(epoch_predicted) is not int or not (1 <= epoch_predicted <= 3):
            _fail("invalid_history")
        if not _is_exact_bool(entry["improved"]):
            _fail("invalid_history")
        if type(entry["best_epoch"]) is not int or not (1 <= entry["best_epoch"] <= index):
            _fail("invalid_history")
        if type(entry["nonimproving_epochs"]) is not int or not (
            0 <= entry["nonimproving_epochs"] <= _EPOCH_MAXIMUM
        ):
            _fail("invalid_history")
        for name in _TRACE_FIELDS:
            if not _is_lower_hex64(entry[name]):
                _fail("invalid_digests")
        if not _is_exact_bool(entry["supcon_enabled"]) or not _is_exact_bool(
            entry["paired_enabled"]
        ):
            _fail("invalid_recipe_flags")
        if (entry["supcon_enabled"], entry["paired_enabled"]) != flags:
            _fail("invalid_recipe_flags")

        key = (-float(entry["validation_balanced_accuracy"]), float(entry["validation_nll"]))
        if best_key is None or key < best_key:
            best_key = key
            expected_best_epoch = index
            nonimproving = 0
            expected_improved = True
        else:
            nonimproving += 1
            expected_improved = False
        if entry["improved"] is not expected_improved:
            _fail("history_sequence_mismatch")
        if entry["best_epoch"] != expected_best_epoch:
            _fail("history_sequence_mismatch")
        if entry["nonimproving_epochs"] != nonimproving:
            _fail("history_sequence_mismatch")

    if sum(entry["zero_gradient_batches"] for entry in history) != zero:
        _fail("invalid_counters")

    last = history[-1]
    for name in _TRACE_FIELDS:
        if last[name] != snapshot[name]:
            _fail("invalid_digests")

    summary_best_epoch = snapshot["best_epoch"]
    if type(summary_best_epoch) is not int or summary_best_epoch != expected_best_epoch:
        _fail("history_sequence_mismatch")
    selected = history[summary_best_epoch - 1]
    if snapshot["best_training_balanced_accuracy"] != selected["train_balanced_accuracy"]:
        _fail("history_sequence_mismatch")
    if snapshot["best_validation_balanced_accuracy"] != selected["validation_balanced_accuracy"]:
        _fail("history_sequence_mismatch")
    if snapshot["best_validation_nll"] != selected["validation_nll"]:
        _fail("history_sequence_mismatch")
    if snapshot["best_validation_macro_f1"] != selected["validation_macro_f1"]:
        _fail("history_sequence_mismatch")
    if (
        snapshot["best_validation_predicted_class_count"]
        != selected["validation_predicted_class_count"]
    ):
        _fail("history_sequence_mismatch")

    _validate_stopping(history, expected_best_epoch)
    return parameter_count


def _validate_stopping(history, expected_best_epoch):
    try:
        inherited_best = _expected_best_epoch(history)
        _check_stopping(history)
    except TrainingRecordError:
        raise
    except P05PilotError:
        _fail("history_stopping_mismatch")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("verification_failed")
    if inherited_best != expected_best_epoch:
        _fail("history_sequence_mismatch")


def _validate_checkpoint_links(snapshot, expected_best, expected_terminal):
    if snapshot["best_state_digest"] != expected_best:
        _fail("checkpoint_link_mismatch")
    if snapshot["terminal_state_digest"] != expected_terminal:
        _fail("checkpoint_link_mismatch")
    if snapshot["best_epoch"] == snapshot["epochs_completed"]:
        if snapshot["best_state_digest"] != snapshot["terminal_state_digest"]:
            _fail("checkpoint_link_mismatch")


def _build_report(snapshot, expected_best, expected_terminal, parameter_count):
    report = {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "training_record_consistent": True,
        "declared_checkpoint_links_consistent": True,
        "external_authentication_verified": False,
        "physical_role_isolation_verified": False,
        "training_completion_verified": False,
        "prediction_parity_verified": False,
        "live_resources_verified": False,
        "class_count": 3,
        "parameter_count": parameter_count,
        "epochs_completed": snapshot["epochs_completed"],
        "optimizer_steps": snapshot["optimizer_steps"],
        "best_epoch": snapshot["best_epoch"],
        "collapse": snapshot["collapse"],
        "record_sha256": sha256_value(snapshot),
        "role_sha256": sha256_value(snapshot["role_id"]),
        "best_state_sha256": expected_best,
        "terminal_state_sha256": expected_terminal,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def _verify_neural_training_record(
    record,
    *,
    expected_role_id,
    expected_recipe_id,
    expected_seed,
    class_count,
    expected_best_state_sha256,
    expected_terminal_state_sha256,
):
    _validate_expected(
        expected_role_id=expected_role_id,
        expected_recipe_id=expected_recipe_id,
        expected_seed=expected_seed,
        class_count=class_count,
        expected_best_state_sha256=expected_best_state_sha256,
        expected_terminal_state_sha256=expected_terminal_state_sha256,
    )
    snapshot = _snapshot_record(record)
    parameter_count = _validate_snapshot(
        snapshot,
        expected_role_id=expected_role_id,
        expected_recipe_id=expected_recipe_id,
        expected_seed=expected_seed,
    )
    _validate_checkpoint_links(snapshot, expected_best_state_sha256, expected_terminal_state_sha256)
    return _build_report(
        snapshot,
        expected_best_state_sha256,
        expected_terminal_state_sha256,
        parameter_count,
    )


def verify_neural_training_record(
    record,
    *,
    expected_role_id,
    expected_recipe_id,
    expected_seed,
    class_count,
    expected_best_state_sha256,
    expected_terminal_state_sha256,
):
    """Verify one in-memory inherited training record.

    Ordinary failures collapse to a static allowlisted
    :class:`TrainingRecordError`; ``KeyboardInterrupt`` and ``SystemExit``
    propagate unchanged as the same object.
    """
    try:
        return _verify_neural_training_record(
            record,
            expected_role_id=expected_role_id,
            expected_recipe_id=expected_recipe_id,
            expected_seed=expected_seed,
            class_count=class_count,
            expected_best_state_sha256=expected_best_state_sha256,
            expected_terminal_state_sha256=expected_terminal_state_sha256,
        )
    except TrainingRecordError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("verification_failed")


def require_scientific_execution(*args, **kwargs):
    """Always deny execution, regardless of forged flags or arguments."""
    _fail("scientific_execution_not_authorized")
