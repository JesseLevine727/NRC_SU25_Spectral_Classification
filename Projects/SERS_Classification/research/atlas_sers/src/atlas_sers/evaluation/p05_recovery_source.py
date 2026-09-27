"""Read-only recovery source resolver and accounting bridge for P05.

This module resolves the source-development stage paths for the legacy
comprehensive run and for an approved recovered run, and normalizes the
source-execution accounting consumed by later refit and evaluation stages.
It reads only the bounded private recovery receipt through the existing strict
input parser, rejects symlinks, partial files and ambiguous dual complete
authority, and never trains, imports ``torch`` or ``numpy``, loads logits or
checkpoints, or writes any file.

The resolver does NOT authenticate the full source evidence.  Manifests,
inventories, selector records, plans, checkpoints, logits and leases remain the
responsibility of the existing full acceptance checks; the caller MUST run
those existing checks after resolution.  The accounting helpers validate
claimed accounting values only and make no claim that any original data file
was read.
"""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_recovery_authority as authority
from atlas_sers.evaluation import p05_recovery_inputs as recovery_inputs
from atlas_sers.evaluation import p05_recovery_receipt as receipt

__all__ = [
    "ACCOUNTING_SCHEMA_VERSION",
    "CLEAN_MAXIMUM_NEW_NEURAL_EXECUTIONS",
    "CLEAN_MAXIMUM_NEW_OPTIMIZER_STEPS",
    "CLEAN_MAXIMUM_SOURCE_OPTIMIZER_STEPS",
    "CLEAN_SOURCE_ATTEMPTS",
    "CLEAN_SOURCE_INTERRUPTED_ATTEMPTS",
    "CLEAN_SOURCE_SUCCESSFUL_FITS",
    "COMPREHENSIVE_NAMESPACE",
    "INTERRUPTED_CHARGED_STEPS",
    "INTERRUPTED_OBSERVED_STEPS",
    "RECOVERED_MAXIMUM_NEW_NEURAL_EXECUTIONS",
    "RECOVERED_MAXIMUM_NEW_OPTIMIZER_STEPS",
    "RECOVERED_MAXIMUM_SOURCE_OPTIMIZER_STEPS",
    "RECOVERED_SOURCE_ATTEMPTS",
    "RECOVERED_SOURCE_INTERRUPTED_ATTEMPTS",
    "RECOVERED_SOURCE_SUCCESSFUL_FITS",
    "RECOVERED_UNIT_COUNT",
    "RecoverySourceError",
    "accounting_from_recovered",
    "clean_accounting",
    "from_authenticated",
    "is_recovered",
    "resolve_paths",
    "validate_accounting",
]

ACCOUNTING_SCHEMA_VERSION = "nato-sers-p05-source-accounting-v1"
RECOVERY_ACCOUNTING_MODE = "recovered"
CLEAN_ACCOUNTING_MODE = "clean"

COMPREHENSIVE_NAMESPACE = "p05comprehensive"
DEVELOP_STAGE_NAME = "develop"
SELECTION_STAGE_NAME = "selection"
DEVELOPMENT_RECEIPT_NAME = "development_receipt.json"
SELECTION_RECEIPT_NAME = "selection_receipt.json"
RECOVERIES_DIR_NAME = "recoveries"
REPLAY_LEASE_NAME = "replay_lease.json"

RECOVERED_UNIT_COUNT = 1242
RECOVERED_SELECTOR_RECORDS = 14940

RECOVERED_SOURCE_SUCCESSFUL_FITS = 14904
RECOVERED_SOURCE_ATTEMPTS = 14905
RECOVERED_SOURCE_INTERRUPTED_ATTEMPTS = 1
CLEAN_SOURCE_SUCCESSFUL_FITS = 14904
CLEAN_SOURCE_ATTEMPTS = 14904
CLEAN_SOURCE_INTERRUPTED_ATTEMPTS = 0

INTERRUPTED_OBSERVED_STEPS = 68
INTERRUPTED_CHARGED_STEPS = 800

RECOVERED_REUSED_SUCCESSFUL_STEPS = 1669388
RECOVERED_NEW_FIT_EXECUTIONS = 6184
RECOVERY_MINIMUM_STEPS_PER_NEW_FIT = 120
RECOVERY_MAXIMUM_STEPS_PER_NEW_FIT = 800
RECOVERY_STEPS_GRANULARITY = 4

RECOVERED_MAXIMUM_SOURCE_OPTIMIZER_STEPS = 11924000
CLEAN_MAXIMUM_SOURCE_OPTIMIZER_STEPS = (
    RECOVERED_MAXIMUM_SOURCE_OPTIMIZER_STEPS - INTERRUPTED_CHARGED_STEPS
)
RECOVERED_MAXIMUM_NEW_NEURAL_EXECUTIONS = authority.MAXIMUM_NEW_NEURAL_EXECUTIONS
CLEAN_MAXIMUM_NEW_NEURAL_EXECUTIONS = RECOVERED_MAXIMUM_NEW_NEURAL_EXECUTIONS - 1
RECOVERED_MAXIMUM_NEW_OPTIMIZER_STEPS = authority.MAXIMUM_NEW_OPTIMIZER_STEPS
CLEAN_MAXIMUM_NEW_OPTIMIZER_STEPS = (
    RECOVERED_MAXIMUM_NEW_OPTIMIZER_STEPS - INTERRUPTED_CHARGED_STEPS
)

_RECEIPT_LIMIT_BYTES = 128 * 1024
_PARSE_DEADLINE = sys.float_info.max

_ACCOUNTING_KEYS = frozenset(
    {
        "schema_version",
        "mode",
        "source_successful_fits",
        "source_attempts",
        "source_interrupted_attempts",
        "source_optimizer_steps_successful_exact",
        "source_optimizer_steps_observed_lower_bound",
        "source_optimizer_steps_charged_upper_bound",
        "source_optimizer_steps_all_attempts_exact",
        "maximum_source_optimizer_steps",
        "maximum_new_neural_executions",
        "maximum_new_optimizer_steps",
        "recovery_permit_sha256",
    }
)


class RecoverySourceError(ValueError):
    """Stable, path-free recovery-source resolution/accounting failure."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        message = reason_code if not detail else f"{reason_code}: {detail}"
        super().__init__(message)


def _strict_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecoverySourceError(code)
    return value


def _strict_nonneg_int(value: Any, code: str) -> int:
    number = _strict_int(value, code)
    if number < 0:
        raise RecoverySourceError(code)
    return number


def _reject_chain(path: Path | str, code: str = "path_rejected") -> None:
    """Reject symlinked ancestors and raw dotdot/nul components before use."""

    try:
        recovery_inputs._reject_symlink_chain(path)
    except recovery_inputs.RecoveryInputsError as error:
        raise RecoverySourceError(code) from error


def _kind(path: Path) -> str | None:
    """Classify a single final path without following its own symlink."""

    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RecoverySourceError("path_unreadable") from error
    mode = info.st_mode
    if stat.S_ISLNK(mode):
        raise RecoverySourceError("symlink_path_rejected")
    if stat.S_ISDIR(mode):
        return "dir"
    if stat.S_ISREG(mode):
        return "file"
    raise RecoverySourceError("entry_type_rejected")


def _scan(directory: Path, code: str) -> dict[str, str]:
    """List direct children as ``name -> kind`` rejecting links and specials."""

    _reject_chain(directory, code)
    if _kind(directory) != "dir":
        raise RecoverySourceError(code)
    try:
        with os.scandir(directory) as entries:
            scanned = list(entries)
    except OSError as error:
        raise RecoverySourceError(code) from error
    result: dict[str, str] = {}
    for entry in scanned:
        try:
            name = recovery_inputs._check_component(entry.name)
        except recovery_inputs.RecoveryInputsError as error:
            raise RecoverySourceError("entry_name_rejected") from error
        try:
            if entry.is_symlink():
                raise RecoverySourceError("symlink_path_rejected")
            if entry.is_dir(follow_symlinks=False):
                result[name] = "dir"
            elif entry.is_file(follow_symlinks=False):
                result[name] = "file"
            else:
                raise RecoverySourceError("entry_type_rejected")
        except OSError as error:
            raise RecoverySourceError(code) from error
    return result


def _read_strict_json(path: Path, code: str) -> Any:
    """Bounded strict JSON read: regular non-link file, unique keys, finite."""

    try:
        raw = recovery_inputs._read_bytes_bounded(path, _RECEIPT_LIMIT_BYTES, code, _PARSE_DEADLINE)
        return recovery_inputs._parse_json(raw, code)
    except recovery_inputs.RecoveryInputsError as error:
        raise RecoverySourceError(code) from error


def is_recovered(record: Any) -> bool:
    """Detect a recovered-run record; non-mappings are documented as ``False``.

    A record is recovered when its schema equals the recovery receipt schema
    or when it carries a ``recovery_permit_sha256`` key, so a mislabelled
    record is never silently treated as a legacy clean record later.
    """

    if not isinstance(record, Mapping):
        return False
    if str(record.get("schema_version")) == receipt.SCHEMA_VERSION:
        return True
    return "recovery_permit_sha256" in record


def resolve_paths(bundle: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve the source-development paths for legacy or recovered authority.

    The returned mapping always has the same five keys as the legacy
    comprehensive freeze boundary: ``run_root``, ``develop``, ``receipt``,
    ``selection`` and ``selection_receipt``.  For an approved recovered run the
    ``develop`` path points at the exclusive recovery stage and ``receipt`` at
    the root recovery receipt, while ``selection`` and ``selection_receipt``
    remain under the original run root.

    This routine performs only layout and receipt-schema validation.  It does
    not authenticate manifests, inventories, selector records, plans or model
    artifacts; an approved caller MUST run the existing full acceptance checks
    after resolution.
    """

    if not isinstance(bundle, Mapping):
        raise RecoverySourceError("bundle_malformed")
    if bundle.get("permit_sha256") != authority.BASECOMPREHENSIVE_PERMIT_SHA256:
        raise RecoverySourceError("base_permit_hash_mismatch")
    artifact_root = bundle.get("artifact_root")
    if not isinstance(artifact_root, (str, Path)) or not str(artifact_root):
        raise RecoverySourceError("artifact_root_invalid")
    _reject_chain(artifact_root)

    run_root = (
        Path(artifact_root)
        / COMPREHENSIVE_NAMESPACE
        / "runs"
        / authority.BASECOMPREHENSIVE_PERMIT_SHA256
    )
    _reject_chain(run_root)

    develop = run_root / DEVELOP_STAGE_NAME
    legacy_receipt = run_root / DEVELOPMENT_RECEIPT_NAME
    selection = run_root / SELECTION_STAGE_NAME
    selection_receipt = run_root / SELECTION_RECEIPT_NAME
    recoveries_root = run_root / RECOVERIES_DIR_NAME
    recovery_receipt_path = run_root / receipt.RECEIPT_NAME

    for path in (develop, legacy_receipt, selection, selection_receipt):
        _reject_chain(path)
        _kind(path)

    has_recoveries = _kind(recoveries_root) is not None
    has_recovery_receipt = _kind(recovery_receipt_path) is not None

    if not has_recoveries and not has_recovery_receipt:
        return {
            "run_root": run_root,
            "develop": develop,
            "receipt": legacy_receipt,
            "selection": selection,
            "selection_receipt": selection_receipt,
        }

    if not has_recoveries or not has_recovery_receipt:
        raise RecoverySourceError("recovery_layout_incomplete")
    if _kind(recoveries_root) != "dir":
        raise RecoverySourceError("recoveries_not_directory")
    if _kind(recovery_receipt_path) != "file":
        raise RecoverySourceError("recovery_receipt_not_file")
    if _kind(develop) != "dir":
        raise RecoverySourceError("original_develop_missing")
    if _kind(legacy_receipt) is not None:
        raise RecoverySourceError("dual_complete_authority")

    recoveries_children = _scan(recoveries_root, "recoveries_unreadable")
    if set(recoveries_children) != {authority.RECOVERY_PERMIT_SHA256}:
        raise RecoverySourceError("recoveries_layout_rejected")
    if recoveries_children[authority.RECOVERY_PERMIT_SHA256] != "dir":
        raise RecoverySourceError("recoveries_layout_rejected")

    approved = recoveries_root / authority.RECOVERY_PERMIT_SHA256
    approved_children = _scan(approved, "recovery_stage_unreadable")
    if set(approved_children) != {DEVELOP_STAGE_NAME, REPLAY_LEASE_NAME}:
        raise RecoverySourceError("recovery_stage_layout_rejected")
    if (
        approved_children[DEVELOP_STAGE_NAME] != "dir"
        or approved_children[REPLAY_LEASE_NAME] != "file"
    ):
        raise RecoverySourceError("recovery_stage_layout_rejected")

    recovery_develop = approved / DEVELOP_STAGE_NAME
    if _kind(recovery_develop) != "dir":
        raise RecoverySourceError("recovery_develop_missing")
    _reject_chain(recovery_develop)
    _reject_chain(approved / REPLAY_LEASE_NAME)
    if _kind(recovery_develop / "failure.json") is not None:
        raise RecoverySourceError("recovery_stage_failed")

    value = _read_strict_json(recovery_receipt_path, "recovery_receipt")
    if not isinstance(value, Mapping):
        raise RecoverySourceError("recovery_receipt_malformed")
    if not is_recovered(value):
        raise RecoverySourceError("recovery_receipt_schema_mismatch")
    try:
        receipt.validate_receipt(value, expected_units=RECOVERED_UNIT_COUNT)
    except RecoverySourceError:
        raise
    except Exception as error:
        raise RecoverySourceError("recovery_receipt_rejected") from error

    return {
        "run_root": run_root,
        "develop": recovery_develop,
        "receipt": recovery_receipt_path,
        "selection": selection,
        "selection_receipt": selection_receipt,
    }


def _recovered_accounting(source_optimizer_steps: int) -> dict[str, Any]:
    return {
        "schema_version": ACCOUNTING_SCHEMA_VERSION,
        "mode": RECOVERY_ACCOUNTING_MODE,
        "source_successful_fits": RECOVERED_SOURCE_SUCCESSFUL_FITS,
        "source_attempts": RECOVERED_SOURCE_ATTEMPTS,
        "source_interrupted_attempts": RECOVERED_SOURCE_INTERRUPTED_ATTEMPTS,
        "source_optimizer_steps_successful_exact": source_optimizer_steps,
        "source_optimizer_steps_observed_lower_bound": (
            source_optimizer_steps + INTERRUPTED_OBSERVED_STEPS
        ),
        "source_optimizer_steps_charged_upper_bound": (
            source_optimizer_steps + INTERRUPTED_CHARGED_STEPS
        ),
        "source_optimizer_steps_all_attempts_exact": False,
        "maximum_source_optimizer_steps": RECOVERED_MAXIMUM_SOURCE_OPTIMIZER_STEPS,
        "maximum_new_neural_executions": RECOVERED_MAXIMUM_NEW_NEURAL_EXECUTIONS,
        "maximum_new_optimizer_steps": RECOVERED_MAXIMUM_NEW_OPTIMIZER_STEPS,
        "recovery_permit_sha256": authority.RECOVERY_PERMIT_SHA256,
    }


def _clean_accounting(source_optimizer_steps: int) -> dict[str, Any]:
    return {
        "schema_version": ACCOUNTING_SCHEMA_VERSION,
        "mode": CLEAN_ACCOUNTING_MODE,
        "source_successful_fits": CLEAN_SOURCE_SUCCESSFUL_FITS,
        "source_attempts": CLEAN_SOURCE_ATTEMPTS,
        "source_interrupted_attempts": CLEAN_SOURCE_INTERRUPTED_ATTEMPTS,
        "source_optimizer_steps_successful_exact": source_optimizer_steps,
        "source_optimizer_steps_observed_lower_bound": source_optimizer_steps,
        "source_optimizer_steps_charged_upper_bound": source_optimizer_steps,
        "source_optimizer_steps_all_attempts_exact": True,
        "maximum_source_optimizer_steps": CLEAN_MAXIMUM_SOURCE_OPTIMIZER_STEPS,
        "maximum_new_neural_executions": CLEAN_MAXIMUM_NEW_NEURAL_EXECUTIONS,
        "maximum_new_optimizer_steps": CLEAN_MAXIMUM_NEW_OPTIMIZER_STEPS,
        "recovery_permit_sha256": None,
    }


def _validated_optimizer_steps(validated: Any) -> int:
    source = validated if isinstance(validated, Mapping) else None
    if source is not None and "optimizer_steps" in source:
        return _strict_nonneg_int(
            source.get("optimizer_steps"), "recovered_optimizer_steps_invalid"
        )
    raise RecoverySourceError("recovered_optimizer_steps_missing")


def accounting_from_recovered(
    summary: Mapping[str, Any], receipt_record: Mapping[str, Any]
) -> dict[str, Any]:
    """Return normalized accounting from a validated recovered receipt pair.

    The pair is checked with the recovery receipt validator; the resulting
    accounting is intended for future refit resource budgeting.  This is not a
    proof that the original data files were read.
    """

    try:
        validated = receipt.validate_pair(
            summary, receipt_record, expected_units=RECOVERED_UNIT_COUNT
        )
    except RecoverySourceError:
        raise
    except Exception as error:
        raise RecoverySourceError("recovery_receipt_invalid") from error
    steps = _validated_optimizer_steps(validated)
    return validate_accounting(_recovered_accounting(steps), source_optimizer_steps=steps)


def clean_accounting(source_optimizer_steps: int) -> dict[str, Any]:
    """Return normalized accounting for a legacy clean source run.

    ``source_optimizer_steps`` keeps its legacy meaning of the exact number of
    successful source optimizer steps; synthetic tests may pass zero.
    """

    steps = _strict_nonneg_int(source_optimizer_steps, "source_optimizer_steps_invalid")
    if steps > CLEAN_MAXIMUM_SOURCE_OPTIMIZER_STEPS:
        raise RecoverySourceError("source_optimizer_steps_out_of_range")
    return _clean_accounting(steps)


def validate_accounting(
    accounting: Mapping[str, Any], *, source_optimizer_steps: int
) -> dict[str, Any]:
    """Validate claimed accounting and return an independent shallow copy.

    This helper checks only the claimed accounting value and its pinned
    constants and relationships.  An authority must still derive recovered
    accounting from a validated recovered receipt; no file authentication is
    claimed here, and no enable flag can widen a pinned ceiling.
    """

    if not isinstance(accounting, Mapping):
        raise RecoverySourceError("source_accounting_malformed")
    if set(accounting) != _ACCOUNTING_KEYS:
        raise RecoverySourceError("source_accounting_keys_mismatch")
    if accounting.get("schema_version") != ACCOUNTING_SCHEMA_VERSION:
        raise RecoverySourceError("source_accounting_schema_mismatch")
    mode = accounting.get("mode")
    if mode not in (RECOVERY_ACCOUNTING_MODE, CLEAN_ACCOUNTING_MODE):
        raise RecoverySourceError("source_accounting_mode_invalid")

    argument = _strict_nonneg_int(source_optimizer_steps, "source_optimizer_steps_invalid")
    successful = _strict_nonneg_int(
        accounting.get("source_optimizer_steps_successful_exact"),
        "source_accounting_steps_invalid",
    )
    if successful != argument:
        raise RecoverySourceError("source_accounting_steps_mismatch")

    observed_lower = _strict_nonneg_int(
        accounting.get("source_optimizer_steps_observed_lower_bound"),
        "source_accounting_steps_invalid",
    )
    charged_upper = _strict_nonneg_int(
        accounting.get("source_optimizer_steps_charged_upper_bound"),
        "source_accounting_steps_invalid",
    )
    exact = accounting.get("source_optimizer_steps_all_attempts_exact")
    if type(exact) is not bool:
        raise RecoverySourceError("source_accounting_flag_invalid")

    if mode == RECOVERY_ACCOUNTING_MODE:
        if exact is not False:
            raise RecoverySourceError("source_accounting_flag_invalid")
        if observed_lower != successful + INTERRUPTED_OBSERVED_STEPS:
            raise RecoverySourceError("source_accounting_lower_bound_mismatch")
        if charged_upper != successful + INTERRUPTED_CHARGED_STEPS:
            raise RecoverySourceError("source_accounting_upper_bound_mismatch")
        expected_fits = RECOVERED_SOURCE_SUCCESSFUL_FITS
        expected_attempts = RECOVERED_SOURCE_ATTEMPTS
        expected_interrupted = RECOVERED_SOURCE_INTERRUPTED_ATTEMPTS
        expected_max_source = RECOVERED_MAXIMUM_SOURCE_OPTIMIZER_STEPS
        expected_max_new = RECOVERED_MAXIMUM_NEW_NEURAL_EXECUTIONS
        expected_max_opt = RECOVERED_MAXIMUM_NEW_OPTIMIZER_STEPS
        lower = (
            RECOVERED_REUSED_SUCCESSFUL_STEPS
            + RECOVERED_NEW_FIT_EXECUTIONS * RECOVERY_MINIMUM_STEPS_PER_NEW_FIT
        )
        upper = (
            RECOVERED_REUSED_SUCCESSFUL_STEPS
            + RECOVERED_NEW_FIT_EXECUTIONS * RECOVERY_MAXIMUM_STEPS_PER_NEW_FIT
        )
        if successful < lower or successful > upper:
            raise RecoverySourceError("source_accounting_steps_out_of_range")
        if successful % RECOVERY_STEPS_GRANULARITY != 0:
            raise RecoverySourceError("source_accounting_steps_out_of_range")
    else:
        if exact is not True:
            raise RecoverySourceError("source_accounting_flag_invalid")
        if observed_lower != successful or charged_upper != successful:
            raise RecoverySourceError("source_accounting_bound_mismatch")
        expected_fits = CLEAN_SOURCE_SUCCESSFUL_FITS
        expected_attempts = CLEAN_SOURCE_ATTEMPTS
        expected_interrupted = CLEAN_SOURCE_INTERRUPTED_ATTEMPTS
        expected_max_source = CLEAN_MAXIMUM_SOURCE_OPTIMIZER_STEPS
        expected_max_new = CLEAN_MAXIMUM_NEW_NEURAL_EXECUTIONS
        expected_max_opt = CLEAN_MAXIMUM_NEW_OPTIMIZER_STEPS
        if successful > CLEAN_MAXIMUM_SOURCE_OPTIMIZER_STEPS:
            raise RecoverySourceError("source_accounting_steps_out_of_range")

    for key, expected in (
        ("source_successful_fits", expected_fits),
        ("source_attempts", expected_attempts),
        ("source_interrupted_attempts", expected_interrupted),
        ("maximum_source_optimizer_steps", expected_max_source),
        ("maximum_new_neural_executions", expected_max_new),
        ("maximum_new_optimizer_steps", expected_max_opt),
    ):
        if _strict_nonneg_int(accounting.get(key), "source_accounting_count_invalid") != expected:
            raise RecoverySourceError("source_accounting_count_mismatch")

    permit = accounting.get("recovery_permit_sha256")
    if mode == RECOVERY_ACCOUNTING_MODE:
        if not isinstance(permit, str) or permit != authority.RECOVERY_PERMIT_SHA256:
            raise RecoverySourceError("source_accounting_permit_mismatch")
    elif permit is not None:
        raise RecoverySourceError("source_accounting_permit_mismatch")

    return dict(accounting)


def from_authenticated(auth: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize accounting from an authenticated stage record.

    When ``source_execution_accounting`` is absent the record is a legacy clean
    run and clean accounting is returned.  When the key is present it MUST be
    valid explicit accounting with the same successful step count; an invalid
    value is rejected and never silently downgraded to clean.
    """

    if not isinstance(auth, Mapping):
        raise RecoverySourceError("authentication_malformed")
    steps = _strict_nonneg_int(auth.get("source_optimizer_steps"), "source_optimizer_steps_invalid")
    if "source_execution_accounting" not in auth:
        return clean_accounting(steps)
    return validate_accounting(auth["source_execution_accounting"], source_optimizer_steps=steps)
