"""Pure-stdlib recovery completion accounting contract for the P05 recovery run.

This module builds and validates the explicit, honest recovery development
receipt for the owner-approved P05 recovery.  It never forges a clean-run
receipt: the original interrupted attempt is always counted as exactly one
failed attempt and the interrupted optimizer-update count is kept as an
observed lower bound rather than promoted to an exact count.  Successful fits
remain exactly counted and are kept separate from the charged attempt upper
bound used for resource accounting.

The module performs no IO, imports no numerical library and never imports torch
or numpy.  It re-uses only the pinned public authority and comprehensive-input
constants.  Callers may construct dictionaries directly; every public validator
re-checks the complete structure independently of the builders.
These pure checks do not authenticate artifact files or reconstruct the recovery
plan. The consuming stage must perform those checks before accepting a receipt.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from atlas_sers.evaluation import p05_comprehensive_inputs as comprehensive_inputs
from atlas_sers.evaluation import p05_recovery_authority as authority

__all__ = [
    "RecoveryReceiptError",
    "SCHEMA_VERSION",
    "RECEIPT_NAME",
    "CLAIM",
    "COMMAND",
    "STAGE_NAME",
    "OPTIMIZER_STEPS_SCOPE",
    "PROTOCOL_VERSION",
    "BASE_PERMIT_SHA256",
    "RECOVERY_PERMIT_SHA256",
    "ORIGINAL_EVIDENCE_ANCHOR_SHA256",
    "CORE_CONTRACT_SHA256",
    "CORE_PLAN_ID",
    "LEDGER_ID",
    "REQUIRED_STARTED",
    "REQUIRED_COMPLETED",
    "REQUIRED_FAILED",
    "ORIGINAL_INTERRUPTED_ATTEMPTS",
    "REQUIRED_NEW_COMPLETIONS",
    "REQUIRED_REUSED_PILOT_SLOTS",
    "REQUIRED_REUSED_ORIGINAL_COMPLETIONS",
    "REQUIRED_RECOVERY_STARTED",
    "REQUIRED_RECOVERY_COMPLETED",
    "REQUIRED_RECOVERY_FAILED",
    "REQUIRED_REPLAY_ATTEMPTS",
    "REQUIRED_ORIGINALLY_UNSTARTED_ATTEMPTS",
    "REQUIRED_SELECTOR_RECORDS",
    "REQUIRED_UNITS_TOTAL",
    "DEFAULT_EXPECTED_UNITS",
    "REUSED_ORIGINAL_OPTIMIZER_STEPS",
    "MINIMUM_RECOVERY_OPTIMIZER_STEPS",
    "MAXIMUM_RECOVERY_OPTIMIZER_STEPS",
    "ORIGINAL_INTERRUPTED_OPTIMIZER_STEPS_OBSERVED_LOWER_BOUND",
    "ORIGINAL_INTERRUPTED_OPTIMIZER_STEPS_CHARGED_UPPER_BOUND",
    "MAXIMUM_OPTIMIZER_STEPS",
    "MAXIMUM_NEW_NEURAL_EXECUTIONS",
    "MAXIMUM_NEW_OPTIMIZER_STEPS",
    "PRIOR_SCIENTIFIC_SECONDS_CHARGED_UPPER_BOUND",
    "PRELAUNCH_AUDIT_RESERVE_SECONDS",
    "MAXIMUM_TOTAL_SECONDS",
    "STORAGE_CEILING_BYTES",
    "MAXIMUM_CUDA_ALLOCATED_BYTES",
    "FROZEN_SOURCE_FIT_CUDA_CAP_BYTES",
    "build_summary",
    "build_receipt",
    "validate_summary",
    "validate_receipt",
    "validate_pair",
]

SCHEMA_VERSION = "nato-sers-p05-recovered-development-v1"
RECEIPT_NAME = "recovery_development_receipt.json"
CLAIM = "complete_recovered_source_development_no_selection_no_refit_no_outer_evaluation"
COMMAND = "run_recovery_development"
STAGE_NAME = "develop"
OPTIMIZER_STEPS_SCOPE = "completed_nonpilot_source_fits_only"

PROTOCOL_VERSION = authority.RECOVERY_PROTOCOL_VERSION

BASE_PERMIT_SHA256 = authority.BASECOMPREHENSIVE_PERMIT_SHA256
RECOVERY_PERMIT_SHA256 = authority.RECOVERY_PERMIT_SHA256
ORIGINAL_EVIDENCE_ANCHOR_SHA256 = authority.ORIGINAL_EVIDENCE_ANCHOR_SHA256
CORE_CONTRACT_SHA256 = comprehensive_inputs.CORE_CONTRACT_SHA256
CORE_PLAN_ID = comprehensive_inputs.CORE_PLAN_ID
LEDGER_ID = comprehensive_inputs.LEDGER_ID

REQUIRED_STARTED = 14905
REQUIRED_COMPLETED = 14904
REQUIRED_FAILED = 1
ORIGINAL_INTERRUPTED_ATTEMPTS = 1
REQUIRED_NEW_COMPLETIONS = 14904
REQUIRED_REUSED_PILOT_SLOTS = 36
REQUIRED_REUSED_ORIGINAL_COMPLETIONS = 8720
REQUIRED_RECOVERY_STARTED = 6184
REQUIRED_RECOVERY_COMPLETED = 6184
REQUIRED_RECOVERY_FAILED = 0
REQUIRED_REPLAY_ATTEMPTS = 1
REQUIRED_ORIGINALLY_UNSTARTED_ATTEMPTS = 6183
REQUIRED_SELECTOR_RECORDS = 14940
REQUIRED_UNITS_TOTAL = 1245
DEFAULT_EXPECTED_UNITS = 1242

REUSED_ORIGINAL_OPTIMIZER_STEPS = 1669388
MINIMUM_UPDATES_PER_RECOVERY_FIT = 120
MAXIMUM_UPDATES_PER_RECOVERY_FIT = 800
MINIMUM_RECOVERY_OPTIMIZER_STEPS = REQUIRED_RECOVERY_COMPLETED * MINIMUM_UPDATES_PER_RECOVERY_FIT
MAXIMUM_RECOVERY_OPTIMIZER_STEPS = REQUIRED_RECOVERY_COMPLETED * MAXIMUM_UPDATES_PER_RECOVERY_FIT
OPTIMIZER_STEPS_DIVISOR = 4
ORIGINAL_INTERRUPTED_OPTIMIZER_STEPS_OBSERVED_LOWER_BOUND = 68
ORIGINAL_INTERRUPTED_OPTIMIZER_STEPS_CHARGED_UPPER_BOUND = 800
MAXIMUM_SOURCE_ATTEMPTS = 14905
MAXIMUM_UPDATES_PER_SOURCE_ATTEMPT = 800
MAXIMUM_OPTIMIZER_STEPS = MAXIMUM_SOURCE_ATTEMPTS * MAXIMUM_UPDATES_PER_SOURCE_ATTEMPT

MAXIMUM_NEW_NEURAL_EXECUTIONS = authority.MAXIMUM_NEW_NEURAL_EXECUTIONS
MAXIMUM_NEW_OPTIMIZER_STEPS = authority.MAXIMUM_NEW_OPTIMIZER_STEPS

PRIOR_SCIENTIFIC_SECONDS_CHARGED_UPPER_BOUND = authority.PRIOR_SCIENTIFIC_SECONDS_CHARGED
PRELAUNCH_AUDIT_RESERVE_SECONDS = authority.PRELAUNCH_AUDIT_RESERVE_SECONDS
MAXIMUM_TOTAL_SECONDS = authority.MAXIMUM_TOTAL_SECONDS
STORAGE_CEILING_BYTES = authority.PRIVATE_STORAGE_CEILING_BYTES
MAXIMUM_CUDA_ALLOCATED_BYTES = authority.MAXIMUM_CUDA_ALLOCATED_BYTES
FROZEN_SOURCE_FIT_CUDA_CAP_BYTES = authority.FROZEN_SOURCE_FIT_CUDA_CAP_BYTES

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")

_COUNTER_KEYS = frozenset(
    {
        "new_started",
        "new_completed",
        "new_failed",
        "new_optimizer_steps",
        "new_optimizer_steps_exact",
        "new_elapsed_seconds",
        "new_peak_cuda_bytes",
        "reused_completed",
        "reused_optimizer_steps",
        "replay_started",
        "unstarted_started",
    }
)

_SUMMARY_KEYS = (
    "schema_version",
    "protocol_version",
    "claim",
    "command",
    "status",
    "device",
    "permit_sha256",
    "recovery_permit_sha256",
    "original_evidence_anchor_sha256",
    "recovery_plan_id",
    "core_contract_sha256",
    "core_plan_id",
    "ledger_id",
    "started",
    "completed",
    "failed",
    "original_interrupted_attempts",
    "new_completions",
    "reused_pilot_slots",
    "reused_original_completions",
    "recovery_started",
    "recovery_completed",
    "recovery_failed",
    "replay_attempts",
    "originally_unstarted_attempts",
    "selector_records",
    "units_total",
    "units_completed",
    "optimizer_steps",
    "optimizer_steps_exact",
    "optimizer_steps_scope",
    "reused_original_optimizer_steps_exact",
    "recovery_optimizer_steps_exact",
    "original_interrupted_optimizer_steps_observed_lower_bound",
    "original_interrupted_optimizer_steps_charged_upper_bound",
    "total_source_optimizer_steps_lower_bound",
    "total_source_optimizer_steps_charged_upper_bound",
    "total_source_optimizer_steps_exact",
    "maximum_optimizer_steps",
    "maximum_new_neural_executions",
    "maximum_new_optimizer_steps",
    "scientific_seconds_this_stage",
    "prior_scientific_seconds_charged_upper_bound",
    "prelaunch_audit_reserve_seconds",
    "scientific_seconds_cumulative_bound",
    "maximum_total_seconds",
    "live_bytes",
    "sum_elapsed_seconds",
    "maximum_peak_cuda_bytes",
    "storage_ceiling_bytes",
    "maximum_cuda_allocated_bytes",
    "frozen_source_fit_cuda_cap_bytes",
    "source_fits_only",
    "selection_authorized",
    "refit_authorized",
    "calibration_authorized",
    "outer_evaluation_authorized",
)

_RECEIPT_KEYS = frozenset(_SUMMARY_KEYS) | {"stage", "stage_manifest_sha256"}
_TIME_VARYING_KEYS = frozenset(
    {"scientific_seconds_this_stage", "scientific_seconds_cumulative_bound"}
)

_FIXED_FIELDS: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "protocol_version": PROTOCOL_VERSION,
    "claim": CLAIM,
    "command": COMMAND,
    "status": "complete",
    "device": "cuda",
    "permit_sha256": BASE_PERMIT_SHA256,
    "recovery_permit_sha256": RECOVERY_PERMIT_SHA256,
    "original_evidence_anchor_sha256": ORIGINAL_EVIDENCE_ANCHOR_SHA256,
    "core_contract_sha256": CORE_CONTRACT_SHA256,
    "core_plan_id": CORE_PLAN_ID,
    "ledger_id": LEDGER_ID,
    "started": REQUIRED_STARTED,
    "completed": REQUIRED_COMPLETED,
    "failed": REQUIRED_FAILED,
    "original_interrupted_attempts": ORIGINAL_INTERRUPTED_ATTEMPTS,
    "new_completions": REQUIRED_NEW_COMPLETIONS,
    "reused_pilot_slots": REQUIRED_REUSED_PILOT_SLOTS,
    "reused_original_completions": REQUIRED_REUSED_ORIGINAL_COMPLETIONS,
    "recovery_started": REQUIRED_RECOVERY_STARTED,
    "recovery_completed": REQUIRED_RECOVERY_COMPLETED,
    "recovery_failed": REQUIRED_RECOVERY_FAILED,
    "replay_attempts": REQUIRED_REPLAY_ATTEMPTS,
    "originally_unstarted_attempts": REQUIRED_ORIGINALLY_UNSTARTED_ATTEMPTS,
    "selector_records": REQUIRED_SELECTOR_RECORDS,
    "units_total": REQUIRED_UNITS_TOTAL,
    "optimizer_steps_exact": True,
    "optimizer_steps_scope": OPTIMIZER_STEPS_SCOPE,
    "reused_original_optimizer_steps_exact": REUSED_ORIGINAL_OPTIMIZER_STEPS,
    "original_interrupted_optimizer_steps_observed_lower_bound": (
        ORIGINAL_INTERRUPTED_OPTIMIZER_STEPS_OBSERVED_LOWER_BOUND
    ),
    "original_interrupted_optimizer_steps_charged_upper_bound": (
        ORIGINAL_INTERRUPTED_OPTIMIZER_STEPS_CHARGED_UPPER_BOUND
    ),
    "total_source_optimizer_steps_exact": False,
    "maximum_optimizer_steps": MAXIMUM_OPTIMIZER_STEPS,
    "maximum_new_neural_executions": MAXIMUM_NEW_NEURAL_EXECUTIONS,
    "maximum_new_optimizer_steps": MAXIMUM_NEW_OPTIMIZER_STEPS,
    "prior_scientific_seconds_charged_upper_bound": (PRIOR_SCIENTIFIC_SECONDS_CHARGED_UPPER_BOUND),
    "prelaunch_audit_reserve_seconds": PRELAUNCH_AUDIT_RESERVE_SECONDS,
    "maximum_total_seconds": MAXIMUM_TOTAL_SECONDS,
    "storage_ceiling_bytes": STORAGE_CEILING_BYTES,
    "maximum_cuda_allocated_bytes": MAXIMUM_CUDA_ALLOCATED_BYTES,
    "frozen_source_fit_cuda_cap_bytes": FROZEN_SOURCE_FIT_CUDA_CAP_BYTES,
    "source_fits_only": True,
    "selection_authorized": False,
    "refit_authorized": False,
    "calibration_authorized": False,
    "outer_evaluation_authorized": False,
}


class RecoveryReceiptError(ValueError):
    """Stable, path-free recovery-receipt failure."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        message = reason_code if not detail else f"{reason_code}: {detail}"
        super().__init__(message)


def _require_mapping(value: Any, code: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise RecoveryReceiptError(code)
    return value


def _require_exact_keys(payload: Mapping, keys: Any, code: str) -> None:
    if set(payload) != set(keys):
        raise RecoveryReceiptError(code)


def _sha256(value: Any, code: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise RecoveryReceiptError(code)
    return value


def _exact_int(value: Any, expected: int, code: str) -> int:
    if type(value) is not int or value != expected:
        raise RecoveryReceiptError(code)
    return value


def _bounded_int(value: Any, low: int, high: int, code: str) -> int:
    if type(value) is not int or value < low or value > high:
        raise RecoveryReceiptError(code)
    return value


def _finite_nonneg(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoveryReceiptError(code)
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise RecoveryReceiptError(code) from error
    if not math.isfinite(result) or result < 0.0:
        raise RecoveryReceiptError(code)
    return result


def _identity_from_bundle(base_bundle: Any) -> None:
    bundle = _require_mapping(base_bundle, "base_bundle_not_mapping")
    if bundle.get("permit_sha256") != BASE_PERMIT_SHA256:
        raise RecoveryReceiptError("base_permit_mismatch")
    contract = bundle.get("core_contract_sha256", bundle.get("contract_sha256"))
    if contract != CORE_CONTRACT_SHA256:
        raise RecoveryReceiptError("base_core_contract_mismatch")
    for key in ("contract_sha256", "core_contract_sha256"):
        if key in bundle and bundle[key] != CORE_CONTRACT_SHA256:
            raise RecoveryReceiptError("base_core_contract_mismatch")
    if bundle.get("core_plan_id") != CORE_PLAN_ID:
        raise RecoveryReceiptError("base_core_plan_mismatch")
    ledger_id = bundle.get("ledger_id")
    if ledger_id is None:
        ledger = bundle.get("ledger")
        if isinstance(ledger, Mapping):
            ledger_id = ledger.get("ledger_id")
    if ledger_id != LEDGER_ID:
        raise RecoveryReceiptError("base_ledger_mismatch")
    if "ledger" in bundle and (
        not isinstance(bundle["ledger"], Mapping) or bundle["ledger"].get("ledger_id") != LEDGER_ID
    ):
        raise RecoveryReceiptError("base_ledger_mismatch")


def _validate_fixed(payload: Mapping) -> None:
    for field, expected in _FIXED_FIELDS.items():
        value = payload[field]
        if type(expected) is bool:
            if value is not expected:
                raise RecoveryReceiptError(f"field_{field}_invalid")
        elif type(expected) is int:
            if type(value) is not int or value != expected:
                raise RecoveryReceiptError(f"field_{field}_invalid")
        elif value != expected:
            raise RecoveryReceiptError(f"field_{field}_invalid")


def _validate_counts_consistency(payload: Mapping) -> None:
    started = payload["started"]
    completed = payload["completed"]
    failed = payload["failed"]
    if started != completed + failed:
        raise RecoveryReceiptError("counts_started_inconsistent")
    if completed != payload["reused_original_completions"] + payload["recovery_completed"]:
        raise RecoveryReceiptError("counts_completed_inconsistent")
    if payload["new_completions"] != completed:
        raise RecoveryReceiptError("counts_new_completions_inconsistent")
    if payload["original_interrupted_attempts"] != failed:
        raise RecoveryReceiptError("counts_interruption_inconsistent")
    if (
        payload["recovery_started"]
        != payload["replay_attempts"] + payload["originally_unstarted_attempts"]
    ):
        raise RecoveryReceiptError("counts_recovery_split_inconsistent")
    if payload["recovery_completed"] != payload["recovery_started"] - payload["recovery_failed"]:
        raise RecoveryReceiptError("counts_recovery_completion_inconsistent")


def _validate_optimizer(payload: Mapping) -> None:
    reused = payload["reused_original_optimizer_steps_exact"]
    if type(reused) is not int or reused != REUSED_ORIGINAL_OPTIMIZER_STEPS:
        raise RecoveryReceiptError("reused_optimizer_steps_invalid")
    recovery = payload["recovery_optimizer_steps_exact"]
    if (
        type(recovery) is not int
        or recovery < MINIMUM_RECOVERY_OPTIMIZER_STEPS
        or recovery > MAXIMUM_RECOVERY_OPTIMIZER_STEPS
    ):
        raise RecoveryReceiptError("recovery_optimizer_steps_invalid")
    if recovery % OPTIMIZER_STEPS_DIVISOR != 0:
        raise RecoveryReceiptError("recovery_optimizer_steps_not_multiple_of_four")
    optimizer_steps = payload["optimizer_steps"]
    if type(optimizer_steps) is not int or optimizer_steps != reused + recovery:
        raise RecoveryReceiptError("optimizer_steps_invalid")
    if optimizer_steps > MAXIMUM_OPTIMIZER_STEPS:
        raise RecoveryReceiptError("optimizer_steps_ceiling_exceeded")
    lower = payload["total_source_optimizer_steps_lower_bound"]
    if (
        type(lower) is not int
        or lower != optimizer_steps + ORIGINAL_INTERRUPTED_OPTIMIZER_STEPS_OBSERVED_LOWER_BOUND
    ):
        raise RecoveryReceiptError("total_source_optimizer_steps_lower_bound_invalid")
    upper = payload["total_source_optimizer_steps_charged_upper_bound"]
    if (
        type(upper) is not int
        or upper != optimizer_steps + ORIGINAL_INTERRUPTED_OPTIMIZER_STEPS_CHARGED_UPPER_BOUND
    ):
        raise RecoveryReceiptError("total_source_optimizer_steps_charged_upper_bound_invalid")
    if upper <= lower:
        raise RecoveryReceiptError("total_source_optimizer_steps_ordering_invalid")
    if upper > MAXIMUM_NEW_OPTIMIZER_STEPS:
        raise RecoveryReceiptError("total_source_optimizer_steps_ceiling_exceeded")


def _validate_time(payload: Mapping) -> None:
    this_stage = _finite_nonneg(
        payload["scientific_seconds_this_stage"], "scientific_seconds_this_stage_invalid"
    )
    cumulative = _finite_nonneg(
        payload["scientific_seconds_cumulative_bound"],
        "scientific_seconds_cumulative_bound_invalid",
    )
    expected = (
        float(PRIOR_SCIENTIFIC_SECONDS_CHARGED_UPPER_BOUND + PRELAUNCH_AUDIT_RESERVE_SECONDS)
        + this_stage
    )
    if cumulative != expected:
        raise RecoveryReceiptError("scientific_seconds_cumulative_mismatch")
    if cumulative > MAXIMUM_TOTAL_SECONDS:
        raise RecoveryReceiptError("scientific_seconds_ceiling_exceeded")


def _validate_resources(payload: Mapping) -> None:
    live = payload["live_bytes"]
    if type(live) is not int or live < 0 or live > STORAGE_CEILING_BYTES:
        raise RecoveryReceiptError("live_bytes_invalid")
    elapsed = _finite_nonneg(payload["sum_elapsed_seconds"], "sum_elapsed_seconds_invalid")
    if elapsed > payload["scientific_seconds_this_stage"]:
        raise RecoveryReceiptError("fit_time_exceeds_stage_time")
    peak = payload["maximum_peak_cuda_bytes"]
    if type(peak) is not int or peak < 0 or peak > FROZEN_SOURCE_FIT_CUDA_CAP_BYTES:
        raise RecoveryReceiptError("maximum_peak_cuda_bytes_invalid")


def _validate_units(payload: Mapping, expected_units: int) -> None:
    if type(expected_units) is not int or expected_units != DEFAULT_EXPECTED_UNITS:
        raise RecoveryReceiptError("expected_units_invalid")
    value = payload["units_completed"]
    if type(value) is not int or value != expected_units:
        raise RecoveryReceiptError("units_completed_invalid")
    if value > payload["units_total"]:
        raise RecoveryReceiptError("units_completed_exceeds_total")


def build_summary(
    *,
    base_bundle: Mapping[str, Any],
    recovery_plan_id: str,
    counters: Mapping[str, Any],
    units_completed: int,
    selector_records: int,
    scientific_seconds: float,
    live_bytes: int,
) -> dict[str, Any]:
    """Build the recovery development summary from planned-runner counters."""

    _identity_from_bundle(base_bundle)
    plan_id = _sha256(recovery_plan_id, "recovery_plan_id_invalid")
    if not isinstance(counters, Mapping):
        raise RecoveryReceiptError("counters_not_mapping")
    if set(counters) != _COUNTER_KEYS:
        raise RecoveryReceiptError("counters_keys_mismatch")

    new_started = _exact_int(
        counters["new_started"], REQUIRED_RECOVERY_STARTED, "counter_new_started_invalid"
    )
    new_completed = _exact_int(
        counters["new_completed"], REQUIRED_RECOVERY_COMPLETED, "counter_new_completed_invalid"
    )
    new_failed = _exact_int(
        counters["new_failed"], REQUIRED_RECOVERY_FAILED, "counter_new_failed_invalid"
    )
    replay_started = _exact_int(
        counters["replay_started"], REQUIRED_REPLAY_ATTEMPTS, "counter_replay_started_invalid"
    )
    unstarted_started = _exact_int(
        counters["unstarted_started"],
        REQUIRED_ORIGINALLY_UNSTARTED_ATTEMPTS,
        "counter_unstarted_started_invalid",
    )
    reused_completed = _exact_int(
        counters["reused_completed"],
        REQUIRED_REUSED_ORIGINAL_COMPLETIONS,
        "counter_reused_completed_invalid",
    )
    reused_optimizer_steps = _exact_int(
        counters["reused_optimizer_steps"],
        REUSED_ORIGINAL_OPTIMIZER_STEPS,
        "counter_reused_optimizer_steps_invalid",
    )
    if counters["new_optimizer_steps_exact"] is not True:
        raise RecoveryReceiptError("counter_new_optimizer_steps_exact_invalid")
    new_optimizer_steps = _bounded_int(
        counters["new_optimizer_steps"],
        MINIMUM_RECOVERY_OPTIMIZER_STEPS,
        MAXIMUM_RECOVERY_OPTIMIZER_STEPS,
        "counter_new_optimizer_steps_invalid",
    )
    if new_optimizer_steps % OPTIMIZER_STEPS_DIVISOR != 0:
        raise RecoveryReceiptError("counter_new_optimizer_steps_not_multiple_of_four")
    new_elapsed_seconds = _finite_nonneg(
        counters["new_elapsed_seconds"], "counter_new_elapsed_seconds_invalid"
    )
    new_peak_cuda_bytes = _bounded_int(
        counters["new_peak_cuda_bytes"],
        0,
        FROZEN_SOURCE_FIT_CUDA_CAP_BYTES,
        "counter_new_peak_cuda_bytes_invalid",
    )
    if new_started != replay_started + unstarted_started:
        raise RecoveryReceiptError("counter_recovery_split_invalid")
    if new_completed != new_started - new_failed:
        raise RecoveryReceiptError("counter_recovery_completion_invalid")

    if type(units_completed) is not int or units_completed != DEFAULT_EXPECTED_UNITS:
        raise RecoveryReceiptError("units_completed_invalid")
    if type(selector_records) is not int or selector_records != REQUIRED_SELECTOR_RECORDS:
        raise RecoveryReceiptError("selector_records_invalid")
    stage_seconds = _finite_nonneg(scientific_seconds, "scientific_seconds_invalid")
    if type(live_bytes) is not int or live_bytes < 0 or live_bytes > STORAGE_CEILING_BYTES:
        raise RecoveryReceiptError("live_bytes_invalid")
    cumulative = (
        float(PRIOR_SCIENTIFIC_SECONDS_CHARGED_UPPER_BOUND + PRELAUNCH_AUDIT_RESERVE_SECONDS)
        + stage_seconds
    )
    if cumulative > MAXIMUM_TOTAL_SECONDS:
        raise RecoveryReceiptError("scientific_seconds_ceiling_exceeded")

    optimizer_steps = reused_optimizer_steps + new_optimizer_steps
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "claim": CLAIM,
        "command": COMMAND,
        "status": "complete",
        "device": "cuda",
        "permit_sha256": BASE_PERMIT_SHA256,
        "recovery_permit_sha256": RECOVERY_PERMIT_SHA256,
        "original_evidence_anchor_sha256": ORIGINAL_EVIDENCE_ANCHOR_SHA256,
        "recovery_plan_id": plan_id,
        "core_contract_sha256": CORE_CONTRACT_SHA256,
        "core_plan_id": CORE_PLAN_ID,
        "ledger_id": LEDGER_ID,
        "started": REQUIRED_STARTED,
        "completed": REQUIRED_COMPLETED,
        "failed": REQUIRED_FAILED,
        "original_interrupted_attempts": ORIGINAL_INTERRUPTED_ATTEMPTS,
        "new_completions": REQUIRED_NEW_COMPLETIONS,
        "reused_pilot_slots": REQUIRED_REUSED_PILOT_SLOTS,
        "reused_original_completions": reused_completed,
        "recovery_started": new_started,
        "recovery_completed": new_completed,
        "recovery_failed": new_failed,
        "replay_attempts": replay_started,
        "originally_unstarted_attempts": unstarted_started,
        "selector_records": REQUIRED_SELECTOR_RECORDS,
        "units_total": REQUIRED_UNITS_TOTAL,
        "units_completed": units_completed,
        "optimizer_steps": optimizer_steps,
        "optimizer_steps_exact": True,
        "optimizer_steps_scope": OPTIMIZER_STEPS_SCOPE,
        "reused_original_optimizer_steps_exact": reused_optimizer_steps,
        "recovery_optimizer_steps_exact": new_optimizer_steps,
        "original_interrupted_optimizer_steps_observed_lower_bound": (
            ORIGINAL_INTERRUPTED_OPTIMIZER_STEPS_OBSERVED_LOWER_BOUND
        ),
        "original_interrupted_optimizer_steps_charged_upper_bound": (
            ORIGINAL_INTERRUPTED_OPTIMIZER_STEPS_CHARGED_UPPER_BOUND
        ),
        "total_source_optimizer_steps_lower_bound": (
            optimizer_steps + ORIGINAL_INTERRUPTED_OPTIMIZER_STEPS_OBSERVED_LOWER_BOUND
        ),
        "total_source_optimizer_steps_charged_upper_bound": (
            optimizer_steps + ORIGINAL_INTERRUPTED_OPTIMIZER_STEPS_CHARGED_UPPER_BOUND
        ),
        "total_source_optimizer_steps_exact": False,
        "maximum_optimizer_steps": MAXIMUM_OPTIMIZER_STEPS,
        "maximum_new_neural_executions": MAXIMUM_NEW_NEURAL_EXECUTIONS,
        "maximum_new_optimizer_steps": MAXIMUM_NEW_OPTIMIZER_STEPS,
        "scientific_seconds_this_stage": stage_seconds,
        "prior_scientific_seconds_charged_upper_bound": (
            PRIOR_SCIENTIFIC_SECONDS_CHARGED_UPPER_BOUND
        ),
        "prelaunch_audit_reserve_seconds": PRELAUNCH_AUDIT_RESERVE_SECONDS,
        "scientific_seconds_cumulative_bound": cumulative,
        "maximum_total_seconds": MAXIMUM_TOTAL_SECONDS,
        "live_bytes": live_bytes,
        "sum_elapsed_seconds": new_elapsed_seconds,
        "maximum_peak_cuda_bytes": new_peak_cuda_bytes,
        "storage_ceiling_bytes": STORAGE_CEILING_BYTES,
        "maximum_cuda_allocated_bytes": MAXIMUM_CUDA_ALLOCATED_BYTES,
        "frozen_source_fit_cuda_cap_bytes": FROZEN_SOURCE_FIT_CUDA_CAP_BYTES,
        "source_fits_only": True,
        "selection_authorized": False,
        "refit_authorized": False,
        "calibration_authorized": False,
        "outer_evaluation_authorized": False,
    }
    return validate_summary(payload)


def build_receipt(
    *,
    summary: Mapping[str, Any],
    stage_manifest_sha256: str,
    scientific_seconds: float,
) -> dict[str, Any]:
    """Build the recovery development receipt from a validated summary."""

    validated = validate_summary(summary)
    manifest = _sha256(stage_manifest_sha256, "stage_manifest_sha256_invalid")
    receipt_seconds = _finite_nonneg(scientific_seconds, "receipt_scientific_seconds_invalid")
    if receipt_seconds < validated["scientific_seconds_this_stage"]:
        raise RecoveryReceiptError("receipt_seconds_before_summary")
    receipt = dict(validated)
    receipt["stage"] = STAGE_NAME
    receipt["stage_manifest_sha256"] = manifest
    receipt["scientific_seconds_this_stage"] = receipt_seconds
    receipt["scientific_seconds_cumulative_bound"] = (
        float(PRIOR_SCIENTIFIC_SECONDS_CHARGED_UPPER_BOUND + PRELAUNCH_AUDIT_RESERVE_SECONDS)
        + receipt_seconds
    )
    return validate_receipt(receipt)


def validate_summary(
    summary: Mapping[str, Any], expected_units: int = DEFAULT_EXPECTED_UNITS
) -> dict[str, Any]:
    """Validate a recovery development summary and return an independent copy."""

    payload = _require_mapping(summary, "summary_not_mapping")
    _require_exact_keys(payload, _SUMMARY_KEYS, "summary_keys_mismatch")
    _validate_fixed(payload)
    _sha256(payload["recovery_plan_id"], "recovery_plan_id_invalid")
    _validate_counts_consistency(payload)
    _validate_optimizer(payload)
    _validate_time(payload)
    _validate_resources(payload)
    _validate_units(payload, expected_units)
    return dict(payload)


def validate_receipt(
    receipt: Mapping[str, Any], expected_units: int = DEFAULT_EXPECTED_UNITS
) -> dict[str, Any]:
    """Validate a recovery development receipt and return an independent copy."""

    payload = _require_mapping(receipt, "receipt_not_mapping")
    _require_exact_keys(payload, _RECEIPT_KEYS, "receipt_keys_mismatch")
    if payload["stage"] != STAGE_NAME:
        raise RecoveryReceiptError("receipt_stage_mismatch")
    _sha256(payload["stage_manifest_sha256"], "stage_manifest_sha256_invalid")
    summary_part = {key: payload[key] for key in _SUMMARY_KEYS}
    validated = validate_summary(summary_part, expected_units)
    result = dict(validated)
    result["stage"] = STAGE_NAME
    result["stage_manifest_sha256"] = payload["stage_manifest_sha256"]
    return result


def validate_pair(
    summary: Mapping[str, Any],
    receipt: Mapping[str, Any],
    expected_units: int = DEFAULT_EXPECTED_UNITS,
) -> dict[str, Any]:
    """Validate that a summary and receipt agree and return the downstream cost payload."""

    validated_summary = validate_summary(summary, expected_units)
    validated_receipt = validate_receipt(receipt, expected_units)
    for key in _SUMMARY_KEYS:
        if key in _TIME_VARYING_KEYS:
            continue
        if validated_summary[key] != validated_receipt[key]:
            raise RecoveryReceiptError("pair_field_mismatch")
    if (
        validated_receipt["scientific_seconds_this_stage"]
        < validated_summary["scientific_seconds_this_stage"]
    ):
        raise RecoveryReceiptError("pair_receipt_time_before_summary")
    if (
        validated_receipt["scientific_seconds_cumulative_bound"]
        < validated_summary["scientific_seconds_cumulative_bound"]
    ):
        raise RecoveryReceiptError("pair_receipt_time_before_summary")
    return {
        "schema_version": validated_summary["schema_version"],
        "protocol_version": validated_summary["protocol_version"],
        "claim": validated_summary["claim"],
        "permit_sha256": validated_summary["permit_sha256"],
        "recovery_permit_sha256": validated_summary["recovery_permit_sha256"],
        "original_evidence_anchor_sha256": validated_summary["original_evidence_anchor_sha256"],
        "recovery_plan_id": validated_summary["recovery_plan_id"],
        "core_contract_sha256": validated_summary["core_contract_sha256"],
        "core_plan_id": validated_summary["core_plan_id"],
        "ledger_id": validated_summary["ledger_id"],
        "started": validated_summary["started"],
        "completed": validated_summary["completed"],
        "failed": validated_summary["failed"],
        "original_interrupted_attempts": validated_summary["original_interrupted_attempts"],
        "new_completions": validated_summary["new_completions"],
        "reused_pilot_slots": validated_summary["reused_pilot_slots"],
        "reused_original_completions": validated_summary["reused_original_completions"],
        "recovery_started": validated_summary["recovery_started"],
        "recovery_completed": validated_summary["recovery_completed"],
        "recovery_failed": validated_summary["recovery_failed"],
        "replay_attempts": validated_summary["replay_attempts"],
        "originally_unstarted_attempts": validated_summary["originally_unstarted_attempts"],
        "selector_records": validated_summary["selector_records"],
        "units_total": validated_summary["units_total"],
        "units_completed": validated_summary["units_completed"],
        "optimizer_steps": validated_summary["optimizer_steps"],
        "optimizer_steps_exact": validated_summary["optimizer_steps_exact"],
        "optimizer_steps_scope": validated_summary["optimizer_steps_scope"],
        "reused_original_optimizer_steps_exact": (
            validated_summary["reused_original_optimizer_steps_exact"]
        ),
        "recovery_optimizer_steps_exact": validated_summary["recovery_optimizer_steps_exact"],
        "original_interrupted_optimizer_steps_observed_lower_bound": (
            validated_summary["original_interrupted_optimizer_steps_observed_lower_bound"]
        ),
        "original_interrupted_optimizer_steps_charged_upper_bound": (
            validated_summary["original_interrupted_optimizer_steps_charged_upper_bound"]
        ),
        "total_source_optimizer_steps_lower_bound": (
            validated_summary["total_source_optimizer_steps_lower_bound"]
        ),
        "total_source_optimizer_steps_charged_upper_bound": (
            validated_summary["total_source_optimizer_steps_charged_upper_bound"]
        ),
        "total_source_optimizer_steps_exact": (
            validated_summary["total_source_optimizer_steps_exact"]
        ),
        "maximum_optimizer_steps": validated_summary["maximum_optimizer_steps"],
        "maximum_new_neural_executions": validated_summary["maximum_new_neural_executions"],
        "maximum_new_optimizer_steps": validated_summary["maximum_new_optimizer_steps"],
        "scientific_seconds_this_stage": validated_receipt["scientific_seconds_this_stage"],
        "prior_scientific_seconds_charged_upper_bound": (
            validated_summary["prior_scientific_seconds_charged_upper_bound"]
        ),
        "prelaunch_audit_reserve_seconds": validated_summary["prelaunch_audit_reserve_seconds"],
        "scientific_seconds_cumulative_bound": (
            validated_receipt["scientific_seconds_cumulative_bound"]
        ),
        "maximum_total_seconds": validated_summary["maximum_total_seconds"],
        "live_bytes": validated_summary["live_bytes"],
        "sum_elapsed_seconds": validated_summary["sum_elapsed_seconds"],
        "maximum_peak_cuda_bytes": validated_summary["maximum_peak_cuda_bytes"],
        "storage_ceiling_bytes": validated_summary["storage_ceiling_bytes"],
        "maximum_cuda_allocated_bytes": validated_summary["maximum_cuda_allocated_bytes"],
        "frozen_source_fit_cuda_cap_bytes": validated_summary["frozen_source_fit_cuda_cap_bytes"],
        "source_fits_only": validated_summary["source_fits_only"],
        "selection_authorized": validated_summary["selection_authorized"],
        "refit_authorized": validated_summary["refit_authorized"],
        "calibration_authorized": validated_summary["calibration_authorized"],
        "outer_evaluation_authorized": validated_summary["outer_evaluation_authorized"],
        "stage": validated_receipt["stage"],
        "stage_manifest_sha256": validated_receipt["stage_manifest_sha256"],
    }
