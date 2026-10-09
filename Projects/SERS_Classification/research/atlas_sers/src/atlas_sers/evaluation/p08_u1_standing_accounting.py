"""P08-U1 standing bounded infrastructure-recovery accounting (T316).

Standalone immutable profile schema ``nato-sers-p08-u1-standing-accounting-v1``.

A profile records the exact completed operations copied out of a parent ledger
plus the current incomplete replay jobs and their cumulative per-operation
replay history.  It derives the accounting view consumed by
:mod:`atlas_sers.evaluation.p08_u1_store`.  Totals are always derived from the
declared identities; no caller-supplied total is trusted.

This module performs no model work, no spectral I/O and no authentication.
Parent audit/approval authentication stays with the outer caller.  It validates
shape, exact bound identities, counts, generation and dependency closure only.
"""

from __future__ import annotations

import math

__all__ = [
    "STANDING_ACCOUNTING_SCHEMA",
    "STANDING_MAX_REPLAYS_PER_JOB",
    "STANDING_MAX_GENERATIONS",
    "STANDING_FIT_SCALAR_REPLAY_CAP",
    "STANDING_TOTAL_REPLAY_CAP",
    "STANDING_HISTORICAL_OVERHEAD",
    "STANDING_SCALAR_OVERHEAD",
    "STANDING_MAX_FIT_TOTAL",
    "STANDING_MAX_UNIQUE_FITS",
    "STANDING_MAX_SCALAR_ATTEMPTS",
    "STANDING_MIN_ACTIVE_SECONDS",
    "STANDING_MIN_ARTIFACT_BYTES",
    "STANDING_EXPECTED_PILOT_FITS",
    "KNOWN_STAGES",
    "MODEL_FIT_STAGES",
    "SCALAR_STAGE",
    "PREDICTION_STAGES",
    "SELECTOR_STAGES",
    "OPERATION_STAGES",
    "is_standing_profile",
    "classify_stage_kind",
    "validate_standing_profile",
    "derive_standing_accounting",
    "validate_standing_jobs",
]

STANDING_ACCOUNTING_SCHEMA = "nato-sers-p08-u1-standing-accounting-v1"

KNOWN_STAGES = frozenset(
    (
        "source_fit",
        "source_validation_prediction",
        "select_hyperparameters",
        "calibration_model_fit",
        "calibration_validation_prediction",
        "calibration_prediction_alias",
        "scalar_calibration",
        "final_refit",
        "held_prediction",
        "seed_ensemble_prediction",
        "select_refit_epochs",
    )
)

MODEL_FIT_STAGES = frozenset(("source_fit", "calibration_model_fit", "final_refit"))
SCALAR_STAGE = "scalar_calibration"
PREDICTION_STAGES = frozenset(
    (
        "source_validation_prediction",
        "calibration_validation_prediction",
        "held_prediction",
    )
)
SELECTOR_STAGES = frozenset(("select_hyperparameters", "select_refit_epochs"))
OPERATION_STAGES = frozenset(("calibration_prediction_alias", "seed_ensemble_prediction"))

STANDING_MAX_REPLAYS_PER_JOB = 2
STANDING_MAX_GENERATIONS = 8
# 32 total fit/calibration replay attempts in the standing window, including the
# current four; at most two replays per operation; at most eight generations.
STANDING_FIT_SCALAR_REPLAY_CAP = 32
# Eight generations at a conservative five workers each.
STANDING_TOTAL_REPLAY_CAP = 40
STANDING_HISTORICAL_OVERHEAD = 14
STANDING_SCALAR_OVERHEAD = 0
STANDING_MAX_FIT_TOTAL = 195248
STANDING_MAX_UNIQUE_FITS = 195202
STANDING_MAX_SCALAR_ATTEMPTS = 3354
STANDING_MIN_ACTIVE_SECONDS = 5600
STANDING_MIN_ARTIFACT_BYTES = 8_000_000_000
STANDING_EXPECTED_PILOT_FITS = 78

_KNOWN_SORTED = tuple(sorted(KNOWN_STAGES))
_HEX = frozenset("0123456789abcdef")
_STANDING_FIELDS = frozenset(
    (
        "schema_version",
        "authority_sha256",
        "parent_audit_sha256",
        "parent_binding_sha256",
        "parent_inventory_sha256",
        "recovery_generation",
        "replay_job_ids",
        "replay_counts_by_stage",
        "completed_job_ids_by_stage",
        "baseline_active_seconds",
        "baseline_artifact_bytes",
    )
)


def _fail(message):
    from atlas_sers.evaluation.p08_u1_store import ValidationError

    raise ValidationError(message)


def _caps():
    from atlas_sers.evaluation import p08_u1_store as ledger

    return ledger.MAX_WALL_SECONDS, ledger.MAX_ARTIFACT_BYTES


def _is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def _is_job_id(value):
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("P08JOB-")
        and all(c in _HEX for c in value[7:])
    )


def is_standing_profile(profile):
    return isinstance(profile, dict) and profile.get("schema_version") == STANDING_ACCOUNTING_SCHEMA


def classify_stage_kind(stage):
    if stage in MODEL_FIT_STAGES:
        return "fit"
    if stage in PREDICTION_STAGES:
        return "prediction"
    if stage in SELECTOR_STAGES:
        return "selector"
    if stage == SCALAR_STAGE:
        return "calibration"
    if stage in OPERATION_STAGES:
        return "operation"
    _fail("standing_stage_unknown")


def _normalize_id_list(value, field):
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        _fail(field + "_invalid")
    items = list(value)
    if any(not _is_job_id(item) for item in items):
        _fail(field + "_invalid")
    if len(set(items)) != len(items):
        _fail(field + "_duplicate")
    if items != sorted(items):
        _fail(field + "_not_sorted")
    return tuple(items)


def _normalize_completed(value):
    if not isinstance(value, dict) or set(value.keys()) != KNOWN_STAGES:
        _fail("completed_job_ids_by_stage_invalid")
    result = {}
    for stage in _KNOWN_SORTED:
        result[stage] = _normalize_id_list(value[stage], "completed_job_ids_by_stage")
    return result


def _normalize_history(value):
    if not isinstance(value, dict) or set(value.keys()) != KNOWN_STAGES:
        _fail("replay_counts_by_stage_invalid")
    result = {}
    for stage in _KNOWN_SORTED:
        inner = value[stage]
        if not isinstance(inner, dict):
            _fail("replay_counts_by_stage_invalid")
        keys = list(inner.keys())
        if any(not _is_job_id(key) for key in keys):
            _fail("replay_counts_by_stage_invalid")
        if keys != sorted(keys):
            _fail("replay_counts_by_stage_not_sorted")
        counts = {}
        for key in keys:
            count = inner[key]
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
                or count > STANDING_MAX_REPLAYS_PER_JOB
            ):
                _fail("replay_count_invalid")
            counts[key] = int(count)
        result[stage] = counts
    return result


def validate_standing_profile(profile):
    """Validate and normalize a standing recovery accounting profile."""
    if not isinstance(profile, dict):
        _fail("standing_accounting_must_be_mapping")
    if set(profile.keys()) != _STANDING_FIELDS:
        _fail("standing_accounting_keys_invalid")
    if profile["schema_version"] != STANDING_ACCOUNTING_SCHEMA:
        _fail("standing_accounting_schema_unsupported")
    for field in (
        "authority_sha256",
        "parent_audit_sha256",
        "parent_binding_sha256",
        "parent_inventory_sha256",
    ):
        if not _is_digest(profile[field]):
            _fail(field + "_invalid")
    generation = profile["recovery_generation"]
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
        or generation > STANDING_MAX_GENERATIONS
    ):
        _fail("recovery_generation_invalid")

    current = _normalize_id_list(profile["replay_job_ids"], "replay_job_ids")
    completed = _normalize_completed(profile["completed_job_ids_by_stage"])
    history = _normalize_history(profile["replay_counts_by_stage"])

    completed_seen = set()
    for stage in _KNOWN_SORTED:
        for job_id in completed[stage]:
            if job_id in completed_seen:
                _fail("standing_completed_duplicate_across_stages")
            completed_seen.add(job_id)

    history_seen = set()
    for stage in _KNOWN_SORTED:
        for job_id in history[stage]:
            if job_id in history_seen:
                _fail("standing_replay_history_duplicate_across_stages")
            history_seen.add(job_id)

    for job_id in current:
        if job_id not in history_seen:
            _fail("standing_current_replay_not_in_history")
        if job_id in completed_seen:
            _fail("standing_current_replay_completed_overlap")

    fit_scalar_replays = 0
    total_replays = 0
    for stage in _KNOWN_SORTED:
        subtotal = sum(history[stage].values())
        total_replays += subtotal
        if stage in MODEL_FIT_STAGES or stage == SCALAR_STAGE:
            fit_scalar_replays += subtotal
    if fit_scalar_replays > STANDING_FIT_SCALAR_REPLAY_CAP:
        _fail("standing_replay_cap_exceeded")
    if total_replays > STANDING_TOTAL_REPLAY_CAP:
        _fail("standing_total_replay_cap_exceeded")

    completed_fits = sum(len(completed[stage]) for stage in MODEL_FIT_STAGES)
    if completed_fits < STANDING_EXPECTED_PILOT_FITS:
        _fail("standing_completed_fit_floor")

    active = profile["baseline_active_seconds"]
    max_wall, max_artifact = _caps()
    if (
        isinstance(active, bool)
        or not isinstance(active, (int, float))
        or not math.isfinite(active)
        or active < STANDING_MIN_ACTIVE_SECONDS
        or active >= max_wall
    ):
        _fail("baseline_active_seconds_invalid")
    artifact = profile["baseline_artifact_bytes"]
    if (
        isinstance(artifact, bool)
        or not isinstance(artifact, int)
        or artifact < STANDING_MIN_ARTIFACT_BYTES
        or artifact >= max_artifact
    ):
        _fail("baseline_artifact_bytes_invalid")

    return {
        "schema_version": STANDING_ACCOUNTING_SCHEMA,
        "authority_sha256": profile["authority_sha256"],
        "parent_audit_sha256": profile["parent_audit_sha256"],
        "parent_binding_sha256": profile["parent_binding_sha256"],
        "parent_inventory_sha256": profile["parent_inventory_sha256"],
        "recovery_generation": int(generation),
        "replay_job_ids": current,
        "replay_counts_by_stage": history,
        "completed_job_ids_by_stage": completed,
        "baseline_active_seconds": float(active),
        "baseline_artifact_bytes": int(artifact),
    }


def derive_standing_accounting(profile):
    """Derive the ledger accounting view from a standing profile."""
    normalized = validate_standing_profile(profile)
    completed = normalized["completed_job_ids_by_stage"]
    history = normalized["replay_counts_by_stage"]

    fit_ids = tuple(sorted(j for stage in MODEL_FIT_STAGES for j in completed[stage]))
    prediction_ids = tuple(sorted(j for stage in PREDICTION_STAGES for j in completed[stage]))
    selector_ids = tuple(sorted(j for stage in SELECTOR_STAGES for j in completed[stage]))
    scalar_ids = tuple(sorted(completed[SCALAR_STAGE]))
    operation_ids = tuple(sorted(j for stage in OPERATION_STAGES for j in completed[stage]))

    historical = STANDING_HISTORICAL_OVERHEAD + sum(
        count for stage in MODEL_FIT_STAGES for count in history[stage].values()
    )
    scalar_overhead = sum(history[SCALAR_STAGE].values())
    current = tuple(normalized["replay_job_ids"])
    history_stage = {job_id: stage for stage in _KNOWN_SORTED for job_id in history[stage]}
    replay_fit = tuple(sorted(j for j in current if history_stage.get(j) in MODEL_FIT_STAGES))

    return {
        "schema_version": STANDING_ACCOUNTING_SCHEMA,
        "standing": True,
        "expected_reuse_fits": len(fit_ids),
        "expected_reuse_predictions": len(prediction_ids),
        "max_reuse_fits": len(fit_ids),
        "max_reuse_predictions": len(prediction_ids),
        "historical_overhead_attempts": historical,
        "scalar_overhead_attempts": scalar_overhead,
        "max_fit_total": STANDING_MAX_FIT_TOTAL,
        "max_unique_fit_jobs": STANDING_MAX_UNIQUE_FITS,
        "max_scalar_attempts": STANDING_MAX_SCALAR_ATTEMPTS,
        "baseline_active_seconds": float(normalized["baseline_active_seconds"]),
        "baseline_artifact_bytes": int(normalized["baseline_artifact_bytes"]),
        "replay_fit_job_ids": replay_fit,
        "all_replay_job_ids": current,
        "current_replay_ids": current,
        "additional_reuse_fit_job_ids": fit_ids,
        "selector_job_ids": selector_ids,
        "scalar_job_ids": scalar_ids,
        "operation_job_ids": operation_ids,
        "prediction_job_ids": prediction_ids,
        "completed_job_ids_by_stage": completed,
        "replay_counts_by_stage": history,
        "recovery_generation": int(normalized["recovery_generation"]),
        "authority_sha256": normalized["authority_sha256"],
        "parent_audit_sha256": normalized["parent_audit_sha256"],
        "parent_binding_sha256": normalized["parent_binding_sha256"],
        "parent_inventory_sha256": normalized["parent_inventory_sha256"],
        "stage_kinds": {stage: classify_stage_kind(stage) for stage in _KNOWN_SORTED},
    }


def validate_standing_jobs(accounting, parsed_jobs):
    """Bind every declared identity to the registered graph and verify closure."""
    by_id = {record["job_id"]: record for record in parsed_jobs}
    completed = accounting["completed_job_ids_by_stage"]
    history = accounting["replay_counts_by_stage"]

    completed_all = set()
    for stage in _KNOWN_SORTED:
        for job_id in completed[stage]:
            record = by_id.get(job_id)
            if record is None or record["stage"] != stage:
                _fail("standing_completed_job_missing")
            completed_all.add(job_id)

    for stage in _KNOWN_SORTED:
        for job_id in history[stage]:
            record = by_id.get(job_id)
            if record is None or record["stage"] != stage:
                _fail("standing_replay_job_missing")

    for job_id in accounting["current_replay_ids"]:
        if job_id in completed_all:
            _fail("standing_current_replay_completed_overlap")

    for stage in _KNOWN_SORTED:
        for job_id in completed[stage]:
            record = by_id[job_id]
            for dep in record["dependencies"]:
                if dep not in completed_all:
                    _fail("standing_completed_dependency_open")
