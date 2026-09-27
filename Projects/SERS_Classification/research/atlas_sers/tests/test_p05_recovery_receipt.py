"""Synthetic contract tests for the P05 recovery receipt module."""

from __future__ import annotations

import pytest

from atlas_sers.evaluation import p05_recovery_receipt as receipt

RECOVERY_PLAN_ID = "0123456789abcdef" * 4
STAGE_MANIFEST_SHA256 = "fedcba9876543210" * 4

VALID_COUNTERS = {
    "new_started": 6184,
    "new_completed": 6184,
    "new_failed": 0,
    "new_optimizer_steps": 2_000_000,
    "new_optimizer_steps_exact": True,
    "new_elapsed_seconds": 123.0,
    "new_peak_cuda_bytes": 166_337_536,
    "reused_completed": 8720,
    "reused_optimizer_steps": 1_669_388,
    "replay_started": 1,
    "unstarted_started": 6183,
}


def _bundle():
    return {
        "permit_sha256": receipt.BASE_PERMIT_SHA256,
        "core_contract_sha256": receipt.CORE_CONTRACT_SHA256,
        "core_plan_id": receipt.CORE_PLAN_ID,
        "ledger_id": receipt.LEDGER_ID,
    }


def _build_summary(**overrides):
    kwargs = dict(
        base_bundle=_bundle(),
        recovery_plan_id=RECOVERY_PLAN_ID,
        counters=dict(VALID_COUNTERS),
        units_completed=receipt.DEFAULT_EXPECTED_UNITS,
        selector_records=receipt.REQUIRED_SELECTOR_RECORDS,
        scientific_seconds=1000.0,
        live_bytes=1024,
    )
    kwargs.update(overrides)
    return receipt.build_summary(**kwargs)


def _tampered_summary(**changes):
    payload = _build_summary()
    payload.update(changes)
    return payload


def _build_receipt(summary=None, scientific_seconds=2000.0):
    if summary is None:
        summary = _build_summary()
    return receipt.build_receipt(
        summary=summary,
        stage_manifest_sha256=STAGE_MANIFEST_SHA256,
        scientific_seconds=scientific_seconds,
    )


def test_build_summary_valid_counts():
    summary = _build_summary()
    assert summary["started"] == 14905
    assert summary["completed"] == 14904
    assert summary["failed"] == 1
    assert summary["original_interrupted_attempts"] == 1
    assert summary["new_completions"] == 14904
    assert summary["reused_pilot_slots"] == 36
    assert summary["reused_original_completions"] == 8720
    assert summary["recovery_started"] == 6184
    assert summary["recovery_completed"] == 6184
    assert summary["recovery_failed"] == 0
    assert summary["replay_attempts"] == 1
    assert summary["originally_unstarted_attempts"] == 6183
    assert summary["selector_records"] == 14940
    assert summary["units_total"] == 1245
    assert summary["units_completed"] == 1242
    assert summary["optimizer_steps_exact"] is True
    assert summary["total_source_optimizer_steps_exact"] is False
    assert summary["optimizer_steps"] == 1_669_388 + 2_000_000
    assert summary["total_source_optimizer_steps_lower_bound"] == summary["optimizer_steps"] + 68
    assert (
        summary["total_source_optimizer_steps_charged_upper_bound"]
        == summary["optimizer_steps"] + 800
    )
    assert summary["maximum_optimizer_steps"] == 14905 * 800
    assert summary["maximum_new_neural_executions"] == 17785
    assert summary["maximum_new_optimizer_steps"] == 14228000


def test_validator_returns_independent_copy():
    summary = _build_summary()
    copy = receipt.validate_summary(summary)
    copy["started"] = 0
    assert summary["started"] == 14905
    assert receipt.validate_summary(summary)["started"] == 14905


def test_build_and_validate_receipt():
    summary = _build_summary()
    rcpt = _build_receipt(summary, scientific_seconds=2000.0)
    assert rcpt["stage"] == "develop"
    assert rcpt["stage_manifest_sha256"] == STAGE_MANIFEST_SHA256
    assert rcpt["scientific_seconds_this_stage"] == 2000.0
    assert rcpt["scientific_seconds_cumulative_bound"] == 39600.0 + 2000.0


def test_validate_pair_returns_cost_payload():
    summary = _build_summary()
    rcpt = _build_receipt(summary)
    cost = receipt.validate_pair(summary, rcpt)
    assert cost["optimizer_steps"] == summary["optimizer_steps"]
    assert (
        cost["total_source_optimizer_steps_charged_upper_bound"]
        == summary["total_source_optimizer_steps_charged_upper_bound"]
    )
    assert (
        cost["scientific_seconds_cumulative_bound"] == rcpt["scientific_seconds_cumulative_bound"]
    )
    assert cost["stage_manifest_sha256"] == STAGE_MANIFEST_SHA256
    assert cost["failed"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("started", 14904),
        ("completed", 14905),
        ("failed", 0),
        ("original_interrupted_attempts", 0),
        ("new_completions", 14905),
        ("reused_pilot_slots", 0),
        ("reused_original_completions", 8721),
        ("recovery_started", 6183),
        ("recovery_completed", 6183),
        ("recovery_failed", 1),
        ("replay_attempts", 0),
        ("originally_unstarted_attempts", 6184),
        ("selector_records", 14939),
        ("units_total", 1244),
    ],
)
def test_forged_counts_rejected(field, value):
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(_tampered_summary(**{field: value}))


def test_forged_zero_failure_clean_run_rejected():
    forged = _build_summary()
    forged["failed"] = 0
    forged["original_interrupted_attempts"] = 0
    forged["started"] = forged["completed"]
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(forged)


@pytest.mark.parametrize(
    "field",
    [
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
        "optimizer_steps",
        "recovery_optimizer_steps_exact",
        "live_bytes",
        "maximum_peak_cuda_bytes",
    ],
)
def test_bool_is_not_int(field):
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(_tampered_summary(**{field: True}))


@pytest.mark.parametrize(
    "field",
    [
        "scientific_seconds_this_stage",
        "sum_elapsed_seconds",
        "scientific_seconds_cumulative_bound",
    ],
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_time_rejected(field, bad):
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(_tampered_summary(**{field: bad}))


@pytest.mark.parametrize(
    "field",
    [
        "scientific_seconds_this_stage",
        "sum_elapsed_seconds",
        "live_bytes",
        "maximum_peak_cuda_bytes",
    ],
)
def test_negative_values_rejected(field):
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(_tampered_summary(**{field: -1.0}))


@pytest.mark.parametrize(
    "field",
    [
        "permit_sha256",
        "recovery_permit_sha256",
        "original_evidence_anchor_sha256",
        "core_contract_sha256",
        "core_plan_id",
        "ledger_id",
    ],
)
def test_identity_and_pin_swaps_rejected(field):
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(_tampered_summary(**{field: "0" * 64}))


def test_permit_swap_between_base_and_recovery_rejected():
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(_tampered_summary(permit_sha256=receipt.RECOVERY_PERMIT_SHA256))
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(
            _tampered_summary(recovery_permit_sha256=receipt.BASE_PERMIT_SHA256)
        )


def test_recovery_plan_id_must_be_lowercase_sha256():
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(_tampered_summary(recovery_plan_id="not-a-hash"))
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(_tampered_summary(recovery_plan_id="A" * 64))


def test_extra_field_rejected():
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(_tampered_summary(extra="x"))


def test_unknown_schema_rejected():
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(
            _tampered_summary(schema_version="nato-sers-p05-comprehensive-development-v1")
        )


def test_missing_field_rejected():
    payload = _build_summary()
    del payload["failed"]
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("optimizer_steps_exact", False),
        ("total_source_optimizer_steps_exact", True),
        ("optimizer_steps_scope", "all_source_fits"),
        ("reused_original_optimizer_steps_exact", 1_669_387),
        ("original_interrupted_optimizer_steps_observed_lower_bound", 0),
        ("original_interrupted_optimizer_steps_charged_upper_bound", 68),
    ],
)
def test_optimizer_exactness_and_charge_rejected(field, value):
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(_tampered_summary(**{field: value}))


def test_charged_upper_bound_understatement_rejected():
    payload = _build_summary()
    payload["total_source_optimizer_steps_charged_upper_bound"] = payload[
        "total_source_optimizer_steps_lower_bound"
    ]
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(payload)


@pytest.mark.parametrize(
    "recovery_steps",
    [0, 741_999, 4_947_201, 2_000_001, 2_000_000.0, True],
)
def test_recovery_optimizer_steps_bounds_and_type(recovery_steps):
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(_tampered_summary(recovery_optimizer_steps_exact=recovery_steps))


def test_time_clock_reset_rejected():
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(_tampered_summary(prior_scientific_seconds_charged_upper_bound=0))


def test_cumulative_mismatch_rejected():
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(_tampered_summary(scientific_seconds_cumulative_bound=1.0))


def test_time_ceiling_rejected():
    with pytest.raises(receipt.RecoveryReceiptError):
        _build_summary(scientific_seconds=200_000.0)


def test_live_bytes_over_ceiling_rejected():
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(_tampered_summary(live_bytes=receipt.STORAGE_CEILING_BYTES + 1))


def test_peak_cuda_over_frozen_cap_rejected():
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(
            _tampered_summary(maximum_peak_cuda_bytes=receipt.FROZEN_SOURCE_FIT_CUDA_CAP_BYTES + 1)
        )


def test_receipt_before_summary_time_rejected():
    summary = _build_summary(scientific_seconds=5000.0)
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.build_receipt(
            summary=summary,
            stage_manifest_sha256=STAGE_MANIFEST_SHA256,
            scientific_seconds=4000.0,
        )


def test_receipt_invalid_manifest_rejected():
    summary = _build_summary()
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.build_receipt(
            summary=summary,
            stage_manifest_sha256="zz",
            scientific_seconds=2000.0,
        )


def test_receipt_wrong_stage_rejected():
    rcpt = _build_receipt()
    rcpt["stage"] = "selection"
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_receipt(rcpt)


def test_receipt_extra_field_rejected():
    rcpt = _build_receipt()
    rcpt["extra"] = 1
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_receipt(rcpt)


def test_validate_receipt_independent_of_builder():
    summary = _build_summary()
    rcpt = dict(summary)
    rcpt["stage"] = receipt.STAGE_NAME
    rcpt["stage_manifest_sha256"] = STAGE_MANIFEST_SHA256
    rcpt["scientific_seconds_this_stage"] = 1500.0
    rcpt["scientific_seconds_cumulative_bound"] = 39600.0 + 1500.0
    validated = receipt.validate_receipt(rcpt)
    assert validated["stage_manifest_sha256"] == STAGE_MANIFEST_SHA256
    assert validated["scientific_seconds_this_stage"] == 1500.0


def test_pair_identity_mismatch_rejected():
    summary = _build_summary()
    rcpt = _build_receipt(summary)
    rcpt["recovery_plan_id"] = "f" * 64
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_pair(summary, rcpt)


def test_pair_receipt_time_before_summary_rejected():
    summary = _build_summary(scientific_seconds=5000.0)
    rcpt = _build_receipt(summary, scientific_seconds=6000.0)
    rcpt["scientific_seconds_this_stage"] = 4000.0
    rcpt["scientific_seconds_cumulative_bound"] = 39600.0 + 4000.0
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_pair(summary, rcpt)


def test_expected_units_parameter():
    summary = _build_summary()
    assert receipt.validate_summary(summary, expected_units=1242)["units_completed"] == 1242
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt.validate_summary(summary, expected_units=1241)


@pytest.mark.parametrize(
    "field,value",
    [
        ("new_started", 6183),
        ("new_completed", 6183),
        ("new_failed", 1),
        ("replay_started", 0),
        ("unstarted_started", 6184),
        ("reused_completed", 8719),
        ("reused_optimizer_steps", 1_669_387),
        ("new_optimizer_steps_exact", False),
        ("new_optimizer_steps", 2_000_001),
        ("new_optimizer_steps", 741_999),
        ("new_optimizer_steps", 4_947_201),
        ("new_optimizer_steps", 2_000_000.0),
        ("new_optimizer_steps", True),
        ("new_elapsed_seconds", -1.0),
        ("new_elapsed_seconds", float("nan")),
        ("new_peak_cuda_bytes", -1),
        ("new_peak_cuda_bytes", receipt.FROZEN_SOURCE_FIT_CUDA_CAP_BYTES + 1),
    ],
)
def test_counter_validation_rejected(field, value):
    counters = dict(VALID_COUNTERS)
    counters[field] = value
    with pytest.raises(receipt.RecoveryReceiptError):
        _build_summary(counters=counters)


def test_counter_keys_rejected():
    counters = dict(VALID_COUNTERS)
    del counters["new_failed"]
    with pytest.raises(receipt.RecoveryReceiptError):
        _build_summary(counters=counters)
    counters = dict(VALID_COUNTERS)
    counters["extra"] = 0
    with pytest.raises(receipt.RecoveryReceiptError):
        _build_summary(counters=counters)


def test_base_bundle_identity_rejected():
    bad = _bundle()
    bad["permit_sha256"] = receipt.RECOVERY_PERMIT_SHA256
    with pytest.raises(receipt.RecoveryReceiptError):
        _build_summary(base_bundle=bad)
    bad = _bundle()
    bad["ledger_id"] = "other"
    with pytest.raises(receipt.RecoveryReceiptError):
        _build_summary(base_bundle=bad)
    with pytest.raises(receipt.RecoveryReceiptError):
        _build_summary(base_bundle=None)


def test_validator_independent_of_builder():
    summary = _build_summary()
    manual = dict(summary)
    assert receipt.validate_summary(manual) == summary
