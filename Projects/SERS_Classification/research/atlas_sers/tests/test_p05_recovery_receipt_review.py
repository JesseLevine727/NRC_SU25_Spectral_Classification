"""Independent checks for fixed completion coverage and accounting identity."""

import pytest

from atlas_sers.evaluation import p05_recovery_receipt as receipt


def test_expected_units_cannot_redefine_complete_coverage():
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt._validate_units({"units_completed": 0, "units_total": 1245}, 0)


def test_unrepresentable_time_has_stable_error():
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt._finite_nonneg(10**1000, "test_invalid_time")


@pytest.mark.parametrize("kind", ["contract", "ledger"])
def test_conflicting_base_bundle_aliases_cannot_hide_bad_canonical_identity(kind):
    base = {
        "permit_sha256": receipt.BASE_PERMIT_SHA256,
        "contract_sha256": receipt.CORE_CONTRACT_SHA256,
        "core_contract_sha256": receipt.CORE_CONTRACT_SHA256,
        "core_plan_id": receipt.CORE_PLAN_ID,
        "ledger_id": receipt.LEDGER_ID,
        "ledger": {"ledger_id": receipt.LEDGER_ID},
    }
    if kind == "contract":
        base["contract_sha256"] = "0" * 64
    else:
        base["ledger"] = {"ledger_id": "0" * 64}
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt._identity_from_bundle(base)


def test_serial_fit_time_cannot_exceed_whole_recovery_stage_time():
    with pytest.raises(receipt.RecoveryReceiptError):
        receipt._validate_resources(
            {
                "live_bytes": 1,
                "sum_elapsed_seconds": 200.0,
                "scientific_seconds_this_stage": 100.0,
                "maximum_peak_cuda_bytes": 1,
            }
        )
