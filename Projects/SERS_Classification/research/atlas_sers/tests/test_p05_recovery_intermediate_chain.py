"""CPU-only intermediate tests for the six recovery-aware P05 chain modules.

The suites under test already provide synthetic fixtures that persist real
manifests, receipts, prediction CSV/parquet bytes and real ``StorageBudget``
boundaries.  Those fixtures intentionally stub the upstream *authority seam*
(``p05_evaluation_authority`` / ``p05_frozen_predictions`` /
``p05_aggregation_authority``) and the numerical inference kernels
(``p05_prediction.predict_refit`` and friends); no fit, no GPU and no private
data is touched here.

This file only upgrades that stubbed authority seam so it advertises a valid
recovered ``source_execution_accounting`` payload.  Everything downstream of
the seam still runs for real: producers persist summary/receipt through the
genuine close-out and ``StorageBudget`` paths, and consumers re-open the
persisted bytes, verify manifests and re-run the real read-only authentication
including ``p05_recovery_source.validate_accounting``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping

import pytest

from atlas_sers.evaluation import p05_aggregation_authority as aggregation_authority
from atlas_sers.evaluation import p05_comparison_authority as comparison_authority
from atlas_sers.evaluation import p05_comprehensive_aggregation as aggregation
from atlas_sers.evaluation import p05_comprehensive_comparison as comparison
from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_evaluation as evaluation
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_evaluation_authority as evaluation_authority
from atlas_sers.evaluation import p05_frozen_predictions as frozen
from atlas_sers.evaluation import p05_recovery_source as source
from tests import test_p05_aggregation_authority as aggregation_authority_fixtures
from tests import test_p05_comparison_authority as comparison_authority_fixtures
from tests import test_p05_comprehensive_aggregation as aggregation_fixtures
from tests import test_p05_comprehensive_comparison as comparison_fixtures
from tests import test_p05_comprehensive_evaluation as evaluation_fixtures
from tests import test_p05_frozen_predictions as frozen_fixtures
from tests import test_p05_recovery_source as recovery_fixtures


def _raw(fixture: object):
    """Return the plain function behind a pytest fixture (or the callable itself)."""

    return fixture.__wrapped__


def _deadline() -> float:
    return time.perf_counter() + development.MAXIMUM_TOTAL_SECONDS


def _accounting_and_steps() -> tuple[dict, int]:
    summary, receipt = recovery_fixtures._recovered_pair()
    accounting = source.accounting_from_recovered(summary, receipt)
    steps = accounting["source_optimizer_steps_successful_exact"]
    assert isinstance(steps, int) and not isinstance(steps, bool)
    assert accounting["mode"] == source.RECOVERY_ACCOUNTING_MODE
    return accounting, steps


def _wrap_upstream(original, accounting: Mapping[str, object], steps: int):
    """Upgrade a stubbed authority seam with recovered accounting, keeping all fields."""

    def upgraded(bundle_arg, **kwargs):
        result = original(bundle_arg, **kwargs)
        assert isinstance(result, Mapping)
        payload = dict(result)
        payload["source_optimizer_steps"] = steps
        payload["source_execution_accounting"] = dict(accounting)
        return payload

    return upgraded


def _mutated_accounting(accounting: Mapping[str, object]) -> dict:
    mutated = dict(accounting)
    mutated["source_interrupted_attempts"] = True
    return mutated


def _install_frozen_accounting(
    world,
    accounting: Mapping[str, object],
    steps: int,
    *,
    drop_summary: bool = False,
    drop_receipt: bool = False,
) -> None:
    summary = core._read_json(world.summary_path, "summary")
    receipt = core._read_json(world.receipt_path, "receipt")
    for payload, drop in ((summary, drop_summary), (receipt, drop_receipt)):
        payload["source_optimizer_steps"] = steps
        if drop:
            payload.pop("source_execution_accounting", None)
            continue
        payload["source_execution_accounting"] = dict(accounting)
    core._atomic_write(world.summary_path, core._canon().canonical_json_bytes(summary))
    core._atomic_write(world.receipt_path, core._canon().canonical_json_bytes(receipt))
    frozen_fixtures._reseal(world)


def test_evaluation_producer_persists_recovery_accounting(tmp_path, monkeypatch):
    world = _raw(evaluation_fixtures.world)(tmp_path, monkeypatch)
    accounting, steps = _accounting_and_steps()

    original = evaluation_authority.authenticate_refits
    monkeypatch.setattr(
        evaluation_authority,
        "authenticate_refits",
        _wrap_upstream(original, accounting, steps),
    )

    receipt = evaluation.run_evaluation(**world.run_kwargs())

    assert receipt["source_optimizer_steps"] == steps
    assert receipt["source_execution_accounting"] == accounting
    assert core._read_json(world.run_root / evaluation.RECEIPT_NAME, "receipt") == receipt
    summary = core._read_json(world.stage / "summary.json", "summary")
    assert summary["source_optimizer_steps"] == steps
    assert summary["source_execution_accounting"] == accounting


def test_frozen_consumer_returns_recovery_accounting(tmp_path, monkeypatch):
    world = frozen_fixtures._build_world(tmp_path, monkeypatch)
    accounting, steps = _accounting_and_steps()
    _install_frozen_accounting(world, accounting, steps)

    original = evaluation_authority.authenticate_refits
    monkeypatch.setattr(
        evaluation_authority,
        "authenticate_refits",
        _wrap_upstream(original, accounting, steps),
    )

    result = frozen.authenticate_predictions(world.bundle, deadline=_deadline())

    assert result["source_optimizer_steps"] == steps
    assert result["source_execution_accounting"] == accounting
    assert set(result["predictions"]) == {spec["refit_id"] for spec in world.specs}


@pytest.mark.parametrize(
    "mode,code",
    [
        ("drop_summary", "summary_source_accounting_missing"),
        ("drop_receipt", "receipt_source_accounting_missing"),
        ("corrupt", "receipt_source_accounting_invalid"),
    ],
)
def test_frozen_consumer_rejects_tampered_accounting(tmp_path, monkeypatch, mode, code):
    world = frozen_fixtures._build_world(tmp_path, monkeypatch)
    accounting, steps = _accounting_and_steps()

    if mode == "drop_summary":
        _install_frozen_accounting(world, accounting, steps, drop_summary=True)
    elif mode == "drop_receipt":
        _install_frozen_accounting(world, accounting, steps, drop_receipt=True)
    else:
        _install_frozen_accounting(world, _mutated_accounting(accounting), steps)

    original = evaluation_authority.authenticate_refits
    monkeypatch.setattr(
        evaluation_authority,
        "authenticate_refits",
        _wrap_upstream(original, accounting, steps),
    )

    with pytest.raises(frozen.P05FrozenPredictionsError) as info:
        frozen.authenticate_predictions(world.bundle, deadline=_deadline())
    assert info.value.reason_code == code


def test_frozen_consumer_rejects_unexpected_accounting_in_clean_mode(tmp_path, monkeypatch):
    world = frozen_fixtures._build_world(tmp_path, monkeypatch)
    accounting, _steps = _accounting_and_steps()

    # Authority stays clean (no accounting key) while the persisted stage claims
    # recovered accounting: the clean-mode branch must reject the extra payload.
    summary = core._read_json(world.summary_path, "summary")
    receipt = core._read_json(world.receipt_path, "receipt")
    for payload in (summary, receipt):
        payload["source_execution_accounting"] = dict(accounting)
    core._atomic_write(world.summary_path, core._canon().canonical_json_bytes(summary))
    core._atomic_write(world.receipt_path, core._canon().canonical_json_bytes(receipt))
    frozen_fixtures._reseal(world)

    with pytest.raises(frozen.P05FrozenPredictionsError) as info:
        frozen.authenticate_predictions(world.bundle, deadline=_deadline())
    assert info.value.reason_code == "receipt_source_accounting_unexpected"


def test_aggregation_chain_preserves_recovery_accounting(tmp_path, monkeypatch):
    world = _raw(aggregation_fixtures.world)(tmp_path, monkeypatch)
    accounting, steps = _accounting_and_steps()

    original = frozen.authenticate_predictions
    monkeypatch.setattr(
        frozen,
        "authenticate_predictions",
        _wrap_upstream(original, accounting, steps),
    )

    receipt = aggregation.run_aggregation(**world.run_kwargs())
    result = aggregation_authority.authenticate_aggregation(world.bundle, deadline=_deadline())

    assert receipt["source_optimizer_steps"] == steps
    assert receipt["source_execution_accounting"] == accounting
    summary = core._read_json(world.stage / aggregation.SUMMARY_NAME, "summary")
    assert summary["source_execution_accounting"] == accounting
    assert result["source_optimizer_steps"] == steps
    assert result["source_execution_accounting"] == accounting
    assert result["aggregation_receipt"]["source_execution_accounting"] == accounting


def test_aggregation_authority_rejects_tampered_summary_accounting(tmp_path, monkeypatch):
    world = _raw(aggregation_fixtures.world)(tmp_path, monkeypatch)
    accounting, steps = _accounting_and_steps()

    original = frozen.authenticate_predictions
    monkeypatch.setattr(
        frozen,
        "authenticate_predictions",
        _wrap_upstream(original, accounting, steps),
    )
    aggregation.run_aggregation(**world.run_kwargs())

    summary_path = world.stage / aggregation.SUMMARY_NAME
    summary = core._read_json(summary_path, "summary")
    summary["source_execution_accounting"]["source_interrupted_attempts"] = True
    core._atomic_write(summary_path, core._canon().canonical_json_bytes(summary))
    aggregation_authority_fixtures._reseal(world)

    with pytest.raises(aggregation_authority.P05AggregationAuthorityError) as info:
        aggregation_authority.authenticate_aggregation(world.bundle, deadline=_deadline())
    assert info.value.reason_code == "summary_source_accounting_invalid"


def test_comparison_chain_preserves_recovery_accounting(tmp_path, monkeypatch):
    world = _raw(comparison_fixtures.world)(tmp_path, monkeypatch)
    prepared = _raw(comparison_authority_fixtures.prepared)(world, monkeypatch)
    accounting, steps = _accounting_and_steps()

    original = aggregation_authority.authenticate_aggregation
    monkeypatch.setattr(
        aggregation_authority,
        "authenticate_aggregation",
        _wrap_upstream(original, accounting, steps),
    )

    receipt = comparison.run_comparison(**prepared.run_kwargs())
    result = comparison_authority.authenticate_comparison(prepared.bundle, deadline=_deadline())

    assert receipt["source_optimizer_steps"] == steps
    assert receipt["source_execution_accounting"] == accounting
    summary = core._read_json(prepared.stage / comparison.SUMMARY_NAME, "summary")
    assert summary["source_execution_accounting"] == accounting
    assert result["source_optimizer_steps"] == steps
    assert result["source_execution_accounting"] == accounting
    assert result["comparison_receipt"]["source_execution_accounting"] == accounting


def test_comparison_authority_rejects_stripped_receipt_accounting(tmp_path, monkeypatch):
    world = _raw(comparison_fixtures.world)(tmp_path, monkeypatch)
    prepared = _raw(comparison_authority_fixtures.prepared)(world, monkeypatch)
    accounting, steps = _accounting_and_steps()

    original = aggregation_authority.authenticate_aggregation
    monkeypatch.setattr(
        aggregation_authority,
        "authenticate_aggregation",
        _wrap_upstream(original, accounting, steps),
    )
    comparison.run_comparison(**prepared.run_kwargs())

    receipt_path = prepared.run_root / comparison.RECEIPT_NAME
    payload = core._read_json(receipt_path, "receipt")
    payload.pop("source_execution_accounting", None)
    core._atomic_write(receipt_path, core._canon().canonical_json_bytes(payload))
    comparison_authority_fixtures._reseal(prepared)

    with pytest.raises(comparison_authority.P05ComparisonAuthorityError) as info:
        comparison_authority.authenticate_comparison(prepared.bundle, deadline=_deadline())
    assert info.value.reason_code == "receipt_source_accounting_missing"
