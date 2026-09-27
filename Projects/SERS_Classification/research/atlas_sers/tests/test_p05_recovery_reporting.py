"""Pure unit tests for recovery-aware P05 reporting cost accounting.

These tests exercise the recovery cost gate on small synthetic bundles: they
never run the pipeline, load feature arrays, fit models or render figures.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from atlas_sers.evaluation import p05_comprehensive_reporting as reporting
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_recovery_source as source
from atlas_sers.evaluation import p05_reporting_inputs as reporting_inputs
from atlas_sers.governance import p05_publication_gate as publication_gate
from tests.test_p05_recovery_source import _recovered_pair

RECOVERY_KEYS = tuple(reporting_inputs.RECOVERY_PUBLIC_COST_KEYS)


class _Constants:
    """Minimal constants-only stand-in for the reporting-inputs module."""

    PUBLIC_COST_KEYS = tuple(reporting_inputs.PUBLIC_COST_KEYS)
    REQUIRED_PUBLIC_COST_KEYS = tuple(reporting_inputs.REQUIRED_PUBLIC_COST_KEYS)


def _normalized_accounting() -> dict[str, Any]:
    return source.accounting_from_recovered(*_recovered_pair())


def _assemble(source_accounting: Any) -> dict[str, Any]:
    source_steps = _normalized_accounting()["source_optimizer_steps_successful_exact"]
    return reporting_inputs._assemble_costs(
        {"scientific_seconds_this_stage": 1000.0},
        {"scientific_seconds_this_stage": 20.0},
        {"peak_cuda_bytes": 1024},
        {"scientific_seconds_cumulative_bound": 40650.0},
        2,
        source_steps,
        240,
        source_accounting=source_accounting,
    )


def _recovered_costs() -> dict[str, Any]:
    return _assemble(_normalized_accounting())


def _clean_costs() -> dict[str, Any]:
    return _assemble(None)


def _receipt(costs: Mapping[str, Any], accounting: Any) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "source_optimizer_steps": costs["new_source_optimizer_updates"],
        "refit_optimizer_steps": costs["refit_optimizer_updates"],
        "prior_scientific_seconds_cumulative_bound": costs[
            "scientific_seconds_cumulative_bound_through_comparison"
        ],
    }
    if accounting is not None:
        receipt["source_execution_accounting"] = dict(accounting)
    return receipt


def test_recovered_cost_bundle_is_complete() -> None:
    costs = _recovered_costs()
    assert set(RECOVERY_KEYS) <= set(costs)
    validated = reporting._check_public_costs({"public_costs": costs}, reporting_inputs)
    assert dict(validated) == costs


def test_legacy_cost_bundle_still_validates() -> None:
    costs = _clean_costs()
    assert not (set(RECOVERY_KEYS) & set(costs))
    validated = reporting._check_public_costs({"public_costs": costs}, reporting_inputs)
    assert dict(validated) == costs


def test_partial_recovery_extras_rejected() -> None:
    costs = _recovered_costs()
    costs.pop(RECOVERY_KEYS[-1])
    with pytest.raises(core.P05CoreError):
        reporting._check_public_costs({"public_costs": costs}, reporting_inputs)


def test_partial_recovery_extras_rejected_with_constants_module() -> None:
    costs = _recovered_costs()
    costs.pop(RECOVERY_KEYS[0])
    with pytest.raises(core.P05CoreError):
        reporting._check_public_costs({"public_costs": costs}, _Constants())


def test_gate_accepts_complete_recovered_costs() -> None:
    accounting = _normalized_accounting()
    costs = _recovered_costs()
    publication_gate._check_cost_agreement(costs, _receipt(costs, accounting))


def test_gate_rejects_clean_costs_for_recovered_receipt() -> None:
    accounting = _normalized_accounting()
    costs = _clean_costs()
    with pytest.raises(core.P05CoreError):
        publication_gate._check_cost_agreement(costs, _receipt(costs, accounting))


def test_gate_rejects_partial_recovery_extras() -> None:
    accounting = _normalized_accounting()
    costs = _recovered_costs()
    costs.pop(RECOVERY_KEYS[-1])
    with pytest.raises(core.P05CoreError):
        publication_gate._check_cost_agreement(costs, _receipt(costs, accounting))


def test_gate_rejects_modified_source_charged_upper() -> None:
    accounting = _normalized_accounting()
    costs = _recovered_costs()
    costs["source_optimizer_updates_charged_upper_bound"] += 1
    with pytest.raises(core.P05CoreError):
        publication_gate._check_cost_agreement(costs, _receipt(costs, accounting))


def test_gate_rejects_modified_combined_charged_upper() -> None:
    accounting = _normalized_accounting()
    costs = _recovered_costs()
    costs["combined_optimizer_updates_charged_upper_bound"] += 1
    with pytest.raises(core.P05CoreError):
        publication_gate._check_cost_agreement(costs, _receipt(costs, accounting))


def test_gate_rejects_modified_attempt_count() -> None:
    accounting = _normalized_accounting()
    costs = _recovered_costs()
    costs["new_source_attempts"] += 1
    with pytest.raises(core.P05CoreError):
        publication_gate._check_cost_agreement(costs, _receipt(costs, accounting))


def test_gate_rejects_modified_recovery_seconds() -> None:
    accounting = _normalized_accounting()
    costs = _recovered_costs()
    costs["recovery_source_scientific_seconds_charged_upper_bound"] += 1.0
    with pytest.raises(core.P05CoreError):
        publication_gate._check_cost_agreement(costs, _receipt(costs, accounting))
