"""P05 reporting-inputs read-only helper.

Consumes the already-authenticated comprehensive-comparison authority result and
exposes the exact reporting inputs derived from the frozen source, selection and
refit evidence: the completed source selector records, a strict allowlisted
numeric cost summary and a hash-only binding map.  It never re-authenticates the
chain, never selects a context, trains, infers, calibrates or writes.  The
companion ``verify_reporting_sources`` re-hashes the exact fixed path map and
re-verifies the develop, selection, refit and comparison stage manifests to
reject post-hoc mutation.  Every failure is a stable, path-free reason code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_comparison_authority as comparison_authority
from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_evaluation_authority as evaluation_authority
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_recovery_source as source
from atlas_sers.evaluation import p05_refit_authority as refit_authority

__all__ = [
    "ReportingInputsError",
    "load_reporting_sources",
    "verify_reporting_sources",
]

NEW_SOURCE_FITS = 14904
REUSED_PILOT_FITS = 36
SOURCE_EVIDENCE_FITS = 14940
STRATEGY_ALIASES = 2880

RECOVERY_SOURCE_ATTEMPTS = 14905
RECOVERY_INTERRUPTED_SOURCE_ATTEMPTS = 1
RECOVERY_REPLAYED_SOURCE_ATTEMPTS = 1
RECOVERY_OBSERVED_STEP_OFFSET = 68
RECOVERY_CHARGED_STEP_OFFSET = 800
RECOVERY_PRIOR_SOURCE_SECONDS = 36000.0
RECOVERY_MAXIMUM_NEW_OPTIMIZER_STEPS = 14228000

DEVELOP_MANIFEST_NAME = "manifest.json"
SOURCE_LEDGER_NAME = "source_ledger.json"
SELECTOR_NAME = "selector.jsonl"
SUMMARY_NAME = "summary.json"

LEGACY_PUBLIC_COST_KEYS = frozenset(
    {
        "new_source_fits",
        "reused_pilot_fits",
        "source_evidence_fits",
        "unique_refitted_models",
        "unique_scalar_calibrations",
        "new_neural_fits_total",
        "strategy_alias_count",
        "new_source_optimizer_updates",
        "refit_optimizer_updates",
        "combined_new_optimizer_updates",
        "source_scientific_seconds",
        "refit_scientific_seconds",
        "scientific_seconds_cumulative_bound_through_comparison",
        "refit_peak_allocated_gpu_bytes",
    }
)

REQUIRED_PUBLIC_COST_KEYS = LEGACY_PUBLIC_COST_KEYS - {"refit_peak_allocated_gpu_bytes"}

RECOVERY_PUBLIC_COST_KEYS = frozenset(
    {
        "new_source_attempts",
        "original_interrupted_source_attempts",
        "replayed_source_attempts",
        "new_neural_attempts_total",
        "source_optimizer_updates_observed_lower_bound",
        "source_optimizer_updates_charged_upper_bound",
        "combined_optimizer_updates_observed_lower_bound",
        "combined_optimizer_updates_charged_upper_bound",
        "prior_source_scientific_seconds_charged_upper_bound",
        "recovery_source_scientific_seconds_charged_upper_bound",
    }
)

PUBLIC_COST_KEYS = LEGACY_PUBLIC_COST_KEYS | RECOVERY_PUBLIC_COST_KEYS

_RECOVERY_INTEGRAL_COST_KEYS = (
    "new_source_attempts",
    "original_interrupted_source_attempts",
    "replayed_source_attempts",
    "new_neural_attempts_total",
    "source_optimizer_updates_observed_lower_bound",
    "source_optimizer_updates_charged_upper_bound",
    "combined_optimizer_updates_observed_lower_bound",
    "combined_optimizer_updates_charged_upper_bound",
)

_RECOVERY_SECONDS_COST_KEYS = (
    "prior_source_scientific_seconds_charged_upper_bound",
    "recovery_source_scientific_seconds_charged_upper_bound",
)

BINDING_KEYS = frozenset(
    {
        "development_receipt_sha256",
        "develop_manifest_sha256",
        "source_ledger_sha256",
        "selector_sha256",
        "selection_receipt_sha256",
        "selection_manifest_sha256",
        "source_bindings_sha256",
        "refit_receipt_sha256",
        "refit_manifest_sha256",
        "comparison_receipt_sha256",
        "comparison_manifest_sha256",
    }
)


class ReportingInputsError(core.P05CoreError):
    """Stable, path-free reporting-input failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise ReportingInputsError(code)


def _finite_seconds(value: Any, code: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), code)
    number = float(value)
    _require(math.isfinite(number), code)
    return number


def _strict_int(value: Any, code: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), code)
    return int(value)


def _hex64(value: Any, code: str) -> str:
    _require(isinstance(value, str) and core._is_hex64(value), code)
    return value


def _stable_file_sha256(path: Path, code: str) -> str:
    core._reject_symlink_chain(path)
    _require(path.is_file() and not path.is_symlink(), code)
    return core._canon().sha256_file(path)


def _run_root(bundle: Mapping[str, Any]) -> Path:
    artifact_root = bundle.get("artifact_root")
    _require(artifact_root is not None, "artifact_root_missing")
    permit_sha256 = bundle.get("permit_sha256")
    _require(isinstance(permit_sha256, str) and bool(permit_sha256), "permit_sha256_malformed")
    return (
        Path(artifact_root)
        / comparison_authority.COMPREHENSIVE_DIR
        / comparison_authority.RUNS_DIR
        / permit_sha256
    )


def _selection_bundle(bundle: Mapping[str, Any], accounting: Mapping[str, Any]) -> dict[str, Any]:
    local_bundle = dict(bundle)
    local_bundle["source_execution_accounting"] = accounting
    return local_bundle


def _check_constants() -> None:
    _require(development.MAXIMUM_NEW_FITS == NEW_SOURCE_FITS, "source_fit_constant_changed")
    _require(
        development.INNER_SLOT_COUNT == SOURCE_EVIDENCE_FITS, "source_evidence_constant_changed"
    )
    _require(development.REUSED_PILOT_SLOTS == REUSED_PILOT_FITS, "reused_pilot_constant_changed")
    _require(
        NEW_SOURCE_FITS + REUSED_PILOT_FITS == SOURCE_EVIDENCE_FITS,
        "source_fit_partition_mismatch",
    )
    _require(freeze.STRATEGY_ALIAS_COUNT == STRATEGY_ALIASES, "strategy_alias_constant_changed")
    _require(
        evaluation_authority.STRATEGY_ALIAS_COUNT == STRATEGY_ALIASES,
        "strategy_alias_constant_changed",
    )
    _require(
        evaluation_authority.MAXIMUM_REFITS == STRATEGY_ALIASES,
        "maximum_refit_constant_changed",
    )
    _require(
        evaluation_authority.SOURCE_FIT_COUNT == NEW_SOURCE_FITS,
        "source_fit_count_constant_changed",
    )
    _require(
        evaluation_authority.REUSED_PILOT_FIT_COUNT == REUSED_PILOT_FITS,
        "reused_pilot_fit_constant_changed",
    )
    _require(
        development.PRELAUNCH_AUDIT_RESERVE_SECONDS
        == evaluation_authority.PRELAUNCH_AUDIT_RESERVE_SECONDS,
        "reserve_constant_mismatch",
    )
    _require(
        development.MAXIMUM_TOTAL_SECONDS == evaluation_authority.MAXIMUM_TOTAL_SECONDS,
        "maximum_total_constant_mismatch",
    )


def _check_comparison_receipt(
    bundle: Mapping[str, Any], authenticated: Mapping[str, Any], plan_id: str
) -> tuple[Mapping[str, Any], str, str]:
    expected = authenticated.get("comparison_receipt")
    _require(isinstance(expected, Mapping), "authenticated_comparison_receipt_missing")
    run_root = _run_root(bundle)
    receipt_path = run_root / comparison_authority.RECEIPT_NAME
    receipt_sha256 = _stable_file_sha256(receipt_path, "comparison_receipt_missing")
    receipt = core._read_json(receipt_path, "comparison_receipt")
    _require(isinstance(receipt, Mapping), "comparison_receipt_malformed")
    _require(dict(receipt) == dict(expected), "comparison_receipt_changed")
    manifest_path = run_root / comparison_authority.STAGE_NAME / comparison_authority.MANIFEST_NAME
    manifest_sha256 = _stable_file_sha256(manifest_path, "comparison_manifest_missing")
    _require(
        receipt.get("stage_manifest_sha256") == manifest_sha256,
        "comparison_manifest_digest_mismatch",
    )
    _require(receipt.get("status") == "complete", "comparison_status_incomplete")
    _require(receipt.get("comparison_complete") is True, "comparison_incomplete")
    _require(
        receipt.get("permit_sha256") == bundle.get("permit_sha256"), "comparison_permit_mismatch"
    )
    _require(str(receipt.get("selection_plan_id")) == plan_id, "comparison_plan_id_mismatch")
    pilot._verify_manifest(run_root / comparison_authority.STAGE_NAME)
    _require(
        core._canon().sha256_file(receipt_path) == receipt_sha256,
        "comparison_receipt_changed",
    )
    return receipt, receipt_sha256, manifest_sha256


def _check_plan_shape(plan: Mapping[str, Any]) -> tuple[Mapping[str, Any], Sequence[Any]]:
    unique_refits = plan.get("unique_refits")
    aliases = plan.get("strategy_aliases")
    _require(isinstance(unique_refits, Mapping), "plan_unique_refits_malformed")
    _require(
        isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)),
        "plan_aliases_malformed",
    )
    unique_count = len(unique_refits)
    _require(
        0 < unique_count <= evaluation_authority.MAXIMUM_REFITS,
        "unique_refit_count_out_of_range",
    )
    _require(len(aliases) == STRATEGY_ALIASES, "strategy_alias_count_mismatch")
    for alias in aliases:
        _require(isinstance(alias, Mapping), "plan_alias_malformed")
        _require(str(alias.get("refit_id")) in unique_refits, "plan_alias_refit_unknown")
    return unique_refits, aliases


def _check_development_counts(receipt: Mapping[str, Any]) -> None:
    _require(
        _strict_int(receipt.get("new_completions"), "development_new_completions_malformed")
        == NEW_SOURCE_FITS,
        "development_new_completions_mismatch",
    )
    _require(
        _strict_int(receipt.get("selector_records"), "development_selector_records_malformed")
        == SOURCE_EVIDENCE_FITS,
        "development_selector_records_mismatch",
    )


def _check_refit_receipt(
    bundle: Mapping[str, Any],
    run_root: Path,
    receipt: Mapping[str, Any],
    plan_id: str,
    unique_count: int,
    alias_count: int,
    source_steps: int,
    refit_steps: int,
    *,
    source_accounting: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    _require(
        str(receipt.get("schema_version")) == evaluation_authority.SCHEMA,
        "refit_receipt_schema_mismatch",
    )
    _require(
        str(receipt.get("stage")) == evaluation_authority.STAGE_NAME,
        "refit_receipt_stage_mismatch",
    )
    _require(
        str(receipt.get("command")) == evaluation_authority.COMMAND,
        "refit_receipt_command_mismatch",
    )
    _require(
        str(receipt.get("claim")) == evaluation_authority.CLAIM,
        "refit_receipt_claim_mismatch",
    )
    _require(str(receipt.get("status")) == "complete", "refit_receipt_status_incomplete")
    _require(receipt.get("refits_complete") is True, "refit_receipt_refits_incomplete")
    _require(receipt.get("calibrations_complete") is True, "refit_receipt_calibrations_incomplete")
    _require(
        _strict_int(
            receipt.get("outer_predictions_started"), "refit_receipt_outer_predictions_malformed"
        )
        == 0,
        "refit_receipt_outer_predictions_started",
    )
    ledger = bundle.get("ledger")
    _require(isinstance(ledger, Mapping), "ledger_malformed")
    for name, expected in (
        ("permit_sha256", bundle.get("permit_sha256")),
        ("core_contract_sha256", bundle.get("contract_sha256")),
        ("core_plan_id", bundle.get("core_plan_id")),
        ("ledger_id", ledger.get("ledger_id")),
    ):
        _require(str(receipt.get(name)) == str(expected), f"refit_receipt_{name}_mismatch")
    _require(str(receipt.get("selection_plan_id")) == plan_id, "refit_receipt_plan_id_mismatch")
    _require(
        _strict_int(receipt.get("source_fit_count"), "refit_receipt_source_fit_count_malformed")
        == NEW_SOURCE_FITS,
        "refit_receipt_source_fit_count_mismatch",
    )
    _require(
        _strict_int(
            receipt.get("reused_pilot_fit_count"), "refit_receipt_reused_pilot_fit_count_malformed"
        )
        == REUSED_PILOT_FITS,
        "refit_receipt_reused_pilot_fit_count_mismatch",
    )
    _require(
        _strict_int(receipt.get("unique_refit_count"), "refit_receipt_unique_refit_count_malformed")
        == unique_count,
        "refit_receipt_unique_refit_count_mismatch",
    )
    _require(
        _strict_int(
            receipt.get("strategy_alias_count"), "refit_receipt_strategy_alias_count_malformed"
        )
        == alias_count,
        "refit_receipt_strategy_alias_count_mismatch",
    )
    _require(
        _strict_int(
            receipt.get("source_optimizer_steps"), "refit_receipt_source_optimizer_steps_malformed"
        )
        == source_steps,
        "refit_receipt_source_optimizer_steps_mismatch",
    )
    counters = receipt.get("counters")
    _require(isinstance(counters, Mapping), "refit_receipt_counters_malformed")
    _require(
        set(counters) == set(evaluation_authority.COUNTER_KEYS),
        "refit_receipt_counter_fields_mismatch",
    )
    for prefix in ("calibration", "neural"):
        _require(
            _strict_int(counters.get(f"{prefix}_started"), f"refit_{prefix}_started_malformed")
            == unique_count,
            f"refit_{prefix}_started_mismatch",
        )
        _require(
            _strict_int(counters.get(f"{prefix}_completed"), f"refit_{prefix}_completed_malformed")
            == unique_count,
            f"refit_{prefix}_completed_mismatch",
        )
        _require(
            _strict_int(counters.get(f"{prefix}_failed"), f"refit_{prefix}_failed_malformed") == 0,
            f"refit_{prefix}_failures_present",
        )
    _require(
        _strict_int(counters.get("optimizer_steps"), "refit_receipt_optimizer_steps_malformed")
        == refit_steps,
        "refit_receipt_optimizer_steps_mismatch",
    )
    _require(counters.get("optimizer_steps_exact") is True, "refit_receipt_optimizer_steps_inexact")
    total_new = _strict_int(
        receipt.get("total_new_optimizer_steps"),
        "refit_receipt_total_new_optimizer_steps_malformed",
    )
    _require(
        total_new == source_steps + refit_steps, "refit_receipt_total_new_optimizer_steps_mismatch"
    )
    if source_accounting is None:
        _require(
            total_new <= evaluation_authority.MAXIMUM_COMBINED_UPDATES,
            "refit_receipt_combined_updates_exceeded",
        )
        _require(
            "source_execution_accounting" not in receipt,
            "refit_receipt_source_execution_accounting_unexpected",
        )
        _require(
            "total_new_optimizer_steps_charged_upper_bound" not in receipt,
            "refit_receipt_total_new_optimizer_steps_charged_upper_bound_unexpected",
        )
        _require(
            "total_new_optimizer_steps_all_attempts_exact" not in receipt,
            "refit_receipt_total_new_optimizer_steps_all_attempts_exact_unexpected",
        )
    else:
        source_accounting = source.validate_accounting(
            source_accounting, source_optimizer_steps=source_steps
        )
        _require(source_accounting["mode"] == "recovered", "recovery_accounting_mode_invalid")
        normalized = source.validate_accounting(
            receipt.get("source_execution_accounting"),
            source_optimizer_steps=source_steps,
        )
        _require(
            dict(normalized) == dict(source_accounting),
            "refit_receipt_source_execution_accounting_mismatch",
        )
        source_charged = _strict_int(
            normalized.get("source_optimizer_steps_charged_upper_bound"),
            "refit_receipt_source_charged_upper_bound_malformed",
        )
        combined_charged = _strict_int(
            receipt.get("total_new_optimizer_steps_charged_upper_bound"),
            "refit_receipt_total_new_optimizer_steps_charged_upper_bound_malformed",
        )
        _require(
            combined_charged == source_charged + refit_steps,
            "refit_receipt_total_new_optimizer_steps_charged_upper_bound_mismatch",
        )
        maximum_new_steps = _strict_int(
            normalized.get("maximum_new_optimizer_steps"),
            "refit_receipt_maximum_new_optimizer_steps_malformed",
        )
        _require(
            combined_charged <= maximum_new_steps,
            "refit_receipt_total_new_optimizer_steps_charged_upper_bound_exceeded",
        )
        _require(
            receipt.get("total_new_optimizer_steps_all_attempts_exact") is False,
            "refit_receipt_total_new_optimizer_steps_all_attempts_exact_not_false",
        )
    manifest_path = run_root / evaluation_authority.STAGE_NAME / evaluation_authority.MANIFEST_NAME
    manifest_sha256 = _stable_file_sha256(manifest_path, "refit_manifest_missing")
    _require(
        manifest_sha256 == str(receipt.get("stage_manifest_sha256")),
        "refit_manifest_digest_mismatch",
    )
    pilot._verify_manifest(run_root / evaluation_authority.STAGE_NAME)
    return manifest_sha256, dict(counters)


def _check_develop_artifacts(
    paths: Mapping[str, Path], development_receipt: Mapping[str, Any]
) -> tuple[str, str]:
    develop_stage = paths["develop"]
    manifest_path = develop_stage / DEVELOP_MANIFEST_NAME
    manifest_sha256 = _stable_file_sha256(manifest_path, "develop_manifest_missing")
    _require(
        manifest_sha256 == str(development_receipt.get("stage_manifest_sha256")),
        "develop_manifest_digest_mismatch",
    )
    source_ledger_path = develop_stage / SOURCE_LEDGER_NAME
    source_ledger_sha256 = _stable_file_sha256(source_ledger_path, "source_ledger_missing")
    pilot._verify_manifest(develop_stage)
    return manifest_sha256, source_ledger_sha256


def _check_selection(
    bundle: Mapping[str, Any],
    paths: Mapping[str, Path],
    plan: Mapping[str, Any],
    plan_id: str,
) -> tuple[str, str, str]:
    selection_stage = paths["selection"]
    receipt_path = paths["selection_receipt"]
    receipt_sha256 = _stable_file_sha256(receipt_path, "selection_receipt_missing")
    receipt = freeze._read_mapping(receipt_path, "selection_receipt_missing")
    _require(
        str(receipt.get("stage")) == freeze.SELECTION_STAGE_NAME,
        "selection_receipt_stage_mismatch",
    )
    _require(str(receipt.get("selection_plan_id")) == plan_id, "selection_receipt_plan_id_mismatch")
    manifest_sha256 = _hex64(
        receipt.get("selection_manifest_sha256"), "selection_receipt_manifest_sha_malformed"
    )
    refit_authority._verify_selection_manifest(selection_stage, manifest_sha256)
    actual_manifest_sha256 = _stable_file_sha256(
        selection_stage / refit_authority.SELECTION_MANIFEST_NAME, "selection_manifest_missing"
    )
    _require(actual_manifest_sha256 == manifest_sha256, "selection_manifest_digest_mismatch")
    bindings_path = selection_stage / refit_authority.SELECTION_BINDINGS_NAME
    bindings_sha256 = _stable_file_sha256(bindings_path, "selection_bindings_missing")
    refit_authority._check_source_bindings(
        bundle, selection_stage, paths["develop"], paths["develop"] / SELECTOR_NAME
    )
    stored_plan = freeze._read_mapping(
        selection_stage / refit_authority.SELECTION_PLAN_NAME, "selection_plan_missing"
    )
    _require(dict(stored_plan) == dict(plan), "selection_plan_mismatch")
    _require(core._canon().sha256_file(receipt_path) == receipt_sha256, "selection_receipt_changed")
    return receipt_sha256, actual_manifest_sha256, bindings_sha256


def _read_selector(paths: Mapping[str, Path], deadline: float) -> tuple[list[dict[str, Any]], str]:
    selector_path = paths["develop"] / SELECTOR_NAME
    selector_sha256 = _stable_file_sha256(selector_path, "selector_missing")
    records = freeze._read_jsonl(selector_path, "selector_missing")
    _require(len(records) == SOURCE_EVIDENCE_FITS, "selector_count_mismatch")
    for record in records:
        freeze._check_deadline(deadline)
        status = record.get("status")
        _require(isinstance(status, str) and bool(status), "selector_status_malformed")
        _require(status == "complete", "selector_record_incomplete")
    _require(core._canon().sha256_file(selector_path) == selector_sha256, "selector_changed")
    return records, selector_sha256


def _assemble_costs(
    development_receipt: Mapping[str, Any],
    refit_receipt: Mapping[str, Any],
    refit_counters: Mapping[str, Any],
    comparison_receipt: Mapping[str, Any],
    unique_count: int,
    source_steps: int,
    refit_steps: int,
    *,
    source_accounting: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    recovery = source_accounting is not None
    development_stage_seconds = _finite_seconds(
        development_receipt.get("scientific_seconds_this_stage"), "development_seconds_malformed"
    )
    if recovery:
        source_accounting = source.validate_accounting(
            source_accounting, source_optimizer_steps=source_steps
        )
        _require(source_accounting["mode"] == "recovered", "recovery_accounting_mode_invalid")
        source_charged = _strict_int(
            source_accounting.get("source_optimizer_steps_charged_upper_bound"),
            "source_accounting_charged_upper_bound_malformed",
        )
        maximum_new_steps = _strict_int(
            source_accounting.get("maximum_new_optimizer_steps"),
            "source_accounting_maximum_new_optimizer_steps_malformed",
        )
        _require(
            source_charged + refit_steps <= maximum_new_steps,
            "source_accounting_charged_budget_exceeded",
        )
        source_seconds = RECOVERY_PRIOR_SOURCE_SECONDS + development_stage_seconds
    else:
        source_seconds = development_stage_seconds
    refit_seconds = _finite_seconds(
        refit_receipt.get("scientific_seconds_this_stage"), "refit_seconds_malformed"
    )
    total_seconds = _finite_seconds(
        comparison_receipt.get("scientific_seconds_cumulative_bound"),
        "comparison_cumulative_seconds_malformed",
    )
    _require(
        0.0 <= source_seconds <= development.MAXIMUM_TOTAL_SECONDS,
        "development_seconds_out_of_range",
    )
    _require(
        0.0 <= refit_seconds <= development.MAXIMUM_TOTAL_SECONDS, "refit_seconds_out_of_range"
    )
    _require(
        0.0 <= total_seconds <= development.MAXIMUM_TOTAL_SECONDS,
        "comparison_cumulative_seconds_out_of_range",
    )
    _require(
        total_seconds
        >= development.PRELAUNCH_AUDIT_RESERVE_SECONDS + source_seconds + refit_seconds,
        "cumulative_seconds_inconsistent",
    )
    combined = source_steps + refit_steps
    if not recovery:
        _require(
            combined <= evaluation_authority.MAXIMUM_COMBINED_UPDATES,
            "combined_updates_exceeded",
        )
    costs: dict[str, Any] = {
        "new_source_fits": NEW_SOURCE_FITS,
        "reused_pilot_fits": REUSED_PILOT_FITS,
        "source_evidence_fits": SOURCE_EVIDENCE_FITS,
        "unique_refitted_models": unique_count,
        "unique_scalar_calibrations": unique_count,
        "new_neural_fits_total": NEW_SOURCE_FITS + unique_count,
        "strategy_alias_count": STRATEGY_ALIASES,
        "new_source_optimizer_updates": source_steps,
        "refit_optimizer_updates": refit_steps,
        "combined_new_optimizer_updates": combined,
        "source_scientific_seconds": source_seconds,
        "refit_scientific_seconds": refit_seconds,
        "scientific_seconds_cumulative_bound_through_comparison": total_seconds,
    }
    peak = refit_counters.get("peak_cuda_bytes")
    if peak is not None:
        peak_bytes = _strict_int(peak, "refit_peak_cuda_malformed")
        _require(
            0 <= peak_bytes <= evaluation_authority.MAXIMUM_CUDA_ALLOCATED_BYTES,
            "refit_peak_cuda_out_of_range",
        )
        costs["refit_peak_allocated_gpu_bytes"] = peak_bytes
    if recovery:
        costs.update(
            _recovery_cost_extras(
                unique_count=unique_count,
                source_steps=source_steps,
                refit_steps=refit_steps,
                recovery_source_seconds=development_stage_seconds,
            )
        )
    _check_public_costs(costs)
    return costs


def _check_public_costs(costs: Mapping[str, Any]) -> None:
    _require(set(costs) <= PUBLIC_COST_KEYS, "public_cost_key_not_allowed")
    _require(REQUIRED_PUBLIC_COST_KEYS <= set(costs), "public_cost_key_missing")
    for name, value in costs.items():
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"public_cost_{name}_malformed",
        )
        if isinstance(value, float):
            _require(math.isfinite(value), f"public_cost_{name}_not_finite")
            _require(value >= 0.0, f"public_cost_{name}_out_of_range")
        else:
            _require(value >= 0, f"public_cost_{name}_out_of_range")
    if set(costs) & RECOVERY_PUBLIC_COST_KEYS:
        _require(RECOVERY_PUBLIC_COST_KEYS <= set(costs), "recovery_public_cost_key_missing")
        _check_recovery_public_costs(costs)


def _check_recovery_public_costs(costs: Mapping[str, Any]) -> None:
    for key in (
        "new_source_fits",
        "reused_pilot_fits",
        "source_evidence_fits",
        "unique_refitted_models",
        "unique_scalar_calibrations",
        "new_neural_fits_total",
        "strategy_alias_count",
        "new_source_optimizer_updates",
        "refit_optimizer_updates",
        "combined_new_optimizer_updates",
    ):
        _strict_int(costs[key], f"public_cost_{key}_malformed")
    unique = costs["unique_refitted_models"]
    _require(0 < unique <= 2880, "public_cost_refit_count_out_of_range")
    for key, value in (
        ("new_source_fits", NEW_SOURCE_FITS),
        ("reused_pilot_fits", REUSED_PILOT_FITS),
        ("source_evidence_fits", SOURCE_EVIDENCE_FITS),
        ("unique_scalar_calibrations", unique),
        ("new_neural_fits_total", NEW_SOURCE_FITS + unique),
        ("strategy_alias_count", STRATEGY_ALIASES),
        (
            "combined_new_optimizer_updates",
            costs["new_source_optimizer_updates"] + costs["refit_optimizer_updates"],
        ),
    ):
        _require(costs[key] == value, f"public_cost_{key}_mismatch")
    for name in _RECOVERY_INTEGRAL_COST_KEYS:
        value = costs[name]
        _require(
            isinstance(value, int) and not isinstance(value, bool),
            f"public_cost_{name}_malformed",
        )
    for name in _RECOVERY_SECONDS_COST_KEYS:
        value = costs[name]
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"public_cost_{name}_malformed",
        )
        _require(math.isfinite(float(value)), f"public_cost_{name}_not_finite")
        _require(float(value) >= 0.0, f"public_cost_{name}_out_of_range")
    expected = _recovery_cost_extras(
        unique_count=costs["unique_refitted_models"],
        source_steps=costs["new_source_optimizer_updates"],
        refit_steps=costs["refit_optimizer_updates"],
        recovery_source_seconds=costs["recovery_source_scientific_seconds_charged_upper_bound"],
    )
    for name, value in expected.items():
        _require(costs[name] == value, f"public_cost_{name}_mismatch")
    _require(
        costs["source_scientific_seconds"]
        == RECOVERY_PRIOR_SOURCE_SECONDS
        + costs["recovery_source_scientific_seconds_charged_upper_bound"],
        "public_cost_source_scientific_seconds_mismatch",
    )
    _require(
        costs["combined_optimizer_updates_charged_upper_bound"]
        <= RECOVERY_MAXIMUM_NEW_OPTIMIZER_STEPS,
        "public_cost_combined_charges_exceed_budget",
    )
    _require(costs["new_neural_attempts_total"] <= 17785, "public_cost_attempts_exceeded")
    _require(
        development.PRELAUNCH_AUDIT_RESERVE_SECONDS
        + costs["source_scientific_seconds"]
        + costs["refit_scientific_seconds"]
        <= costs["scientific_seconds_cumulative_bound_through_comparison"]
        <= development.MAXIMUM_TOTAL_SECONDS,
        "public_cost_cumulative_seconds_inconsistent",
    )


def _check_bindings(bindings: Mapping[str, Any]) -> None:
    _require(set(bindings) == BINDING_KEYS, "binding_key_set_mismatch")
    for name, value in bindings.items():
        _hex64(value, f"binding_{name}_malformed")


def _recovery_cost_extras(
    *,
    unique_count: int,
    source_steps: int,
    refit_steps: int,
    recovery_source_seconds: float,
) -> dict[str, Any]:
    combined_steps = source_steps + refit_steps
    return {
        "new_source_attempts": RECOVERY_SOURCE_ATTEMPTS,
        "original_interrupted_source_attempts": RECOVERY_INTERRUPTED_SOURCE_ATTEMPTS,
        "replayed_source_attempts": RECOVERY_REPLAYED_SOURCE_ATTEMPTS,
        "new_neural_attempts_total": RECOVERY_SOURCE_ATTEMPTS + unique_count,
        "source_optimizer_updates_observed_lower_bound": source_steps
        + RECOVERY_OBSERVED_STEP_OFFSET,
        "source_optimizer_updates_charged_upper_bound": source_steps + RECOVERY_CHARGED_STEP_OFFSET,
        "combined_optimizer_updates_observed_lower_bound": combined_steps
        + RECOVERY_OBSERVED_STEP_OFFSET,
        "combined_optimizer_updates_charged_upper_bound": combined_steps
        + RECOVERY_CHARGED_STEP_OFFSET,
        "prior_source_scientific_seconds_charged_upper_bound": RECOVERY_PRIOR_SOURCE_SECONDS,
        "recovery_source_scientific_seconds_charged_upper_bound": recovery_source_seconds,
    }


def load_reporting_sources(
    bundle: Mapping[str, Any], *, authenticated: Mapping[str, Any], deadline: Any
) -> dict[str, Any]:
    """Return the authenticated reporting selector records, costs and bindings."""

    _require(isinstance(bundle, Mapping), "bundle_malformed")
    _require(isinstance(authenticated, Mapping), "authenticated_malformed")
    deadline = _finite_seconds(deadline, "deadline_malformed")
    freeze._check_deadline(deadline)
    _check_constants()

    plan = authenticated.get("plan")
    _require(isinstance(plan, Mapping), "authority_plan_malformed")
    plan_id = str(plan.get("plan_id"))
    _require(core._is_hex64(plan_id), "authority_plan_id_malformed")

    comparison_receipt, comparison_receipt_sha256, comparison_manifest_sha256 = (
        _check_comparison_receipt(bundle, authenticated, plan_id)
    )
    freeze._check_deadline(deadline)

    source_steps = _strict_int(
        authenticated.get("source_optimizer_steps"), "authority_source_steps_malformed"
    )
    refit_steps = _strict_int(
        authenticated.get("refit_optimizer_steps"), "authority_refit_steps_malformed"
    )
    _require(source_steps >= 0, "authority_source_steps_out_of_range")
    _require(refit_steps >= 0, "authority_refit_steps_out_of_range")

    accounting = source.from_authenticated(authenticated)
    _require(isinstance(accounting, Mapping), "source_accounting_malformed")
    recovered = accounting.get("mode") == "recovered"

    unique_refits, _aliases = _check_plan_shape(plan)
    unique_count = len(unique_refits)
    alias_count = len(_aliases)

    paths = freeze._paths(bundle)
    run_root = _run_root(bundle)
    _require(paths["run_root"] == run_root, "run_root_mismatch")

    development_receipt_path = paths["receipt"]
    development_receipt_sha256 = _stable_file_sha256(
        development_receipt_path, "development_receipt_missing"
    )
    development_receipt = freeze._read_mapping(
        development_receipt_path, "development_receipt_missing"
    )
    freeze._check_receipt(development_receipt, freeze._expected_new_units(bundle))
    _check_development_counts(development_receipt)
    development_recovered = bool(source.is_recovered(development_receipt))
    _require(development_recovered == recovered, "source_recovery_mode_mismatch")
    if recovered:
        summary = freeze._read_mapping(paths["develop"] / SUMMARY_NAME, "source_summary_missing")
        recovered_accounting = source.accounting_from_recovered(summary, development_receipt)
        _require(
            dict(recovered_accounting) == dict(accounting),
            "source_recovery_accounting_mismatch",
        )
    _require(
        _strict_int(
            development_receipt.get("optimizer_steps"), "development_optimizer_steps_malformed"
        )
        == source_steps,
        "development_optimizer_steps_mismatch",
    )
    _require(
        core._canon().sha256_file(development_receipt_path) == development_receipt_sha256,
        "development_receipt_changed",
    )
    freeze._check_deadline(deadline)

    refit_receipt_path = run_root / evaluation_authority.RECEIPT_NAME
    refit_receipt_sha256 = _stable_file_sha256(refit_receipt_path, "refit_receipt_missing")
    refit_receipt = core._read_json(refit_receipt_path, "refit_receipt")
    _require(isinstance(refit_receipt, Mapping), "refit_receipt_malformed")
    refit_receipt = dict(refit_receipt)
    refit_manifest_sha256, refit_counters = _check_refit_receipt(
        bundle,
        run_root,
        refit_receipt,
        plan_id,
        unique_count,
        alias_count,
        source_steps,
        refit_steps,
        **({"source_accounting": accounting} if recovered else {}),
    )
    _require(
        core._canon().sha256_file(refit_receipt_path) == refit_receipt_sha256,
        "refit_receipt_changed",
    )
    freeze._check_deadline(deadline)

    develop_manifest_sha256, source_ledger_sha256 = _check_develop_artifacts(
        paths, development_receipt
    )
    freeze._check_deadline(deadline)

    selection_receipt_sha256, selection_manifest_sha256, source_bindings_sha256 = _check_selection(
        _selection_bundle(bundle, accounting) if recovered else bundle, paths, plan, plan_id
    )
    freeze._check_deadline(deadline)

    selector_records, selector_sha256 = _read_selector(paths, deadline)
    freeze._check_deadline(deadline)

    authority_total = _finite_seconds(
        authenticated.get("prior_seconds"), "authority_prior_malformed"
    )
    _require(
        authority_total
        == _finite_seconds(
            comparison_receipt.get("scientific_seconds_cumulative_bound"),
            "comparison_cumulative_seconds_malformed",
        ),
        "authority_prior_mismatch",
    )

    _require(
        core._canon().sha256_file(development_receipt_path) == development_receipt_sha256,
        "development_receipt_changed",
    )
    _require(
        core._canon().sha256_file(refit_receipt_path) == refit_receipt_sha256,
        "refit_receipt_changed",
    )
    _require(
        core._canon().sha256_file(paths["develop"] / DEVELOP_MANIFEST_NAME)
        == develop_manifest_sha256,
        "develop_manifest_changed",
    )
    _require(
        core._canon().sha256_file(paths["develop"] / SOURCE_LEDGER_NAME) == source_ledger_sha256,
        "source_ledger_changed",
    )
    _require(
        core._canon().sha256_file(paths["develop"] / SELECTOR_NAME) == selector_sha256,
        "selector_changed",
    )
    _require(
        core._canon().sha256_file(paths["selection_receipt"]) == selection_receipt_sha256,
        "selection_receipt_changed",
    )
    _require(
        core._canon().sha256_file(paths["selection"] / refit_authority.SELECTION_MANIFEST_NAME)
        == selection_manifest_sha256,
        "selection_manifest_changed",
    )
    _require(
        core._canon().sha256_file(paths["selection"] / refit_authority.SELECTION_BINDINGS_NAME)
        == source_bindings_sha256,
        "selection_bindings_changed",
    )
    _require(
        core._canon().sha256_file(run_root / comparison_authority.RECEIPT_NAME)
        == comparison_receipt_sha256,
        "comparison_receipt_changed",
    )
    _require(
        core._canon().sha256_file(
            run_root / comparison_authority.STAGE_NAME / comparison_authority.MANIFEST_NAME
        )
        == comparison_manifest_sha256,
        "comparison_manifest_changed",
    )
    _require(
        core._canon().sha256_file(
            run_root / evaluation_authority.STAGE_NAME / evaluation_authority.MANIFEST_NAME
        )
        == refit_manifest_sha256,
        "refit_manifest_changed",
    )
    freeze._check_deadline(deadline)

    public_costs = _assemble_costs(
        development_receipt,
        refit_receipt,
        refit_counters,
        comparison_receipt,
        unique_count,
        source_steps,
        refit_steps,
        **({"source_accounting": accounting} if recovered else {}),
    )
    bindings = {
        "development_receipt_sha256": development_receipt_sha256,
        "develop_manifest_sha256": develop_manifest_sha256,
        "source_ledger_sha256": source_ledger_sha256,
        "selector_sha256": selector_sha256,
        "selection_receipt_sha256": selection_receipt_sha256,
        "selection_manifest_sha256": selection_manifest_sha256,
        "source_bindings_sha256": source_bindings_sha256,
        "refit_receipt_sha256": refit_receipt_sha256,
        "refit_manifest_sha256": refit_manifest_sha256,
        "comparison_receipt_sha256": comparison_receipt_sha256,
        "comparison_manifest_sha256": comparison_manifest_sha256,
    }
    _check_bindings(bindings)
    verify_reporting_sources(bundle, bindings=bindings, deadline=deadline)

    return {
        "selector_records": selector_records,
        "public_costs": public_costs,
        "bindings": bindings,
    }


def verify_reporting_sources(
    bundle: Mapping[str, Any], *, bindings: Mapping[str, Any], deadline: Any
) -> dict[str, Any]:
    """Re-hash the fixed reporting path map and re-verify every stage manifest."""

    _require(isinstance(bundle, Mapping), "bundle_malformed")
    _require(isinstance(bindings, Mapping), "bindings_malformed")
    deadline = _finite_seconds(deadline, "deadline_malformed")
    freeze._check_deadline(deadline)
    _check_bindings(bindings)

    paths = freeze._paths(bundle)
    run_root = _run_root(bundle)
    _require(paths["run_root"] == run_root, "run_root_mismatch")

    known: dict[str, Path] = {
        "development_receipt_sha256": paths["receipt"],
        "develop_manifest_sha256": paths["develop"] / DEVELOP_MANIFEST_NAME,
        "source_ledger_sha256": paths["develop"] / SOURCE_LEDGER_NAME,
        "selector_sha256": paths["develop"] / SELECTOR_NAME,
        "selection_receipt_sha256": paths["selection_receipt"],
        "selection_manifest_sha256": paths["selection"] / refit_authority.SELECTION_MANIFEST_NAME,
        "source_bindings_sha256": paths["selection"] / refit_authority.SELECTION_BINDINGS_NAME,
        "refit_receipt_sha256": run_root / evaluation_authority.RECEIPT_NAME,
        "refit_manifest_sha256": (
            run_root / evaluation_authority.STAGE_NAME / evaluation_authority.MANIFEST_NAME
        ),
        "comparison_receipt_sha256": run_root / comparison_authority.RECEIPT_NAME,
        "comparison_manifest_sha256": (
            run_root / comparison_authority.STAGE_NAME / comparison_authority.MANIFEST_NAME
        ),
    }
    _require(set(known) == BINDING_KEYS, "binding_key_set_mismatch")
    for name, path in known.items():
        freeze._check_deadline(deadline)
        expected = _hex64(bindings.get(name), f"binding_{name}_malformed")
        core._reject_symlink_chain(path)
        _require(path.is_file() and not path.is_symlink(), "reporting_source_missing")
        _require(core._canon().sha256_file(path) == expected, f"binding_{name}_changed")

    for stage in (
        paths["develop"],
        paths["selection"],
        run_root / evaluation_authority.STAGE_NAME,
        run_root / comparison_authority.STAGE_NAME,
    ):
        freeze._check_deadline(deadline)
        core._reject_symlink_chain(stage)
        _require(stage.is_dir() and not stage.is_symlink(), "reporting_stage_missing")
        pilot._verify_manifest(stage)

    # Receipts live outside their stage manifests. Recheck them (and the
    # pinned selectors/manifests) after the full inventory scans as well.
    for name, path in known.items():
        _require(
            _stable_file_sha256(path, "reporting_source_missing") == bindings[name],
            f"binding_{name}_changed",
        )
    freeze._check_deadline(deadline)
    return {"verified": True, "bindings": dict(bindings)}
