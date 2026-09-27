"""P05 comprehensive read-only selection freeze boundary.

Bounded, supervisor-gated stage after the P05 comprehensive source-development
stage.  It re-authenticates the development receipt/summary/manifest and the
frozen pilot run, reconstructs every selector record from the persisted
execution summaries, and freezes the deterministic refit plan for all 320
contexts.  No logits, checkpoint or spectrum is loaded, no fit, refit or
outer-test prediction runs, no global winning recipe is chosen and no refit is
launched.  Read-only stage time is charged against the single cumulative
48-hour bound recorded by the development receipt with no reset.  The exclusive
single-use ``selection`` stage directory holds the frozen plan, source
bindings, before/after provenance, summary and manifest; a path-free
``selection_receipt.json`` is written at the run root.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_recovery_acceptance as acceptance
from atlas_sers.evaluation import p05_recovery_receipt as recovery_receipt
from atlas_sers.evaluation import p05_recovery_source as source
from atlas_sers.evaluation.p05_comprehensive_storage import (
    P05StorageError,
    StorageBudget,
)
from atlas_sers.evaluation.p05_core_run import P05CoreError
from atlas_sers.evaluation.p05_refit_plan import (
    MAXIMUM_STRATEGY_ALIAS_COUNT,
    RefitPlanError,
    build_refit_plan,
)

__all__ = ["FreezeSelectionError", "freeze_selection"]

NAMESPACE = development.NAMESPACE
SCHEMA_VERSION = "nato-sers-p05-comprehensive-freeze-v1"
PROTOCOL_VERSION = development.PROTOCOL_VERSION
SELECTION_CLAIM = "read_only_source_selection_freeze_no_refit_no_outer_evaluation"
DEVELOP_STAGE_NAME = "develop"
SELECTION_STAGE_NAME = "selection"
RECEIPT_NAME = "development_receipt.json"
SELECTION_RECEIPT_NAME = "selection_receipt.json"

MAXIMUM_TOTAL_SECONDS = development.MAXIMUM_TOTAL_SECONDS
STORAGE_CEILING_BYTES = development.STORAGE_CEILING_BYTES
PRELAUNCH_AUDIT_RESERVE_SECONDS = development.PRELAUNCH_AUDIT_RESERVE_SECONDS
MAXIMUM_NEW_FITS = development.MAXIMUM_NEW_FITS
INNER_SLOT_COUNT = development.INNER_SLOT_COUNT
REUSED_PILOT_SLOTS = development.REUSED_PILOT_SLOTS
SLOTS_PER_UNIT = development.SLOTS_PER_UNIT
UNIT_COUNT = development.UNIT_COUNT
CONTEXT_COUNT = inputs.CONTEXT_COUNT
STRATEGY_ALIAS_COUNT = MAXIMUM_STRATEGY_ALIAS_COUNT
HEADROOM_BYTES = 8 * 1024 * 1024
SOURCE_ACCOUNTING_KEY = "source_execution_accounting"
SELECTION_ONLY_FLAGS = {
    "selection_only": True,
    "refit_authorized": False,
    "outer_evaluation_authorized": False,
}
_IDENTITY_EXPECTED = (
    ("permit_sha256", inputs.COMPREHENSIVE_PERMIT_SHA256),
    ("core_contract_sha256", inputs.CORE_CONTRACT_SHA256),
    ("core_plan_id", inputs.CORE_PLAN_ID),
    ("ledger_id", inputs.LEDGER_ID),
)


class FreezeSelectionError(P05CoreError):
    """Stable freeze-selection failure with a path-free reason code."""


def _err(reason_code: str) -> FreezeSelectionError:
    return FreezeSelectionError(reason_code)


def _canon() -> Any:
    return core._canon()


def _check_deadline(deadline: float) -> None:
    if time.perf_counter() > float(deadline):
        raise _err("global_deadline_exceeded")


def _paths(bundle: Mapping[str, Any]) -> dict[str, Path]:
    try:
        return source.resolve_paths(bundle)
    except source.RecoverySourceError as error:
        raise _err("source_paths_unresolved") from error


def _identity_fields(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "permit_sha256": str(bundle["permit_sha256"]),
        "core_contract_sha256": str(bundle["contract_sha256"]),
        "core_plan_id": str(bundle["core_plan_id"]),
        "ledger_id": str(bundle["ledger"]["ledger_id"]),
    }


def _check_identity(record: Mapping[str, Any], code: str) -> None:
    if any(str(record.get(key)) != value for key, value in _IDENTITY_EXPECTED):
        raise _err(code)


def _read_mapping(path: Path, code: str) -> dict[str, Any]:
    core._reject_symlink_chain(path)
    if not path.is_file():
        raise _err(code)
    value = core._read_json(path, code)
    if not isinstance(value, Mapping):
        raise _err(code)
    return dict(value)


def _read_jsonl(path: Path, code: str) -> list[dict[str, Any]]:
    core._reject_symlink_chain(path)
    if not path.is_file():
        raise _err(code)
    records: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line.decode("utf-8"))
            if not isinstance(record, Mapping):
                raise _err(code)
            records.append(dict(record))
    return records


def _reject_preexisting(path: Path, code: str) -> None:
    core._reject_symlink_chain(path)
    if path.exists() or path.is_symlink():
        raise _err(code)


def _budgeted_write(path: Path, payload: Any, budget: StorageBudget) -> None:
    content = _canon().canonical_json_bytes(payload)
    budget.check(headroom_bytes=HEADROOM_BYTES + len(content))
    core._atomic_write(path, content)
    budget.account_new_file(path)
    budget.check(headroom_bytes=HEADROOM_BYTES)


def _expected_new_units(bundle: Mapping[str, Any]) -> int:
    reused = len(bundle["pilot_bundle"]["units"])
    if reused != REUSED_PILOT_SLOTS // SLOTS_PER_UNIT:
        raise _err("pilot_unit_count_mismatch")
    return UNIT_COUNT - reused


def _check_receipt(receipt: Mapping[str, Any], expected_units: int) -> None:
    if source.is_recovered(receipt):
        try:
            recovery_receipt.validate_receipt(receipt, expected_units)
        except Exception as error:
            raise _err("receipt_recovery_invalid") from error
        return
    if str(receipt.get("schema_version")) != development.SCHEMA_VERSION:
        raise _err("receipt_schema_mismatch")
    if str(receipt.get("stage")) != DEVELOP_STAGE_NAME:
        raise _err("receipt_stage_mismatch")
    _check_identity(receipt, "receipt_identity_mismatch")
    for key, expected in (
        ("new_completions", MAXIMUM_NEW_FITS),
        ("selector_records", INNER_SLOT_COUNT),
        ("units_completed", expected_units),
    ):
        if int(receipt.get(key, -1)) != expected:
            raise _err("receipt_count_mismatch")
    if str(receipt.get("claim")) != development.COMPREHENSIVE_CLAIM:
        raise _err("receipt_claim_mismatch")
    if receipt.get("source_fits_only") is not True:
        raise _err("receipt_authorization_invalid")
    if receipt.get("outer_evaluation_authorized") is not False:
        raise _err("receipt_authorization_invalid")


def _check_prior_bound(receipt: Mapping[str, Any]) -> float:
    if source.is_recovered(receipt):
        try:
            validated = recovery_receipt.validate_receipt(receipt)
        except Exception as error:
            raise _err("receipt_recovery_invalid") from error
        cumulative = validated["scientific_seconds_cumulative_bound"]
        if isinstance(cumulative, bool) or not isinstance(cumulative, (int, float)):
            raise _err("receipt_cumulative_seconds_malformed")
        cumulative = float(cumulative)
        if not math.isfinite(cumulative) or cumulative <= 0.0:
            raise _err("receipt_cumulative_seconds_out_of_range")
        if cumulative > MAXIMUM_TOTAL_SECONDS:
            raise _err("receipt_cumulative_exceeds_total")
        return cumulative
    this_stage = receipt.get("scientific_seconds_this_stage")
    cumulative = receipt.get("scientific_seconds_cumulative_bound")
    for value, code in (
        (this_stage, "receipt_stage_seconds_malformed"),
        (cumulative, "receipt_cumulative_seconds_malformed"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _err(code)
    this_stage, cumulative = float(this_stage), float(cumulative)
    if not math.isfinite(this_stage) or this_stage < 0.0:
        raise _err("receipt_stage_seconds_out_of_range")
    if not math.isfinite(cumulative) or cumulative <= 0.0:
        raise _err("receipt_cumulative_seconds_out_of_range")
    if cumulative < PRELAUNCH_AUDIT_RESERVE_SECONDS:
        raise _err("receipt_cumulative_below_reserve")
    if cumulative != this_stage + PRELAUNCH_AUDIT_RESERVE_SECONDS:
        raise _err("receipt_cumulative_inconsistent")
    if cumulative > MAXIMUM_TOTAL_SECONDS:
        raise _err("receipt_cumulative_exceeds_total")
    if int(receipt.get("maximum_total_seconds", -1)) != int(MAXIMUM_TOTAL_SECONDS):
        raise _err("receipt_maximum_total_mismatch")
    return cumulative


def _check_develop_summary(summary: Mapping[str, Any], expected_units: int) -> None:
    if source.is_recovered(summary):
        try:
            recovery_receipt.validate_summary(summary, expected_units)
        except Exception as error:
            raise _err("develop_summary_recovery_invalid") from error
        return
    if str(summary.get("status")) != "complete":
        raise _err("develop_not_complete")
    if str(summary.get("command")) != "run_development":
        raise _err("develop_command_mismatch")
    _check_identity(summary, "develop_identity_mismatch")
    for key, expected in (
        ("started", MAXIMUM_NEW_FITS),
        ("completed", MAXIMUM_NEW_FITS),
        ("new_completions", MAXIMUM_NEW_FITS),
        ("selector_records", INNER_SLOT_COUNT),
        ("units_completed", expected_units),
        ("failed", 0),
    ):
        if int(summary.get(key, -1)) != expected:
            raise _err("develop_count_mismatch")
    if str(summary.get("claim")) != development.COMPREHENSIVE_CLAIM:
        raise _err("develop_claim_mismatch")
    if summary.get("outer_evaluation_authorized") is not False:
        raise _err("develop_authorization_invalid")


def _verify_develop_manifest(receipt: Mapping[str, Any], stage: Path) -> None:
    manifest_path = stage / "manifest.json"
    core._reject_symlink_chain(manifest_path)
    if not manifest_path.is_file():
        raise _err("develop_manifest_missing")
    if _canon().sha256_file(manifest_path) != str(receipt.get("stage_manifest_sha256")):
        raise _err("develop_manifest_digest_mismatch")
    pilot._verify_manifest(stage)


def _verify_ledger(bundle: Mapping[str, Any], stage: Path) -> None:
    path = stage / "ledger.json"
    core._reject_symlink_chain(path)
    if not path.is_file():
        raise _err("develop_ledger_missing")
    if core._read_bytes(path, "develop_ledger") != _canon().canonical_json_bytes(bundle["ledger"]):
        raise _err("develop_ledger_mismatch")


def _verify_pilot_manifest(bundle: Mapping[str, Any]) -> None:
    pinned = str(bundle["permit"].get("pilot_manifest_sha256"))
    if pinned != inputs.PILOT_MANIFEST_SHA256:
        raise _err("pilot_manifest_pin_mismatch")
    run_dir = inputs._pilot_run_dir(bundle["artifact_root"])
    manifest_path = run_dir / "manifest.json"
    core._reject_symlink_chain(manifest_path)
    if not manifest_path.is_file():
        raise _err("pilot_manifest_missing")
    if _canon().sha256_file(manifest_path) != pinned:
        raise _err("pilot_manifest_digest_mismatch")
    pilot._verify_manifest(run_dir)


def _expected_source_ledger(bundle: Mapping[str, Any]) -> dict[str, Any]:
    ledger = bundle["ledger"]
    pilot_unit_ids = {str(unit["unit_id"]) for unit in bundle["pilot_bundle"]["units"]}
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for slot in ledger["slots"]:
        by_unit.setdefault(str(slot["unit_id"]), []).append(dict(slot))
    units: list[dict[str, Any]] = []
    for unit in ledger["units"]:
        unit_id = str(unit["unit_id"])
        if unit_id in pilot_unit_ids:
            continue
        group = sorted(
            by_unit.get(unit_id, []),
            key=lambda item: (str(item["recipe_id"]), int(item["seed"])),
        )
        units.append(
            {
                "unit_id": unit_id,
                "station": str(unit["station"]),
                "fitting_uid_set_sha256": str(unit["fitting_uid_set_sha256"]),
                "validation_uid_set_sha256": str(unit["validation_uid_set_sha256"]),
                "slot_ids": [str(item["slot_id"]) for item in group],
            }
        )
    return {
        **_identity_fields(bundle),
        "schema_version": development.SCHEMA_VERSION,
        "units": units,
    }


def _verify_source_ledger(bundle: Mapping[str, Any], stage: Path) -> None:
    path = stage / "source_ledger.json"
    core._reject_symlink_chain(path)
    if not path.is_file():
        raise _err("source_ledger_missing")
    expected = _canon().canonical_json_bytes(_expected_source_ledger(bundle))
    if core._read_bytes(path, "source_ledger") != expected:
        raise _err("source_ledger_mismatch")


def _authenticate_selector(
    bundle: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    stage: Path,
    deadline: float,
) -> list[dict[str, Any]]:
    if len(records) != INNER_SLOT_COUNT:
        raise _err("selector_count_mismatch")
    unit_by_id = {str(unit["unit_id"]): unit for unit in bundle["ledger"]["units"]}
    slot_by_id = {str(slot["slot_id"]): dict(slot) for slot in bundle["ledger"]["slots"]}
    if len(slot_by_id) != INNER_SLOT_COUNT:
        raise _err("ledger_slot_identity_mismatch")
    pilot_slot_ids = inputs.pilot_slot_ids(bundle)
    pilot_run_dir = inputs._pilot_run_dir(bundle["artifact_root"])
    seen: set[str] = set()
    for record in records:
        _check_deadline(deadline)
        slot_id = str(record.get("slot_id"))
        if not slot_id or slot_id in seen:
            raise _err("selector_slot_duplicate")
        seen.add(slot_id)
        slot = slot_by_id.get(slot_id)
        if slot is None:
            raise _err("selector_slot_unknown")
        unit = unit_by_id.get(str(slot["unit_id"]))
        if unit is None:
            raise _err("selector_unit_unknown")
        run_dir = (
            pilot_run_dir if slot_id in pilot_slot_ids else stage / "units" / str(unit["unit_id"])
        )
        summary = _read_mapping(
            run_dir / "executions" / pilot.execution_id(unit, slot) / "summary.json",
            "execution_summary_missing",
        )
        if inputs.selector_record(unit, slot, summary) != record:
            raise _err("selector_reconstruction_mismatch")
    if seen != set(slot_by_id):
        raise _err("selector_coverage_mismatch")
    return [dict(record) for record in records]


def _support_view(support: Any) -> tuple[Any, Any]:
    contexts = getattr(support, "contexts", None)
    roles = getattr(support, "roles", None)
    if contexts is None or roles is None:
        raise _err("support_view_unavailable")
    return contexts, roles


def _build_plan(bundle: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    contexts, roles = _support_view(bundle["support"])
    try:
        plan = build_refit_plan(
            ledger=bundle["ledger"],
            contexts=contexts,
            roles=roles,
            results=records,
            permit_sha256=str(bundle["permit_sha256"]),
        )
    except RefitPlanError as error:
        raise _err("refit_plan_failed") from error
    decisions = plan.get("decisions")
    counts = plan.get("counts")
    unique_refits = plan.get("unique_refits")
    if not isinstance(decisions, Sequence) or isinstance(decisions, (str, bytes)):
        raise _err("refit_decisions_malformed")
    if len(decisions) != CONTEXT_COUNT:
        raise _err("refit_decision_count_mismatch")
    if not isinstance(counts, Mapping) or not isinstance(unique_refits, Mapping):
        raise _err("refit_counts_malformed")
    if int(counts.get("context_count", -1)) != CONTEXT_COUNT:
        raise _err("refit_context_count_mismatch")
    aliases = int(counts.get("strategy_alias_count", -1))
    expected = int(counts.get("expected_strategy_alias_count", -2))
    if aliases != STRATEGY_ALIAS_COUNT or expected != STRATEGY_ALIAS_COUNT:
        raise _err("refit_alias_count_mismatch")
    unique = int(counts.get("unique_refit_count", -1))
    if unique != len(unique_refits) or not 1 <= unique <= STRATEGY_ALIAS_COUNT:
        raise _err("refit_unique_count_mismatch")
    if not core._is_hex64(str(plan.get("plan_id"))):
        raise _err("refit_plan_id_malformed")
    return dict(plan)


def _base_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        **_identity_fields(bundle),
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "pilot_permit_sha256": inputs.PILOT_PERMIT_SHA256,
        "pilot_plan_id": inputs.PILOT_PLAN_ID,
        "pilot_manifest_sha256": inputs.PILOT_MANIFEST_SHA256,
        "context_count": CONTEXT_COUNT,
        "unit_count": UNIT_COUNT,
        "reused_pilot_slots": REUSED_PILOT_SLOTS,
        "source_new_fits": MAXIMUM_NEW_FITS,
        "fits_started": 0,
        "selector_records": INNER_SLOT_COUNT,
        "claim": SELECTION_CLAIM,
        **SELECTION_ONLY_FLAGS,
    }
    if SOURCE_ACCOUNTING_KEY in bundle:
        accounting = bundle[SOURCE_ACCOUNTING_KEY]
        if not isinstance(accounting, Mapping):
            raise _err("source_accounting_malformed")
        try:
            payload[SOURCE_ACCOUNTING_KEY] = source.validate_accounting(
                accounting,
                source_optimizer_steps=accounting.get("source_optimizer_steps_successful_exact"),
            )
        except Exception as error:
            raise _err("source_accounting_invalid") from error
    return payload


def _source_bindings(
    bundle: Mapping[str, Any],
    develop_manifest_sha256: str,
    source_ledger_sha256: str,
    selector_sha256: str,
) -> dict[str, Any]:
    return {
        **_base_payload(bundle),
        "develop_manifest_sha256": str(develop_manifest_sha256),
        "source_ledger_sha256": str(source_ledger_sha256),
        "selector_sha256": str(selector_sha256),
    }


def _selection_summary(
    bundle: Mapping[str, Any], plan: Mapping[str, Any], prior: float, seconds: float
) -> dict[str, Any]:
    counts = plan["counts"]
    return {
        **_base_payload(bundle),
        "status": "complete",
        "command": "freeze_selection",
        "refit_decision_count": len(plan["decisions"]),
        "strategy_alias_count": int(counts["strategy_alias_count"]),
        "unique_refit_count": int(counts["unique_refit_count"]),
        "plan_id": str(plan["plan_id"]),
        "prior_scientific_seconds_cumulative_bound": float(prior),
        "scientific_seconds_this_stage": float(seconds),
        "scientific_seconds_cumulative_bound": float(prior + seconds),
        "prelaunch_audit_reserve_seconds": PRELAUNCH_AUDIT_RESERVE_SECONDS,
        "maximum_total_seconds": MAXIMUM_TOTAL_SECONDS,
    }


def _selection_receipt(
    bundle: Mapping[str, Any],
    plan: Mapping[str, Any],
    manifest_sha256: str,
    prior: float,
    seconds: float,
) -> dict[str, Any]:
    return {
        **_base_payload(bundle),
        "stage": SELECTION_STAGE_NAME,
        "selection_plan_id": str(plan["plan_id"]),
        "selection_manifest_sha256": str(manifest_sha256),
        "prior_scientific_seconds_cumulative_bound": float(prior),
        "scientific_seconds_this_stage": float(seconds),
        "scientific_seconds_cumulative_bound": float(prior + seconds),
        "prelaunch_audit_reserve_seconds": PRELAUNCH_AUDIT_RESERVE_SECONDS,
        "maximum_total_seconds": MAXIMUM_TOTAL_SECONDS,
    }


def _write_failure_summary(
    stage: Path,
    bundle: Mapping[str, Any],
    error: BaseException,
    prior: float | None,
    wall_seconds: float,
) -> None:
    cumulative = float(prior + wall_seconds) if prior is not None else None
    core._atomic_write(
        stage / "summary.json",
        _canon().canonical_json_bytes(
            {
                **_base_payload(bundle),
                "status": "fail",
                "command": "freeze_selection",
                "reason_code": getattr(error, "reason_code", type(error).__name__),
                "prior_scientific_seconds_cumulative_bound": (
                    float(prior) if prior is not None else None
                ),
                "scientific_seconds_this_stage": float(wall_seconds),
                "scientific_seconds_cumulative_bound": cumulative,
                "wall_seconds": float(wall_seconds),
            }
        ),
    )


def freeze_selection(
    *,
    project_root: Path | str,
    artifact_root: Path | str,
    contract_path: Path | str,
    permit_path: Path | str,
) -> dict[str, Any]:
    """Freeze the read-only P05 selection plan once, or fail closed."""

    core._configure_environment()
    wall_start = time.perf_counter()
    bundle = inputs.prepare(
        project_root, artifact_root, contract_path, permit_path, require_unstarted=False
    )
    paths = _paths(bundle)
    expected_units = _expected_new_units(bundle)
    develop_receipt = _read_mapping(paths["receipt"], "development_receipt_missing")
    _check_receipt(develop_receipt, expected_units)
    develop_summary = _read_mapping(paths["develop"] / "summary.json", "develop_summary_missing")
    _check_develop_summary(develop_summary, expected_units)
    prior = _check_prior_bound(develop_receipt)
    deadline = wall_start + (MAXIMUM_TOTAL_SECONDS - prior)
    _check_deadline(deadline)
    recovered_receipt = source.is_recovered(develop_receipt)
    recovered_summary = source.is_recovered(develop_summary)
    if recovered_receipt != recovered_summary:
        raise _err("source_flavor_mismatch")
    if recovered_receipt:
        try:
            normalized = acceptance.authenticate_completed_source(
                bundle,
                paths=paths,
                summary=develop_summary,
                receipt_record=develop_receipt,
                deadline=deadline,
            )
        except Exception as error:
            raise _err("source_acceptance_failed") from error
        if not isinstance(normalized, Mapping):
            raise _err("source_accounting_malformed")
        bundle = {**bundle, SOURCE_ACCOUNTING_KEY: dict(normalized)}
    stage = paths["selection"]
    _reject_preexisting(paths["selection_receipt"], "selection_receipt_exists")
    core._mkdir_exclusive(stage, "selection_stage_exists")
    budget: StorageBudget | None = None
    try:
        budget = StorageBudget(
            bundle["artifact_root"], paths["run_root"], ceiling=STORAGE_CEILING_BYTES
        )
        provenance_before = core._capture_provenance(
            bundle["repository_root"], bundle["project_root"], bundle["artifact_root"]
        )
        _budgeted_write(stage / "provenance_before.json", provenance_before, budget)
        _verify_develop_manifest(develop_receipt, paths["develop"])
        _verify_ledger(bundle, paths["develop"])
        _verify_source_ledger(bundle, paths["develop"])
        selector_path = paths["develop"] / "selector.jsonl"
        records = _authenticate_selector(
            bundle, _read_jsonl(selector_path, "selector_missing"), paths["develop"], deadline
        )
        _verify_pilot_manifest(bundle)
        _check_deadline(deadline)
        plan = _build_plan(bundle, records)
        _check_deadline(deadline)
        _budgeted_write(stage / "plan.json", plan, budget)
        _budgeted_write(
            stage / "source_bindings.json",
            _source_bindings(
                bundle,
                _canon().sha256_file(paths["develop"] / "manifest.json"),
                _canon().sha256_file(paths["develop"] / "source_ledger.json"),
                _canon().sha256_file(selector_path),
            ),
            budget,
        )
        provenance_after = pilot._post_run_reauth(
            bundle["artifact_root"],
            bundle["contract"],
            bundle["support"],
            provenance_before,
            bundle["repository_root"],
            bundle["project_root"],
        )
        _budgeted_write(stage / "provenance_after.json", provenance_after, budget)
        freeze_seconds = time.perf_counter() - wall_start
        if prior + freeze_seconds > MAXIMUM_TOTAL_SECONDS:
            raise _err("total_wall_exceeded")
        summary = _selection_summary(bundle, plan, prior, freeze_seconds)
        _budgeted_write(stage / "summary.json", summary, budget)
        budget.check(headroom_bytes=HEADROOM_BYTES)
        core._write_manifest(stage)
        budget.account_new_file(stage / "manifest.json")
        budget.check(headroom_bytes=HEADROOM_BYTES)
        manifest_sha256 = _canon().sha256_file(stage / "manifest.json")
        pilot._verify_manifest(stage)
        budget.reconcile()
        _check_deadline(deadline)
        receipt_seconds = time.perf_counter() - wall_start
        if prior + receipt_seconds > MAXIMUM_TOTAL_SECONDS:
            raise _err("total_wall_exceeded")
        _budgeted_write(
            paths["selection_receipt"],
            _selection_receipt(bundle, plan, manifest_sha256, prior, receipt_seconds),
            budget,
        )
        budget.check()
        _check_deadline(deadline)
        summary["scientific_seconds_this_stage"] = receipt_seconds
        summary["scientific_seconds_cumulative_bound"] = prior + receipt_seconds
        return summary
    except BaseException as error:
        wall_seconds = time.perf_counter() - wall_start
        try:
            _write_failure_summary(stage, bundle, error, prior, wall_seconds)
            if budget is not None:
                budget.account_new_file(stage / "summary.json")
        except Exception:
            pass
        try:
            core._write_manifest(stage)
        except Exception:
            pass
        if isinstance(error, (KeyboardInterrupt, SystemExit, P05CoreError, P05StorageError)):
            raise
        raise FreezeSelectionError("freeze_execution_failed") from error
