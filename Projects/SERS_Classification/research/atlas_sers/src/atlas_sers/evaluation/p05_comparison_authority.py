"""P05 comparison authority: read-only authentication of the comparison stage.

Authenticates the completed comprehensive-comparison stage after the frozen
predictions and the comprehensive aggregation have themselves been
authenticated.  The four persisted comparison tables are re-derived from the
authenticated aggregation ensemble, the two pinned legacy reference shards and
the registered context rows; the receipt, summary, manifest and reference
bindings are checked for mutual consistency.  This module writes nothing, loads
no feature array, builds no model, runs no optimizer, fits no temperature and
never predicts.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_sers.evaluation import p05_aggregation_authority as aggregation_authority
from atlas_sers.evaluation import p05_comparison as comparison
from atlas_sers.evaluation import p05_comprehensive_comparison as producer
from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_evaluation as evaluation
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_legacy_references as legacy_references
from atlas_sers.evaluation import p05_outer_inputs as outer_inputs
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_recovery_source as recovery_source
from atlas_sers.governance.canonical import sha256_value

__all__ = ["P05ComparisonAuthorityError", "authenticate_comparison"]

SCHEMA = producer.SCHEMA
PROTOCOL = producer.PROTOCOL
COMMAND = producer.COMMAND
STAGE_NAME = producer.STAGE_NAME
RECEIPT_NAME = producer.RECEIPT_NAME
MANIFEST_NAME = producer.MANIFEST_NAME
SUMMARY_NAME = producer.SUMMARY_NAME
PROVENANCE_BEFORE_NAME = producer.PROVENANCE_BEFORE_NAME
PROVENANCE_AFTER_NAME = producer.PROVENANCE_AFTER_NAME
BINDINGS_NAME = producer.BINDINGS_NAME
COMPREHENSIVE_DIR = producer.COMPREHENSIVE_DIR
RUNS_DIR = producer.RUNS_DIR
TABLE_NAMES = producer.TABLE_NAMES
COUNTER_KEYS = producer.COUNTER_KEYS
ZERO_COUNTERS = producer.ZERO_COUNTERS
AGGREGATION_RECEIPT_NAME = producer.AGGREGATION_RECEIPT_NAME
AGGREGATION_MANIFEST_NAME = producer.AGGREGATION_MANIFEST_NAME
AGGREGATION_STAGE_NAME = producer.AGGREGATION_STAGE_NAME
EVALUATION_RECEIPT_NAME = evaluation.RECEIPT_NAME
EVALUATION_MANIFEST_NAME = evaluation.MANIFEST_NAME
EVALUATION_STAGE_NAME = evaluation.STAGE_NAME
ELAPSED_KEY = "elapsed_seconds"
COUNTER_FIELDS = frozenset(COUNTER_KEYS) | {ELAPSED_KEY}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class P05ComparisonAuthorityError(core.P05CoreError):
    """Stable, path-free comparison-authority failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05ComparisonAuthorityError(code)


def _finite_seconds(value: Any, code: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), code)
    number = float(value)
    _require(math.isfinite(number), code)
    return number


def _integer(value: Any, code: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), code)
    return int(value)


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _mapping(path: Path, name: str) -> Mapping[str, Any]:
    core._reject_symlink_chain(path)
    _require(path.is_file() and not path.is_symlink(), f"{name}_missing")
    value = core._read_json(path, name)
    _require(isinstance(value, Mapping), f"{name}_malformed")
    return value


def _frames_equal(saved: Any, recomputed: Any) -> bool:
    if not isinstance(saved, pd.DataFrame) or not isinstance(recomputed, pd.DataFrame):
        return False
    try:
        pd.testing.assert_frame_equal(
            saved.reset_index(drop=True),
            recomputed.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError:
        return False
    return True


def _source_accounting(auth: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        return recovery_source.from_authenticated(auth)
    except recovery_source.RecoverySourceError as error:
        raise P05ComparisonAuthorityError("source_accounting_invalid") from error


def _require_accounting(
    payload: Mapping[str, Any],
    *,
    label: str,
    source_steps: int,
    expected: Mapping[str, Any] | None,
) -> None:
    present = "source_execution_accounting" in payload
    if expected is None:
        _require(not present, f"{label}_source_accounting_unexpected")
        return
    _require(present, f"{label}_source_accounting_missing")
    try:
        validated = recovery_source.validate_accounting(
            payload["source_execution_accounting"], source_optimizer_steps=source_steps
        )
    except recovery_source.RecoverySourceError as error:
        raise P05ComparisonAuthorityError(f"{label}_source_accounting_invalid") from error
    _require(
        core._canon().canonical_json_bytes(validated)
        == core._canon().canonical_json_bytes(dict(expected)),
        f"{label}_source_accounting_mismatch",
    )


def authenticate_comparison(bundle: Any, *, deadline: Any) -> dict[str, Any]:
    """Authenticate the completed comprehensive-comparison stage read-only."""

    _require(isinstance(bundle, Mapping), "bundle_malformed")
    deadline = _finite_seconds(deadline, "deadline_malformed")
    freeze._check_deadline(deadline)

    auth = aggregation_authority.authenticate_aggregation(bundle, deadline=deadline)
    _require(isinstance(auth, Mapping), "authority_malformed")
    plan = auth.get("plan")
    _require(isinstance(plan, Mapping), "authority_plan_malformed")
    plan_id = plan.get("plan_id")
    _require(plan_id is not None and str(plan_id) != "", "authority_plan_id_malformed")
    aggregation_tables = auth.get("aggregation_tables")
    _require(isinstance(aggregation_tables, Mapping), "authority_tables_malformed")
    _require("ensemble_predictions" in aggregation_tables, "authority_ensemble_missing")
    aggregation_receipt = auth.get("aggregation_receipt")
    _require(isinstance(aggregation_receipt, Mapping), "authority_aggregation_receipt_malformed")
    evaluation_receipt = auth.get("evaluation_receipt")
    _require(isinstance(evaluation_receipt, Mapping), "authority_evaluation_receipt_malformed")
    source_steps = _integer(auth.get("source_optimizer_steps"), "authority_source_steps_malformed")
    refit_steps = _integer(auth.get("refit_optimizer_steps"), "authority_refit_steps_malformed")
    source_accounting = _source_accounting(auth)
    recovered = source_accounting["mode"] == recovery_source.RECOVERY_ACCOUNTING_MODE
    expected_accounting = dict(source_accounting) if recovered else None
    aggregation_prior_seconds = _finite_seconds(
        auth.get("prior_seconds"), "authority_prior_malformed"
    )
    _require(
        development.PRELAUNCH_AUDIT_RESERVE_SECONDS
        <= aggregation_prior_seconds
        <= development.MAXIMUM_TOTAL_SECONDS,
        "authority_prior_out_of_range",
    )

    permit_sha256 = bundle.get("permit_sha256")
    _require(isinstance(permit_sha256, str) and bool(permit_sha256), "permit_sha256_malformed")
    ledger = bundle.get("ledger")
    _require(isinstance(ledger, Mapping), "ledger_malformed")
    ledger_id = ledger.get("ledger_id")
    _require(isinstance(ledger_id, str) and bool(ledger_id), "ledger_id_malformed")
    contract_sha256 = bundle.get("contract_sha256")
    _require(contract_sha256 is not None, "contract_sha256_missing")
    core_plan_id = bundle.get("core_plan_id")
    _require(core_plan_id is not None, "core_plan_id_missing")
    _require(bundle.get("artifact_root") is not None, "artifact_root_missing")

    run_root = Path(bundle["artifact_root"]) / COMPREHENSIVE_DIR / RUNS_DIR / permit_sha256
    stage = run_root / STAGE_NAME
    aggregation_stage = run_root / AGGREGATION_STAGE_NAME
    aggregation_receipt_path = run_root / AGGREGATION_RECEIPT_NAME
    aggregation_manifest_path = aggregation_stage / AGGREGATION_MANIFEST_NAME
    evaluation_stage = run_root / EVALUATION_STAGE_NAME
    evaluation_receipt_path = run_root / EVALUATION_RECEIPT_NAME
    evaluation_manifest_path = evaluation_stage / EVALUATION_MANIFEST_NAME

    core._reject_symlink_chain(aggregation_receipt_path)
    _require(
        aggregation_receipt_path.is_file() and not aggregation_receipt_path.is_symlink(),
        "aggregation_receipt_missing",
    )
    aggregation_receipt_sha256 = core._canon().sha256_file(aggregation_receipt_path)
    _require(
        dict(_mapping(aggregation_receipt_path, "aggregation_receipt"))
        == dict(aggregation_receipt),
        "aggregation_receipt_changed",
    )
    core._reject_symlink_chain(aggregation_manifest_path)
    _require(
        aggregation_manifest_path.is_file() and not aggregation_manifest_path.is_symlink(),
        "aggregation_manifest_missing",
    )
    aggregation_manifest_sha256 = core._canon().sha256_file(aggregation_manifest_path)
    _require(
        aggregation_receipt.get("stage_manifest_sha256") == aggregation_manifest_sha256,
        "aggregation_manifest_mismatch",
    )
    core._reject_symlink_chain(evaluation_receipt_path)
    _require(
        evaluation_receipt_path.is_file() and not evaluation_receipt_path.is_symlink(),
        "evaluation_receipt_missing",
    )
    evaluation_receipt_sha256 = core._canon().sha256_file(evaluation_receipt_path)
    _require(
        dict(_mapping(evaluation_receipt_path, "evaluation_receipt")) == dict(evaluation_receipt),
        "evaluation_receipt_changed",
    )
    core._reject_symlink_chain(evaluation_manifest_path)
    _require(
        evaluation_manifest_path.is_file() and not evaluation_manifest_path.is_symlink(),
        "evaluation_manifest_missing",
    )
    evaluation_manifest_sha256 = core._canon().sha256_file(evaluation_manifest_path)
    _require(
        evaluation_receipt.get("stage_manifest_sha256") == evaluation_manifest_sha256,
        "evaluation_manifest_mismatch",
    )

    core._reject_symlink_chain(stage)
    _require(stage.is_dir() and not stage.is_symlink(), "comparison_stage_missing")
    receipt = _mapping(run_root / RECEIPT_NAME, "comparison_receipt")
    receipt_sha256 = core._canon().sha256_file(run_root / RECEIPT_NAME)
    summary = _mapping(stage / SUMMARY_NAME, "comparison_summary")
    manifest_path = stage / MANIFEST_NAME
    core._reject_symlink_chain(manifest_path)
    _require(
        manifest_path.is_file() and not manifest_path.is_symlink(), "comparison_manifest_missing"
    )
    stage_manifest_sha256 = core._canon().sha256_file(manifest_path)
    _require(
        receipt.get("stage_manifest_sha256") == stage_manifest_sha256,
        "comparison_manifest_sha256_mismatch",
    )
    pilot._verify_manifest(stage)

    bindings = _mapping(stage / BINDINGS_NAME, "comparison_bindings")
    reference_binding_digest = sha256_value(dict(bindings))
    _require(bindings.get("p03_run_id") == legacy_references.P03_RUN_ID, "binding_p03_run_mismatch")
    _require(bindings.get("p04_run_id") == legacy_references.P04_RUN_ID, "binding_p04_run_mismatch")
    p03_protected = bindings.get("p03_protected_state_sha256")
    _require(
        _is_hex64(p03_protected) and p03_protected.startswith(legacy_references.P03_EXECUTION_ID),
        "binding_p03_protected_mismatch",
    )
    _require(
        bindings.get("p04_shard_protected_state_sha256")
        == legacy_references.P04_AGGREGATION_STATE_SHA256,
        "binding_p04_shard_protected_mismatch",
    )
    _require(
        bindings.get("p04_execution_protected_state_sha256")
        == legacy_references.P04_PROTECTED_STATE_SHA256,
        "binding_p04_execution_protected_mismatch",
    )
    for name in (
        "p03_state_sha256",
        "p04_state_sha256",
        "p03_predictions_sha256",
        "p04_predictions_sha256",
        "evaluation_receipt_sha256",
    ):
        _require(_is_hex64(bindings.get(name)), f"binding_{name}_malformed")
    _require(
        bindings.get("evaluation_receipt_sha256") == evaluation_receipt_sha256,
        "binding_evaluation_receipt_mismatch",
    )

    expected_identity = {
        "schema_version": SCHEMA,
        "protocol_version": PROTOCOL,
        "command": COMMAND,
        "stage": STAGE_NAME,
        "permit_sha256": permit_sha256,
        "core_contract_sha256": contract_sha256,
        "core_plan_id": core_plan_id,
        "ledger_id": ledger_id,
        "selection_plan_id": str(plan_id),
        "aggregation_receipt_sha256": aggregation_receipt_sha256,
        "source_optimizer_steps": source_steps,
        "refit_optimizer_steps": refit_steps,
        "reference_binding_digest": reference_binding_digest,
    }
    if recovered:
        expected_identity["source_execution_accounting"] = dict(source_accounting)
    for label, payload in (("receipt", receipt), ("summary", summary)):
        for name, value in expected_identity.items():
            if name == "source_execution_accounting":
                continue
            _require(payload.get(name) == value, f"{label}_{name}_mismatch")
        _require_accounting(
            payload, label=label, source_steps=source_steps, expected=expected_accounting
        )
        for name in ("source_optimizer_steps", "refit_optimizer_steps"):
            _integer(payload.get(name), f"{label}_{name}_invalid")
        _require(payload.get("status") == "complete", f"{label}_status_incomplete")
        _require(payload.get("comparison_complete") is True, f"{label}_incomplete")

    receipt_counters = receipt.get("counters")
    summary_counters = summary.get("counters")
    _require(isinstance(receipt_counters, Mapping), "receipt_counters_malformed")
    _require(isinstance(summary_counters, Mapping), "summary_counters_malformed")
    _require(set(receipt_counters) == COUNTER_FIELDS, "receipt_counter_fields_mismatch")
    _require(set(summary_counters) == COUNTER_FIELDS, "summary_counter_fields_mismatch")
    for name in ZERO_COUNTERS:
        _require(
            _integer(receipt_counters.get(name), f"receipt_{name}_invalid") == 0,
            f"receipt_{name}_nonzero",
        )
        _require(
            _integer(summary_counters.get(name), f"summary_{name}_invalid") == 0,
            f"summary_{name}_nonzero",
        )

    receipt_elapsed = _finite_seconds(receipt_counters[ELAPSED_KEY], "receipt_elapsed_malformed")
    summary_elapsed = _finite_seconds(summary_counters[ELAPSED_KEY], "summary_elapsed_malformed")
    _require(0.0 <= summary_elapsed <= receipt_elapsed, "summary_elapsed_exceeds_receipt")
    for label, payload in (("receipt", receipt), ("summary", summary)):
        prior_bound = _finite_seconds(
            payload.get("prior_scientific_seconds_cumulative_bound"), f"{label}_prior_malformed"
        )
        _require(prior_bound == aggregation_prior_seconds, f"{label}_prior_mismatch")
        stage_seconds = _finite_seconds(
            payload.get("scientific_seconds_this_stage"), f"{label}_stage_malformed"
        )
        counter_elapsed = _finite_seconds(
            payload["counters"][ELAPSED_KEY], f"{label}_counter_elapsed_malformed"
        )
        _require(stage_seconds == counter_elapsed, f"{label}_stage_seconds_inconsistent")
        cumulative = _finite_seconds(
            payload.get("scientific_seconds_cumulative_bound"), f"{label}_cumulative_malformed"
        )
        _require(
            cumulative == aggregation_prior_seconds + stage_seconds,
            f"{label}_cumulative_mismatch",
        )
        _require(
            development.PRELAUNCH_AUDIT_RESERVE_SECONDS
            <= cumulative
            <= development.MAXIMUM_TOTAL_SECONDS,
            f"{label}_cumulative_out_of_range",
        )
        _require(
            _finite_seconds(
                payload.get("prelaunch_audit_reserve_seconds"), f"{label}_reserve_malformed"
            )
            == development.PRELAUNCH_AUDIT_RESERVE_SECONDS,
            f"{label}_reserve_mismatch",
        )
        _require(
            _finite_seconds(payload.get("maximum_total_seconds"), f"{label}_maximum_malformed")
            == development.MAXIMUM_TOTAL_SECONDS,
            f"{label}_maximum_mismatch",
        )
    comparison_cumulative = _finite_seconds(
        receipt.get("scientific_seconds_cumulative_bound"), "comparison_cumulative_malformed"
    )

    expected_files = {f"{name}.parquet" for name in TABLE_NAMES} | {
        PROVENANCE_BEFORE_NAME,
        PROVENANCE_AFTER_NAME,
        SUMMARY_NAME,
        MANIFEST_NAME,
        BINDINGS_NAME,
    }
    actual_files: set[str] = set()
    for entry in stage.iterdir():
        _require(not entry.is_symlink(), "comparison_symlink_rejected")
        _require(entry.is_file(), "comparison_inventory_not_file")
        actual_files.add(entry.name)
    _require(actual_files == expected_files, "comparison_inventory_mismatch")

    tables: dict[str, pd.DataFrame] = {}
    for name in TABLE_NAMES:
        freeze._check_deadline(deadline)
        path = stage / f"{name}.parquet"
        core._reject_symlink_chain(path)
        _require(path.is_file() and not path.is_symlink(), "comparison_table_missing")
        try:
            frame = pd.read_parquet(path)
        except Exception as error:  # noqa: BLE001
            raise P05ComparisonAuthorityError("comparison_table_read_failed") from error
        _require(isinstance(frame, pd.DataFrame), "comparison_table_malformed")
        tables[name] = frame

    freeze._check_deadline(deadline)
    legacy = legacy_references.load_references(bundle, authenticated=auth, deadline=deadline)
    _require(isinstance(legacy, Mapping), "legacy_malformed")
    loaded_bindings = legacy.get("bindings")
    _require(isinstance(loaded_bindings, Mapping), "legacy_bindings_malformed")
    _require(dict(loaded_bindings) == dict(bindings), "reference_bindings_changed")
    for name in ("p04_ensemble", "p03_predictions"):
        _require(isinstance(legacy.get(name), pd.DataFrame), "legacy_frame_malformed")

    freeze._check_deadline(deadline)
    full_contexts = outer_inputs.load_context_rows(bundle)
    contexts = pd.DataFrame(full_contexts)
    _require(
        isinstance(contexts, pd.DataFrame) and len(contexts) > 0, "context_registry_empty"
    )
    try:
        held = comparison._held_contexts(contexts)
    except comparison.P05ComparisonError as error:
        raise P05ComparisonAuthorityError("held_contexts_invalid") from error
    held_ids = set(held.context_id.astype(str))

    freeze._check_deadline(deadline)
    try:
        recomputed = comparison.compare_predictions(
            p05_ensemble=aggregation_tables["ensemble_predictions"],
            p04_ensemble=legacy["p04_ensemble"],
            p03_predictions=legacy["p03_predictions"],
            contexts=contexts,
        )
    except comparison.P05ComparisonError as error:
        raise P05ComparisonAuthorityError("recomputed_comparison_failed") from error
    _require(isinstance(recomputed, Mapping), "recomputed_malformed")
    _require(set(recomputed) == set(TABLE_NAMES), "recomputed_table_set_mismatch")
    for name in TABLE_NAMES:
        freeze._check_deadline(deadline)
        _require(_frames_equal(tables[name], recomputed[name]), f"table_{name}_mismatch")

    freeze._check_deadline(deadline)
    try:
        recomputed_counters = producer._comparison_counters(recomputed, held_context_ids=held_ids)
    except producer.P05ComprehensiveComparisonError as error:
        raise P05ComparisonAuthorityError("recomputed_counters_invalid") from error
    _require(set(recomputed_counters) == set(COUNTER_KEYS), "recomputed_counter_fields_mismatch")
    for name in COUNTER_KEYS:
        expected_count = _integer(recomputed_counters[name], f"recomputed_{name}_invalid")
        _require(
            _integer(receipt_counters.get(name), f"receipt_{name}_invalid") == expected_count,
            f"receipt_{name}_mismatch",
        )
        _require(
            _integer(summary_counters.get(name), f"summary_{name}_invalid") == expected_count,
            f"summary_{name}_mismatch",
        )

    freeze._check_deadline(deadline)
    _require(
        core._canon().sha256_file(run_root / RECEIPT_NAME) == receipt_sha256,
        "comparison_receipt_changed",
    )
    _require(
        core._canon().sha256_file(manifest_path) == stage_manifest_sha256,
        "comparison_manifest_changed",
    )
    _require(
        core._canon().sha256_file(aggregation_receipt_path) == aggregation_receipt_sha256,
        "aggregation_receipt_changed",
    )
    _require(
        core._canon().sha256_file(aggregation_manifest_path) == aggregation_manifest_sha256,
        "aggregation_manifest_changed",
    )
    _require(
        core._canon().sha256_file(evaluation_receipt_path) == evaluation_receipt_sha256,
        "evaluation_receipt_changed",
    )
    _require(
        core._canon().sha256_file(evaluation_manifest_path) == evaluation_manifest_sha256,
        "evaluation_manifest_changed",
    )
    _require(
        dict(_mapping(stage / BINDINGS_NAME, "comparison_bindings")) == dict(bindings),
        "comparison_bindings_changed",
    )
    pilot._verify_manifest(stage)
    pilot._verify_manifest(aggregation_stage)
    pilot._verify_manifest(evaluation_stage)
    freeze._check_deadline(deadline)

    return {
        **auth,
        "comparison_receipt": receipt,
        "comparison_tables": tables,
        "reference_bindings": dict(bindings),
        "aggregation_prior_seconds": aggregation_prior_seconds,
        "prior_seconds": comparison_cumulative,
    }
