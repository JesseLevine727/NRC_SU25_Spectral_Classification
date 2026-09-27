"""Comprehensive P05 aggregation: freeze the five pure result tables.

This stage reads only authenticated frozen predictions, aggregates them in
memory and persists the five result tables into an exclusive aggregation
stage. It builds no model, runs no optimizer, fits no temperature, loads no
feature array and selects nothing: every table is a deterministic function of
evidence already authenticated by the frozen-prediction gate.
"""

from __future__ import annotations

import io
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_evaluation as evaluation
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_recovery_source as recovery_source

__all__ = ["P05ComprehensiveAggregationError", "run_aggregation"]

SCHEMA = "nato-sers-p05-comprehensive-aggregation-v1"
PROTOCOL = development.PROTOCOL_VERSION
COMMAND = "run_comprehensive_aggregation"
STAGE_NAME = "aggregation"
RECEIPT_NAME = "aggregation_receipt.json"
EVALUATION_RECEIPT_NAME = evaluation.RECEIPT_NAME
MANIFEST_NAME = evaluation.MANIFEST_NAME
SUMMARY_NAME = "summary.json"
PROVENANCE_BEFORE_NAME = "provenance_before.json"
PROVENANCE_AFTER_NAME = "provenance_after.json"
COMPREHENSIVE_DIR = evaluation.COMPREHENSIVE_DIR
RUNS_DIR = evaluation.RUNS_DIR
MAXIMUM_STORAGE_BYTES = evaluation.MAXIMUM_STORAGE_BYTES
MINIMUM_BUDGET_HEADROOM_BYTES = evaluation.MINIMUM_BUDGET_HEADROOM_BYTES
PRIOR_FIELD = evaluation.PRIOR_FIELD
TABLE_NAMES = (
    "seed_predictions",
    "ensemble_predictions",
    "spectrum_metrics",
    "master_metrics",
    "coverage",
)
STRATEGY_COUNT = 3
COUNTER_KEYS = (
    "rows_seed_predictions",
    "rows_ensemble_predictions",
    "rows_spectrum_metrics",
    "rows_master_metrics",
    "rows_coverage",
    "fits",
    "calibrations",
    "outer_predictions",
    "updates",
)


class P05ComprehensiveAggregationError(core.P05CoreError):
    """Stable, path-free comprehensive-aggregation failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05ComprehensiveAggregationError(code)


def _finite_seconds(value: Any, code: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), code)
    number = float(value)
    _require(math.isfinite(number), code)
    return number


def _integer(value: Any, code: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), code)
    return int(value)


def _source_accounting(auth: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        return recovery_source.from_authenticated(auth)
    except recovery_source.RecoverySourceError as error:
        raise P05ComprehensiveAggregationError("source_accounting_invalid") from error


def _import_runtime() -> dict[str, Any]:
    core._configure_environment()
    import torch

    from atlas_sers.evaluation import p05_frozen_predictions as frozen
    from atlas_sers.evaluation import p05_outer_inputs as outer_inputs
    from atlas_sers.evaluation import p05_pilot as pilot
    from atlas_sers.evaluation import p05_results as results

    return {
        "torch": torch,
        "frozen": frozen,
        "outer_inputs": outer_inputs,
        "pilot": pilot,
        "results": results,
    }


def _write_table(*, stage: Path, budget: Any, name: str, frame: Any) -> int:
    _require(isinstance(frame, pd.DataFrame), "table_not_dataframe")
    buffer = io.BytesIO()
    try:
        frame.to_parquet(buffer, index=False)
    except Exception as error:  # noqa: BLE001
        raise P05ComprehensiveAggregationError("table_serialization_failed") from error
    payload = buffer.getvalue()
    _require(bool(payload), "table_payload_empty")
    budget.check(headroom_bytes=MINIMUM_BUDGET_HEADROOM_BYTES + len(payload))
    path = stage / f"{name}.parquet"
    freeze._reject_preexisting(path, "table_file_exists")
    core._atomic_write(path, payload)
    size = budget.account_new_file(path)
    _require(size == len(payload), "table_size_mismatch")
    stored = core._read_bytes(path, "aggregation_table")
    _require(stored == payload, "table_bytes_mismatch")
    try:
        restored = pd.read_parquet(io.BytesIO(stored))
    except Exception as error:  # noqa: BLE001
        raise P05ComprehensiveAggregationError("table_deserialization_failed") from error
    try:
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True), restored.reset_index(drop=True), check_exact=True
        )
    except AssertionError as error:
        raise P05ComprehensiveAggregationError("table_roundtrip_mismatch") from error
    return int(len(frame))


def _bound_payload(
    identity: Mapping[str, Any],
    counters: Mapping[str, Any],
    prior: float,
    elapsed: float,
) -> dict[str, Any]:
    _require(
        math.isfinite(prior) and math.isfinite(elapsed) and elapsed >= 0.0,
        "aggregation_cumulative_time_invalid",
    )
    _require(
        development.PRELAUNCH_AUDIT_RESERVE_SECONDS <= prior,
        "aggregation_cumulative_time_invalid",
    )
    _require(
        prior + elapsed <= development.MAXIMUM_TOTAL_SECONDS,
        "aggregation_cumulative_time_invalid",
    )
    return {
        **identity,
        "status": "complete",
        "aggregation_complete": True,
        "counters": {**counters, "elapsed_seconds": elapsed},
        "scientific_seconds_this_stage": elapsed,
        "prior_scientific_seconds_cumulative_bound": prior,
        "scientific_seconds_cumulative_bound": prior + elapsed,
        "prelaunch_audit_reserve_seconds": development.PRELAUNCH_AUDIT_RESERVE_SECONDS,
        "maximum_total_seconds": development.MAXIMUM_TOTAL_SECONDS,
    }


def _record_failure(
    *,
    stage: Path,
    identity: Mapping[str, Any],
    counters: Mapping[str, Any],
    prior: float,
    wall_start: float,
    error: BaseException,
) -> None:
    elapsed = time.perf_counter() - wall_start
    payload = {
        **identity,
        "status": "fail",
        "aggregation_complete": False,
        "reason_code": getattr(error, "reason_code", type(error).__name__),
        "counters": {**counters, "elapsed_seconds": elapsed},
        "scientific_seconds_this_stage": elapsed,
        "prior_scientific_seconds_cumulative_bound": prior,
        "scientific_seconds_cumulative_bound": prior + elapsed,
    }
    try:
        core._atomic_write(stage / SUMMARY_NAME, core._canon().canonical_json_bytes(payload))
        core._write_manifest(stage)
    except BaseException:
        pass


def run_aggregation(
    *,
    project_root: Any,
    artifact_root: Any,
    contract_path: Any,
    permit_path: Any,
) -> dict[str, Any]:
    """Aggregate frozen P05 predictions into the five result tables."""

    wall_start = time.perf_counter()
    core._configure_environment()
    bundle = inputs.prepare(
        project_root, artifact_root, contract_path, permit_path, require_unstarted=False
    )
    _require(isinstance(bundle, Mapping), "bundle_malformed")

    runtime = _import_runtime()
    runtime["torch"].set_num_threads(1)
    frozen = runtime["frozen"]
    results = runtime["results"]
    outer_inputs = runtime["outer_inputs"]
    pilot = runtime["pilot"]

    permit_id = str(bundle["permit_sha256"])
    root = Path(bundle["artifact_root"])
    run_root = root / COMPREHENSIVE_DIR / RUNS_DIR / permit_id
    receipt_path = run_root / EVALUATION_RECEIPT_NAME
    evaluation_receipt = freeze._read_mapping(receipt_path, "evaluation_receipt")
    prior = _finite_seconds(evaluation_receipt.get(PRIOR_FIELD), "evaluation_prior_invalid")
    _require(
        development.PRELAUNCH_AUDIT_RESERVE_SECONDS <= prior,
        "evaluation_prior_out_of_range",
    )
    _require(prior <= development.MAXIMUM_TOTAL_SECONDS, "evaluation_prior_out_of_range")
    deadline = wall_start + development.MAXIMUM_TOTAL_SECONDS - prior
    freeze._check_deadline(deadline)

    auth = frozen.authenticate_predictions(bundle, deadline=deadline)
    _require(isinstance(auth, Mapping), "authority_malformed")
    _require(
        _finite_seconds(auth.get("prior_seconds"), "authority_prior_malformed") == prior,
        "authority_prior_mismatch",
    )
    _require(
        dict(auth["evaluation_receipt"]) == dict(evaluation_receipt),
        "evaluation_receipt_changed",
    )
    plan = auth.get("plan")
    _require(isinstance(plan, Mapping), "authority_plan_malformed")
    predictions = auth.get("predictions")
    _require(isinstance(predictions, Mapping), "authority_predictions_malformed")
    source_steps = _integer(auth.get("source_optimizer_steps"), "authority_source_steps_malformed")
    refit_steps = _integer(auth.get("refit_optimizer_steps"), "authority_refit_steps_malformed")
    source_accounting = _source_accounting(auth)
    evaluation_receipt_sha256 = core._canon().sha256_file(receipt_path)

    full_contexts = outer_inputs.load_context_rows(bundle)
    context_count = len(full_contexts)
    _require(context_count > 0, "context_count_empty")
    provenance_before = core._capture_provenance(
        bundle["repository_root"], bundle["project_root"], bundle["artifact_root"]
    )

    identity = {
        "schema_version": SCHEMA,
        "protocol_version": PROTOCOL,
        "command": COMMAND,
        "stage": STAGE_NAME,
        "permit_sha256": permit_id,
        "core_contract_sha256": bundle["contract_sha256"],
        "core_plan_id": bundle["core_plan_id"],
        "ledger_id": bundle["ledger"]["ledger_id"],
        "selection_plan_id": str(plan["plan_id"]),
        "evaluation_receipt_sha256": evaluation_receipt_sha256,
        "source_optimizer_steps": source_steps,
        "refit_optimizer_steps": refit_steps,
        "context_count": context_count,
        "aggregation_complete": False,
    }
    if source_accounting["mode"] == recovery_source.RECOVERY_ACCOUNTING_MODE:
        identity["source_execution_accounting"] = dict(source_accounting)
    counters: dict[str, Any] = {key: 0 for key in COUNTER_KEYS}

    stage = run_root / STAGE_NAME
    consumed = False
    try:
        freeze._reject_preexisting(stage, "aggregation_stage_exists")
        freeze._reject_preexisting(run_root / RECEIPT_NAME, "aggregation_receipt_exists")
        freeze._check_deadline(deadline)
        core._mkdir_exclusive(stage, "aggregation_stage_exists")
        consumed = True
        budget = freeze.StorageBudget(root, run_root, ceiling=MAXIMUM_STORAGE_BYTES)
        freeze._budgeted_write(stage / PROVENANCE_BEFORE_NAME, provenance_before, budget)

        tables = results.aggregate_predictions(
            plan=plan,
            contexts=full_contexts,
            manifest=list(bundle["support"].manifest),
            predictions=predictions,
        )
        _require(isinstance(tables, Mapping), "aggregation_malformed")
        _require(set(tables) == set(TABLE_NAMES), "aggregation_table_set_mismatch")
        coverage = tables["coverage"]
        _require(len(coverage) == STRATEGY_COUNT * context_count, "coverage_row_count_mismatch")
        _require(set(coverage["status"].astype(str)) == {"pass"}, "coverage_status_failure")
        for name in ("spectrum_metrics", "master_metrics"):
            _require(
                len(tables[name]) == STRATEGY_COUNT * context_count, "metric_row_count_mismatch"
            )
        ensemble_rows = STRATEGY_COUNT * sum(len(e["test_uids"]) for e in plan["endpoints"])
        _require(
            len(tables["ensemble_predictions"]) == ensemble_rows, "ensemble_row_count_mismatch"
        )
        _require(
            len(tables["seed_predictions"]) == STRATEGY_COUNT * ensemble_rows,
            "seed_row_count_mismatch",
        )

        for name in TABLE_NAMES:
            freeze._check_deadline(deadline)
            counters[f"rows_{name}"] = _write_table(
                stage=stage, budget=budget, name=name, frame=tables[name]
            )
            budget.check(headroom_bytes=MINIMUM_BUDGET_HEADROOM_BYTES)

        freeze._check_deadline(deadline)
        provenance_after = pilot._post_run_reauth(
            bundle["artifact_root"],
            bundle["contract"],
            bundle["support"],
            provenance_before,
            bundle["repository_root"],
            bundle["project_root"],
        )
        freeze._budgeted_write(stage / PROVENANCE_AFTER_NAME, provenance_after, budget)
        inputs.prepare(
            bundle["project_root"],
            bundle["artifact_root"],
            contract_path,
            permit_path,
            require_unstarted=False,
        )
        freeze._check_deadline(deadline)
        _require(
            core._canon().sha256_file(receipt_path) == evaluation_receipt_sha256,
            "evaluation_receipt_changed",
        )
        _require(
            dict(freeze._read_mapping(receipt_path, "evaluation_receipt"))
            == dict(evaluation_receipt),
            "evaluation_receipt_changed",
        )
        evaluation_stage = run_root / evaluation.STAGE_NAME
        _require(
            core._canon().sha256_file(evaluation_stage / MANIFEST_NAME)
            == evaluation_receipt["stage_manifest_sha256"],
            "evaluation_manifest_changed",
        )
        pilot._verify_manifest(evaluation_stage)

        elapsed = time.perf_counter() - wall_start
        freeze._budgeted_write(
            stage / SUMMARY_NAME, _bound_payload(identity, counters, prior, elapsed), budget
        )
        budget.check(headroom_bytes=MINIMUM_BUDGET_HEADROOM_BYTES)
        core._write_manifest(stage)
        budget.account_new_file(stage / MANIFEST_NAME)
        pilot._verify_manifest(stage)
        budget.reconcile()
        freeze._check_deadline(deadline)
        final_elapsed = time.perf_counter() - wall_start
        receipt = {
            **_bound_payload(identity, counters, prior, final_elapsed),
            "stage_manifest_sha256": core._canon().sha256_file(stage / MANIFEST_NAME),
        }
        freeze._reject_preexisting(run_root / RECEIPT_NAME, "aggregation_receipt_exists")
        freeze._budgeted_write(run_root / RECEIPT_NAME, receipt, budget)
        budget.check()
        freeze._check_deadline(deadline)
        return receipt
    except BaseException as error:
        if consumed:
            _record_failure(
                stage=stage,
                identity=identity,
                counters=counters,
                prior=prior,
                wall_start=wall_start,
                error=error,
            )
        raise
