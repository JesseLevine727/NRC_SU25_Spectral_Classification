"""Comprehensive P05 comparison: descriptive refit-versus-reference report.

This stage authenticates the completed comprehensive-aggregation stage, opens
only the two pinned legacy prediction shards through the gated reference
loader, and compares the three frozen P05 strategies with the four classical
procedures and the historical ``D0-ERM`` reference.  It builds no model, runs
no optimizer, fits no temperature, loads no feature array and selects nothing:
every table is a deterministic function of evidence already authenticated by
the frozen-prediction and aggregation gates.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_sers.evaluation import p05_comparison as comparison
from atlas_sers.evaluation import p05_comprehensive_aggregation as aggregation
from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_evaluation as evaluation
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_recovery_source as recovery_source
from atlas_sers.governance.canonical import sha256_value

__all__ = ["P05ComprehensiveComparisonError", "run_comparison"]

SCHEMA = "nato-sers-p05-comprehensive-comparison-v1"
PROTOCOL = development.PROTOCOL_VERSION
COMMAND = "run_comprehensive_comparison"
STAGE_NAME = "comparison"
RECEIPT_NAME = "comparison_receipt.json"
MANIFEST_NAME = evaluation.MANIFEST_NAME
SUMMARY_NAME = "summary.json"
PROVENANCE_BEFORE_NAME = "provenance_before.json"
PROVENANCE_AFTER_NAME = "provenance_after.json"
BINDINGS_NAME = "reference_bindings.json"
COMPREHENSIVE_DIR = evaluation.COMPREHENSIVE_DIR
RUNS_DIR = evaluation.RUNS_DIR
MAXIMUM_STORAGE_BYTES = evaluation.MAXIMUM_STORAGE_BYTES
MINIMUM_BUDGET_HEADROOM_BYTES = evaluation.MINIMUM_BUDGET_HEADROOM_BYTES
PRIOR_FIELD = evaluation.PRIOR_FIELD
AGGREGATION_RECEIPT_NAME = aggregation.RECEIPT_NAME
AGGREGATION_MANIFEST_NAME = aggregation.MANIFEST_NAME
AGGREGATION_STAGE_NAME = aggregation.STAGE_NAME
TABLE_NAMES = ("endpoint_metrics", "paired_metrics", "coverage", "summary")
COUNTER_KEYS = (
    "rows_endpoint_metrics",
    "rows_paired_metrics",
    "rows_coverage",
    "rows_summary",
    "held_contexts",
    "complete_model_contexts",
    "incomplete_model_contexts",
    "fits",
    "calibrations",
    "outer_predictions",
    "updates",
)
ZERO_COUNTERS = ("fits", "calibrations", "outer_predictions", "updates")
EXPECTED_MODEL_COUNT = 8
EXPECTED_PAIR_COUNT = 17
EXPECTED_AGGREGATION_COUNT = 2


class P05ComprehensiveComparisonError(core.P05CoreError):
    """Stable, path-free comprehensive-comparison failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05ComprehensiveComparisonError(code)


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
        raise P05ComprehensiveComparisonError("source_accounting_invalid") from error


def _import_runtime() -> dict[str, Any]:
    core._configure_environment()
    import torch

    from atlas_sers.evaluation import p05_aggregation_authority as authority
    from atlas_sers.evaluation import p05_legacy_references as legacy_references
    from atlas_sers.evaluation import p05_outer_inputs as outer_inputs
    from atlas_sers.evaluation import p05_pilot as pilot

    return {
        "torch": torch,
        "authority": authority,
        "legacy_references": legacy_references,
        "outer_inputs": outer_inputs,
        "pilot": pilot,
    }


def _comparison_counters(
    tables: Mapping[str, Any], *, held_context_ids: set[str]
) -> dict[str, int]:
    for name in TABLE_NAMES:
        _require(isinstance(tables.get(name), pd.DataFrame), f"table_{name}_malformed")
    endpoint = tables["endpoint_metrics"]
    paired = tables["paired_metrics"]
    coverage = tables["coverage"]
    summary = tables["summary"]

    model_count = len(comparison.ALL_MODELS)
    aggregation_count = len(comparison.AGGREGATIONS)
    pair_count = len(comparison.PAIRS)
    _require(model_count == EXPECTED_MODEL_COUNT, "model_count_mismatch")
    _require(pair_count == EXPECTED_PAIR_COUNT, "pair_count_mismatch")
    _require(aggregation_count == EXPECTED_AGGREGATION_COUNT, "aggregation_count_mismatch")
    _require("complete" in coverage.columns, "coverage_complete_missing")
    _require(
        pd.api.types.is_bool_dtype(coverage["complete"]) and not coverage["complete"].isna().any(),
        "coverage_complete_invalid",
    )
    _require(len(coverage) % model_count == 0, "coverage_row_count_mismatch")

    held_contexts = len(coverage) // model_count
    _require(held_contexts > 0, "held_contexts_empty")
    _require(held_contexts == len(held_context_ids), "held_context_count_mismatch")
    coverage_keys = list(zip(coverage.context_id, coverage.model_id, strict=True))
    expected_coverage = {
        (context_id, model) for context_id in held_context_ids for model in comparison.ALL_MODELS
    }
    _require(
        len(set(coverage_keys)) == len(coverage_keys) and set(coverage_keys) == expected_coverage,
        "coverage_key_mismatch",
    )
    complete = int(coverage["complete"].sum())
    incomplete = len(coverage) - complete
    _require(
        len(endpoint) == complete * aggregation_count,
        "endpoint_row_count_mismatch",
    )
    _require(
        len(paired) == held_contexts * pair_count * aggregation_count,
        "paired_row_count_mismatch",
    )
    endpoint_keys = list(
        zip(endpoint.context_id, endpoint.model_id, endpoint.aggregation_id, strict=True)
    )
    expected_endpoints = {
        (row.context_id, row.model_id, endpoint_id)
        for row in coverage.itertuples(index=False)
        if row.complete
        for endpoint_id in comparison.AGGREGATIONS
    }
    _require(
        len(set(endpoint_keys)) == len(endpoint_keys) and set(endpoint_keys) == expected_endpoints,
        "endpoint_key_mismatch",
    )
    paired_keys = list(
        zip(
            paired.context_id,
            paired.model_id,
            paired.reference_model_id,
            paired.aggregation_id,
            strict=True,
        )
    )
    expected_pairs = {
        (context_id, model, reference, endpoint_id)
        for context_id in held_context_ids
        for model, reference in comparison.PAIRS
        for endpoint_id in comparison.AGGREGATIONS
    }
    _require(
        len(set(paired_keys)) == len(paired_keys) and set(paired_keys) == expected_pairs,
        "paired_key_mismatch",
    )

    counters = {
        "rows_endpoint_metrics": int(len(endpoint)),
        "rows_paired_metrics": int(len(paired)),
        "rows_coverage": int(len(coverage)),
        "rows_summary": int(len(summary)),
        "held_contexts": int(held_contexts),
        "complete_model_contexts": complete,
        "incomplete_model_contexts": incomplete,
        "fits": 0,
        "calibrations": 0,
        "outer_predictions": 0,
        "updates": 0,
    }
    _require(set(counters) == set(COUNTER_KEYS), "counter_fields_mismatch")
    for name in ZERO_COUNTERS:
        _require(
            _integer(counters[name], f"counter_{name}_invalid") == 0, f"counter_{name}_nonzero"
        )
    return counters


def _bound_payload(
    identity: Mapping[str, Any],
    counters: Mapping[str, Any],
    prior: float,
    elapsed: float,
) -> dict[str, Any]:
    _require(
        math.isfinite(prior) and math.isfinite(elapsed) and elapsed >= 0.0,
        "comparison_cumulative_time_invalid",
    )
    _require(
        development.PRELAUNCH_AUDIT_RESERVE_SECONDS <= prior,
        "comparison_cumulative_time_invalid",
    )
    _require(
        prior + elapsed <= development.MAXIMUM_TOTAL_SECONDS,
        "comparison_cumulative_time_invalid",
    )
    return {
        **identity,
        "status": "complete",
        "comparison_complete": True,
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
        "comparison_complete": False,
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


def run_comparison(
    *,
    project_root: Any,
    artifact_root: Any,
    contract_path: Any,
    permit_path: Any,
) -> dict[str, Any]:
    """Compare frozen P05 predictions against frozen P04/P03 references."""

    wall_start = time.perf_counter()
    core._configure_environment()
    bundle = inputs.prepare(
        project_root, artifact_root, contract_path, permit_path, require_unstarted=False
    )
    _require(isinstance(bundle, Mapping), "bundle_malformed")

    permit_id = str(bundle["permit_sha256"])
    root = Path(bundle["artifact_root"])
    run_root = root / COMPREHENSIVE_DIR / RUNS_DIR / permit_id
    aggregation_receipt_path = run_root / AGGREGATION_RECEIPT_NAME
    aggregation_receipt = freeze._read_mapping(aggregation_receipt_path, "aggregation_receipt")
    prior = _finite_seconds(aggregation_receipt.get(PRIOR_FIELD), "aggregation_prior_invalid")
    _require(development.PRELAUNCH_AUDIT_RESERVE_SECONDS <= prior, "aggregation_prior_out_of_range")
    _require(prior <= development.MAXIMUM_TOTAL_SECONDS, "aggregation_prior_out_of_range")
    deadline = wall_start + development.MAXIMUM_TOTAL_SECONDS - prior
    freeze._check_deadline(deadline)

    runtime = _import_runtime()
    runtime["torch"].set_num_threads(1)
    authority = runtime["authority"]
    legacy_references = runtime["legacy_references"]
    outer_inputs = runtime["outer_inputs"]
    pilot = runtime["pilot"]

    auth = authority.authenticate_aggregation(bundle, deadline=deadline)
    _require(isinstance(auth, Mapping), "authority_malformed")
    _require(
        _finite_seconds(auth.get("prior_seconds"), "authority_prior_malformed") == prior,
        "authority_prior_mismatch",
    )
    _require(
        dict(auth["aggregation_receipt"]) == dict(aggregation_receipt),
        "aggregation_receipt_changed",
    )
    plan = auth.get("plan")
    _require(isinstance(plan, Mapping), "authority_plan_malformed")
    plan_id = plan.get("plan_id")
    _require(plan_id is not None and str(plan_id) != "", "authority_plan_id_malformed")
    _require(isinstance(auth.get("aggregation_tables"), Mapping), "authority_tables_malformed")
    source_steps = _integer(auth.get("source_optimizer_steps"), "authority_source_steps_malformed")
    refit_steps = _integer(auth.get("refit_optimizer_steps"), "authority_refit_steps_malformed")
    source_accounting = _source_accounting(auth)
    aggregation_receipt_sha256 = core._canon().sha256_file(aggregation_receipt_path)

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
        "selection_plan_id": str(plan_id),
        "aggregation_receipt_sha256": aggregation_receipt_sha256,
        "source_optimizer_steps": source_steps,
        "refit_optimizer_steps": refit_steps,
        "comparison_complete": False,
    }
    if source_accounting["mode"] == recovery_source.RECOVERY_ACCOUNTING_MODE:
        identity["source_execution_accounting"] = dict(source_accounting)
    counters: dict[str, Any] = {key: 0 for key in COUNTER_KEYS}

    stage = run_root / STAGE_NAME
    consumed = False
    try:
        freeze._reject_preexisting(stage, "comparison_stage_exists")
        freeze._reject_preexisting(run_root / RECEIPT_NAME, "comparison_receipt_exists")
        freeze._check_deadline(deadline)
        core._mkdir_exclusive(stage, "comparison_stage_exists")
        consumed = True
        budget = freeze.StorageBudget(root, run_root, ceiling=MAXIMUM_STORAGE_BYTES)
        freeze._budgeted_write(stage / PROVENANCE_BEFORE_NAME, provenance_before, budget)

        legacy = legacy_references.load_references(bundle, authenticated=auth, deadline=deadline)
        _require(isinstance(legacy, Mapping), "legacy_malformed")
        bindings = legacy.get("bindings")
        _require(isinstance(bindings, Mapping), "legacy_bindings_malformed")
        binding_digest = sha256_value(dict(bindings))
        identity = {**identity, "reference_binding_digest": binding_digest}
        freeze._budgeted_write(stage / BINDINGS_NAME, dict(bindings), budget)

        full_contexts = outer_inputs.load_context_rows(bundle)
        contexts = pd.DataFrame(full_contexts)
        _require(isinstance(contexts, pd.DataFrame) and len(contexts) > 0, "context_registry_empty")

        p05_ensemble = auth["aggregation_tables"]["ensemble_predictions"]
        tables = comparison.compare_predictions(
            p05_ensemble=p05_ensemble,
            p04_ensemble=legacy["p04_ensemble"],
            p03_predictions=legacy["p03_predictions"],
            contexts=contexts,
        )
        _require(isinstance(tables, Mapping), "comparison_malformed")
        _require(set(tables) == set(TABLE_NAMES), "comparison_table_set_mismatch")
        held = comparison._held_contexts(contexts)
        counters.update(
            _comparison_counters(tables, held_context_ids=set(held.context_id.astype(str)))
        )

        for name in TABLE_NAMES:
            freeze._check_deadline(deadline)
            written = aggregation._write_table(
                stage=stage, budget=budget, name=name, frame=tables[name]
            )
            _require(
                _integer(counters[f"rows_{name}"], f"rows_{name}_invalid") == written,
                "table_row_count_mismatch",
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

        legacy_again = legacy_references.load_references(
            bundle, authenticated=auth, deadline=deadline
        )
        _require(isinstance(legacy_again, Mapping), "legacy_malformed")
        _require(
            dict(legacy_again["bindings"]) == dict(bindings),
            "legacy_bindings_changed",
        )
        _require(
            core._canon().sha256_file(aggregation_receipt_path) == aggregation_receipt_sha256,
            "aggregation_receipt_changed",
        )
        _require(
            dict(freeze._read_mapping(aggregation_receipt_path, "aggregation_receipt"))
            == dict(aggregation_receipt),
            "aggregation_receipt_changed",
        )
        aggregation_stage = run_root / AGGREGATION_STAGE_NAME
        _require(
            core._canon().sha256_file(aggregation_stage / AGGREGATION_MANIFEST_NAME)
            == aggregation_receipt["stage_manifest_sha256"],
            "aggregation_manifest_changed",
        )
        pilot._verify_manifest(aggregation_stage)
        _require(
            dict(freeze._read_mapping(stage / BINDINGS_NAME, "reference_bindings"))
            == dict(bindings),
            "persisted_reference_bindings_changed",
        )

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
        freeze._reject_preexisting(run_root / RECEIPT_NAME, "comparison_receipt_exists")
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
