"""P05 aggregation authority: read-only authentication of the aggregation stage.

Authenticates the completed comprehensive-aggregation stage after the frozen
predictions have themselves been authenticated. The five persisted result
tables are re-derived from the authenticated frozen predictions and compared
with the saved parquet tables; the receipt, summary and manifest are checked
for mutual consistency. This module writes nothing, loads no feature arrays,
builds no model, runs no optimizer and never predicts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_sers.evaluation import p05_comprehensive_aggregation as producer
from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_evaluation as evaluation
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_frozen_predictions as frozen
from atlas_sers.evaluation import p05_outer_inputs as outer_inputs
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_results as results

__all__ = ["P05AggregationAuthorityError", "authenticate_aggregation"]

SCHEMA = producer.SCHEMA
PROTOCOL = producer.PROTOCOL
COMMAND = producer.COMMAND
STAGE_NAME = producer.STAGE_NAME
RECEIPT_NAME = producer.RECEIPT_NAME
MANIFEST_NAME = producer.MANIFEST_NAME
SUMMARY_NAME = producer.SUMMARY_NAME
PROVENANCE_BEFORE_NAME = producer.PROVENANCE_BEFORE_NAME
PROVENANCE_AFTER_NAME = producer.PROVENANCE_AFTER_NAME
COMPREHENSIVE_DIR = producer.COMPREHENSIVE_DIR
RUNS_DIR = producer.RUNS_DIR
TABLE_NAMES = producer.TABLE_NAMES
COUNTER_KEYS = producer.COUNTER_KEYS
ZERO_COUNTERS = ("fits", "calibrations", "outer_predictions", "updates")
ELAPSED_KEY = "elapsed_seconds"
COUNTER_FIELDS = frozenset(COUNTER_KEYS) | {ELAPSED_KEY}


class P05AggregationAuthorityError(core.P05CoreError):
    """Stable, path-free aggregation-authority failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05AggregationAuthorityError(code)


def _finite_seconds(value: Any, code: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), code)
    number = float(value)
    _require(math.isfinite(number), code)
    return number


def _integer(value: Any, code: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), code)
    return int(value)


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
            saved.reset_index(drop=True), recomputed.reset_index(drop=True), check_exact=True
        )
    except AssertionError:
        return False
    return True


def authenticate_aggregation(bundle: Any, *, deadline: Any) -> dict[str, Any]:
    """Authenticate the completed comprehensive-aggregation stage read-only."""

    _require(isinstance(bundle, Mapping), "bundle_malformed")
    deadline = _finite_seconds(deadline, "deadline_malformed")
    freeze._check_deadline(deadline)

    auth = frozen.authenticate_predictions(bundle, deadline=deadline)
    _require(isinstance(auth, Mapping), "authority_malformed")
    plan = auth.get("plan")
    _require(isinstance(plan, Mapping), "authority_plan_malformed")
    predictions = auth.get("predictions")
    _require(isinstance(predictions, Mapping), "authority_predictions_malformed")
    evaluation_receipt = auth.get("evaluation_receipt")
    _require(isinstance(evaluation_receipt, Mapping), "authority_evaluation_receipt_malformed")
    evaluation_prior = _finite_seconds(auth.get("prior_seconds"), "authority_prior_malformed")
    source_steps = _integer(auth.get("source_optimizer_steps"), "authority_source_steps_malformed")
    refit_steps = _integer(auth.get("refit_optimizer_steps"), "authority_refit_steps_malformed")

    permit_sha256 = bundle.get("permit_sha256")
    _require(isinstance(permit_sha256, str) and bool(permit_sha256), "permit_sha256_malformed")
    ledger = bundle.get("ledger")
    _require(isinstance(ledger, Mapping), "ledger_malformed")
    ledger_id = ledger.get("ledger_id")
    _require(isinstance(ledger_id, str) and bool(ledger_id), "ledger_id_malformed")
    _require(bundle.get("artifact_root") is not None, "artifact_root_missing")
    support = bundle.get("support")
    _require(support is not None and hasattr(support, "manifest"), "support_malformed")
    endpoints = plan.get("endpoints")
    _require(
        isinstance(endpoints, Sequence) and not isinstance(endpoints, (str, bytes)),
        "plan_endpoints_malformed",
    )

    run_root = Path(bundle["artifact_root"]) / COMPREHENSIVE_DIR / RUNS_DIR / permit_sha256
    stage = run_root / STAGE_NAME
    evaluation_stage = run_root / evaluation.STAGE_NAME
    evaluation_receipt_path = run_root / producer.EVALUATION_RECEIPT_NAME
    evaluation_manifest_path = evaluation_stage / evaluation.MANIFEST_NAME
    manifest_path = stage / MANIFEST_NAME

    core._reject_symlink_chain(evaluation_receipt_path)
    _require(
        evaluation_receipt_path.is_file() and not evaluation_receipt_path.is_symlink(),
        "evaluation_receipt_missing",
    )
    core._reject_symlink_chain(evaluation_manifest_path)
    _require(
        evaluation_manifest_path.is_file() and not evaluation_manifest_path.is_symlink(),
        "evaluation_manifest_missing",
    )
    evaluation_receipt_sha256 = core._canon().sha256_file(evaluation_receipt_path)
    evaluation_manifest_sha256 = core._canon().sha256_file(evaluation_manifest_path)
    _require(
        dict(_mapping(evaluation_receipt_path, "evaluation_receipt")) == dict(evaluation_receipt),
        "evaluation_receipt_changed",
    )
    _require(
        evaluation_receipt.get("stage_manifest_sha256") == evaluation_manifest_sha256,
        "evaluation_manifest_mismatch",
    )

    receipt = _mapping(run_root / RECEIPT_NAME, "aggregation_receipt")
    receipt_sha256 = core._canon().sha256_file(run_root / RECEIPT_NAME)
    summary = _mapping(stage / SUMMARY_NAME, "aggregation_summary")
    core._reject_symlink_chain(manifest_path)
    _require(
        manifest_path.is_file() and not manifest_path.is_symlink(), "aggregation_manifest_missing"
    )
    stage_manifest_sha256 = core._canon().sha256_file(manifest_path)
    _require(
        receipt.get("stage_manifest_sha256") == stage_manifest_sha256,
        "aggregation_manifest_sha256_mismatch",
    )
    pilot._verify_manifest(stage)

    full_contexts = outer_inputs.load_context_rows(bundle)
    context_count = len(full_contexts)
    _require(context_count > 0, "context_count_empty")
    _require(context_count == len(endpoints), "context_count_plan_mismatch")

    expected_identity = {
        "schema_version": SCHEMA,
        "protocol_version": PROTOCOL,
        "command": COMMAND,
        "stage": STAGE_NAME,
        "permit_sha256": permit_sha256,
        "core_contract_sha256": bundle.get("contract_sha256"),
        "core_plan_id": bundle.get("core_plan_id"),
        "ledger_id": ledger_id,
        "selection_plan_id": str(plan.get("plan_id")),
        "evaluation_receipt_sha256": evaluation_receipt_sha256,
        "source_optimizer_steps": source_steps,
        "refit_optimizer_steps": refit_steps,
        "context_count": context_count,
    }
    for label, payload in (("receipt", receipt), ("summary", summary)):
        for name, value in expected_identity.items():
            _require(payload.get(name) == value, f"{label}_{name}_mismatch")
        for name in ("source_optimizer_steps", "refit_optimizer_steps", "context_count"):
            _integer(payload.get(name), f"{label}_{name}_invalid")
        _require(payload.get("status") == "complete", f"{label}_status_incomplete")
        _require(payload.get("aggregation_complete") is True, f"{label}_incomplete")

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
        _require(prior_bound == evaluation_prior, f"{label}_prior_mismatch")
        stage_seconds = _finite_seconds(
            payload.get("scientific_seconds_this_stage"), f"{label}_stage_malformed"
        )
        _require(
            stage_seconds == payload["counters"][ELAPSED_KEY],
            f"{label}_stage_seconds_inconsistent",
        )
        cumulative = _finite_seconds(
            payload.get("scientific_seconds_cumulative_bound"), f"{label}_cumulative_malformed"
        )
        _require(cumulative == evaluation_prior + stage_seconds, f"{label}_cumulative_mismatch")
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
    aggregation_cumulative = _finite_seconds(
        receipt.get("scientific_seconds_cumulative_bound"), "aggregation_cumulative_malformed"
    )

    core._reject_symlink_chain(stage)
    _require(stage.is_dir() and not stage.is_symlink(), "aggregation_stage_missing")
    expected_files = {f"{name}.parquet" for name in TABLE_NAMES} | {
        PROVENANCE_BEFORE_NAME,
        PROVENANCE_AFTER_NAME,
        SUMMARY_NAME,
        MANIFEST_NAME,
    }
    actual_files: set[str] = set()
    for entry in stage.iterdir():
        _require(not entry.is_symlink(), "aggregation_symlink_rejected")
        _require(entry.is_file(), "aggregation_inventory_not_file")
        actual_files.add(entry.name)
    _require(actual_files == expected_files, "aggregation_inventory_mismatch")

    tables: dict[str, pd.DataFrame] = {}
    for name in TABLE_NAMES:
        freeze._check_deadline(deadline)
        path = stage / f"{name}.parquet"
        core._reject_symlink_chain(path)
        _require(path.is_file() and not path.is_symlink(), "aggregation_table_missing")
        try:
            frame = pd.read_parquet(path)
        except Exception as error:  # noqa: BLE001
            raise P05AggregationAuthorityError("aggregation_table_read_failed") from error
        _require(isinstance(frame, pd.DataFrame), "aggregation_table_malformed")
        tables[name] = frame

    freeze._check_deadline(deadline)
    recomputed = results.aggregate_predictions(
        plan=plan,
        contexts=full_contexts,
        manifest=list(support.manifest),
        predictions=predictions,
    )
    _require(isinstance(recomputed, Mapping), "recomputed_malformed")
    _require(set(recomputed) == set(TABLE_NAMES), "recomputed_table_set_mismatch")
    for name in TABLE_NAMES:
        freeze._check_deadline(deadline)
        _require(_frames_equal(tables[name], recomputed[name]), f"table_{name}_mismatch")
        expected_rows = len(recomputed[name])
        _require(
            _integer(receipt_counters[f"rows_{name}"], f"receipt_rows_{name}_invalid")
            == expected_rows,
            f"receipt_rows_{name}_mismatch",
        )
        _require(
            _integer(summary_counters[f"rows_{name}"], f"summary_rows_{name}_invalid")
            == expected_rows,
            f"summary_rows_{name}_mismatch",
        )

    freeze._check_deadline(deadline)
    _require(
        core._canon().sha256_file(evaluation_receipt_path) == evaluation_receipt_sha256,
        "evaluation_receipt_changed",
    )
    _require(
        core._canon().sha256_file(evaluation_manifest_path) == evaluation_manifest_sha256,
        "evaluation_manifest_changed",
    )
    _require(
        core._canon().sha256_file(manifest_path) == stage_manifest_sha256,
        "aggregation_manifest_changed",
    )
    _require(
        core._canon().sha256_file(run_root / RECEIPT_NAME) == receipt_sha256,
        "aggregation_receipt_changed",
    )
    pilot._verify_manifest(stage)
    pilot._verify_manifest(evaluation_stage)
    freeze._check_deadline(deadline)

    return {
        **auth,
        "aggregation_receipt": receipt,
        "aggregation_tables": tables,
        "prior_seconds": aggregation_cumulative,
        "evaluation_prior_seconds": evaluation_prior,
    }
