"""P05 frozen-prediction authentication: read-only final gate for evaluation.

Authenticates the completed comprehensive-evaluation stage before any P03/P04
prediction evidence is opened for comparison. Source/refit authentication is delegated to
:func:`p05_evaluation_authority.authenticate_refits`; this module then verifies
the receipt, summary, manifests, leases, prediction frames and audits,
re-deriving each calibrated probability vector from the persisted logits and
calibration.  It writes nothing, loads no feature arrays, builds no model, runs
no optimizer and never predicts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation import classical
from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_evaluation as evaluation
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_evaluation_authority as authority
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_prediction as prediction
from atlas_sers.evaluation import p05_prediction_io as prediction_io
from atlas_sers.evaluation import p05_recovery_source as recovery_source

__all__ = ["P05FrozenPredictionsError", "authenticate_predictions"]

SCHEMA = evaluation.SCHEMA
PROTOCOL = evaluation.PROTOCOL
COMMAND = evaluation.COMMAND
STAGE_NAME = evaluation.STAGE_NAME
COMPREHENSIVE_DIR = evaluation.COMPREHENSIVE_DIR
RUNS_DIR = evaluation.RUNS_DIR
RECEIPT_NAME = evaluation.RECEIPT_NAME
REFIT_RECEIPT_NAME = evaluation.REFIT_RECEIPT_NAME
MANIFEST_NAME = evaluation.MANIFEST_NAME
LEASE_NAME = evaluation.LEASE_NAME
SUMMARY_NAME = "summary.json"
UNITS_NAME = "units"
REFITS_DIR = "refits"
PREDICTIONS_FILENAME = prediction_io.PREDICTIONS_FILENAME
AUDIT_FILENAME = prediction_io.AUDIT_FILENAME
UID_COLUMN = prediction_io.UID_COLUMN
PROBABILITY_TOLERANCE = 1e-12
ELAPSED_TOLERANCE = 1e-9
COUNTER_KEYS = (
    "started",
    "completed",
    "failed",
    "rows",
    "contexts_completed",
    "peak_cuda_bytes",
    "sum_prediction_elapsed_seconds",
    "optimizer_steps",
    "elapsed_seconds",
)


class P05FrozenPredictionsError(core.P05CoreError):
    """Stable, path-free frozen-prediction authentication failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05FrozenPredictionsError(code)


def _finite(value: Any, code: str) -> float:
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


def _directory(path: Path, name: str) -> None:
    core._reject_symlink_chain(path)
    _require(path.is_dir() and not path.is_symlink(), f"{name}_missing")


def _source_accounting(auth: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        return recovery_source.from_authenticated(auth)
    except recovery_source.RecoverySourceError as error:
        raise P05FrozenPredictionsError("source_accounting_invalid") from error


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
        raise P05FrozenPredictionsError(f"{label}_source_accounting_invalid") from error
    _require(
        core._canon().canonical_json_bytes(validated)
        == core._canon().canonical_json_bytes(dict(expected)),
        f"{label}_source_accounting_mismatch",
    )


def authenticate_predictions(bundle: Any, *, deadline: Any) -> dict[str, Any]:
    """Authenticate the completed comprehensive-evaluation stage read-only."""

    _require(isinstance(bundle, Mapping), "bundle_malformed")
    deadline = _finite(deadline, "deadline_malformed")
    freeze._check_deadline(deadline)

    auth = authority.authenticate_refits(bundle, deadline=deadline)
    _require(isinstance(auth, Mapping), "authority_malformed")
    plan = auth.get("plan")
    _require(isinstance(plan, Mapping), "authority_plan_malformed")
    refit_prior = _finite(auth.get("prior_seconds"), "authority_prior_malformed")
    source_steps = _integer(auth.get("source_optimizer_steps"), "authority_source_steps_malformed")
    refit_steps = _integer(auth.get("refit_optimizer_steps"), "authority_refit_steps_malformed")
    source_accounting = _source_accounting(auth)
    recovered = source_accounting["mode"] == recovery_source.RECOVERY_ACCOUNTING_MODE
    expected_accounting = dict(source_accounting) if recovered else None

    plan_id = plan.get("plan_id")
    _require(isinstance(plan_id, str) and bool(plan_id), "plan_id_malformed")
    unique_refits = plan.get("unique_refits")
    _require(isinstance(unique_refits, Mapping), "plan_unique_refits_malformed")
    aliases = plan.get("strategy_aliases")
    _require(
        isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)),
        "plan_aliases_malformed",
    )
    unique_count = len(unique_refits)
    _require(unique_count > 0, "plan_unique_refits_empty")

    permit_sha256 = bundle.get("permit_sha256")
    _require(isinstance(permit_sha256, str) and bool(permit_sha256), "permit_sha256_malformed")
    ledger = bundle.get("ledger")
    _require(isinstance(ledger, Mapping), "ledger_malformed")
    ledger_id = ledger.get("ledger_id")
    _require(isinstance(ledger_id, str) and bool(ledger_id), "ledger_id_malformed")
    _require(bundle.get("artifact_root") is not None, "artifact_root_missing")

    run_root = Path(bundle["artifact_root"]) / COMPREHENSIVE_DIR / RUNS_DIR / permit_sha256
    refit_receipt_path = run_root / REFIT_RECEIPT_NAME
    core._reject_symlink_chain(refit_receipt_path)
    _require(
        refit_receipt_path.is_file() and not refit_receipt_path.is_symlink(),
        "refit_receipt_missing",
    )
    refit_receipt_sha256 = core._canon().sha256_file(refit_receipt_path)
    _require(
        dict(_mapping(refit_receipt_path, "refit_receipt")) == dict(auth["refit_receipt"]),
        "refit_receipt_changed",
    )

    index = prediction_io.index_endpoints(plan)
    _require(len(index) == prediction_io.EXPECTED_CONTEXTS, "context_count_mismatch")

    identity = {
        "schema_version": SCHEMA,
        "protocol_version": PROTOCOL,
        "command": COMMAND,
        "stage": STAGE_NAME,
        "permit_sha256": permit_sha256,
        "core_contract_sha256": bundle.get("contract_sha256"),
        "core_plan_id": bundle.get("core_plan_id"),
        "ledger_id": ledger_id,
        "selection_plan_id": plan_id,
        "refit_receipt_sha256": refit_receipt_sha256,
        "unique_prediction_count": unique_count,
        "strategy_alias_count": len(aliases),
        "context_count": prediction_io.EXPECTED_CONTEXTS,
        "fits_started": 0,
        "calibrations_started": 0,
        "optimizer_steps": 0,
        "predictions_frozen": True,
        "source_optimizer_steps": source_steps,
        "refit_optimizer_steps": refit_steps,
    }
    if recovered:
        identity["source_execution_accounting"] = dict(source_accounting)

    stage = run_root / STAGE_NAME
    _directory(stage, "evaluation_stage")
    receipt = _mapping(run_root / RECEIPT_NAME, "evaluation_receipt")
    summary = _mapping(stage / SUMMARY_NAME, "evaluation_summary")
    for name, value in identity.items():
        if name == "source_execution_accounting":
            continue
        _require(receipt.get(name) == value, f"receipt_{name}_mismatch")
        _require(summary.get(name) == value, f"summary_{name}_mismatch")
    _require_accounting(
        receipt, label="receipt", source_steps=source_steps, expected=expected_accounting
    )
    _require_accounting(
        summary, label="summary", source_steps=source_steps, expected=expected_accounting
    )
    for label, payload in (("receipt", receipt), ("summary", summary)):
        _require(payload.get("predictions_frozen") is True, f"{label}_predictions_unfrozen")
        _require(payload.get("predictions_complete") is True, f"{label}_predictions_incomplete")
        _require(payload.get("all_complete") is True, f"{label}_all_incomplete")
        _require(payload.get("status") == "complete", f"{label}_status_incomplete")
        for name in ("fits_started", "calibrations_started", "optimizer_steps"):
            _require(
                _integer(payload.get(name), f"{label}_{name}_invalid") == 0,
                f"{label}_{name}_nonzero",
            )

    manifest_path = stage / MANIFEST_NAME
    core._reject_symlink_chain(manifest_path)
    _require(
        manifest_path.is_file() and not manifest_path.is_symlink(), "evaluation_manifest_missing"
    )
    stage_manifest_sha256 = receipt.get("stage_manifest_sha256")
    _require(
        isinstance(stage_manifest_sha256, str) and bool(stage_manifest_sha256),
        "stage_manifest_sha256_malformed",
    )
    _require(
        core._canon().sha256_file(manifest_path) == stage_manifest_sha256,
        "stage_manifest_sha256_mismatch",
    )
    pilot._verify_manifest(stage)
    freeze._check_deadline(deadline)

    receipt_counters = receipt.get("counters")
    summary_counters = summary.get("counters")
    _require(isinstance(receipt_counters, Mapping), "receipt_counters_malformed")
    _require(isinstance(summary_counters, Mapping), "summary_counters_malformed")
    _require(
        set(receipt_counters) == set(summary_counters) == set(COUNTER_KEYS),
        "counter_fields_mismatch",
    )
    for name in COUNTER_KEYS:
        if name != "elapsed_seconds":
            _require(
                receipt_counters.get(name) == summary_counters.get(name),
                f"counters_{name}_mismatch",
            )

    receipt_elapsed = _finite(receipt_counters["elapsed_seconds"], "receipt_elapsed_malformed")
    summary_elapsed = _finite(summary_counters["elapsed_seconds"], "summary_elapsed_malformed")
    _require(0.0 <= summary_elapsed <= receipt_elapsed, "summary_elapsed_exceeds_receipt")
    sum_prediction = _finite(
        receipt_counters["sum_prediction_elapsed_seconds"], "sum_prediction_elapsed_malformed"
    )
    _require(0.0 <= sum_prediction <= summary_elapsed, "sum_prediction_exceeds_elapsed")

    for label, payload in (("receipt", receipt), ("summary", summary)):
        prior = _finite(
            payload.get("prior_scientific_seconds_cumulative_bound"), f"{label}_prior_malformed"
        )
        _require(prior == refit_prior, f"{label}_prior_mismatch")
        stage_seconds = _finite(
            payload.get("scientific_seconds_this_stage"), f"{label}_stage_malformed"
        )
        _require(
            stage_seconds == payload["counters"]["elapsed_seconds"],
            f"{label}_stage_seconds_inconsistent",
        )
        cumulative = _finite(
            payload.get("scientific_seconds_cumulative_bound"), f"{label}_cumulative_malformed"
        )
        _require(cumulative == refit_prior + stage_seconds, f"{label}_cumulative_mismatch")
        _require(
            development.PRELAUNCH_AUDIT_RESERVE_SECONDS
            <= cumulative
            <= development.MAXIMUM_TOTAL_SECONDS,
            f"{label}_cumulative_out_of_range",
        )
        _require(
            _finite(payload.get("prelaunch_audit_reserve_seconds"), f"{label}_reserve_malformed")
            == development.PRELAUNCH_AUDIT_RESERVE_SECONDS,
            f"{label}_reserve_mismatch",
        )
        _require(
            _finite(payload.get("maximum_total_seconds"), f"{label}_maximum_malformed")
            == development.MAXIMUM_TOTAL_SECONDS,
            f"{label}_maximum_mismatch",
        )
    evaluation_cumulative = _finite(
        receipt["scientific_seconds_cumulative_bound"], "receipt_cumulative_malformed"
    )

    _require(
        _integer(receipt_counters["started"], "started_malformed") == unique_count,
        "started_mismatch",
    )
    _require(
        _integer(receipt_counters["completed"], "completed_malformed") == unique_count,
        "completed_mismatch",
    )
    _require(_integer(receipt_counters["failed"], "failed_malformed") == 0, "failed_nonzero")
    _require(
        _integer(receipt_counters["contexts_completed"], "contexts_malformed") == len(index),
        "contexts_completed_mismatch",
    )
    _require(
        _integer(receipt_counters["optimizer_steps"], "optimizer_steps_malformed") == 0,
        "optimizer_steps_nonzero",
    )
    peak_counter = _integer(receipt_counters["peak_cuda_bytes"], "peak_cuda_malformed")
    _require(0 <= peak_counter <= prediction_io.MAX_PEAK_CUDA_BYTES, "peak_cuda_exceeded")
    expected_rows = sum(
        len(entry["endpoint"]["test_uids"]) for entry in index.values() for _ in entry["specs"]
    )
    _require(
        _integer(receipt_counters["rows"], "rows_malformed") == expected_rows,
        "row_count_mismatch",
    )

    units_dir = stage / UNITS_NAME
    _directory(units_dir, "units_dir")
    present_units = set()
    for entry in sorted(units_dir.iterdir(), key=lambda path: path.name):
        _require(not entry.is_symlink(), "unit_symlink_rejected")
        _require(entry.is_dir(), "unit_entry_not_directory")
        present_units.add(entry.name)
    _require(
        present_units == set(str(key) for key in unique_refits), "units_directory_set_mismatch"
    )
    freeze._check_deadline(deadline)

    predictions: dict[str, pd.DataFrame] = {}
    row_sum = 0
    elapsed_sum = 0.0
    peak_cuda = 0
    for context_id in sorted(index):
        freeze._check_deadline(deadline)
        endpoint = index[context_id]["endpoint"]
        test_uids = prediction_io._validate_endpoint_uids(endpoint)
        for spec in index[context_id]["specs"]:
            freeze._check_deadline(deadline)
            refit_id = str(spec["refit_id"])
            _require(refit_id not in predictions, "prediction_duplicate_refit")
            unit_dir = units_dir / refit_id
            _directory(unit_dir, "unit_directory")
            pilot._verify_manifest(unit_dir)

            lease = _mapping(unit_dir / LEASE_NAME, "unit_lease")
            _require(
                core._canon().canonical_json_bytes(dict(lease))
                == core._canon().canonical_json_bytes(
                    {"selection_plan_id": plan_id, "spec": spec, "endpoint": endpoint}
                ),
                "unit_lease_mismatch",
            )

            predictions_path = unit_dir / PREDICTIONS_FILENAME
            core._reject_symlink_chain(predictions_path)
            _require(
                predictions_path.is_file() and not predictions_path.is_symlink(),
                "predictions_missing",
            )
            frame = pd.read_csv(
                predictions_path,
                dtype={UID_COLUMN: str},
                keep_default_na=False,
                float_precision="round_trip",
            )
            prediction_io._validate_frame(frame, test_uids)

            audit = _mapping(unit_dir / AUDIT_FILENAME, "prediction_audit")
            prediction_io._validate_audit(audit, spec, test_uids, len(frame))

            refit_unit_dir = run_root / REFITS_DIR / UNITS_NAME / refit_id
            refit_summary = _mapping(refit_unit_dir / SUMMARY_NAME, "refit_summary")
            _require(
                audit["model_state_sha256"] == refit_summary.get("terminal_state_digest"),
                "audit_model_digest_mismatch",
            )
            calibration, calibration_sha256 = prediction._load_calibration(refit_unit_dir, spec)
            _require(
                audit["calibration_state_sha256"] == calibration_sha256,
                "audit_calibration_digest_mismatch",
            )

            logits = np.column_stack(
                [
                    frame[f"logit_{class_index}"].to_numpy(dtype=np.float64)
                    for class_index in range(prediction_io.EXPECTED_CLASS_COUNT)
                ]
            )
            calibrated = classical.apply_temperature(logits, calibration)
            for class_index in range(prediction_io.EXPECTED_CLASS_COUNT):
                stored = frame[f"probability_{class_index}"].to_numpy(dtype=np.float64)
                _require(
                    bool(
                        np.allclose(
                            calibrated[:, class_index],
                            stored,
                            rtol=0.0,
                            atol=PROBABILITY_TOLERANCE,
                        )
                    ),
                    "probability_vector_mismatch",
                )

            predictions[refit_id] = frame
            row_sum += int(len(frame))
            elapsed_sum += float(audit["elapsed_seconds"])
            peak_cuda = max(peak_cuda, int(audit["peak_cuda_bytes"]))
    freeze._check_deadline(deadline)

    _require(set(predictions) == set(str(key) for key in unique_refits), "prediction_set_mismatch")
    _require(row_sum == expected_rows, "prediction_row_total_mismatch")
    _require(
        math.isclose(elapsed_sum, sum_prediction, rel_tol=0.0, abs_tol=ELAPSED_TOLERANCE),
        "sum_prediction_elapsed_mismatch",
    )
    _require(peak_cuda == peak_counter, "peak_cuda_mismatch")

    result = {
        "plan": plan,
        "predictions": predictions,
        "evaluation_receipt": receipt,
        "prior_seconds": evaluation_cumulative,
        "source_optimizer_steps": source_steps,
        "refit_optimizer_steps": refit_steps,
    }
    if recovered:
        result["source_execution_accounting"] = dict(source_accounting)
    return result
