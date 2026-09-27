"""Comprehensive P05 evaluation orchestration.

Predicts and persists every selected comprehensive refit into an exclusive
evaluation stage, delegating persistence, receipts and cumulative-time
accounting to the close-out helper.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_core_run as core

__all__ = ["P05ComprehensiveEvaluationError", "run_evaluation"]

SCHEMA = "nato-sers-p05-comprehensive-evaluation-v1"
PROTOCOL = development.PROTOCOL_VERSION
COMMAND = "run_comprehensive_evaluation"
STAGE_NAME = "evaluation"
COMPREHENSIVE_DIR = Path("p05comprehensive")
RUNS_DIR = Path("runs")
RECEIPT_NAME = "evaluation_receipt.json"
REFIT_RECEIPT_NAME = "refit_receipt.json"
EVENTS_NAME = "events.jsonl"
PROGRESS_NAME = "progress.jsonl"
MANIFEST_NAME = "manifest.json"
LEASE_NAME = "lease.json"
PRIOR_FIELD = "scientific_seconds_cumulative_bound"
MINIMUM_BUDGET_HEADROOM_BYTES = 8 * 1024 * 1024
MINIMUM_FREE_CUDA_BYTES = 5368709120
MAXIMUM_STORAGE_BYTES = 107374182400


class P05ComprehensiveEvaluationError(core.P05CoreError):
    """Stable, path-free comprehensive-evaluation failure."""


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05ComprehensiveEvaluationError(code)


def _finite_seconds(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _import_runtime() -> dict[str, Any]:
    core._configure_environment()
    import torch

    from atlas_sers.evaluation import p05_evaluation_authority as authority
    from atlas_sers.evaluation import p05_evaluation_close as close
    from atlas_sers.evaluation import p05_outer_inputs as outer_inputs
    from atlas_sers.evaluation import p05_pilot as pilot
    from atlas_sers.evaluation import p05_prediction as prediction
    from atlas_sers.evaluation import p05_prediction_io as prediction_io

    return {
        "torch": torch,
        "authority": authority,
        "close": close,
        "outer_inputs": outer_inputs,
        "pilot": pilot,
        "prediction": prediction,
        "prediction_io": prediction_io,
    }


def _configure_cuda(torch: Any, pilot: Any, device: str) -> None:
    torch.set_num_threads(1)
    _require(bool(torch.cuda.is_available()), "cuda_unavailable")
    _require(
        pilot._free_cuda_bytes(torch) >= MINIMUM_FREE_CUDA_BYTES, "cuda_free_bytes_insufficient"
    )
    pilot._enforce_cuda_cap(torch, device)


def _event(phase: str, context_id: str, refit_id: str) -> dict[str, Any]:
    return {
        "command": COMMAND,
        "stage": STAGE_NAME,
        "phase": phase,
        "context_id": context_id,
        "refit_id": refit_id,
    }


def _progress(counters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "started": int(counters["started"]),
        "completed": int(counters["completed"]),
        "failed": int(counters["failed"]),
        "rows": int(counters["rows"]),
        "contexts_completed": int(counters["contexts_completed"]),
        "peak_cuda_bytes": int(counters["peak_cuda_bytes"]),
    }


def run_evaluation(
    *,
    project_root: Any,
    artifact_root: Any,
    contract_path: Any,
    permit_path: Any,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the comprehensive evaluation stage and return its receipt."""

    wall_start = time.perf_counter()
    _require(device == "cuda", "device_not_cuda")
    core._configure_environment()
    bundle = inputs.prepare(
        project_root, artifact_root, contract_path, permit_path, require_unstarted=False
    )
    _require(isinstance(bundle, Mapping), "bundle_malformed")

    runtime = _import_runtime()
    torch = runtime["torch"]
    authority = runtime["authority"]
    close = runtime["close"]
    outer_inputs = runtime["outer_inputs"]
    pilot = runtime["pilot"]
    prediction = runtime["prediction"]
    prediction_io = runtime["prediction_io"]

    permit_id = str(bundle["permit_sha256"])
    artifact_root = Path(bundle["artifact_root"])
    run_root = artifact_root / COMPREHENSIVE_DIR / RUNS_DIR / permit_id
    refit_receipt = freeze._read_mapping(run_root / REFIT_RECEIPT_NAME, "refit_receipt")
    prior = refit_receipt.get(PRIOR_FIELD)
    _require(_finite_seconds(prior), "refit_prior_invalid")
    prior = float(prior)
    _require(prior >= development.PRELAUNCH_AUDIT_RESERVE_SECONDS, "refit_prior_below_reserve")
    _require(prior <= development.MAXIMUM_TOTAL_SECONDS, "refit_prior_above_maximum")
    deadline = wall_start + development.MAXIMUM_TOTAL_SECONDS - prior
    freeze._check_deadline(deadline)

    auth = authority.authenticate_refits(bundle, deadline=deadline)
    _require(isinstance(auth, Mapping), "authority_result_malformed")
    plan = auth["plan"]
    _require(_finite_seconds(auth["prior_seconds"]), "authority_prior_invalid")
    _require(float(auth["prior_seconds"]) == prior, "authority_prior_mismatch")
    _require(dict(auth["refit_receipt"]) == dict(refit_receipt), "refit_receipt_changed")
    _require(type(auth["source_optimizer_steps"]) is int, "authority_source_steps_invalid")
    _require(type(auth["refit_optimizer_steps"]) is int, "authority_refit_steps_invalid")

    _configure_cuda(torch, pilot, device)
    provenance_before = core._capture_provenance(
        bundle["repository_root"], bundle["project_root"], bundle["artifact_root"]
    )

    index = prediction_io.index_endpoints(plan)
    full_contexts = outer_inputs.load_context_rows(bundle)
    _require(len(full_contexts) == prediction_io.EXPECTED_CONTEXTS, "context_count_mismatch")
    _require(
        set(index) == {str(row["context_id"]) for row in full_contexts}, "context_set_mismatch"
    )

    selection_plan_id = str(plan["plan_id"])
    specs = plan["unique_refits"]
    identity = {
        "schema_version": SCHEMA,
        "protocol_version": PROTOCOL,
        "command": COMMAND,
        "stage": STAGE_NAME,
        "permit_sha256": permit_id,
        "core_contract_sha256": bundle["contract_sha256"],
        "core_plan_id": bundle["core_plan_id"],
        "ledger_id": bundle["ledger"]["ledger_id"],
        "selection_plan_id": selection_plan_id,
        "refit_receipt_sha256": core._canon().sha256_file(run_root / REFIT_RECEIPT_NAME),
        "unique_prediction_count": len(specs),
        "strategy_alias_count": len(plan["strategy_aliases"]),
        "context_count": len(full_contexts),
        "fits_started": 0,
        "calibrations_started": 0,
        "optimizer_steps": 0,
        "predictions_frozen": False,
        "source_optimizer_steps": auth["source_optimizer_steps"],
        "refit_optimizer_steps": auth["refit_optimizer_steps"],
    }
    counters: dict[str, Any] = {
        "started": 0,
        "completed": 0,
        "failed": 0,
        "rows": 0,
        "contexts_completed": 0,
        "peak_cuda_bytes": 0,
        "sum_prediction_elapsed_seconds": 0.0,
        "optimizer_steps": 0,
        "elapsed_seconds": 0.0,
    }

    stage = run_root / STAGE_NAME
    units_dir = stage / "units"
    events_path = stage / EVENTS_NAME
    progress_path = stage / PROGRESS_NAME
    consumed = False
    try:
        freeze._reject_preexisting(stage, "evaluation_stage_exists")
        freeze._reject_preexisting(run_root / RECEIPT_NAME, "evaluation_receipt_exists")
        freeze._check_deadline(deadline)
        core._mkdir_exclusive(stage, "evaluation_stage_exists")
        consumed = True
        budget = freeze.StorageBudget(artifact_root, run_root, ceiling=MAXIMUM_STORAGE_BYTES)
        core._mkdir_exclusive(units_dir, "evaluation_units_exists")
        budget.register_growing(events_path)
        budget.register_growing(progress_path)
        freeze._budgeted_write(stage / "provenance_before.json", provenance_before, budget)

        completed_ids: set[str] = set()
        for context_id in sorted(index):
            freeze._check_deadline(deadline)
            entry = index[context_id]
            endpoint = entry["endpoint"]
            prepared = outer_inputs.prepare_outer_inputs(bundle, endpoint)
            for spec in entry["specs"]:
                freeze._check_deadline(deadline)
                budget.check(headroom_bytes=MINIMUM_BUDGET_HEADROOM_BYTES)
                refit_id = str(spec["refit_id"])
                _require(refit_id not in completed_ids, "evaluation_duplicate_refit")
                _require(
                    tuple(prepared["classes"]) == tuple(spec["classes"]),
                    "prediction_source_classes_mismatch",
                )
                unit_dir = units_dir / refit_id
                budget.activate_unit(unit_dir)
                try:
                    core._atomic_write(
                        unit_dir / LEASE_NAME,
                        core._canon().canonical_json_bytes(
                            {
                                "selection_plan_id": selection_plan_id,
                                "spec": spec,
                                "endpoint": endpoint,
                            }
                        ),
                    )
                    budget.check(headroom_bytes=MINIMUM_BUDGET_HEADROOM_BYTES)
                    freeze._check_deadline(deadline)
                    counters["started"] += 1
                    development._append_jsonl(
                        events_path, _event("predict_start", context_id, refit_id)
                    )
                    frame, audit = prediction.predict_refit(
                        spec=spec,
                        unit_dir=run_root / "refits" / "units" / refit_id,
                        values=prepared["values"],
                        observation_uids=prepared["observation_uids"],
                        device=device,
                        deadline=deadline,
                    )
                    budget.check(headroom_bytes=MINIMUM_BUDGET_HEADROOM_BYTES)
                    freeze._check_deadline(deadline)
                    result = prediction_io.persist_prediction(
                        unit_dir, spec, endpoint, frame, audit
                    )
                    counters["rows"] += int(result["row_count"])
                    counters["peak_cuda_bytes"] = max(
                        int(counters["peak_cuda_bytes"]), int(result["peak_cuda_bytes"])
                    )
                    counters["sum_prediction_elapsed_seconds"] += float(result["elapsed_seconds"])
                    core._write_manifest(unit_dir)
                    pilot._verify_manifest(unit_dir)
                except BaseException:
                    try:
                        budget.close_unit()
                    except BaseException:
                        pass
                    raise
                budget.close_unit()
                counters["completed"] += 1
                completed_ids.add(refit_id)
                development._append_jsonl(
                    events_path, _event("predict_complete", context_id, refit_id)
                )
                development._append_jsonl(progress_path, _progress(counters))
                pilot._enforce_cuda_cap(torch, device)
                freeze._check_deadline(deadline)
                del frame, audit, result
            counters["contexts_completed"] += 1
            del prepared

        expected_ids = {str(refit_id) for refit_id in specs}
        expected_rows = sum(
            len(index[str(spec["context_id"])]["endpoint"]["test_uids"]) for spec in specs.values()
        )
        _require(completed_ids == expected_ids, "evaluation_refit_coverage_mismatch")
        _require(counters["started"] == len(expected_ids), "evaluation_started_count_mismatch")
        _require(counters["completed"] == len(expected_ids), "evaluation_completed_count_mismatch")
        _require(counters["failed"] == 0, "evaluation_failed_count_nonzero")
        _require(counters["contexts_completed"] == len(index), "evaluation_context_count_mismatch")
        _require(counters["optimizer_steps"] == 0, "evaluation_optimizer_steps_nonzero")
        _require(
            counters["peak_cuda_bytes"] <= prediction_io.MAX_PEAK_CUDA_BYTES,
            "evaluation_peak_cuda_exceeded",
        )
        _require(counters["rows"] == expected_rows, "evaluation_row_count_mismatch")

        return close.finish_stage(
            bundle=bundle,
            stage=stage,
            run_root=run_root,
            budget=budget,
            identity={
                **identity,
                "predictions_frozen": True,
                "predictions_complete": True,
                "all_complete": True,
            },
            counters=counters,
            prior=prior,
            wall_start=wall_start,
            deadline=deadline,
            provenance_before=provenance_before,
            contract_path=contract_path,
            permit_path=permit_path,
        )
    except BaseException as error:
        if consumed:
            close.record_failure(
                stage=stage,
                identity=identity,
                counters=counters,
                prior=prior,
                wall_start=wall_start,
                error=error,
            )
        raise
