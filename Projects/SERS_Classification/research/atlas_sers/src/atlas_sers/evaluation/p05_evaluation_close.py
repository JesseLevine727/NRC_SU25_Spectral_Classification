"""Persistence-only close-out helpers for frozen P05 evaluation stages."""

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
from atlas_sers.evaluation import p05_pilot as pilot

__all__ = ["finish_stage", "record_failure"]

RECEIPT_NAME = "evaluation_receipt.json"
MINIMUM_BUDGET_HEADROOM_BYTES = 8 * 1024 * 1024


def _failure_counters(counters: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(counters)
    started = int(updated.get("started", 0))
    completed = int(updated.get("completed", 0))
    failed = int(updated.get("failed", 0))
    updated["failed"] = max(failed, started - completed)
    updated.setdefault("optimizer_steps", 0)
    return updated


def record_failure(
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
        "predictions_frozen": False,
        "reason_code": getattr(error, "reason_code", type(error).__name__),
        "counters": {**_failure_counters(counters), "elapsed_seconds": elapsed},
        "predictions_complete": False,
        "all_complete": False,
        "prior_scientific_seconds_cumulative_bound": prior,
        "scientific_seconds_this_stage": elapsed,
        "scientific_seconds_cumulative_bound": prior + elapsed,
    }
    try:
        core._atomic_write(stage / "summary.json", core._canon().canonical_json_bytes(payload))
        core._write_manifest(stage)
    except BaseException:
        pass


def _bound_payload(
    identity: Mapping[str, Any],
    counters: Mapping[str, Any],
    prior: float,
    elapsed: float,
) -> dict[str, Any]:
    if (
        not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in (prior, elapsed)
        )
        or prior < development.PRELAUNCH_AUDIT_RESERVE_SECONDS
        or elapsed < 0
        or prior + elapsed > development.MAXIMUM_TOTAL_SECONDS
    ):
        raise core.P05CoreError("evaluation_cumulative_time_invalid")
    return {
        **identity,
        "status": "complete",
        "counters": {**counters, "elapsed_seconds": elapsed},
        "scientific_seconds_this_stage": elapsed,
        "prior_scientific_seconds_cumulative_bound": prior,
        "scientific_seconds_cumulative_bound": prior + elapsed,
        "prelaunch_audit_reserve_seconds": development.PRELAUNCH_AUDIT_RESERVE_SECONDS,
        "maximum_total_seconds": development.MAXIMUM_TOTAL_SECONDS,
    }


def finish_stage(
    *,
    bundle: Mapping[str, Any],
    stage: Path,
    run_root: Path,
    budget: Any,
    identity: Mapping[str, Any],
    counters: Mapping[str, Any],
    prior: float,
    wall_start: float,
    deadline: float,
    provenance_before: Mapping[str, Any],
    contract_path: Any,
    permit_path: Any,
) -> dict[str, Any]:
    freeze._check_deadline(deadline)
    freeze._reject_preexisting(run_root / RECEIPT_NAME, "evaluation_receipt_exists")
    provenance_after = pilot._post_run_reauth(
        bundle["artifact_root"],
        bundle["contract"],
        bundle["support"],
        provenance_before,
        bundle["repository_root"],
        bundle["project_root"],
    )
    freeze._budgeted_write(stage / "provenance_after.json", provenance_after, budget)
    inputs.prepare(
        bundle["project_root"],
        bundle["artifact_root"],
        contract_path,
        permit_path,
        require_unstarted=False,
    )
    freeze._check_deadline(deadline)
    elapsed = time.perf_counter() - wall_start
    freeze._budgeted_write(
        stage / "summary.json", _bound_payload(identity, counters, prior, elapsed), budget
    )
    budget.check(headroom_bytes=MINIMUM_BUDGET_HEADROOM_BYTES)
    core._write_manifest(stage)
    budget.account_new_file(stage / "manifest.json")
    pilot._verify_manifest(stage)
    budget.reconcile()
    freeze._check_deadline(deadline)
    final_elapsed = time.perf_counter() - wall_start
    receipt = {
        **_bound_payload(identity, counters, prior, final_elapsed),
        "stage_manifest_sha256": core._canon().sha256_file(stage / "manifest.json"),
    }
    freeze._reject_preexisting(run_root / RECEIPT_NAME, "evaluation_receipt_exists")
    freeze._budgeted_write(run_root / RECEIPT_NAME, receipt, budget)
    budget.check()
    freeze._check_deadline(deadline)
    return receipt
