"""Comprehensive P05 refit runner.

Executes the frozen, already-authorized comprehensive refit plan: one fixed
epoch kernel per unique refit plus one scalar calibration, durable per-unit
evidence and one authoritative run-root receipt.  Produces no held
prediction, benchmark metric or superiority claim.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_recovery_source as source
from atlas_sers.evaluation import p05_refit_authority as authority
from atlas_sers.evaluation import p05_refit_evidence as evidence

__all__ = ["P05ComprehensiveRefitError", "run_refits"]

SCHEMA = "nato-sers-p05-comprehensive-refits-v1"
STAGE_NAME = "refits"
RECEIPT_NAME = "refit_receipt.json"
COMMAND = "run_comprehensive_refits"
CLAIM = "source_refits_and_calibration_no_held_predictions"
SOURCE_FIT_COUNT = 14904
REUSED_PILOT_FIT_COUNT = 36
MAXIMUM_REFITS = 2880
MAXIMUM_UPDATES_PER_REFIT = 800
SOURCE_MAXIMUM_UPDATES = SOURCE_FIT_COUNT * MAXIMUM_UPDATES_PER_REFIT
MAXIMUM_COMBINED_UPDATES = 14227200
MAXIMUM_FIT_SECONDS = 120.0
MAXIMUM_STORAGE_BYTES = 107374182400
MINIMUM_BUDGET_HEADROOM_BYTES = 8 * 1024 * 1024
MAXIMUM_CUDA_ALLOCATED_BYTES = 4294967296
MINIMUM_FREE_CUDA_BYTES = 5368709120


class P05ComprehensiveRefitError(core.P05CoreError):
    """Stable, path-free refit-stage failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05ComprehensiveRefitError(code)


def _finite(value: Any, code: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), code)
    number = float(value)
    _require(math.isfinite(number), code)
    return number


def _integer(value: Any, code: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), code)
    return int(value)


def _run_root(artifact_root: Any, permit_sha256: Any) -> Path:
    return Path(artifact_root) / "p05comprehensive" / "runs" / str(permit_sha256)


def _recovery_resources(torch: Any, phase: str) -> None:
    from atlas_sers.evaluation import p05_recovery_authority as recovery_authority

    recovery_authority.check_resources(torch, phase=phase)


def _new_counters() -> dict[str, Any]:
    return {
        "calibration_started": 0,
        "calibration_completed": 0,
        "calibration_failed": 0,
        "neural_started": 0,
        "neural_completed": 0,
        "neural_failed": 0,
        "optimizer_steps": 0,
        "optimizer_steps_exact": True,
        "peak_cuda_bytes": 0,
        "elapsed_seconds": 0.0,
        "sum_fit_elapsed_seconds": 0.0,
    }


def _mark_in_flight(counters: dict[str, Any]) -> None:
    for prefix in ("calibration", "neural"):
        pending = (
            int(counters[f"{prefix}_started"])
            - int(counters[f"{prefix}_completed"])
            - int(counters[f"{prefix}_failed"])
        )
        if pending > 0:
            counters[f"{prefix}_failed"] = int(counters[f"{prefix}_failed"]) + pending


def _require_headroom(budget: Any) -> None:
    budget.check(headroom_bytes=MINIMUM_BUDGET_HEADROOM_BYTES)


def _register_growing(budget: Any, path: Path) -> None:
    budget.register_growing(path)


def _static_write(budget: Any, path: Path, payload: Any) -> None:
    freeze._budgeted_write(path, payload, budget)


def _emit(stage: Path, payload: Mapping[str, Any]) -> None:
    development._append_jsonl(stage / "events.jsonl", dict(payload))


def _record_failure(
    stage: Path,
    counters: dict[str, Any],
    error: BaseException,
    identity: dict[str, Any],
    prior: float,
    elapsed: float,
) -> None:
    payload = {
        **identity,
        "schema_version": SCHEMA,
        "protocol_version": development.PROTOCOL_VERSION,
        "stage": STAGE_NAME,
        "command": COMMAND,
        "status": "fail",
        "reason_code": getattr(error, "reason_code", type(error).__name__),
        "counters": dict(counters),
        "refits_complete": False,
        "calibrations_complete": False,
        "outer_predictions_started": 0,
        "prior_scientific_seconds_cumulative_bound": prior,
        "scientific_seconds_this_stage": elapsed,
        "scientific_seconds_cumulative_bound": prior + elapsed,
    }
    try:
        core._atomic_write(stage / "summary.json", core._canon().canonical_json_bytes(payload))
        core._write_manifest(stage)
    except BaseException:
        pass


def run_refits(
    *,
    project_root: Any,
    artifact_root: Any,
    contract_path: Any,
    permit_path: Any,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the authorized comprehensive refits and calibrations once."""

    wall_start = time.perf_counter()
    return _run_refits(
        project_root=project_root,
        artifact_root=artifact_root,
        contract_path=contract_path,
        permit_path=permit_path,
        device=device,
        wall_start=wall_start,
    )


def _run_refits(
    *,
    project_root: Any,
    artifact_root: Any,
    contract_path: Any,
    permit_path: Any,
    device: str,
    wall_start: float,
) -> dict[str, Any]:
    counters = _new_counters()
    stage: Path | None = None
    consumed = False
    prior = 0.0
    identity: dict[str, Any] = {}
    try:
        _require(device == "cuda", "device_invalid")
        core._configure_environment()
        bundle = inputs.prepare(
            project_root, artifact_root, contract_path, permit_path, require_unstarted=False
        )
        _require(isinstance(bundle, Mapping), "bundle_malformed")
        support = bundle["support"]
        ledger = bundle["ledger"]
        contract = bundle["contract"]
        paths = freeze._paths(bundle)
        repository_root = bundle["repository_root"]
        project_root = bundle["project_root"]
        artifact_root = bundle["artifact_root"]

        selection_receipt = freeze._read_mapping(
            paths["selection_receipt"], "selection_receipt_missing"
        )
        prior = _finite(
            selection_receipt.get("scientific_seconds_cumulative_bound"),
            "selection_receipt_cumulative_malformed",
        )
        _require(
            development.PRELAUNCH_AUDIT_RESERVE_SECONDS
            <= prior
            <= development.MAXIMUM_TOTAL_SECONDS,
            "selection_prior_out_of_range",
        )
        deadline = wall_start + float(development.MAXIMUM_TOTAL_SECONDS) - prior
        freeze._check_deadline(deadline)

        authenticated = authority.authenticate_selection(bundle, deadline=deadline)
        _require(
            _finite(authenticated.get("prior_seconds"), "authority_prior_malformed") == prior,
            "authority_prior_mismatch",
        )
        plan = authenticated["plan"]
        source_optimizer_steps = _integer(
            authenticated.get("source_optimizer_steps"), "source_optimizer_steps_malformed"
        )
        try:
            source_accounting = source.from_authenticated(authenticated)
        except source.RecoverySourceError as error:
            raise P05ComprehensiveRefitError("source_accounting_rejected") from error
        recovered = source_accounting.get("mode") == source.RECOVERY_ACCOUNTING_MODE
        unique_refits = plan["unique_refits"]
        aliases = plan["strategy_aliases"]
        plan_id = str(plan["plan_id"])

        _require(isinstance(unique_refits, Mapping), "plan_unique_refits_malformed")
        _require(
            isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)),
            "plan_aliases_malformed",
        )
        unique_count = len(unique_refits)
        alias_count = len(aliases)
        _require(0 < unique_count <= MAXIMUM_REFITS, "unique_refit_count_out_of_range")
        _require(alias_count == freeze.STRATEGY_ALIAS_COUNT, "strategy_alias_count_mismatch")
        _require(alias_count == MAXIMUM_REFITS, "strategy_alias_count_not_exact")
        for alias in aliases:
            _require(isinstance(alias, Mapping), "alias_malformed")
            _require(str(alias["refit_id"]) in unique_refits, "alias_refit_unknown")
        if recovered:
            maximum_source_optimizer_steps = _integer(
                source_accounting["maximum_source_optimizer_steps"], "source_maximum_malformed"
            )
            source_charged_upper_bound = _integer(
                source_accounting["source_optimizer_steps_charged_upper_bound"],
                "source_charged_upper_malformed",
            )
            maximum_combined_updates = _integer(
                source_accounting["maximum_new_optimizer_steps"], "source_maximum_malformed"
            )
            maximum_new_neural_executions = _integer(
                source_accounting["maximum_new_neural_executions"], "source_maximum_malformed"
            )
            source_attempts = _integer(
                source_accounting["source_attempts"], "source_attempts_malformed"
            )
            _require(
                0 <= source_optimizer_steps <= maximum_source_optimizer_steps,
                "source_updates_exceeded",
            )
            _require(
                source_charged_upper_bound <= maximum_source_optimizer_steps,
                "source_charged_upper_exceeded",
            )
        else:
            _require(
                0 <= source_optimizer_steps <= SOURCE_MAXIMUM_UPDATES, "source_updates_exceeded"
            )
            source_charged_upper_bound = source_optimizer_steps
            maximum_source_optimizer_steps = SOURCE_MAXIMUM_UPDATES
            maximum_combined_updates = MAXIMUM_COMBINED_UPDATES
            source_attempts = SOURCE_FIT_COUNT
            maximum_new_neural_executions = SOURCE_FIT_COUNT + MAXIMUM_REFITS

        identity = {
            "schema_version": SCHEMA,
            "protocol_version": development.PROTOCOL_VERSION,
            "command": COMMAND,
            "stage": STAGE_NAME,
            "claim": CLAIM,
            "permit_sha256": bundle["permit_sha256"],
            "core_contract_sha256": bundle["contract_sha256"],
            "core_plan_id": bundle["core_plan_id"],
            "ledger_id": ledger["ledger_id"],
            "selection_plan_id": plan_id,
            "source_fit_count": SOURCE_FIT_COUNT,
            "reused_pilot_fit_count": REUSED_PILOT_FIT_COUNT,
            "unique_refit_count": unique_count,
            "strategy_alias_count": alias_count,
            "source_optimizer_steps": source_optimizer_steps,
            "outer_predictions_started": 0,
        }
        if recovered:
            identity["source_execution_accounting"] = dict(source_accounting)

        import torch

        from atlas_sers.evaluation import p05_calibration as calibration
        from atlas_sers.evaluation import p05_recovery_unit as recovery_unit
        from atlas_sers.evaluation import p05_refit as refit
        from atlas_sers.evaluation import p05_refit_io as refit_io

        torch.set_num_threads(1)
        _require(bool(torch.cuda.is_available()), "cuda_unavailable")
        _require(
            pilot._free_cuda_bytes(torch) >= MINIMUM_FREE_CUDA_BYTES, "insufficient_free_cuda_bytes"
        )
        pilot._enforce_cuda_cap(torch, device)
        if recovered:
            _recovery_resources(torch, "launch")
        pilot._checkpoint_preflight(torch, artifact_root)
        freeze._check_deadline(deadline)

        run_root = _run_root(artifact_root, bundle["permit_sha256"])
        stage = run_root / STAGE_NAME
        freeze._reject_preexisting(stage, "refits_stage_exists")
        freeze._reject_preexisting(run_root / RECEIPT_NAME, "refit_receipt_exists")
        core._mkdir_exclusive(stage, "refits_stage_exists")
        consumed = True
        budget = freeze.StorageBudget(artifact_root, run_root, ceiling=MAXIMUM_STORAGE_BYTES)
        core._mkdir_exclusive(stage / "units", "units_dir_exists")
        _register_growing(budget, stage / "events.jsonl")
        _register_growing(budget, stage / "progress.jsonl")

        provenance_before = core._capture_provenance(repository_root, project_root, artifact_root)
        _static_write(budget, stage / "provenance_before.json", provenance_before)

        load_logits = evidence.make_logit_loader(bundle)
        groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        completed_refits: set[str] = set()
        ordered = sorted(
            (dict(spec) for spec in unique_refits.values()),
            key=lambda spec: (
                str(spec["context_id"]),
                int(spec["seed"]),
                str(spec["recipe_id"]),
                str(spec["refit_id"]),
            ),
        )
        expected_sizes = {
            key: len(list(items))
            for key, items in itertools.groupby(
                ordered, key=lambda spec: (str(spec["context_id"]), int(spec["seed"]))
            )
        }

        for spec in ordered:
            freeze._check_deadline(deadline)
            if recovered:
                _recovery_resources(torch, "fit")
            refit_id = str(spec["refit_id"])
            unit_dir = stage / "units" / refit_id
            _require_headroom(budget)
            budget.activate_unit(unit_dir)
            try:
                core._atomic_write(
                    unit_dir / "lease.json",
                    core._canon().canonical_json_bytes(
                        {
                            "selection_plan_id": plan_id,
                            "spec": spec,
                        }
                    ),
                )
                _emit(stage, {"event": "refit_started", "refit_id": refit_id})

                prepared = refit_io.prepare_refit_inputs(bundle, spec)
                _require_headroom(budget)
                freeze._check_deadline(deadline)
                if recovered:
                    _recovery_resources(torch, "fit")
                _require(
                    counters["calibration_started"] < MAXIMUM_REFITS, "calibration_ceiling_exceeded"
                )
                counters["calibration_started"] += 1
                _emit(stage, {"event": "calibration_started", "refit_id": refit_id})
                try:
                    calibrated, audit = calibration.calibrate_spec(
                        spec=spec,
                        ledger=ledger,
                        manifest=support.manifest,
                        load_logits=load_logits,
                    )
                    evidence.persist_calibration(unit_dir, calibrated, audit, spec)
                except BaseException:
                    counters["calibration_failed"] += 1
                    raise
                counters["calibration_completed"] += 1
                _emit(stage, {"event": "calibration_completed", "refit_id": refit_id})

                _require_headroom(budget)
                freeze._check_deadline(deadline)
                _require(counters["neural_started"] < MAXIMUM_REFITS, "refit_ceiling_exceeded")
                _require(
                    source_attempts + counters["neural_started"] + 1
                    <= maximum_new_neural_executions,
                    "source_attempts_exceeded",
                )
                expected_steps = _integer(spec["epochs"], "spec_epochs_malformed") * 4
                _require(
                    120 <= expected_steps <= MAXIMUM_UPDATES_PER_REFIT, "spec_epoch_budget_invalid"
                )
                _require(
                    source_charged_upper_bound + counters["optimizer_steps"] + expected_steps
                    <= maximum_combined_updates,
                    "combined_updates_exceeded",
                )
                recorder = pilot.open_history_recorder(unit_dir, refit_id)
                if recovered:
                    guarded = recovery_unit._GuardedEpochRecorder(recorder, budget, deadline, torch)
                else:
                    guarded = development._GuardedRecorder(recorder, budget, deadline)
                steps_accounted = False
                try:
                    counters["neural_started"] += 1
                    _emit(stage, {"event": "neural_started", "refit_id": refit_id})
                    result = refit.train_refit(
                        **prepared,
                        device=device,
                        maximum_fit_seconds=MAXIMUM_FIT_SECONDS,
                        global_deadline=deadline,
                        maximum_cuda_allocated_bytes=MAXIMUM_CUDA_ALLOCATED_BYTES,
                        on_epoch=guarded,
                    )
                    steps = _integer(result.optimizer_steps, "result_optimizer_steps_malformed")
                    _require(steps >= 0, "result_optimizer_steps_malformed")
                    counters["optimizer_steps"] += steps
                    steps_accounted = True
                finally:
                    if not steps_accounted:
                        lower = development._lower_bound_updates(unit_dir, refit_id)
                        counters["optimizer_steps"] += _integer(lower, "lower_bound_malformed")
                        counters["optimizer_steps_exact"] = False
                    guarded.close()

                peak = _integer(result.peak_cuda_bytes, "result_peak_malformed")
                counters["peak_cuda_bytes"] = max(counters["peak_cuda_bytes"], peak)

                _require_headroom(budget)
                refit_io.persist_refit_result(torch, unit_dir, spec, result)
                fit_seconds = _finite(result.elapsed_seconds, "result_seconds_malformed")
                counters["sum_fit_elapsed_seconds"] += fit_seconds
                _require(
                    0 <= fit_seconds <= MAXIMUM_FIT_SECONDS,
                    "fit_seconds_exceeded",
                )
                _require(0 <= peak <= MAXIMUM_CUDA_ALLOCATED_BYTES, "peak_cuda_exceeded")
                _require(steps <= expected_steps, "refit_updates_exceeded")
                _require(
                    source_charged_upper_bound + counters["optimizer_steps"]
                    <= maximum_combined_updates,
                    "combined_updates_exceeded",
                )
                refit_io.check_completed_refit(torch, unit_dir, spec, result)
                counters["neural_completed"] += 1
                _emit(stage, {"event": "neural_completed", "refit_id": refit_id})

                summary = evidence.summarize_result(spec, result)
                key = (str(spec["context_id"]), int(spec["seed"]))
                bucket = groups.setdefault(key, [])
                bucket.append(summary)
                if len(bucket) == expected_sizes[key]:
                    pairs = evidence.cross_instrument_master_count(support, spec)
                    evidence.check_recipe_group(bucket, cross_instrument_pairs=pairs)
                    del groups[key]
                del result, prepared, calibrated, audit

                core._write_manifest(unit_dir)
                pilot._verify_manifest(unit_dir)
                development._append_jsonl(
                    stage / "progress.jsonl",
                    {"refit_id": refit_id, "status": "complete", "counters": dict(counters)},
                )
                completed_refits.add(refit_id)
            except BaseException:
                try:
                    budget.close_unit()
                except Exception:
                    pass
                raise
            else:
                budget.close_unit()
            pilot._enforce_cuda_cap(torch, device)
            if recovered:
                _recovery_resources(torch, "fit")
            freeze._check_deadline(deadline)

        _require(completed_refits == set(unique_refits), "unique_coverage_incomplete")
        if recovered:
            _recovery_resources(torch, "fit")
        _require(
            source_attempts + counters["neural_started"] <= maximum_new_neural_executions,
            "source_attempts_exceeded",
        )
        _require(counters["calibration_started"] == unique_count, "calibration_started_mismatch")
        _require(
            counters["calibration_completed"] == unique_count, "calibration_completed_mismatch"
        )
        _require(counters["calibration_failed"] == 0, "calibration_failures_present")
        _require(counters["neural_started"] == unique_count, "neural_started_mismatch")
        _require(counters["neural_completed"] == unique_count, "neural_completed_mismatch")
        _require(counters["neural_failed"] == 0, "neural_failures_present")
        _require(counters["optimizer_steps_exact"] is True, "optimizer_steps_inexact")
        _require(
            counters["optimizer_steps"] == sum(int(spec["epochs"]) * 4 for spec in ordered),
            "exact_optimizer_steps_mismatch",
        )
        _require(
            counters["optimizer_steps"] <= unique_count * MAXIMUM_UPDATES_PER_REFIT,
            "refit_updates_exceeded",
        )
        _require(
            source_charged_upper_bound + counters["optimizer_steps"] <= maximum_combined_updates,
            "combined_updates_exceeded",
        )
        _require(not groups, "recipe_groups_incomplete")
        freeze._check_deadline(deadline)

        identity.update(
            {
                "refits_complete": True,
                "calibrations_complete": True,
                "total_new_optimizer_steps": source_optimizer_steps + counters["optimizer_steps"],
            }
        )
        if recovered:
            identity["total_new_optimizer_steps_charged_upper_bound"] = (
                source_charged_upper_bound + counters["optimizer_steps"]
            )
            identity["total_new_optimizer_steps_all_attempts_exact"] = False

        provenance_after = pilot._post_run_reauth(
            artifact_root,
            contract,
            support,
            provenance_before,
            repository_root,
            project_root,
        )
        _static_write(budget, stage / "provenance_after.json", provenance_after)
        inputs.prepare(
            project_root, artifact_root, contract_path, permit_path, require_unstarted=False
        )

        summary_elapsed = time.perf_counter() - wall_start
        freeze._check_deadline(deadline)
        counters["elapsed_seconds"] = summary_elapsed
        summary_payload = {
            **identity,
            "status": "complete",
            "counters": dict(counters),
            "scientific_seconds_this_stage": summary_elapsed,
            "prior_scientific_seconds_cumulative_bound": prior,
            "scientific_seconds_cumulative_bound": prior + summary_elapsed,
            "prelaunch_audit_reserve_seconds": development.PRELAUNCH_AUDIT_RESERVE_SECONDS,
            "maximum_total_seconds": development.MAXIMUM_TOTAL_SECONDS,
        }
        _static_write(budget, stage / "summary.json", summary_payload)
        _require_headroom(budget)
        core._write_manifest(stage)
        budget.account_new_file(stage / "manifest.json")
        pilot._verify_manifest(stage)
        budget.reconcile()
        freeze._check_deadline(deadline)

        elapsed = time.perf_counter() - wall_start
        counters["elapsed_seconds"] = elapsed
        receipt = {
            **identity,
            "status": "complete",
            "counters": dict(counters),
            "stage_manifest_sha256": core._canon().sha256_file(stage / "manifest.json"),
            "scientific_seconds_this_stage": elapsed,
            "prior_scientific_seconds_cumulative_bound": prior,
            "scientific_seconds_cumulative_bound": prior + elapsed,
            "prelaunch_audit_reserve_seconds": development.PRELAUNCH_AUDIT_RESERVE_SECONDS,
            "maximum_total_seconds": development.MAXIMUM_TOTAL_SECONDS,
        }
        _static_write(budget, run_root / RECEIPT_NAME, receipt)
        budget.check()
        freeze._check_deadline(deadline)
        return receipt
    except BaseException as error:
        _mark_in_flight(counters)
        elapsed = time.perf_counter() - wall_start
        counters["elapsed_seconds"] = elapsed
        if consumed and stage is not None:
            _record_failure(stage, counters, error, identity, prior, elapsed)
        raise
