"""P05 comprehensive source-fit development boundary.

``run_development`` authenticates the owner-approved comprehensive permit
through the read-only :mod:`p05_comprehensive_inputs` boundary, re-imports the
36 completed pilot records and executes exactly the remaining 14,904 canonical
source-development fits (1,245 ledger units x 12 slots minus the 36 reused pilot
slots) serially on CUDA.  Only source fits run here: no selection, refit,
calibration or outer-test evaluation.  All artifacts are written under an
exclusive consumed run root (``p05comprehensive/runs/<permit>``) with explicit
storage accounting; the 3 pilot units are skipped as fully reused.  Any failure
stops the run, preserves the partial state/history/lease evidence and reports an
honest elapsed time; there is no resume and no automatic retry.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation.p05_comprehensive_storage import (
    P05StorageError,
    StorageBudget,
)
from atlas_sers.evaluation.p05_core_run import P05CoreError

__all__ = [
    "COMPREHENSIVE_CLAIM",
    "INNER_SLOT_COUNT",
    "MAXIMUM_NEW_FITS",
    "MAXIMUM_TOTAL_SECONDS",
    "NAMESPACE",
    "P05ComprehensiveDevelopmentError",
    "PRELAUNCH_AUDIT_RESERVE_SECONDS",
    "REUSED_PILOT_SLOTS",
    "SLOTS_PER_UNIT",
    "UNIT_COUNT",
    "run_development",
]

NAMESPACE = "p05comprehensive"
SCHEMA_VERSION = "nato-sers-p05-comprehensive-development-v1"
PROTOCOL_VERSION = "nato-sers-p05-comprehensive-20260926-v1"
COMPREHENSIVE_CLAIM = "complete_source_development_fits_no_selection_no_refit_no_outer_evaluation"

STORAGE_CEILING_BYTES = 107374182400
MAXIMUM_NEW_FITS = 14904
INNER_SLOT_COUNT = 14940
REUSED_PILOT_SLOTS = 36
UNIT_COUNT = 1245
SLOTS_PER_UNIT = 12
MAXIMUM_TOTAL_SECONDS = 172800.0
UNIT_BUDGET_HEADROOM_BYTES = 8 * 1024 * 1024
MAXIMUM_FIT_STEPS = MAXIMUM_NEW_FITS * pilot.BATCH_DRAWS_PER_EPOCH * pilot.MAXIMUM_EPOCHS

# Conservative reservation within the approved 48 hours for all read-only
# prelaunch real-data audits. Synthetic tests and implementation are separate.
PRELAUNCH_AUDIT_RESERVE_SECONDS = 3600.0

_EXCLUDED_RESULT_FIELDS = frozenset(
    {
        "state_dict",
        "best_state_dict",
        "terminal_state_dict",
        "validation_logits",
        "validation_uids",
        "classes",
    }
)


class P05ComprehensiveDevelopmentError(P05CoreError):
    """Stable comprehensive-development failure with a path-free reason code."""


def _err(reason_code: str) -> P05ComprehensiveDevelopmentError:
    return P05ComprehensiveDevelopmentError(reason_code)


def _canon() -> Any:
    return core._canon()


def _check_deadline(deadline: float) -> None:
    if time.perf_counter() > float(deadline):
        raise _err("global_deadline_exceeded")


def _append_jsonl(path: Path, entry: Mapping[str, Any]) -> None:
    with Path(path).open("ab") as stream:
        stream.write(_canon().canonical_json_bytes(dict(entry)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _lower_bound_updates(unit_dir: Path, identifier: str) -> int:
    path = Path(unit_dir) / "histories" / f"{identifier}.jsonl"
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            if line.strip():
                count += 1
    return count * pilot.BATCH_DRAWS_PER_EPOCH


class _GuardedRecorder:
    """Persist each epoch, then enforce the global deadline and storage bounds."""

    __slots__ = ("_recorder", "_budget", "_deadline")

    def __init__(self, recorder: Any, budget: StorageBudget, deadline: float) -> None:
        self._recorder = recorder
        self._budget = budget
        self._deadline = deadline

    def __call__(self, record: Any) -> None:
        self._recorder(record)
        _check_deadline(self._deadline)
        self._budget.check()

    def close(self) -> None:
        self._recorder.close()


def _plan_units(
    bundle: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], list[tuple[str, list[dict[str, Any]]]], set[str]]:
    """Bind the exact 12-slot product per unit and split reused pilot units out."""

    ledger = bundle["ledger"]
    units = list(ledger["units"])
    slots = list(ledger["slots"])
    if len(units) != UNIT_COUNT or len(slots) != INNER_SLOT_COUNT:
        raise _err("ledger_count_mismatch")
    expected = {
        (str(recipe), int(seed)) for recipe in pilot.PILOT_RECIPES for seed in pilot.PILOT_SEEDS
    }
    if len(expected) != SLOTS_PER_UNIT:
        raise _err("recipe_seed_product_mismatch")
    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    if len(unit_by_id) != UNIT_COUNT:
        raise _err("ledger_unit_identity_mismatch")
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for slot in slots:
        by_unit.setdefault(str(slot["unit_id"]), []).append(dict(slot))
    if len(by_unit) != UNIT_COUNT:
        raise _err("ledger_unit_coverage_mismatch")
    pilot_bundle = bundle["pilot_bundle"]
    reused_unit_ids = {str(unit["unit_id"]) for unit in pilot_bundle["units"]}
    reused_slot_ids = inputs.pilot_slot_ids(bundle)
    if len(reused_unit_ids) != 3 or len(reused_slot_ids) != REUSED_PILOT_SLOTS:
        raise _err("pilot_reuse_identity_mismatch")
    if not reused_unit_ids <= set(unit_by_id):
        raise _err("pilot_unit_unknown")
    pilot_observed: set[str] = set()
    all_slot_ids: set[str] = set()
    ordered: list[tuple[str, list[dict[str, Any]]]] = []
    for unit in units:
        unit_id = str(unit["unit_id"])
        group = by_unit.get(unit_id, [])
        if len(group) != SLOTS_PER_UNIT:
            raise _err("unit_slot_count_mismatch")
        if {(str(s["recipe_id"]), int(s["seed"])) for s in group} != expected:
            raise _err("unit_slot_product_mismatch")
        for slot in group:
            if bool(slot.get("excluded_by_protocol")):
                raise _err("slot_excluded")
            if str(slot.get("fitting_role_id")) != str(unit["fitting_role_id"]):
                raise _err("slot_fitting_role_mismatch")
            if str(slot.get("validation_role_id")) != str(unit["validation_role_id"]):
                raise _err("slot_validation_role_mismatch")
        ordered_group = sorted(group, key=lambda s: (str(s["recipe_id"]), int(s["seed"])))
        for slot in ordered_group:
            all_slot_ids.add(str(slot["slot_id"]))
        if unit_id in reused_unit_ids:
            for slot in ordered_group:
                if str(slot["slot_id"]) not in reused_slot_ids:
                    raise _err("pilot_slot_unknown")
                pilot_observed.add(str(slot["slot_id"]))
            continue
        ordered.append((unit_id, ordered_group))
    if pilot_observed != reused_slot_ids:
        raise _err("pilot_slot_coverage_mismatch")
    if len(all_slot_ids) != INNER_SLOT_COUNT:
        raise _err("slot_identity_coverage_mismatch")
    if sum(len(group) for _, group in ordered) != MAXIMUM_NEW_FITS:
        raise _err("new_fit_count_mismatch")
    return unit_by_id, ordered, all_slot_ids


def _reserve_slot_lease(
    artifact_root: Path,
    contract_sha256: str,
    core_plan_id: str,
    slot: Mapping[str, Any],
    permit_sha256: str,
) -> Path:
    directory = pilot._slot_lease_dir(
        artifact_root, contract_sha256, core_plan_id, str(slot["slot_id"])
    )
    core._mkdir_exclusive(directory, "slot_lease_exists")
    path = directory / "lease.json"
    core._atomic_write(
        path,
        _canon().canonical_json_bytes(
            {
                "slot_id": str(slot["slot_id"]),
                "unit_id": str(slot["unit_id"]),
                "recipe_id": str(slot["recipe_id"]),
                "seed": int(slot["seed"]),
                "contract_sha256": str(contract_sha256),
                "core_plan_id": str(core_plan_id),
                "permit_sha256": str(permit_sha256),
            }
        ),
    )
    return path


def _source_identity_adapter(
    result: Any, unit: Mapping[str, Any], slot: Mapping[str, Any]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for field in dataclasses.fields(result):
        if field.name in _EXCLUDED_RESULT_FIELDS:
            continue
        value = getattr(result, field.name)
        summary[field.name] = list(value) if isinstance(value, tuple) else value
    summary["unit_id"] = str(unit["unit_id"])
    summary["slot_id"] = str(slot["slot_id"])
    summary["recipe_id"] = str(slot["recipe_id"])
    summary["seed"] = int(slot["seed"])
    return summary


def _drop_unit_states(items: list[dict[str, Any]]) -> None:
    for item in items:
        result = item.get("result")
        if result is None:
            continue
        for name in _EXCLUDED_RESULT_FIELDS:
            try:
                object.__setattr__(result, name, None)
            except Exception:
                pass
    items.clear()


def _write_ledger(stage: Path, bundle: Mapping[str, Any], budget: StorageBudget) -> None:
    path = Path(stage) / "ledger.json"
    core._atomic_write(path, _canon().canonical_json_bytes(bundle["ledger"]))
    budget.account_new_file(path)


def _write_source_ledger(
    stage: Path,
    bundle: Mapping[str, Any],
    ordered: list[tuple[str, list[dict[str, Any]]]],
    unit_by_id: Mapping[str, Mapping[str, Any]],
    budget: StorageBudget,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "permit_sha256": str(bundle["permit_sha256"]),
        "core_contract_sha256": str(bundle["contract_sha256"]),
        "core_plan_id": str(bundle["core_plan_id"]),
        "ledger_id": str(bundle["ledger"]["ledger_id"]),
        "units": [
            {
                "unit_id": unit_id,
                "station": str(unit_by_id[unit_id]["station"]),
                "fitting_uid_set_sha256": str(unit_by_id[unit_id]["fitting_uid_set_sha256"]),
                "validation_uid_set_sha256": str(unit_by_id[unit_id]["validation_uid_set_sha256"]),
                "slot_ids": [str(slot["slot_id"]) for slot in slots],
            }
            for unit_id, slots in ordered
        ],
    }
    path = Path(stage) / "source_ledger.json"
    core._atomic_write(path, _canon().canonical_json_bytes(payload))
    budget.account_new_file(path)


def _write_progress(
    path: Path,
    state: Mapping[str, Any],
    new_completions: int,
    units_done: int,
    units_total: int,
) -> None:
    core._atomic_write(
        Path(path),
        _canon().canonical_json_bytes(
            {
                "started": int(state["started"]),
                "completed": int(state["completed"]),
                "failed": int(state["failed"]),
                "new_completions": int(new_completions),
                "units_completed": int(units_done),
                "units_total": int(units_total),
                "optimizer_steps": int(state["steps"]),
                "optimizer_steps_exact": bool(state["steps_exact"]),
            }
        ),
    )


def _aggregate(
    bundle: Mapping[str, Any],
    device: str,
    state: Mapping[str, Any],
    new_completions: int,
    selector_count: int,
    units_completed: int,
    scientific_seconds: float,
    live_bytes: int,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "command": "run_development",
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "permit_sha256": str(bundle["permit_sha256"]),
        "core_contract_sha256": str(bundle["contract_sha256"]),
        "core_plan_id": str(bundle["core_plan_id"]),
        "ledger_id": str(bundle["ledger"]["ledger_id"]),
        "device": str(device),
        "reused_pilot_slots": REUSED_PILOT_SLOTS,
        "units_total": UNIT_COUNT,
        "units_completed": int(units_completed),
        "started": int(state["started"]),
        "completed": int(state["completed"]),
        "failed": int(state["failed"]),
        "new_completions": int(new_completions),
        "selector_records": int(selector_count),
        "optimizer_steps": int(state["steps"]),
        "maximum_optimizer_steps": MAXIMUM_FIT_STEPS,
        "sum_elapsed_seconds": float(state["elapsed"]),
        "maximum_peak_cuda_bytes": int(state["peak"]),
        "scientific_seconds_this_stage": float(scientific_seconds),
        "prelaunch_audit_reserve_seconds": PRELAUNCH_AUDIT_RESERVE_SECONDS,
        "scientific_seconds_cumulative_bound": (
            float(scientific_seconds) + PRELAUNCH_AUDIT_RESERVE_SECONDS
        ),
        "live_bytes": int(live_bytes),
        "storage_ceiling_bytes": STORAGE_CEILING_BYTES,
        "claim": COMPREHENSIVE_CLAIM,
        "source_fits_only": True,
        "selection_authorized": False,
        "refit_authorized": False,
        "calibration_authorized": False,
        "outer_evaluation_authorized": False,
    }


def _receipt(
    bundle: Mapping[str, Any],
    device: str,
    state: Mapping[str, Any],
    new_completions: int,
    selector_count: int,
    units_completed: int,
    scientific_seconds: float,
    stage: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "stage": "develop",
        "permit_sha256": str(bundle["permit_sha256"]),
        "core_contract_sha256": str(bundle["contract_sha256"]),
        "core_plan_id": str(bundle["core_plan_id"]),
        "ledger_id": str(bundle["ledger"]["ledger_id"]),
        "device": str(device),
        "stage_manifest_sha256": _canon().sha256_file(Path(stage) / "manifest.json"),
        "scientific_seconds_this_stage": float(scientific_seconds),
        "prelaunch_audit_reserve_seconds": PRELAUNCH_AUDIT_RESERVE_SECONDS,
        "scientific_seconds_cumulative_bound": (
            float(scientific_seconds) + PRELAUNCH_AUDIT_RESERVE_SECONDS
        ),
        "units_completed": int(units_completed),
        "new_completions": int(new_completions),
        "selector_records": int(selector_count),
        "optimizer_steps": int(state["steps"]),
        "maximum_total_seconds": MAXIMUM_TOTAL_SECONDS,
        "claim": COMPREHENSIVE_CLAIM,
        "source_fits_only": True,
        "outer_evaluation_authorized": False,
    }


def _write_failure_summary(
    stage: Path,
    bundle: Mapping[str, Any],
    state: Mapping[str, Any],
    error: BaseException,
    wall_seconds: float,
) -> None:
    core._atomic_write(
        Path(stage) / "summary.json",
        _canon().canonical_json_bytes(
            {
                "status": "fail",
                "command": "run_development",
                "permit_sha256": str(bundle["permit_sha256"]),
                "core_contract_sha256": str(bundle["contract_sha256"]),
                "core_plan_id": str(bundle["core_plan_id"]),
                "ledger_id": str(bundle["ledger"]["ledger_id"]),
                "reason_code": getattr(error, "reason_code", type(error).__name__),
                "started": int(state["started"]),
                "completed": int(state["completed"]),
                "failed": int(state["failed"]),
                "unstarted": MAXIMUM_NEW_FITS - int(state["started"]),
                "optimizer_steps": int(state["steps"]),
                "optimizer_steps_exact": bool(state["steps_exact"]),
                "wall_seconds": float(wall_seconds),
                "claim": COMPREHENSIVE_CLAIM,
                "source_fits_only": True,
                "outer_evaluation_authorized": False,
            }
        ),
    )


def run_development(
    *,
    project_root: Path | str,
    artifact_root: Path | str,
    contract_path: Path | str,
    permit_path: Path | str,
    device: str = "cuda",
) -> dict[str, Any]:
    """Execute the remaining 14,904 source fits once, or fail closed."""

    core._configure_environment()
    wall_start = time.perf_counter()
    deadline = wall_start + MAXIMUM_TOTAL_SECONDS - PRELAUNCH_AUDIT_RESERVE_SECONDS
    if str(device) != "cuda":
        raise _err("device_invalid")
    bundle = inputs.prepare(
        project_root, artifact_root, contract_path, permit_path, require_unstarted=True
    )
    artifact = Path(bundle["artifact_root"])
    contract = bundle["contract"]
    support = bundle["support"]
    permit_digest = str(bundle["permit_sha256"])
    contract_sha256 = str(bundle["contract_sha256"])
    core_plan_id = str(bundle["core_plan_id"])
    state: dict[str, Any] = {
        "started": 0,
        "completed": 0,
        "failed": 0,
        "steps": 0,
        "steps_exact": True,
        "elapsed": 0.0,
        "peak": 0,
    }
    new_completions = 0
    units_completed = 0
    selector_seen: set[str] = set()
    current_unit_dir: Path | None = None
    current_identifier: str | None = None
    result_available = False
    in_flight = False
    stage: Path | None = None
    budget: StorageBudget | None = None
    try:
        torch = importlib.import_module("torch")
        torch.set_num_threads(1)
        pilot._development_kernel()
        if not bool(torch.cuda.is_available()):
            raise _err("cuda_unavailable")
        if pilot._free_cuda_bytes(torch) < pilot.MINIMUM_FREE_CUDA_BYTES:
            raise _err("insufficient_free_cuda")
        pilot._checkpoint_preflight(torch, artifact)
        provenance_before = core._capture_provenance(
            bundle["repository_root"], bundle["project_root"], artifact
        )
        pilot_records = inputs.import_pilot(bundle, device=device)
        if len(pilot_records) != REUSED_PILOT_SLOTS:
            raise _err("pilot_record_count_mismatch")
        for record in pilot_records:
            selector_seen.add(str(record["slot_id"]))
        if selector_seen != inputs.pilot_slot_ids(bundle):
            raise _err("pilot_selector_identity_mismatch")
        unit_by_id, ordered, all_slot_ids = _plan_units(bundle)
        _check_deadline(deadline)
        runs_root = artifact / NAMESPACE / "runs"
        core._reject_symlink_chain(runs_root)
        runs_root.mkdir(parents=True, exist_ok=True)
        run_root = runs_root / permit_digest
        core._reject_symlink_chain(run_root)
        core._mkdir_exclusive(run_root, "comprehensive_run_exists")
        stage = run_root / "develop"
        core._mkdir_exclusive(stage, "comprehensive_stage_exists")
        units_root = stage / "units"
        units_root.mkdir(parents=True, exist_ok=True)
        budget = StorageBudget(artifact, run_root, ceiling=STORAGE_CEILING_BYTES)
        events_path = stage / "events.jsonl"
        selector_path = stage / "selector.jsonl"
        progress_path = stage / "progress.json"
        budget.register_growing(events_path)
        budget.register_growing(selector_path)
        budget.register_growing(progress_path)
        _write_ledger(stage, bundle, budget)
        _write_source_ledger(stage, bundle, ordered, unit_by_id, budget)
        input_manifest = stage / "input_manifest.json"
        core._atomic_write(
            input_manifest,
            _canon().canonical_json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "permit_sha256": permit_digest,
                    "core_contract_sha256": contract_sha256,
                    "core_plan_id": core_plan_id,
                    "ledger_id": str(bundle["ledger"]["ledger_id"]),
                    "reused_pilot_slots": REUSED_PILOT_SLOTS,
                    "new_fits": MAXIMUM_NEW_FITS,
                    "units": len(ordered),
                }
            ),
        )
        budget.account_new_file(input_manifest)
        provenance_before_path = stage / "provenance_before.json"
        core._atomic_write(provenance_before_path, _canon().canonical_json_bytes(provenance_before))
        budget.account_new_file(provenance_before_path)
        _append_jsonl(
            events_path,
            {
                "event": "run_started",
                "permit_sha256": permit_digest,
                "core_plan_id": core_plan_id,
                "ledger_id": str(bundle["ledger"]["ledger_id"]),
                "device": str(device),
                "new_units": len(ordered),
                "new_fits": MAXIMUM_NEW_FITS,
            },
        )
        for record in pilot_records:
            _append_jsonl(selector_path, record)
        for unit_id, unit_slots in ordered:
            _check_deadline(deadline)
            budget.check(headroom_bytes=UNIT_BUDGET_HEADROOM_BYTES)
            unit = unit_by_id[unit_id]
            unit_dir = units_root / unit_id
            budget.activate_unit(unit_dir)
            current_unit_dir = unit_dir
            unit_inputs = pilot.prepare_role_inputs({**bundle, "units": [unit]})[unit_id]
            _append_jsonl(
                events_path,
                {
                    "event": "unit_started",
                    "unit_id": unit_id,
                    "slot_count": len(unit_slots),
                },
            )
            unit_items: list[dict[str, Any]] = []
            for slot in unit_slots:
                _check_deadline(deadline)
                budget.check(headroom_bytes=UNIT_BUDGET_HEADROOM_BYTES)
                if state["started"] >= MAXIMUM_NEW_FITS:
                    raise _err("execution_ceiling_exceeded")
                if state["steps"] > MAXIMUM_FIT_STEPS:
                    raise _err("optimizer_steps_exceeded")
                identifier = pilot.execution_id(unit, slot)
                lease_path = _reserve_slot_lease(
                    artifact, contract_sha256, core_plan_id, slot, permit_digest
                )
                budget.account_new_file(lease_path)
                current_identifier = identifier
                result_available = False
                in_flight = True
                state["started"] += 1
                _append_jsonl(
                    events_path,
                    {
                        "event": "started",
                        "execution_id": identifier,
                        "slot_id": str(slot["slot_id"]),
                        "unit_id": unit_id,
                        "recipe_id": str(slot["recipe_id"]),
                        "seed": int(slot["seed"]),
                        "used_fit_count": state["started"],
                    },
                )
                recorder = pilot.open_history_recorder(unit_dir, identifier)
                guarded = _GuardedRecorder(recorder, budget, deadline)
                try:
                    result = pilot.train_fit(unit_inputs, unit, slot, device, deadline, guarded)
                finally:
                    guarded.close()
                result_available = True
                state["steps"] += int(result.optimizer_steps)
                state["elapsed"] += float(result.elapsed_seconds)
                state["peak"] = max(state["peak"], int(result.peak_cuda_bytes))
                pilot.persist_result(torch, unit_dir, unit, slot, result)
                pilot.check_completed_result(
                    result, unit_dir, unit, slot, contract, unit_inputs, torch, device
                )
                pilot.check_sparse_support(unit, slot, contract, result)
                record = inputs.selector_record(
                    unit, slot, _source_identity_adapter(result, unit, slot)
                )
                _append_jsonl(selector_path, record)
                selector_seen.add(str(slot["slot_id"]))
                unit_items.append(
                    {
                        "slot": slot,
                        "unit": unit,
                        "unit_id": unit_id,
                        "seed": int(slot["seed"]),
                        "result": result,
                    }
                )
                state["completed"] += 1
                new_completions += 1
                in_flight = False
                _append_jsonl(
                    events_path,
                    {
                        "event": "completed",
                        "execution_id": identifier,
                        "slot_id": str(slot["slot_id"]),
                        "status": str(result.status),
                        "optimizer_steps": int(result.optimizer_steps),
                    },
                )
                _check_deadline(deadline)
                budget.check(headroom_bytes=UNIT_BUDGET_HEADROOM_BYTES)
                pilot._enforce_cuda_cap(torch, device)
                if device == "cuda":
                    state["peak"] = max(state["peak"], int(torch.cuda.max_memory_allocated()))
            if len(unit_items) != SLOTS_PER_UNIT:
                raise _err("unit_execution_count_mismatch")
            pilot.check_shared_prefixes(unit_items)
            pilot.check_sparse_equivalences(unit_items, [unit], contract)
            core._write_manifest(unit_dir)
            pilot._verify_manifest(unit_dir)
            _drop_unit_states(unit_items)
            budget.close_unit()
            units_completed += 1
            _write_progress(progress_path, state, new_completions, units_completed, len(ordered))
            _append_jsonl(
                events_path,
                {
                    "event": "unit_completed",
                    "unit_id": unit_id,
                    "completed": state["completed"],
                    "started": state["started"],
                    "optimizer_steps": state["steps"],
                },
            )
            del unit_inputs
            current_unit_dir = None
            current_identifier = None
        _check_deadline(deadline)
        if new_completions != MAXIMUM_NEW_FITS:
            raise _err("new_completion_count_mismatch")
        if selector_seen != all_slot_ids:
            raise _err("selector_coverage_mismatch")
        if units_completed != len(ordered):
            raise _err("unit_completion_mismatch")
        if (
            state["started"] != MAXIMUM_NEW_FITS
            or state["completed"] != MAXIMUM_NEW_FITS
            or state["failed"] != 0
        ):
            raise _err("execution_state_mismatch")
        if state["steps"] > MAXIMUM_FIT_STEPS:
            raise _err("optimizer_steps_exceeded")
        inputs.prepare(
            project_root, artifact_root, contract_path, permit_path, require_unstarted=False
        )
        provenance_after = pilot._post_run_reauth(
            artifact,
            contract,
            support,
            provenance_before,
            bundle["repository_root"],
            bundle["project_root"],
        )
        provenance_after_path = stage / "provenance_after.json"
        core._atomic_write(provenance_after_path, _canon().canonical_json_bytes(provenance_after))
        budget.account_new_file(provenance_after_path)
        scientific_seconds = time.perf_counter() - wall_start
        if scientific_seconds > MAXIMUM_TOTAL_SECONDS:
            raise _err("total_wall_exceeded")
        _check_deadline(deadline)
        live_bytes = budget.check()
        summary = _aggregate(
            bundle,
            device,
            state,
            new_completions,
            len(selector_seen),
            units_completed,
            scientific_seconds,
            live_bytes,
        )
        summary_path = stage / "summary.json"
        core._atomic_write(summary_path, _canon().canonical_json_bytes(summary))
        budget.account_new_file(summary_path)
        core._write_manifest(stage)
        budget.account_new_file(stage / "manifest.json")
        pilot._verify_manifest(stage)
        _check_deadline(deadline)
        budget.reconcile()
        pilot._enforce_cuda_cap(torch, device)
        receipt_seconds = time.perf_counter() - wall_start
        if receipt_seconds > MAXIMUM_TOTAL_SECONDS:
            raise _err("total_wall_exceeded")
        receipt_path = run_root / "development_receipt.json"
        core._atomic_write(
            receipt_path,
            _canon().canonical_json_bytes(
                _receipt(
                    bundle,
                    device,
                    state,
                    new_completions,
                    len(selector_seen),
                    units_completed,
                    receipt_seconds,
                    stage,
                )
            ),
        )
        budget.account_new_file(receipt_path)
        budget.check()
        _check_deadline(deadline)
        summary["scientific_seconds_this_stage"] = receipt_seconds
        summary["scientific_seconds_cumulative_bound"] = (
            receipt_seconds + PRELAUNCH_AUDIT_RESERVE_SECONDS
        )
        return summary
    except BaseException as error:
        wall_seconds = time.perf_counter() - wall_start
        if in_flight:
            state["failed"] += 1
            if (
                not result_available
                and current_identifier is not None
                and current_unit_dir is not None
            ):
                state["steps"] += _lower_bound_updates(current_unit_dir, current_identifier)
                state["steps_exact"] = False
        if budget is not None:
            try:
                budget.check()
            except Exception:
                pass
        if stage is not None:
            try:
                _write_failure_summary(stage, bundle, state, error, wall_seconds)
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
        raise P05ComprehensiveDevelopmentError("comprehensive_execution_failed") from error
