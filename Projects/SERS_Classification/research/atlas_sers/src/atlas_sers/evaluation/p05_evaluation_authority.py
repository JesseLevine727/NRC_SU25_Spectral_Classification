"""P05 evaluation authority: read-only pre-prediction authentication of the
completed comprehensive refit stage.

Verifies the persisted refit receipt, stage and unit manifests, unit leases,
per-unit summaries, terminal checkpoints and calibrations, then reconciles the
receipt/summary counters against the authenticated refit plan and the on-disk
history.  It trains nothing, fits no temperature, builds no model, runs no
forward pass, loads no feature arrays and writes nothing.  The outer runner
MUST call this before any predict_refit.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_prediction as prediction
from atlas_sers.evaluation import p05_refit_authority as authority
from atlas_sers.evaluation import p05_refit_evidence as evidence

__all__ = ["P05EvaluationAuthorityError", "authenticate_refits"]

SCHEMA = "nato-sers-p05-comprehensive-refits-v1"
PROTOCOL_VERSION = development.PROTOCOL_VERSION
STAGE_NAME = "refits"
RECEIPT_NAME = "refit_receipt.json"
SUMMARY_NAME = "summary.json"
MANIFEST_NAME = "manifest.json"
LEASE_NAME = "lease.json"
UNITS_NAME = "units"
COMMAND = "run_comprehensive_refits"
CLAIM = "source_refits_and_calibration_no_held_predictions"
SOURCE_FIT_COUNT = 14904
REUSED_PILOT_FIT_COUNT = 36
STRATEGY_ALIAS_COUNT = 2880
MAXIMUM_REFITS = 2880
MAXIMUM_UPDATES_PER_REFIT = 800
SOURCE_MAXIMUM_UPDATES = SOURCE_FIT_COUNT * MAXIMUM_UPDATES_PER_REFIT
MAXIMUM_COMBINED_UPDATES = 14227200
MAXIMUM_FIT_SECONDS = 120.0
MAXIMUM_CUDA_ALLOCATED_BYTES = 4294967296
PRELAUNCH_AUDIT_RESERVE_SECONDS = 3600
MAXIMUM_TOTAL_SECONDS = 172800
COUNTER_KEYS = (
    "calibration_started",
    "calibration_completed",
    "calibration_failed",
    "neural_started",
    "neural_completed",
    "neural_failed",
    "optimizer_steps",
    "optimizer_steps_exact",
    "peak_cuda_bytes",
    "elapsed_seconds",
    "sum_fit_elapsed_seconds",
)


class P05EvaluationAuthorityError(core.P05CoreError):
    """Stable, path-free evaluation-authority failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05EvaluationAuthorityError(code)


def _finite(value: Any, code: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), code)
    number = float(value)
    _require(math.isfinite(number), code)
    return number


def _integer(value: Any, code: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), code)
    return int(value)


def _read_history(path: Path) -> list[Any]:
    core._reject_symlink_chain(path)
    _require(path.is_file() and not path.is_symlink(), "unit_history_missing")
    records: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except ValueError:
                raise P05EvaluationAuthorityError("unit_history_malformed") from None
    return records


def authenticate_refits(bundle: Any, *, deadline: Any) -> dict[str, Any]:
    """Authenticate the completed comprehensive refit stage read-only."""

    _require(isinstance(bundle, Mapping), "bundle_malformed")
    deadline = _finite(deadline, "deadline_malformed")
    freeze._check_deadline(deadline)

    authenticated = authority.authenticate_selection(bundle, deadline=deadline)
    _require(isinstance(authenticated, Mapping), "authority_malformed")
    plan = authenticated.get("plan")
    _require(isinstance(plan, Mapping), "authority_plan_malformed")
    prior_seconds = _finite(authenticated.get("prior_seconds"), "authority_prior_malformed")
    _require(
        PRELAUNCH_AUDIT_RESERVE_SECONDS <= prior_seconds <= MAXIMUM_TOTAL_SECONDS,
        "authority_prior_out_of_range",
    )
    source_optimizer_steps = _integer(
        authenticated.get("source_optimizer_steps"), "authority_source_steps_malformed"
    )
    selection_receipt = authenticated.get("selection_receipt")
    _require(isinstance(selection_receipt, Mapping), "selection_receipt_malformed")

    unique_refits = plan.get("unique_refits")
    aliases = plan.get("strategy_aliases")
    plan_id = plan.get("plan_id")
    _require(isinstance(plan_id, str) and bool(plan_id), "plan_id_malformed")
    _require(isinstance(unique_refits, Mapping), "plan_unique_refits_malformed")
    _require(
        isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)),
        "plan_aliases_malformed",
    )
    unique_count = len(unique_refits)
    alias_count = len(aliases)
    _require(0 < unique_count <= MAXIMUM_REFITS, "unique_refit_count_out_of_range")
    _require(alias_count == STRATEGY_ALIAS_COUNT, "strategy_alias_count_mismatch")
    for alias in aliases:
        _require(isinstance(alias, Mapping), "alias_malformed")
        _require(str(alias.get("refit_id")) in unique_refits, "alias_refit_unknown")
    _require(0 <= source_optimizer_steps <= SOURCE_MAXIMUM_UPDATES, "source_updates_exceeded")

    permit_sha256 = bundle.get("permit_sha256")
    contract_sha256 = bundle.get("contract_sha256")
    core_plan_id = bundle.get("core_plan_id")
    ledger = bundle.get("ledger")
    _require(isinstance(ledger, Mapping), "ledger_malformed")
    ledger_id = ledger.get("ledger_id")
    _require(isinstance(ledger_id, str) and bool(ledger_id), "ledger_id_malformed")
    support = bundle.get("support")
    _require(support is not None, "support_missing")
    artifact_root = bundle.get("artifact_root")
    _require(artifact_root is not None, "artifact_root_missing")

    run_root = Path(artifact_root) / "p05comprehensive" / "runs" / str(permit_sha256)
    stage = run_root / STAGE_NAME
    core._reject_symlink_chain(stage)
    receipt_path = run_root / RECEIPT_NAME
    core._reject_symlink_chain(receipt_path)
    _require(receipt_path.is_file() and not receipt_path.is_symlink(), "refit_receipt_missing")
    refit_receipt = core._read_json(receipt_path, "refit_receipt")
    _require(isinstance(refit_receipt, Mapping), "refit_receipt_malformed")

    manifest_path = stage / MANIFEST_NAME
    core._reject_symlink_chain(manifest_path)
    _require(manifest_path.is_file() and not manifest_path.is_symlink(), "refit_manifest_missing")
    stage_manifest_sha256 = refit_receipt.get("stage_manifest_sha256")
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

    identity = {
        "schema_version": SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "command": COMMAND,
        "stage": STAGE_NAME,
        "claim": CLAIM,
        "permit_sha256": permit_sha256,
        "core_contract_sha256": contract_sha256,
        "core_plan_id": core_plan_id,
        "ledger_id": ledger_id,
        "selection_plan_id": plan_id,
        "source_fit_count": SOURCE_FIT_COUNT,
        "reused_pilot_fit_count": REUSED_PILOT_FIT_COUNT,
        "unique_refit_count": unique_count,
        "strategy_alias_count": alias_count,
        "source_optimizer_steps": source_optimizer_steps,
        "outer_predictions_started": 0,
    }
    for name, value in identity.items():
        _require(refit_receipt.get(name) == value, f"receipt_{name}_mismatch")
    _require(refit_receipt.get("refits_complete") is True, "receipt_refits_incomplete")
    _require(refit_receipt.get("calibrations_complete") is True, "receipt_calibrations_incomplete")
    _require(refit_receipt.get("status") == "complete", "receipt_status_incomplete")

    summary_path = stage / SUMMARY_NAME
    core._reject_symlink_chain(summary_path)
    _require(summary_path.is_file() and not summary_path.is_symlink(), "refit_summary_missing")
    summary = core._read_json(summary_path, "refit_summary")
    _require(isinstance(summary, Mapping), "refit_summary_malformed")
    for name, value in identity.items():
        _require(summary.get(name) == value, f"summary_{name}_mismatch")
    _require(summary.get("refits_complete") is True, "summary_refits_incomplete")
    _require(summary.get("calibrations_complete") is True, "summary_calibrations_incomplete")
    _require(summary.get("status") == "complete", "summary_status_incomplete")

    receipt_counters = refit_receipt.get("counters")
    summary_counters = summary.get("counters")
    _require(isinstance(receipt_counters, Mapping), "receipt_counters_malformed")
    _require(isinstance(summary_counters, Mapping), "summary_counters_malformed")
    _require(
        set(receipt_counters) == set(summary_counters) == set(COUNTER_KEYS),
        "counter_fields_mismatch",
    )
    for name in COUNTER_KEYS:
        _require(name in receipt_counters, f"receipt_counter_{name}_missing")
        _require(name in summary_counters, f"summary_counter_{name}_missing")
    for name in sorted(set(receipt_counters) | set(summary_counters)):
        if name == "elapsed_seconds":
            continue
        _require(
            receipt_counters.get(name) == summary_counters.get(name),
            f"counters_{name}_mismatch",
        )
    summary_elapsed = _finite(summary_counters["elapsed_seconds"], "summary_elapsed_malformed")
    receipt_elapsed = _finite(receipt_counters["elapsed_seconds"], "receipt_elapsed_malformed")
    _require(0.0 <= summary_elapsed <= receipt_elapsed, "summary_elapsed_exceeds_receipt")

    for prefix in ("calibration", "neural"):
        _require(
            _integer(receipt_counters[f"{prefix}_started"], f"{prefix}_started_malformed")
            == unique_count,
            f"{prefix}_started_mismatch",
        )
        _require(
            _integer(receipt_counters[f"{prefix}_completed"], f"{prefix}_completed_malformed")
            == unique_count,
            f"{prefix}_completed_mismatch",
        )
        _require(
            _integer(receipt_counters[f"{prefix}_failed"], f"{prefix}_failed_malformed") == 0,
            f"{prefix}_failures_present",
        )
    _require(receipt_counters["optimizer_steps_exact"] is True, "optimizer_steps_inexact")
    refit_optimizer_steps = _integer(
        receipt_counters["optimizer_steps"], "optimizer_steps_malformed"
    )
    _require(refit_optimizer_steps >= 0, "optimizer_steps_malformed")
    _require(
        refit_optimizer_steps <= unique_count * MAXIMUM_UPDATES_PER_REFIT,
        "refit_updates_exceeded",
    )
    peak_cuda_bytes = _integer(receipt_counters["peak_cuda_bytes"], "peak_cuda_malformed")
    _require(0 <= peak_cuda_bytes <= MAXIMUM_CUDA_ALLOCATED_BYTES, "peak_cuda_exceeded")
    sum_fit_elapsed = _finite(
        receipt_counters["sum_fit_elapsed_seconds"], "sum_fit_elapsed_malformed"
    )
    _require(
        0.0 <= sum_fit_elapsed <= MAXIMUM_FIT_SECONDS * unique_count,
        "sum_fit_elapsed_exceeded",
    )

    total_new_optimizer_steps = _integer(
        refit_receipt.get("total_new_optimizer_steps"), "total_new_optimizer_steps_malformed"
    )
    _require(
        total_new_optimizer_steps == source_optimizer_steps + refit_optimizer_steps,
        "total_new_optimizer_steps_mismatch",
    )
    _require(total_new_optimizer_steps <= MAXIMUM_COMBINED_UPDATES, "combined_updates_exceeded")
    _require(
        _integer(summary.get("total_new_optimizer_steps"), "summary_total_updates_malformed")
        == total_new_optimizer_steps,
        "summary_total_updates_mismatch",
    )

    _require(
        development.PRELAUNCH_AUDIT_RESERVE_SECONDS == PRELAUNCH_AUDIT_RESERVE_SECONDS,
        "reserve_constant_mismatch",
    )
    _require(
        development.MAXIMUM_TOTAL_SECONDS == MAXIMUM_TOTAL_SECONDS,
        "maximum_constant_mismatch",
    )
    cumulative_prior = _finite(
        refit_receipt.get("prior_scientific_seconds_cumulative_bound"),
        "receipt_prior_malformed",
    )
    _require(cumulative_prior == prior_seconds, "authority_prior_mismatch")
    for label, payload in (("receipt", refit_receipt), ("summary", summary)):
        _require(
            _finite(
                payload.get("prior_scientific_seconds_cumulative_bound"),
                f"{label}_prior_malformed",
            )
            == prior_seconds,
            f"{label}_prior_mismatch",
        )
        stage_seconds = _finite(
            payload.get("scientific_seconds_this_stage"), f"{label}_stage_malformed"
        )
        _require(
            stage_seconds == payload["counters"]["elapsed_seconds"]
            and 0 <= sum_fit_elapsed <= stage_seconds,
            f"{label}_stage_seconds_inconsistent",
        )
        cumulative = _finite(
            payload.get("scientific_seconds_cumulative_bound"), f"{label}_cumulative_malformed"
        )
        _require(cumulative == prior_seconds + stage_seconds, f"{label}_cumulative_mismatch")
        _require(cumulative <= MAXIMUM_TOTAL_SECONDS, f"{label}_cumulative_exceeded")
        _require(
            _finite(payload.get("prelaunch_audit_reserve_seconds"), f"{label}_reserve_malformed")
            == PRELAUNCH_AUDIT_RESERVE_SECONDS,
            f"{label}_reserve_mismatch",
        )
        _require(
            _finite(payload.get("maximum_total_seconds"), f"{label}_maximum_malformed")
            == MAXIMUM_TOTAL_SECONDS,
            f"{label}_maximum_mismatch",
        )
    _require(
        _finite(refit_receipt.get("scientific_seconds_this_stage"), "receipt_stage_malformed")
        >= _finite(summary.get("scientific_seconds_this_stage"), "summary_stage_malformed"),
        "receipt_stage_less_than_summary",
    )

    units_dir = stage / UNITS_NAME
    core._reject_symlink_chain(units_dir)
    _require(units_dir.is_dir() and not units_dir.is_symlink(), "units_dir_missing")
    present_units = set()
    for entry in sorted(units_dir.iterdir(), key=lambda path: path.name):
        _require(not entry.is_symlink(), "unit_symlink_rejected")
        _require(entry.is_dir(), "unit_entry_not_directory")
        present_units.add(entry.name)
    _require(present_units == set(unique_refits), "units_directory_set_mismatch")
    freeze._check_deadline(deadline)

    ordered = sorted(
        (dict(spec) for spec in unique_refits.values()),
        key=lambda spec: (
            str(spec["context_id"]),
            int(spec["seed"]),
            str(spec["recipe_id"]),
            str(spec["refit_id"]),
        ),
    )
    expected_steps = sum(
        _integer(spec.get("epochs"), "spec_epochs_malformed") * 4 for spec in ordered
    )
    _require(refit_optimizer_steps == expected_steps, "exact_optimizer_steps_mismatch")
    expected_sizes = {
        key: len(list(items))
        for key, items in itertools.groupby(
            ordered, key=lambda spec: (str(spec["context_id"]), int(spec["seed"]))
        )
    }

    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    actual_elapsed = 0.0
    actual_peak = 0
    for spec in ordered:
        freeze._check_deadline(deadline)
        refit_id = str(spec["refit_id"])
        unit_dir = units_dir / refit_id
        core._reject_symlink_chain(unit_dir)
        _require(unit_dir.is_dir() and not unit_dir.is_symlink(), "unit_directory_missing")
        pilot._verify_manifest(unit_dir)

        lease_path = unit_dir / LEASE_NAME
        core._reject_symlink_chain(lease_path)
        _require(lease_path.is_file() and not lease_path.is_symlink(), "unit_lease_missing")
        lease = core._read_json(lease_path, "unit_lease")
        _require(isinstance(lease, Mapping), "unit_lease_malformed")
        _require(dict(lease) == {"selection_plan_id": plan_id, "spec": spec}, "unit_lease_mismatch")

        loaded_state, _model_sha256 = prediction._load_result(unit_dir, spec)
        del loaded_state
        _calibration, _calibration_sha256 = prediction._load_calibration(unit_dir, spec)
        del _calibration

        unit_summary = core._read_json(unit_dir / SUMMARY_NAME, "unit_summary")
        _require(isinstance(unit_summary, Mapping), "unit_summary_malformed")
        result = SimpleNamespace(**dict(unit_summary))
        _require(
            _integer(getattr(result, "optimizer_steps", None), "result_optimizer_steps_malformed")
            == _integer(spec.get("epochs"), "spec_epochs_malformed") * 4,
            "result_optimizer_steps_mismatch",
        )
        elapsed = _finite(getattr(result, "elapsed_seconds", None), "result_seconds_malformed")
        _require(0.0 <= elapsed <= MAXIMUM_FIT_SECONDS, "result_seconds_out_of_range")
        peak = _integer(getattr(result, "peak_cuda_bytes", None), "result_peak_malformed")
        _require(0 <= peak <= MAXIMUM_CUDA_ALLOCATED_BYTES, "result_peak_out_of_range")
        actual_elapsed += elapsed
        actual_peak = max(actual_peak, peak)

        history = _read_history(unit_dir / "histories" / f"{refit_id}.jsonl")
        persisted = list(getattr(result, "history", ()))
        _require(persisted == history, "unit_history_mismatch")
        _require(
            len(history) == _integer(getattr(result, "epochs", None), "result_epochs_malformed"),
            "unit_history_length_mismatch",
        )

        summary_evidence = evidence.summarize_result(spec, result)
        key = (str(spec["context_id"]), int(spec["seed"]))
        bucket = groups.setdefault(key, [])
        bucket.append(summary_evidence)
        if len(bucket) == expected_sizes[key]:
            pairs = evidence.cross_instrument_master_count(support, spec)
            evidence.check_recipe_group(bucket, cross_instrument_pairs=pairs)
            del groups[key]
        del result, summary_evidence, unit_summary, history, persisted
        freeze._check_deadline(deadline)
    _require(not groups, "recipe_groups_incomplete")

    _require(
        math.isclose(actual_elapsed, sum_fit_elapsed, rel_tol=0.0, abs_tol=1e-9),
        "sum_fit_elapsed_mismatch",
    )
    _require(actual_peak == peak_cuda_bytes, "peak_cuda_mismatch")
    freeze._check_deadline(deadline)

    return {
        "plan": plan,
        "prior_seconds": _finite(
            refit_receipt["scientific_seconds_cumulative_bound"], "receipt_cumulative_malformed"
        ),
        "source_optimizer_steps": source_optimizer_steps,
        "refit_optimizer_steps": refit_optimizer_steps,
        "refit_receipt": refit_receipt,
        "selection_receipt": selection_receipt,
    }
