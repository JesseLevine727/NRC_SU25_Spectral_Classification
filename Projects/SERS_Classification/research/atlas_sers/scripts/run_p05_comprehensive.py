#!/usr/bin/env python3
"""Thin sanitized CLI for the reviewed P05 comprehensive development modules.

``inspect`` authenticates the permit, contract and rebuilt ledger in
metadata-only mode without importing torch.  ``develop`` lazily imports
``p05_comprehensive_development.run_development`` and executes ONLY the source
stage, never the benchmark or outer evaluation.  ``status`` prints whitelisted
aggregate progress counters. ``freeze-selection``, ``refits``, ``evaluate``,
``aggregate``, ``compare`` and ``report`` lazily import their reviewed stage
modules and never chain into another stage.
``recover-develop`` runs only the approved recovery source stage;
``recovery-status`` reads its aggregate progress without importing torch.
Every command emits one canonical JSON object and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (_REPOSITORY_ROOT / "src", _REPOSITORY_ROOT):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

_STATUS_FIELDS = (
    "started",
    "completed",
    "failed",
    "new_completions",
    "units_completed",
    "units_total",
    "optimizer_steps",
    "optimizer_steps_exact",
)

_RECOVERY_PROGRESS_FIELDS = (
    "new_started",
    "new_completed",
    "new_failed",
    "new_optimizer_steps",
    "new_optimizer_steps_exact",
    "new_elapsed_seconds",
    "new_peak_cuda_bytes",
    "reused_completed",
    "reused_optimizer_steps",
    "replay_started",
    "unstarted_started",
    "units_completed",
    "units_total",
)

_RECOVERY_DEVELOP_FIELDS = (
    "started",
    "completed",
    "failed",
    "recovery_started",
    "recovery_completed",
    "recovery_failed",
    "reused_original_completions",
    "reused_pilot_slots",
    "replay_attempts",
    "originally_unstarted_attempts",
    "units_completed",
    "units_total",
    "selector_records",
    "optimizer_steps",
    "optimizer_steps_exact",
    "optimizer_steps_scope",
    "total_source_optimizer_steps_lower_bound",
    "total_source_optimizer_steps_charged_upper_bound",
    "total_source_optimizer_steps_exact",
    "scientific_seconds_this_stage",
    "scientific_seconds_cumulative_bound",
)

_RECOVERY_DEVELOP_BOOL_FIELDS = frozenset(
    {"optimizer_steps_exact", "total_source_optimizer_steps_exact"}
)
_RECOVERY_DEVELOP_FLOAT_FIELDS = frozenset(
    {"scientific_seconds_this_stage", "scientific_seconds_cumulative_bound"}
)


class _InvalidArguments(Exception):
    """Raised by the sanitized CLI parser for invalid command-line arguments."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _InvalidArguments(message)


def _module(name: str) -> Any:
    return importlib.import_module(name)


def _inputs() -> Any:
    return _module("atlas_sers.evaluation.p05_comprehensive_inputs")


def _canon() -> Any:
    return _module("atlas_sers.governance.canonical")


class _RecoveryError(Exception):
    """Sanitized, path-free failure for the recovery commands."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_finite_nonneg(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return value >= 0 and math.isfinite(value)
    except OverflowError:
        return False


def _recovery_authority() -> Any:
    return _module("atlas_sers.evaluation.p05_recovery_authority")


def _recovery_inputs() -> Any:
    return _module("atlas_sers.evaluation.p05_recovery_inputs")


def _emit(payload: Mapping[str, Any]) -> None:
    encoded = _canon().canonical_json_bytes(dict(payload))
    sys.stdout.write(encoded.decode("utf-8") + "\n")


def _failure(command: str, reason_code: str) -> int:
    _emit({"status": "fail", "command": command, "reason_code": reason_code})
    return 1


def _inspect(arguments: argparse.Namespace) -> dict[str, Any]:
    bundle = _inputs().prepare(
        arguments.project_root,
        arguments.artifact_root,
        arguments.contract,
        arguments.permit,
        require_unstarted=True,
    )
    ledger = bundle["ledger"]
    summary = ledger["summary"]
    return {
        "status": "ok",
        "command": "inspect",
        "schema_version": ledger["schema_version"],
        "contract_sha256": bundle["contract_sha256"],
        "permit_sha256": bundle["permit_sha256"],
        "ledger_id": ledger["ledger_id"],
        "core_plan_id": bundle["core_plan_id"],
        "contexts": summary["context_count"],
        "contexts_by_selection_mode": summary["contexts_by_selection_mode"],
        "contexts_by_phase_gate": summary["contexts_by_phase_gate"],
        "eligible_units": summary["eligible_unit_count"],
        "excluded_units": summary["excluded_unit_count"],
        "slots": summary["slot_count"],
        "eligible_slots": summary["eligible_slot_count"],
        "excluded_slots": summary["excluded_slot_count"],
        "reused_pilot_slots": len(bundle["pilot_bundle"]["slots"]),
        "arrays_loaded": False,
        "fits_started": 0,
        "execution_authorized": False,
    }


def _develop(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run ONLY the reviewed source stage; no benchmark or outer evaluation."""

    summary = _module("atlas_sers.evaluation.p05_comprehensive_development").run_development(
        project_root=arguments.project_root,
        artifact_root=arguments.artifact_root,
        contract_path=arguments.contract,
        permit_path=arguments.permit,
        device=arguments.device,
    )
    return {**dict(summary), "command": "develop"}


def _status(arguments: argparse.Namespace) -> dict[str, Any]:
    inputs = _inputs()
    _permit, permit_digest = inputs._load_permit(arguments.permit)
    _project, artifact, _repository = _module("atlas_sers.evaluation.p05_pilot")._resolve_paths(
        arguments.project_root, arguments.artifact_root
    )
    progress = _module("atlas_sers.evaluation.p05_core_run")._read_json(
        artifact / "p05comprehensive" / "runs" / permit_digest / "develop" / "progress.json",
        "development_progress",
    )
    if not isinstance(progress, Mapping):
        raise inputs.ComprehensiveInputsError("development_progress_malformed")
    report = {"status": "ok", "command": "status"}
    for field in _STATUS_FIELDS:
        value = progress.get(field)
        if field == "optimizer_steps_exact":
            valid = isinstance(value, bool)
        else:
            valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        if not valid:
            raise inputs.ComprehensiveInputsError("development_progress_malformed")
        report[field] = value
    return report


def _recover_develop(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run ONLY the reviewed recovery-development stage; no chaining or retries."""

    claim = _module("atlas_sers.evaluation.p05_recovery_development").run_recovery_development(
        project_root=arguments.project_root,
        artifact_root=arguments.artifact_root,
        contract_path=arguments.contract,
        base_permit_path=arguments.permit,
        recovery_permit_path=arguments.recovery_permit,
        device=arguments.device,
    )
    if not isinstance(claim, Mapping):
        raise _RecoveryError("recovery_development_unverified")
    receipt = _module("atlas_sers.evaluation.p05_recovery_receipt")
    fixed = {
        "schema_version": receipt.SCHEMA_VERSION,
        "claim": receipt.CLAIM,
        "started": 14905,
        "completed": 14904,
        "failed": 1,
        "recovery_started": 6184,
        "recovery_completed": 6184,
        "recovery_failed": 0,
        "reused_original_completions": 8720,
        "reused_pilot_slots": 36,
        "replay_attempts": 1,
        "originally_unstarted_attempts": 6183,
        "units_completed": 1242,
        "units_total": receipt.REQUIRED_UNITS_TOTAL,
        "selector_records": 14940,
        "optimizer_steps_exact": True,
        "optimizer_steps_scope": receipt.OPTIMIZER_STEPS_SCOPE,
        "total_source_optimizer_steps_exact": False,
    }
    for field, expected in fixed.items():
        if type(claim.get(field)) is not type(expected) or claim[field] != expected:
            raise _RecoveryError("recovery_development_unverified")
    report: dict[str, Any] = {"status": "complete", "command": "recover-develop"}
    for field in _RECOVERY_DEVELOP_FIELDS:
        if field not in claim:
            raise _RecoveryError("recovery_development_unverified")
        value = claim[field]
        if field in _RECOVERY_DEVELOP_BOOL_FIELDS:
            valid = isinstance(value, bool)
        elif field in _RECOVERY_DEVELOP_FLOAT_FIELDS:
            valid = _is_finite_nonneg(value)
        elif field == "optimizer_steps_scope":
            valid = isinstance(value, str) and bool(value)
        else:
            valid = _is_count(value)
        if not valid:
            raise _RecoveryError("recovery_development_unverified")
        report[field] = value
    return report


def _recovery_status(arguments: argparse.Namespace) -> dict[str, Any]:
    """Read ONLY the bounded recovery progress counters; metadata only."""

    inputs = _inputs()
    _base_permit, base_digest = inputs._load_permit(arguments.permit)
    authority = _recovery_authority()
    base_pin = authority.BASECOMPREHENSIVE_PERMIT_SHA256
    if base_digest != base_pin:
        raise _RecoveryError("base_permit_mismatch")
    authority.load_recovery_permit(arguments.recovery_permit)
    _project, artifact, _repository = _module("atlas_sers.evaluation.p05_pilot")._resolve_paths(
        arguments.project_root, arguments.artifact_root
    )
    progress_path = (
        artifact
        / "p05comprehensive"
        / "runs"
        / base_pin
        / "recoveries"
        / authority.RECOVERY_PERMIT_SHA256
        / "develop"
        / "progress.json"
    )
    progress = _recovery_inputs()._read_json_mapping(
        progress_path,
        "recovery_progress",
        deadline=time.perf_counter() + 30,
    )
    if not isinstance(progress, Mapping):
        raise _RecoveryError("recovery_progress_malformed")
    report: dict[str, Any] = {"status": "ok", "command": "recovery-status"}
    for field in _RECOVERY_PROGRESS_FIELDS:
        if field not in progress:
            raise _RecoveryError("recovery_progress_malformed")
        value = progress[field]
        if field == "new_optimizer_steps_exact":
            valid = isinstance(value, bool)
        elif field == "new_elapsed_seconds":
            valid = _is_finite_nonneg(value)
        else:
            valid = _is_count(value)
        if not valid:
            raise _RecoveryError("recovery_progress_malformed")
        report[field] = value
    return report


def _freeze_selection(arguments: argparse.Namespace) -> dict[str, Any]:
    summary = _module("atlas_sers.evaluation.p05_comprehensive_freeze").freeze_selection(
        project_root=arguments.project_root,
        artifact_root=arguments.artifact_root,
        contract_path=arguments.contract,
        permit_path=arguments.permit,
    )
    return {**dict(summary), "command": "freeze-selection"}


def _refits(arguments: argparse.Namespace) -> dict[str, Any]:
    summary = _module("atlas_sers.evaluation.p05_comprehensive_refits").run_refits(
        project_root=arguments.project_root,
        artifact_root=arguments.artifact_root,
        contract_path=arguments.contract,
        permit_path=arguments.permit,
        device=arguments.device,
    )
    return {**dict(summary), "command": "refits"}


def _evaluate(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run ONLY the reviewed outer evaluation stage; no chaining or retries."""

    summary = _module("atlas_sers.evaluation.p05_comprehensive_evaluation").run_evaluation(
        project_root=arguments.project_root,
        artifact_root=arguments.artifact_root,
        contract_path=arguments.contract,
        permit_path=arguments.permit,
        device=arguments.device,
    )
    return {**dict(summary), "command": "evaluate"}


def _aggregate(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run ONLY the reviewed aggregation stage; no chaining or retries."""

    summary = _module("atlas_sers.evaluation.p05_comprehensive_aggregation").run_aggregation(
        project_root=arguments.project_root,
        artifact_root=arguments.artifact_root,
        contract_path=arguments.contract,
        permit_path=arguments.permit,
    )
    return {**dict(summary), "command": "aggregate"}


def _compare(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run ONLY the reviewed comparison stage; no chaining or retries."""

    summary = _module("atlas_sers.evaluation.p05_comprehensive_comparison").run_comparison(
        project_root=arguments.project_root,
        artifact_root=arguments.artifact_root,
        contract_path=arguments.contract,
        permit_path=arguments.permit,
    )
    return {**dict(summary), "command": "compare"}


def _report(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run ONLY the reviewed reporting stage; no chaining or retries."""

    summary = _module("atlas_sers.evaluation.p05_comprehensive_reporting").run_reporting(
        project_root=arguments.project_root,
        artifact_root=arguments.artifact_root,
        contract_path=arguments.contract,
        permit_path=arguments.permit,
    )
    return {**dict(summary), "command": "report"}


def _build_parser() -> _Parser:
    parser = _Parser(prog="run_p05_comprehensive")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "inspect",
        "develop",
        "status",
        "freeze-selection",
        "refits",
        "evaluate",
        "aggregate",
        "compare",
        "report",
        "recover-develop",
        "recovery-status",
    ):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--project-root", required=True)
        subparser.add_argument("--artifact-root", required=True)
        subparser.add_argument("--contract", required=True)
        subparser.add_argument("--permit", required=True)
        if name in ("recover-develop", "recovery-status"):
            subparser.add_argument("--recovery-permit", required=True)
        if name in ("develop", "refits", "evaluate", "recover-develop"):
            subparser.add_argument("--device", default="cuda", choices=("cuda",))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _build_parser().parse_args(argv)
    except _InvalidArguments:
        return _failure("unknown", "invalid_arguments")
    command = arguments.command
    try:
        if command == "inspect":
            report = _inspect(arguments)
        elif command == "develop":
            report = _develop(arguments)
        elif command == "status":
            report = _status(arguments)
        elif command == "freeze-selection":
            report = _freeze_selection(arguments)
        elif command == "refits":
            report = _refits(arguments)
        elif command == "evaluate":
            report = _evaluate(arguments)
        elif command == "compare":
            report = _compare(arguments)
        elif command == "report":
            report = _report(arguments)
        elif command == "recover-develop":
            report = _recover_develop(arguments)
        elif command == "recovery-status":
            report = _recovery_status(arguments)
        else:
            report = _aggregate(arguments)
    except Exception as error:
        reason_code = getattr(error, "reason_code", None) or type(error).__name__
        return _failure(command, str(reason_code))
    _emit(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
