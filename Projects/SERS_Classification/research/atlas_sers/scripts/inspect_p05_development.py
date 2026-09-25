#!/usr/bin/env python3
"""Read-only P05 development ledger inspection.

Authenticates the pinned P05 core contract and prerequisite artifacts, rebuilds
the immutable core plan without writing anything, compares its canonical digest
against the pinned plan identifier, and prints an aggregate-only public summary
of the executable development metadata ledger. It never reads spectra, imports
torch, fits a model, or authorizes training or outer evaluation. The shared
canonical serialization helper imports NumPy without loading spectral arrays.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (_REPOSITORY_ROOT / "src", _REPOSITORY_ROOT):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))


class _InvalidArguments(Exception):
    """Raised by the sanitized CLI parser for invalid command-line arguments."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _InvalidArguments(message)


def _core_run() -> Any:
    return importlib.import_module("atlas_sers.evaluation.p05_core_run")


def _ledger_module() -> Any:
    return importlib.import_module("atlas_sers.evaluation.p05_development_plan")


def _canon() -> Any:
    return importlib.import_module("atlas_sers.governance.canonical")


def _public_summary(ledger: dict[str, Any]) -> dict[str, Any]:
    """Return the aggregate whitelist: no private IDs, paths, or observations."""

    summary = ledger["summary"]
    return {
        "status": "ok",
        "command": "inspect",
        "schema_version": ledger["schema_version"],
        "contract_sha256": ledger["contract_sha256"],
        "plan_id": ledger["plan_id"],
        "ledger_id": ledger["ledger_id"],
        "contexts": summary["context_count"],
        "contexts_by_selection_mode": summary["contexts_by_selection_mode"],
        "contexts_by_phase_gate": summary["contexts_by_phase_gate"],
        "inherited_units": summary["inherited_unit_count"],
        "guard_units": summary["guard_unit_count"],
        "eligible_units": summary["eligible_unit_count"],
        "excluded_units": summary["excluded_unit_count"],
        "slots": summary["slot_count"],
        "eligible_slots": summary["eligible_slot_count"],
        "excluded_slots": summary["excluded_slot_count"],
        "maximum_masters": summary["maximum_masters_per_unit"],
        "maximum_fitting_masters": summary["maximum_fitting_masters"],
        "maximum_validation_masters": summary["maximum_validation_masters"],
        "maximum_rows": summary["maximum_rows_per_unit"],
        "maximum_sampling_capacity": summary["maximum_sampling_capacity"],
        "minimum_scheduled_optimizer_updates": summary[
            "minimum_scheduled_optimizer_updates"
        ],
        "maximum_scheduled_optimizer_updates": summary[
            "maximum_scheduled_optimizer_updates"
        ],
        "checkpoint_tensor_bytes_per_fit_max": summary[
            "checkpoint_tensor_bytes_per_fit_max"
        ],
        "checkpoint_tensor_bytes_total_eligible": summary[
            "checkpoint_tensor_bytes_total_eligible"
        ],
        "checkpoint_estimate_note": summary["checkpoint_estimate_note"],
        "execution_authorized": False,
        "arrays_loaded": False,
        "fits_started": 0,
        "outer_evaluation_authorized": False,
    }


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(_canon().canonical_json_bytes(payload).decode("utf-8") + "\n")


def _failure(reason_code: str) -> int:
    _emit({"status": "fail", "command": "inspect", "reason_code": reason_code})
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _Parser(prog="inspect_p05_development")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", default=None)
    try:
        arguments = parser.parse_args(argv)
    except _InvalidArguments:
        return _failure("invalid_arguments")

    core = _core_run()
    ledger_module = _ledger_module()
    contract_sha256 = (
        arguments.contract_sha256 or ledger_module.PINNED_CONTRACT_SHA256
    )
    try:
        project_root = Path(arguments.project_root)
        artifact_root = Path(arguments.artifact_root)
        repository_root = core._repository_root(project_root)
        core._assert_artifact_location(project_root, repository_root, artifact_root)
        contract, resolved_sha256 = core._load_contract(
            Path(arguments.contract), contract_sha256
        )
        support, _p01_run, _p04_run = core._authenticate(artifact_root, contract)
        plan = core._build_plan(support, contract, project_root)
        core._minimal_plan_checks(plan, contract)
        plan_id = _canon().sha256_bytes(_canon().canonical_json_bytes(plan))
        if plan_id != ledger_module.PINNED_PLAN_ID:
            raise core.P05CoreError("plan_authority_mismatch")
        ledger = ledger_module.build_development_ledger(
            plan=plan, support=support, contract=contract
        )
        if ledger["plan_id"] != plan_id:
            raise core.P05CoreError("ledger_plan_identity_mismatch")
        if ledger["contract_sha256"] != resolved_sha256:
            raise core.P05CoreError("ledger_contract_identity_mismatch")
        report = _public_summary(ledger)
    except core.P05CoreError as error:
        return _failure(error.reason_code)
    except ledger_module.DevelopmentLedgerError as error:
        return _failure(error.reason_code)
    except Exception:
        return _failure("inspection_failed")
    _emit(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
