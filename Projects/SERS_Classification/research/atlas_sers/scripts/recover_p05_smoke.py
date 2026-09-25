#!/usr/bin/env python3
"""CLI entry point for P05-T012 bounded checkpoint recovery."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _emit(payload: Mapping[str, Any]) -> None:
    canonical = importlib.import_module("atlas_sers.governance.canonical")
    sys.stdout.write(canonical.canonical_json_bytes(dict(payload)).decode("utf-8") + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    recovery = importlib.import_module("atlas_sers.evaluation.p05_recovery")
    core = importlib.import_module("atlas_sers.evaluation.p05_core_run")
    core._configure_environment()
    parser = argparse.ArgumentParser(prog="recover_p05_smoke")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--permit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("recover")
    arguments = parser.parse_args(argv)
    permit_path = arguments.permit or str(
        Path(arguments.project_root) / "plan" / "contracts" / "p05_checkpoint_recovery.json"
    )
    try:
        if arguments.command == "preflight":
            report = recovery.preflight_recovery(
                project_root=arguments.project_root,
                artifact_root=arguments.artifact_root,
                contract_path=arguments.contract,
                plan_id=arguments.plan_id,
                contract_sha256=arguments.contract_sha256,
                permit_path=permit_path,
            )
        else:
            report = recovery.run_recovery(
                project_root=arguments.project_root,
                artifact_root=arguments.artifact_root,
                contract_path=arguments.contract,
                plan_id=arguments.plan_id,
                contract_sha256=arguments.contract_sha256,
                permit_path=permit_path,
            )
    except (recovery.P05RecoveryError, core.P05CoreError) as error:
        _emit(
            {
                "status": "fail",
                "command": arguments.command,
                "reason_code": error.reason_code,
            }
        )
        return 1
    _emit(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
