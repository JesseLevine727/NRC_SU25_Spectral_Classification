"""Command-line entry points for the ATLAS P01 data and representation freeze."""

from __future__ import annotations

import argparse
import json

from atlas_sers.governance.registries import load_governance, validate_governance
from atlas_sers.paths import artifact_root, native_data_root, private_data_root, project_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ATLAS P01 data-freeze command")
    parser.add_argument("command", choices=("audit", "dry-run", "build", "validate"))
    return parser


def _audit() -> dict[str, object]:
    root = project_root()
    bundle = load_governance(root / "plan")
    governance = validate_governance(bundle)
    contract = bundle.contracts["p01_governance_contract.json"]
    registered_figures = {
        row["figure_id"] for row in bundle.rows("figure_registry.csv") if row["phase"] == "P01"
    }
    declared_figures = set(contract["required_figures"])
    checks = {
        "governance_passes": governance["status"] == "pass",
        "eight_representations_declared": len(contract["representations"]) == 8,
        "representation_ids_unique": len(
            {row["representation_id"] for row in contract["representations"]}
        )
        == 8,
        "figure_registry_matches_contract": registered_figures == declared_figures,
        "model_and_split_operations_prohibited": not contract["model_fitting_authorized"]
        and not contract["split_construction_authorized"],
    }
    return {
        "schema_version": "p01-structural-audit-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "governance_errors": governance["errors"],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = project_root()
    if args.command == "audit":
        report = _audit()
    elif args.command == "dry-run":
        from atlas_sers.governance.p01 import p01_dry_run

        report = p01_dry_run(root)
    elif args.command == "validate":
        from atlas_sers.governance.p01 import validate_latest_p01

        report = validate_latest_p01(artifact_root())
    else:
        from atlas_sers.governance.p01 import execute_p01

        report, _, action = execute_p01(
            project_root=root,
            private_root=private_data_root(),
            native_root=native_data_root(),
            artifact_root=artifact_root(),
        )
        report = {
            "status": report["status"],
            "run_id": report["run_id"],
            "action": action,
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return {"pass": 0, "blocked": 2, "fail": 1}[str(report["status"])]


if __name__ == "__main__":
    raise SystemExit(main())
