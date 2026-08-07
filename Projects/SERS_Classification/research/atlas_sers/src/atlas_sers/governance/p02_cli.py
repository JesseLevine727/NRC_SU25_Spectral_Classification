"""Command-line entry points for the ATLAS P02 evaluation-design freeze."""

from __future__ import annotations

import argparse
import json

from atlas_sers.governance.registries import load_governance, validate_governance
from atlas_sers.paths import artifact_root, private_data_root, project_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ATLAS P02 evaluation-freeze command")
    parser.add_argument("command", choices=("audit", "dry-run", "build", "validate"))
    return parser


def _audit() -> dict[str, object]:
    root = project_root()
    bundle = load_governance(root / "plan")
    governance = validate_governance(bundle)
    split = bundle.contracts["split_contract.json"]
    contract = bundle.contracts["p02_governance_contract.json"]
    checks = {
        "governance_passes": governance["status"] == "pass",
        "five_repeat_seeds": len(split["outer_repeat_seeds"]) == 5,
        "four_outer_folds": split["outer_folds_per_station"] == 4,
        "thirteen_primary_domains": len(split["primary_domain_eligibility"]["domains"])
        == 13,
        "qc_library_size_frozen": contract["qc_gate_enumeration"][
            "expected_candidates"
        ]
        == 124,
        "predictive_model_fitting_prohibited": not contract[
            "predictive_model_fitting_authorized"
        ],
        "figures_f10_f11": set(contract["required_figures"]) == {"F10", "F11"},
    }
    return {
        "schema_version": "p02-structural-audit-v1",
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
        from atlas_sers.governance.p02 import p02_dry_run

        report = p02_dry_run(root)
    elif args.command == "validate":
        from atlas_sers.governance.p02 import validate_latest_p02

        report = validate_latest_p02(artifact_root())
    else:
        from atlas_sers.governance.p02 import execute_p02

        report, _, action = execute_p02(
            project_root=root,
            private_root=private_data_root(),
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
