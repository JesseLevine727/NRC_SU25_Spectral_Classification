"""Command-line entry points for P00 structural audit and definitive dry run."""

from __future__ import annotations

import argparse
import json

from atlas_sers.governance.p00 import execute_p00
from atlas_sers.governance.registries import load_governance, validate_governance
from atlas_sers.paths import artifact_root, private_data_root, project_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ATLAS P00 no-training governance command")
    parser.add_argument("command", choices=("audit", "dry-run"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = project_root()
    if args.command == "audit":
        report = validate_governance(load_governance(root / "plan"))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "pass" else 1
    report, _, action = execute_p00(
        project_root=root,
        private_root=private_data_root(),
        artifact_root=artifact_root(),
    )
    print(
        json.dumps(
            {"action": action, "run_id": report["run_id"], "status": report["status"]},
            indent=2,
            sort_keys=True,
        )
    )
    return {"pass": 0, "blocked": 2, "fail": 1}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
