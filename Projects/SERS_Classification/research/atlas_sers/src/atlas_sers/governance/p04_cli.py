"""Command-line boundary for locked P04 planning and execution."""

from __future__ import annotations

import argparse

from atlas_sers.governance.canonical import canonical_json_bytes
from atlas_sers.paths import artifact_root, private_data_root, project_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "validate-plan",
            "execute-shard",
            "execute-batch",
            "freeze-development",
            "aggregate",
            "compare",
        ),
    )
    parser.add_argument("--shard-index", type=int)
    parser.add_argument(
        "--phase",
        choices=("development", "held_evaluation", "all"),
        default="development",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        from atlas_sers.governance.p04 import execute_p04_plan

        report, _, action = execute_p04_plan(
            project_root=project_root(),
            private_root=private_data_root(),
            artifact_root=artifact_root(),
        )
        output = {
            "status": report["status"],
            "run_id": report["run_id"],
            "action": action,
            "counts": report["counts"],
            "architecture": report["architecture"],
        }
    elif args.command == "validate-plan":
        from atlas_sers.governance.p04 import validate_latest_p04_plan

        output = validate_latest_p04_plan(artifact_root())
    elif args.command == "execute-shard":
        if args.shard_index is None:
            raise SystemExit("execute-shard requires --shard-index")
        from atlas_sers.governance.p04_execution import execute_shard

        _, action, run_id = execute_shard(
            artifact_root=artifact_root(),
            project_root=project_root(),
            shard_index=args.shard_index,
        )
        output = {
            "status": "pass",
            "run_id": run_id,
            "shard_index": args.shard_index,
            "action": action,
        }
    elif args.command == "execute-batch":
        from atlas_sers.governance.p04_execution import execute_batch

        output = execute_batch(
            artifact_root=artifact_root(),
            project_root=project_root(),
            phase=args.phase,
            start_index=args.start_index,
            stop_index=args.stop_index,
        )
    elif args.command == "freeze-development":
        from atlas_sers.governance.p04_execution import freeze_development

        output = freeze_development(
            artifact_root=artifact_root(), project_root=project_root()
        )
        output["status"] = output["g2_status"]
    elif args.command == "aggregate":
        from atlas_sers.governance.p04_results import aggregate_p04

        output = aggregate_p04(
            artifact_root=artifact_root(), project_root=project_root()
        )
    else:
        from atlas_sers.governance.p04_reporting import freeze_p04_comparison

        output = freeze_p04_comparison(
            artifact_root=artifact_root(), project_root=project_root()
        )
    print(canonical_json_bytes(output, pretty=True).decode(), end="")
    return 0 if output["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
