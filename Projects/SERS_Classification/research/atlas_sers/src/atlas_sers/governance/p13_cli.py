"""Command-line boundary for locked P13 planning and execution."""

from __future__ import annotations

import argparse
import json

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
            "aggregate",
        ),
    )
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "aggregate":
        from atlas_sers.governance.p13_results import aggregate_p13

        report, _, action = aggregate_p13(
            artifact_root=artifact_root(), project_root=project_root()
        )
        output = {
            "status": report["status"],
            "run_id": report["run_id"],
            "action": action,
            "counts": report["counts"],
        }
    elif args.command == "execute-shard":
        if args.shard_index is None:
            raise SystemExit("execute-shard requires --shard-index")
        from atlas_sers.governance.p13_execution import execute_shard

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
        from atlas_sers.governance.p13_execution import execute_batch

        output = execute_batch(
            artifact_root=artifact_root(),
            project_root=project_root(),
            worker_index=args.worker_index,
            worker_count=args.worker_count,
            start_index=args.start_index,
            stop_index=args.stop_index,
        )
    elif args.command == "plan":
        from atlas_sers.governance.p13 import execute_p13_plan

        report, _, action = execute_p13_plan(
            project_root=project_root(),
            private_root=private_data_root(),
            artifact_root=artifact_root(),
        )
        output = {
            "status": report["status"],
            "run_id": report["run_id"],
            "action": action,
            "scientific_fitting_authorized": report[
                "scientific_fitting_authorized"
            ],
            "compute": report["compute"],
        }
    else:
        from atlas_sers.governance.p13 import validate_latest_p13_plan

        output = validate_latest_p13_plan(artifact_root())
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
