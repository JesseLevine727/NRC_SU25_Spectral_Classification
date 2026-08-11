"""Command-line boundary for the P03 no-fit execution plan."""

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
            "validate",
            "execute-selection",
            "execute-selection-batch",
            "aggregate-selection",
            "execute-outer",
            "execute-outer-batch",
            "aggregate-final",
            "validate-execution",
        ),
    )
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--outer-index", type=int)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    parser.add_argument("--max-tasks", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "execute-selection":
        if args.shard_id is None or args.shard_id < 0:
            raise SystemExit("execute-selection requires a nonnegative --shard-id")
        from atlas_sers.governance.p03_execution import execute_latest_selection_shard

        path, action, run_id = execute_latest_selection_shard(
            project_root=project_root(),
            artifact_root=artifact_root(),
            shard_id=args.shard_id,
        )
        report = {
            "status": "pass",
            "run_id": run_id,
            "action": action,
            "shard_id": args.shard_id,
            "artifact_kind": path.parent.name,
        }
    elif args.command == "execute-selection-batch":
        from atlas_sers.governance.p03_execution import (
            execute_latest_selection_batch,
        )

        report = execute_latest_selection_batch(
            project_root=project_root(),
            artifact_root=artifact_root(),
            worker_index=args.worker_index,
            worker_count=args.worker_count,
            start_index=args.start_index,
            stop_index=args.stop_index,
            max_tasks=args.max_tasks,
        )
    elif args.command == "aggregate-selection":
        from atlas_sers.governance.p03_execution import aggregate_latest_selection

        path, action, run_id = aggregate_latest_selection(
            project_root=project_root(),
            artifact_root=artifact_root(),
        )
        report = {
            "status": "pass",
            "run_id": run_id,
            "action": action,
            "artifact_kind": path.parent.parent.name,
        }
    elif args.command == "execute-outer":
        if args.outer_index is None or args.outer_index < 0:
            raise SystemExit("execute-outer requires a nonnegative --outer-index")
        from atlas_sers.governance.p03_execution import execute_latest_outer_index

        path, action, run_id = execute_latest_outer_index(
            project_root=project_root(),
            artifact_root=artifact_root(),
            outer_index=args.outer_index,
        )
        report = {
            "status": "pass",
            "run_id": run_id,
            "action": action,
            "outer_index": args.outer_index,
            "artifact_kind": path.parent.parent.name,
        }
    elif args.command == "execute-outer-batch":
        from atlas_sers.governance.p03_execution import execute_latest_outer_batch

        report = execute_latest_outer_batch(
            project_root=project_root(),
            artifact_root=artifact_root(),
            worker_index=args.worker_index,
            worker_count=args.worker_count,
            start_index=args.start_index,
            stop_index=args.stop_index,
            max_tasks=args.max_tasks,
        )
    elif args.command == "aggregate-final":
        from atlas_sers.governance.p03_execution import aggregate_latest_final

        path, action, run_id = aggregate_latest_final(
            project_root=project_root(),
            artifact_root=artifact_root(),
        )
        report = {
            "status": "pass",
            "run_id": run_id,
            "action": action,
            "artifact_kind": path.parent.parent.name,
        }
    elif args.command == "validate-execution":
        from atlas_sers.governance.p03_execution import validate_latest_p03_execution

        report, path, action = validate_latest_p03_execution(
            project_root=project_root(),
            artifact_root=artifact_root(),
        )
        report = {
            **report,
            "action": action,
            "artifact_kind": path.parent.parent.name,
        }
    elif args.command == "validate":
        from atlas_sers.governance.p03 import validate_latest_p03_plan

        report = validate_latest_p03_plan(artifact_root())
    else:
        from atlas_sers.governance.p03 import execute_p03_plan

        report, _, action = execute_p03_plan(
            project_root=project_root(),
            private_root=private_data_root(),
            artifact_root=artifact_root(),
        )
        report = {
            "status": report["status"],
            "run_id": report["run_id"],
            "action": action,
            "scientific_fitting_authorized": report["scientific_fitting_authorized"],
            "diagnostics": report["diagnostics"],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
