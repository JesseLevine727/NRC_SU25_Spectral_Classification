"""Fail-closed verification of immutable private P00 inputs."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from atlas_sers.governance.canonical import sha256_file, sha256_value

PRIVATE_PREFIX = "${ATLAS_PRIVATE_ROOT}/"


def resolve_private_path(private_root: Path, logical_path: str) -> Path:
    if not logical_path.startswith(PRIVATE_PREFIX):
        raise ValueError("Authoritative input is not rooted at ATLAS_PRIVATE_ROOT.")
    relative = logical_path.removeprefix(PRIVATE_PREFIX)
    resolved = (private_root / relative).resolve()
    try:
        resolved.relative_to(private_root.resolve())
    except ValueError as exc:
        raise ValueError("Authoritative input escapes ATLAS_PRIVATE_ROOT.") from exc
    return resolved


def _csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _npz_shape_check(path: Path, expected: list[int]) -> tuple[bool, dict[str, Any]]:
    import numpy as np

    expected_tuple = tuple(expected)
    with np.load(path, allow_pickle=False) as archive:
        shapes = {name: list(archive[name].shape) for name in archive.files}
    matches = sorted(name for name, shape in shapes.items() if tuple(shape) == expected_tuple)
    return bool(matches), {"archive_shapes": shapes, "matching_arrays": matches}


def _json_status(path: Path) -> str | None:
    value = json.loads(path.read_text())
    return value.get("status") if isinstance(value, dict) else None


def private_inputs_are_untracked(repository_root: Path, private_root: Path) -> bool:
    """Return true when the private root is external or has no indexed files."""

    try:
        relative = private_root.resolve().relative_to(repository_root.resolve())
    except ValueError:
        return True
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", relative.as_posix()],
        cwd=repository_root,
        capture_output=True,
        check=True,
    )
    return not result.stdout


def verify_authoritative_inputs(
    research_contract: dict[str, Any],
    *,
    private_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for declaration in research_contract["authoritative_inputs"]:
        logical = str(declaration["path"])
        record: dict[str, Any] = {
            "logical_path": logical,
            "expected_sha256": declaration["sha256"],
            "actual_sha256": None,
            "checks": {},
            "diagnostics": {},
            "status": "fail",
        }
        try:
            path = resolve_private_path(private_root, logical)
        except ValueError as exc:
            record["diagnostics"]["error"] = str(exc)
            records.append(record)
            continue
        exists = path.is_file()
        record["checks"]["exists"] = exists
        if not exists:
            record["diagnostics"]["error"] = "declared private input is missing"
            records.append(record)
            continue
        actual_hash = sha256_file(path)
        record["actual_sha256"] = actual_hash
        record["checks"]["sha256_matches"] = actual_hash == declaration["sha256"]
        if "expected_rows" in declaration:
            actual_rows = _csv_rows(path)
            record["diagnostics"]["actual_rows"] = actual_rows
            record["checks"]["row_count_matches"] = actual_rows == declaration["expected_rows"]
        if "expected_shape" in declaration:
            shape_ok, shape_details = _npz_shape_check(path, declaration["expected_shape"])
            record["diagnostics"].update(shape_details)
            record["checks"]["shape_matches"] = shape_ok
        if "required_status" in declaration:
            actual_status = _json_status(path)
            record["diagnostics"]["actual_status"] = actual_status
            record["checks"]["required_status_matches"] = (
                actual_status == declaration["required_status"]
            )
        record["status"] = "pass" if all(record["checks"].values()) else "fail"
        records.append(record)

    root_untracked = private_inputs_are_untracked(repository_root, private_root)
    checks = {
        "all_declared_inputs_pass": bool(records)
        and all(record["status"] == "pass" for record in records),
        "private_inputs_are_not_git_tracked": root_untracked,
    }
    bundle_state = [
        {
            "logical_path": record["logical_path"],
            "expected_sha256": record["expected_sha256"],
            "actual_sha256": record["actual_sha256"],
            "status": record["status"],
        }
        for record in records
    ]
    all_pass = all(checks.values())
    any_missing = any(not record["checks"].get("exists", False) for record in records)
    status = "pass" if all_pass else ("blocked" if any_missing else "fail")
    return {
        "schema_version": "p00-input-verification-v1",
        "status": status,
        "checks": checks,
        "inputs": records,
        "input_bundle_sha256": sha256_value(bundle_state),
    }
