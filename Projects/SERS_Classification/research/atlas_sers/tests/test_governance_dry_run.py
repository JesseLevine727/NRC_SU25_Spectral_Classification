from __future__ import annotations

import csv
import io
from pathlib import Path

from atlas_sers.governance.dry_run import build_dry_run_outputs, dry_run_bundle_sha256
from atlas_sers.governance.registries import load_governance

PROJECT = Path(__file__).resolve().parents[1]


def test_dry_run_is_deterministic_complete_and_authorizes_no_fits() -> None:
    bundle = load_governance(PROJECT / "plan")
    arguments = {
        "protocol_version": "atlas-sers-research-master-plan-v1",
        "code_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "input_sha256": "3" * 64,
        "resource_snapshot": {
            "artifact_filesystem_total_bytes": 1000,
            "artifact_filesystem_free_bytes_at_capture": 500,
        },
    }
    first = build_dry_run_outputs(bundle, **arguments)
    second = build_dry_run_outputs(bundle, **arguments)
    assert first == second
    assert dry_run_bundle_sha256(first) == dry_run_bundle_sha256(second)
    expected_files = set(
        bundle.contracts["p00_governance_contract.json"]["required_dry_run_outputs"]
    )
    assert set(first) == expected_files
    rows = list(csv.DictReader(io.StringIO(first["expected_run_registry.csv"].decode())))
    assert len(rows) == 43
    assert all(row["fit_authorized"] == "false" for row in rows)
    assert all(row["finalized"] == "false" for row in rows)
