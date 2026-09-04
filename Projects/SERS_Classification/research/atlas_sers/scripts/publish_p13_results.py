#!/usr/bin/env python3
"""Publish aggregate-only P13 tables and generate F45--F47."""

from __future__ import annotations

import json

import pandas as pd

from atlas_sers.governance.canonical import canonical_json_bytes, sha256_file
from atlas_sers.paths import artifact_root, project_root
from atlas_sers.visualization.p13_figures import generate_p13_figures

PUBLIC_TABLES = (
    "domain_metrics.csv",
    "interval_table.csv",
    "domain_claims.csv",
    "class_cell_claims.csv",
    "substrate_claims.csv",
    "preprocessing_sensitivity.csv",
    "procedure_comparison.csv",
    "crossover_effects.csv",
    "field_log_results.csv",
    "failure_table.csv",
)
PROHIBITED_COLUMNS = {
    "observation_uid",
    "source_observation_uid",
    "master_sample_id",
    "source_logical_id",
    "source_primary_reference",
}
PROHIBITED_BYTES = (
    bytes((47, 104, 111, 109, 101, 47)),
    bytes((92, 117, 115, 101, 114, 115, 92)),
    b"github_pat_",
    b"ghp_",
    b"gho_",
)


def main() -> int:
    project = project_root()
    artifacts = artifact_root()
    latest = json.loads((artifacts / "p13/LATEST.json").read_text())
    aggregate = (
        artifacts
        / "p13/runs"
        / str(latest["run_id"])
        / "aggregation/shards/shard-000000"
    )
    report = json.loads((aggregate / "P13_EXECUTION_VALIDATION_REPORT.json").read_text())
    if report["status"] != "pass":
        raise RuntimeError("Only a passing P13 aggregate may be published.")
    results = project / "results/p13_portability"
    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name in PUBLIC_TABLES:
        frame = pd.read_csv(aggregate / name, low_memory=False)
        if PROHIBITED_COLUMNS & set(frame):
            raise ValueError(f"Private identifier field entered public P13 table {name}.")
        destination = tables / name
        destination.write_bytes((aggregate / name).read_bytes())
        lowered = destination.read_bytes().lower()
        if any(token in lowered for token in PROHIBITED_BYTES):
            raise ValueError(f"Private path or credential pattern entered {name}.")
        hashes[f"tables/{name}"] = sha256_file(destination)
    figures = generate_p13_figures(results_root=results, plan_root=project / "plan")
    figures.to_csv(
        results / "p13_figure_manifest.csv",
        index=False,
        lineterminator="\n",
    )
    hashes["p13_figure_manifest.csv"] = sha256_file(
        results / "p13_figure_manifest.csv"
    )
    summary = {
        "schema_version": "nato-sers-p13-public-release-v1",
        "protocol_version": "nato-sers-p13-v1-locked",
        "run_id": latest["run_id"],
        "execution_protected_state_sha256": latest[
            "execution_protected_state_sha256"
        ],
        "aggregation_protected_state_sha256": latest[
            "aggregation_protected_state_sha256"
        ],
        "private_execution_validation_report_sha256": sha256_file(
            aggregate / "P13_EXECUTION_VALIDATION_REPORT.json"
        ),
        "public_files": hashes,
        "privacy_boundary": "aggregate tables only; no observation or master identifiers",
    }
    (results / "release_manifest.json").write_bytes(
        canonical_json_bytes(summary, pretty=True)
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "run_id": latest["run_id"],
                "tables": len(PUBLIC_TABLES),
                "figures": len(figures),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
