#!/usr/bin/env python3
"""Validate the provenance, matrices, tiers, and HTML artifacts of the fresh audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=repository / "Workspace" / "nato_sers_raw_reexploration_v1",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    tables = workspace / "tables"
    summary = json.loads((workspace / "initial_audit_summary.json").read_text())
    inventory = pd.read_csv(tables / "raw_file_inventory.csv")
    manifest = pd.read_csv(tables / "recordings_manifest_fresh.csv")
    selected = pd.read_csv(tables / "analysis_manifest.csv")
    source = pd.read_csv(tables / "source_spectrum_inventory.csv")
    master = pd.read_csv(tables / "master_embeddings_and_clusters.csv")
    bundle = np.load(workspace / "arrays" / "unique_attributable_common_support_raw.npz")
    axis = bundle["axis_cm1"]
    matrix = bundle["intensity"]
    uids = bundle["observation_uid"].astype(str)

    checks: dict[str, bool] = {}
    checks["inventory_count_matches_summary"] = len(inventory) == summary["raw_files"]
    checks["inventory_bytes_match_summary"] = int(inventory.size_bytes.sum()) == summary["raw_bytes"]
    checks["recording_count_matches_summary"] = len(manifest) == summary["expanded_log_observations"]
    for column, key in (
        ("tier_all_parseable_sers", "all_parseable_sers"),
        ("tier_attributable_sers", "attributable_sers"),
        ("tier_unique_attributable_sers", "unique_attributable_sers"),
        ("tier_notes_clear_sers", "notes_clear_sers"),
    ):
        checks[f"{column}_count"] = int(manifest[column].sum()) == summary[key]
    checks["tiers_are_nested"] = bool(
        (~manifest.tier_notes_clear_sers | manifest.tier_unique_attributable_sers).all()
        and (~manifest.tier_unique_attributable_sers | manifest.tier_attributable_sers).all()
        and (~manifest.tier_attributable_sers | manifest.tier_all_parseable_sers).all()
    )
    checks["matrix_shape"] = matrix.shape == (len(selected), len(axis))
    checks["matrix_finite"] = bool(np.isfinite(matrix).all())
    checks["matrix_uid_order"] = np.array_equal(uids, selected.observation_uid.astype(str).to_numpy())
    checks["one_selected_row_per_source"] = not selected.duplicated(["instrument", "source_scan_id"]).any()
    checks["no_common_grid_extrapolation"] = bool(
        (selected.effective_axis_min_cm1 <= axis[0]).all()
        and (selected.effective_axis_max_cm1 >= axis[-1]).all()
    )
    checks["strict_axis"] = bool(np.all(np.diff(axis) > 0) and np.allclose(np.diff(axis), 1.0))
    checks["all_selected_source_qc_pass"] = bool(selected.numeric_qc_status.eq("pass").all())
    checks["source_rows_have_load_result"] = bool(source.load_status.notna().all())
    checks["master_count"] = len(master) == summary["analysis_masters"] == selected.master_sample_id.nunique()
    checks["master_counts_sum_to_spectra"] = int(master.n_spectra.sum()) == len(selected)

    registry = json.loads((workspace / "html_artifact_registry.json").read_text())
    html_paths = [workspace / "html" / item["file"] for item in registry]
    checks["html_registry_unique"] = len(html_paths) == len(set(html_paths))
    checks["all_html_exist"] = all(path.is_file() for path in html_paths)
    checks["all_html_are_standalone_documents"] = all(
        "<html" in path.read_text(errors="ignore")[:1000].lower()
        and "</html>" in path.read_text(errors="ignore")[-1000:].lower()
        for path in html_paths
    )
    checks["index_exists"] = (workspace / "index.html").is_file()

    artifact_hashes = {}
    for path in sorted(item for item in workspace.rglob("*") if item.is_file()):
        if path.name in {"validation_report.json", "artifact_hashes.json"}:
            continue
        artifact_hashes[str(path.relative_to(workspace))] = sha256(path)
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "checked_artifacts": len(artifact_hashes),
        "html_plots": len(registry),
    }
    (workspace / "artifact_hashes.json").write_text(json.dumps(artifact_hashes, indent=2, sort_keys=True) + "\n")
    (workspace / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
