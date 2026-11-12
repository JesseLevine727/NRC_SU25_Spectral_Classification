#!/usr/bin/env python3
"""Completion audit for the NATO SERS raw-data restart.

This validator checks raw-file provenance, tier accounting, common-axis integrity,
preprocessing matrices, leakage-safe split feasibility, metric tables, standalone
HTML registries, documentation, and reproducibility assets. It then freezes SHA-256
hashes for every final artifact and every script/configuration used in the restart.
"""

from __future__ import annotations

import hashlib
import json
import py_compile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY / "Workspace" / "nato_sers_raw_reexploration_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def standalone_html(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(errors="ignore").lower()
    return "<html" in text[:2000] and "</html>" in text[-2000:]


def check_source_tree(root: Path, inventory: pd.DataFrame) -> tuple[bool, list[str]]:
    observed = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    expected = sorted(inventory["relative_path"].astype(str))
    failures: list[str] = []
    if observed != expected:
        failures.append("relative file list differs")
        return False, failures
    expected_rows = inventory.set_index("relative_path")
    for relative in expected:
        path = root / relative
        row = expected_rows.loc[relative]
        if int(path.stat().st_size) != int(row["size_bytes"]):
            failures.append(f"size:{relative}")
            continue
        if sha256(path) != str(row["sha256"]):
            failures.append(f"sha256:{relative}")
    return not failures, failures[:20]


def split_audit(manifest: pd.DataFrame) -> tuple[bool, dict[str, object]]:
    details: dict[str, object] = {}
    valid = True
    tasks = {"aggregate": manifest}
    tasks.update({f"station_{s}": d for s, d in manifest.groupby("station")})
    for name, frame in tasks.items():
        n_splits = 5 if name == "aggregate" else 4
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=20260805)
        task_valid = True
        folds = 0
        for train, test in splitter.split(frame, frame["target_analyte"], frame["master_sample_id"]):
            train_groups = set(frame.iloc[train]["master_sample_id"])
            test_groups = set(frame.iloc[test]["master_sample_id"])
            task_valid &= train_groups.isdisjoint(test_groups)
            task_valid &= len(train_groups) + len(test_groups) == frame["master_sample_id"].nunique()
            folds += 1
        details[name] = {"folds": folds, "master_disjoint": bool(task_valid)}
        valid &= task_valid

    instrument_rows = pd.read_csv(
        WORKSPACE / "identifiability" / "tables" / "within_station_leave_instrument_out.csv"
    )
    reported_domains = instrument_rows[["station", "held_out_instrument"]].drop_duplicates()
    adequate_domains = instrument_rows[
        instrument_rows["n_test_classes"].eq(3) & instrument_rows["n_test_masters"].ge(15)
    ][["station", "held_out_instrument"]].drop_duplicates()
    domain_valid = True
    reconstructed_domains: set[tuple[str, str]] = set()
    for station_name, station in manifest.groupby("station"):
        station = station.reset_index(drop=True)
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20260805)
        folds = list(splitter.split(station, station["target_analyte"], station["master_sample_id"]))
        for instrument in station["instrument"].sort_values().unique():
            truth: list[str] = []
            for train_master, test_master in folds:
                train = train_master[station.iloc[train_master]["instrument"].ne(instrument).to_numpy()]
                test = test_master[station.iloc[test_master]["instrument"].eq(instrument).to_numpy()]
                if not len(test) or station.iloc[train]["target_analyte"].nunique() < station["target_analyte"].nunique():
                    continue
                train_groups = set(station.iloc[train]["master_sample_id"])
                test_groups = set(station.iloc[test]["master_sample_id"])
                domain_valid &= train_groups.isdisjoint(test_groups)
                domain_valid &= instrument not in set(station.iloc[train]["instrument"])
                domain_valid &= set(station.iloc[test]["target_analyte"]).issubset(
                    set(station.iloc[train]["target_analyte"])
                )
                truth.extend(station.iloc[test]["target_analyte"].astype(str))
            if len(set(truth)) >= 2:
                reconstructed_domains.add((str(station_name), str(instrument)))
    reported_domain_set = set(map(tuple, reported_domains.to_numpy()))
    domain_valid &= reconstructed_domains == reported_domain_set
    details["leave_instrument_out"] = {
        "reported_domains": int(len(reported_domains)),
        "adequately_supported_primary_domains": int(len(adequate_domains)),
        "held_instrument_absent_from_training": bool(domain_valid),
    }
    return bool(
        valid and domain_valid and len(reported_domains) == 17 and len(adequate_domains) == 13
    ), details


def main() -> None:
    w = WORKSPACE
    checks: dict[str, bool] = {}
    diagnostics: dict[str, object] = {}

    summary = json.loads((w / "initial_audit_summary.json").read_text())
    raw_validation = json.loads((w / "validation_report.json").read_text())
    inventory = pd.read_csv(w / "tables" / "raw_file_inventory.csv")
    recordings = pd.read_csv(w / "tables" / "recordings_manifest_fresh.csv")
    selected = pd.read_csv(w / "tables" / "tier_unique_attributable_sers.csv")
    notes_clear = pd.read_csv(w / "tables" / "tier_notes_clear_sers.csv")
    source_spectra = pd.read_csv(w / "tables" / "source_spectrum_inventory.csv")

    checks["prior_raw_validation_passes"] = raw_validation.get("status") == "pass"
    checks["raw_inventory_counts_and_bytes"] = (
        len(inventory) == 7148
        and int(inventory["size_bytes"].sum()) == 1_397_661_278
        and len(inventory) == summary["raw_files"]
        and int(inventory["size_bytes"].sum()) == summary["raw_bytes"]
    )
    source_ok, source_failures = check_source_tree(Path(summary["source_root"]), inventory)
    mirror_ok, mirror_failures = check_source_tree(Path(summary["mirror_root"]), inventory)
    checks["authoritative_source_matches_inventory_hashes"] = source_ok
    checks["project_mirror_is_byte_identical"] = mirror_ok
    diagnostics["source_hash_failures_first20"] = source_failures
    diagnostics["mirror_hash_failures_first20"] = mirror_failures

    expected_counts = {
        "expanded_log_observations": 1172,
        "named_sers_observations": 721,
        "all_parseable_sers": 626,
        "attributable_sers": 615,
        "unique_attributable_sers": 598,
        "notes_clear_sers": 500,
        "analysis_masters": 69,
    }
    checks["reported_count_flow_is_locked"] = all(summary[k] == v for k, v in expected_counts.items())
    checks["manifest_tier_counts_match"] = (
        len(recordings) == 1172
        and int(recordings["is_named_sers"].sum()) == 721
        and int(recordings["tier_all_parseable_sers"].sum()) == 626
        and int(recordings["tier_attributable_sers"].sum()) == 615
        and int(recordings["tier_unique_attributable_sers"].sum()) == 598
        and int(recordings["tier_notes_clear_sers"].sum()) == 500
        and len(selected) == 598
        and len(notes_clear) == 500
    )
    checks["selection_is_unique_attributable_and_qc_pass"] = (
        selected["observation_uid"].is_unique
        and not selected.duplicated(["instrument", "source_scan_id"]).any()
        and selected["master_sample_id"].notna().all()
        and selected["target_analyte"].notna().all()
        and selected["numeric_qc_status"].eq("pass").all()
        and selected["master_sample_id"].nunique() == 69
        and selected["instrument"].nunique() == 10
        and selected["sensor_family"].nunique() == 4
        and selected["target_analyte"].nunique() == 7
        and selected["station"].nunique() == 3
    )
    checks["source_numeric_inventory_is_complete"] = (
        len(source_spectra) == 1373
        and source_spectra["load_status"].eq("pass").all()
        and int(source_spectra["numeric_qc_status"].eq("pass").sum()) == 1367
    )

    raw = np.load(w / "arrays" / "unique_attributable_common_support_raw.npz")
    axis = raw["axis_cm1"]
    intensity = raw["intensity"]
    uid = raw["observation_uid"].astype(str)
    checks["raw_matrix_integrity"] = (
        intensity.shape == (598, 1450)
        and axis.shape == (1450,)
        and np.isfinite(intensity).all()
        and np.isfinite(axis).all()
        and np.all(np.diff(axis) > 0)
        and np.allclose(np.diff(axis), 1.0)
        and axis[0] == 400
        and axis[-1] == 1849
        and np.array_equal(uid, selected["observation_uid"].astype(str).to_numpy())
    )
    checks["common_axis_never_extrapolates"] = bool(
        (selected["effective_axis_min_cm1"] <= axis[0]).all()
        and (selected["effective_axis_max_cm1"] >= axis[-1]).all()
    )

    metadata_catalog = pd.read_csv(w / "deep_audit" / "tables" / "metadata_field_catalog_selected.csv")
    deep_summary = json.loads((w / "deep_audit" / "summary.json").read_text())
    mira1 = selected[selected["instrument"].eq("Mira-1")]
    checks["metadata_and_workbooks_audited"] = (
        len(metadata_catalog) == 44
        and deep_summary["metadata_fields_audited"] == 44
        and (w / "deep_audit" / "tables" / "workbook_sheet_audit.csv").is_file()
        and (w / "deep_audit" / "tables" / "source_notes_transcription.csv").is_file()
        and selected["instrument_serial"].notna().sum() == 486
        and len(mira1) == 23
        and mira1["system_suitability"].astype(str).str.lower().str.contains("fail").all()
    )

    candidates = np.load(w / "preprocessing" / "preprocessing_candidates.npz")
    required_candidates = {
        "minimal_minmax",
        "minimal_snv",
        "minimal_vector",
        "minimal_area",
        "despike_minmax",
        "despike_savgol_minmax",
        "rubberband_minmax",
        "asls_minmax",
        "arpls_minmax",
        "arpls_savgol_minmax",
        "arpls_savgol_snv",
        "savgol_derivative1_snv",
    }
    candidate_keys = set(candidates.files) - {"axis_cm1"}
    checks["all_prespecified_preprocessing_candidates_exist"] = required_candidates == candidate_keys
    checks["all_preprocessing_matrices_finite_and_aligned"] = all(
        candidates[k].shape == (598, 1450) and np.isfinite(candidates[k]).all()
        for k in required_candidates
    ) and np.array_equal(candidates["axis_cm1"], axis)
    prep_metrics = pd.read_csv(w / "preprocessing" / "tables" / "preprocessing_metrics.csv")
    range_metrics = pd.read_csv(w / "preprocessing" / "tables" / "range_sensitivity.csv")
    prep_instrument = pd.read_csv(
        w / "preprocessing" / "tables" / "preprocessing_metrics_by_instrument.csv"
    )
    checks["preprocessing_metrics_are_complete"] = (
        len(prep_metrics) == 12
        and set(prep_metrics["candidate"]) == required_candidates
        and prep_metrics.select_dtypes(include=[np.number]).notna().all().all()
        and len(range_metrics) > 0
        and len(prep_instrument) >= 10 * 12
    )

    split_ok, split_details = split_audit(selected)
    checks["master_grouped_and_unseen_instrument_splits_are_feasible_and_disjoint"] = split_ok
    diagnostics["split_audit"] = split_details

    classical = pd.read_csv(w / "benchmarks" / "tables" / "closed_set_fold_metrics.csv")
    station = pd.read_csv(w / "identifiability" / "tables" / "within_station_fold_metrics.csv")
    shift = pd.read_csv(w / "identifiability" / "tables" / "within_station_leave_instrument_out.csv")
    cross_station = pd.read_csv(w / "identifiability" / "tables" / "cross_station_binary.csv")
    cnn_aggregate = pd.read_csv(w / "benchmarks" / "cnn" / "fold_metrics.csv")
    cnn_station = pd.read_csv(w / "identifiability" / "cnn" / "fold_metrics.csv")
    quality = pd.read_csv(w / "quality_sensitivity" / "fold_metrics.csv")
    metric_tables = [classical, station, shift, cross_station, cnn_aggregate, cnn_station, quality]
    checks["metric_tables_are_nonempty_and_finite"] = all(
        len(d) > 0 and d.select_dtypes(include=[np.number]).notna().all().all() for d in metric_tables
    )
    checks["classical_neural_and_shift_models_present"] = (
        {"dummy_prior", "nearest_centroid", "pca_lda", "ridge", "rbf_svm", "random_forest", "extra_trees"}
        .issubset(set(classical["model"]))
        and len(cnn_aggregate) == 5
        and set(cnn_station["station"]) == {"cwa", "pills", "surfaces"}
        and shift[["station", "held_out_instrument"]].drop_duplicates().shape[0] == 17
        and shift[
            shift["n_test_classes"].eq(3) & shift["n_test_masters"].ge(15)
        ][["station", "held_out_instrument"]].drop_duplicates().shape[0] == 13
    )

    registries = [
        (w / "html_artifact_registry.json", w / "html", 35),
        (w / "preprocessing" / "html_artifact_registry.json", w / "preprocessing" / "html", 17),
        (w / "benchmarks" / "html_artifact_registry.json", w / "benchmarks" / "html", 5),
        (w / "deep_audit" / "html_artifact_registry.json", w / "deep_audit" / "html", 54),
        (w / "identifiability" / "html_artifact_registry.json", w / "identifiability" / "html", 6),
    ]
    html_count = 0
    html_registry_ok = True
    for registry_path, directory, expected in registries:
        registry = json.loads(registry_path.read_text())
        paths = [directory / item["file"] for item in registry]
        html_count += len(paths)
        html_registry_ok &= len(paths) == expected and len(paths) == len(set(paths))
        html_registry_ok &= all(standalone_html(path) for path in paths)
    checks["all_117_registered_plots_are_standalone_html"] = html_registry_ok and html_count == 117
    required_html = [
        w / "ALL_RESULTS.html",
        w / "index.html",
        w / "preprocessing" / "index.html",
        w / "benchmarks" / "index.html",
        w / "deep_audit" / "index.html",
        w / "identifiability" / "index.html",
        w / "benchmarks" / "cnn" / "training_history.html",
    ]
    checks["all_indexes_and_cnn_history_are_standalone_html"] = all(
        standalone_html(path) for path in required_html
    )

    literature = pd.read_csv(w / "literature_matrix.csv")
    project_scores = pd.read_csv(w / "project_option_scores.csv")
    protocol = REPOSITORY / "docs" / "NATO_SERS_ACQUISITION_SHIFT_PROTOCOL_V1.md"
    final_report = w / "FINAL_REPORT.md"
    checks["literature_and_project_decision_are_documented"] = (
        len(literature) >= 12
        and literature["url"].str.startswith("http").all()
        and len(project_scores) >= 9
        and "PRIMARY" in set(project_scores["recommendation"])
        and "FALLBACK" in set(project_scores["recommendation"])
        and protocol.stat().st_size > 10_000
        and final_report.stat().st_size > 20_000
    )

    scripts = sorted(REPOSITORY.glob("scripts/*nato_sers*.py"))
    configs = sorted(REPOSITORY.glob("configs/nato_sers*.json"))
    compile_ok = True
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except py_compile.PyCompileError:
            compile_ok = False
    json_ok = True
    for config in configs:
        try:
            json.loads(config.read_text())
        except json.JSONDecodeError:
            json_ok = False
    checks["all_nato_sers_scripts_compile_and_configs_parse"] = compile_ok and json_ok

    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "diagnostics": diagnostics,
        "source_files_rehashed_per_tree": len(inventory),
        "registered_standalone_html_plots": html_count,
        "workspace": str(w),
    }

    hash_paths = [p for p in w.rglob("*") if p.is_file()]
    hash_paths += scripts + configs + [protocol]
    excluded = {w / "FINAL_ARTIFACT_HASHES.json", w / "FINAL_VALIDATION_REPORT.json"}
    hash_paths = sorted({p.resolve() for p in hash_paths if p.resolve() not in {q.resolve() for q in excluded}})
    hashes = {
        str(path.relative_to(REPOSITORY)): sha256(path)
        for path in hash_paths
    }
    (w / "FINAL_ARTIFACT_HASHES.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n"
    )
    report["hashed_artifacts"] = len(hashes)
    (w / "FINAL_VALIDATION_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
