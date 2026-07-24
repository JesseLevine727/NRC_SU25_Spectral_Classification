#!/usr/bin/env python3
"""Validate the frozen SERS representation-baseline bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import sers_baseline_common as common


EXPECTED_ROWS = {
    "ae_search_fold_metrics.csv": 400,
    "dae_search_fold_metrics.csv": 200,
    "control_metrics.csv": 15,
    "outer_fold_metrics.csv": 360,
    "outer_fold_predictions.csv": 28_704,
    "outer_fold_reconstruction_metrics.csv": 14_352,
    "outer_fold_corruption_metrics.csv": 3_780,
    "poster_metrics.csv": 108,
    "poster_predictions.csv": 7_425,
    "poster_reconstruction_metrics.csv": 3_300,
    "poster_corruption_metrics.csv": 1_008,
    "domain_transfer_metrics.csv": 1_008,
    "domain_transfer_predictions.csv": 79_056,
    "domain_transfer_reconstruction_metrics.csv": 26_352,
    "domain_transfer_corruption_metrics.csv": 7_056,
}

EXPECTED_REGISTRIES = {
    "outer_fold_run_registry.json": (225, 150, 150),
    "poster_run_registry.json": (72, 72, 72),
    "domain_transfer_run_registry.json": (504, 504, 504),
}

EXPECTED_BINARY_COUNTS = {
    "checkpoints": 726,
    "embeddings": 801,
    "reconstructions": 564,
}

EXPECTED_COHORT_ROWS = {
    "strict_core": 598,
    "quality_pass": 500,
    "field_quality_stress": 98,
}

HASH_EXCLUSIONS = {
    "artifact_hashes.json",
    "validation_report.json",
    "clean_rebuild_comparison.json",
}


class Audit:
    """Collect validation outcomes and fail with a complete report."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(
        self,
        condition: bool,
        name: str,
        details: Any = None,
    ) -> None:
        record: dict[str, Any] = {
            "name": name,
            "status": "pass" if bool(condition) else "fail",
        }
        if details is not None:
            record["details"] = details
        self.checks.append(record)

    def run(self, name: str, function: Any) -> None:
        try:
            details = function()
            self.check(True, name, details)
        except Exception as exc:  # report all independent integrity failures
            self.check(False, name, f"{type(exc).__name__}: {exc}")

    @property
    def passed(self) -> bool:
        return all(item["status"] == "pass" for item in self.checks)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def duplicate_count(frame: pd.DataFrame, columns: list[str]) -> int:
    require_columns(frame, columns, "duplicate-key frame")
    return int(frame.duplicated(columns, keep=False).sum())


def validate_required_artifacts(
    audit: Audit,
    output_dir: Path,
    protocol_path: Path,
) -> None:
    protocol = load_json(protocol_path)
    required = protocol["required_artifacts"]
    missing_records = [
        name for name in required["records"] if not (output_dir / name).is_file()
    ]
    audit.check(not missing_records, "required record files", missing_records)
    missing_directories = [
        name
        for name in required["binary_directories"]
        if not (output_dir / name).is_dir()
    ]
    if not (output_dir / required["figure_directory"]).is_dir():
        missing_directories.append(required["figure_directory"])
    audit.check(
        not missing_directories,
        "required artifact directories",
        missing_directories,
    )


def validate_json_files(audit: Audit, output_dir: Path) -> None:
    failures: list[str] = []
    for path in sorted(output_dir.rglob("*.json")):
        if "search_cache" in path.parts or "run_cache" in path.parts:
            continue
        try:
            load_json(path)
        except Exception as exc:
            failures.append(f"{path.relative_to(output_dir)}: {exc}")
    audit.check(not failures, "strict JSON parsing", failures)


def validate_hash_catalog(audit: Audit, output_dir: Path) -> None:
    catalog = load_json(output_dir / "artifact_hashes.json")
    failures: list[str] = []
    for relative, expected in catalog.items():
        path = output_dir / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = common.sha256_file(path)
        if actual != expected:
            failures.append(
                f"hash mismatch: {relative}: expected {expected}, got {actual}"
            )
    actual_files = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name not in HASH_EXCLUSIONS
        and "search_cache" not in path.parts
        and "run_cache" not in path.parts
    }
    uncatalogued = sorted(actual_files - set(catalog))
    stale = sorted(set(catalog) - actual_files)
    failures.extend(f"uncatalogued: {name}" for name in uncatalogued)
    failures.extend(f"stale catalog entry: {name}" for name in stale)
    audit.check(
        not failures,
        "artifact SHA-256 catalog",
        {"entries": len(catalog), "failures": failures},
    )


def validate_protocol_and_selection(
    audit: Audit,
    output_dir: Path,
    protocol_path: Path,
) -> None:
    protocol = load_json(protocol_path)
    copied_protocol = load_json(output_dir / "predeclared_protocol.json")
    selected = load_json(output_dir / "selected_configurations.json")
    version = load_json(output_dir / "dataset_version.json")
    decisions = load_json(output_dir / "final_decisions.json")

    audit.check(
        protocol == copied_protocol,
        "predeclared protocol is copied exactly",
    )
    audit.check(
        copied_protocol["protocol_version"] == common.PROTOCOL_VERSION,
        "protocol version",
        copied_protocol["protocol_version"],
    )
    audit.check(
        copied_protocol["terminal_scope"]["preprocessing_may_be_reopened"]
        is False,
        "preprocessing remains closed",
    )
    audit.check(
        selected["selection_closed"] is True
        and selected["outer_test_used"] is False
        and selected["field_quality_stress_used"] is False
        and selected["poster_used"] is False
        and selected["selection_data"] == "NATO nested inner validation only",
        "selection leakage flags",
        {
            key: selected[key]
            for key in (
                "selection_closed",
                "outer_test_used",
                "field_quality_stress_used",
                "poster_used",
                "selection_data",
            )
        },
    )
    audit.check(
        version["selection_closed"] is True
        and version["vae_models_run"] is False,
        "baseline terminates before VAE models",
    )
    audit.check(
        decisions["standard_vae_starting_point"]
        == {
            "representation": "arpls_minmax",
            "channels": [8, 16],
            "bottleneck_dimension": 64,
            "reconstruction_loss": "spectral_composite",
            "clean_curriculum": "clean",
            "denoising_comparator_curriculum": "mixed_uniform",
            "unchanged_from_baseline_selection": True,
        },
        "frozen next-goal starting point",
    )

    ae = selected["autoencoders"]
    dae = selected["denoising_autoencoders"]
    audit.check(
        set(ae) == {"minimal_minmax", "arpls_minmax"}
        and set(dae) == {"minimal_minmax", "arpls_minmax"},
        "selected reconstructive representations",
    )
    audit.check(
        all(item["eligible_to_advance"] is False for item in ae.values()),
        "all clean AEs fail at least one gate",
    )
    audit.check(
        dae["arpls_minmax"]["identifier"]
        == "c8x16_z64_spectral_composite_mixed_uniform",
        "advancing arPLS DAE configuration",
    )

    prohibited = tuple(
        value.lower()
        for value in copied_protocol["terminal_scope"]["prohibited_models"]
    )
    model_text = " ".join(
        path.read_text(errors="replace").lower()
        for path in (
            output_dir / "outer_fold_metrics.csv",
            output_dir / "poster_metrics.csv",
            output_dir / "domain_transfer_metrics.csv",
        )
    )
    audit.check(
        not any(value in model_text for value in prohibited),
        "no prohibited VAE-family model output",
    )


def validate_preprocessing(
    audit: Audit,
    nato_bundle: Path,
) -> None:
    dataset = common.load_nato_dataset(nato_bundle)
    audit.check(
        len(dataset.observation_uid) == 598,
        "immutable NATO core loads with verified hashes",
        {
            "rows": len(dataset.observation_uid),
            "axis_points": len(dataset.axis_cm1),
            "representations": sorted(dataset.representations),
        },
    )


def validate_table_sizes(audit: Audit, output_dir: Path) -> None:
    for name, expected in EXPECTED_ROWS.items():
        frame = pd.read_csv(output_dir / name)
        audit.check(
            len(frame) == expected,
            f"{name} row count",
            {"expected": expected, "actual": len(frame)},
        )


def validate_selection_search(audit: Audit, output_dir: Path) -> None:
    ae = pd.read_csv(output_dir / "ae_search_fold_metrics.csv")
    dae = pd.read_csv(output_dir / "dae_search_fold_metrics.csv")
    expected = {
        "ae": {
            ("strict_core", "minimal_minmax"): (160, 8),
            ("strict_core", "arpls_minmax"): (160, 8),
            ("quality_pass", "minimal_minmax"): (40, 2),
            ("quality_pass", "arpls_minmax"): (40, 2),
        },
        "dae": {
            ("strict_core", "minimal_minmax"): (60, 3),
            ("strict_core", "arpls_minmax"): (60, 3),
            ("quality_pass", "minimal_minmax"): (40, 2),
            ("quality_pass", "arpls_minmax"): (40, 2),
        },
    }
    for family, frame in (("ae", ae), ("dae", dae)):
        failures: list[str] = []
        for (subset, representation), (rows, configurations) in expected[
            family
        ].items():
            part = frame[
                (frame["subset"] == subset)
                & (frame["representation"] == representation)
            ]
            if len(part) != rows:
                failures.append(
                    f"{subset}/{representation}: {len(part)} != {rows}"
                )
            if part["configuration"].nunique() != configurations:
                failures.append(
                    f"{subset}/{representation}: "
                    f"{part['configuration'].nunique()} configs != "
                    f"{configurations}"
                )
            counts = part.groupby("configuration").size()
            if not (counts == 20).all():
                failures.append(
                    f"{subset}/{representation}: not 20 nested folds/config"
                )
        audit.check(
            not failures,
            f"{family.upper()} nested-search coverage",
            failures,
        )
        audit.check(
            duplicate_count(frame, ["run_identifier"]) == 0,
            f"{family.upper()} search run IDs unique",
        )


def validate_controls(audit: Audit, output_dir: Path) -> None:
    frame = pd.read_csv(output_dir / "control_metrics.csv")
    permutation = frame[frame["model_family"] == "negative_control"]
    identity = frame[frame["model_family"] == "identity_control"]
    audit.check(
        len(permutation) == 5
        and abs(permutation["balanced_accuracy_supported"].mean() - 1 / 7)
        < 0.03,
        "group-label permutation is near chance",
        float(permutation["balanced_accuracy_supported"].mean()),
    )
    audit.check(
        len(identity) == 10
        and np.allclose(identity["reconstruction_mse"], 0.0)
        and np.allclose(
            identity["reconstruction_median_row_correlation"], 1.0
        )
        and np.allclose(identity["repeatable_peak_recall"], 1.0),
        "identity reconstruction control",
    )


def validate_metric_prediction_alignment(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    extra_keys: list[str],
) -> list[str]:
    keys = [
        *extra_keys,
        "scenario",
        "model_family",
        "model",
        "representation",
        "seed",
    ]
    failures: list[str] = []
    if duplicate_count(predictions, [*keys, "observation_uid"]):
        failures.append("duplicate per-run observation predictions")
    counts = (
        predictions.groupby(keys, dropna=False)
        .size()
        .rename("prediction_rows")
        .reset_index()
    )
    merged = metrics.merge(counts, on=keys, how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        failures.append(
            f"metric/prediction run mismatch: "
            f"{merged['_merge'].value_counts().to_dict()}"
        )
    both = merged[merged["_merge"] == "both"]
    if not (both["n_test"] == both["prediction_rows"]).all():
        failures.append("n_test does not equal per-run prediction rows")
    return failures


def validate_outer(audit: Audit, output_dir: Path) -> None:
    metrics = pd.read_csv(output_dir / "outer_fold_metrics.csv")
    predictions = pd.read_csv(output_dir / "outer_fold_predictions.csv")
    failures = validate_metric_prediction_alignment(metrics, predictions, [])
    scenario_counts = metrics.groupby("scenario").size()
    if len(scenario_counts) != 15 or not (scenario_counts == 24).all():
        failures.append("expected 15 scenarios with 24 metric rows each")
    if set(metrics["outer_fold"].astype(int)) != set(range(5)):
        failures.append("outer folds are not exactly 0..4")

    predictions["cohort"] = predictions["scenario"].str.extract(
        r"test_(.+)$"
    )[0]
    descriptor = [
        "cohort",
        "model_family",
        "model",
        "representation",
        "seed",
    ]
    coverage = (
        predictions.groupby(descriptor)["observation_uid"].nunique().reset_index()
    )
    for cohort, expected in EXPECTED_COHORT_ROWS.items():
        values = coverage.loc[
            coverage["cohort"] == cohort, "observation_uid"
        ]
        if values.empty or not (values == expected).all():
            failures.append(
                f"{cohort} model coverage is not exactly {expected}: "
                f"{sorted(values.unique().tolist())}"
            )
    audit.check(
        not failures,
        "sealed outer-fold coverage and prediction alignment",
        failures,
    )


def validate_poster(audit: Audit, output_dir: Path) -> None:
    metrics = pd.read_csv(output_dir / "poster_metrics.csv")
    predictions = pd.read_csv(output_dir / "poster_predictions.csv")
    failures = validate_metric_prediction_alignment(
        metrics,
        predictions,
        ["heldout_substrate_family"],
    )
    families = {"Ag", "Au", "PICO", "pSERS"}
    if set(metrics["heldout_substrate_family"]) != families:
        failures.append("unexpected poster substrate families")
    scenario_counts = metrics.groupby("scenario").size()
    if len(scenario_counts) != 4 or not (scenario_counts == 27).all():
        failures.append("expected four poster scenarios with 27 rows each")
    if predictions["observation_uid"].nunique() != 275:
        failures.append("poster predictions do not cover 275 chemical rows")
    limitation = load_json(output_dir / "poster_evaluation_limitation.json")
    if limitation.get("physical_preparation_ids_available") is not False:
        failures.append("poster preparation-independence limit is missing")
    audit.check(
        not failures,
        "poster transfer coverage and limitation",
        failures,
    )


def validate_domain(audit: Audit, output_dir: Path) -> None:
    metrics = pd.read_csv(output_dir / "domain_transfer_metrics.csv")
    predictions = pd.read_csv(output_dir / "domain_transfer_predictions.csv")
    extra = [
        "evaluation_subset",
        "domain_protocol",
        "domain_type",
        "heldout_domain",
    ]
    failures = validate_metric_prediction_alignment(metrics, predictions, extra)
    scenarios = metrics.groupby("scenario").size()
    if len(scenarios) != 56 or not (scenarios == 18).all():
        failures.append("expected 56 domain scenarios with 18 metric rows each")
    expected_grid = (
        metrics[
            [
                "evaluation_subset",
                "domain_protocol",
                "domain_type",
                "heldout_domain",
            ]
        ]
        .drop_duplicates()
        .groupby(
            ["evaluation_subset", "domain_protocol", "domain_type"]
        )
        .size()
    )
    for subset in ("strict_core", "quality_pass"):
        for protocol in ("domain_only", "domain_and_sample"):
            if expected_grid.get((subset, protocol, "instrument"), 0) != 10:
                failures.append(f"{subset}/{protocol}: not 10 instruments")
            if expected_grid.get((subset, protocol, "sensor_family"), 0) != 4:
                failures.append(f"{subset}/{protocol}: not 4 sensor families")
    if int((metrics["n_test_unsupported"] > 0).sum()) == 0:
        failures.append("unsupported-class accounting unexpectedly absent")
    audit.check(
        not failures,
        "domain-transfer grid, support flags, and prediction alignment",
        failures,
    )


def validate_registries_and_checkpoints(
    audit: Audit,
    output_dir: Path,
) -> None:
    registry_rows: list[dict[str, Any]] = []
    for name, (rows, run_ids, checkpoints) in EXPECTED_REGISTRIES.items():
        records = load_json(output_dir / name)
        details = {
            "rows": len(records),
            "run_ids": len({item["run_identifier"] for item in records}),
            "checkpoints": len({item["checkpoint"] for item in records}),
        }
        audit.check(
            details == {
                "rows": rows,
                "run_ids": run_ids,
                "checkpoints": checkpoints,
            },
            f"{name} coverage",
            details,
        )
        registry_rows.extend(records)

    checkpoint_expectations: dict[str, tuple[str, str]] = {}
    conflicts: list[str] = []
    for record in registry_rows:
        relative = record["checkpoint"]
        expected = (
            record["state_sha256"],
            record["run_identifier"],
        )
        previous = checkpoint_expectations.get(relative)
        if previous is not None and previous != expected:
            conflicts.append(relative)
        checkpoint_expectations[relative] = expected
    audit.check(not conflicts, "registry checkpoint metadata consistency", conflicts)

    failures: list[str] = []
    for relative, (expected_hash, expected_run) in checkpoint_expectations.items():
        path = output_dir / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            actual_hash = common.state_dict_sha256(payload["state_dict"])
            if actual_hash != expected_hash:
                failures.append(f"registry state hash mismatch: {relative}")
            if payload["state_sha256"] != actual_hash:
                failures.append(f"payload state hash mismatch: {relative}")
            if payload["metadata"]["run_identifier"] != expected_run:
                failures.append(f"payload run ID mismatch: {relative}")
        except Exception as exc:
            failures.append(f"{relative}: {type(exc).__name__}: {exc}")
    audit.check(
        not failures,
        "all selected neural checkpoint tensors and metadata",
        {
            "checked": len(checkpoint_expectations),
            "failures": failures,
        },
    )


def validate_binary_artifacts(audit: Audit, output_dir: Path) -> None:
    for directory, expected in EXPECTED_BINARY_COUNTS.items():
        paths = sorted((output_dir / directory).rglob("*"))
        files = [path for path in paths if path.is_file()]
        audit.check(
            len(files) == expected,
            f"{directory} file count",
            {"expected": expected, "actual": len(files)},
        )
    failures: list[str] = []
    for directory, required_keys in (
        ("embeddings", {"observation_uid", "latent"}),
        (
            "reconstructions",
            {"observation_uid", "clean", "reconstructed"},
        ),
    ):
        for path in sorted((output_dir / directory).glob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as archive:
                    if set(archive.files) != required_keys:
                        failures.append(
                            f"{path.name}: keys {sorted(archive.files)}"
                        )
                        continue
                    if len(archive["observation_uid"]) == 0:
                        failures.append(f"{path.name}: empty")
                    arrays = [
                        archive[key]
                        for key in archive.files
                        if key != "observation_uid"
                    ]
                    if not all(np.isfinite(value).all() for value in arrays):
                        failures.append(f"{path.name}: nonfinite values")
            except Exception as exc:
                failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
    audit.check(
        not failures,
        "embedding and reconstruction archives",
        {"checked": 1_365, "failures": failures},
    )


def validate_figures(audit: Audit, output_dir: Path) -> None:
    expected = {
        "outer_performance.pdf",
        "outer_performance.png",
        "corruption_robustness.pdf",
        "corruption_robustness.png",
        "strict_domain_transfer.pdf",
        "strict_domain_transfer.png",
        "poster_transfer.pdf",
        "poster_transfer.png",
    }
    found = {
        path.name for path in (output_dir / "figures").iterdir() if path.is_file()
    }
    audit.check(
        found == expected,
        "publication figure set",
        {"expected": sorted(expected), "found": sorted(found)},
    )


def validate_clean_rebuild(
    audit: Audit,
    output_dir: Path,
    required: bool,
) -> None:
    path = output_dir / "clean_rebuild_comparison.json"
    if not path.is_file():
        audit.check(
            not required,
            "independent clean rebuild comparison",
            "not yet present",
        )
        return
    report = load_json(path)
    audit.check(
        report.get("status") == "exact_match"
        and report.get("all_required_comparisons_exact") is True,
        "independent clean rebuild comparison",
        {
            "status": report.get("status"),
            "all_required_comparisons_exact": report.get(
                "all_required_comparisons_exact"
            ),
        },
    )


def write_report(
    output_dir: Path,
    audit: Audit,
    require_clean_rebuild: bool,
) -> dict[str, Any]:
    report = {
        "validator": "sers-baseline-bundle-validator-v1",
        "protocol_version": common.PROTOCOL_VERSION,
        "status": "pass" if audit.passed else "fail",
        "require_clean_rebuild": require_clean_rebuild,
        "summary": {
            "checks": len(audit.checks),
            "passed": sum(
                item["status"] == "pass" for item in audit.checks
            ),
            "failed": sum(
                item["status"] == "fail" for item in audit.checks
            ),
        },
        "checks": audit.checks,
    }
    path = output_dir / "validation_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_representation_baselines"
        / "baselines_v1",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repository / "configs" / "sers_representation_baselines_v1.json",
    )
    parser.add_argument(
        "--nato-bundle",
        type=Path,
        default=repository
        / "Workspace"
        / "nato_sers_field_trial"
        / "preprocessing_v2",
    )
    parser.add_argument(
        "--require-clean-rebuild",
        action="store_true",
        help="Fail unless an exact clean-rebuild comparison is present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    output_dir = args.output_dir.resolve()
    protocol_path = args.protocol.resolve()
    nato_bundle = args.nato_bundle.resolve()
    audit = Audit()

    audit.run(
        "required artifact declarations",
        lambda: validate_required_artifacts(audit, output_dir, protocol_path),
    )
    validate_json_files(audit, output_dir)
    validate_protocol_and_selection(audit, output_dir, protocol_path)
    audit.run(
        "immutable preprocessing-v2 verification",
        lambda: validate_preprocessing(audit, nato_bundle),
    )
    validate_table_sizes(audit, output_dir)
    validate_selection_search(audit, output_dir)
    validate_controls(audit, output_dir)
    validate_outer(audit, output_dir)
    validate_poster(audit, output_dir)
    validate_domain(audit, output_dir)
    validate_registries_and_checkpoints(audit, output_dir)
    validate_binary_artifacts(audit, output_dir)
    validate_figures(audit, output_dir)
    validate_clean_rebuild(audit, output_dir, args.require_clean_rebuild)
    validate_hash_catalog(audit, output_dir)

    report = write_report(
        output_dir,
        audit,
        args.require_clean_rebuild,
    )
    print(json.dumps(report["summary"] | {"status": report["status"]}))
    return 0 if audit.passed else 1


if __name__ == "__main__":
    sys.exit(main())
