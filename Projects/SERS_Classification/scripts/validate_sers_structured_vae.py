#!/usr/bin/env python3
"""Validate the complete SERS structured-VAE study and an optional rebuild."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image

import sers_baseline_common as baseline


EXPECTED_ROWS = {
    "identity_control_metrics.csv": 20,
    "identity_control_histories.csv": 10_000,
    "controls_fold_metrics.csv": 80,
    "controls_histories.csv": 40_000,
    "instrument_adversary_fold_metrics.csv": 60,
    "instrument_adversary_histories.csv": 30_000,
    "pair_fold_metrics.csv": 60,
    "pair_histories.csv": 30_000,
    "dependence_fold_metrics.csv": 40,
    "dependence_histories.csv": 20_000,
    "sensitivity_fold_metrics.csv": 60,
    "sensitivity_training_histories.csv": 30_000,
    "locked_outer_metrics.csv": 90,
    "locked_outer_predictions.csv": 7_176,
    "locked_outer_reconstruction_metrics.csv": 2_392,
    "locked_outer_corruption_metrics.csv": 630,
    "locked_outer_swap_metrics.csv": 30,
    "locked_outer_training_histories.csv": 10_000,
    "locked_domain_metrics.csv": 168,
    "locked_domain_predictions.csv": 13_176,
    "locked_domain_reconstruction_metrics.csv": 4_392,
    "locked_domain_corruption_metrics.csv": 1_176,
    "locked_domain_training_histories.csv": 28_000,
    "locked_poster_metrics.csv": 24,
    "locked_poster_predictions.csv": 1_650,
    "locked_poster_reconstruction_metrics.csv": 550,
    "locked_poster_corruption_metrics.csv": 168,
    "locked_poster_training_histories.csv": 4_000,
    "negative_control_chemical_permutation.csv": 20,
}

EXPECTED_REGISTRIES = {
    "sensitivity_run_registry.json": 60,
    "locked_outer_run_registry.json": 20,
    "locked_domain_run_registry.json": 56,
    "locked_poster_run_registry.json": 8,
}

EXPECTED_STAGE_CONFIGURATIONS = {
    "controls_summary.csv": 4,
    "instrument_adversary_summary.csv": 3,
    "pair_summary.csv": 3,
    "dependence_summary.csv": 2,
}

EXPECTED_FIGURES = {
    "inner_mechanism_tradeoffs",
    "locked_outer_comparators",
    "partition_preprocessing_sensitivity",
    "heldout_domain_heatmap",
    "corruption_robustness",
    "locked_swap_examples",
}

EXPECTED_CACHE_COUNTS = {
    "identity_control": 20,
    "selection_cache": 240,
    "confirmation_cache": 144,
}

REQUIRED_FILES = {
    "FINAL_REPORT.md",
    "PREDECLARED_PROTOCOL.md",
    "README.md",
    "artifact_hashes.json",
    "compute_accounting.json",
    "environment.json",
    "failure_attribution.json",
    "identity_control_summary.json",
    "inner_gate_matrix.csv",
    "inner_selection_closure.json",
    "inner_stage_winners.csv",
    "input_hashes.json",
    "negative_control_summary.json",
    "predeclared_protocol.json",
    "reproduction_commands.sh",
    "terminal_decision.json",
    "audit/METADATA_IDENTIFIABILITY_AUDIT.md",
    "audit/audit_summary.json",
}

PROBABILITY_COLUMNS = {
    "balanced_accuracy_supported",
    "macro_f1_supported",
    "macro_f1_union",
    "prediction_confidence",
    "reconstruction_median_row_correlation",
    "repeatable_peak_recall",
    "partition_maximum_canonical_correlation",
}

SCIENTIFIC_JSONS = {
    "audit/audit_summary.json",
    "compute_accounting.json",
    "controls_decision.json",
    "dependence_decision.json",
    "failure_attribution.json",
    "identity_control_summary.json",
    "inner_selection_closure.json",
    "instrument_adversary_decision.json",
    "negative_control_summary.json",
    "pair_decision.json",
    "predeclared_protocol.json",
    "sensitivity_decision.json",
    "terminal_decision.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> Any:
    return json.loads(path.read_text())


def tensor_tree_digest(value: Any) -> tuple[str, int]:
    """Hash every tensor in a nested checkpoint without hashing metadata."""
    digest = hashlib.sha256()
    tensor_count = 0

    def walk(item: Any, path: str) -> None:
        nonlocal tensor_count
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(path.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
            tensor_count += 1
            return
        if isinstance(item, dict):
            for key in sorted(item, key=lambda candidate: str(candidate)):
                walk(item[key], f"{path}/{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{path}/{index}")

    walk(value, "")
    return digest.hexdigest(), tensor_count


class Validator:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, condition: bool, name: str, detail: str = "") -> None:
        self.checks.append(
            {
                "name": name,
                "passed": bool(condition),
                "detail": detail,
            }
        )

    def equal(self, actual: Any, expected: Any, name: str) -> None:
        self.check(
            actual == expected,
            name,
            f"actual={actual!r}; expected={expected!r}",
        )

    @property
    def passed(self) -> bool:
        return all(record["passed"] for record in self.checks)

    def report(self) -> dict[str, Any]:
        failures = [
            record for record in self.checks if not record["passed"]
        ]
        return {
            "protocol": "sers-structured-vae-v1",
            "status": "passed" if not failures else "failed",
            "check_count": len(self.checks),
            "failure_count": len(failures),
            "failures": failures,
            "checks": self.checks,
        }


def validate_files(root: Path, validator: Validator) -> None:
    missing = sorted(
        relative
        for relative in REQUIRED_FILES
        if not (root / relative).is_file()
    )
    validator.check(not missing, "required artifact files", str(missing))
    for relative, expected in EXPECTED_ROWS.items():
        path = root / relative
        if not path.is_file():
            validator.check(False, f"row count: {relative}", "file missing")
            continue
        actual = len(pd.read_csv(path))
        validator.equal(actual, expected, f"row count: {relative}")
    for relative, expected in EXPECTED_REGISTRIES.items():
        path = root / relative
        if not path.is_file():
            validator.check(False, f"registry count: {relative}", "missing")
            continue
        validator.equal(
            len(json_load(path)), expected, f"registry count: {relative}"
        )
    for relative, expected in EXPECTED_STAGE_CONFIGURATIONS.items():
        path = root / relative
        if not path.is_file():
            validator.check(False, f"candidate count: {relative}", "missing")
            continue
        validator.equal(
            len(pd.read_csv(path)),
            expected,
            f"candidate count: {relative}",
        )
    for directory, expected in EXPECTED_CACHE_COUNTS.items():
        path = root / directory
        actual = len(list(path.rglob("*.pt"))) if path.is_dir() else -1
        validator.equal(actual, expected, f"checkpoint count: {directory}")
    validator.equal(
        len(list((root / "embeddings").glob("*.npz"))),
        94,
        "embedding bundle count",
    )
    validator.equal(
        len(list((root / "reconstructions").glob("*.npz"))),
        94,
        "reconstruction bundle count",
    )
    validator.equal(
        len(list((root / "swaps").glob("*.npz"))),
        30,
        "locked swap bundle count",
    )


def validate_protocol_decisions(root: Path, validator: Validator) -> None:
    audit = json_load(root / "audit/audit_summary.json")
    for key, expected in {
        "spectra": 598,
        "master_samples": 69,
        "targets": 7,
        "instruments": 10,
        "sensor_families": 4,
        "cross_instrument_pairs": 2473,
        "cross_sensor_pairs": 1171,
    }.items():
        validator.equal(audit.get(key), expected, f"audit: {key}")
    validator.check(
        math.isclose(
            audit.get("target_instrument_support_fraction", -1),
            44 / 70,
            abs_tol=1e-15,
        ),
        "audit: target×instrument support",
    )
    validator.check(
        math.isclose(
            audit.get("target_sensor_support_fraction", -1),
            17 / 28,
            abs_tol=1e-15,
        ),
        "audit: target×sensor support",
    )

    identity = json_load(root / "identity_control_summary.json")
    for key in (
        "identity_gate_passed",
        "all_checkpoint_tensors_exact",
        "all_optimizer_states_exact",
    ):
        validator.check(bool(identity.get(key)), f"identity: {key}")
    validator.equal(identity.get("fold_count"), 20, "identity fold count")
    validator.equal(
        identity.get("maximum_history_absolute_difference"),
        0.0,
        "identity history exactness",
    )

    closure = json_load(root / "inner_selection_closure.json")
    selected = (
        "zc48__zn16__chem0__ni0__ns0__cond0__ai0__as0__pair0__"
        "xrec0__dep0p001__e500"
    )
    validator.check(closure.get("selection_closed"), "selection is frozen")
    validator.check(
        not closure.get("selection_used_locked_outcomes", True),
        "selection did not use locked outcomes",
    )
    validator.equal(
        closure.get("selected_by_registered_hierarchy"),
        selected,
        "registered selection",
    )
    validator.equal(
        closure.get("selected_gate_count"), 15, "selected gate count"
    )
    validator.equal(
        closure.get("selected_gate_total"), 17, "selected gate total"
    )
    validator.check(
        not closure.get("selected_passes_all_gates", True),
        "selected model recorded as ineligible",
    )
    validator.check(
        not closure["sensor_adversary"]["opened"],
        "sensor branch correctly remained closed",
    )
    validator.check(
        not closure["combination"]["opened"],
        "combination branch correctly remained closed",
    )
    validator.equal(
        closure.get("eligible_candidate_count"),
        0,
        "eligible candidate count",
    )

    negative = json_load(root / "negative_control_summary.json")
    validator.check(
        negative.get("all_applicable_controls_passed"),
        "applicable negative controls passed",
    )
    permuted = negative["chemical_group_permutation"]
    validator.equal(permuted.get("fold_count"), 20, "permutation fold count")
    validator.check(
        permuted["maximum_balanced_accuracy"]
        <= permuted["registered_per_fold_maximum"],
        "permuted chemical labels remained below registered bound",
    )

    terminal = json_load(root / "terminal_decision.json")
    validator.equal(
        terminal.get("terminal_classification"),
        "unsuccessful",
        "terminal classification",
    )
    validator.check(
        terminal.get("idea_worked_as_disentanglement") is False,
        "no unsupported disentanglement claim",
    )
    validator.check(
        terminal.get("idea_worked_as_general_nuisance_filter") is False,
        "no unsupported nuisance-filter claim",
    )
    validator.equal(
        terminal.get("selected_gate_count"), 15, "terminal gate count"
    )
    validator.equal(
        terminal.get("failed_selected_gates"),
        ["gate_same_master", "gate_fold_chemical_direction"],
        "terminal failed-gate identity",
    )

    accounting = json_load(root / "compute_accounting.json")
    validator.equal(
        accounting.get("authoritative_training_run_count"),
        404,
        "authoritative training-run count",
    )
    validator.equal(
        accounting.get("authoritative_optimizer_epoch_count"),
        202_000,
        "authoritative optimizer-epoch count",
    )
    validator.equal(
        sum(accounting["authoritative_run_counts"].values()),
        404,
        "component run-count sum",
    )


def validate_numeric_tables(root: Path, validator: Validator) -> None:
    metric_files = [
        "locked_outer_metrics.csv",
        "locked_domain_metrics.csv",
        "locked_poster_metrics.csv",
    ]
    for relative in metric_files:
        frame = pd.read_csv(root / relative)
        validator.check(
            set(frame["partition"]) == {"chemical", "nuisance", "union"},
            f"partition coverage: {relative}",
            str(sorted(set(frame["partition"]))),
        )
        for column in PROBABILITY_COLUMNS.intersection(frame.columns):
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            validator.check(
                bool(((values >= -1.0) & (values <= 1.0)).all()),
                f"bounded metric: {relative}:{column}",
            )
        numeric = frame.select_dtypes(include=[np.number])
        validator.check(
            not np.isinf(numeric.to_numpy(dtype=float)).any(),
            f"no infinite metrics: {relative}",
        )
    for relative in (
        "locked_outer_predictions.csv",
        "locked_domain_predictions.csv",
        "locked_poster_predictions.csv",
    ):
        frame = pd.read_csv(root / relative)
        confidence = frame["prediction_confidence"]
        validator.check(
            bool(confidence.between(0.0, 1.0).all()),
            f"prediction confidence range: {relative}",
        )
        validator.check(
            bool(frame["correct"].isin([True, False]).all()),
            f"binary correctness: {relative}",
        )
    histories = [
        relative
        for relative in EXPECTED_ROWS
        if "histor" in relative
    ]
    for relative in histories:
        frame = pd.read_csv(root / relative)
        validator.check(
            "epoch" in frame and frame["epoch"].between(1, 500).all(),
            f"registered epoch range: {relative}",
        )
        numeric = frame.select_dtypes(include=[np.number])
        validator.check(
            not np.isinf(numeric.to_numpy(dtype=float)).any(),
            f"no infinite history values: {relative}",
        )


def validate_outer_coverage(
    root: Path, repository: Path, validator: Validator
) -> None:
    manifest = pd.read_csv(
        repository
        / "Workspace/nato_sers_field_trial/preprocessing_v2/"
        "core_preprocessing_manifest.csv"
    )
    manifest["observation_uid"] = manifest["observation_uid"].astype(str)
    manifest = manifest.set_index("observation_uid", drop=False)
    expected_sets = {
        "strict_core": set(
            manifest.loc[
                manifest["include_sers_core"].astype(bool), "observation_uid"
            ]
        ),
        "quality_pass": set(
            manifest.loc[
                manifest["include_sers_qc_pass"].astype(bool),
                "observation_uid",
            ]
        ),
        "field_quality_stress": set(
            manifest.loc[
                manifest["field_quality_stress"].astype(bool),
                "observation_uid",
            ]
        ),
    }
    for subset, expected in {
        "strict_core": 598,
        "quality_pass": 500,
        "field_quality_stress": 98,
    }.items():
        validator.equal(
            len(expected_sets[subset]),
            expected,
            f"source cohort size: {subset}",
        )

    predictions = pd.read_csv(root / "locked_outer_predictions.csv")
    keys = [
        "test_subset",
        "representation",
        "partition",
        "observation_uid",
    ]
    validator.check(
        not predictions.duplicated(keys).any(),
        "locked outer prediction keys are unique",
    )
    for subset, expected_uids in expected_sets.items():
        for representation in ("arpls_minmax", "minimal_minmax"):
            for partition in ("chemical", "nuisance", "union"):
                observed = set(
                    predictions.loc[
                        predictions["test_subset"].eq(subset)
                        & predictions["representation"].eq(representation)
                        & predictions["partition"].eq(partition),
                        "observation_uid",
                    ].astype(str)
                )
                validator.check(
                    observed == expected_uids,
                    (
                        "outer coverage: "
                        f"{subset}/{representation}/{partition}"
                    ),
                    (
                        f"observed={len(observed)}; "
                        f"expected={len(expected_uids)}"
                    ),
                )
    uid_to_fold = manifest["grouped_sample_fold_5"].astype(int).to_dict()
    observed_folds = predictions["observation_uid"].astype(str).map(uid_to_fold)
    validator.check(
        observed_folds.notna().all(),
        "all locked outer predictions map to source observations",
    )
    validator.check(
        bool(
            (
                observed_folds.to_numpy(dtype=int)
                == predictions["outer_fold"].to_numpy(dtype=int)
            ).all()
        ),
        "locked outer test folds match frozen master-group folds",
    )
    master_fold_counts = (
        manifest.reset_index(drop=True)
        .groupby("master_sample_id")["grouped_sample_fold_5"]
        .nunique()
    )
    validator.check(
        bool((master_fold_counts == 1).all()),
        "master_sample_id never crosses an outer fold",
    )


def validate_checkpoints(root: Path, validator: Validator) -> None:
    registry_paths: list[tuple[str, dict[str, Any]]] = []
    for registry_name in EXPECTED_REGISTRIES:
        for record in json_load(root / registry_name):
            registry_paths.append((registry_name, record))
    seen: set[str] = set()
    for registry_name, record in registry_paths:
        relative = record["checkpoint"]
        path = root / relative
        validator.check(
            relative not in seen,
            f"unique registry checkpoint: {relative}",
        )
        seen.add(relative)
        if not path.is_file():
            validator.check(False, f"checkpoint exists: {relative}", "missing")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state_hash = baseline.state_dict_sha256(payload["state"])
        validator.equal(
            state_hash,
            payload["state_sha256"],
            f"checkpoint internal state hash: {relative}",
        )
        validator.equal(
            state_hash,
            record["state_sha256"],
            f"registry state hash: {relative}",
        )
        validator.equal(
            len(payload["history"]),
            500,
            f"checkpoint history length: {relative}",
        )
        metadata = payload["metadata"]
        for key in (
            "execution_fingerprint",
            "run_identifier",
            "run_seed",
            "train_uids_sha256",
            "train_values_sha256",
            "validation_uids_sha256",
            "validation_values_sha256",
        ):
            validator.equal(
                metadata[key],
                record[key],
                f"registry metadata {key}: {relative}",
            )

    selection_metrics = pd.concat(
        [
            pd.read_csv(root / f"{stage}_fold_metrics.csv")
            for stage in (
                "controls",
                "instrument_adversary",
                "pair",
                "dependence",
            )
        ],
        ignore_index=True,
    )
    selection_lookup = selection_metrics.set_index(
        ["stage", "outer_fold", "inner_fold", "identifier"]
    )["state_sha256"].to_dict()
    selection_files = list((root / "selection_cache").rglob("*.pt"))
    for path in selection_files:
        stage = path.parent.name
        stem = path.stem
        prefix, identifier = stem.split("__arpls_minmax__", maxsplit=1)
        outer_token, inner_token = prefix.split("__")[-2:]
        outer_fold = int(outer_token.removeprefix("o"))
        inner_fold = int(inner_token.removeprefix("i"))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["states"][500]
        state_hash = baseline.state_dict_sha256(state)
        key = (stage, outer_fold, inner_fold, identifier)
        validator.equal(
            state_hash,
            selection_lookup.get(key),
            f"selection checkpoint state hash: {path.relative_to(root)}",
        )
        validator.equal(
            len(payload["history"]),
            500,
            f"selection checkpoint history: {path.relative_to(root)}",
        )
        validator.check(
            500 in payload["optimizer_states"],
            f"selection optimizer state present: {path.relative_to(root)}",
        )


def validate_npz_and_figures(root: Path, validator: Validator) -> None:
    embedding_names = {
        path.name for path in (root / "embeddings").glob("*.npz")
    }
    reconstruction_names = {
        path.name for path in (root / "reconstructions").glob("*.npz")
    }
    validator.check(
        embedding_names == reconstruction_names,
        "embedding/reconstruction bundle names match",
    )
    for directory in ("embeddings", "reconstructions", "swaps"):
        for path in sorted((root / directory).glob("*.npz")):
            with np.load(path, allow_pickle=False) as bundle:
                validator.check(
                    bool(bundle.files),
                    f"nonempty NPZ: {path.relative_to(root)}",
                )
                for key in bundle.files:
                    array = bundle[key]
                    if np.issubdtype(array.dtype, np.number):
                        validator.check(
                            bool(np.isfinite(array).all()),
                            (
                                "finite NPZ array: "
                                f"{path.relative_to(root)}:{key}"
                            ),
                        )
    swap_metrics = pd.read_csv(root / "locked_outer_swap_metrics.csv")
    validator.equal(
        int(swap_metrics["real_pair_count"].sum()),
        2_270,
        "locked swap real-pair row count",
    )
    validator.equal(
        int(swap_metrics["real_pair_count"].eq(0).sum()),
        2,
        "locked swap no-partner representation rows",
    )
    for column in (
        "same_master_invariant",
        "different_instrument_invariant",
        "same_target_invariant",
    ):
        validator.check(
            bool(swap_metrics[column].all()),
            f"locked swap invariant: {column}",
        )
    validator.equal(
        len(
            swap_metrics[
                ["scenario", "representation"]
            ].drop_duplicates()
        ),
        30,
        "locked swap scenario/representation coverage",
    )
    for path in sorted((root / "swaps").glob("*.npz")):
        with np.load(path, allow_pickle=False) as bundle:
            required = {
                "axis_cm1",
                "source_observation_uid",
                "partner_observation_uid",
                "source_master_sample_id",
                "partner_master_sample_id",
                "source_target_analyte",
                "partner_target_analyte",
                "source_instrument",
                "partner_instrument",
                "source_sensor_family",
                "partner_sensor_family",
                "source_clean",
                "partner_clean",
                "source_standard_reconstruction",
                "source_chemical_mu",
                "partner_nuisance_mu",
                "swapped_reconstruction",
            }
            validator.check(
                set(bundle.files) == required,
                f"locked swap schema: {path.name}",
            )
            count = len(bundle["source_observation_uid"])
            validator.check(
                all(
                    len(bundle[key]) == count
                    for key in required - {"axis_cm1"}
                ),
                f"locked swap aligned rows: {path.name}",
            )
            validator.check(
                np.array_equal(
                    bundle["source_master_sample_id"],
                    bundle["partner_master_sample_id"],
                ),
                f"locked swap same master: {path.name}",
            )
            validator.check(
                np.array_equal(
                    bundle["source_target_analyte"],
                    bundle["partner_target_analyte"],
                ),
                f"locked swap same analyte: {path.name}",
            )
            validator.check(
                bool(
                    (
                        bundle["source_instrument"]
                        != bundle["partner_instrument"]
                    ).all()
                ),
                f"locked swap different instrument: {path.name}",
            )
            validator.check(
                np.array_equal(
                    bundle["axis_cm1"],
                    np.arange(400, 1801, dtype=np.float32),
                ),
                f"locked swap common axis: {path.name}",
            )
            validator.equal(
                bundle["swapped_reconstruction"].shape,
                (count, 1_401),
                f"locked swap spectral shape: {path.name}",
            )
    figure_dir = root / "figures"
    for stem in EXPECTED_FIGURES:
        png = figure_dir / f"{stem}.png"
        pdf = figure_dir / f"{stem}.pdf"
        validator.check(png.is_file(), f"figure PNG exists: {stem}")
        validator.check(pdf.is_file(), f"figure PDF exists: {stem}")
        if png.is_file():
            with Image.open(png) as image:
                width, height = image.size
                dpi = image.info.get("dpi", (0, 0))
                validator.check(
                    width >= 1_500 and height >= 1_000,
                    f"figure raster dimensions: {stem}",
                    f"{width}×{height}",
                )
                validator.check(
                    min(dpi) >= 590,
                    f"figure raster resolution: {stem}",
                    str(dpi),
                )
        if pdf.is_file():
            validator.check(
                pdf.stat().st_size > 10_000,
                f"figure PDF nontrivial: {stem}",
                str(pdf.stat().st_size),
            )


def validate_hash_manifest(
    root: Path, repository: Path, validator: Validator
) -> None:
    manifest = json_load(root / "artifact_hashes.json")
    for relative, expected in manifest.items():
        path = repository / relative
        validator.check(path.is_file(), f"hashed artifact exists: {relative}")
        if path.is_file():
            validator.equal(
                sha256_file(path),
                expected,
                f"artifact SHA-256: {relative}",
            )
    validator.check(
        "scripts/validate_sers_structured_vae.py" in manifest,
        "validator source is included in the artifact manifest",
    )


def compare_exact_rebuild(
    root: Path, reference: Path, validator: Validator
) -> None:
    """Apply the preregistered exact comparison to scientific artifacts."""
    validator.check(reference.is_dir(), "rebuild reference exists", str(reference))
    if not reference.is_dir():
        return

    root_csvs = {
        path.relative_to(root)
        for path in root.rglob("*.csv")
        if "cache" not in path.parts
    }
    reference_csvs = {
        path.relative_to(reference)
        for path in reference.rglob("*.csv")
        if "cache" not in path.parts
    }
    validator.check(
        root_csvs == reference_csvs,
        "rebuild scientific CSV file set exact",
    )
    for relative in sorted(root_csvs & reference_csvs):
        validator.equal(
            sha256_file(root / relative),
            sha256_file(reference / relative),
            f"rebuild scientific table exact: {relative}",
        )

    for relative_text in sorted(SCIENTIFIC_JSONS):
        relative = Path(relative_text)
        validator.check(
            (root / relative).is_file()
            and (reference / relative).is_file(),
            f"rebuild scientific JSON exists: {relative}",
        )
        if (root / relative).is_file() and (reference / relative).is_file():
            validator.check(
                json_load(root / relative) == json_load(reference / relative),
                f"rebuild scientific JSON exact: {relative}",
            )

    left_inputs = json_load(root / "input_hashes.json")
    right_inputs = json_load(reference / "input_hashes.json")
    validator.equal(
        set(left_inputs),
        set(right_inputs),
        "rebuild frozen-input key set exact",
    )
    for key in set(left_inputs).intersection(right_inputs):
        validator.equal(
            left_inputs[key]["sha256"],
            right_inputs[key]["sha256"],
            f"rebuild frozen-input digest exact: {key}",
        )

    for directory in ("embeddings", "reconstructions", "swaps"):
        left = {
            path.relative_to(root)
            for path in (root / directory).glob("*.npz")
        }
        right = {
            path.relative_to(reference)
            for path in (reference / directory).glob("*.npz")
        }
        validator.check(left == right, f"rebuild {directory} file set exact")
        for relative in sorted(left & right):
            with np.load(root / relative, allow_pickle=False) as left_bundle:
                with np.load(
                    reference / relative, allow_pickle=False
                ) as right_bundle:
                    validator.equal(
                        left_bundle.files,
                        right_bundle.files,
                        f"rebuild array keys exact: {relative}",
                    )
                    for key in set(left_bundle.files).intersection(
                        right_bundle.files
                    ):
                        left_array = left_bundle[key]
                        right_array = right_bundle[key]
                        numeric = np.issubdtype(
                            left_array.dtype, np.number
                        )
                        validator.check(
                            np.array_equal(
                                left_array,
                                right_array,
                                equal_nan=numeric,
                            ),
                            f"rebuild array exact: {relative}:{key}",
                        )

    for directory in (
        "identity_control",
        "selection_cache",
        "confirmation_cache",
    ):
        left = {
            path.relative_to(root)
            for path in (root / directory).rglob("*.pt")
        }
        right = {
            path.relative_to(reference)
            for path in (reference / directory).rglob("*.pt")
        }
        validator.check(left == right, f"rebuild {directory} file set exact")
        for relative in sorted(left & right):
            left_payload = torch.load(
                root / relative, map_location="cpu", weights_only=False
            )
            right_payload = torch.load(
                reference / relative, map_location="cpu", weights_only=False
            )
            left_digest, left_count = tensor_tree_digest(left_payload)
            right_digest, right_count = tensor_tree_digest(right_payload)
            validator.equal(
                left_count,
                right_count,
                f"rebuild tensor count exact: {relative}",
            )
            validator.equal(
                left_digest,
                right_digest,
                f"rebuild checkpoint/optimizer tensors exact: {relative}",
            )


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace/sers_structured_vae/structured_vae_v1",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help=(
            "Optional independently built output to compare exactly against "
            "--output-dir."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "Validation JSON destination; defaults to "
            "<output-dir>/validation_report.json."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    repository = Path(__file__).resolve().parents[1]
    root = args.output_dir.resolve()
    report_path = (
        args.report.resolve()
        if args.report
        else root / "validation_report.json"
    )
    validator = Validator()
    validator.check(root.is_dir(), "output directory exists", str(root))
    if root.is_dir():
        validate_files(root, validator)
        validate_protocol_decisions(root, validator)
        validate_numeric_tables(root, validator)
        validate_outer_coverage(root, repository, validator)
        validate_checkpoints(root, validator)
        validate_npz_and_figures(root, validator)
        validate_hash_manifest(root, repository, validator)
        if args.reference_dir is not None:
            compare_exact_rebuild(
                root, args.reference_dir.resolve(), validator
            )
    report = validator.report()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "check_count": report["check_count"],
                "failure_count": report["failure_count"],
                "report": str(report_path),
            },
            indent=2,
        )
    )
    if not validator.passed:
        for failure in report["failures"]:
            print(
                f"FAILED: {failure['name']}: {failure['detail']}",
                file=sys.stderr,
            )
    return 0 if validator.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
