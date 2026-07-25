#!/usr/bin/env python3
"""Run inner-only convergence and bounded adequacy selection for SERS VAE."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

import run_sers_representation_baselines as baseline_selection
import run_sers_standard_vae_selection as standard_selection
import sers_baseline_common as baseline
import sers_vae_adequacy_common as adequacy
import sers_vae_common as standard


CACHE_SCHEMA = "sers-vae-adequacy-selection-cache-v1"


def json_clean(value: Any) -> Any:
    return baseline_selection.json_clean(value)


def write_json(path: Path, value: Any) -> None:
    baseline_selection.write_json(path, value)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != adequacy.PROTOCOL_VERSION:
        raise ValueError("Unexpected adequacy protocol version")
    if protocol["status_before_model_execution"] != "predeclared":
        raise ValueError("Adequacy protocol was not predeclared")
    return protocol


def verify_inputs(
    protocol: dict[str, Any],
    nato_bundle: Path,
    standard_bundle: Path,
    baseline_bundle: Path,
) -> None:
    immutable = protocol["immutable_inputs"]
    if baseline.sha256_file(nato_bundle / "artifact_hashes.json") != immutable[
        "nato_artifact_catalog_sha256"
    ]:
        raise ValueError("Frozen NATO artifact catalog changed")
    baseline.verify_hash_catalog(nato_bundle)
    selected = json.loads(
        (standard_bundle / "selected_configuration.json").read_text()
    )
    if selected["identifier"] != (
        "c8x16_z64_spectral_composite_beta1_cyclical_4"
    ):
        raise ValueError("Original standard-VAE reference changed")
    validation = json.loads((standard_bundle / "validation_report.json").read_text())
    if validation["summary"]["failed"] != 0:
        raise ValueError("Original standard-VAE reference is not validated")
    if not (baseline_bundle / "selected_configurations.json").exists():
        raise ValueError("Frozen baseline selection bundle is missing")


def execution_fingerprint(
    protocol_path: Path, amendment_path: Path | None = None
) -> str:
    digest = hashlib.sha256()
    for path in (
        protocol_path.resolve(),
        Path(baseline.__file__).resolve(),
        Path(adequacy.__file__).resolve(),
        Path(__file__).resolve(),
    ):
        digest.update(str(path).encode())
        digest.update(baseline.sha256_file(path).encode())
    if amendment_path is not None:
        digest.update(str(amendment_path.resolve()).encode())
        digest.update(baseline.sha256_file(amendment_path).encode())
    digest.update(CACHE_SCHEMA.encode())
    for function in (
        evaluate_checkpoint,
        run_grid,
        aggregate_checkpoint_metrics,
        apply_gates,
    ):
        digest.update(inspect.getsource(function).encode())
    return digest.hexdigest()


def initialize_output(
    output_dir: Path,
    protocol_path: Path,
    nato_bundle: Path,
    standard_bundle: Path,
    baseline_bundle: Path,
    amendment_path: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = output_dir / "predeclared_protocol.json"
    if copied.exists() and copied.read_bytes() != protocol_path.read_bytes():
        raise ValueError("Output directory contains a different protocol")
    if not copied.exists():
        shutil.copyfile(protocol_path, copied)
    if amendment_path is not None:
        amendment_copy = output_dir / "protocol_amendment_1.json"
        if (
            amendment_copy.exists()
            and amendment_copy.read_bytes() != amendment_path.read_bytes()
        ):
            raise ValueError("Output directory contains a different amendment")
        if not amendment_copy.exists():
            shutil.copyfile(amendment_path, amendment_copy)
    inputs = {
        "protocol": protocol_path,
        "nato_artifact_catalog": nato_bundle / "artifact_hashes.json",
        "standard_vae_artifact_catalog": standard_bundle / "artifact_hashes.json",
        "standard_vae_selected_configuration": (
            standard_bundle / "selected_configuration.json"
        ),
        "standard_vae_final_decisions": standard_bundle / "final_decisions.json",
        "baseline_artifact_catalog": baseline_bundle / "artifact_hashes.json",
        "baseline_selected_configurations": (
            baseline_bundle / "selected_configurations.json"
        ),
    }
    if amendment_path is not None:
        inputs["protocol_amendment_1"] = amendment_path
    write_json(
        output_dir / "input_hashes.json",
        {
            key: {"path": str(path), "sha256": baseline.sha256_file(path)}
            for key, path in inputs.items()
        },
    )
    write_json(
        output_dir / "environment.json",
        {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "deterministic_algorithms_required": True,
            "canonical_metric_device": "cpu",
        },
    )


def existing_run_audit(standard_bundle: Path) -> dict[str, Any]:
    metrics = pd.read_csv(standard_bundle / "selection_fold_metrics.csv")
    history = pd.read_csv(standard_bundle / "selection_training_histories.csv")
    selected = metrics[
        (metrics["kl_schedule"] == "cyclical_4")
        & (metrics["representation"].isin(["arpls_minmax", "minimal_minmax"]))
    ]
    groups: dict[str, Any] = {}
    for (subset, representation), frame in selected.groupby(
        ["subset", "representation"]
    ):
        best = frame["best_epoch"].astype(int)
        histories = history[
            (history["kl_schedule"] == "cyclical_4")
            & (history["subset"] == subset)
            & (history["representation"] == representation)
        ]
        changes: list[float] = []
        relative: list[float] = []
        for _, run in histories.groupby("run_identifier"):
            indexed = run.set_index("epoch")
            if 90 in indexed.index and 100 in indexed.index:
                before = float(indexed.loc[90, "validation_loss"])
                after = float(indexed.loc[100, "validation_loss"])
                changes.append(after - before)
                relative.append(100.0 * (after - before) / before)
        groups[f"{subset}__{representation}"] = {
            "run_count": int(len(frame)),
            "best_epoch_minimum": int(best.min()),
            "best_epoch_median": float(best.median()),
            "best_epoch_mean": float(best.mean()),
            "best_epoch_maximum": int(best.max()),
            "best_epoch_at_100_count": int((best == 100).sum()),
            "best_epoch_at_least_95_count": int((best >= 95).sum()),
            "epoch_90_to_100_run_count": int(len(changes)),
            "epoch_90_to_100_validation_loss_change_median": (
                float(np.median(changes)) if changes else None
            ),
            "epoch_90_to_100_validation_loss_percent_change_median": (
                float(np.median(relative)) if relative else None
            ),
            "epoch_90_to_100_improved_fraction": (
                float(np.mean(np.asarray(changes) < 0)) if changes else None
            ),
            "n_train_minimum": int(frame["n_train"].min()),
            "n_train_median": float(frame["n_train"].median()),
            "n_train_maximum": int(frame["n_train"].max()),
            "parameter_count": int(frame["parameter_count"].iloc[0]),
        }
    decisions = json.loads((standard_bundle / "final_decisions.json").read_text())
    return {
        "source": str(standard_bundle),
        "groups": groups,
        "selected_configuration": decisions["selected_configuration"],
        "selection_gate_failures": decisions["selection_gate_failures"],
        "cycle_interaction": {
            "fourth_cycle_begins_epoch": 76,
            "fourth_cycle_reaches_beta_one_epoch": 89,
            "beta_one_epochs_before_cap": 12,
            "early_stopping_patience": 15,
            "finding": (
                "The cap binds most folds; one strict-core run could stop "
                "during the final ramp because patience spans that ramp."
            ),
        },
        "preregistered_interpretation": (
            "The prior result is a valid controlled 100-epoch comparator, "
            "but convergence is not established."
        ),
    }


def original_seed_lookup(standard_bundle: Path) -> dict[tuple[int, int], int]:
    metrics = pd.read_csv(standard_bundle / "selection_fold_metrics.csv")
    selected = metrics[
        (metrics["subset"] == "strict_core")
        & (metrics["representation"] == "arpls_minmax")
        & (metrics["kl_schedule"] == "cyclical_4")
    ]
    if len(selected) != 20:
        raise ValueError("Expected 20 original selected strict-core runs")
    return {
        (int(row.outer_fold), int(row.inner_fold)): int(row.run_seed)
        for row in selected.itertuples(index=False)
    }


def model_from_state(
    config: adequacy.AdequacyConfig,
    input_length: int,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.nn.Module:
    model = adequacy.build_model(
        input_length, config.architecture, config.latent_dimension
    )
    model.load_state_dict(state)
    return model.to(device)


def evaluate_checkpoint(
    dataset: baseline.SpectralDataset,
    representation: str,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    config: adequacy.AdequacyConfig,
    checkpoint_epoch: int,
    state: dict[str, torch.Tensor],
    run_seed: int,
    repeatable_validation: list[set[int]],
    raw_context: dict[str, Any],
    metric_device: torch.device,
) -> dict[str, Any]:
    values = dataset.representations[representation]
    manifest = dataset.manifest
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    train_values = values[train_indices]
    validation_values = values[validation_indices]
    train_targets = (
        manifest.iloc[train_indices]["target_analyte"].astype(str).to_numpy()
    )
    validation_targets = (
        manifest.iloc[validation_indices]["target_analyte"].astype(str).to_numpy()
    )
    model = model_from_state(
        config, train_values.shape[1], state, metric_device
    )
    _, train_mu, _ = adequacy.outputs(model, train_values, metric_device)
    validation_reconstruction, validation_mu, validation_log_variance = (
        adequacy.outputs(model, validation_values, metric_device)
    )
    clean_metrics = baseline.aggregate_reconstruction_metrics(
        baseline.reconstruction_metrics(
            validation_values,
            validation_reconstruction,
            dataset.observation_uid[validation_indices],
            repeatable_validation,
        )
    )
    variability = adequacy.sample_reconstruction_variability(
        model,
        validation_values,
        metric_device,
        baseline.stable_seed(run_seed, "sample_variability"),
    )
    variational = standard.variational_metrics(
        validation_mu,
        validation_log_variance,
        normalization_divisor=config.kl_normalization_divisor,
        sample_reconstruction_variability=variability,
    )
    latent_probe = baseline.fit_latent_probe_model(
        train_mu,
        train_targets,
        baseline.stable_seed(run_seed, "latent_probe"),
    )
    clean_prediction = latent_probe.predict(validation_mu).astype(str)
    clean_classification = baseline.classification_summary(
        validation_targets, clean_prediction
    )
    corrupted_values = baseline_selection.deterministic_corrupted_matrix(
        validation_values,
        dataset.observation_uid[validation_indices],
        "composite",
        1.0,
    )
    corrupted_reconstruction, corrupted_mu, _ = adequacy.outputs(
        model, corrupted_values, metric_device
    )
    corrupted_metrics = baseline.aggregate_reconstruction_metrics(
        baseline.reconstruction_metrics(
            validation_values,
            corrupted_reconstruction,
            dataset.observation_uid[validation_indices],
            repeatable_validation,
        )
    )
    corrupted_prediction = latent_probe.predict(corrupted_mu).astype(str)
    corrupted_classification = baseline.classification_summary(
        validation_targets, corrupted_prediction
    )
    train_instrument = (
        manifest.iloc[train_indices]["instrument"].astype(str).to_numpy()
    )
    validation_instrument = (
        manifest.iloc[validation_indices]["instrument"].astype(str).to_numpy()
    )
    train_sensor = (
        manifest.iloc[train_indices]["sensor_family"].astype(str).to_numpy()
    )
    validation_sensor = (
        manifest.iloc[validation_indices]["sensor_family"].astype(str).to_numpy()
    )
    latent_instrument = baseline_selection.safe_domain_probe(
        train_mu,
        validation_mu,
        train_targets,
        validation_targets,
        train_instrument,
        validation_instrument,
        baseline.stable_seed(run_seed, "instrument_probe"),
    )
    latent_sensor = baseline_selection.safe_domain_probe(
        train_mu,
        validation_mu,
        train_targets,
        validation_targets,
        train_sensor,
        validation_sensor,
        baseline.stable_seed(run_seed, "sensor_probe"),
    )
    validation_manifest = manifest.iloc[validation_indices].reset_index(drop=True)
    latent_geometry = baseline.geometry_metrics(
        validation_mu,
        validation_manifest,
        np.ones(len(validation_manifest), dtype=bool),
    )
    raw_geometry = raw_context["geometry"]
    raw_instrument = raw_context["instrument_probe"]
    raw_sensor = raw_context["sensor_probe"]
    return {
        **config.record(),
        "checkpoint_epoch": int(checkpoint_epoch),
        "representation": representation,
        "model_family": "vae_adequacy",
        "run_seed": int(run_seed),
        "n_train": int(len(train_indices)),
        "n_validation": int(len(validation_indices)),
        "parameter_count": baseline.model_parameter_count(model),
        "state_sha256": baseline.state_dict_sha256(state),
        "raw_target_balanced_accuracy": raw_context["target_classification"][
            "balanced_accuracy"
        ],
        "latent_target_balanced_accuracy": clean_classification[
            "balanced_accuracy"
        ],
        "latent_target_macro_f1_supported": clean_classification[
            "macro_f1_supported"
        ],
        "latent_target_macro_f1_union": clean_classification["macro_f1_union"],
        "corrupted_target_balanced_accuracy": corrupted_classification[
            "balanced_accuracy"
        ],
        "corrupted_prediction_agreement": float(
            np.mean(corrupted_prediction == clean_prediction)
        ),
        "corrupted_latent_cosine_drift": float(
            np.mean(baseline_selection.cosine_drift(validation_mu, corrupted_mu))
        ),
        "raw_instrument_probe_increment": raw_instrument[
            "increment_over_target_only"
        ],
        "latent_instrument_probe_increment": latent_instrument[
            "increment_over_target_only"
        ],
        "instrument_probe_increment_delta": (
            latent_instrument["increment_over_target_only"]
            - raw_instrument["increment_over_target_only"]
        ),
        "raw_sensor_probe_increment": raw_sensor["increment_over_target_only"],
        "latent_sensor_probe_increment": latent_sensor[
            "increment_over_target_only"
        ],
        "sensor_probe_increment_delta": (
            latent_sensor["increment_over_target_only"]
            - raw_sensor["increment_over_target_only"]
        ),
        "raw_same_master_cross_instrument_distance": raw_geometry[
            "same_master_cross_domain_mean_distance"
        ],
        "latent_same_master_cross_instrument_distance": latent_geometry[
            "same_master_cross_domain_mean_distance"
        ],
        "same_master_distance_delta": (
            latent_geometry["same_master_cross_domain_mean_distance"]
            - raw_geometry["same_master_cross_domain_mean_distance"]
        ),
        "latent_different_target_cross_instrument_distance": latent_geometry[
            "different_target_cross_domain_mean_distance"
        ],
        "latent_cross_instrument_separation_margin": latent_geometry[
            "cross_domain_separation_margin"
        ],
        **variational,
        **{f"clean_{key}": value for key, value in clean_metrics.items()},
        **{
            f"corrupted_{key}": value
            for key, value in corrupted_metrics.items()
        },
    }


def cache_path(
    output_dir: Path, stage: str, run_identifier: str
) -> Path:
    return output_dir / "selection_cache" / stage / f"{run_identifier}.pt"


def run_grid(
    dataset: baseline.SpectralDataset,
    representation: str,
    subset: str,
    configs: Sequence[adequacy.AdequacyConfig],
    checkpoints: Sequence[int],
    output_dir: Path,
    stage: str,
    fingerprint: str,
    training_device: torch.device,
    metric_device: torch.device,
    original_seeds: dict[tuple[int, int], int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = dataset.manifest
    folds = manifest["grouped_sample_fold_5"].to_numpy(dtype=int)
    quality = manifest["include_sers_qc_pass"].astype(bool).to_numpy()
    subset_mask = (
        np.ones(len(manifest), dtype=bool)
        if subset == "strict_core"
        else quality
    )
    records: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    for outer_fold in range(5):
        for inner_fold in sorted(set(range(5)) - {outer_fold}):
            train_mask = (
                subset_mask & (folds != outer_fold) & (folds != inner_fold)
            )
            validation_mask = subset_mask & (folds == inner_fold)
            development_mask = subset_mask & (folds != outer_fold)
            repeatable = baseline_selection.repeatable_for_validation(
                dataset.representations[representation],
                manifest,
                development_mask,
                validation_mask,
            )
            raw_context = baseline_selection.raw_baseline_context(
                dataset,
                representation,
                train_mask,
                validation_mask,
                baseline.stable_seed(
                    adequacy.PROTOCOL_VERSION,
                    stage,
                    subset,
                    outer_fold,
                    inner_fold,
                    representation,
                    "raw_context",
                ),
            )
            for config in configs:
                run_seed = (
                    original_seeds[(outer_fold, inner_fold)]
                    if original_seeds is not None
                    else baseline.stable_seed(
                        adequacy.PROTOCOL_VERSION,
                        stage,
                        subset,
                        outer_fold,
                        inner_fold,
                        representation,
                        config.identifier,
                        1729,
                    )
                )
                run_identifier = (
                    f"{subset}__o{outer_fold}__i{inner_fold}__"
                    f"{representation}__{config.identifier}"
                )
                path = cache_path(output_dir, stage, run_identifier)
                if path.exists():
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    if payload["execution_fingerprint"] != fingerprint:
                        raise ValueError(
                            f"Stale adequacy cache: {run_identifier}"
                        )
                    history = payload["history"]
                    states = payload["states"]
                else:
                    history, states, optimizer_states = (
                        adequacy.train_registered_checkpoints(
                            dataset.representations[representation][train_mask],
                            dataset.observation_uid[train_mask],
                            dataset.representations[representation][validation_mask],
                            dataset.observation_uid[validation_mask],
                            config,
                            run_seed,
                            checkpoints,
                            training_device,
                        )
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "execution_fingerprint": fingerprint,
                            "config": config.record(),
                            "run_seed": run_seed,
                            "history": history,
                            "states": states,
                            "optimizer_states": optimizer_states,
                        },
                        path,
                    )
                history = history.copy()
                history["stage"] = stage
                history["subset"] = subset
                history["outer_fold"] = outer_fold
                history["inner_fold"] = inner_fold
                history["representation"] = representation
                history["configuration"] = config.identifier
                history["run_identifier"] = run_identifier
                histories.append(history)
                for checkpoint in checkpoints:
                    record = evaluate_checkpoint(
                        dataset,
                        representation,
                        train_mask,
                        validation_mask,
                        config,
                        int(checkpoint),
                        states[int(checkpoint)],
                        run_seed,
                        repeatable,
                        raw_context,
                        metric_device,
                    )
                    record.update(
                        {
                            "stage": stage,
                            "subset": subset,
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                        }
                    )
                    records.append(record)
                print(
                    json.dumps(
                        {
                            "stage": stage,
                            "outer": outer_fold,
                            "inner": inner_fold,
                            "configuration": config.identifier,
                            "last_epoch": max(checkpoints),
                            "last_ba": records[-1][
                                "latent_target_balanced_accuracy"
                            ],
                            "last_correlation": records[-1][
                                "clean_reconstruction_median_row_correlation"
                            ],
                            "last_peak_recall": records[-1][
                                "clean_repeatable_peak_recall"
                            ],
                        }
                    ),
                    flush=True,
                )
    return pd.DataFrame(records), pd.concat(histories, ignore_index=True)


def aggregate_checkpoint_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "identifier",
        "architecture",
        "latent_dimension",
        "reconstruction_loss",
        "beta_target",
        "optimizer_policy",
        "maximum_epoch",
        "checkpoint_epoch",
        "representation",
        "model_family",
        "subset",
    ]
    excluded = set(
        identifiers
        + [
            "stage",
            "outer_fold",
            "inner_fold",
            "run_seed",
            "state_sha256",
        ]
    )
    numeric = [
        column
        for column in frame.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    result = frame.groupby(identifiers, as_index=False)[numeric].mean()
    counts = (
        frame.groupby(identifiers).size().rename("fold_count").reset_index()
    )
    return result.merge(counts, on=identifiers, how="left")


def apply_gates(
    summary: pd.DataFrame,
    protocol: dict[str, Any],
    ae_reference: dict[str, Any],
) -> pd.DataFrame:
    result = summary.copy()
    gates = protocol["eligibility_gates"]
    ae = ae_reference["strict_core_metrics"]
    result["gate_clean_correlation"] = (
        result["clean_reconstruction_median_row_correlation"]
        >= float(ae["clean_reconstruction_median_row_correlation"])
        - float(gates["clean_correlation_drop_maximum"])
    )
    result["gate_peak_recall"] = (
        result["clean_repeatable_peak_recall"]
        >= float(ae["clean_repeatable_peak_recall"])
        - float(gates["repeatable_peak_recall_drop_maximum"])
    )
    result["gate_chemical_probe"] = (
        result["latent_target_balanced_accuracy"]
        >= float(ae["latent_target_balanced_accuracy"])
        - float(gates["chemical_balanced_accuracy_drop_maximum"])
    )
    result["gate_instrument_probe"] = (
        result["latent_instrument_probe_increment"]
        <= float(ae["latent_instrument_probe_increment"])
        + float(gates["instrument_probe_increment_increase_maximum"])
    )
    result["gate_same_master_distance"] = (
        result["latent_same_master_cross_instrument_distance"]
        <= float(ae["latent_same_master_cross_instrument_distance"])
        + float(gates["same_master_distance_increase_maximum"])
    )
    result["gate_active_units"] = (
        result["vae_active_units_var_mu_gt_0_01"]
        >= int(gates["minimum_active_units"])
    )
    result["gate_kl_dimensions"] = (
        result["vae_dimensions_mean_kl_gt_0_01"]
        >= int(gates["minimum_kl_dimensions"])
    )
    result["gate_kl_range"] = result[
        "vae_kl_unnormalized_per_observation"
    ].between(
        float(gates["minimum_kl_per_observation"]),
        float(gates["maximum_kl_per_observation"]),
    )
    finite_columns = [
        "latent_target_balanced_accuracy",
        "clean_reconstruction_median_row_correlation",
        "clean_reconstruction_mse",
        "clean_repeatable_peak_recall",
        "vae_kl_unnormalized_per_observation",
    ]
    result["gate_finite"] = np.isfinite(
        result[finite_columns].to_numpy(dtype=float)
    ).all(axis=1)
    gate_columns = [column for column in result if column.startswith("gate_")]
    result["gate_count"] = result[gate_columns].astype(int).sum(axis=1)
    result["gate_total"] = len(gate_columns)
    result["passes_all_gates"] = result[gate_columns].all(axis=1)
    result["kl_health_score"] = result[
        [
            "gate_active_units",
            "gate_kl_dimensions",
            "gate_kl_range",
            "gate_finite",
        ]
    ].astype(float).mean(axis=1)
    objectives = {
        "latent_target_balanced_accuracy": ("max", 0.25),
        "clean_reconstruction_median_row_correlation": ("max", 0.075),
        "clean_reconstruction_mse": ("min", 0.075),
        "clean_repeatable_peak_recall": ("max", 0.15),
        "corrupted_target_balanced_accuracy": ("max", 0.05),
        "corrupted_prediction_agreement": ("max", 0.05),
        "latent_instrument_probe_increment": ("min", 0.10),
        "latent_same_master_cross_instrument_distance": ("min", 0.10),
        "vae_active_units_var_mu_gt_0_01": ("max", 0.10),
        "kl_health_score": ("max", 0.05),
    }
    result["selection_utility"] = baseline_selection.normalize_utility(
        result, objectives
    )
    return result


def convergence_table(
    histories: pd.DataFrame, summary: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["configuration", "outer_fold", "inner_fold"]
    for key, run in histories.groupby(keys):
        indexed = run.set_index("epoch")
        for checkpoint in sorted(
            value
            for value in summary["checkpoint_epoch"].unique()
            if value >= 150
        ):
            if checkpoint not in indexed.index or checkpoint - 50 not in indexed.index:
                continue
            before = float(indexed.loc[checkpoint - 50, "validation_loss"])
            after = float(indexed.loc[checkpoint, "validation_loss"])
            rows.append(
                {
                    "identifier": key[0],
                    "outer_fold": key[1],
                    "inner_fold": key[2],
                    "checkpoint_epoch": int(checkpoint),
                    "validation_elbo_relative_improvement_50": (
                        (before - after) / max(abs(before), 1.0e-12)
                    ),
                }
            )
    per_fold = pd.DataFrame(rows)
    aggregate = (
        per_fold.groupby(["identifier", "checkpoint_epoch"])[
            "validation_elbo_relative_improvement_50"
        ]
        .agg(
            median_relative_improvement_50="median",
            mean_relative_improvement_50="mean",
            maximum_relative_improvement_50="max",
            folds_improving_at_least_1_percent=lambda values: float(
                np.mean(values >= 0.01)
            ),
            fold_count="size",
        )
        .reset_index()
    )
    aggregate["converged"] = (
        aggregate["median_relative_improvement_50"] < 0.005
    ) & (
        aggregate["folds_improving_at_least_1_percent"] < 0.25
    )
    return aggregate


def verify_first_100_reproduction(
    histories: pd.DataFrame, standard_bundle: Path
) -> dict[str, Any]:
    old = pd.read_csv(standard_bundle / "selection_training_histories.csv")
    old = old[
        (old["subset"] == "strict_core")
        & (old["representation"] == "arpls_minmax")
        & (old["kl_schedule"] == "cyclical_4")
    ]
    metrics = [
        "beta",
        "train_loss",
        "train_reconstruction_loss",
        "train_kl_unnormalized",
        "train_kl_normalized",
        "validation_loss",
        "validation_reconstruction_loss",
        "validation_kl_unnormalized",
        "validation_kl_normalized",
    ]
    comparisons: list[dict[str, Any]] = []
    for (outer_fold, inner_fold), old_run in old.groupby(
        ["outer_fold", "inner_fold"]
    ):
        new_run = histories[
            (histories["outer_fold"] == outer_fold)
            & (histories["inner_fold"] == inner_fold)
            & histories["configuration"].str.contains("__constant_lr__")
        ]
        shared = old_run.merge(new_run, on="epoch", suffixes=("_old", "_new"))
        maximum = 0.0
        for metric in metrics:
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            shared[f"{metric}_old"].to_numpy(dtype=float)
                            - shared[f"{metric}_new"].to_numpy(dtype=float)
                        )
                    )
                ),
            )
        comparisons.append(
            {
                "outer_fold": int(outer_fold),
                "inner_fold": int(inner_fold),
                "old_epoch_count": int(len(old_run)),
                "shared_epoch_count": int(len(shared)),
                "maximum_absolute_difference": maximum,
            }
        )
    frame = pd.DataFrame(comparisons)
    return {
        "run_count": int(len(frame)),
        "all_shared_histories_exact": bool(
            (frame["maximum_absolute_difference"] == 0).all()
        ),
        "all_shared_histories_within_predeclared_tolerance": bool(
            (frame["maximum_absolute_difference"] <= 1.0e-12).all()
        ),
        "predeclared_history_tolerance": 1.0e-12,
        "maximum_absolute_difference": float(
            frame["maximum_absolute_difference"].max()
        ),
        "minimum_shared_epoch_count": int(frame["shared_epoch_count"].min()),
        "per_fold": frame.to_dict(orient="records"),
    }


def run_convergence(
    protocol: dict[str, Any],
    protocol_path: Path,
    dataset: baseline.SpectralDataset,
    standard_bundle: Path,
    baseline_bundle: Path,
    output_dir: Path,
    training_device: torch.device,
    amendment_path: Path | None = None,
    amendment: dict[str, Any] | None = None,
) -> None:
    fingerprint = execution_fingerprint(protocol_path, amendment_path)
    convergence = (
        amendment["stage_1b_convergence_extension"]
        if amendment is not None
        else protocol["stage_1_convergence"]
    )
    checkpoints = [int(value) for value in convergence["metric_checkpoints"]]
    maximum_epoch = int(
        convergence.get(
            "maximum_epoch", protocol["stage_1_convergence"]["maximum_epoch"]
        )
    )
    configs = [
        adequacy.AdequacyConfig(
            optimizer_policy=policy, maximum_epoch=maximum_epoch
        )
        for policy in convergence["optimizer_policies"]
    ]
    records, histories = run_grid(
        dataset,
        "arpls_minmax",
        "strict_core",
        configs,
        checkpoints,
        output_dir,
        "stage_1_convergence",
        fingerprint,
        training_device,
        torch.device("cpu"),
        original_seed_lookup(standard_bundle),
    )
    records.to_csv(output_dir / "checkpoint_metrics.csv", index=False)
    histories.to_csv(output_dir / "training_histories.csv", index=False)
    ae_reference = json.loads(
        (baseline_bundle / "selected_configurations.json").read_text()
    )["autoencoders"]["arpls_minmax"]
    summary = apply_gates(
        aggregate_checkpoint_metrics(records), protocol, ae_reference
    )
    convergence_summary = convergence_table(histories, summary)
    summary = summary.merge(
        convergence_summary,
        on=["identifier", "checkpoint_epoch"],
        how="left",
    )
    summary.to_csv(output_dir / "stage_summaries.csv", index=False)
    reproduction = verify_first_100_reproduction(histories, standard_bundle)
    write_json(output_dir / "first_100_reproduction.json", reproduction)
    candidates = summary[
        (summary["checkpoint_epoch"] >= 150)
        & summary["converged"].fillna(False)
    ].copy()
    if candidates.empty:
        candidates = summary[
            summary["checkpoint_epoch"] == max(checkpoints)
        ].copy()
    candidates = candidates.sort_values(
        [
            "passes_all_gates",
            "gate_count",
            "selection_utility",
            "parameter_count",
            "identifier",
        ],
        ascending=[False, False, False, True, True],
    )
    winner = candidates.iloc[0]
    strict_pass = bool(winner["passes_all_gates"])
    decision = {
        "stage": "stage_1_convergence",
        "selection_closed": True,
        "selected_identifier": str(winner["identifier"]),
        "selected_checkpoint_epoch": int(winner["checkpoint_epoch"]),
        "selected_optimizer_policy": str(winner["optimizer_policy"]),
        "strict_core_passes_all_gates": strict_pass,
        "architecture_ablation_required": not strict_pass,
        "selected_metrics": json_clean(winner.to_dict()),
        "first_100_reproduction": reproduction,
        "outer_used": False,
        "field_quality_stress_used": False,
        "domain_used": False,
        "poster_used": False,
    }
    write_json(output_dir / "stage_1_decision.json", decision)


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--amendment",
        type=Path,
        default=repository
        / "configs"
        / "sers_vae_adequacy_v1_amendment_1.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repository / "configs" / "sers_vae_adequacy_v1.json",
    )
    parser.add_argument(
        "--nato-bundle",
        type=Path,
        default=repository / "Workspace" / "nato_sers_field_trial" / "preprocessing_v2",
    )
    parser.add_argument(
        "--standard-bundle",
        type=Path,
        default=repository / "Workspace" / "sers_standard_vae" / "standard_vae_v1",
    )
    parser.add_argument(
        "--baseline-bundle",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_representation_baselines"
        / "baselines_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "Workspace" / "sers_vae_adequacy" / "adequacy_v1",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    protocol = load_protocol(args.protocol)
    amendment = json.loads(args.amendment.read_text())
    if (
        amendment["amendment_version"]
        != "sers-vae-adequacy-v1-amendment-1"
        or amendment["status_before_amended_model_execution"] != "predeclared"
    ):
        raise ValueError("Unexpected or unregistered adequacy amendment")
    verify_inputs(
        protocol, args.nato_bundle, args.standard_bundle, args.baseline_bundle
    )
    initialize_output(
        args.output_dir,
        args.protocol,
        args.nato_bundle,
        args.standard_bundle,
        args.baseline_bundle,
        args.amendment,
    )
    audit = existing_run_audit(args.standard_bundle)
    write_json(args.output_dir / "existing_run_audit.json", audit)
    dataset = baseline.load_nato_dataset(args.nato_bundle)
    run_convergence(
        protocol,
        args.protocol,
        dataset,
        args.standard_bundle,
        args.baseline_bundle,
        args.output_dir,
        torch.device(args.device),
        args.amendment,
        amendment,
    )


if __name__ == "__main__":
    main()
