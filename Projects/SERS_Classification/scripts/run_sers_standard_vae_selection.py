#!/usr/bin/env python3
"""Run leakage-safe inner-only selection for SERS standard-VAE protocol v1."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

import run_sers_representation_baselines as baseline_selection
import sers_baseline_common as baseline
import sers_vae_common as vae


CACHE_SCHEMA = "sers-standard-vae-selection-cache-v1"
SCHEMA_ONLY_COMPATIBLE_FINGERPRINTS = {
    # Initial 60-run core screen. The only subsequent source change corrected
    # access to the already-frozen JSON key
    # ``minimum_dimensions_mean_kl_gt_0.01`` during post-training gate
    # aggregation; no training or evaluation function changed.
    "6357fb532d5267a910f1e1df9b0453ea0556daeec113c9b9317069dc620bc167"
}


def json_clean(value: Any) -> Any:
    return baseline_selection.json_clean(value)


def write_json(path: Path, value: Any) -> None:
    baseline_selection.write_json(path, value)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol["protocol_version"] != vae.PROTOCOL_VERSION:
        raise ValueError("Unexpected standard-VAE protocol version")
    if protocol["status_before_model_execution"] != "predeclared":
        raise ValueError("Protocol was not frozen before model execution")
    return protocol


def verify_frozen_inputs(
    protocol: dict[str, Any],
    nato_bundle: Path,
    baseline_bundle: Path,
    poster_csv: Path,
) -> None:
    frozen = protocol["immutable_inputs"]
    paths = {
        "nato_artifact_catalog_sha256": nato_bundle / "artifact_hashes.json",
        "baseline_selected_configurations_sha256": (
            baseline_bundle / "selected_configurations.json"
        ),
        "baseline_final_decisions_sha256": (
            baseline_bundle / "final_decisions.json"
        ),
        "baseline_artifact_catalog_sha256": (
            baseline_bundle / "artifact_hashes.json"
        ),
        "poster_source_sha256": poster_csv,
    }
    for field, path in paths.items():
        actual = baseline.sha256_file(path)
        if actual != frozen[field]:
            raise ValueError(
                f"Frozen input mismatch for {path}: "
                f"expected {frozen[field]}, got {actual}"
            )
    baseline.verify_hash_catalog(nato_bundle)


def execution_fingerprint(protocol_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        protocol_path.resolve(),
        Path(baseline.__file__).resolve(),
        Path(vae.__file__).resolve(),
        Path(__file__).resolve(),
    ):
        digest.update(str(path).encode())
        digest.update(baseline.sha256_file(path).encode())
    digest.update(CACHE_SCHEMA.encode())
    for function in (
        evaluate_vae_run,
        run_fold_grid,
        aggregate_records,
        apply_gates,
    ):
        digest.update(inspect.getsource(function).encode())
    return digest.hexdigest()


def initialize_output(
    output_dir: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    nato_bundle: Path,
    baseline_bundle: Path,
    poster_csv: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = output_dir / "predeclared_protocol.json"
    if copied.exists() and copied.read_bytes() != protocol_path.read_bytes():
        raise ValueError("Output directory contains a different protocol")
    if not copied.exists():
        shutil.copyfile(protocol_path, copied)
    write_json(
        output_dir / "input_hashes.json",
        {
            "protocol": {
                "path": str(protocol_path),
                "sha256": baseline.sha256_file(protocol_path),
            },
            "nato_artifact_catalog": {
                "path": str(nato_bundle / "artifact_hashes.json"),
                "sha256": baseline.sha256_file(
                    nato_bundle / "artifact_hashes.json"
                ),
            },
            "baseline_artifact_catalog": {
                "path": str(baseline_bundle / "artifact_hashes.json"),
                "sha256": baseline.sha256_file(
                    baseline_bundle / "artifact_hashes.json"
                ),
            },
            "baseline_selected_configurations": {
                "path": str(baseline_bundle / "selected_configurations.json"),
                "sha256": baseline.sha256_file(
                    baseline_bundle / "selected_configurations.json"
                ),
            },
            "poster_source": {
                "path": str(poster_csv),
                "sha256": baseline.sha256_file(poster_csv),
            },
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
            "canonical_final_inference": "cpu",
        },
    )


def config_from_protocol(
    protocol: dict[str, Any], schedule: str
) -> vae.VAETrainingConfig:
    frozen = protocol["frozen_model"]
    training = frozen["training"]
    return vae.VAETrainingConfig(
        channels=tuple(int(value) for value in frozen["channels"]),
        latent_dimension=int(frozen["latent_dimension"]),
        reconstruction_loss=str(frozen["reconstruction_loss"]["name"]),
        kl_schedule=schedule,
        beta=float(frozen["beta"]),
        kl_normalization_divisor=int(frozen["kl_normalization_divisor"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        batch_size=int(training["batch_size"]),
        maximum_epochs=int(training["maximum_epochs"]),
        minimum_epochs=int(training["minimum_epochs"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        early_stopping_minimum_delta=float(
            training["early_stopping_minimum_delta"]
        ),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
    )


def evaluate_vae_run(
    dataset: baseline.SpectralDataset,
    representation: str,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    config: vae.VAETrainingConfig,
    run_seed: int,
    device: torch.device,
    repeatable_validation: list[set[int]],
    raw_context: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    values = dataset.representations[representation]
    manifest = dataset.manifest
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    train_values = values[train_indices]
    validation_values = values[validation_indices]
    train_uids = dataset.observation_uid[train_indices]
    validation_uids = dataset.observation_uid[validation_indices]
    train_targets = (
        manifest.iloc[train_indices]["target_analyte"].astype(str).to_numpy()
    )
    validation_targets = (
        manifest.iloc[validation_indices]["target_analyte"]
        .astype(str)
        .to_numpy()
    )

    trained = vae.train_vae(
        train_values,
        train_uids,
        validation_values,
        validation_uids,
        config,
        run_seed,
        device,
    )
    train_reconstruction, train_mu, train_log_variance = vae.vae_outputs(
        trained.model, train_values, device
    )
    del train_reconstruction, train_log_variance
    validation_reconstruction, validation_mu, validation_log_variance = (
        vae.vae_outputs(trained.model, validation_values, device)
    )
    clean_per_row = baseline.reconstruction_metrics(
        validation_values,
        validation_reconstruction,
        validation_uids,
        repeatable_validation,
    )
    clean_metrics = baseline.aggregate_reconstruction_metrics(clean_per_row)
    sample_variability = vae.posterior_sample_reconstruction_variability(
        trained.model,
        validation_values,
        device,
        baseline.stable_seed(run_seed, "sample_variability"),
    )
    variational = vae.variational_metrics(
        validation_mu,
        validation_log_variance,
        normalization_divisor=config.kl_normalization_divisor,
        sample_reconstruction_variability=sample_variability,
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
    raw_classification = raw_context["target_classification"]

    corrupted_values = baseline_selection.deterministic_corrupted_matrix(
        validation_values,
        validation_uids,
        "composite",
        1.0,
    )
    corrupted_reconstruction, corrupted_mu, _ = vae.vae_outputs(
        trained.model, corrupted_values, device
    )
    corrupted_metrics = baseline.aggregate_reconstruction_metrics(
        baseline.reconstruction_metrics(
            validation_values,
            corrupted_reconstruction,
            validation_uids,
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

    record: dict[str, Any] = {
        "configuration": config.identifier,
        "representation": representation,
        "model_family": "vae",
        "curriculum": "clean",
        "kl_schedule": config.kl_schedule,
        "beta": config.beta,
        "channels": "x".join(str(value) for value in config.channels),
        "bottleneck_dimension": config.latent_dimension,
        "loss_name": config.reconstruction_loss,
        "run_seed": run_seed,
        "n_train": len(train_indices),
        "n_validation": len(validation_indices),
        "parameter_count": trained.parameter_count,
        "best_epoch": trained.best_epoch,
        "best_validation_loss": trained.best_validation_loss,
        "state_sha256": trained.state_sha256,
        "raw_target_balanced_accuracy": raw_classification["balanced_accuracy"],
        "latent_target_balanced_accuracy": clean_classification[
            "balanced_accuracy"
        ],
        "latent_target_macro_f1_supported": clean_classification[
            "macro_f1_supported"
        ],
        "latent_target_macro_f1_union": clean_classification[
            "macro_f1_union"
        ],
        "target_balanced_accuracy_delta_from_raw": (
            clean_classification["balanced_accuracy"]
            - raw_classification["balanced_accuracy"]
        ),
        "corrupted_target_balanced_accuracy": corrupted_classification[
            "balanced_accuracy"
        ],
        "corrupted_prediction_agreement": float(
            np.mean(corrupted_prediction == clean_prediction)
        ),
        "corrupted_latent_cosine_drift": float(
            np.mean(
                baseline_selection.cosine_drift(
                    validation_mu, corrupted_mu
                )
            )
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
    history = trained.history.copy()
    history.insert(0, "configuration", config.identifier)
    history.insert(1, "representation", representation)
    history.insert(2, "kl_schedule", config.kl_schedule)
    history.insert(3, "run_seed", run_seed)
    return record, history


def cache_paths(
    output_dir: Path, stage: str, run_identifier: str
) -> tuple[Path, Path]:
    directory = output_dir / "selection_cache" / stage
    return directory / f"{run_identifier}.json", directory / f"{run_identifier}.csv"


def run_or_load_cached(
    output_dir: Path,
    stage: str,
    run_identifier: str,
    fingerprint: str,
    execute: Callable[[], tuple[dict[str, Any], pd.DataFrame]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    record_path, history_path = cache_paths(output_dir, stage, run_identifier)
    if record_path.exists() and history_path.exists():
        payload = json.loads(record_path.read_text())
        cached_fingerprint = payload.get("execution_fingerprint")
        if (
            cached_fingerprint != fingerprint
            and cached_fingerprint not in SCHEMA_ONLY_COMPATIBLE_FINGERPRINTS
        ):
            raise ValueError(
                f"Stale selection cache for {run_identifier}; use a new output directory"
            )
        record = payload["record"]
        record["cache_status"] = "loaded"
        return record, pd.read_csv(history_path)
    record, history = execute()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        record_path,
        {
            "execution_fingerprint": fingerprint,
            "record": record,
        },
    )
    history.to_csv(history_path, index=False)
    record["cache_status"] = "trained"
    return record, history


def run_fold_grid(
    dataset: baseline.SpectralDataset,
    representation: str,
    subset: str,
    configurations: list[vae.VAETrainingConfig],
    output_dir: Path,
    stage: str,
    fingerprint: str,
    selection_seed: int,
    device: torch.device,
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
                subset_mask
                & (folds != outer_fold)
                & (folds != inner_fold)
            )
            validation_mask = subset_mask & (folds == inner_fold)
            development_mask = subset_mask & (folds != outer_fold)
            repeatable = baseline_selection.repeatable_for_validation(
                dataset.representations[representation],
                manifest,
                development_mask,
                validation_mask,
            )
            raw_seed = baseline.stable_seed(
                vae.PROTOCOL_VERSION,
                dataset.name,
                stage,
                subset,
                outer_fold,
                inner_fold,
                representation,
                "raw_context",
            )
            raw_context = baseline_selection.raw_baseline_context(
                dataset,
                representation,
                train_mask,
                validation_mask,
                raw_seed,
            )
            for config in configurations:
                run_seed = baseline.stable_seed(
                    vae.PROTOCOL_VERSION,
                    dataset.name,
                    stage,
                    subset,
                    outer_fold,
                    inner_fold,
                    representation,
                    config.identifier,
                    selection_seed,
                )
                run_identifier = (
                    f"{subset}__o{outer_fold}__i{inner_fold}__"
                    f"{representation}__{config.identifier}"
                )

                def execute() -> tuple[dict[str, Any], pd.DataFrame]:
                    return evaluate_vae_run(
                        dataset,
                        representation,
                        train_mask,
                        validation_mask,
                        config,
                        run_seed,
                        device,
                        repeatable,
                        raw_context,
                    )

                record, history = run_or_load_cached(
                    output_dir,
                    stage,
                    run_identifier,
                    fingerprint,
                    execute,
                )
                record.update(
                    {
                        "stage": stage,
                        "subset": subset,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                    }
                )
                history["stage"] = stage
                history["subset"] = subset
                history["outer_fold"] = outer_fold
                history["inner_fold"] = inner_fold
                history["run_identifier"] = run_identifier
                records.append(record)
                histories.append(history)
                print(
                    json.dumps(
                        {
                            "stage": stage,
                            "outer": outer_fold,
                            "inner": inner_fold,
                            "representation": representation,
                            "schedule": config.kl_schedule,
                            "target_ba": record[
                                "latent_target_balanced_accuracy"
                            ],
                            "peak_recall": record[
                                "clean_repeatable_peak_recall"
                            ],
                            "active_units": record[
                                "vae_active_units_var_mu_gt_0_01"
                            ],
                            "kl": record[
                                "vae_kl_unnormalized_per_observation"
                            ],
                        }
                    ),
                    flush=True,
                )
    return pd.DataFrame(records), pd.concat(histories, ignore_index=True)


def aggregate_records(frame: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "configuration",
        "representation",
        "model_family",
        "curriculum",
        "kl_schedule",
        "beta",
        "channels",
        "bottleneck_dimension",
        "loss_name",
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
            "cache_status",
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
        frame.groupby(identifiers)
        .size()
        .rename("fold_count")
        .reset_index()
    )
    return result.merge(counts, on=identifiers, how="left")


def apply_gates(
    frame: pd.DataFrame,
    protocol: dict[str, Any],
    ae_reference: dict[str, Any],
    quality_stage: bool,
) -> pd.DataFrame:
    result = frame.copy()
    relative = protocol["eligibility_gates"][
        "relative_to_frozen_arpls_ae"
    ]
    health = protocol["eligibility_gates"]["variational_health"]
    ae_metrics = ae_reference["strict_core_metrics"]
    result["gate_clean_correlation"] = (
        result["clean_reconstruction_median_row_correlation"]
        >= float(ae_metrics["clean_reconstruction_median_row_correlation"])
        - float(relative["clean_median_row_correlation_drop_maximum"])
    )
    result["gate_peak_recall"] = (
        result["clean_repeatable_peak_recall"]
        >= float(ae_metrics["clean_repeatable_peak_recall"])
        - float(relative["clean_repeatable_peak_recall_drop_maximum"])
    )
    result["gate_chemical_probe"] = (
        result["latent_target_balanced_accuracy"]
        >= float(ae_metrics["latent_target_balanced_accuracy"])
        - float(relative["chemical_probe_balanced_accuracy_drop_maximum"])
    )
    result["gate_instrument_probe"] = (
        result["latent_instrument_probe_increment"]
        <= float(ae_metrics["latent_instrument_probe_increment"])
        + float(
            relative[
                "target_adjusted_instrument_probe_increment_increase_maximum"
            ]
        )
    )
    result["gate_same_master_distance"] = (
        result["latent_same_master_cross_instrument_distance"]
        <= float(ae_metrics["latent_same_master_cross_instrument_distance"])
        + float(
            relative[
                "same_master_cross_instrument_distance_increase_maximum"
            ]
        )
    )
    result["gate_active_units"] = (
        result["vae_active_units_var_mu_gt_0_01"]
        >= int(health["minimum_active_units_var_mu_gt_0.01"])
        if "minimum_active_units_var_mu_gt_0.01" in health
        else result["vae_active_units_var_mu_gt_0_01"]
        >= int(health["minimum_active_units_var_mu_gt_0_01"])
    )
    result["gate_kl_dimensions"] = (
        result["vae_dimensions_mean_kl_gt_0_01"]
        >= int(health["minimum_dimensions_mean_kl_gt_0.01"])
    )
    result["gate_kl_range"] = result[
        "vae_kl_unnormalized_per_observation"
    ].between(
        float(health["minimum_unnormalized_kl_per_observation"]),
        float(health["maximum_unnormalized_kl_per_observation"]),
    )
    finite_columns = [
        "latent_target_balanced_accuracy",
        "clean_reconstruction_mse",
        "clean_reconstruction_median_row_correlation",
        "clean_repeatable_peak_recall",
        "vae_kl_unnormalized_per_observation",
    ]
    result["gate_finite"] = np.isfinite(
        result[finite_columns].to_numpy(dtype=float)
    ).all(axis=1)
    gate_columns = [
        "gate_clean_correlation",
        "gate_peak_recall",
        "gate_chemical_probe",
        "gate_instrument_probe",
        "gate_same_master_distance",
        "gate_active_units",
        "gate_kl_dimensions",
        "gate_kl_range",
        "gate_finite",
    ]
    if quality_stage:
        result["gate_quality_sensitivity"] = (
            result["quality_target_drop_from_core"]
            >= -float(
                protocol["eligibility_gates"]["quality_sensitivity"][
                    "chemical_probe_drop_from_strict_core_maximum"
                ]
            )
        )
        gate_columns.append("gate_quality_sensitivity")
    result["passes_vae_gates"] = result[gate_columns].all(axis=1)
    result["kl_health_score"] = (
        result[
            [
                "gate_active_units",
                "gate_kl_dimensions",
                "gate_kl_range",
                "gate_finite",
            ]
        ]
        .astype(float)
        .mean(axis=1)
    )
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


def run_selection(
    dataset: baseline.SpectralDataset,
    protocol: dict[str, Any],
    protocol_path: Path,
    baseline_bundle: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    fingerprint = execution_fingerprint(protocol_path)
    search = protocol["vae_specific_search"]
    seed = int(search["selection_seed"])
    primary = protocol["frozen_model"]["primary_representation"]
    sensitivity = protocol["frozen_model"][
        "mandatory_sensitivity_representation"
    ]
    all_configs = [
        config_from_protocol(protocol, schedule)
        for schedule in search["candidates"]
    ]
    core_records, core_histories = run_fold_grid(
        dataset,
        primary,
        "strict_core",
        all_configs,
        output_dir,
        "vae_search_core",
        fingerprint,
        seed,
        device,
    )
    ae_reference = json.loads(
        (baseline_bundle / "selected_configurations.json").read_text()
    )["autoencoders"][primary]
    core_summary = apply_gates(
        aggregate_records(core_records),
        protocol,
        ae_reference,
        quality_stage=False,
    )
    eligible = core_summary[core_summary["passes_vae_gates"].astype(bool)]
    carried_pool = eligible if not eligible.empty else core_summary
    carried_schedules = (
        carried_pool.sort_values(
            ["selection_utility", "kl_schedule"],
            ascending=[False, True],
        )
        .head(2)["kl_schedule"]
        .astype(str)
        .tolist()
    )
    config_lookup = {config.kl_schedule: config for config in all_configs}
    quality_records, quality_histories = run_fold_grid(
        dataset,
        primary,
        "quality_pass",
        [config_lookup[name] for name in carried_schedules],
        output_dir,
        "vae_search_quality",
        fingerprint,
        seed,
        device,
    )
    quality_summary = aggregate_records(quality_records)
    merge_keys = [
        "configuration",
        "representation",
        "model_family",
        "curriculum",
        "kl_schedule",
        "beta",
        "channels",
        "bottleneck_dimension",
        "loss_name",
    ]
    carried_core = core_summary[
        core_summary["kl_schedule"].isin(carried_schedules)
    ].copy()
    quality_target = quality_summary[
        merge_keys + ["latent_target_balanced_accuracy"]
    ].rename(
        columns={
            "latent_target_balanced_accuracy": (
                "quality_latent_target_balanced_accuracy"
            )
        }
    )
    combined = carried_core.merge(quality_target, on=merge_keys, how="left")
    combined["quality_target_drop_from_core"] = (
        combined["quality_latent_target_balanced_accuracy"]
        - combined["latent_target_balanced_accuracy"]
    )
    combined = apply_gates(
        combined,
        protocol,
        ae_reference,
        quality_stage=True,
    )
    eligible_final = combined[combined["passes_vae_gates"].astype(bool)]
    final_pool = eligible_final if not eligible_final.empty else combined
    winner = final_pool.sort_values(
        ["selection_utility", "kl_schedule"],
        ascending=[False, True],
    ).iloc[0]
    selected_config = config_lookup[str(winner["kl_schedule"])]

    minimal_core, minimal_core_histories = run_fold_grid(
        dataset,
        sensitivity,
        "strict_core",
        [selected_config],
        output_dir,
        "vae_minimal_sensitivity_core",
        fingerprint,
        seed,
        device,
    )
    minimal_quality, minimal_quality_histories = run_fold_grid(
        dataset,
        sensitivity,
        "quality_pass",
        [selected_config],
        output_dir,
        "vae_minimal_sensitivity_quality",
        fingerprint,
        seed,
        device,
    )
    minimal_summary = pd.concat(
        [
            aggregate_records(minimal_core),
            aggregate_records(minimal_quality),
        ],
        ignore_index=True,
    )

    all_records = pd.concat(
        [core_records, quality_records, minimal_core, minimal_quality],
        ignore_index=True,
    ).drop(columns=["cache_status"], errors="ignore").sort_index(axis=1)
    all_histories = pd.concat(
        [
            core_histories,
            quality_histories,
            minimal_core_histories,
            minimal_quality_histories,
        ],
        ignore_index=True,
    ).sort_index(axis=1)
    all_records.to_csv(
        output_dir / "selection_fold_metrics.csv", index=False
    )
    all_histories.to_csv(
        output_dir / "selection_training_histories.csv", index=False
    )
    core_summary.to_csv(output_dir / "selection_core_summary.csv", index=False)
    combined.to_csv(output_dir / "selection_summary.csv", index=False)
    minimal_summary.to_csv(
        output_dir / "minimal_sensitivity_summary.csv", index=False
    )
    selected = {
        **vae.config_record(selected_config),
        "selection_data": "NATO master-sample-grouped nested inner validation only",
        "selection_representation": primary,
        "mandatory_sensitivity_representation": sensitivity,
        "carried_to_quality": carried_schedules,
        "eligible_to_advance": bool(winner["passes_vae_gates"]),
        "selection_utility": float(winner["selection_utility"]),
        "strict_core_and_quality_metrics": json_clean(winner.to_dict()),
        "selection_closed": True,
        "outer_used": False,
        "field_quality_stress_used": False,
        "domain_used": False,
        "poster_used": False,
    }
    write_json(output_dir / "selected_configuration.json", selected)
    return selected


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repository / "configs" / "sers_standard_vae_v1.json",
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
        "--baseline-bundle",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_representation_baselines"
        / "baselines_v1",
    )
    parser.add_argument(
        "--poster-csv",
        type=Path,
        default=repository
        / "Workspace"
        / "data"
        / "processed"
        / "consolidated_SERS.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_standard_vae"
        / "standard_vae_v1",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    protocol_path = args.protocol.resolve()
    nato_bundle = args.nato_bundle.resolve()
    baseline_bundle = args.baseline_bundle.resolve()
    poster_csv = args.poster_csv.resolve()
    output_dir = args.output_dir.resolve()
    protocol = load_protocol(protocol_path)
    verify_frozen_inputs(
        protocol, nato_bundle, baseline_bundle, poster_csv
    )
    dataset = baseline.load_nato_dataset(nato_bundle)
    initialize_output(
        output_dir,
        protocol_path,
        protocol,
        nato_bundle,
        baseline_bundle,
        poster_csv,
    )
    device = baseline.resolve_device(args.device)
    selected = run_selection(
        dataset,
        protocol,
        protocol_path,
        baseline_bundle,
        output_dir,
        device,
    )
    print(
        json.dumps(
            {
                "stage": "standard_vae_selection",
                "status": "complete",
                "output_dir": str(output_dir),
                "selected": selected,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
