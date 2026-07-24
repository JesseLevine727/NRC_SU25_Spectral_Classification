#!/usr/bin/env python3
"""Run the frozen SERS classical/Siamese/AE/DAE baseline protocol."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

import sers_baseline_common as common


TRAINING_CACHE_SCHEMA_VERSION = "sers_representation_baselines_training_v1"
LEGACY_COMPATIBLE_CACHE_FINGERPRINTS = {
    # Pre-aggregation-fix fingerprint. The only subsequent code change was to
    # post-training aggregation/cache scoping; model fitting and metric
    # evaluation were unchanged.
    "fe0e71f2b4003a8f253eef49e2f0f289426394ed9695ba7c95e878a3d8f40ba3",
}


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_clean(item) for item in value.tolist()]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_clean(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Non-standard JSON constant: {value}")
        ),
    )
    if protocol["protocol_version"] != common.PROTOCOL_VERSION:
        raise ValueError("Unexpected baseline protocol version")
    if protocol["status_before_model_execution"] != "predeclared":
        raise ValueError("Protocol was not marked predeclared")
    return protocol


def execution_fingerprint(protocol_path: Path) -> str:
    """Fingerprint only code that can change a cached fitted-run result."""
    digest = hashlib.sha256()
    for path in (
        protocol_path.resolve(),
        Path(common.__file__).resolve(),
    ):
        digest.update(str(path).encode("utf-8"))
        digest.update(common.sha256_file(path).encode("utf-8"))
    digest.update(TRAINING_CACHE_SCHEMA_VERSION.encode("utf-8"))
    for function in (
        configuration_grid,
        selected_ae_with_curriculum,
        deterministic_corrupted_matrix,
        repeatable_for_validation,
        safe_domain_probe,
        cosine_drift,
        raw_baseline_context,
        evaluate_autoencoder_run,
        run_search_stage,
    ):
        digest.update(function.__name__.encode("utf-8"))
        digest.update(inspect.getsource(function).encode("utf-8"))
    return digest.hexdigest()


def initialize_output(
    output_dir: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    nato_bundle: Path,
    poster_csv: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_protocol = output_dir / "predeclared_protocol.json"
    if copied_protocol.exists():
        if copied_protocol.read_bytes() != protocol_path.read_bytes():
            raise ValueError(
                "Output directory belongs to a different protocol snapshot"
            )
    else:
        shutil.copyfile(protocol_path, copied_protocol)
    input_hashes = {
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": common.sha256_file(protocol_path),
        },
        "nato_artifact_hash_catalog": {
            "path": str((nato_bundle / "artifact_hashes.json").resolve()),
            "sha256": common.sha256_file(
                nato_bundle / "artifact_hashes.json"
            ),
        },
        "nato_dataset_version": {
            "path": str((nato_bundle / "dataset_version.json").resolve()),
            "sha256": common.sha256_file(nato_bundle / "dataset_version.json"),
        },
        "poster_source": {
            "path": str(poster_csv.resolve()),
            "sha256": common.sha256_file(poster_csv),
        },
        "common_script": {
            "path": str(Path(common.__file__).resolve()),
            "sha256": common.sha256_file(Path(common.__file__).resolve()),
        },
        "orchestrator": {
            "path": str(Path(__file__).resolve()),
            "sha256": common.sha256_file(Path(__file__).resolve()),
        },
    }
    expected_poster = protocol["poster_input"]["source_sha256"]
    if input_hashes["poster_source"]["sha256"] != expected_poster:
        raise ValueError("Poster source hash differs from the predeclared input")
    write_json(output_dir / "input_hashes.json", input_hashes)
    write_json(
        output_dir / "environment.json",
        {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "cublas_workspace_config": common.os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "deterministic_algorithms_required": True,
        },
    )


def configuration_grid(
    protocol: dict[str, Any],
    curriculum: str = "clean",
) -> list[common.AutoencoderTrainingConfig]:
    search = protocol["autoencoder_search"]
    training = search["training"]
    configurations: list[common.AutoencoderTrainingConfig] = []
    for architecture in search["encoder_families"].values():
        channels = tuple(int(value) for value in architecture["channels"])
        for bottleneck in search["bottleneck_dimensions"]:
            for loss_name in search["reconstruction_losses"]:
                configurations.append(
                    common.AutoencoderTrainingConfig(
                        channels=channels,
                        bottleneck_dimension=int(bottleneck),
                        loss_name=str(loss_name),
                        curriculum=curriculum,
                        learning_rate=float(training["learning_rate"]),
                        weight_decay=float(training["weight_decay"]),
                        batch_size=int(training["batch_size"]),
                        maximum_epochs=int(training["maximum_epochs"]),
                        minimum_epochs=int(training["minimum_epochs"]),
                        early_stopping_patience=int(
                            training["early_stopping_patience"]
                        ),
                        early_stopping_minimum_delta=float(
                            training["early_stopping_minimum_delta"]
                        ),
                        gradient_clip_norm=float(
                            training["gradient_clip_norm"]
                        ),
                    )
                )
    return configurations


def selected_ae_with_curriculum(
    selected: dict[str, Any],
    curriculum: str,
) -> common.AutoencoderTrainingConfig:
    return common.AutoencoderTrainingConfig(
        channels=tuple(selected["channels"]),
        bottleneck_dimension=int(selected["bottleneck_dimension"]),
        loss_name=str(selected["loss_name"]),
        curriculum=curriculum,
        learning_rate=float(selected["learning_rate"]),
        weight_decay=float(selected["weight_decay"]),
        batch_size=int(selected["batch_size"]),
        maximum_epochs=int(selected["maximum_epochs"]),
        minimum_epochs=int(selected["minimum_epochs"]),
        early_stopping_patience=int(selected["early_stopping_patience"]),
        early_stopping_minimum_delta=float(
            selected["early_stopping_minimum_delta"]
        ),
        gradient_clip_norm=float(selected["gradient_clip_norm"]),
    )


def deterministic_corrupted_matrix(
    clean: np.ndarray,
    observation_uids: Iterable[str],
    corruption: str,
    severity: float = 1.0,
) -> np.ndarray:
    rows = []
    for values, uid in zip(clean, observation_uids):
        rng = np.random.default_rng(
            common.stable_seed(
                common.PROTOCOL_VERSION,
                "heldout_corruption",
                uid,
                corruption,
                severity,
            )
        )
        rows.append(
            common.apply_corruption(values, corruption, severity, rng)
        )
    return np.asarray(rows, dtype=np.float32)


def repeatable_for_validation(
    values: np.ndarray,
    manifest: pd.DataFrame,
    development_mask: np.ndarray,
    validation_mask: np.ndarray,
) -> list[set[int]]:
    development_indices = np.flatnonzero(development_mask)
    development_manifest = (
        manifest.iloc[development_indices].reset_index(drop=True)
    )
    repeatable = common.repeatable_peak_positions(
        values[development_indices], development_manifest
    )
    position = {
        original: local for local, original in enumerate(development_indices)
    }
    return [repeatable[position[index]] for index in np.flatnonzero(validation_mask)]


def safe_domain_probe(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_targets: np.ndarray,
    validation_targets: np.ndarray,
    train_domains: np.ndarray,
    validation_domains: np.ndarray,
    seed: int,
) -> dict[str, float]:
    if len(np.unique(train_domains)) < 2 or len(np.unique(validation_domains)) < 2:
        return {
            "cell_balanced_accuracy": np.nan,
            "target_only_null_cell_balanced_accuracy": np.nan,
            "increment_over_target_only": np.nan,
        }
    return common.target_adjusted_domain_probe(
        train_features,
        validation_features,
        train_targets,
        validation_targets,
        train_domains,
        validation_domains,
        seed,
    )


def cosine_drift(clean: np.ndarray, corrupted: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(clean, axis=1) * np.linalg.norm(
        corrupted, axis=1
    )
    similarity = np.sum(clean * corrupted, axis=1) / np.maximum(
        denominator, 1.0e-12
    )
    return 1.0 - similarity


def raw_baseline_context(
    dataset: common.SpectralDataset,
    representation: str,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    seed: int,
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
        manifest.iloc[validation_indices]["target_analyte"]
        .astype(str)
        .to_numpy()
    )
    train_projected, validation_projected = common.pca_project_train_test(
        train_values,
        validation_values,
        common.stable_seed(seed, "raw_pca"),
    )
    target_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=common.stable_seed(seed, "raw_target"),
    )
    target_model.fit(train_projected, train_targets)
    raw_prediction = target_model.predict(validation_projected).astype(str)
    raw_classification = common.classification_summary(
        validation_targets, raw_prediction
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
    instrument = safe_domain_probe(
        train_projected,
        validation_projected,
        train_targets,
        validation_targets,
        train_instrument,
        validation_instrument,
        common.stable_seed(seed, "raw_instrument"),
    )
    sensor = safe_domain_probe(
        train_projected,
        validation_projected,
        train_targets,
        validation_targets,
        train_sensor,
        validation_sensor,
        common.stable_seed(seed, "raw_sensor"),
    )
    validation_manifest = (
        manifest.iloc[validation_indices].reset_index(drop=True)
    )
    geometry = common.geometry_metrics(
        validation_values,
        validation_manifest,
        np.ones(len(validation_manifest), dtype=bool),
    )
    return {
        "target_classification": raw_classification,
        "instrument_probe": instrument,
        "sensor_probe": sensor,
        "geometry": geometry,
    }


def evaluate_autoencoder_run(
    dataset: common.SpectralDataset,
    representation: str,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    configuration: common.AutoencoderTrainingConfig,
    run_seed: int,
    device: torch.device,
    repeatable_validation: list[set[int]],
    baseline_context: dict[str, Any],
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

    trained = common.train_autoencoder(
        train_values,
        train_uids,
        validation_values,
        validation_uids,
        configuration,
        run_seed,
        device,
    )
    train_reconstruction, train_latent = common.autoencoder_outputs(
        trained.model, train_values, device
    )
    validation_reconstruction, validation_latent = (
        common.autoencoder_outputs(
            trained.model, validation_values, device
        )
    )
    clean_per_row = common.reconstruction_metrics(
        validation_values,
        validation_reconstruction,
        validation_uids,
        repeatable_validation,
    )
    clean_metrics = common.aggregate_reconstruction_metrics(clean_per_row)

    raw_classification = baseline_context["target_classification"]
    latent_probe = common.fit_latent_probe_model(
        train_latent,
        train_targets,
        common.stable_seed(run_seed, "latent_probe"),
    )
    clean_prediction = latent_probe.predict(validation_latent).astype(str)
    clean_classification = common.classification_summary(
        validation_targets, clean_prediction
    )

    corrupted_values = deterministic_corrupted_matrix(
        validation_values,
        validation_uids,
        "composite",
        1.0,
    )
    corrupted_reconstruction, corrupted_latent = (
        common.autoencoder_outputs(
            trained.model, corrupted_values, device
        )
    )
    corrupted_per_row = common.reconstruction_metrics(
        validation_values,
        corrupted_reconstruction,
        validation_uids,
        repeatable_validation,
    )
    corrupted_metrics = common.aggregate_reconstruction_metrics(
        corrupted_per_row
    )
    corrupted_prediction = latent_probe.predict(corrupted_latent).astype(str)
    corrupted_classification = common.classification_summary(
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
    raw_instrument = baseline_context["instrument_probe"]
    latent_instrument = safe_domain_probe(
        train_latent,
        validation_latent,
        train_targets,
        validation_targets,
        train_instrument,
        validation_instrument,
        common.stable_seed(run_seed, "latent_instrument"),
    )
    raw_sensor = baseline_context["sensor_probe"]
    latent_sensor = safe_domain_probe(
        train_latent,
        validation_latent,
        train_targets,
        validation_targets,
        train_sensor,
        validation_sensor,
        common.stable_seed(run_seed, "latent_sensor"),
    )

    validation_manifest = (
        manifest.iloc[validation_indices].reset_index(drop=True)
    )
    raw_geometry = baseline_context["geometry"]
    latent_geometry = common.geometry_metrics(
        validation_latent,
        validation_manifest,
        np.ones(len(validation_manifest), dtype=bool),
    )
    prediction_agreement = float(
        np.mean(corrupted_prediction == clean_prediction)
    )
    latent_drift = float(
        np.mean(cosine_drift(validation_latent, corrupted_latent))
    )
    record: dict[str, Any] = {
        "configuration": configuration.identifier,
        "representation": representation,
        "model_family": (
            "ae" if configuration.curriculum == "clean" else "dae"
        ),
        "curriculum": configuration.curriculum,
        "channels": "x".join(str(value) for value in configuration.channels),
        "bottleneck_dimension": configuration.bottleneck_dimension,
        "loss_name": configuration.loss_name,
        "run_seed": run_seed,
        "n_train": len(train_indices),
        "n_validation": len(validation_indices),
        "parameter_count": trained.parameter_count,
        "best_epoch": trained.best_epoch,
        "best_validation_loss": trained.best_validation_loss,
        "state_sha256": trained.state_sha256,
        "raw_target_balanced_accuracy": raw_classification[
            "balanced_accuracy"
        ],
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
        "corrupted_prediction_agreement": prediction_agreement,
        "corrupted_latent_cosine_drift": latent_drift,
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
        "raw_sensor_probe_increment": raw_sensor[
            "increment_over_target_only"
        ],
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
        "latent_cross_instrument_separation_margin": latent_geometry[
            "cross_domain_separation_margin"
        ],
        **{f"clean_{key}": value for key, value in clean_metrics.items()},
        **{
            f"corrupted_{key}": value
            for key, value in corrupted_metrics.items()
        },
    }
    history = trained.history.copy()
    history.insert(0, "configuration", configuration.identifier)
    history.insert(1, "representation", representation)
    history.insert(2, "curriculum", configuration.curriculum)
    history.insert(3, "run_seed", run_seed)
    return record, history


def cache_paths(
    output_dir: Path,
    stage: str,
    run_identifier: str,
) -> tuple[Path, Path]:
    directory = output_dir / "search_cache" / stage
    return directory / f"{run_identifier}.json", directory / f"{run_identifier}.csv"


def run_or_load_cached(
    output_dir: Path,
    stage: str,
    run_identifier: str,
    cache_fingerprint: str,
    runner: Any,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metric_path, history_path = cache_paths(
        output_dir, stage, run_identifier
    )
    if metric_path.exists() and history_path.exists():
        record = json.loads(metric_path.read_text())
        recorded_fingerprint = record.get("execution_fingerprint")
        if (
            recorded_fingerprint != cache_fingerprint
            and recorded_fingerprint
            not in LEGACY_COMPATIBLE_CACHE_FINGERPRINTS
        ):
            raise ValueError(f"Stale cache implementation for {run_identifier}")
        return record, pd.read_csv(history_path)
    record, history = runner()
    record["run_identifier"] = run_identifier
    record["execution_fingerprint"] = cache_fingerprint
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(metric_path, record)
    history.to_csv(history_path, index=False)
    return record, history


def normalize_utility(
    frame: pd.DataFrame,
    objectives: dict[str, tuple[str, float]],
) -> np.ndarray:
    utility = np.zeros(len(frame), dtype=float)
    for column, (direction, weight) in objectives.items():
        values = frame[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            normalized = np.full(len(values), 0.5)
        else:
            low = float(np.nanmin(values))
            high = float(np.nanmax(values))
            normalized = (
                np.full(len(values), 0.5)
                if high - low <= 1.0e-12
                else (values - low) / (high - low)
            )
            normalized[~finite] = 0.0
        if direction == "min":
            normalized = 1.0 - normalized
        utility += weight * normalized
    return utility


def aggregate_search(records: pd.DataFrame) -> pd.DataFrame:
    identifier_columns = [
        "model_family",
        "representation",
        "configuration",
        "curriculum",
        "channels",
        "bottleneck_dimension",
        "loss_name",
        "subset",
    ]
    numeric_columns = [
        column
        for column in records.select_dtypes(include=[np.number]).columns
        if column
        not in set(identifier_columns)
        | {
            "outer_fold",
            "inner_fold",
            "run_seed",
            "n_train",
            "n_validation",
            "best_epoch",
        }
    ]
    aggregate = (
        records.groupby(identifier_columns, dropna=False)[numeric_columns]
        .mean()
        .reset_index()
    )
    counts = (
        records.groupby(identifier_columns, dropna=False)
        .size()
        .rename("fold_count")
        .reset_index()
    )
    return aggregate.merge(counts, on=identifier_columns, how="left")


def apply_ae_gates(
    aggregate: pd.DataFrame,
    protocol: dict[str, Any],
    quality_stage: bool,
) -> pd.DataFrame:
    frame = aggregate.copy()
    gates = protocol["eligibility_gates"]["ae"]
    frame["gate_clean_correlation"] = (
        frame["clean_reconstruction_median_row_correlation"]
        >= gates["clean_median_row_correlation_minimum"]
    )
    frame["gate_peak_recall"] = (
        frame["clean_repeatable_peak_recall"]
        >= gates["clean_repeatable_peak_recall_minimum"]
    )
    frame["gate_chemical_probe"] = (
        frame["target_balanced_accuracy_delta_from_raw"]
        >= -gates[
            "chemical_probe_balanced_accuracy_drop_from_same_view_classical_maximum"
        ]
    )
    frame["gate_instrument_leakage"] = (
        frame["instrument_probe_increment_delta"]
        <= gates["target_adjusted_instrument_leakage_increase_maximum"]
    )
    frame["gate_same_master_distance"] = (
        frame["same_master_distance_delta"]
        <= gates[
            "same_master_cross_instrument_correlation_distance_increase_maximum"
        ]
    )
    gate_columns = [
        "gate_clean_correlation",
        "gate_peak_recall",
        "gate_chemical_probe",
        "gate_instrument_leakage",
        "gate_same_master_distance",
    ]
    if quality_stage and "quality_target_drop_from_core" in frame:
        frame["gate_quality_sensitivity"] = (
            frame["quality_target_drop_from_core"]
            >= -gates[
                "quality_pass_chemical_probe_drop_from_strict_core_maximum"
            ]
        )
        gate_columns.append("gate_quality_sensitivity")
    frame["passes_ae_gates"] = frame[gate_columns].all(axis=1)
    objectives = {
        "latent_target_balanced_accuracy": ("max", 0.30),
        "clean_reconstruction_median_row_correlation": ("max", 0.10),
        "clean_reconstruction_mse": ("min", 0.10),
        "corrupted_reconstruction_mse": ("min", 0.20),
        "clean_repeatable_peak_recall": ("max", 0.15),
        "latent_instrument_probe_increment": ("min", 0.10),
        "latent_same_master_cross_instrument_distance": ("min", 0.05),
    }
    frame["selection_utility"] = np.nan
    for representation, indices in frame.groupby("representation").groups.items():
        frame.loc[indices, "selection_utility"] = normalize_utility(
            frame.loc[indices], objectives
        )
    return frame


def top_candidates(
    frame: pd.DataFrame,
    representation: str,
    count: int,
    gate_column: str,
) -> list[str]:
    candidates = frame[frame["representation"] == representation].copy()
    eligible = candidates[candidates[gate_column].astype(bool)]
    pool = eligible if not eligible.empty else candidates
    pool = pool.sort_values(
        ["selection_utility", "parameter_count", "configuration"],
        ascending=[False, True, True],
    )
    return pool["configuration"].head(count).astype(str).tolist()


def run_search_stage(
    dataset: common.SpectralDataset,
    configurations_by_representation: dict[
        str, list[common.AutoencoderTrainingConfig]
    ],
    subset: str,
    output_dir: Path,
    stage: str,
    cache_fingerprint: str,
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
            for representation, configurations in (
                configurations_by_representation.items()
            ):
                repeatable = repeatable_for_validation(
                    dataset.representations[representation],
                    manifest,
                    development_mask,
                    validation_mask,
                )
                baseline_seed = common.stable_seed(
                    common.PROTOCOL_VERSION,
                    dataset.name,
                    stage,
                    subset,
                    outer_fold,
                    inner_fold,
                    representation,
                    "raw_baseline",
                )
                baseline_context = raw_baseline_context(
                    dataset,
                    representation,
                    train_mask,
                    validation_mask,
                    baseline_seed,
                )
                for configuration in configurations:
                    run_seed = common.stable_seed(
                        common.PROTOCOL_VERSION,
                        dataset.name,
                        stage,
                        subset,
                        outer_fold,
                        inner_fold,
                        representation,
                        configuration.identifier,
                        selection_seed,
                    )
                    run_identifier = (
                        f"{subset}__o{outer_fold}__i{inner_fold}__"
                        f"{representation}__{configuration.identifier}"
                    )

                    def execute() -> tuple[dict[str, Any], pd.DataFrame]:
                        return evaluate_autoencoder_run(
                            dataset,
                            representation,
                            train_mask,
                            validation_mask,
                            configuration,
                            run_seed,
                            device,
                            repeatable,
                            baseline_context,
                        )

                    record, history = run_or_load_cached(
                        output_dir,
                        stage,
                        run_identifier,
                        cache_fingerprint,
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
                                "subset": subset,
                                "outer": outer_fold,
                                "inner": inner_fold,
                                "representation": representation,
                                "configuration": configuration.identifier,
                                "target": record[
                                    "latent_target_balanced_accuracy"
                                ],
                                "correlation": record[
                                    "clean_reconstruction_median_row_correlation"
                                ],
                            }
                        ),
                        flush=True,
                    )
    return pd.DataFrame(records), pd.concat(histories, ignore_index=True)


def configuration_lookup(
    configurations: Iterable[common.AutoencoderTrainingConfig],
) -> dict[str, common.AutoencoderTrainingConfig]:
    return {configuration.identifier: configuration for configuration in configurations}


def run_ae_selection(
    dataset: common.SpectralDataset,
    protocol: dict[str, Any],
    protocol_path: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    cache_fingerprint = execution_fingerprint(protocol_path)
    selection_seed = int(protocol["determinism"]["selection_seeds"][0])
    configurations = configuration_grid(protocol, curriculum="clean")
    by_representation = {
        representation: configurations
        for representation in common.INTENSITY_REPRESENTATIONS
    }
    core_records, core_histories = run_search_stage(
        dataset,
        by_representation,
        "strict_core",
        output_dir,
        "ae_search_core",
        cache_fingerprint,
        selection_seed,
        device,
    )
    core_aggregate = apply_ae_gates(
        aggregate_search(core_records), protocol, quality_stage=False
    )
    carried: dict[str, list[str]] = {
        representation: top_candidates(
            core_aggregate,
            representation,
            count=2,
            gate_column="passes_ae_gates",
        )
        for representation in common.INTENSITY_REPRESENTATIONS
    }
    lookup = configuration_lookup(configurations)
    quality_configurations = {
        representation: [lookup[name] for name in names]
        for representation, names in carried.items()
    }
    quality_records, quality_histories = run_search_stage(
        dataset,
        quality_configurations,
        "quality_pass",
        output_dir,
        "ae_search_quality",
        cache_fingerprint,
        selection_seed,
        device,
    )
    quality_aggregate = aggregate_search(quality_records)
    core_selected = core_aggregate[
        core_aggregate["configuration"].isin(
            [name for names in carried.values() for name in names]
        )
    ].copy()
    quality_selected = quality_aggregate.copy()
    merge_columns = [
        "representation",
        "configuration",
        "channels",
        "bottleneck_dimension",
        "loss_name",
    ]
    quality_target = quality_selected[
        merge_columns + ["latent_target_balanced_accuracy"]
    ].rename(
        columns={
            "latent_target_balanced_accuracy": "quality_latent_target_balanced_accuracy"
        }
    )
    combined = core_selected.merge(
        quality_target, on=merge_columns, how="left"
    )
    combined["quality_target_drop_from_core"] = (
        combined["quality_latent_target_balanced_accuracy"]
        - combined["latent_target_balanced_accuracy"]
    )
    combined = apply_ae_gates(combined, protocol, quality_stage=True)
    selected: dict[str, Any] = {}
    for representation in common.INTENSITY_REPRESENTATIONS:
        candidates = combined[
            combined["representation"] == representation
        ].copy()
        eligible = candidates[candidates["passes_ae_gates"].astype(bool)]
        pool = eligible if not eligible.empty else candidates
        winner = pool.sort_values(
            ["selection_utility", "parameter_count", "configuration"],
            ascending=[False, True, True],
        ).iloc[0]
        configuration = lookup[str(winner["configuration"])]
        selected[representation] = {
            **common.autoencoder_config_record(configuration),
            "eligible_to_advance": bool(
                winner["passes_ae_gates"]
            ),
            "selection_utility": float(winner["selection_utility"]),
            "strict_core_metrics": json_clean(winner.to_dict()),
        }

    # Cache JSON is written with sorted keys, whereas freshly computed records
    # retain insertion order. Canonical column ordering makes fresh and resumed
    # selection outputs byte-stable without changing any row or metric value.
    all_records = pd.concat(
        [core_records, quality_records], ignore_index=True
    ).sort_index(axis=1)
    all_histories = pd.concat(
        [core_histories, quality_histories], ignore_index=True
    ).sort_index(axis=1)
    all_records.to_csv(output_dir / "ae_search_fold_metrics.csv", index=False)
    all_histories.to_csv(
        output_dir / "ae_search_training_histories.csv", index=False
    )
    core_aggregate.to_csv(
        output_dir / "ae_search_core_summary.csv", index=False
    )
    combined.to_csv(output_dir / "ae_selection_metrics.csv", index=False)
    write_json(
        output_dir / "ae_selected_configurations.json",
        {
            "selection_data": "NATO nested inner validation only",
            "stress_used": False,
            "poster_used": False,
            "outer_used": False,
            "carried_to_quality": carried,
            "selected": selected,
        },
    )
    return selected


def dae_gate_and_utility(
    dae_aggregate: pd.DataFrame,
    ae_aggregate: pd.DataFrame,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    ae_reference = ae_aggregate[
        [
            "representation",
            "latent_target_balanced_accuracy",
            "clean_reconstruction_median_row_correlation",
            "clean_repeatable_peak_recall",
            "corrupted_reconstruction_mse",
            "corrupted_prediction_agreement",
            "corrupted_latent_cosine_drift",
        ]
    ].rename(
        columns={
            column: f"ae_{column}"
            for column in [
                "latent_target_balanced_accuracy",
                "clean_reconstruction_median_row_correlation",
                "clean_repeatable_peak_recall",
                "corrupted_reconstruction_mse",
                "corrupted_prediction_agreement",
                "corrupted_latent_cosine_drift",
            ]
        }
    )
    frame = dae_aggregate.merge(
        ae_reference, on="representation", how="left"
    )
    gates = protocol["eligibility_gates"]["dae_relative_to_matched_ae"]
    frame["clean_correlation_delta_from_ae"] = (
        frame["clean_reconstruction_median_row_correlation"]
        - frame["ae_clean_reconstruction_median_row_correlation"]
    )
    frame["clean_peak_recall_delta_from_ae"] = (
        frame["clean_repeatable_peak_recall"]
        - frame["ae_clean_repeatable_peak_recall"]
    )
    frame["clean_target_delta_from_ae"] = (
        frame["latent_target_balanced_accuracy"]
        - frame["ae_latent_target_balanced_accuracy"]
    )
    frame["corrupted_recovery_relative_improvement"] = (
        frame["ae_corrupted_reconstruction_mse"]
        - frame["corrupted_reconstruction_mse"]
    ) / np.maximum(frame["ae_corrupted_reconstruction_mse"], 1.0e-12)
    frame["corrupted_prediction_agreement_delta_from_ae"] = (
        frame["corrupted_prediction_agreement"]
        - frame["ae_corrupted_prediction_agreement"]
    )
    frame["corrupted_latent_drift_relative_reduction"] = (
        frame["ae_corrupted_latent_cosine_drift"]
        - frame["corrupted_latent_cosine_drift"]
    ) / np.maximum(frame["ae_corrupted_latent_cosine_drift"], 1.0e-12)
    frame["gate_clean_correlation"] = (
        frame["clean_correlation_delta_from_ae"]
        >= -gates["clean_median_row_correlation_drop_maximum"]
    )
    frame["gate_clean_peak_recall"] = (
        frame["clean_peak_recall_delta_from_ae"]
        >= -gates["clean_repeatable_peak_recall_drop_maximum"]
    )
    frame["gate_clean_target"] = (
        frame["clean_target_delta_from_ae"]
        >= -gates["clean_chemical_probe_balanced_accuracy_drop_maximum"]
    )
    frame["benefit_corrupted_recovery"] = (
        frame["corrupted_recovery_relative_improvement"] >= 0.05
    )
    frame["benefit_prediction_agreement"] = (
        frame["corrupted_prediction_agreement_delta_from_ae"] >= 0.02
    )
    frame["benefit_latent_drift"] = (
        frame["corrupted_latent_drift_relative_reduction"] >= 0.05
    )
    frame["passes_any_denoising_benefit"] = frame[
        [
            "benefit_corrupted_recovery",
            "benefit_prediction_agreement",
            "benefit_latent_drift",
        ]
    ].any(axis=1)
    frame["passes_dae_gates"] = frame[
        [
            "gate_clean_correlation",
            "gate_clean_peak_recall",
            "gate_clean_target",
            "passes_any_denoising_benefit",
        ]
    ].all(axis=1)
    objectives = {
        "latent_target_balanced_accuracy": ("max", 0.30),
        "clean_reconstruction_median_row_correlation": ("max", 0.10),
        "clean_reconstruction_mse": ("min", 0.10),
        "corrupted_reconstruction_mse": ("min", 0.20),
        "clean_repeatable_peak_recall": ("max", 0.15),
        "latent_instrument_probe_increment": ("min", 0.10),
        "latent_same_master_cross_instrument_distance": ("min", 0.05),
    }
    frame["selection_utility"] = np.nan
    for representation, indices in frame.groupby("representation").groups.items():
        frame.loc[indices, "selection_utility"] = normalize_utility(
            frame.loc[indices], objectives
        )
    return frame


def run_dae_selection(
    dataset: common.SpectralDataset,
    protocol: dict[str, Any],
    protocol_path: Path,
    output_dir: Path,
    ae_selected: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    cache_fingerprint = execution_fingerprint(protocol_path)
    selection_seed = int(protocol["determinism"]["selection_seeds"][0])
    curricula = list(protocol["denoising_search"]["curricula"])
    configurations_by_representation = {
        representation: [
            selected_ae_with_curriculum(configuration, curriculum)
            for curriculum in curricula
        ]
        for representation, configuration in ae_selected.items()
    }
    core_records, core_histories = run_search_stage(
        dataset,
        configurations_by_representation,
        "strict_core",
        output_dir,
        "dae_search_core",
        cache_fingerprint,
        selection_seed,
        device,
    )
    ae_core_records = pd.read_csv(output_dir / "ae_search_fold_metrics.csv")
    ae_core_records = ae_core_records[
        ae_core_records["subset"] == "strict_core"
    ]
    selected_ae_names = {
        representation: configuration["identifier"]
        for representation, configuration in ae_selected.items()
    }
    ae_core_records = ae_core_records[
        [
            row["configuration"]
            == selected_ae_names[row["representation"]]
            for _, row in ae_core_records.iterrows()
        ]
    ]
    ae_core_aggregate = aggregate_search(ae_core_records)
    core_aggregate = dae_gate_and_utility(
        aggregate_search(core_records), ae_core_aggregate, protocol
    )
    carried: dict[str, list[str]] = {
        representation: top_candidates(
            core_aggregate,
            representation,
            count=2,
            gate_column="passes_dae_gates",
        )
        for representation in common.INTENSITY_REPRESENTATIONS
    }
    lookup = {
        representation: configuration_lookup(configurations)
        for representation, configurations in configurations_by_representation.items()
    }
    quality_configurations = {
        representation: [
            lookup[representation][name] for name in names
        ]
        for representation, names in carried.items()
    }
    quality_records, quality_histories = run_search_stage(
        dataset,
        quality_configurations,
        "quality_pass",
        output_dir,
        "dae_search_quality",
        cache_fingerprint,
        selection_seed,
        device,
    )
    quality_aggregate = aggregate_search(quality_records)
    core_carried = core_aggregate[
        core_aggregate["configuration"].isin(
            [name for names in carried.values() for name in names]
        )
    ].copy()
    quality_columns = [
        "representation",
        "configuration",
        "latent_target_balanced_accuracy",
        "clean_reconstruction_median_row_correlation",
        "clean_repeatable_peak_recall",
        "corrupted_reconstruction_mse",
        "corrupted_prediction_agreement",
        "corrupted_latent_cosine_drift",
    ]
    combined = core_carried.merge(
        quality_aggregate[quality_columns],
        on=["representation", "configuration"],
        how="left",
        suffixes=("", "_quality"),
    )
    # The strict-core eligibility decision remains primary. Quality-pass
    # sensitivity acts as a veto for large clean-target degradation.
    combined["quality_target_drop_from_core"] = (
        combined["latent_target_balanced_accuracy_quality"]
        - combined["latent_target_balanced_accuracy"]
    )
    combined["gate_quality_sensitivity"] = (
        combined["quality_target_drop_from_core"] >= -0.05
    )
    combined["passes_dae_gates_with_quality"] = (
        combined["passes_dae_gates"].astype(bool)
        & combined["gate_quality_sensitivity"].astype(bool)
    )
    selected: dict[str, Any] = {}
    for representation in common.INTENSITY_REPRESENTATIONS:
        candidates = combined[
            combined["representation"] == representation
        ].copy()
        eligible = candidates[
            candidates["passes_dae_gates_with_quality"].astype(bool)
        ]
        if eligible.empty:
            selected[representation] = {
                "selected": False,
                "reason": "No DAE curriculum passed all clean-preservation, denoising-benefit, and quality-sensitivity gates.",
            }
            continue
        winner = eligible.sort_values(
            ["selection_utility", "configuration"],
            ascending=[False, True],
        ).iloc[0]
        configuration = lookup[representation][str(winner["configuration"])]
        selected[representation] = {
            "selected": True,
            **common.autoencoder_config_record(configuration),
            "selection_utility": float(winner["selection_utility"]),
            "metrics": json_clean(winner.to_dict()),
        }
    all_records = pd.concat(
        [core_records, quality_records], ignore_index=True
    ).sort_index(axis=1)
    all_histories = pd.concat(
        [core_histories, quality_histories], ignore_index=True
    ).sort_index(axis=1)
    all_records.to_csv(
        output_dir / "dae_search_fold_metrics.csv", index=False
    )
    all_histories.to_csv(
        output_dir / "dae_search_training_histories.csv", index=False
    )
    core_aggregate.to_csv(
        output_dir / "dae_search_core_summary.csv", index=False
    )
    combined.to_csv(output_dir / "dae_selection_metrics.csv", index=False)
    write_json(
        output_dir / "dae_selected_configurations.json",
        {
            "selection_data": "NATO nested inner validation only",
            "stress_used": False,
            "poster_used": False,
            "outer_used": False,
            "carried_to_quality": carried,
            "selected": selected,
        },
    )
    return selected


def run_selection(
    dataset: common.SpectralDataset,
    protocol: dict[str, Any],
    protocol_path: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    ae_selected_path = output_dir / "ae_selected_configurations.json"
    if ae_selected_path.exists():
        ae_selected = json.loads(ae_selected_path.read_text())["selected"]
    else:
        ae_selected = run_ae_selection(
            dataset,
            protocol,
            protocol_path,
            output_dir,
            device,
        )
    dae_selected_path = output_dir / "dae_selected_configurations.json"
    if dae_selected_path.exists():
        dae_selected = json.loads(dae_selected_path.read_text())["selected"]
    else:
        dae_selected = run_dae_selection(
            dataset,
            protocol,
            protocol_path,
            output_dir,
            ae_selected,
            device,
        )
    selected = {
        "selection_closed": True,
        "selection_data": "NATO nested inner validation only",
        "outer_test_used": False,
        "field_quality_stress_used": False,
        "poster_used": False,
        "autoencoders": ae_selected,
        "denoising_autoencoders": dae_selected,
    }
    # Also canonicalize a resumed selection whose final configuration JSON
    # already exists. This closes the fresh-vs-cache schema difference without
    # retraining or changing any selected value.
    for name in (
        "ae_search_fold_metrics.csv",
        "ae_search_training_histories.csv",
        "ae_search_core_summary.csv",
        "ae_selection_metrics.csv",
        "dae_search_fold_metrics.csv",
        "dae_search_training_histories.csv",
        "dae_search_core_summary.csv",
        "dae_selection_metrics.csv",
    ):
        path = output_dir / name
        if path.is_file():
            pd.read_csv(path).sort_index(axis=1).to_csv(path, index=False)
    write_json(output_dir / "selected_configurations.json", selected)
    return selected


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
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
        / "sers_representation_baselines"
        / "baselines_v1",
    )
    parser.add_argument(
        "--stage",
        choices=["selection", "all"],
        default="all",
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
    poster_csv = args.poster_csv.resolve()
    output_dir = args.output_dir.resolve()
    protocol = load_protocol(protocol_path)
    # Loading NATO verifies every immutable preprocessing-v2 artifact hash.
    nato = common.load_nato_dataset(nato_bundle)
    initialize_output(
        output_dir,
        protocol_path,
        protocol,
        nato_bundle,
        poster_csv,
    )
    device = common.resolve_device(args.device)
    selected = run_selection(
        nato,
        protocol,
        protocol_path,
        output_dir,
        device,
    )
    print(
        json.dumps(
            {
                "stage": "selection",
                "status": "complete",
                "output_dir": str(output_dir),
                "selected": selected,
            },
            indent=2,
        )
    )
    if args.stage == "all":
        print(
            "Final outer/domain/poster evaluation is not yet implemented; "
            "selection artifacts are complete.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
