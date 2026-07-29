#!/usr/bin/env python3
"""Run grouped-inner structured-VAE controls and mechanism selection."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from dataclasses import replace
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
import sers_baseline_common as baseline
import sers_structured_vae_common as structured
import sers_vae_common as standard


CACHE_SCHEMA = "sers-structured-vae-selection-cache-v1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def execution_fingerprint(protocol_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        protocol_path,
        Path(__file__),
        Path(structured.__file__),
        Path(baseline.__file__),
        Path(baseline_selection.__file__),
    ):
        digest.update(str(path.resolve()).encode())
        digest.update(baseline.sha256_file(path).encode())
    for function in (
        evaluate_checkpoint,
        run_grid,
        apply_gates,
        select_candidate,
    ):
        digest.update(inspect.getsource(function).encode())
    digest.update(CACHE_SCHEMA.encode())
    return digest.hexdigest()


def mappings(
    manifest: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    return tuple(
        {
            label: index
            for index, label in enumerate(
                sorted(manifest[column].astype(str).unique())
            )
        }
        for column in ("target_analyte", "instrument", "sensor_family")
    )  # type: ignore[return-value]


def model_indices(
    manifest: pd.DataFrame,
    instrument_mapping: dict[str, int],
    sensor_mapping: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        manifest["instrument"]
        .astype(str)
        .map(instrument_mapping)
        .to_numpy(dtype=np.int64),
        manifest["sensor_family"]
        .astype(str)
        .map(sensor_mapping)
        .to_numpy(dtype=np.int64),
    )


def classification_and_probe(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_targets: np.ndarray,
    validation_targets: np.ndarray,
    train_instruments: np.ndarray,
    validation_instruments: np.ndarray,
    train_sensors: np.ndarray,
    validation_sensors: np.ndarray,
    validation_manifest: pd.DataFrame,
    run_seed: int,
    name: str,
) -> tuple[dict[str, Any], np.ndarray]:
    target_model = baseline.fit_latent_probe_model(
        train_features,
        train_targets,
        baseline.stable_seed(run_seed, name, "target"),
    )
    prediction = target_model.predict(validation_features).astype(str)
    target = baseline.classification_summary(
        validation_targets, prediction
    )
    instrument = baseline_selection.safe_domain_probe(
        train_features,
        validation_features,
        train_targets,
        validation_targets,
        train_instruments,
        validation_instruments,
        baseline.stable_seed(run_seed, name, "instrument"),
    )
    sensor = baseline_selection.safe_domain_probe(
        train_features,
        validation_features,
        train_targets,
        validation_targets,
        train_sensors,
        validation_sensors,
        baseline.stable_seed(run_seed, name, "sensor"),
    )
    geometry = baseline.geometry_metrics(
        validation_features,
        validation_manifest,
        np.ones(len(validation_manifest), dtype=bool),
    )
    result = {
        f"{name}_target_balanced_accuracy": target["balanced_accuracy"],
        f"{name}_target_macro_f1_supported": target["macro_f1_supported"],
        f"{name}_instrument_probe_increment": instrument[
            "increment_over_target_only"
        ],
        f"{name}_sensor_probe_increment": sensor[
            "increment_over_target_only"
        ],
        f"{name}_same_master_cross_instrument_distance": geometry[
            "same_master_cross_domain_mean_distance"
        ],
        f"{name}_different_target_cross_instrument_distance": geometry[
            "different_target_cross_domain_mean_distance"
        ],
        f"{name}_cross_instrument_separation_margin": geometry[
            "cross_domain_separation_margin"
        ],
    }
    return result, prediction


def sampling_variability(
    model: structured.PartitionedSERSVAE,
    values: np.ndarray,
    instrument_indices: np.ndarray,
    sensor_indices: np.ndarray,
    device: torch.device,
    seed: int,
    draws: int = 8,
) -> float:
    rng = np.random.default_rng(seed)
    deviations: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 128):
            stop = start + 128
            batch = torch.from_numpy(
                np.asarray(values[start:stop], dtype=np.float32)
            ).unsqueeze(1).to(device)
            instruments = torch.from_numpy(
                instrument_indices[start:stop].astype(np.int64)
            ).to(device)
            sensors = torch.from_numpy(
                sensor_indices[start:stop].astype(np.int64)
            ).to(device)
            mu, log_variance = model.encode(batch)
            decoded = []
            for _ in range(draws):
                epsilon = torch.from_numpy(
                    rng.standard_normal(mu.shape).astype(np.float32)
                ).to(device)
                union = model.reparameterize(mu, log_variance, epsilon)
                decoded.append(
                    model.decode_parts(
                        union[:, : model.chemical_dimension],
                        union[:, model.chemical_dimension :],
                        instruments,
                        sensors,
                    )
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )
            deviations.append(np.std(np.stack(decoded), axis=0))
    return float(np.mean(np.vstack(deviations)))


def partner_indices(
    manifest: pd.DataFrame, run_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    master = manifest["master_sample_id"].astype(str).to_numpy()
    instrument = manifest["instrument"].astype(str).to_numpy()
    uids = manifest["observation_uid"].astype(str).to_numpy()
    partners = np.arange(len(manifest), dtype=int)
    valid = np.zeros(len(manifest), dtype=bool)
    for index in range(len(manifest)):
        choices = np.flatnonzero(
            (master == master[index]) & (instrument != instrument[index])
        )
        if len(choices):
            seed = baseline.stable_seed(
                structured.PROTOCOL_VERSION,
                run_seed,
                499,
                uids[index],
                "real_partner",
            )
            partners[index] = int(choices[seed % len(choices)])
            valid[index] = True
    return partners, valid


def swapped_reconstruction(
    model: structured.PartitionedSERSVAE,
    outputs: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    instrument_indices: np.ndarray,
    sensor_indices: np.ndarray,
    run_seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    partners, valid = partner_indices(manifest, run_seed)
    selected = np.flatnonzero(valid)
    swapped: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(selected), 128):
            current = selected[start : start + 128]
            paired = partners[current]
            chemical = torch.from_numpy(
                outputs["chemical_mu"][current].astype(np.float32)
            ).to(device)
            nuisance = torch.from_numpy(
                outputs["nuisance_mu"][paired].astype(np.float32)
            ).to(device)
            instruments = torch.from_numpy(
                instrument_indices[paired].astype(np.int64)
            ).to(device)
            sensors = torch.from_numpy(
                sensor_indices[paired].astype(np.int64)
            ).to(device)
            swapped.append(
                model.decode_parts(
                    chemical, nuisance, instruments, sensors
                )
                .squeeze(1)
                .cpu()
                .numpy()
            )
    return np.vstack(swapped), partners[selected], selected


def evaluate_checkpoint(
    dataset: baseline.SpectralDataset,
    representation: str,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    config: structured.StructuredConfig,
    checkpoint_epoch: int,
    state: dict[str, torch.Tensor],
    run_seed: int,
    repeatable_validation: list[set[int]],
    raw_context: dict[str, Any],
    target_mapping: dict[str, int],
    instrument_mapping: dict[str, int],
    sensor_mapping: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    values = dataset.representations[representation]
    manifest = dataset.manifest
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    train_manifest = manifest.iloc[train_indices].reset_index(drop=True)
    validation_manifest = manifest.iloc[validation_indices].reset_index(
        drop=True
    )
    train_values = values[train_indices]
    validation_values = values[validation_indices]
    train_instrument_indices, train_sensor_indices = model_indices(
        train_manifest, instrument_mapping, sensor_mapping
    )
    validation_instrument_indices, validation_sensor_indices = model_indices(
        validation_manifest, instrument_mapping, sensor_mapping
    )
    model = structured.build_model_from_state(
        values.shape[1],
        config,
        len(target_mapping),
        len(instrument_mapping),
        len(sensor_mapping),
        state,
        device,
    )
    train_outputs = structured.outputs(
        model,
        train_values,
        train_instrument_indices,
        train_sensor_indices,
        device,
    )
    validation_outputs = structured.outputs(
        model,
        validation_values,
        validation_instrument_indices,
        validation_sensor_indices,
        device,
    )
    train_targets = train_manifest["target_analyte"].astype(str).to_numpy()
    validation_targets = (
        validation_manifest["target_analyte"].astype(str).to_numpy()
    )
    train_instruments = train_manifest["instrument"].astype(str).to_numpy()
    validation_instruments = (
        validation_manifest["instrument"].astype(str).to_numpy()
    )
    train_sensors = (
        train_manifest["sensor_family"].astype(str).to_numpy()
    )
    validation_sensors = (
        validation_manifest["sensor_family"].astype(str).to_numpy()
    )
    metrics: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, key in (
        ("chemical", "chemical_mu"),
        ("nuisance", "nuisance_mu"),
        ("union", "union_mu"),
    ):
        part, prediction = classification_and_probe(
            train_outputs[key],
            validation_outputs[key],
            train_targets,
            validation_targets,
            train_instruments,
            validation_instruments,
            train_sensors,
            validation_sensors,
            validation_manifest,
            run_seed,
            name,
        )
        metrics.update(part)
        predictions[name] = prediction
    clean_reconstruction = baseline.aggregate_reconstruction_metrics(
        baseline.reconstruction_metrics(
            validation_values,
            validation_outputs["reconstruction"],
            validation_manifest["observation_uid"].astype(str).to_numpy(),
            repeatable_validation,
        )
    )
    corrupted_values = baseline_selection.deterministic_corrupted_matrix(
        validation_values,
        validation_manifest["observation_uid"],
        "composite",
        1.0,
    )
    corrupted_outputs = structured.outputs(
        model,
        corrupted_values,
        validation_instrument_indices,
        validation_sensor_indices,
        device,
    )
    corrupted_reconstruction = baseline.aggregate_reconstruction_metrics(
        baseline.reconstruction_metrics(
            validation_values,
            corrupted_outputs["reconstruction"],
            validation_manifest["observation_uid"].astype(str).to_numpy(),
            repeatable_validation,
        )
    )
    for name, key in (
        ("chemical", "chemical_mu"),
        ("nuisance", "nuisance_mu"),
        ("union", "union_mu"),
    ):
        probe = baseline.fit_latent_probe_model(
            train_outputs[key],
            train_targets,
            baseline.stable_seed(run_seed, name, "target"),
        )
        corrupted_prediction = probe.predict(
            corrupted_outputs[key]
        ).astype(str)
        classification = baseline.classification_summary(
            validation_targets, corrupted_prediction
        )
        metrics.update(
            {
                f"corrupted_{name}_target_balanced_accuracy": classification[
                    "balanced_accuracy"
                ],
                f"corrupted_{name}_prediction_agreement": float(
                    np.mean(corrupted_prediction == predictions[name])
                ),
                f"corrupted_{name}_cosine_drift": float(
                    np.mean(
                        baseline_selection.cosine_drift(
                            validation_outputs[key], corrupted_outputs[key]
                        )
                    )
                ),
            }
        )
    variability = sampling_variability(
        model,
        validation_values,
        validation_instrument_indices,
        validation_sensor_indices,
        device,
        baseline.stable_seed(run_seed, "sampling_variability"),
    )
    for name, mu_key, logvar_key in (
        ("chemical", "chemical_mu", "chemical_log_variance"),
        ("nuisance", "nuisance_mu", "nuisance_log_variance"),
        ("union", "union_mu", "union_log_variance"),
    ):
        variational = standard.variational_metrics(
            validation_outputs[mu_key],
            validation_outputs[logvar_key],
            normalization_divisor=config.kl_normalization_divisor,
            sample_reconstruction_variability=(
                variability if name == "union" else np.nan
            ),
        )
        metrics.update(
            {f"{name}_{key}": value for key, value in variational.items()}
        )
    metrics.update(
        structured.partition_dependence(
            validation_outputs["chemical_mu"],
            validation_outputs["nuisance_mu"],
        )
    )
    swapped, partner_rows, selected_rows = swapped_reconstruction(
        model,
        validation_outputs,
        validation_manifest,
        validation_instrument_indices,
        validation_sensor_indices,
        run_seed,
        device,
    )
    swapped_repeatable = [
        repeatable_validation[index] for index in partner_rows
    ]
    cross_metrics = baseline.aggregate_reconstruction_metrics(
        baseline.reconstruction_metrics(
            validation_values[partner_rows],
            swapped,
            validation_manifest.iloc[partner_rows][
                "observation_uid"
            ].astype(str).to_numpy(),
            swapped_repeatable,
        )
    )
    chemical_pair_distance = baseline.correlation_distance_rows(
        validation_outputs["chemical_mu"][selected_rows],
        validation_outputs["chemical_mu"][partner_rows],
    )
    result = {
        **config.record(),
        "checkpoint_epoch": checkpoint_epoch,
        "representation": representation,
        "model_family": "structured_vae",
        "run_seed": run_seed,
        "n_train": len(train_indices),
        "n_validation": len(validation_indices),
        "parameter_count": baseline.model_parameter_count(model),
        "state_sha256": baseline.state_dict_sha256(state),
        "raw_target_balanced_accuracy": raw_context[
            "target_classification"
        ]["balanced_accuracy"],
        "raw_instrument_probe_increment": raw_context[
            "instrument_probe"
        ]["increment_over_target_only"],
        "raw_sensor_probe_increment": raw_context["sensor_probe"][
            "increment_over_target_only"
        ],
        "raw_same_master_cross_instrument_distance": raw_context[
            "geometry"
        ]["same_master_cross_domain_mean_distance"],
        "real_pair_count": len(selected_rows),
        "chemical_real_pair_correlation_distance": float(
            np.mean(chemical_pair_distance)
        ),
        **metrics,
        **{
            f"clean_{key}": value
            for key, value in clean_reconstruction.items()
        },
        **{
            f"corrupted_{key}": value
            for key, value in corrupted_reconstruction.items()
        },
        **{
            f"cross_reconstruction_{key}": value
            for key, value in cross_metrics.items()
        },
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_grid(
    dataset: baseline.SpectralDataset,
    configs: Sequence[structured.StructuredConfig],
    stage: str,
    output_dir: Path,
    fingerprint: str,
    training_device: torch.device,
    metric_device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = dataset.manifest
    folds = manifest["grouped_sample_fold_5"].to_numpy(dtype=int)
    target_mapping, instrument_mapping, sensor_mapping = mappings(manifest)
    records: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    for outer_fold in range(5):
        for inner_fold in sorted(set(range(5)) - {outer_fold}):
            train_mask = (folds != outer_fold) & (folds != inner_fold)
            validation_mask = folds == inner_fold
            development_mask = folds != outer_fold
            repeatable = baseline_selection.repeatable_for_validation(
                dataset.representations["arpls_minmax"],
                manifest,
                development_mask,
                validation_mask,
            )
            raw_context = baseline_selection.raw_baseline_context(
                dataset,
                "arpls_minmax",
                train_mask,
                validation_mask,
                baseline.stable_seed(
                    structured.PROTOCOL_VERSION,
                    stage,
                    outer_fold,
                    inner_fold,
                    "raw",
                ),
            )
            train_manifest = manifest.loc[train_mask].reset_index(drop=True)
            validation_manifest = manifest.loc[
                validation_mask
            ].reset_index(drop=True)
            for config in configs:
                run_seed = baseline.stable_seed(
                    structured.PROTOCOL_VERSION,
                    "paired_inner",
                    outer_fold,
                    inner_fold,
                    1729,
                )
                run_identifier = (
                    f"strict_core__o{outer_fold}__i{inner_fold}__"
                    f"arpls_minmax__{config.identifier}"
                )
                cache_path = (
                    output_dir
                    / "selection_cache"
                    / stage
                    / f"{run_identifier}.pt"
                )
                if cache_path.exists():
                    payload = torch.load(
                        cache_path, map_location="cpu", weights_only=False
                    )
                    if payload["execution_fingerprint"] != fingerprint:
                        raise ValueError(f"Stale cache: {run_identifier}")
                    history = payload["history"]
                    states = payload["states"]
                else:
                    history, states, optimizer_states = (
                        structured.train_registered_checkpoints(
                            dataset.representations["arpls_minmax"][
                                train_mask
                            ],
                            train_manifest,
                            dataset.representations["arpls_minmax"][
                                validation_mask
                            ],
                            validation_manifest,
                            target_mapping,
                            instrument_mapping,
                            sensor_mapping,
                            config,
                            run_seed,
                            [100, 300, 400, 500],
                            training_device,
                        )
                    )
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "execution_fingerprint": fingerprint,
                            "config": config.record(),
                            "run_seed": run_seed,
                            "history": history,
                            "states": states,
                            "optimizer_states": optimizer_states,
                        },
                        cache_path,
                    )
                annotated = history.copy()
                annotated.insert(0, "stage", stage)
                annotated.insert(1, "configuration", config.identifier)
                annotated.insert(2, "outer_fold", outer_fold)
                annotated.insert(3, "inner_fold", inner_fold)
                histories.append(annotated)
                record = evaluate_checkpoint(
                    dataset,
                    "arpls_minmax",
                    train_mask,
                    validation_mask,
                    config,
                    500,
                    states[500],
                    run_seed,
                    repeatable,
                    raw_context,
                    target_mapping,
                    instrument_mapping,
                    sensor_mapping,
                    metric_device,
                )
                record.update(
                    {
                        "stage": stage,
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
                            "chemical_ba": record[
                                "chemical_target_balanced_accuracy"
                            ],
                            "instrument_probe": record[
                                "chemical_instrument_probe_increment"
                            ],
                            "same_master": record[
                                "chemical_same_master_cross_instrument_distance"
                            ],
                        }
                    ),
                    flush=True,
                )
    return pd.DataFrame(records), pd.concat(histories, ignore_index=True)


def convergence_summary(histories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (configuration, outer_fold, inner_fold), run in histories.groupby(
        ["configuration", "outer_fold", "inner_fold"]
    ):
        indexed = run.set_index("epoch")
        before = float(indexed.loc[450, "validation_total"])
        after = float(indexed.loc[500, "validation_total"])
        rows.append(
            {
                "identifier": configuration,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "relative_improvement_final_50": (
                    before - after
                )
                / max(abs(before), 1.0e-12),
            }
        )
    per_fold = pd.DataFrame(rows)
    aggregate = (
        per_fold.groupby("identifier")["relative_improvement_final_50"]
        .agg(
            median_relative_improvement_50="median",
            mean_relative_improvement_50="mean",
            maximum_relative_improvement_50="max",
            folds_improving_at_least_1_percent=lambda values: float(
                np.mean(values >= 0.01)
            ),
        )
        .reset_index()
    )
    aggregate["converged"] = (
        aggregate["median_relative_improvement_50"] < 0.005
    ) & (
        aggregate["folds_improving_at_least_1_percent"] < 0.25
    )
    return aggregate


def aggregate_metrics(
    folds: pd.DataFrame, histories: pd.DataFrame
) -> pd.DataFrame:
    identifiers = [
        "identifier",
        "chemical_dimension",
        "nuisance_dimension",
        "chemical_supervision_weight",
        "instrument_supervision_weight",
        "sensor_supervision_weight",
        "condition_decoder",
        "instrument_adversary_weight",
        "sensor_adversary_weight",
        "same_master_consistency_weight",
        "cross_reconstruction_weight",
        "dependence_weight",
        "checkpoint_epoch",
        "representation",
        "model_family",
        "stage",
    ]
    excluded = set(
        identifiers
        + [
            "outer_fold",
            "inner_fold",
            "run_seed",
            "state_sha256",
        ]
    )
    numeric = [
        column
        for column in folds
        if column not in excluded
        and pd.api.types.is_numeric_dtype(folds[column])
    ]
    result = folds.groupby(identifiers, as_index=False)[numeric].mean()
    counts = (
        folds.groupby(identifiers).size().rename("fold_count").reset_index()
    )
    result = result.merge(counts, on=identifiers, how="left")
    return result.merge(convergence_summary(histories), on="identifier")


def paired_directional_fractions(
    fold_metrics: pd.DataFrame, adequacy_bundle: Path
) -> pd.DataFrame:
    reference = pd.read_csv(
        adequacy_bundle / "stage_2_beta_fold_metrics.csv"
    )
    reference = reference[
        reference["identifier"].eq(
            "base_maxpool__z64__spectral_composite__"
            "beta0p25__constant_lr__e500"
        )
    ][
        [
            "outer_fold",
            "inner_fold",
            "latent_target_balanced_accuracy",
            "latent_instrument_probe_increment",
        ]
    ].rename(
        columns={
            "latent_target_balanced_accuracy": "reference_chemical_ba",
            "latent_instrument_probe_increment": (
                "reference_instrument_probe"
            ),
        }
    )
    merged = fold_metrics.merge(
        reference, on=["outer_fold", "inner_fold"], validate="many_to_one"
    )
    merged["preserves_chemical"] = (
        merged["chemical_target_balanced_accuracy"]
        >= merged["reference_chemical_ba"] - 0.03
    )
    merged["improves_instrument"] = (
        merged["chemical_instrument_probe_increment"]
        < merged["reference_instrument_probe"]
    )
    return (
        merged.groupby("identifier")
        .agg(
            fold_fraction_preserving_chemical=(
                "preserves_chemical",
                "mean",
            ),
            fold_fraction_improving_instrument=(
                "improves_instrument",
                "mean",
            ),
        )
        .reset_index()
    )


def apply_gates(
    summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    protocol: dict[str, Any],
    adequacy_bundle: Path,
) -> pd.DataFrame:
    result = summary.merge(
        paired_directional_fractions(fold_metrics, adequacy_bundle),
        on="identifier",
    )
    gates = protocol["eligibility_gates"]
    result["gate_chemical_ba"] = (
        result["chemical_target_balanced_accuracy"]
        >= gates["chemical_balanced_accuracy_minimum"]
    )
    result["gate_chemical_f1"] = (
        result["chemical_target_macro_f1_supported"]
        >= gates["chemical_macro_f1_minimum"]
    )
    result["gate_instrument_probe"] = (
        result["chemical_instrument_probe_increment"]
        <= gates["instrument_probe_increment_maximum"]
    )
    result["gate_same_master"] = (
        result["chemical_same_master_cross_instrument_distance"]
        <= gates["same_master_distance_maximum"]
    )
    result["gate_separation_margin"] = (
        result["chemical_cross_instrument_separation_margin"]
        >= gates["cross_instrument_separation_margin_minimum"]
    )
    result["gate_reconstruction_correlation"] = (
        result["clean_reconstruction_median_row_correlation"]
        >= gates["reconstruction_correlation_minimum"]
    )
    result["gate_peak_recall"] = (
        result["clean_repeatable_peak_recall"]
        >= gates["repeatable_peak_recall_minimum"]
    )
    result["gate_chemical_active"] = (
        result["chemical_vae_active_units_var_mu_gt_0_01"]
        >= gates["chemical_active_units_minimum"]
    )
    result["gate_nuisance_active"] = (
        result["nuisance_vae_active_units_var_mu_gt_0_01"]
        >= gates["nuisance_active_units_minimum"]
    )
    result["gate_kl_dimensions"] = (
        result["chemical_vae_dimensions_mean_kl_gt_0_01"]
        >= gates["kl_dimensions_per_nonempty_partition_minimum"]
    ) & (
        result["nuisance_vae_dimensions_mean_kl_gt_0_01"]
        >= gates["kl_dimensions_per_nonempty_partition_minimum"]
    )
    result["gate_kl_range"] = result[
        "union_vae_kl_unnormalized_per_observation"
    ].between(
        gates["kl_per_observation_minimum"],
        gates["kl_per_observation_maximum"],
    )
    result["gate_fold_instrument_direction"] = (
        result["fold_fraction_improving_instrument"]
        >= gates["fold_fraction_with_instrument_probe_improvement_minimum"]
    )
    result["gate_fold_chemical_direction"] = (
        result["fold_fraction_preserving_chemical"]
        >= gates["fold_fraction_preserving_chemical_accuracy_minimum"]
    )
    sensor_claimed = result["sensor_adversary_weight"] > 0
    result["gate_sensor_probe"] = (
        ~sensor_claimed
        | (
            result["chemical_sensor_probe_increment"]
            <= gates["sensor_probe_increment_maximum_when_sensor_claimed"]
        )
    )
    capture_claimed = result["instrument_supervision_weight"] > 0
    result["gate_nuisance_capture"] = (
        ~capture_claimed
        | (
            result["nuisance_instrument_probe_increment"]
            >= gates[
                "nuisance_instrument_probe_minimum_when_capture_claimed"
            ]
        )
    )
    finite_columns = [
        "chemical_target_balanced_accuracy",
        "chemical_instrument_probe_increment",
        "chemical_same_master_cross_instrument_distance",
        "clean_reconstruction_median_row_correlation",
        "clean_repeatable_peak_recall",
        "union_vae_kl_unnormalized_per_observation",
        "partition_cross_covariance_mean_square",
    ]
    result["gate_finite"] = np.isfinite(
        result[finite_columns].to_numpy(dtype=float)
    ).all(axis=1)
    result["gate_converged"] = result["converged"].astype(bool)
    gate_columns = [
        column for column in result if column.startswith("gate_")
    ]
    result["gate_count"] = result[gate_columns].astype(int).sum(axis=1)
    result["gate_total"] = len(gate_columns)
    result["passes_all_gates"] = result[gate_columns].all(axis=1)
    result["posterior_health"] = result[
        [
            "gate_chemical_active",
            "gate_nuisance_active",
            "gate_kl_dimensions",
            "gate_kl_range",
            "gate_finite",
        ]
    ].astype(float).mean(axis=1)
    objectives = {
        "chemical_target_balanced_accuracy": ("max", 0.20),
        "chemical_target_macro_f1_supported": ("max", 0.05),
        "chemical_instrument_probe_increment": ("min", 0.15),
        "chemical_sensor_probe_increment": ("min", 0.10),
        "chemical_same_master_cross_instrument_distance": ("min", 0.15),
        "chemical_cross_instrument_separation_margin": ("max", 0.05),
        "clean_reconstruction_median_row_correlation": ("max", 0.05),
        "clean_repeatable_peak_recall": ("max", 0.10),
        "nuisance_instrument_probe_increment": ("max", 0.05),
        "partition_cross_covariance_mean_square": ("min", 0.05),
        "posterior_health": ("max", 0.05),
    }
    result["selection_utility"] = (
        baseline_selection.normalize_utility(result, objectives)
    )
    return result


def select_candidate(summary: pd.DataFrame) -> pd.Series:
    ranked = summary.sort_values(
        [
            "converged",
            "passes_all_gates",
            "gate_count",
            "selection_utility",
            "parameter_count",
            "identifier",
        ],
        ascending=[False, False, False, False, True, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


def control_configs() -> list[structured.StructuredConfig]:
    return [
        structured.StructuredConfig(
            chemical_dimension=48, nuisance_dimension=16
        ),
        structured.StructuredConfig(
            chemical_dimension=32, nuisance_dimension=32
        ),
        structured.StructuredConfig(
            chemical_dimension=48,
            nuisance_dimension=16,
            chemical_supervision_weight=0.005,
        ),
        structured.StructuredConfig(
            chemical_dimension=48,
            nuisance_dimension=16,
            chemical_supervision_weight=0.005,
            instrument_supervision_weight=0.0025,
            sensor_supervision_weight=0.0025,
            condition_decoder=True,
        ),
    ]


def config_from_record(record: dict[str, Any]) -> structured.StructuredConfig:
    allowed = set(structured.StructuredConfig.__dataclass_fields__)
    values = {key: value for key, value in record.items() if key in allowed}
    for key in (
        "chemical_dimension",
        "nuisance_dimension",
        "maximum_epoch",
        "batch_size",
        "kl_normalization_divisor",
    ):
        if key in values:
            values[key] = int(values[key])
    if "condition_decoder" in values:
        values["condition_decoder"] = bool(values["condition_decoder"])
    return structured.StructuredConfig(**values)


def stage_configs(
    stage: str,
    output_dir: Path,
    protocol: dict[str, Any],
) -> list[structured.StructuredConfig]:
    if stage == "controls":
        return control_configs()
    decision_path = output_dir / "controls_decision.json"
    if not decision_path.is_file():
        raise ValueError("Controls must close before mechanism search")
    base = config_from_record(
        json.loads(decision_path.read_text())["selected_configuration"]
    )
    search = protocol["mechanism_search"]
    if stage == "instrument_adversary":
        return [
            replace(base, instrument_adversary_weight=float(weight))
            for weight in search["instrument_adversary_weights"]
        ]
    if stage == "pair":
        return [
            replace(
                base,
                same_master_consistency_weight=float(weight),
                cross_reconstruction_weight=float(
                    search["cross_reconstruction_weight"]
                ),
            )
            for weight in search["same_master_consistency_weights"]
        ]
    if stage == "dependence":
        return [
            replace(base, dependence_weight=float(weight))
            for weight in search["dependence_weights"]
        ]
    raise ValueError(f"Unsupported stage: {stage}")


def save_stage(
    stage: str,
    fold_metrics: pd.DataFrame,
    histories: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    fold_metrics.to_csv(
        output_dir / f"{stage}_fold_metrics.csv", index=False
    )
    histories.to_csv(output_dir / f"{stage}_histories.csv", index=False)
    summary.to_csv(output_dir / f"{stage}_summary.csv", index=False)
    selected = select_candidate(summary)
    write_json(
        output_dir / f"{stage}_decision.json",
        {
            "protocol": structured.PROTOCOL_VERSION,
            "stage": stage,
            "selection_used_locked_outcomes": False,
            "selected_identifier": selected["identifier"],
            "selected_configuration": json_clean(
                {
                    key: selected[key]
                    for key in structured.StructuredConfig.__dataclass_fields__
                }
            ),
            "converged": bool(selected["converged"]),
            "passes_all_gates": bool(selected["passes_all_gates"]),
            "gate_count": int(selected["gate_count"]),
            "gate_total": int(selected["gate_total"]),
            "selection_utility": float(selected["selection_utility"]),
        },
    )


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["controls", "instrument_adversary", "pair", "dependence"],
        required=True,
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repository / "configs" / "sers_structured_vae_v1.json",
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
        "--adequacy-bundle",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_vae_adequacy"
        / "adequacy_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_structured_vae"
        / "structured_vae_v1",
    )
    parser.add_argument(
        "--training-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--metric-device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("protocol_version") != structured.PROTOCOL_VERSION
        or protocol.get("status_before_structured_model_execution")
        != "predeclared"
    ):
        raise ValueError("Structured protocol is not preregistered")
    identity = json.loads(
        (args.output_dir / "identity_control_summary.json").read_text()
    )
    if identity.get("identity_gate_passed") is not True:
        raise ValueError("Exact identity gate did not pass")
    configs = stage_configs(args.stage, args.output_dir, protocol)
    dataset = baseline.load_nato_dataset(args.nato_bundle)
    fingerprint = execution_fingerprint(args.protocol)
    fold_metrics, histories = run_grid(
        dataset,
        configs,
        args.stage,
        args.output_dir,
        fingerprint,
        torch.device(args.training_device),
        torch.device(args.metric_device),
    )
    summary = aggregate_metrics(fold_metrics, histories)
    summary = apply_gates(
        summary, fold_metrics, protocol, args.adequacy_bundle
    )
    save_stage(
        args.stage, fold_metrics, histories, summary, args.output_dir
    )
    selected = select_candidate(summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "stage": args.stage,
                "selected": selected["identifier"],
                "converged": bool(selected["converged"]),
                "gates": f"{int(selected['gate_count'])}/"
                f"{int(selected['gate_total'])}",
                "passes_all_gates": bool(selected["passes_all_gates"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
