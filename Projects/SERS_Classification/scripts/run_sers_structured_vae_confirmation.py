#!/usr/bin/env python3
"""Run sensitivity and locked confirmation for the frozen structured SERS VAE."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
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

import run_sers_baseline_final as baseline_final
import run_sers_representation_baselines as baseline_selection
import run_sers_structured_vae_selection as selection
import sers_baseline_common as baseline
import sers_structured_vae_common as structured
import sers_vae_common as standard


CONFIRMATION_VERSION = "sers-structured-vae-confirmation-v1"
DECLARED_SEED = 1729


def write_json(path: Path, value: Any) -> None:
    selection.write_json(path, selection.json_clean(value))


def array_sha256(values: Sequence[str] | np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def execution_fingerprint(protocol_path: Path, stage: str) -> str:
    digest = hashlib.sha256()
    for path in (
        protocol_path,
        Path(structured.__file__),
        Path(selection.__file__),
        Path(baseline.__file__),
        Path(baseline_final.__file__),
    ):
        digest.update(str(path.resolve()).encode())
        digest.update(baseline.sha256_file(path).encode())
    functions = [load_or_train]
    if stage == "sensitivity":
        functions.extend([sensitivity_grid, run_sensitivity])
    elif stage == "outer":
        functions.extend([evaluate_structured, train_and_evaluate, run_outer])
    elif stage == "domain":
        functions.extend([evaluate_structured, train_and_evaluate, run_domain])
    elif stage == "poster":
        functions.extend([evaluate_structured, train_and_evaluate, run_poster])
    else:
        raise ValueError(f"Unknown fingerprint stage: {stage}")
    for function in functions:
        digest.update(inspect.getsource(function).encode())
    digest.update(stage.encode())
    digest.update(CONFIRMATION_VERSION.encode())
    return digest.hexdigest()


def frozen_configuration(output_dir: Path) -> structured.StructuredConfig:
    closure = json.loads(
        (output_dir / "inner_selection_closure.json").read_text()
    )
    if not closure.get("selection_closed", False):
        raise ValueError("Inner selection is not closed")
    if closure.get("selection_used_locked_outcomes", True):
        raise ValueError("Locked outcomes were used during selection")
    return selection.config_from_record(closure["selected_configuration"])


def load_or_train(
    output_dir: Path,
    stage: str,
    run_identifier: str,
    train_values: np.ndarray,
    train_manifest: pd.DataFrame,
    validation_values: np.ndarray,
    validation_manifest: pd.DataFrame,
    target_mapping: dict[str, int],
    instrument_mapping: dict[str, int],
    sensor_mapping: dict[str, int],
    config: structured.StructuredConfig,
    run_seed: int,
    fingerprint: str,
    device: torch.device,
) -> tuple[
    structured.PartitionedSERSVAE,
    pd.DataFrame,
    dict[str, Any],
]:
    checkpoint = (
        output_dir / "confirmation_cache" / stage / f"{run_identifier}.pt"
    )
    expected = {
        "confirmation_version": CONFIRMATION_VERSION,
        "execution_fingerprint": fingerprint,
        "run_identifier": run_identifier,
        "configuration": config.record(),
        "run_seed": run_seed,
        "train_uids_sha256": array_sha256(
            train_manifest["observation_uid"].astype(str).to_numpy()
        ),
        "train_values_sha256": array_sha256(
            np.asarray(train_values, dtype=np.float32)
        ),
        "validation_uids_sha256": array_sha256(
            validation_manifest["observation_uid"].astype(str).to_numpy()
        ),
        "validation_values_sha256": array_sha256(
            np.asarray(validation_values, dtype=np.float32)
        ),
        "target_mapping": target_mapping,
        "instrument_mapping": instrument_mapping,
        "sensor_mapping": sensor_mapping,
    }
    if checkpoint.is_file():
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        if payload["metadata"] != expected:
            raise ValueError(f"Stale confirmation cache: {run_identifier}")
        state = payload["state"]
        actual = baseline.state_dict_sha256(state)
        if actual != payload["state_sha256"]:
            raise ValueError(f"Corrupt confirmation cache: {run_identifier}")
        history = payload["history"]
        cache_status = "loaded"
    else:
        history, states, _ = structured.train_registered_checkpoints(
            train_values,
            train_manifest,
            validation_values,
            validation_manifest,
            target_mapping,
            instrument_mapping,
            sensor_mapping,
            config,
            run_seed,
            [config.maximum_epoch],
            device,
        )
        state = states[config.maximum_epoch]
        actual = baseline.state_dict_sha256(state)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": expected,
                "history": history,
                "state": state,
                "state_sha256": actual,
            },
            checkpoint,
        )
        cache_status = "trained"
    model = structured.build_model_from_state(
        train_values.shape[1],
        config,
        len(target_mapping),
        len(instrument_mapping),
        len(sensor_mapping),
        state,
        device,
    )
    registry = {
        **expected,
        "state_sha256": actual,
        "parameter_count": baseline.model_parameter_count(model),
        "cache_status": cache_status,
        "checkpoint": str(checkpoint.relative_to(output_dir)),
    }
    return model, history, registry


def sensitivity_grid(
    dataset: baseline.SpectralDataset,
    subset: str,
    representation: str,
    config: structured.StructuredConfig,
    output_dir: Path,
    fingerprint: str,
    training_device: torch.device,
    metric_device: torch.device,
    global_mappings: tuple[
        dict[str, int], dict[str, int], dict[str, int]
    ],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    manifest = dataset.manifest
    folds = manifest["grouped_sample_fold_5"].to_numpy(dtype=int)
    target_mapping, instrument_mapping, sensor_mapping = global_mappings
    records: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    registries: list[dict[str, Any]] = []
    for outer_fold in range(5):
        for inner_fold in sorted(set(range(5)) - {outer_fold}):
            train_mask = (folds != outer_fold) & (folds != inner_fold)
            validation_mask = folds == inner_fold
            development_mask = folds != outer_fold
            train_manifest = manifest.loc[train_mask].reset_index(drop=True)
            validation_manifest = manifest.loc[
                validation_mask
            ].reset_index(drop=True)
            run_seed = baseline.stable_seed(
                structured.PROTOCOL_VERSION,
                "paired_inner",
                outer_fold,
                inner_fold,
                DECLARED_SEED,
            )
            run_identifier = (
                f"{subset}__o{outer_fold}__i{inner_fold}__"
                f"{representation}__{config.identifier}"
            )
            model, history, registry = load_or_train(
                output_dir,
                "sensitivity",
                run_identifier,
                dataset.representations[representation][train_mask],
                train_manifest,
                dataset.representations[representation][validation_mask],
                validation_manifest,
                target_mapping,
                instrument_mapping,
                sensor_mapping,
                config,
                run_seed,
                fingerprint,
                training_device,
            )
            del model
            payload = torch.load(
                output_dir
                / "confirmation_cache"
                / "sensitivity"
                / f"{run_identifier}.pt",
                map_location="cpu",
                weights_only=False,
            )
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
                baseline.stable_seed(run_seed, "raw"),
            )
            record = selection.evaluate_checkpoint(
                dataset,
                representation,
                train_mask,
                validation_mask,
                config,
                500,
                payload["state"],
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
                    "stage": "sensitivity",
                    "subset": subset,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                }
            )
            records.append(record)
            annotated = history.copy()
            annotated.insert(0, "subset", subset)
            annotated.insert(1, "representation", representation)
            annotated.insert(2, "outer_fold", outer_fold)
            annotated.insert(3, "inner_fold", inner_fold)
            histories.append(annotated)
            registry.update(
                {
                    "subset": subset,
                    "representation": representation,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                }
            )
            registries.append(registry)
            print(
                json.dumps(
                    {
                        "stage": "sensitivity",
                        "subset": subset,
                        "representation": representation,
                        "outer": outer_fold,
                        "inner": inner_fold,
                        "chemical_ba": record[
                            "chemical_target_balanced_accuracy"
                        ],
                    }
                ),
                flush=True,
            )
    return (
        pd.DataFrame(records),
        pd.concat(histories, ignore_index=True),
        registries,
    )


def run_sensitivity(
    output_dir: Path,
    nato_bundle: Path,
    config: structured.StructuredConfig,
    fingerprint: str,
    protocol: dict[str, Any],
    training_device: torch.device,
    metric_device: torch.device,
) -> None:
    core = baseline.load_nato_dataset(nato_bundle)
    quality = baseline_final.load_nato_subset(nato_bundle, "quality_pass")
    global_mappings = selection.mappings(core.manifest)
    specifications = (
        (quality, "quality_pass", "arpls_minmax"),
        (core, "strict_core", "minimal_minmax"),
        (quality, "quality_pass", "minimal_minmax"),
    )
    metric_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    registries: list[dict[str, Any]] = []
    for dataset, subset, representation in specifications:
        metrics, histories, registry = sensitivity_grid(
            dataset,
            subset,
            representation,
            config,
            output_dir,
            fingerprint,
            training_device,
            metric_device,
            global_mappings,
        )
        metric_frames.append(metrics)
        history_frames.append(histories)
        registries.extend(registry)
    metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    histories = pd.concat(history_frames, ignore_index=True, sort=False)
    metrics.to_csv(output_dir / "sensitivity_fold_metrics.csv", index=False)
    histories.to_csv(
        output_dir / "sensitivity_training_histories.csv", index=False
    )
    write_json(output_dir / "sensitivity_run_registry.json", registries)
    numeric = [
        column
        for column in metrics
        if pd.api.types.is_numeric_dtype(metrics[column])
        and column not in {"outer_fold", "inner_fold", "run_seed"}
    ]
    summary = (
        metrics.groupby(["subset", "representation"], as_index=False)[numeric]
        .mean()
    )
    summary.to_csv(output_dir / "sensitivity_summary.csv", index=False)
    reference = pd.read_csv(output_dir / "dependence_fold_metrics.csv")[
        [
            "outer_fold",
            "inner_fold",
            "chemical_target_balanced_accuracy",
        ]
    ].rename(
        columns={
            "chemical_target_balanced_accuracy": "strict_arpls_chemical_ba"
        }
    )
    paired = metrics[
        [
            "subset",
            "representation",
            "outer_fold",
            "inner_fold",
            "chemical_target_balanced_accuracy",
        ]
    ].merge(reference, on=["outer_fold", "inner_fold"], how="left")
    quality_arpls = paired[
        paired["subset"].eq("quality_pass")
        & paired["representation"].eq("arpls_minmax")
    ]
    strict_minimal = paired[
        paired["subset"].eq("strict_core")
        & paired["representation"].eq("minimal_minmax")
    ]
    quality_limit = float(
        protocol["eligibility_gates"][
            "quality_balanced_accuracy_drop_maximum"
        ]
    )
    quality_delta = float(
        (
            quality_arpls["chemical_target_balanced_accuracy"]
            - quality_arpls["strict_arpls_chemical_ba"]
        ).mean()
    )
    minimal_delta = float(
        (
            strict_minimal["chemical_target_balanced_accuracy"]
            - strict_minimal["strict_arpls_chemical_ba"]
        ).mean()
    )
    decision = {
        "protocol": structured.PROTOCOL_VERSION,
        "configuration": config.identifier,
        "quality_arpls_mean_ba_delta_vs_strict_arpls": quality_delta,
        "quality_drop_limit": quality_limit,
        "quality_sensitivity_passed": quality_delta >= -quality_limit,
        "strict_minimal_mean_ba_delta_vs_strict_arpls": minimal_delta,
        "minimal_sensitivity_can_change_selection": False,
        "selection_remains_closed": True,
        "locked_outcomes_used": False,
    }
    write_json(output_dir / "sensitivity_decision.json", decision)


def model_domain_indices(
    manifest: pd.DataFrame,
    instrument_mapping: dict[str, int],
    sensor_mapping: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    instruments = (
        manifest["instrument"]
        .astype(str)
        .map(instrument_mapping)
        .fillna(-1)
        .to_numpy(dtype=np.int64)
    )
    sensors = (
        manifest["sensor_family"]
        .astype(str)
        .map(sensor_mapping)
        .fillna(-1)
        .to_numpy(dtype=np.int64)
    )
    return instruments, sensors


def evaluate_structured(
    model: structured.PartitionedSERSVAE,
    train_values: np.ndarray,
    test_values: np.ndarray,
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    config: structured.StructuredConfig,
    target_mapping: dict[str, int],
    instrument_mapping: dict[str, int],
    sensor_mapping: dict[str, int],
    context: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    artifact_stem: str,
) -> dict[str, Any]:
    model.to(device)
    train_instrument_indices, train_sensor_indices = model_domain_indices(
        train_manifest, instrument_mapping, sensor_mapping
    )
    test_instrument_indices, test_sensor_indices = model_domain_indices(
        test_manifest, instrument_mapping, sensor_mapping
    )
    train_outputs = structured.outputs(
        model,
        train_values,
        train_instrument_indices,
        train_sensor_indices,
        device,
    )
    test_outputs = structured.outputs(
        model,
        test_values,
        test_instrument_indices,
        test_sensor_indices,
        device,
    )
    train_targets = train_manifest["target_analyte"].astype(str).to_numpy()
    test_targets = test_manifest["target_analyte"].astype(str).to_numpy()
    train_instruments = train_manifest["instrument"].astype(str).to_numpy()
    test_instruments = test_manifest["instrument"].astype(str).to_numpy()
    train_sensors = train_manifest["sensor_family"].astype(str).to_numpy()
    test_sensors = test_manifest["sensor_family"].astype(str).to_numpy()
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    probes: dict[str, Any] = {}
    clean_predictions: dict[str, np.ndarray] = {}
    for partition, key in (
        ("chemical", "chemical_mu"),
        ("nuisance", "nuisance_mu"),
        ("union", "union_mu"),
    ):
        probe = baseline.fit_latent_probe_model(
            train_outputs[key],
            train_targets,
            baseline.stable_seed(artifact_stem, partition, "target"),
        )
        probes[partition] = probe
        prediction = probe.predict(test_outputs[key]).astype(str)
        clean_predictions[partition] = prediction
        confidence = np.max(probe.predict_proba(test_outputs[key]), axis=1)
        classification = baseline_final.supported_classification(
            train_targets, test_targets, prediction
        )
        instrument_probe = baseline_selection.safe_domain_probe(
            train_outputs[key],
            test_outputs[key],
            train_targets,
            test_targets,
            train_instruments,
            test_instruments,
            baseline.stable_seed(artifact_stem, partition, "instrument"),
        )
        sensor_probe = baseline_selection.safe_domain_probe(
            train_outputs[key],
            test_outputs[key],
            train_targets,
            test_targets,
            train_sensors,
            test_sensors,
            baseline.stable_seed(artifact_stem, partition, "sensor"),
        )
        geometry = baseline.geometry_metrics(
            test_outputs[key],
            test_manifest,
            np.ones(len(test_manifest), dtype=bool),
        )
        mu = test_outputs[key]
        log_variance = test_outputs[
            {
                "chemical": "chemical_log_variance",
                "nuisance": "nuisance_log_variance",
                "union": "union_log_variance",
            }[partition]
        ]
        variational = standard.variational_metrics(
            mu,
            log_variance,
            normalization_divisor=config.kl_normalization_divisor,
            sample_reconstruction_variability=np.nan,
        )
        partition_context = {**context, "partition": partition}
        metric_rows.append(
            {
                **baseline_final.classification_metric_row(
                    partition_context, classification
                ),
                "instrument_probe_increment": instrument_probe[
                    "increment_over_target_only"
                ],
                "sensor_probe_increment": sensor_probe[
                    "increment_over_target_only"
                ],
                **geometry,
                **variational,
            }
        )
        prediction_rows.extend(
            baseline_final.prediction_rows(
                partition_context,
                test_manifest,
                test_targets,
                prediction,
                confidence,
                classification["supported_mask"],
            )
        )
    repeatable = baseline_final.repeatable_test_positions(
        train_values, train_manifest, test_values, test_manifest
    )
    per_reconstruction = baseline.reconstruction_metrics(
        test_values,
        test_outputs["reconstruction"],
        test_manifest["observation_uid"].astype(str),
        repeatable,
    )
    per_reconstruction = pd.concat(
        [
            pd.DataFrame([context] * len(per_reconstruction)),
            per_reconstruction,
        ],
        axis=1,
    )
    reconstruction_summary = baseline.aggregate_reconstruction_metrics(
        per_reconstruction
    )
    dependence = structured.partition_dependence(
        test_outputs["chemical_mu"], test_outputs["nuisance_mu"]
    )
    for row in metric_rows:
        row.update(reconstruction_summary)
        row.update(dependence)
    corruption_rows: list[dict[str, Any]] = []
    for corruption in baseline.CORRUPTION_NAMES:
        for severity in (0.5, 1.0, 1.5):
            corrupted = baseline_selection.deterministic_corrupted_matrix(
                test_values,
                test_manifest["observation_uid"].astype(str),
                corruption,
                severity,
            )
            corrupted_outputs = structured.outputs(
                model,
                corrupted,
                test_instrument_indices,
                test_sensor_indices,
                device,
            )
            recovered = baseline.aggregate_reconstruction_metrics(
                baseline.reconstruction_metrics(
                    test_values,
                    corrupted_outputs["reconstruction"],
                    test_manifest["observation_uid"].astype(str),
                    repeatable,
                )
            )
            row: dict[str, Any] = {
                **context,
                "corruption": corruption,
                "severity": severity,
                **recovered,
            }
            for partition, key in (
                ("chemical", "chemical_mu"),
                ("nuisance", "nuisance_mu"),
                ("union", "union_mu"),
            ):
                prediction = probes[partition].predict(
                    corrupted_outputs[key]
                ).astype(str)
                classification = baseline_final.supported_classification(
                    train_targets, test_targets, prediction
                )
                row.update(
                    {
                        f"{partition}_balanced_accuracy_supported": (
                            classification["balanced_accuracy"]
                        ),
                        f"{partition}_macro_f1_supported": classification[
                            "macro_f1_supported"
                        ],
                        f"{partition}_prediction_agreement": float(
                            np.mean(
                                prediction == clean_predictions[partition]
                            )
                        ),
                        f"{partition}_latent_cosine_drift": float(
                            np.mean(
                                baseline_selection.cosine_drift(
                                    test_outputs[key],
                                    corrupted_outputs[key],
                                )
                            )
                        ),
                    }
                )
            corruption_rows.append(row)
    baseline_final.save_array_artifact(
        output_dir / "embeddings" / f"{artifact_stem}.npz",
        test_manifest["observation_uid"].astype(str),
        chemical_mu=test_outputs["chemical_mu"].astype(np.float32),
        chemical_log_variance=test_outputs[
            "chemical_log_variance"
        ].astype(np.float32),
        nuisance_mu=test_outputs["nuisance_mu"].astype(np.float32),
        nuisance_log_variance=test_outputs[
            "nuisance_log_variance"
        ].astype(np.float32),
        union_mu=test_outputs["union_mu"].astype(np.float32),
        union_log_variance=test_outputs[
            "union_log_variance"
        ].astype(np.float32),
    )
    baseline_final.save_array_artifact(
        output_dir / "reconstructions" / f"{artifact_stem}.npz",
        test_manifest["observation_uid"].astype(str),
        clean=test_values.astype(np.float32),
        reconstructed=test_outputs["reconstruction"].astype(np.float32),
    )
    return {
        "metrics": metric_rows,
        "predictions": prediction_rows,
        "reconstruction": per_reconstruction,
        "corruption": corruption_rows,
    }


def empty_outputs() -> dict[str, list[Any]]:
    return {
        "metrics": [],
        "predictions": [],
        "reconstruction": [],
        "corruption": [],
        "histories": [],
        "registry": [],
    }


def merge_outputs(
    target: dict[str, list[Any]], source: dict[str, Any]
) -> None:
    for key in ("metrics", "predictions", "corruption"):
        target[key].extend(source[key])
    target["reconstruction"].append(source["reconstruction"])


def save_outputs(
    output_dir: Path, prefix: str, outputs: dict[str, list[Any]]
) -> None:
    pd.DataFrame(outputs["metrics"]).to_csv(
        output_dir / f"{prefix}_metrics.csv", index=False
    )
    pd.DataFrame(outputs["predictions"]).to_csv(
        output_dir / f"{prefix}_predictions.csv", index=False
    )
    pd.concat(outputs["reconstruction"], ignore_index=True).to_csv(
        output_dir / f"{prefix}_reconstruction_metrics.csv", index=False
    )
    pd.DataFrame(outputs["corruption"]).to_csv(
        output_dir / f"{prefix}_corruption_metrics.csv", index=False
    )
    pd.concat(outputs["histories"], ignore_index=True).to_csv(
        output_dir / f"{prefix}_training_histories.csv", index=False
    )
    write_json(output_dir / f"{prefix}_run_registry.json", outputs["registry"])


def train_and_evaluate(
    output_dir: Path,
    stage: str,
    scenario: str,
    training_scenario: str,
    train_dataset: baseline.SpectralDataset,
    test_dataset: baseline.SpectralDataset,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    representation: str,
    config: structured.StructuredConfig,
    global_mappings: tuple[
        dict[str, int], dict[str, int], dict[str, int]
    ],
    fingerprint: str,
    training_device: torch.device,
    metric_device: torch.device,
    context: dict[str, Any],
    training_validation_dataset: baseline.SpectralDataset | None = None,
    training_validation_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    target_mapping, instrument_mapping, sensor_mapping = global_mappings
    train_manifest = train_dataset.manifest.loc[train_mask].reset_index(
        drop=True
    )
    test_manifest = test_dataset.manifest.loc[test_mask].reset_index(drop=True)
    train_values = train_dataset.representations[representation][train_mask]
    test_values = test_dataset.representations[representation][test_mask]
    validation_dataset = training_validation_dataset or test_dataset
    validation_mask = (
        training_validation_mask
        if training_validation_mask is not None
        else test_mask
    )
    validation_manifest = validation_dataset.manifest.loc[
        validation_mask
    ].reset_index(drop=True)
    validation_values = validation_dataset.representations[representation][
        validation_mask
    ]
    run_seed = baseline.stable_seed(
        structured.PROTOCOL_VERSION,
        "locked",
        stage,
        training_scenario,
        representation,
        DECLARED_SEED,
    )
    run_identifier = (
        f"{training_scenario}__{representation}__{config.identifier}"
    )
    model, history, registry = load_or_train(
        output_dir,
        stage,
        run_identifier,
        train_values,
        train_manifest,
        validation_values,
        validation_manifest,
        target_mapping,
        instrument_mapping,
        sensor_mapping,
        config,
        run_seed,
        fingerprint,
        training_device,
    )
    full_context = {
        **context,
        "stage": stage,
        "scenario": scenario,
        "training_scenario": training_scenario,
        "model_family": "structured_vae",
        "configuration": config.identifier,
        "representation": representation,
        "seed": DECLARED_SEED,
        "n_train": len(train_manifest),
        "n_test_partition": len(test_manifest),
    }
    evaluated = evaluate_structured(
        model,
        train_values,
        test_values,
        train_manifest,
        test_manifest,
        config,
        target_mapping,
        instrument_mapping,
        sensor_mapping,
        full_context,
        metric_device,
        output_dir,
        (
            f"{stage}__{scenario}__{representation}__"
            f"{config.identifier}__s{DECLARED_SEED}"
        ),
    )
    annotated_history = history.copy()
    for key, value in full_context.items():
        annotated_history[key] = value
    registry.update(full_context)
    del model
    if training_device.type == "cuda":
        torch.cuda.empty_cache()
    return evaluated, annotated_history, registry


def run_outer(
    output_dir: Path,
    nato_bundle: Path,
    config: structured.StructuredConfig,
    fingerprint: str,
    training_device: torch.device,
    metric_device: torch.device,
) -> None:
    core = baseline.load_nato_dataset(nato_bundle)
    quality = baseline_final.load_nato_subset(nato_bundle, "quality_pass")
    stress = baseline_final.load_nato_subset(
        nato_bundle, "field_quality_stress"
    )
    global_mappings = selection.mappings(core.manifest)
    outputs = empty_outputs()
    for outer_fold in range(5):
        for representation in ("arpls_minmax", "minimal_minmax"):
            core_folds = core.manifest[
                "grouped_sample_fold_5"
            ].to_numpy(dtype=int)
            scenario = (
                f"nato_outer_o{outer_fold}__train_strict_core__"
                "test_strict_core"
            )
            result, history, registry = train_and_evaluate(
                output_dir,
                "locked_outer",
                scenario,
                f"nato_outer_o{outer_fold}__train_strict_core",
                core,
                core,
                core_folds != outer_fold,
                core_folds == outer_fold,
                representation,
                config,
                global_mappings,
                fingerprint,
                training_device,
                metric_device,
                {
                    "outer_fold": outer_fold,
                    "train_subset": "strict_core",
                    "test_subset": "strict_core",
                },
            )
            merge_outputs(outputs, result)
            outputs["histories"].append(history)
            outputs["registry"].append(registry)
            quality_folds = quality.manifest[
                "grouped_sample_fold_5"
            ].to_numpy(dtype=int)
            stress_folds = stress.manifest[
                "grouped_sample_fold_5"
            ].to_numpy(dtype=int)
            training_scenario = (
                f"nato_outer_o{outer_fold}__train_quality_pass"
            )
            for test_dataset, test_subset, test_mask in (
                (
                    quality,
                    "quality_pass",
                    quality_folds == outer_fold,
                ),
                (
                    stress,
                    "field_quality_stress",
                    stress_folds == outer_fold,
                ),
            ):
                scenario = (
                    f"{training_scenario}__test_{test_subset}"
                )
                result, history, registry = train_and_evaluate(
                    output_dir,
                    "locked_outer",
                    scenario,
                    training_scenario,
                    quality,
                    test_dataset,
                    quality_folds != outer_fold,
                    test_mask,
                    representation,
                    config,
                    global_mappings,
                    fingerprint,
                    training_device,
                    metric_device,
                    {
                        "outer_fold": outer_fold,
                        "train_subset": "quality_pass",
                        "test_subset": test_subset,
                    },
                    training_validation_dataset=quality,
                    training_validation_mask=quality_folds == outer_fold,
                )
                merge_outputs(outputs, result)
                if test_subset == "quality_pass":
                    outputs["histories"].append(history)
                    outputs["registry"].append(registry)
            print(
                json.dumps(
                    {
                        "stage": "locked_outer",
                        "outer_fold": outer_fold,
                        "representation": representation,
                        "metric_rows": len(outputs["metrics"]),
                    }
                ),
                flush=True,
            )
    save_outputs(output_dir, "locked_outer", outputs)


def run_domain(
    output_dir: Path,
    nato_bundle: Path,
    config: structured.StructuredConfig,
    fingerprint: str,
    training_device: torch.device,
    metric_device: torch.device,
) -> None:
    core = baseline.load_nato_dataset(nato_bundle)
    datasets = {
        "strict_core": core,
        "quality_pass": baseline_final.load_nato_subset(
            nato_bundle, "quality_pass"
        ),
    }
    partition_files = {
        "strict_core": nato_bundle / "domain_evaluation_partitions_core.csv",
        "quality_pass": (
            nato_bundle / "domain_evaluation_partitions_quality.csv"
        ),
    }
    global_mappings = selection.mappings(core.manifest)
    outputs = empty_outputs()
    for subset, dataset in datasets.items():
        partitions = pd.read_csv(partition_files[subset])
        uids = dataset.manifest["observation_uid"].astype(str)
        for keys, frame in partitions.groupby(
            ["protocol", "domain_type", "heldout_domain"], sort=True
        ):
            protocol_name, domain_type, heldout_domain = [
                str(value) for value in keys
            ]
            train_uids = set(
                frame.loc[
                    frame["partition"].eq("train"), "observation_uid"
                ].astype(str)
            )
            test_uids = set(
                frame.loc[
                    frame["partition"].eq("test"), "observation_uid"
                ].astype(str)
            )
            train_mask = uids.isin(train_uids).to_numpy()
            test_mask = uids.isin(test_uids).to_numpy()
            scenario = (
                f"domain__{subset}__{protocol_name}__{domain_type}__"
                f"{heldout_domain}"
            )
            result, history, registry = train_and_evaluate(
                output_dir,
                "locked_domain",
                scenario,
                scenario,
                dataset,
                dataset,
                train_mask,
                test_mask,
                "arpls_minmax",
                config,
                global_mappings,
                fingerprint,
                training_device,
                metric_device,
                {
                    "evaluation_subset": subset,
                    "domain_protocol": protocol_name,
                    "domain_type": domain_type,
                    "heldout_domain": heldout_domain,
                },
            )
            merge_outputs(outputs, result)
            outputs["histories"].append(history)
            outputs["registry"].append(registry)
            print(
                json.dumps(
                    {
                        "stage": "locked_domain",
                        "subset": subset,
                        "protocol": protocol_name,
                        "domain_type": domain_type,
                        "heldout_domain": heldout_domain,
                    }
                ),
                flush=True,
            )
    save_outputs(output_dir, "locked_domain", outputs)


def poster_dataset_with_domains(
    poster_csv: Path,
) -> tuple[
    baseline.SpectralDataset,
    tuple[dict[str, int], dict[str, int], dict[str, int]],
]:
    original = baseline.load_poster_dataset(poster_csv)
    manifest = original.manifest.copy()
    manifest["instrument"] = "poster_unknown"
    manifest["sensor_family"] = manifest["substrate_family"].astype(str)
    manifest["master_sample_id"] = (
        manifest["observation_uid"].astype(str) + "__poster_master"
    )
    targets = sorted(manifest["target_analyte"].astype(str).unique())
    target_labels = targets + [
        f"__unused_target_{index}"
        for index in range(7 - len(targets))
    ]
    target_mapping = {
        label: index for index, label in enumerate(target_labels)
    }
    instrument_labels = ["poster_unknown"] + [
        f"__unused_instrument_{index}" for index in range(9)
    ]
    instrument_mapping = {
        label: index for index, label in enumerate(instrument_labels)
    }
    sensors = sorted(manifest["sensor_family"].astype(str).unique())
    if len(sensors) != 4:
        raise ValueError("Poster transfer expects four substrate families")
    sensor_mapping = {
        label: index for index, label in enumerate(sensors)
    }
    dataset = baseline.SpectralDataset(
        name=original.name,
        axis_cm1=original.axis_cm1,
        observation_uid=original.observation_uid,
        representations=original.representations,
        manifest=manifest,
    )
    return dataset, (target_mapping, instrument_mapping, sensor_mapping)


def run_poster(
    output_dir: Path,
    poster_csv: Path,
    config: structured.StructuredConfig,
    fingerprint: str,
    training_device: torch.device,
    metric_device: torch.device,
) -> None:
    poster, global_mappings = poster_dataset_with_domains(poster_csv)
    families = poster.manifest["sensor_family"].astype(str).to_numpy()
    outputs = empty_outputs()
    for heldout in sorted(np.unique(families)):
        for representation in ("arpls_minmax", "minimal_minmax"):
            scenario = f"poster_holdout_{heldout}"
            result, history, registry = train_and_evaluate(
                output_dir,
                "locked_poster",
                scenario,
                scenario,
                poster,
                poster,
                families != heldout,
                families == heldout,
                representation,
                config,
                global_mappings,
                fingerprint,
                training_device,
                metric_device,
                {
                    "heldout_substrate_family": heldout,
                    "poster_axis": "400-1800_common",
                    "interpretation": (
                        "descriptive architecture transfer; poster analyte "
                        "labels differ from NATO labels"
                    ),
                },
            )
            merge_outputs(outputs, result)
            outputs["histories"].append(history)
            outputs["registry"].append(registry)
            print(
                json.dumps(
                    {
                        "stage": "locked_poster",
                        "heldout_substrate_family": heldout,
                        "representation": representation,
                    }
                ),
                flush=True,
            )
    save_outputs(output_dir, "locked_poster", outputs)


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("sensitivity", "outer", "domain", "poster", "all"),
        default="all",
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
        / "sers_structured_vae"
        / "structured_vae_v1",
    )
    parser.add_argument(
        "--training-device",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument(
        "--metric-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    protocol = json.loads(args.protocol.read_text())
    config = frozen_configuration(args.output_dir)
    training_device = torch.device(args.training_device)
    metric_device = torch.device(args.metric_device)
    if args.stage in {"sensitivity", "all"}:
        run_sensitivity(
            args.output_dir,
            args.nato_bundle,
            config,
            execution_fingerprint(args.protocol, "sensitivity"),
            protocol,
            training_device,
            metric_device,
        )
    if args.stage in {"outer", "all"}:
        if not (args.output_dir / "sensitivity_decision.json").is_file():
            raise ValueError("Sensitivity must close before locked outcomes")
        run_outer(
            args.output_dir,
            args.nato_bundle,
            config,
            execution_fingerprint(args.protocol, "outer"),
            training_device,
            metric_device,
        )
    if args.stage in {"domain", "all"}:
        if not (args.output_dir / "locked_outer_metrics.csv").is_file():
            raise ValueError("Locked outer evaluation must run before domain")
        run_domain(
            args.output_dir,
            args.nato_bundle,
            config,
            execution_fingerprint(args.protocol, "domain"),
            training_device,
            metric_device,
        )
    if args.stage in {"poster", "all"}:
        if not (args.output_dir / "locked_domain_metrics.csv").is_file():
            raise ValueError("Locked domain evaluation must run before poster")
        run_poster(
            args.output_dir,
            args.poster_csv,
            config,
            execution_fingerprint(args.protocol, "poster"),
            training_device,
            metric_device,
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "stage": args.stage,
                "configuration": config.identifier,
                "training_device": str(training_device),
                "metric_device": str(metric_device),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
