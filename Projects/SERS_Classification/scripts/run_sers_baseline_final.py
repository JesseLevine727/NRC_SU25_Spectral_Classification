#!/usr/bin/env python3
"""Run sealed controls and final evaluations for SERS baseline protocol v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

# Exact rebuilds require deterministic CPU probe fits as well as deterministic
# Torch kernels. These must be set before importing NumPy/scikit-learn through
# the shared module.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import run_sers_representation_baselines as selection
import sers_baseline_common as common


ADVANCING_VIEW = "arpls_minmax"
FINAL_EVALUATION_VERSION = "sers-baseline-final-v1"


def write_json(path: Path, value: Any) -> None:
    selection.write_json(path, value)


def array_sha256(values: Sequence[str] | np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_nato_subset(bundle: Path, subset: str) -> common.SpectralDataset:
    common.verify_hash_catalog(bundle)
    files = {
        "quality_pass": (
            "final_model_inputs_quality.npz",
            "quality_preprocessing_manifest.csv",
            "NATO-Q500",
        ),
        "field_quality_stress": (
            "final_model_inputs_field_quality_stress.npz",
            "field_quality_stress_manifest.csv",
            "NATO-S98",
        ),
    }
    archive_name, manifest_name, dataset_name = files[subset]
    archive = np.load(bundle / archive_name)
    manifest = pd.read_csv(bundle / manifest_name)
    axis = archive["axis_cm1"].astype(np.float32)
    representations = {
        name: archive[name].astype(np.float32)
        for name in common.AUTHORIZED_REPRESENTATIONS
    }
    return common.SpectralDataset(
        name=dataset_name,
        axis_cm1=axis,
        observation_uid=archive["observation_uid"].astype(str),
        representations=representations,
        manifest=manifest,
    )


def config_from_record(record: dict[str, Any]) -> common.AutoencoderTrainingConfig:
    return common.AutoencoderTrainingConfig(
        channels=tuple(int(value) for value in record["channels"]),
        bottleneck_dimension=int(record["bottleneck_dimension"]),
        loss_name=str(record["loss_name"]),
        curriculum=str(record["curriculum"]),
        learning_rate=float(record["learning_rate"]),
        weight_decay=float(record["weight_decay"]),
        batch_size=int(record["batch_size"]),
        maximum_epochs=int(record["maximum_epochs"]),
        minimum_epochs=int(record["minimum_epochs"]),
        early_stopping_patience=int(record["early_stopping_patience"]),
        early_stopping_minimum_delta=float(
            record["early_stopping_minimum_delta"]
        ),
        gradient_clip_norm=float(record["gradient_clip_norm"]),
    )


def pre_outer_view_decision(
    output_dir: Path, selected: dict[str, Any]
) -> dict[str, Any]:
    path = output_dir / "pre_outer_advancing_view_decision.json"
    decision = {
        "decision_version": FINAL_EVALUATION_VERSION,
        "created_before_outer_evaluation": True,
        "outer_test_used": False,
        "poster_used": False,
        "field_quality_stress_used": False,
        "advancing_view_for_domain_transfer": ADVANCING_VIEW,
        "basis": (
            "Inner-only DAE comparison prioritized chemical separation, "
            "target-adjusted instrument invariance, same-master cross-instrument "
            "geometry, corrupted prediction agreement, and latent stability. "
            "Peak preservation remains a declared adverse trade-off."
        ),
        "candidate_metrics": {
            view: selected["denoising_autoencoders"][view]["metrics"]
            for view in common.INTENSITY_REPRESENTATIONS
        },
    }
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != selection.json_clean(decision):
            raise ValueError("Pre-outer advancing-view decision changed")
    else:
        write_json(path, decision)
    return decision


def selected_epoch_count(
    output_dir: Path,
    model_family: str,
    subset: str,
    outer_fold: int | None,
    representation: str,
    configuration: str,
) -> int:
    filename = (
        "ae_search_fold_metrics.csv"
        if model_family == "ae"
        else "dae_search_fold_metrics.csv"
    )
    frame = pd.read_csv(output_dir / filename)
    selected = frame[
        (frame["subset"] == subset)
        & (frame["representation"] == representation)
        & (frame["configuration"] == configuration)
    ]
    if outer_fold is not None:
        selected = selected[selected["outer_fold"] == outer_fold]
    if selected.empty:
        fallback = "strict_core"
        selected = frame[
            (frame["subset"] == fallback)
            & (frame["representation"] == representation)
            & (frame["configuration"] == configuration)
        ]
        if outer_fold is not None:
            selected = selected[selected["outer_fold"] == outer_fold]
    if selected.empty:
        raise ValueError(
            f"No epoch evidence for {model_family}/{subset}/{outer_fold}/"
            f"{representation}/{configuration}"
        )
    return max(1, int(math.floor(float(selected["best_epoch"].median()) + 0.5)))


def train_autoencoder_fixed(
    train_values: np.ndarray,
    train_uids: Sequence[str],
    config: common.AutoencoderTrainingConfig,
    epochs: int,
    run_seed: int,
    device: torch.device,
) -> tuple[common.ConvAutoencoder, pd.DataFrame, str]:
    common.configure_determinism(run_seed)
    model = common.ConvAutoencoder(
        train_values.shape[1],
        config.channels,
        config.bottleneck_dimension,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    dataset = common.ReconstructionDataset(
        train_values,
        train_uids,
        config.curriculum,
        run_seed,
        epochs,
    )
    generator = torch.Generator()
    generator.manual_seed(common.stable_seed(run_seed, "final_loader"))
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    rows: list[dict[str, Any]] = []
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        model.train()
        total = 0.0
        count = 0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            reconstruction, _ = model(inputs)
            loss = common.reconstruction_loss(
                reconstruction, targets, config.loss_name
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            total += float(loss.detach().cpu()) * len(inputs)
            count += len(inputs)
        rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / max(count, 1),
            }
        )
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    return model, pd.DataFrame(rows), common.state_dict_sha256(state)


def neural_paths(
    output_dir: Path, stage: str, run_identifier: str
) -> tuple[Path, Path]:
    checkpoint = output_dir / "checkpoints" / stage / f"{run_identifier}.pt"
    history = output_dir / "run_cache" / stage / f"{run_identifier}.csv"
    return checkpoint, history


def load_or_train_autoencoder(
    output_dir: Path,
    stage: str,
    run_identifier: str,
    train_values: np.ndarray,
    train_uids: Sequence[str],
    config: common.AutoencoderTrainingConfig,
    epochs: int,
    run_seed: int,
    device: torch.device,
) -> tuple[common.ConvAutoencoder, pd.DataFrame, dict[str, Any]]:
    checkpoint_path, history_path = neural_paths(
        output_dir, stage, run_identifier
    )
    expected = {
        "evaluation_version": FINAL_EVALUATION_VERSION,
        "run_identifier": run_identifier,
        "configuration": common.autoencoder_config_record(config),
        "epochs": epochs,
        "run_seed": run_seed,
        "train_uids_sha256": array_sha256(np.asarray(train_uids, dtype=str)),
        "train_values_sha256": array_sha256(
            np.asarray(train_values, dtype=np.float32)
        ),
    }
    if checkpoint_path.exists() and history_path.exists():
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if payload["metadata"] != expected:
            raise ValueError(f"Stale final AE checkpoint: {run_identifier}")
        actual = common.state_dict_sha256(payload["state_dict"])
        if actual != payload["state_sha256"]:
            raise ValueError(f"Corrupt final AE checkpoint: {run_identifier}")
        model = common.ConvAutoencoder(
            train_values.shape[1],
            config.channels,
            config.bottleneck_dimension,
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return model, pd.read_csv(history_path), {
            **expected,
            "state_sha256": actual,
            "parameter_count": common.model_parameter_count(model),
            "cache_status": "loaded",
            "checkpoint": str(checkpoint_path.relative_to(output_dir)),
        }
    model, history, state_hash = train_autoencoder_fixed(
        train_values,
        train_uids,
        config,
        epochs,
        run_seed,
        device,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    common.save_checkpoint(checkpoint_path, model, expected)
    history.to_csv(history_path, index=False)
    return model, history, {
        **expected,
        "state_sha256": state_hash,
        "parameter_count": common.model_parameter_count(model),
        "cache_status": "trained",
        "checkpoint": str(checkpoint_path.relative_to(output_dir)),
    }


def load_or_train_siamese(
    output_dir: Path,
    stage: str,
    run_identifier: str,
    train_values: np.ndarray,
    train_labels: Sequence[str],
    train_domains: Sequence[str],
    train_uids: Sequence[str],
    run_seed: int,
    device: torch.device,
    circular_shift: bool,
) -> tuple[common.SiameseEncoder, pd.DataFrame, dict[str, Any]]:
    checkpoint_path, history_path = neural_paths(
        output_dir, stage, run_identifier
    )
    expected = {
        "evaluation_version": FINAL_EVALUATION_VERSION,
        "run_identifier": run_identifier,
        "model": "fixed_siamese_triplet",
        "epochs": 100,
        "run_seed": run_seed,
        "circular_shift": circular_shift,
        "train_uids_sha256": array_sha256(np.asarray(train_uids, dtype=str)),
        "train_values_sha256": array_sha256(
            np.asarray(train_values, dtype=np.float32)
        ),
    }
    if checkpoint_path.exists() and history_path.exists():
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if payload["metadata"] != expected:
            raise ValueError(f"Stale final Siamese checkpoint: {run_identifier}")
        actual = common.state_dict_sha256(payload["state_dict"])
        if actual != payload["state_sha256"]:
            raise ValueError(f"Corrupt final Siamese checkpoint: {run_identifier}")
        model = common.SiameseEncoder(train_values.shape[1])
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return model, pd.read_csv(history_path), {
            **expected,
            "state_sha256": actual,
            "parameter_count": common.model_parameter_count(model),
            "cache_status": "loaded",
            "checkpoint": str(checkpoint_path.relative_to(output_dir)),
        }
    trained = common.train_siamese(
        train_values,
        train_labels,
        train_domains,
        train_uids,
        run_seed,
        device,
        circular_shift=circular_shift,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    common.save_checkpoint(checkpoint_path, trained.model, expected)
    trained.history.to_csv(history_path, index=False)
    return trained.model, trained.history, {
        **expected,
        "state_sha256": trained.state_sha256,
        "parameter_count": trained.parameter_count,
        "cache_status": "trained",
        "checkpoint": str(checkpoint_path.relative_to(output_dir)),
    }


def supported_classification(
    train_labels: Sequence[str],
    true_labels: Sequence[str],
    predictions: Sequence[str],
) -> dict[str, Any]:
    train_classes = set(np.asarray(train_labels, dtype=str))
    true_array = np.asarray(true_labels, dtype=str)
    prediction_array = np.asarray(predictions, dtype=str)
    supported = np.asarray(
        [label in train_classes for label in true_array], dtype=bool
    )
    metrics = (
        common.classification_summary(
            true_array[supported], prediction_array[supported]
        )
        if supported.any()
        else {
            "balanced_accuracy": np.nan,
            "macro_f1_supported": np.nan,
            "macro_f1_union": np.nan,
        }
    )
    return {
        **metrics,
        "n_test": len(true_array),
        "n_test_supported": int(supported.sum()),
        "n_test_unsupported": int((~supported).sum()),
        "supported_mask": supported,
        "unsupported_classes": sorted(set(true_array[~supported])),
    }


def repeatable_test_positions(
    train_values: np.ndarray,
    train_manifest: pd.DataFrame,
    test_values: np.ndarray,
    test_manifest: pd.DataFrame,
) -> list[set[int]] | None:
    del train_values, train_manifest
    required_columns = {"master_sample_id", "instrument"}
    if required_columns.issubset(test_manifest.columns):
        # This is a definition of the held-out clean reference peaks, not a
        # model-selection operation. It exactly matches the frozen
        # same-master/cross-instrument repeatability rule used in selection.
        return common.repeatable_peak_positions(
            test_values, test_manifest.reset_index(drop=True)
        )
    return None


def prediction_rows(
    context: dict[str, Any],
    manifest: pd.DataFrame,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    confidence: np.ndarray | None,
    supported: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, uid in enumerate(manifest["observation_uid"].astype(str)):
        rows.append(
            {
                **context,
                "observation_uid": uid,
                "true_label": str(true_labels[index]),
                "predicted_label": str(predictions[index]),
                "prediction_confidence": (
                    float(confidence[index])
                    if confidence is not None
                    else np.nan
                ),
                "test_class_supported": bool(supported[index]),
                "correct": bool(predictions[index] == true_labels[index]),
                "instrument": str(manifest.iloc[index].get("instrument", "")),
                "sensor_family": str(
                    manifest.iloc[index].get("sensor_family", "")
                ),
                "substrate_family": str(
                    manifest.iloc[index].get("substrate_family", "")
                ),
                "source_substrate": str(
                    manifest.iloc[index].get("source_substrate", "")
                ),
                "target_analyte": str(true_labels[index]),
            }
        )
    return rows


def classification_metric_row(
    context: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        **context,
        "balanced_accuracy_supported": metrics["balanced_accuracy"],
        "macro_f1_supported": metrics["macro_f1_supported"],
        "macro_f1_union": metrics["macro_f1_union"],
        "n_test": metrics["n_test"],
        "n_test_supported": metrics["n_test_supported"],
        "n_test_unsupported": metrics["n_test_unsupported"],
        "unsupported_classes": "|".join(metrics["unsupported_classes"]),
    }


def save_array_artifact(
    path: Path,
    observation_uid: Sequence[str],
    **arrays: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observation_uid=np.asarray(observation_uid, dtype=str),
        **{key: np.asarray(value) for key, value in arrays.items()},
    )


def evaluate_autoencoder(
    model: common.ConvAutoencoder,
    train_values: np.ndarray,
    test_values: np.ndarray,
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    context: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    artifact_stem: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    pd.DataFrame,
    list[dict[str, Any]],
]:
    train_targets = train_manifest["target_analyte"].astype(str).to_numpy()
    test_targets = test_manifest["target_analyte"].astype(str).to_numpy()
    train_reconstruction, train_latent = common.autoencoder_outputs(
        model, train_values, device
    )
    test_reconstruction, test_latent = common.autoencoder_outputs(
        model, test_values, device
    )
    probe = common.fit_latent_probe_model(
        train_latent,
        train_targets,
        common.stable_seed(artifact_stem, "final_probe"),
    )
    predictions = probe.predict(test_latent).astype(str)
    confidence = np.max(probe.predict_proba(test_latent), axis=1)
    classification = supported_classification(
        train_targets, test_targets, predictions
    )
    repeatable = repeatable_test_positions(
        train_values,
        train_manifest,
        test_values,
        test_manifest,
    )
    per_reconstruction = common.reconstruction_metrics(
        test_values,
        test_reconstruction,
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
    reconstruction = common.aggregate_reconstruction_metrics(
        per_reconstruction
    )
    save_array_artifact(
        output_dir / "embeddings" / f"{artifact_stem}.npz",
        test_manifest["observation_uid"].astype(str),
        latent=test_latent.astype(np.float32),
    )
    save_array_artifact(
        output_dir / "reconstructions" / f"{artifact_stem}.npz",
        test_manifest["observation_uid"].astype(str),
        clean=test_values.astype(np.float32),
        reconstructed=test_reconstruction.astype(np.float32),
    )
    corruption_rows: list[dict[str, Any]] = []
    for corruption in common.CORRUPTION_NAMES:
        for severity in (0.5, 1.0, 1.5):
            corrupted = selection.deterministic_corrupted_matrix(
                test_values,
                test_manifest["observation_uid"].astype(str),
                corruption,
                severity,
            )
            recovered, corrupted_latent = common.autoencoder_outputs(
                model, corrupted, device
            )
            corrupted_prediction = probe.predict(corrupted_latent).astype(str)
            corrupted_classification = supported_classification(
                train_targets, test_targets, corrupted_prediction
            )
            recovery = common.aggregate_reconstruction_metrics(
                common.reconstruction_metrics(
                    test_values,
                    recovered,
                    test_manifest["observation_uid"].astype(str),
                    repeatable,
                )
            )
            corruption_rows.append(
                {
                    **context,
                    "corruption": corruption,
                    "severity": severity,
                    "balanced_accuracy_supported": corrupted_classification[
                        "balanced_accuracy"
                    ],
                    "macro_f1_supported": corrupted_classification[
                        "macro_f1_supported"
                    ],
                    "prediction_agreement": float(
                        np.mean(corrupted_prediction == predictions)
                    ),
                    "latent_cosine_drift": float(
                        np.mean(
                            selection.cosine_drift(
                                test_latent, corrupted_latent
                            )
                        )
                    ),
                    **recovery,
                }
            )
    metric = {
        **classification_metric_row(context, classification),
        **reconstruction,
    }
    predictions_rows = prediction_rows(
        context,
        test_manifest,
        test_targets,
        predictions,
        confidence,
        classification["supported_mask"],
    )
    return metric, predictions_rows, per_reconstruction, corruption_rows


def evaluate_classical(
    model_name: str,
    representation: str,
    train_dataset: common.SpectralDataset,
    test_dataset: common.SpectralDataset,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    context: dict[str, Any],
    seed: int,
    train_labels_override: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_manifest = train_dataset.manifest.loc[train_mask].reset_index(drop=True)
    test_manifest = test_dataset.manifest.loc[test_mask].reset_index(drop=True)
    train_labels = (
        train_labels_override
        if train_labels_override is not None
        else train_manifest["target_analyte"].astype(str).to_numpy()
    )
    test_labels = test_manifest["target_analyte"].astype(str).to_numpy()
    predictions, confidence = common.fit_classical_predict(
        model_name,
        train_dataset.representations[representation][train_mask],
        train_labels,
        test_dataset.representations[representation][test_mask],
        seed,
    )
    metrics = supported_classification(train_labels, test_labels, predictions)
    return (
        classification_metric_row(context, metrics),
        prediction_rows(
            context,
            test_manifest,
            test_labels,
            predictions,
            confidence,
            metrics["supported_mask"],
        ),
    )


def evaluate_siamese(
    model: common.SiameseEncoder,
    train_values: np.ndarray,
    test_values: np.ndarray,
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    context: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    artifact_stem: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_labels = train_manifest["target_analyte"].astype(str).to_numpy()
    test_labels = test_manifest["target_analyte"].astype(str).to_numpy()
    train_embedding = common.embed_siamese(model, train_values, device)
    test_embedding = common.embed_siamese(model, test_values, device)
    predictions, confidence, _ = common.nearest_prototype_predict(
        train_embedding, train_labels, test_embedding
    )
    metrics = supported_classification(train_labels, test_labels, predictions)
    save_array_artifact(
        output_dir / "embeddings" / f"{artifact_stem}.npz",
        test_manifest["observation_uid"].astype(str),
        latent=test_embedding.astype(np.float32),
    )
    return (
        classification_metric_row(context, metrics),
        prediction_rows(
            context,
            test_manifest,
            test_labels,
            predictions,
            confidence,
            metrics["supported_mask"],
        ),
    )


def evaluate_model_suite(
    output_dir: Path,
    stage: str,
    scenario: str,
    train_dataset: common.SpectralDataset,
    test_dataset: common.SpectralDataset,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    selected: dict[str, Any],
    device: torch.device,
    outer_fold: int | None,
    epoch_subset: str,
    domain_column: str,
    include_all_views: bool,
    circular_siamese: bool = False,
    training_scenario: str | None = None,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, list[Any]]:
    training_scenario = training_scenario or scenario
    extra_context = extra_context or {}
    train_manifest = train_dataset.manifest.loc[train_mask].reset_index(drop=True)
    test_manifest = test_dataset.manifest.loc[test_mask].reset_index(drop=True)
    train_labels = train_manifest["target_analyte"].astype(str).to_numpy()
    outputs: dict[str, list[Any]] = {
        "metrics": [],
        "predictions": [],
        "reconstruction": [],
        "corruption": [],
        "histories": [],
        "registry": [],
    }
    classical_representations = list(common.AUTHORIZED_REPRESENTATIONS)
    for representation in classical_representations:
        for model_name in ("nearest_centroid", "pca_logistic", "linear_svm"):
            context = {
                **extra_context,
                "stage": stage,
                "scenario": scenario,
                "outer_fold": outer_fold,
                "model_family": "classical",
                "model": model_name,
                "representation": representation,
                "curriculum": "none",
                "seed": 2026,
            }
            metric, predictions = evaluate_classical(
                model_name,
                representation,
                train_dataset,
                test_dataset,
                train_mask,
                test_mask,
                context,
                common.stable_seed(scenario, model_name, representation, 2026),
            )
            outputs["metrics"].append(metric)
            outputs["predictions"].extend(predictions)

    siamese_representation = "derivative_1"
    train_siamese_values = train_dataset.representations[siamese_representation][
        train_mask
    ]
    test_siamese_values = test_dataset.representations[siamese_representation][
        test_mask
    ]
    train_domains = train_manifest[domain_column].astype(str).to_numpy()
    for declared_seed in (1729, 2718, 3141):
        run_seed = common.stable_seed(
            common.PROTOCOL_VERSION,
            train_dataset.name,
            stage,
            training_scenario,
            "siamese",
            declared_seed,
        )
        run_identifier = (
            f"{training_scenario}__siamese__{siamese_representation}__"
            f"s{declared_seed}"
        )
        model, history, registry = load_or_train_siamese(
            output_dir,
            stage,
            run_identifier,
            train_siamese_values,
            train_labels,
            train_domains,
            train_manifest["observation_uid"].astype(str),
            run_seed,
            device,
            circular_siamese,
        )
        context = {
            **extra_context,
            "stage": stage,
            "scenario": scenario,
            "outer_fold": outer_fold,
            "model_family": "siamese",
            "model": (
                "historical_circular_triplet"
                if circular_siamese
                else "deterministic_edge_triplet"
            ),
            "representation": siamese_representation,
            "curriculum": "triplet",
            "seed": declared_seed,
        }
        metric, predictions = evaluate_siamese(
            model,
            train_siamese_values,
            test_siamese_values,
            train_manifest,
            test_manifest,
            context,
            device,
            output_dir,
            (
                f"{stage}__{scenario}__siamese__"
                f"{siamese_representation}__s{declared_seed}"
            ),
        )
        outputs["metrics"].append(metric)
        outputs["predictions"].extend(predictions)
        history = history.copy()
        for key, value in context.items():
            history[key] = value
        outputs["histories"].append(history)
        outputs["registry"].append(registry)

    views = (
        common.INTENSITY_REPRESENTATIONS
        if include_all_views
        else (ADVANCING_VIEW,)
    )
    for model_family, selection_key in (
        ("ae", "autoencoders"),
        ("dae", "denoising_autoencoders"),
    ):
        for representation in views:
            record = selected[selection_key][representation]
            if model_family == "dae" and not record.get("selected", False):
                continue
            config = config_from_record(record)
            epochs = selected_epoch_count(
                output_dir,
                model_family,
                epoch_subset,
                outer_fold,
                representation,
                config.identifier,
            )
            train_values = train_dataset.representations[representation][
                train_mask
            ]
            test_values = test_dataset.representations[representation][
                test_mask
            ]
            for declared_seed in (1729, 2718, 3141):
                run_seed = common.stable_seed(
                    common.PROTOCOL_VERSION,
                    train_dataset.name,
                    stage,
                    training_scenario,
                    model_family,
                    representation,
                    config.identifier,
                    declared_seed,
                )
                run_identifier = (
                    f"{training_scenario}__{model_family}__{representation}__"
                    f"{config.identifier}__s{declared_seed}"
                )
                model, history, registry = load_or_train_autoencoder(
                    output_dir,
                    stage,
                    run_identifier,
                    train_values,
                    train_manifest["observation_uid"].astype(str),
                    config,
                    epochs,
                    run_seed,
                    device,
                )
                context = {
                    **extra_context,
                    "stage": stage,
                    "scenario": scenario,
                    "outer_fold": outer_fold,
                    "model_family": model_family,
                    "model": config.identifier,
                    "representation": representation,
                    "curriculum": config.curriculum,
                    "seed": declared_seed,
                }
                metric, predictions, reconstruction, corruption = (
                    evaluate_autoencoder(
                        model,
                        train_values,
                        test_values,
                        train_manifest,
                        test_manifest,
                        context,
                        device,
                        output_dir,
                        (
                            f"{stage}__{scenario}__{model_family}__"
                            f"{representation}__{config.identifier}__"
                            f"s{declared_seed}"
                        ),
                    )
                )
                outputs["metrics"].append(metric)
                outputs["predictions"].extend(predictions)
                outputs["reconstruction"].append(reconstruction)
                outputs["corruption"].extend(corruption)
                history = history.copy()
                for key, value in context.items():
                    history[key] = value
                history["fixed_epoch_count"] = epochs
                outputs["histories"].append(history)
                outputs["registry"].append(registry)
    return outputs


def merge_outputs(
    target: dict[str, list[Any]], source: dict[str, list[Any]]
) -> None:
    for key in target:
        target[key].extend(source[key])


def empty_outputs() -> dict[str, list[Any]]:
    return {
        "metrics": [],
        "predictions": [],
        "reconstruction": [],
        "corruption": [],
        "histories": [],
        "registry": [],
    }


def save_stage_outputs(
    output_dir: Path,
    prefix: str,
    outputs: dict[str, list[Any]],
) -> None:
    pd.DataFrame(outputs["metrics"]).to_csv(
        output_dir / f"{prefix}_metrics.csv", index=False
    )
    pd.DataFrame(outputs["predictions"]).to_csv(
        output_dir / f"{prefix}_predictions.csv", index=False
    )
    reconstruction = (
        pd.concat(outputs["reconstruction"], ignore_index=True)
        if outputs["reconstruction"]
        else pd.DataFrame()
    )
    reconstruction.to_csv(
        output_dir / f"{prefix}_reconstruction_metrics.csv", index=False
    )
    pd.DataFrame(outputs["corruption"]).to_csv(
        output_dir / f"{prefix}_corruption_metrics.csv", index=False
    )
    histories = (
        pd.concat(outputs["histories"], ignore_index=True)
        if outputs["histories"]
        else pd.DataFrame()
    )
    histories.to_csv(
        output_dir / f"{prefix}_training_histories.csv", index=False
    )
    write_json(output_dir / f"{prefix}_run_registry.json", outputs["registry"])


def run_controls(
    output_dir: Path,
    core: common.SpectralDataset,
) -> None:
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    folds = core.manifest["grouped_sample_fold_5"].to_numpy(dtype=int)
    for outer_fold in range(5):
        train_mask = folds != outer_fold
        test_mask = folds == outer_fold
        train_manifest = core.manifest.loc[train_mask].reset_index(drop=True)
        group_targets = (
            train_manifest.groupby("master_sample_id")["target_analyte"]
            .first()
            .sort_index()
        )
        rng = np.random.default_rng(
            common.stable_seed(
                common.PROTOCOL_VERSION,
                "label_permutation",
                outer_fold,
                2026,
            )
        )
        permuted_values = rng.permutation(group_targets.to_numpy(dtype=str))
        mapping = dict(zip(group_targets.index, permuted_values))
        permuted = (
            train_manifest["master_sample_id"].map(mapping).astype(str).to_numpy()
        )
        context = {
            "stage": "controls",
            "scenario": f"nato_outer_o{outer_fold}_label_permutation",
            "outer_fold": outer_fold,
            "model_family": "negative_control",
            "model": "pca_logistic_group_label_permutation",
            "representation": ADVANCING_VIEW,
            "curriculum": "none",
            "seed": 2026,
        }
        metric, pred = evaluate_classical(
            "pca_logistic",
            ADVANCING_VIEW,
            core,
            core,
            train_mask,
            test_mask,
            context,
            common.stable_seed("label_permutation", outer_fold),
            train_labels_override=permuted,
        )
        rows.append(metric)
        predictions.extend(pred)

        for representation in common.INTENSITY_REPRESENTATIONS:
            test_values = core.representations[representation][test_mask]
            manifest = core.manifest.loc[test_mask].reset_index(drop=True)
            identity = common.aggregate_reconstruction_metrics(
                common.reconstruction_metrics(
                    test_values,
                    test_values.copy(),
                    manifest["observation_uid"].astype(str),
                )
            )
            rows.append(
                {
                    "stage": "controls",
                    "scenario": f"nato_outer_o{outer_fold}_identity",
                    "outer_fold": outer_fold,
                    "model_family": "identity_control",
                    "model": "identity_mapping",
                    "representation": representation,
                    "curriculum": "none",
                    "seed": 0,
                    **identity,
                }
            )
    pd.DataFrame(rows).to_csv(
        output_dir / "control_metrics.csv", index=False
    )
    pd.DataFrame(predictions).to_csv(
        output_dir / "control_predictions.csv", index=False
    )
    write_json(
        output_dir / "control_interpretation.json",
        {
            "label_permutation": (
                "Negative control; performance should approach chance and is "
                "not a model estimate."
            ),
            "identity_mapping": (
                "Reconstruction ceiling demonstrating that reconstruction "
                "metrics alone do not establish useful representation learning."
            ),
            "poster_independence": (
                "Poster rows are map locations without physical-preparation "
                "identifiers; substrate-family holdout is descriptive domain "
                "transfer, not independent-preparation validation."
            ),
        },
    )


def run_outer(
    output_dir: Path,
    nato_bundle: Path,
    selected: dict[str, Any],
    device: torch.device,
) -> None:
    core = common.load_nato_dataset(nato_bundle)
    quality = load_nato_subset(nato_bundle, "quality_pass")
    stress = load_nato_subset(nato_bundle, "field_quality_stress")
    outputs = empty_outputs()
    for outer_fold in range(5):
        for train_dataset, test_dataset, train_subset, test_subset in (
            (core, core, "strict_core", "strict_core"),
            (quality, quality, "quality_pass", "quality_pass"),
            (
                quality,
                stress,
                "quality_pass",
                "field_quality_stress",
            ),
        ):
            train_folds = train_dataset.manifest[
                "grouped_sample_fold_5"
            ].to_numpy(dtype=int)
            test_folds = test_dataset.manifest[
                "grouped_sample_fold_5"
            ].to_numpy(dtype=int)
            train_mask = train_folds != outer_fold
            test_mask = test_folds == outer_fold
            scenario = (
                f"nato_outer_o{outer_fold}__train_{train_subset}__test_{test_subset}"
            )
            result = evaluate_model_suite(
                output_dir,
                "outer",
                scenario,
                train_dataset,
                test_dataset,
                train_mask,
                test_mask,
                selected,
                device,
                outer_fold,
                train_subset,
                "instrument",
                include_all_views=True,
                training_scenario=(
                    f"nato_outer_o{outer_fold}__train_{train_subset}"
                ),
            )
            merge_outputs(outputs, result)
            print(
                json.dumps(
                    {
                        "stage": "outer",
                        "outer_fold": outer_fold,
                        "test_subset": test_subset,
                        "metrics": len(result["metrics"]),
                    }
                ),
                flush=True,
            )
    save_stage_outputs(output_dir, "outer_fold", outputs)


def run_poster_historical_siamese(
    output_dir: Path,
    source_csv: Path,
    device: torch.device,
) -> dict[str, list[Any]]:
    _, values, metadata = common.load_poster_historical_derivative(source_csv)
    outputs = empty_outputs()
    for heldout in sorted(metadata["substrate_family"].astype(str).unique()):
        train_mask = (
            metadata["substrate_family"].astype(str).to_numpy() != heldout
        )
        test_mask = ~train_mask
        train_manifest = metadata.loc[train_mask].reset_index(drop=True)
        test_manifest = metadata.loc[test_mask].reset_index(drop=True)
        train_values = values[train_mask]
        test_values = values[test_mask]
        train_labels = (
            train_manifest["target_analyte"].astype(str).to_numpy()
        )
        train_domains = (
            train_manifest["substrate_family"].astype(str).to_numpy()
        )
        scenario = f"poster_holdout_{heldout}"
        for declared_seed in (1729, 2718, 3141):
            run_seed = common.stable_seed(
                common.PROTOCOL_VERSION,
                "Poster-275-historical",
                "poster",
                scenario,
                "historical_siamese",
                declared_seed,
            )
            run_identifier = (
                f"{scenario}__historical_siamese__derivative_1_330_1800__"
                f"s{declared_seed}"
            )
            model, history, registry = load_or_train_siamese(
                output_dir,
                "poster",
                run_identifier,
                train_values,
                train_labels,
                train_domains,
                train_manifest["observation_uid"].astype(str),
                run_seed,
                device,
                circular_shift=True,
            )
            context = {
                "heldout_substrate_family": heldout,
                "poster_axis": "330-1800_native",
                "stage": "poster",
                "scenario": scenario,
                "outer_fold": np.nan,
                "model_family": "siamese",
                "model": "historical_circular_triplet",
                "representation": "derivative_1_historical",
                "curriculum": "triplet",
                "seed": declared_seed,
            }
            metric, predictions = evaluate_siamese(
                model,
                train_values,
                test_values,
                train_manifest,
                test_manifest,
                context,
                device,
                output_dir,
                (
                    f"poster__{scenario}__historical_siamese__"
                    f"s{declared_seed}"
                ),
            )
            outputs["metrics"].append(metric)
            outputs["predictions"].extend(predictions)
            history = history.copy()
            for key, value in context.items():
                history[key] = value
            outputs["histories"].append(history)
            outputs["registry"].append(registry)
    return outputs


def run_poster(
    output_dir: Path,
    poster_csv: Path,
    selected: dict[str, Any],
    device: torch.device,
) -> None:
    poster = common.load_poster_dataset(poster_csv)
    outputs = empty_outputs()
    substrates = sorted(
        poster.manifest["substrate_family"].astype(str).unique()
    )
    for heldout in substrates:
        families = poster.manifest["substrate_family"].astype(str).to_numpy()
        train_mask = families != heldout
        test_mask = families == heldout
        scenario = f"poster_holdout_{heldout}"
        result = evaluate_model_suite(
            output_dir,
            "poster",
            scenario,
            poster,
            poster,
            train_mask,
            test_mask,
            selected,
            device,
            outer_fold=None,
            epoch_subset="strict_core",
            domain_column="substrate_family",
            include_all_views=True,
            circular_siamese=False,
            extra_context={
                "heldout_substrate_family": heldout,
                "poster_axis": "400-1800_common",
            },
        )
        merge_outputs(outputs, result)
        print(
            json.dumps(
                {
                    "stage": "poster",
                    "heldout_substrate_family": heldout,
                    "metrics": len(result["metrics"]),
                }
            ),
            flush=True,
        )
    historical = run_poster_historical_siamese(
        output_dir, poster_csv, device
    )
    merge_outputs(outputs, historical)
    save_stage_outputs(output_dir, "poster", outputs)

    predictions = pd.DataFrame(outputs["predictions"])
    ag_4np = predictions[
        (predictions["heldout_substrate_family"] == "Ag")
        & (predictions["true_label"] == "4np")
    ].copy()
    ag_4np["localized_case"] = "heldout_Ag__all_4np"
    agnp = ag_4np[ag_4np["source_substrate"] == "AgNP"].copy()
    agnp["localized_case"] = "heldout_Ag__AgNP_source__4np"
    pd.concat([ag_4np, agnp], ignore_index=True).to_csv(
        output_dir / "poster_localized_4np_failures.csv", index=False
    )
    write_json(
        output_dir / "poster_evaluation_limitation.json",
        {
            "physical_preparation_ids_available": False,
            "interpretation": (
                "Leave-one-substrate-family-out evaluates substrate-domain "
                "transfer over map-location rows. It must not be interpreted "
                "as validation on independent chemical preparations."
            ),
            "historical_control": (
                "330-1800 cm-1 SNV + Savitzky-Golay first derivative + L2, "
                "fixed triplet network, circular shift retained only for exact "
                "legacy-method control."
            ),
        },
    )


def run_domain_transfer(
    output_dir: Path,
    nato_bundle: Path,
    selected: dict[str, Any],
    device: torch.device,
) -> None:
    datasets = {
        "strict_core": common.load_nato_dataset(nato_bundle),
        "quality_pass": load_nato_subset(nato_bundle, "quality_pass"),
    }
    partition_files = {
        "strict_core": nato_bundle / "domain_evaluation_partitions_core.csv",
        "quality_pass": nato_bundle
        / "domain_evaluation_partitions_quality.csv",
    }
    outputs = empty_outputs()
    for subset, dataset in datasets.items():
        partitions = pd.read_csv(partition_files[subset])
        scenario_columns = ["protocol", "domain_type", "heldout_domain"]
        for keys, scenario_frame in partitions.groupby(
            scenario_columns, sort=True
        ):
            protocol_name, domain_type, heldout_domain = [
                str(value) for value in keys
            ]
            train_uids = set(
                scenario_frame.loc[
                    scenario_frame["partition"] == "train", "observation_uid"
                ].astype(str)
            )
            test_uids = set(
                scenario_frame.loc[
                    scenario_frame["partition"] == "test", "observation_uid"
                ].astype(str)
            )
            uids = dataset.manifest["observation_uid"].astype(str)
            train_mask = uids.isin(train_uids).to_numpy()
            test_mask = uids.isin(test_uids).to_numpy()
            if int(train_mask.sum()) != len(train_uids):
                raise ValueError("Domain train UID coverage mismatch")
            if int(test_mask.sum()) != len(test_uids):
                raise ValueError("Domain test UID coverage mismatch")
            scenario = (
                f"domain__{subset}__{protocol_name}__{domain_type}__"
                f"{heldout_domain}"
            )
            result = evaluate_model_suite(
                output_dir,
                "domain",
                scenario,
                dataset,
                dataset,
                train_mask,
                test_mask,
                selected,
                device,
                outer_fold=None,
                epoch_subset=subset,
                domain_column=domain_type,
                include_all_views=False,
                extra_context={
                    "evaluation_subset": subset,
                    "domain_protocol": protocol_name,
                    "domain_type": domain_type,
                    "heldout_domain": heldout_domain,
                    "n_train": int(train_mask.sum()),
                    "n_test_partition": int(test_mask.sum()),
                },
            )
            merge_outputs(outputs, result)
            print(
                json.dumps(
                    {
                        "stage": "domain",
                        "subset": subset,
                        "protocol": protocol_name,
                        "domain_type": domain_type,
                        "heldout_domain": heldout_domain,
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        "metrics": len(result["metrics"]),
                    }
                ),
                flush=True,
            )
    save_stage_outputs(output_dir, "domain_transfer", outputs)


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["controls", "outer", "poster", "domain", "all"],
        default="all",
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
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    output_dir = args.output_dir.resolve()
    nato_bundle = args.nato_bundle.resolve()
    poster_csv = args.poster_csv.resolve()
    selected_path = output_dir / "selected_configurations.json"
    if not selected_path.exists():
        raise FileNotFoundError(
            "Selection must be closed before final evaluation"
        )
    selected = json.loads(selected_path.read_text())
    if not selected.get("selection_closed", False):
        raise ValueError("Selection is not closed")
    pre_outer_view_decision(output_dir, selected)
    device = common.resolve_device(args.device)
    core = common.load_nato_dataset(nato_bundle)
    if args.stage in {"controls", "all"}:
        run_controls(output_dir, core)
    if args.stage in {"outer", "all"}:
        run_outer(output_dir, nato_bundle, selected, device)
    if args.stage in {"poster", "all"}:
        run_poster(output_dir, poster_csv, selected, device)
    if args.stage in {"domain", "all"}:
        run_domain_transfer(output_dir, nato_bundle, selected, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
